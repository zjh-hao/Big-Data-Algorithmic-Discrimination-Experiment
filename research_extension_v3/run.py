"""Resumable strong-predictor and business-aware fairness comparison."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from threadpoolctl import threadpool_limits
from .common import HERE, ROOT, BASELINE_SOURCES, dump, sha, now, load_config
from .prepare import load_prepared
from .business import fit_policies, evaluate_policies, metrics_for_decisions, apply_policy

SOURCES = BASELINE_SOURCES

class FrozenGoodScore(ClassifierMixin, BaseEstimator):
    """Scikit-learn wrapper for already frozen favorable-outcome scores."""
    def fit(self, X=None, y=None):
        self.classes_ = np.array([0,1])
        self.n_features_in_ = 1
        return self
    def predict_proba(self, X):
        p = np.asarray(X).reshape(-1)
        return np.column_stack([1-p,p])
    def predict(self, X):
        return (np.asarray(X).reshape(-1)>=.5).astype(int)

def fairlearn_policies(val, test, cfg, name, run):
    sys.path.insert(0, str(SOURCES/'fairlearn'))
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn import __version__
    rows, frozen = [], []
    for objective in cfg['fairlearn_objectives']:
        for tol in cfg['fairlearn_tolerances']:
            ident = f'fairlearn_{objective}_tol{tol:g}'
            processor = ThresholdOptimizer(estimator=FrozenGoodScore().fit(), prefit=True,
                    predict_method='predict_proba', constraints='true_positive_rate_parity',
                    objective=objective, tol=tol, grid_size=1000, flip=False)
            processor.fit((1-val.probability.to_numpy()).reshape(-1,1), 1-val.TARGET.to_numpy(),
                          sensitive_features=val.thin.to_numpy())
            qval = processor._pmf_predict((1-val.probability.to_numpy()).reshape(-1,1), sensitive_features=val.thin.to_numpy())[:,1]
            interpolation = processor.interpolated_thresholder_.interpolation_dict
            # Freeze operator strings and mixture coefficients before accessing test scores.
            def serial(value):
                if isinstance(value, dict): return {str(k): serial(v) for k,v in value.items()}
                if isinstance(value, (np.generic,)): return value.item()
                if isinstance(value, (float,int,str,bool)) or value is None: return value
                return str(value)
            rule = dict(policy_id=ident, library_version=__version__, objective=objective, epsilon=tol,
                        constraints='true_positive_rate_parity', favorable_label='1-TARGET',
                        interpolation=serial(interpolation), frozen_utc=now(),
                        validation_metrics=metrics_for_decisions(val,qval,cfg['primary_loss_ratio'],cfg))
            dump(run/'policies'/f'{name}_{ident}.json', rule)
            q = processor._pmf_predict((1-test.probability.to_numpy()).reshape(-1,1), sensitive_features=test.thin.to_numpy())[:,1]
            m = metrics_for_decisions(test,q,cfg['primary_loss_ratio'],cfg)
            rows.append(dict(policy_id=ident, family='fairlearn', objective=objective,
                             utility_basis='unit_utility',status='PASS',feasible=True,fit_status='PASS',
                             epsilon=tol,risk_budget=cfg['primary_risk_budget'],
                             noharm_delta=None,capital_fraction=1.0,method='fairlearn',
                             evaluation_type='expected_randomized_policy',
                             test_risk_constraint_met=m['risk']<=cfg['primary_risk_budget'],
                             test_gap_constraint_met=m['abs_frr_gap']<=tol+1e-12,
                             validation_risk=rule['validation_metrics']['risk'],
                             **m))
            decisions = test[['SK_ID_CURR','TARGET','thin','AMT_CREDIT','probability']].copy()
            decisions['approval_probability'] = q
            decisions.to_parquet(run/'decisions'/f'{name}_{ident}.parquet',index=False)
            joblib.dump(processor,run/'checkpoints'/name/f'{ident}.joblib')
            frozen.append(rule)
    return rows, frozen

def train_basic(model_name, X, dense, records, split, cfg, seed, artifact):
    import lightgbm as lgb
    from xgboost import XGBClassifier
    tr, ref = split['train'],split['reference']
    y = records.TARGET.to_numpy()
    checkpoint = artifact/'basic.joblib'
    if checkpoint.exists():
        saved = joblib.load(checkpoint)
        guard=saved.get('input_fingerprint')
        expected=dict(config=cfg, native_data_sha256=sha(HERE/'cache/native.npy'),
                      dense_data_sha256=sha(HERE/'cache/dense.npy'),split_sha256=sha(HERE/'cache/splits.npz'),
                      training_source_sha256=sha(__file__),model=model_name,seed=seed)
        if guard != expected:
            raise ValueError('Checkpoint data/config/training code differs or predates guards; use a fresh output directory')
        obj, scaler = saved['model'], saved.get('scaler')
        predict = (lambda z: obj.predict_proba(scaler.transform(z))[:,1]) if scaler is not None else (lambda z: obj.predict_proba(z)[:,1])
        return obj,predict,saved['metadata']
    started=time.perf_counter()
    candidates=[]
    if model_name=='logistic':
        scaler=StandardScaler().fit(dense[tr])
        ztr,zref=scaler.transform(dense[tr]),scaler.transform(dense[ref])
        for C in [.1,1.0]:
            model=LogisticRegression(C=C,max_iter=600,tol=1e-5,solver='lbfgs',random_state=seed)
            model.fit(ztr,y[tr])
            candidates.append((float(roc_auc_score(y[ref],model.predict_proba(zref)[:,1])),model,dict(C=C)))
    elif model_name=='xgboost':
        scaler=None
        model=XGBClassifier(n_estimators=200,max_depth=5,learning_rate=.05,tree_method='hist',
                 device='cpu',max_bin=128,n_jobs=cfg['threads'],subsample=.9,colsample_bytree=.9,
                 reg_lambda=5,objective='binary:logistic',eval_metric='auc',random_state=seed)
        model.fit(X[tr],y[tr])
        candidates.append((float(roc_auc_score(y[ref],model.predict_proba(X[ref])[:,1])),model,dict(max_depth=5,n_estimators=200)))
    elif model_name=='lightgbm':
        scaler=None
        for leaves,child in [(31,100),(63,100),(31,250)]:
            model=lgb.LGBMClassifier(n_estimators=1200,num_leaves=leaves,max_depth=-1,
                   min_child_samples=child,learning_rate=.03,subsample=.9,subsample_freq=1,
                   colsample_bytree=.9,reg_lambda=5,n_jobs=cfg['threads'],verbosity=-1,
                   random_state=seed,deterministic=True,force_col_wise=True)
            model.fit(X[tr],y[tr],eval_set=[(X[ref],y[ref])],eval_metric='auc',
                      callbacks=[lgb.early_stopping(60,verbose=False)])
            score=float(roc_auc_score(y[ref],model.predict_proba(X[ref])[:,1]))
            candidates.append((score,model,dict(num_leaves=leaves,min_child_samples=child,best_iteration=model.best_iteration_)))
            print(f'{model_name} seed{seed}: leaves={leaves} child={child} reference AUC={score:.5f}',flush=True)
    else:
        raise ValueError(model_name)
    score,obj,parameters=max(candidates,key=lambda v:v[0])
    metadata=dict(model=model_name,seed=seed,train_seconds=time.perf_counter()-started,
                  reference_auc=score,parameters=parameters,
                  tuning=[dict(reference_auc=a,parameters=c) for a,b,c in candidates],
                  selection_split='reference',train_n=len(tr),input_features=int(dense.shape[1] if model_name=='logistic' else X.shape[1]))
    guard=dict(config=cfg,native_data_sha256=sha(HERE/'cache/native.npy'),
               dense_data_sha256=sha(HERE/'cache/dense.npy'),split_sha256=sha(HERE/'cache/splits.npz'),
               training_source_sha256=sha(__file__),model=model_name,seed=seed)
    joblib.dump(dict(model=obj,scaler=scaler,metadata=metadata,input_fingerprint=guard),checkpoint)
    predict=(lambda z: obj.predict_proba(scaler.transform(z))[:,1]) if scaler is not None else (lambda z: obj.predict_proba(z)[:,1])
    return obj,predict,metadata

def aggregate(run):
    metrics,policies,status=[],[],[]
    for path in sorted((run/'jobs').glob('*.json')):
        obj=json.loads(path.read_text(encoding='utf-8'))
        status.append({k:obj.get(k) for k in ['model','seed','status','error','completed_utc']})
        if obj['status']=='PASS':
            metrics.append(obj['model_metrics']);policies.extend(obj['policy_metrics'])
    if metrics: pd.DataFrame(metrics).to_csv(run/'model_metrics.csv',index=False)
    if policies: pd.DataFrame(policies).to_csv(run/'policy_metrics.csv',index=False)
    dump(run/'status.json',dict(updated_utc=now(),jobs=status,completed=sum(x['status']=='PASS' for x in status),
                               failed=sum(x['status']=='FAIL' for x in status)))

def execute(model_name,seed,data,cfg,run,train_override=None):
    X,dense,records,split,info=data
    name=f'{model_name}_s{seed}'
    jobpath=run/'jobs'/f'{name}.json'
    if jobpath.exists() and json.loads(jobpath.read_text(encoding='utf-8'))['status']=='PASS':
        print('Resume completed',name,flush=True);return
    dump(jobpath,dict(model=model_name,seed=seed,status='RUNNING',started_utc=now()))
    artifact=run/'checkpoints'/name
    artifact.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    try:
        if train_override is not None:
            model,predict,meta,matrix=train_override(model_name,seed,data,cfg,artifact)
        elif model_name in ('logistic','xgboost','lightgbm'):
            model,predict,meta=train_basic(model_name,X,dense,records,split,cfg,seed,artifact)
            matrix=dense if model_name=='logistic' else X
        else:
            from .advanced_search import train_advanced
            matrix=dense if model_name=='tabm' else X
            tr,ref=split['train'],split['reference']
            kw={} if model_name=='tabm' else dict(group_train=records.thin.to_numpy()[tr],group_stop=records.thin.to_numpy()[ref])
            model,predict,meta=train_advanced(model_name,matrix[tr],records.TARGET.to_numpy()[tr],
                         matrix[ref],records.TARGET.to_numpy()[ref],cfg,seed,artifact,**kw)
            meta.setdefault('train_seconds',meta.get('training_seconds',time.perf_counter()-started))
            meta.setdefault('reference_auc',meta.get('best_stop_auc',meta.get('stop_auc')))
        y=records.TARGET.to_numpy()
        cal=split['calibration']
        pcal=np.asarray(predict(matrix[cal]),dtype=float)
        platt=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(pcal,1e-7,1-1e-7)).reshape(-1,1),y[cal])
        slope,intercept=float(platt.coef_[0,0]),float(platt.intercept_[0])
        if slope<=0: raise ValueError('Nonpositive calibration slope')
        def make_scores(which):
            idx=split[which]
            raw=np.asarray(predict(matrix[idx]),dtype=float)
            frame=records.iloc[idx].drop(columns='split').reset_index(drop=True).copy()
            frame['raw_probability']=raw
            frame['probability']=expit(slope*logit(np.clip(raw,1e-7,1-1e-7))+intercept)
            return frame
        val=make_scores('validation')
        frozen=fit_policies(val,cfg)
        frozen['model']=model_name;frozen['seed']=seed;frozen['frozen_utc']=now()
        frozen['validation_ids_sha256']=sha(HERE/'cache/split_manifest.parquet')
        frozen['calibration']=dict(slope=slope,intercept=intercept)
        frozen['business_code_sha256']=sha(HERE/'business.py')
        freeze_path=run/'policies'/f'{name}.json'
        dump(freeze_path,frozen)
        # All model, calibration, threshold and tuning choices are now frozen.
        test=make_scores('test')
        val.to_parquet(run/'predictions'/f'{name}_validation.parquet',index=False)
        test.to_parquet(run/'predictions'/f'{name}_test.parquet',index=False)
        rows=evaluate_policies(test,frozen,cfg)
        for row in rows:
            row.update(model=model_name,seed=seed,method='threshold',evaluation_type='deterministic_policy')
        fair_rows,fair_rules=fairlearn_policies(val,test,cfg,name,run)
        for row in fair_rows: row.update(model=model_name,seed=seed)
        rows.extend(fair_rows)
        model_metrics=dict(model=model_name,seed=seed,auc=float(roc_auc_score(test.TARGET,test.probability)),
             average_precision=float(average_precision_score(test.TARGET,test.probability)),
             brier=float(brier_score_loss(test.TARGET,test.probability)),
             raw_brier=float(brier_score_loss(test.TARGET,test.raw_probability)),
             train_seconds=meta['train_seconds'],calibration_slope=slope,calibration_intercept=intercept,
             n_test=len(test),reference_auc=meta.get('reference_auc'),
             input_features=meta.get('input_features',meta.get('n_features')))
        dump(jobpath,dict(model=model_name,seed=seed,status='PASS',completed_utc=now(),
             wall_seconds=time.perf_counter()-started,metadata=meta,model_metrics=model_metrics,
             frozen_policy_sha256=sha(freeze_path),policy_metrics=rows,
             prediction_sha256={s:sha(run/'predictions'/f'{name}_{s}.parquet') for s in ['validation','test']}))
        print(f'PASS {name}: test AUC={model_metrics["auc"]:.5f}, {len(rows)} policies',flush=True)
    except Exception as exc:
        dump(jobpath,dict(model=model_name,seed=seed,status='FAIL',error=str(exc),traceback=traceback.format_exc(),completed_utc=now()))
        traceback.print_exc()
        raise
    finally:
        aggregate(run)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='research_extension_v3/results/run_001')
    parser.add_argument('--models',default=None)
    parser.add_argument('--seeds',default=None)
    parser.add_argument('--aggregate',action='store_true')
    args=parser.parse_args()
    cfg=load_config();run=Path(args.output).resolve()
    run.mkdir(parents=True,exist_ok=True)
    for sub in ['jobs','predictions','policies','checkpoints','decisions']: (run/sub).mkdir(exist_ok=True)
    if args.aggregate: aggregate(run);return
    identity=run/'config_frozen.json'
    if identity.exists() and json.loads(identity.read_text(encoding='utf-8'))!=cfg: raise ValueError('Run config changed; choose new output')
    dump(identity,cfg)
    data=load_prepared(cfg)
    shutil.copyfile(HERE/'cache/split_manifest.parquet',run/'split_manifest.parquet')
    dump(run/'data_audit.json',dict(**data[-1],split_manifest_sha256=sha(run/'split_manifest.parquet')))
    models=args.models.split(',') if args.models else cfg['models']
    seeds=[int(x) for x in args.seeds.split(',')] if args.seeds else cfg['seeds']
    with threadpool_limits(limits=cfg['threads']):
        for model in models:
            for seed in seeds:
                print('Starting',model,seed,flush=True)
                execute(model,seed,data,cfg,run)
    dump(run/'runtime_versions.json',{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','pyarrow']})
    aggregate(run)

if __name__=='__main__': main()
