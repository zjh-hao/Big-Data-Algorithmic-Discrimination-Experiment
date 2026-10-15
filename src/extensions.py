"""Modules B (decision objectives) and C (selective-label feedback simulation)."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits
from experiment import ROOT, load_features, biased_indices, make_model, choose_threshold, metrics, dump, sha

CONFIG=json.loads((ROOT/'extensions_config.json').read_text(encoding='utf-8'))
OUT=ROOT/'results/extensions'
MOD=ROOT/'models/extensions'


def clean_json(value):
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def safe_dump(path,value): dump(path,clean_json(value))


def score_metrics(y,g,p,a):
    result=metrics(y,g,p,a)
    result['accuracy']=float(accuracy_score(y,~a))
    result['approved_count']=int(a.sum())
    return result


def threshold_points(p,y,grid=None,extra=None):
    order=np.argsort(p,kind='stable'); s=p[order]; labels=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:])+1,len(s)]
    if grid and len(ends)>grid:
        ends=ends[np.unique(np.rint(np.linspace(0,len(ends)-1,grid)).astype(int))]
    thresholds=np.r_[-1.,s[ends-1]]
    if extra is not None: thresholds=np.unique(np.r_[thresholds,extra])
    counts=np.searchsorted(s,thresholds,side='right')
    bad=np.r_[0,np.cumsum(labels)][counts]
    good=counts-bad
    return thresholds,counts,bad,good


def accuracy_rule(p,y):
    t,n,bad,good=threshold_points(p,y)
    # Predicted TARGET=1 for rejected: error = approved bad + rejected good.
    errors=bad+(int((y==0).sum())-good)
    i=np.flatnonzero(errors==errors.min())[0]
    return {'thresholds':[float(t[i])]*2,'feasible':True,'validation_accuracy':float(1-errors[i]/len(y))}


def fair_rule(p,y,g,budget,tolerance,minimum,grid,common_threshold):
    data=[]
    for group in [0,1]:
        mask=g==group
        if not np.any(y[mask]==0): raise ValueError('No normal-label samples in group')
        t,n,bad,good=threshold_points(p[mask],y[mask],grid,common_threshold)
        frr=1-good/(y[mask]==0).sum()
        data.append((t,n,bad,frr))
    t0,n0,b0,f0=data[0]; t1,n1,b1,f1=data[1]
    n=n0[:,None]+n1[None,:]; bad=b0[:,None]+b1[None,:]
    gap=np.abs(f0[:,None]-f1[None,:])
    possible=(n>=minimum)&(bad<=budget*n+1e-12)&(gap<=tolerance+1e-12)
    i,j=np.where(possible)
    if not len(i): return {'thresholds':[-1.,-1.],'feasible':False,'tolerance':tolerance}
    order=np.lexsort((j,i,gap[i,j],bad[i,j]/n[i,j],-n[i,j]))
    a,b=i[order[0]],j[order[0]]
    return {'thresholds':[float(t0[a]),float(t1[b])],'feasible':True,'tolerance':tolerance,
            'validation_approvals':int(n[a,b]),'validation_risk':float(bad[a,b]/n[a,b]),
            'validation_abs_frr_gap':float(gap[a,b]),'grid_counts':[len(t0),len(t1)]}


def apply_rule(p,g,rule):
    return p<=np.asarray(rule['thresholds'])[g]


def objective_module():
    decisions=[]
    for model_name in CONFIG['objective']['base_models']:
        v=pd.read_csv(ROOT/f'results/predictions/{model_name}_validation.csv',float_precision='round_trip')
        p=v.probability.to_numpy(dtype=np.float32); y=v.TARGET.to_numpy(); g=v.thin.to_numpy()
        business=choose_threshold(p,y,CONFIG['risk_budget'],CONFIG['minimum_validation_approvals'])
        policies={'accuracy':accuracy_rule(p,y), 'business':dict(business,thresholds=[business['threshold']]*2)}
        policies['business_group']=fair_rule(p,y,g,CONFIG['risk_budget'],1.,
                     CONFIG['minimum_validation_approvals'],CONFIG['objective']['threshold_grid_size_per_group'],business['threshold'])
        for tol in CONFIG['objective']['fairness_tolerances']:
            policies[f'fair_{int(round(tol*100)):02d}']=fair_rule(p,y,g,CONFIG['risk_budget'],tol,
                     CONFIG['minimum_validation_approvals'],CONFIG['objective']['threshold_grid_size_per_group'],business['threshold'])
        for policy,rule in policies.items():
            a=apply_rule(p,g,rule)
            decisions.append({'base_model':model_name,'policy':policy,'rule':rule,
                              'validation_metrics':score_metrics(y,g,p,a)})
    safe_dump(OUT/'objective/frozen_decisions.json',decisions)
    results=[]
    for model_name in CONFIG['objective']['base_models']:
        pframe=pd.read_csv(ROOT/f'results/predictions/{model_name}_test.csv',float_precision='round_trip')
        p=pframe.probability.to_numpy(dtype=np.float32); y=pframe.TARGET.to_numpy(); g=pframe.thin.to_numpy()
        exported=pframe[['SK_ID_CURR','TARGET','thin']].copy()
        exported['probability']=p
        for d in [r for r in decisions if r['base_model']==model_name]:
            a=apply_rule(p,g,d['rule'])
            exported[d['policy']]=a.astype(int)
            record={'base_model':model_name,'policy':d['policy'],'feasible':d['rule']['feasible'],**score_metrics(y,g,p,a)}
            record['test_risk_exceeds_budget']=bool(record['approved_risk'] is not None and record['approved_risk']>CONFIG['risk_budget'])
            record['test_gap_exceeds_tolerance']=bool('tolerance' in d['rule'] and record['abs_frr_gap']>d['rule']['tolerance'])
            results.append(record)
        exported.to_csv(OUT/f'objective/{model_name}_test.csv',index=False)
    pd.DataFrame(results).to_csv(OUT/'objective/metrics.csv',index=False)
    print('Module B: 42 decision policies frozen and evaluated.',flush=True)


def cal_predict(model,x,coef):
    raw=model.predict_proba(x)[:,1].astype(np.float64)
    return expit(coef['slope']*logit(np.clip(raw,1e-7,1-1e-7))+coef['intercept'])


def fit_and_freeze(name,matrix,observed,y,g,weights,cal,val,ids):
    # observed lists only revealed records, never candidates still awaiting a decision.
    if len(np.unique(observed))!=len(observed): raise ValueError('Training contains repeated customer')
    model=make_model(CONFIG['feedback']['parameters'])
    normalized=weights/weights.mean()
    start=time.perf_counter()
    model.fit(matrix[observed],y[observed],sample_weight=normalized)
    raw=model.predict_proba(matrix[cal])[:,1].astype(float)
    lr=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(raw,1e-7,1-1e-7)).reshape(-1,1),y[cal])
    coef={'slope':float(lr.coef_[0,0]),'intercept':float(lr.intercept_[0])}
    if coef['slope']<=0: raise ValueError('Invalid calibrator')
    pv=cal_predict(model,matrix[val],coef)
    rule=choose_threshold(pv,y[val],CONFIG['risk_budget'],CONFIG['minimum_validation_approvals'])
    used_cells=2*g[observed]+y[observed]
    counts=np.bincount(used_cells,minlength=4)
    weighted=np.bincount(used_cells,weights=weights,minlength=4)
    decision={'name':name,'calibration':coef,'rule':rule,'train_size':len(observed),
              'thin_good_share':float(counts[2]/counts.sum()),
              'weighted_thin_good_share':float(weighted[2]/weighted.sum()),
              'effective_sample_size':float(weights.sum()**2/(weights**2).sum()),
              'max_normalized_weight':float(normalized.max()),'seconds':time.perf_counter()-start,
              'train_cells':counts.tolist()}
    model.save_model(MOD/f'{name}.json')
    np.savez_compressed(OUT/f'feedback/training/{name}.npz',ids=ids[observed],weight=weights)
    safe_dump(MOD/f'{name}_decision.json',decision)
    return model,decision


def reveal_candidates(proposed,uniform,epsilon):
    explored=(~proposed)&(uniform<epsilon)
    observed=proposed|explored
    probability=np.where(proposed,1.,epsilon)
    return observed,explored,probability


def feedback_module():
    print('Module C: prepare features and five independent simulated streams.',flush=True)
    x,y,g,ids,cat,audit=load_features()
    manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv').set_index('SK_ID_CURR')
    assignment=manifest.loc[ids,'split'].to_numpy()
    pool=np.flatnonzero(assignment=='train')
    cal=np.flatnonzero(assignment=='calibration'); val=np.flatnonzero(assignment=='validation')
    test=np.flatnonzero(assignment=='test')
    pd.DataFrame({'SK_ID_CURR':ids[test],'TARGET':y[test],'thin':g[test]}).to_csv(OUT/'feedback/test_population.csv',index=False)
    safe_dump(OUT/'feedback/source_audit.json',audit)
    references=[]; events=[]
    for seed in CONFIG['feedback']['seeds']:
        checkpoint=OUT/f'feedback/seed_{seed}_complete.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            references.extend(saved['models']); events.extend(saved['events'])
            print(f'Resume completed simulation seed {seed}',flush=True)
            continue
        order=np.random.default_rng(seed).permutation(pool)
        n0=int(len(pool)*CONFIG['feedback']['initial_fraction'])
        initial=order[:n0]
        batches=np.array_split(order[n0:],CONFIG['feedback']['rounds'])
        base=initial[biased_indices(y[initial],g[initial],CONFIG['feedback']['initial_thin_good_retention'],seed+500)]
        # Fit vocabulary on the actual initial training data, not future arrivals.
        pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cat)],remainder='passthrough')
        pre.fit(x.iloc[base]); matrix=pre.transform(x).astype(np.float32)
        joblib.dump(pre,MOD/f'seed_{seed}_preprocessor.joblib')
        stream_rows=[pd.DataFrame({'SK_ID_CURR':ids[initial],'arrival_round':0,'initial_observed':np.isin(initial,base).astype(int)})]
        stream_rows.extend(pd.DataFrame({'SK_ID_CURR':ids[b],'arrival_round':r,'initial_observed':0}) for r,b in enumerate(batches,1))
        pd.concat(stream_rows).to_csv(OUT/f'feedback/stream_{seed}.csv',index=False)
        initial_name=f's{seed}_initial'
        base_model,base_decision=fit_and_freeze(initial_name,matrix,base,y,g,np.ones(len(base)),cal,val,ids)
        states={branch:{'model':base_model,'decision':base_decision,'observed':base.copy(),'weight':np.ones(len(base))} for branch in CONFIG['feedback']['branches']}
        seed_refs=[]; seed_events=[]
        for branch in states: seed_refs.append({'seed':seed,'round':0,'branch':branch,'model':initial_name})
        for round_number,batch in enumerate(batches,1):
            uniform=np.random.default_rng(seed*100+round_number).random(len(batch))
            for branch,state in states.items():
                p=cal_predict(state['model'],matrix[batch],state['decision']['calibration'])
                proposed=p<=state['decision']['rule']['threshold']
                epsilon=CONFIG['feedback']['exploration_probability'] if branch.startswith('explore') else 0.
                visible,explored,pi=reveal_candidates(proposed,uniform,epsilon)
                revealed=batch[visible]
                # Persist an observable ledger: outcomes remain blank for unobserved customers.
                ledger=pd.DataFrame({'SK_ID_CURR':ids[batch],'thin':g[batch],'probability':p,
                        'proposed':proposed.astype(int),'explored':explored.astype(int),'visible':visible.astype(int),
                        'inclusion_probability':pi,'outcome':np.where(visible,y[batch],np.nan)})
                ledger.to_csv(OUT/f'feedback/ledgers/s{seed}_{branch}_r{round_number}.csv.gz',index=False,compression='gzip')
                event={'seed':seed,'round':round_number,'branch':branch,'arrivals':len(batch),
                       'actual_approved':int(visible.sum()),'exploration_approved':int(explored.sum()),
                       'actual_approved_risk':float(y[revealed].mean()) if len(revealed) else None,
                       'exploration_risk':float(y[batch[explored]].mean()) if explored.any() else None,
                       'thin_observation_rate':float(visible[g[batch]==1].mean()),
                       'thick_observation_rate':float(visible[g[batch]==0].mean()),
                       'thin_good_in_observed':int(((g[revealed]==1)&(y[revealed]==0)).sum()),
                       'source_model':state['decision']['name']}
                seed_events.append(event)
                if branch!='static':
                    new_weights=1/pi[visible] if branch=='explore_ipw' else np.ones(len(revealed))
                    state['observed']=np.r_[state['observed'],revealed]
                    state['weight']=np.r_[state['weight'],new_weights]
                    name=f's{seed}_{branch}_r{round_number}'
                    state['model'],state['decision']=fit_and_freeze(name,matrix,state['observed'],y,g,state['weight'],cal,val,ids)
                seed_refs.append({'seed':seed,'round':round_number,'branch':branch,'model':state['decision']['name']})
            print(f'Simulation seed {seed}, round {round_number}/5 frozen.',flush=True)
        safe_dump(checkpoint,{'models':seed_refs,'events':seed_events})
        references.extend(seed_refs);events.extend(seed_events)
        del matrix
    safe_dump(OUT/'feedback/frozen_decisions.json',references)
    pd.DataFrame(events).to_csv(OUT/'feedback/stream_metrics.csv',index=False)
    # No test labels influence the online policy or round selection. Evaluate only
    # after ALL seed trajectories have been frozen.
    evaluations=[]
    for seed in CONFIG['feedback']['seeds']:
        pre=joblib.load(MOD/f'seed_{seed}_preprocessor.joblib')
        test_x=pre.transform(x.iloc[test]).astype(np.float32)
        cache={}
        for row in [r for r in references if r['seed']==seed]:
            name=row['model']
            if name not in cache:
                d=json.loads((MOD/f'{name}_decision.json').read_text())
                model=make_model(CONFIG['feedback']['parameters']);model.load_model(MOD/f'{name}.json')
                p=cal_predict(model,test_x,d['calibration']);a=p<=d['rule']['threshold']
                np.savez_compressed(OUT/f'feedback/predictions/{name}.npz',probability=p,approved=a)
                cache[name]={**score_metrics(y[test],g[test],p,a),**{k:d[k] for k in ['train_size','thin_good_share','weighted_thin_good_share','effective_sample_size','max_normalized_weight']}}
            evaluations.append({**row,**cache[name]})
    pd.DataFrame(evaluations).to_csv(OUT/'feedback/test_metrics.csv',index=False)
    print('Module C: all 120 seed/branch/round evaluations complete.',flush=True)


def main():
    start=time.perf_counter()
    for directory in ['objective','feedback/training','feedback/ledgers','feedback/predictions']:
        (OUT/directory).mkdir(parents=True,exist_ok=True)
    MOD.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'config.json',ROOT/'extensions_config.json',ROOT/'EXTENSION_PROTOCOL.md',Path(__file__),ROOT/'src/experiment.py',ROOT/'data/processed/split_manifest.csv']}
    # Bind extension inputs to the original validated data and score files.
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'data/raw').glob('*.parquet')})
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results/predictions').glob('*_validation.csv')})
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'results/predictions').glob('*_test.csv')})
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    identity=OUT/'identity.json'
    if identity.exists() and json.loads(identity.read_text())['fingerprint']!=fingerprint:
        raise RuntimeError('Extension protocol/code/input changed; preserve previous run and use a fresh extension output directory.')
    safe_dump(identity,{'fingerprint':fingerprint,'hashes':hashes,'configuration':CONFIG,'time_started_unix':time.time()})
    with threadpool_limits(limits=CONFIG['threads']):
        objective_module()
        feedback_module()
    safe_dump(OUT/'completion.json',{'seconds_this_invocation':time.perf_counter()-start,'fingerprint':fingerprint,
         'objective_policies':42,'feedback_fits':80,'feedback_seed_branch_round_evaluations':120,'completed_unix':time.time()})


if __name__=='__main__': main()
