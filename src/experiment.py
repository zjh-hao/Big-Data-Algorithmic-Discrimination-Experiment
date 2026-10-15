"""Local CPU credit-risk experiment. Freeze all decisions before test evaluation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = str(CFG['threads'])
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, DMatrix


def dump(path, value):
    def convert(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(str(type(x)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def inputs():
    raw = ROOT / 'data/raw'
    wanted = ['application_train.csv', 'bureau.csv', 'HomeCredit_columns_description.csv']
    for archive in raw.glob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                base = Path(name).name
                if base in wanted and not (raw / base).exists():
                    with z.open(name) as source, (raw / base).open('wb') as dest:
                        shutil.copyfileobj(source, dest)
    app = raw / 'application_train.csv'
    bureau = raw / 'bureau.csv'
    if not app.exists(): app = raw / 'app_train.parquet'
    if not bureau.exists(): bureau = raw / 'bureau.parquet'
    desc = raw / wanted[2]
    missing = [str(p) for p in (app, bureau, desc) if not p.exists()]
    if missing: raise FileNotFoundError('Missing data: ' + ', '.join(missing))
    return app, bureau, desc


def read_table(path):
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def load_features():
    app_path, bureau_path, desc_path = inputs()
    app = read_table(app_path)
    if app.shape != (307511, 122): raise ValueError(f'Unexpected application shape: {app.shape}')
    if not app.SK_ID_CURR.is_unique or app.SK_ID_CURR.isna().any(): raise ValueError('Application IDs invalid')
    if app.TARGET.value_counts().to_dict() != {0: 282686, 1: 24825}: raise ValueError('Unexpected original labels')
    # Description confirms target means payment difficulties; it does not expose X/Y.
    desc = pd.read_csv(desc_path, encoding='cp1252')
    target_desc = str(desc.loc[desc.Row == 'TARGET', 'Description'].iloc[0])
    if 'payment difficulties' not in target_desc.lower(): raise ValueError('Unrecognized target definition')
    cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT', 'DAYS_CREDIT_UPDATE',
            'CREDIT_ACTIVE', 'CREDIT_DAY_OVERDUE', 'AMT_CREDIT_SUM',
            'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_OVERDUE']
    if bureau_path.suffix == '.parquet':
        pf = pq.ParquetFile(bureau_path)
        chunks = (b.to_pandas() for b in pf.iter_batches(batch_size=200000, columns=cols))
    else:
        chunks = pd.read_csv(bureau_path, usecols=cols, chunksize=200000)
    parts, bureau_ids = [], []
    total = future = unlinked = 0
    known = set(app.SK_ID_CURR)
    for b in chunks:
        total += len(b)
        bureau_ids.append(b.SK_ID_BUREAU.to_numpy())
        invalid = (b.DAYS_CREDIT > 0) | (b.DAYS_CREDIT_UPDATE > 0)
        future += int(invalid.sum())
        unlinked += int((~b.SK_ID_CURR.isin(known)).sum())
        b = b.loc[b.SK_ID_CURR.isin(known) & ~invalid].copy()
        b['ACTIVE'] = (b.CREDIT_ACTIVE == 'Active').astype(int)
        part = b.groupby('SK_ID_CURR').agg(
            BUREAU_COUNT=('SK_ID_BUREAU', 'size'), BUREAU_ACTIVE=('ACTIVE', 'sum'),
            BUREAU_DAYS_MIN=('DAYS_CREDIT', 'min'), BUREAU_DAYS_MAX=('DAYS_CREDIT', 'max'),
            BUREAU_OVERDUE_MAX=('CREDIT_DAY_OVERDUE', 'max'),
            BUREAU_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum'),
            BUREAU_DEBT_SUM=('AMT_CREDIT_SUM_DEBT', 'sum'),
            BUREAU_DEBT_OBS=('AMT_CREDIT_SUM_DEBT', 'count'),
            BUREAU_OVERDUE_SUM=('AMT_CREDIT_SUM_OVERDUE', 'sum'))
        parts.append(part)
    all_ids = np.concatenate(bureau_ids)
    if total != 1716428 or len(np.unique(all_ids)) != total: raise ValueError('Incomplete/duplicate bureau table')
    reducers = {c: ('min' if c.endswith('_MIN') else 'max' if c.endswith('_MAX') else 'sum') for c in parts[0].columns}
    bureau = pd.concat(parts).groupby(level=0).agg(reducers)
    app = app.join(bureau, on='SK_ID_CURR', validate='one_to_one')
    app['BUREAU_COUNT'] = app.BUREAU_COUNT.fillna(0).astype(int)
    app['BUREAU_MISSING'] = (app.BUREAU_COUNT == 0).astype(int)
    # Absence means unobserved history, NOT a risk label. Other aggregates retain NaN.
    app.loc[app.BUREAU_DEBT_OBS == 0, 'BUREAU_DEBT_SUM'] = np.nan
    g = (app.BUREAU_COUNT <= 1).to_numpy(dtype=np.int8)
    y = app.TARGET.to_numpy(dtype=np.int8)
    ids = app.SK_ID_CURR.to_numpy()
    app['EMPLOYMENT_SENTINEL'] = (app.DAYS_EMPLOYED == 365243).astype(int)
    app['DAYS_EMPLOYED'] = app.DAYS_EMPLOYED.replace(365243, np.nan)
    for name, numerator, denominator in [
        ('CREDIT_INCOME_RATIO', 'AMT_CREDIT', 'AMT_INCOME_TOTAL'),
        ('ANNUITY_INCOME_RATIO', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'),
        ('ANNUITY_CREDIT_RATIO', 'AMT_ANNUITY', 'AMT_CREDIT'),
        ('BUREAU_DEBT_CREDIT_RATIO', 'BUREAU_DEBT_SUM', 'BUREAU_CREDIT_SUM')]:
        app[name] = app[numerator] / app[denominator].where(app[denominator] > 0)
    x = app.drop(columns=['SK_ID_CURR', 'TARGET', 'CODE_GENDER']).replace([np.inf, -np.inf], np.nan)
    cat = x.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    x[cat] = x[cat].fillna('__MISSING__').astype(str)
    numeric = [c for c in x if c not in cat]
    x[numeric] = x[numeric].astype(np.float32)
    audit = {'applications': len(app), 'original_application_columns': 122, 'bureau_rows': total,
             'bureau_future_rows_excluded': future, 'bureau_rows_outside_labeled_applications': unlinked,
             'target_description': target_desc, 'target_1': int(y.sum()), 'thin_count': int(g.sum()),
             'thin_definition': '0 or 1 observable bureau records after time checks',
             'feature_count_before_encoding': len(x.columns), 'feature_columns': list(x.columns),
             'source_files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in inputs()}}
    return x, y, g, ids, cat, audit


def split_indices(y, g, seed):
    all_idx = np.arange(len(y))
    train, rest = train_test_split(all_idx, train_size=.5, stratify=2*g+y, random_state=seed)
    ref, rest = train_test_split(rest, train_size=.2, stratify=(2*g+y)[rest], random_state=seed+1)
    cal, rest = train_test_split(rest, train_size=.25, stratify=(2*g+y)[rest], random_state=seed+2)
    val, test = train_test_split(rest, train_size=1/3, stratify=(2*g+y)[rest], random_state=seed+3)
    return {k: np.sort(v) for k, v in zip(['train', 'reference', 'calibration', 'validation', 'test'], [train, ref, cal, val, test])}


def biased_indices(y, g, retention, seed):
    eligible = np.flatnonzero((g == 1) & (y == 0))
    keep = np.random.default_rng(seed).choice(eligible, int(np.floor(len(eligible)*retention)), replace=False)
    return np.sort(np.r_[np.flatnonzero(~((g == 1) & (y == 0))), keep])


def correction_weights(y, g, ref_y, ref_g):
    observed = np.bincount(2*g+y, minlength=4).astype(float)
    reference = np.bincount(2*ref_g+ref_y, minlength=4).astype(float)
    if min(observed.min(), reference.min()) == 0: raise ValueError('Empty group/label cell')
    cell_weights = (reference/reference.sum()) / (observed/observed.sum())
    return cell_weights[2*g+y], cell_weights


def choose_threshold(p, y, budget, minimum):
    order = np.argsort(p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    # Only evaluate complete ties, so identical scores cannot receive different decisions.
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(p)-1]
    n = ends + 1
    risk = np.cumsum(sorted_y)[ends] / n
    ok = (n >= minimum) & (risk <= budget)
    if not ok.any(): return {'threshold': -1.0, 'validation_approvals': 0, 'validation_risk': None, 'feasible': False}
    j = np.flatnonzero(ok)[-1]
    return {'threshold': float(sorted_p[ends[j]]), 'validation_approvals': int(n[j]),
            'validation_risk': float(risk[j]), 'feasible': True}


def calibrated(raw, coefficients):
    return expit(coefficients['slope'] * logit(np.clip(raw, 1e-7, 1-1e-7)) + coefficients['intercept'])


def metrics(y, g, p, approved):
    out = {'auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
           'brier': float(brier_score_loss(y, p)), 'approval_rate': float(approved.mean()),
           'approved_risk': float(y[approved].mean()) if approved.any() else None}
    ece = 0.
    for cell in np.array_split(np.argsort(p), 10):
        ece += len(cell)/len(p) * abs(p[cell].mean()-y[cell].mean())
    out['ece_10_equal_count'] = float(ece)
    for group, label in [(1, 'thin'), (0, 'thick')]:
        mask = g == group
        good = mask & (y == 0)
        out[label+'_frr'] = float((~approved[good]).mean())
        out[label+'_approval'] = float(approved[mask].mean())
        out[label+'_calibration_bias'] = float((p[mask]-y[mask]).mean())
    out['signed_frr_gap'] = out['thin_frr'] - out['thick_frr']
    out['abs_frr_gap'] = abs(out['signed_frr_gap'])
    return out


def make_model(parameters):
    return XGBClassifier(**parameters, learning_rate=.05, tree_method='hist', device='cpu',
                         max_bin=128, n_jobs=CFG['threads'], subsample=.9, colsample_bytree=.9,
                         reg_lambda=5, objective='binary:logistic', eval_metric='auc',
                         random_state=CFG['model_seed'])


def run():
    start = time.perf_counter()
    for folder in ['models', 'results/predictions', 'data/processed', 'figures']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    print('Loading and validating full application + bureau data...', flush=True)
    x, y, g, ids, cat, audit = load_features()
    split = split_indices(y, g, CFG['split_seed'])
    split_rows = []
    for name, index in split.items():
        counts = np.bincount(2*g[index]+y[index], minlength=4)
        if counts.min() < CFG['minimum_cell_size']: raise ValueError(f'Insufficient cell size: {name}')
        audit.setdefault('split_cells', {})[name] = counts.tolist()
        split_rows.append(pd.DataFrame({'SK_ID_CURR': ids[index], 'split': name, 'TARGET': y[index], 'thin': g[index]}))
    pd.concat(split_rows).to_csv(ROOT/'data/processed/split_manifest.csv', index=False)
    dump(ROOT/'results/data_audit.json', audit)
    signature = hashlib.sha256((json.dumps(CFG, sort_keys=True)+json.dumps(audit['source_files'], sort_keys=True)+sha(__file__)).encode()).hexdigest()
    identity = ROOT/'results/run_identity.json'
    if identity.exists() and json.loads(identity.read_text())['signature'] != signature:
        raise RuntimeError('Config/data/code changed: use a fresh output directory rather than mixing runs.')
    dump(identity, {'signature': signature, 'config': CFG, 'python': sys.version, 'platform': platform.platform(),
                    'started_utc': datetime.now(timezone.utc).isoformat(), 'device': 'cpu', 'threads': CFG['threads']})
    train, ref, cal, val, test = [split[k] for k in ['train','reference','calibration','validation','test']]
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), cat)],
                            remainder='passthrough', verbose_feature_names_out=False)
    pre.fit(x.iloc[train])
    # Numeric NaN is native XGBoost missingness, not median imputation fitted on test.
    matrix = pre.transform(x).astype(np.float32)
    del x
    joblib.dump(pre, ROOT/'models/preprocessor.joblib')
    feature_names = list(pre.get_feature_names_out())
    dump(ROOT/'results/features.json', feature_names)
    print(f'Matrix: {matrix.shape}, {matrix.nbytes/2**20:.1f} MiB', flush=True)
    tuning = []
    selection_path = ROOT/'results/selection.json'
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())['parameters']
    else:
        for params in CFG['candidates']:
            model = make_model(params).fit(matrix[train], y[train])
            auc = float(roc_auc_score(y[val], model.predict_proba(matrix[val])[:,1]))
            tuning.append({'parameters': params, 'validation_auc': auc})
            print(f'Tuning {params}: validation AUC={auc:.5f}', flush=True)
        selected = max(tuning, key=lambda r: r['validation_auc'])['parameters']
        dump(selection_path, {'parameters': selected, 'candidates': tuning, 'criterion': 'validation AUC, candidate order breaks ties'})
    specifications = [('full', 1., 0, 'full', np.arange(len(train)))]
    for rate in CFG['retention_rates']:
        for seed in CFG['sampling_seeds']:
            subset = biased_indices(y[train], g[train], rate, seed)
            random_subset = np.sort(np.random.default_rng(seed+10000).choice(len(train), len(subset), replace=False))
            prefix = f'r{int(rate*100)}_s{seed}'
            for condition, rows in [('random',random_subset),('biased',subset),('corrected',subset)]:
                specifications.append((prefix+'_'+condition, rate, seed, condition, rows))
    decisions = []
    # Finish every training/calibration/threshold decision BEFORE test prediction/evaluation.
    for number, (name, rate, seed, condition, rows) in enumerate(specifications, 1):
        checkpoint = ROOT/f'models/{name}.json'
        decision_path = ROOT/f'models/{name}_decision.json'
        if checkpoint.exists() and decision_path.exists():
            decisions.append(json.loads(decision_path.read_text()))
            print(f'Resume {number}/{len(specifications)}: {name}', flush=True)
            continue
        used = train[rows]
        weights = np.ones(len(used))
        cell_weights = np.ones(4)
        if condition == 'corrected': weights, cell_weights = correction_weights(y[used],g[used],y[ref],g[ref])
        tick = time.perf_counter()
        model = make_model(selected).fit(matrix[used], y[used], sample_weight=weights)
        raw_cal = model.predict_proba(matrix[cal])[:,1]
        lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000).fit(logit(np.clip(raw_cal,1e-7,1-1e-7)).reshape(-1,1), y[cal])
        coef = {'slope': float(lr.coef_[0,0]), 'intercept': float(lr.intercept_[0])}
        if coef['slope'] <= 0: raise ValueError('Non-increasing calibration: inspect model')
        raw_val = model.predict_proba(matrix[val])[:,1]
        p_val = calibrated(raw_val, coef)
        rules = {str(b): choose_threshold(p_val, y[val], b, CFG['minimum_validation_approvals']) for b in CFG['risk_budgets']}
        decision = {'name': name, 'retention': rate, 'seed': seed, 'condition': condition,
                    'train_size': len(used), 'cell_counts': np.bincount(2*g[used]+y[used],minlength=4).tolist(),
                    'cell_weights': cell_weights.tolist(), 'calibration': coef, 'rules': rules,
                    'seconds': time.perf_counter()-tick}
        model.save_model(checkpoint)
        pd.DataFrame({'SK_ID_CURR':ids[used], 'weight':weights}).to_csv(ROOT/f'data/processed/{name}_training.csv',index=False)
        pd.DataFrame({'SK_ID_CURR':ids[val], 'TARGET':y[val], 'thin':g[val], 'raw_probability':raw_val, 'probability':p_val}).to_csv(ROOT/f'results/predictions/{name}_validation.csv',index=False)
        dump(decision_path, decision)
        decisions.append(decision)
        print(f'Trained {number}/{len(specifications)} {name}: {len(used)} rows, {decision["seconds"]:.1f}s',flush=True)
    dump(ROOT/'results/frozen_decisions.json', decisions)
    all_metrics = []
    for decision in decisions:
        name = decision['name']
        model = make_model(selected)
        model.load_model(ROOT/f'models/{name}.json')
        raw = model.predict_proba(matrix[test])[:,1]
        p = calibrated(raw, decision['calibration'])
        predictions = pd.DataFrame({'SK_ID_CURR':ids[test], 'TARGET':y[test], 'thin':g[test], 'raw_probability':raw, 'probability':p})
        for budget in CFG['risk_budgets']:
            threshold = decision['rules'][str(budget)]['threshold']
            approved = p <= threshold
            predictions[f'approved_{budget}'] = approved.astype(int)
            record = {k:decision[k] for k in ['name','retention','seed','condition','train_size']}
            record.update(budget=budget, threshold=threshold, **metrics(y[test],g[test],p,approved))
            raw_metrics = metrics(y[test],g[test],raw,approved)
            for k in ['brier','thin_calibration_bias','thick_calibration_bias']:
                record['raw_'+k] = raw_metrics[k]
            all_metrics.append(record)
        predictions.to_csv(ROOT/f'results/predictions/{name}_test.csv',index=False)
        if name == 'full':
            # Tree SHAP contributions are in raw log-odds; not a causal explanation.
            sample = np.random.default_rng(2026).choice(test,min(1500,len(test)),replace=False)
            contributions = model.get_booster().predict(DMatrix(matrix[sample]),pred_contribs=True)
            pd.DataFrame({'feature':feature_names,'mean_abs_shap_logodds':np.abs(contributions[:,:-1]).mean(axis=0)}).sort_values('mean_abs_shap_logodds',ascending=False).to_csv(ROOT/'results/feature_importance.csv',index=False)
    pd.DataFrame(all_metrics).to_csv(ROOT/'results/metrics.csv',index=False)
    dump(ROOT/'results/completion.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
         'wall_seconds_this_invocation':time.perf_counter()-start,'models':len(decisions),
         'sum_training_seconds':sum(d['seconds'] for d in decisions), 'signature':signature,
         'versions':{pkg:importlib.metadata.version(pkg) for pkg in ['numpy','pandas','scikit-learn','xgboost','pyarrow']}})
    print('All training and test evaluations complete.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-data',action='store_true')
    args = parser.parse_args()
    if args.check_data:
        for p in inputs(): print(p.name, p.stat().st_size, 'bytes')
    else:
        with threadpool_limits(limits=CFG['threads']): run()
