"""CPU score recovery, freeze, evaluation and isolated selective-label replay."""
import os
for _key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ[_key] = '4'
import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from research_extension_v2.common import HERE, ROOT, CFG, sha, dump, metrics, read_scores, validate
from research_extension_v2.policy_noharm import fit_policies, approve
from research_extension_v2.subgroup_audit import fit_income_boundaries, income_groups, audit
from research_extension_v2.selective_label_ipw import exploration_plan, reveal, estimate, oracle_metrics
from research_extension_v2.exploration_risk import risk_audit
from research_extension_v2.statistics import policy_intervals, auc_interval, replay_intervals
from experiment import load_features, split_indices, biased_indices, make_model


def snapshot():
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '5a43e3758933f3440b6c53a1dd2224407c62577d'], cwd=ROOT, text=True).splitlines()
    return {f: sha(ROOT/f) for f in files}


def prepare(out):
    x, y, g, ids, cats, data_audit = load_features()
    split = split_indices(y, g, 20261013)
    legacy = ROOT/'data/processed/split_manifest.csv'
    if legacy.exists():
        old = pd.read_csv(legacy).set_index('SK_ID_CURR')
        for kind, index in split.items():
            if set(ids[index]) != set(old.index[old.split==kind]): raise ValueError('Legacy split differs')
    assignment = np.empty(len(y), dtype=object)
    for kind, index in split.items(): assignment[index] = kind
    manifest = pd.DataFrame(dict(SK_ID_CURR=ids, split=assignment, TARGET=y, thin=g))
    manifest.to_csv(out/'split_manifest.csv', index=False)
    dump(out/'data_audit.json', data_audit)
    tr, cal, val, test = [split[k] for k in ['train', 'calibration', 'validation', 'test']]
    cuts = fit_income_boundaries(x.iloc[tr].AMT_INCOME_TOTAL.to_numpy(), CFG['income_quantiles'])
    dump(out/'income_boundaries.json', dict(cuts=cuts, training_ids_sha256=sha(out/'split_manifest.csv'), source='full original training split only', quantiles=CFG['income_quantiles']))
    frozen = {}; sources = []; models = {}
    specifications = [('full', tr)] + [(f'r50_s{s}_biased', tr[biased_indices(y[tr], g[tr], CFG['retention'], s)]) for s in CFG['sampling_seeds']]
    pre = None; matrix = None
    for name, used in specifications:
        folder = out/'scores'/name; folder.mkdir(parents=True)
        vp = ROOT/f'results/predictions/{name}_validation.csv'; tp = ROOT/f'results/predictions/{name}_test.csv'
        if vp.exists() and tp.exists():
            v = read_scores(vp)
            expected = manifest.set_index('SK_ID_CURR').loc[v.SK_ID_CURR]
            if set(v.SK_ID_CURR)!=set(ids[val]) or not np.array_equal(v.TARGET, expected.TARGET) or not np.array_equal(v.thin, expected.thin):
                raise ValueError('Legacy validation alignment mismatch')
            v.to_csv(folder/'validation.csv', index=False)
            models[name] = ('reuse', tp)
            sources.append(dict(model=name, status='REUSED_PREDICTIONS', validation=str(vp), test=str(tp), validation_sha=sha(vp), test_sha=sha(tp)))
        else:
            if pre is None:
                pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), cats)], remainder='passthrough', verbose_feature_names_out=False)
                pre.fit(x.iloc[tr]); matrix = pre.transform(x).astype(np.float32)
                joblib.dump(pre, out/'models/preprocessor.joblib')
            model = make_model(CFG['parameters']).fit(matrix[used], y[used], sample_weight=np.ones(len(used)))
            raw = model.predict_proba(matrix[cal])[:, 1].astype(float)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(np.clip(raw, 1e-7, 1-1e-7)).reshape(-1, 1), y[cal])
            coef = dict(slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))
            if coef['slope']<=0: raise ValueError('Nonmonotonic calibrator')
            def score(index):
                raw = model.predict_proba(matrix[index])[:, 1].astype(float)
                return expit(coef['slope']*logit(np.clip(raw, 1e-7, 1-1e-7))+coef['intercept'])
            v = pd.DataFrame(dict(SK_ID_CURR=ids[val], TARGET=y[val], thin=g[val], probability=score(val)))
            v.to_csv(folder/'validation.csv', index=False)
            model.save_model(out/f'models/{name}.json'); dump(out/f'models/{name}_calibration.json', coef)
            pd.DataFrame(dict(SK_ID_CURR=ids[used], weight=np.ones(len(used)))).to_csv(out/f'models/{name}_training.csv', index=False)
            models[name] = ('restore', model, coef)
            sources.append(dict(model=name, status='RESTORED_FIXED_MODEL', train_n=len(used), parameters=CFG['parameters']))
        frozen[name] = fit_policies(v, CFG)
        print('FROZEN', name, 'policies', len(frozen[name]), flush=True)
    dump(out/'frozen_policies.json', dict(frozen_unix=time.time(), policies=frozen, sources=sources))
    # Every model's policy has now been frozen. Only now export/evaluate test labels.
    for name, spec in models.items():
        if spec[0]=='reuse':
            f = read_scores(spec[1]); f = f.set_index('SK_ID_CURR').loc[ids[test]].reset_index()
            if not np.array_equal(f.TARGET, y[test]) or not np.array_equal(f.thin, g[test]): raise ValueError('Test alignment mismatch')
        else:
            _, model, coef = spec
            raw = model.predict_proba(matrix[test])[:, 1].astype(float)
            p = expit(coef['slope']*logit(np.clip(raw, 1e-7, 1-1e-7))+coef['intercept'])
            f = pd.DataFrame(dict(SK_ID_CURR=ids[test], TARGET=y[test], thin=g[test], probability=p))
        f['income'] = x.iloc[test].AMT_INCOME_TOTAL.to_numpy()
        f.to_csv(out/f'scores/{name}/test.csv', index=False)
    return frozen, cuts


def evaluate(out, frozen, cuts):
    comparisons = []; intervals = []; subgroups = []; aucs = []
    for model, policies in frozen.items():
        f = read_scores(out/f'scores/{model}/test.csv')
        y, g, p = [f[c].to_numpy() for c in ['TARGET', 'thin', 'probability']]
        exported = f.copy(); decisions = {}
        baseline = policies['business']
        for key, rule in policies.items():
            row = dict(model=model, policy=key, **rule)
            if rule['feasible']:
                a = approve(p, g, rule); decisions[key] = a; exported[key] = a.astype(int)
                row.update(metrics(y, g, p, a))
                row['test_risk_violation'] = row['approved_risk'] > CFG['risk_budget']
                row['test_gap_violation'] = rule['epsilon'] is not None and row['abs_frr_gap'] > rule['epsilon']
                for group in ['thin', 'thick']:
                    row['test_'+group+'_noharm_violation'] = (rule['delta'] is not None and
                        row[group+'_frr']>metrics(y, g, p, decisions['business'])[group+'_frr']+rule['delta'])
            comparisons.append(row)
        exported.to_csv(out/f'{model}_test_decisions.csv.gz', index=False, compression='gzip')
        for r in policy_intervals(y, g, income_groups(f.income, cuts), decisions, CFG['bootstrap_repetitions'], CFG['bootstrap_seed']):
            intervals.append(dict(model=model, **r))
        primary = {k: decisions[k] for k in ['business', 'gap_e0.02', 'noharm_e0.02_d0.01'] if k in decisions}
        for r in audit(y, g, p, primary, decisions['business'], f.income, cuts, CFG['minimum_subgroup_n'], CFG['minimum_subgroup_good']): subgroups.append(dict(model=model, **r))
        aucs.append(dict(model=model, **auc_interval(y, p)))
        print('EVALUATED', model, flush=True)
    pd.DataFrame(comparisons).to_csv(out/'COMPARISON_TABLES.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    pd.DataFrame(subgroups).to_csv(out/'subgroups.csv', index=False)
    pd.DataFrame(aucs).to_csv(out/'auc_intervals.csv', index=False)


def simulate(out, frozen):
    estimates = []; risks = []
    for model, policies in frozen.items():
        business = policies['business']
        if not business['feasible']: continue
        f = read_scores(out/f'scores/{model}/test.csv'); proposed = approve(f.probability, f.thin, business)
        # All plans contain no outcome. They freeze before any reveal for this model.
        plans = {}
        for seed in CFG['simulation_seeds']:
            u = np.random.default_rng(CFG['simulation_seed_offset']+seed).random(len(f))
            for rate in CFG['exploration_rates']:
                key = f'{model}_s{seed}_e{rate:.2f}'
                plan = exploration_plan(f.SK_ID_CURR, f.thin, proposed, rate, u)
                plan.to_csv(out/f'plans/{key}.csv.gz', index=False, compression='gzip'); plans[key] = (seed, rate, plan)
        dump(out/f'plans/{model}_frozen.json', dict(frozen_unix=time.time(), plans={k:sha(out/f'plans/{k}.csv.gz') for k in plans}))
        for key, (seed, rate, plan) in plans.items():
            visible = plan.visible.to_numpy(dtype=bool)
            ledger = reveal(plan, f.loc[visible, 'SK_ID_CURR'], f.loc[visible, 'TARGET'].to_numpy())
            ledger.to_csv(out/f'ledgers/{key}.csv.gz', index=False, compression='gzip')
            est = estimate(ledger, CFG['minimum_ess'])
            oracle = oracle_metrics(f[['SK_ID_CURR', 'TARGET']], plan)
            ci = replay_intervals(ledger, f.TARGET.to_numpy(), CFG['bootstrap_repetitions'], CFG['bootstrap_seed'])
            for r in est:
                group = r['group']; r.update(oracle_frr=oracle[group], **{k:v for k,v in ci[group].items() if k!='group'})
                for method in ['naive', 'ipw']:
                    value = r[method+'_frr']; r[method+'_error'] = value-oracle[group] if value is not None else None
                estimates.append(dict(model=model, seed=seed, exploration_rate=rate, **r))
            risk = risk_audit(ledger)
            risks.append(dict(model=model, seed=seed, exploration_rate=rate, **risk,
                **{k:v for k,v in ci[0].items() if '_risk_' in k},
                actual_exceeds_budget=risk['actual_risk'] is not None and risk['actual_risk']>CFG['risk_budget']))
        print('REPLAYED', model, '15 fixed plans', flush=True)
    pd.DataFrame(estimates).to_csv(out/'ipw_estimates.csv', index=False)
    pd.DataFrame(risks).to_csv(out/'exploration_risk.csv', index=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default=str(HERE/'results/run_001'))
    args = parser.parse_args(); out = Path(args.output).resolve()
    if out.exists(): raise FileExistsError('Immutable run directory exists; choose a fresh --output')
    out.mkdir(parents=True)
    for name in ['scores', 'models', 'plans', 'ledgers']: (out/name).mkdir()
    start = time.time()
    original = snapshot(); dump(out/'legacy_hashes.json', original)
    inputs = [HERE/'config.json', HERE/'PROTOCOL.md'] + sorted(HERE.glob('*.py')) + [ROOT/'src/experiment.py', ROOT/'src/extensions.py', ROOT/'config.json']
    inputs += [p for p in (ROOT/'data/raw').iterdir() if p.suffix in ['.parquet', '.csv']]
    dump(out/'identity.json', dict(started_unix=start, command=sys.argv, python=sys.version, platform=platform.platform(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}, config=CFG,
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','pyarrow','matplotlib','joblib']}))
    try:
        with threadpool_limits(limits=CFG['threads']):
            frozen, cuts = prepare(out); evaluate(out, frozen, cuts); simulate(out, frozen)
        if snapshot()!=original: raise RuntimeError('Legacy tracked file changed')
        dump(out/'completion.json', dict(status='PASS', seconds=time.time()-start, models=len(frozen),
            policy_rows=sum(len(p) for p in frozen.values()), budgeted_exploration='NOT_RUN'))
    except Exception as error:
        dump(out/'failure.json', dict(status='BLOCKED_NOT_RUN' if isinstance(error, FileNotFoundError) else 'FAIL', error=repr(error)))
        raise


if __name__ == '__main__': main()
