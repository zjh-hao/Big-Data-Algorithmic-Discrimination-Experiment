"""Independently verify decision constraints and each selective-feedback ledger."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from experiment import ROOT, sha
from extensions import OUT, MOD, CONFIG, safe_dump


def recompute(y,g,p,a):
    f1=np.mean(~a[(g==1)&(y==0)]); f0=np.mean(~a[(g==0)&(y==0)])
    return {'thin_frr':f1,'thick_frr':f0,'abs_frr_gap':abs(f1-f0),
            'approval_rate':np.mean(a),'approved_risk':np.mean(y[a]) if np.any(a) else np.nan,
            'auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p),'accuracy':accuracy_score(y,~a)}


def main():
    checks=[]
    def check(condition,name):
        if not bool(condition):raise AssertionError(name)
        checks.append(name)
    identity=json.loads((OUT/'identity.json').read_text())
    for name,value in identity['hashes'].items():check(sha(ROOT/name)==value,'Frozen input/source '+name)
    population=pd.read_csv(ROOT/'data/processed/split_manifest.csv').set_index('SK_ID_CURR')
    all_test=set(population[population.split=='test'].index)
    train_ids=set(population[population.split=='train'].index)
    objectives=pd.read_csv(OUT/'objective/metrics.csv')
    decisions=json.loads((OUT/'objective/frozen_decisions.json').read_text())
    check(len(objectives)==42 and len(decisions)==42,'42 fixed objective policies')
    for d in decisions:
        name=d['base_model'];policy=d['policy'];rule=d['rule'];label=name+' '+policy
        validation=pd.read_csv(ROOT/f'results/predictions/{name}_validation.csv',float_precision='round_trip')
        p=validation.probability.to_numpy(dtype=np.float32);y=validation.TARGET.to_numpy();g=validation.thin.to_numpy()
        a=p<=np.asarray(rule['thresholds'])[g]
        if policy!='accuracy' and rule['feasible']:
            check(a.sum()>=CONFIG['minimum_validation_approvals'],label+' minimum approvals')
            check(y[a].mean()<=CONFIG['risk_budget']+1e-12,label+' validation risk budget')
            if 'tolerance' in rule:
                check(recompute(y,g,p,a)['abs_frr_gap']<=rule['tolerance']+1e-12,label+' validation fairness constraint')
        for key,value in recompute(y,g,p,a).items():
            expected=d['validation_metrics'][key]
            check(np.isclose(value,np.nan if expected is None else expected,atol=1e-9,equal_nan=True),label+' validation '+key)
        test=pd.read_csv(OUT/f'objective/{name}_test.csv',float_precision='round_trip')
        p=test.probability.to_numpy(dtype=np.float32);y=test.TARGET.to_numpy();g=test.thin.to_numpy()
        a=p<=np.asarray(rule['thresholds'])[g]
        check(set(test.SK_ID_CURR)==all_test,label+' correct test population')
        check(np.array_equal(a,test[policy].to_numpy(dtype=bool)),label+' frozen thresholds applied')
        row=objectives[(objectives.base_model==name)&(objectives.policy==policy)].iloc[0]
        check(all(np.isclose(row[k],v,atol=1e-9,equal_nan=True) for k,v in recompute(y,g,p,a).items()),label+' independently recomputed test metrics')
    for name in CONFIG['objective']['base_models']:
        check(objectives[objectives.base_model==name].auc.nunique()==1,name+' AUC invariant to decision postprocessing')
    reference=json.loads((OUT/'feedback/frozen_decisions.json').read_text())
    scores=pd.read_csv(OUT/'feedback/test_metrics.csv');events=pd.read_csv(OUT/'feedback/stream_metrics.csv')
    check(len(reference)==120 and len(scores)==120,'All 120 seed/branch/round states retained')
    check(len(events)==100,'All 100 seed/branch/arrival events retained')
    testpop=pd.read_csv(OUT/'feedback/test_population.csv')
    check(set(testpop.SK_ID_CURR)==all_test,'Feedback test population matches original holdout')
    y=testpop.TARGET.to_numpy();g=testpop.thin.to_numpy()
    for seed in CONFIG['feedback']['seeds']:
        arrivals=pd.read_csv(OUT/f'feedback/stream_{seed}.csv').set_index('SK_ID_CURR')
        check(arrivals.index.is_unique and set(arrivals.index)==train_ids,f'{seed} disjoint exhaustive initial/arrival batches')
        initial=set(arrivals[(arrivals.arrival_round==0)&(arrivals.initial_observed==1)].index)
        counts=[]
        for branch in CONFIG['feedback']['branches']:
            expected_weights={int(i):1. for i in initial}
            for r in range(6):
                name=f's{seed}_initial' if r==0 or branch=='static' else f's{seed}_{branch}_r{r}'
                d=json.loads((MOD/f'{name}_decision.json').read_text())
                if r:
                    ledger=pd.read_csv(OUT/f'feedback/ledgers/s{seed}_{branch}_r{r}.csv.gz',float_precision='round_trip')
                    check(set(ledger.SK_ID_CURR)==set(arrivals[arrivals.arrival_round==r].index),f'{seed} {branch} {r} exact current batch')
                    visible=ledger.visible.to_numpy(dtype=bool);proposed=ledger.proposed.to_numpy(dtype=bool);explored=ledger.explored.to_numpy(dtype=bool)
                    check(np.array_equal(visible,proposed|explored) and not np.any(proposed&explored),f'{seed} {branch} {r} visibility mechanism')
                    check(ledger.loc[~visible,'outcome'].isna().all(),f'{seed} {branch} {r} denied labels hidden')
                    check(np.array_equal(ledger.loc[visible,'outcome'],population.loc[ledger.loc[visible,'SK_ID_CURR'],'TARGET']),f'{seed} {branch} {r} observed outcomes authentic')
                    epsilon=.1 if branch.startswith('explore') else 0.
                    check(np.allclose(ledger.inclusion_probability,np.where(proposed,1.,epsilon)),f'{seed} {branch} {r} known propensity')
                    previous=f's{seed}_initial' if r==1 or branch=='static' else f's{seed}_{branch}_r{r-1}'
                    event=events[(events.seed==seed)&(events.branch==branch)&(events['round']==r)].iloc[0]
                    check(event.source_model==previous,f'{seed} {branch} {r} decisions precede current outcomes')
                    prev_d=json.loads((MOD/f'{previous}_decision.json').read_text())
                    check(np.array_equal(proposed,ledger.probability.to_numpy()<=prev_d['rule']['threshold']),f'{seed} {branch} {r} previous frozen threshold')
                    check(int(event.actual_approved)==int(visible.sum()) and int(event.exploration_approved)==int(explored.sum()),f'{seed} {branch} {r} exploration costs counted')
                    check(np.isclose(event.actual_approved_risk,ledger.loc[visible,'outcome'].mean(),equal_nan=True),f'{seed} {branch} {r} actual risk recomputed')
                    if branch!='static':
                        check(set(ledger.loc[visible,'SK_ID_CURR']).isdisjoint(expected_weights),f'{seed} {branch} {r} new observations are disjoint')
                        for row in ledger.loc[visible].itertuples():
                            expected_weights[int(row.SK_ID_CURR)]=1/row.inclusion_probability if branch=='explore_ipw' else 1.
                with np.load(OUT/f'feedback/training/{name}.npz') as trained:
                    training_ids=trained['ids'];weights=trained['weight']
                check(set(training_ids)==set(expected_weights),f'{seed} {branch} {r} only initial plus revealed training IDs')
                check(len(training_ids)==len(np.unique(training_ids)),f'{seed} {branch} {r} no duplicated training customer')
                check(np.allclose(weights,[expected_weights[int(i)] for i in training_ids]),f'{seed} {branch} {r} weights from recorded observation probabilities')
                check(np.all(arrivals.loc[training_ids,'arrival_round']<=r),f'{seed} {branch} {r} no future arrival trained')
                check(np.isclose(d['effective_sample_size'],weights.sum()**2/(weights**2).sum()),f'{seed} {branch} {r} effective sample size')
                with np.load(OUT/f'feedback/predictions/{name}.npz') as f:p=f['probability'];a=f['approved']
                check(np.array_equal(a,p<=d['rule']['threshold']),f'{seed} {branch} {r} test policy recomputed')
                row=scores[(scores.seed==seed)&(scores.branch==branch)&(scores['round']==r)].iloc[0]
                check(all(np.isclose(row[k],v,atol=1e-9,equal_nan=True) for k,v in recompute(y,g,p,a).items()),f'{seed} {branch} {r} test metrics recomputed')
                check((OUT/f'feedback/predictions/{name}.npz').stat().st_mtime>=(OUT/'feedback/frozen_decisions.json').stat().st_mtime,f'{seed} {branch} {r} test evaluated after all trajectories frozen')
                if r==0:counts.append(row.thin_frr)
            if branch=='static':check(scores[(scores.seed==seed)&(scores.branch==branch)].thin_frr.nunique()==1,f'{seed} static model does not change')
        check(np.ptp(counts)==0,f'{seed} all branches start with identical predictions')
    diagnostic_scores=pd.read_csv(OUT/'feedback/diagnostic_metrics.csv')
    check(len(diagnostic_scores)==30,'All 30 diagnostic reference states retained')
    diagnostic_identity=json.loads((OUT/'feedback/diagnostic_identity.json').read_text())
    for name,value in diagnostic_identity['hashes'].items():check(sha(ROOT/name)==value,'Diagnostic frozen input/source '+name)
    for seed in CONFIG['feedback']['seeds']:
        arrivals=pd.read_csv(OUT/f'feedback/stream_{seed}.csv')
        with np.load(OUT/f'feedback/training/s{seed}_initial.npz') as f:expected=list(f['ids'])
        for r in range(1,6):
            n=int(events[(events.seed==seed)&(events.branch=='selected_only')&(events['round']==r)].actual_approved.iloc[0])
            batch=arrivals.loc[arrivals.arrival_round==r,'SK_ID_CURR'].to_numpy()
            expected.extend(np.random.default_rng(seed*1000+r+70000).choice(batch,n,replace=False))
            name=f's{seed}_random_reference_r{r}'
            with np.load(OUT/f'feedback/training/{name}.npz') as f:
                trained=f['ids'];weights=f['weight']
            with np.load(OUT/f'feedback/training/s{seed}_selected_only_r{r}.npz') as f:selected_count=len(f['ids'])
            check(np.array_equal(trained,expected),f'{name} exact label-independent random draw')
            check(len(trained)==selected_count,f'{name} matches selected-only training size')
            check(len(np.unique(trained))==len(trained) and set(trained).issubset(train_ids),f'{name} unique training-only membership')
            check(np.all(weights==1),f'{name} equal weights')
            with np.load(OUT/f'feedback/predictions/{name}.npz') as f:p=f['probability'];a=f['approved']
            d=json.loads((MOD/f'{name}_decision.json').read_text())
            check(np.array_equal(a,p<=d['rule']['threshold']),f'{name} frozen test decisions')
            row=diagnostic_scores[(diagnostic_scores.seed==seed)&(diagnostic_scores['round']==r)].iloc[0]
            check(all(np.isclose(row[k],v,atol=1e-9,equal_nan=True) for k,v in recompute(y,g,p,a).items()),f'{name} independent metrics')
            check((OUT/f'feedback/predictions/{name}.npz').stat().st_mtime>=(OUT/'feedback/diagnostic_frozen.json').stat().st_mtime,f'{name} tested after diagnostic freeze')
    check(len(pd.read_csv(OUT/'paired_intervals.csv'))==60,'All primary and diagnostic paired interval comparisons')
    for name in ['06_objectives','07_objective_tradeoff','08_feedback_trajectory','09_feedback_representation','10_exploration_cost','11_feedback_effects']:
        for suffix in ['png','pdf']:check((ROOT/f'figures/{name}.{suffix}').stat().st_size>1000,name+' '+suffix+' exists')
    safe_dump(OUT/'verification.json',{'passed':True,'check_count':len(checks),'checks':checks})
    print(f'{len(checks)} independent structural, label-visibility, and metric checks passed.')


if __name__=='__main__':main()
