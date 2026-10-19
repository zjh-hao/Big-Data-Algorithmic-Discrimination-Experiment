"""Estimator accepts an observable ledger only. Oracle is a separate evaluator."""
import numpy as np
import pandas as pd


def validate_ledger(ledger):
    needed = ['SK_ID_CURR', 'thin', 'proposed_approval', 'actual_approval', 'visible', 'inclusion_probability', 'outcome']
    if set(needed) - set(ledger): raise ValueError('Missing ledger fields')
    if not ledger.SK_ID_CURR.is_unique: raise ValueError('Duplicate record ID')
    for c in ['thin', 'proposed_approval', 'actual_approval', 'visible']:
        if not ledger[c].isin([0, 1]).all(): raise ValueError('Invalid boolean/group')
    pi = ledger.inclusion_probability
    if pi.isna().any() or not pi.between(0, 1).all(): raise ValueError('Invalid observation probability')
    visible = ledger.visible.astype(bool)
    if ledger.loc[~visible, 'outcome'].notna().any(): raise ValueError('Hidden labels must be absent')
    if not ledger.loc[visible, 'outcome'].isin([0, 1]).all(): raise ValueError('Invalid observed label')
    if (pi[visible] <= 0).any(): raise ValueError('Observed record with zero propensity')
    if not (ledger.visible == ledger.actual_approval).all(): raise ValueError('Replay assumes all actual approvals mature')
    if (ledger.proposed_approval > ledger.actual_approval).any(): raise ValueError('Proposed approval omitted')


def estimate(ledger, minimum_ess):
    validate_ledger(ledger); rows = []
    for group in [0, 1]:
        pop = ledger[ledger.thin == group]; v = pop[pop.visible == 1]
        good = v[v.outcome == 0]; w = 1 / good.inclusion_probability.to_numpy()
        reject = 1 - good.proposed_approval.to_numpy()
        identifiable = len(pop)>0 and (pop.inclusion_probability>0).all()
        ess = float(w.sum()**2 / (w@w)) if len(w) else 0.
        naive = float(reject.mean()) if len(good) else None
        value = float(w@reject / w.sum()) if identifiable and len(w) else None
        rows.append(dict(group=group, n=len(pop), visible_n=len(v), observed_good_n=len(good),
            naive_frr=naive, ipw_frr=value, ess_good=ess,
            ess_visible=float((1/v.inclusion_probability).sum()**2 / ((1/v.inclusion_probability)**2).sum()) if len(v) else 0.,
            max_weight=float((1/v.inclusion_probability).max()) if len(v) else None,
            coverage=len(v)/len(pop) if len(pop) else None,
            status=('NOT_IDENTIFIABLE' if not identifiable else 'INSUFFICIENT_SUPPORT' if value is None or ess<minimum_ess else 'PASS')))
    return rows


def exploration_plan(ids, g, proposed, rate, uniform):
    """Decisions and propensities have no label argument."""
    if not 0<=rate<=1: raise ValueError('Invalid exploration rate')
    proposed = np.asarray(proposed, dtype=bool); u = np.asarray(uniform)
    if not np.isfinite(u).all() or not ((u>=0)&(u<1)).all(): raise ValueError('Invalid random draws')
    explore = ~proposed & (u < rate); actual = proposed | explore
    return pd.DataFrame(dict(SK_ID_CURR=ids, thin=g, proposed_approval=proposed.astype(int),
        explored=explore.astype(int), actual_approval=actual.astype(int), visible=actual.astype(int),
        inclusion_probability=np.where(proposed, 1., rate)))


def reveal(plan, visible_ids, visible_outcomes):
    """Only the simulator may access full outcomes, then passes this subset."""
    out = plan.copy(); labels = pd.Series(visible_outcomes, index=visible_ids)
    if not labels.index.is_unique or set(labels.index)!=set(out.loc[out.visible==1, 'SK_ID_CURR']):
        raise ValueError('Visible label IDs do not align')
    out['outcome'] = out.SK_ID_CURR.map(labels)
    validate_ledger(out)
    return out


def oracle_metrics(oracle, plan):
    """Final offline benchmark, keyed alignment, no path back to estimator."""
    if not oracle.SK_ID_CURR.is_unique or set(oracle.SK_ID_CURR)!=set(plan.SK_ID_CURR): raise ValueError('Oracle ID mismatch')
    y = oracle.set_index('SK_ID_CURR').loc[plan.SK_ID_CURR, 'TARGET'].to_numpy()
    return {g: float((1-plan.proposed_approval.to_numpy())[(plan.thin.to_numpy()==g)&(y==0)].mean())
            for g in [0, 1]}
