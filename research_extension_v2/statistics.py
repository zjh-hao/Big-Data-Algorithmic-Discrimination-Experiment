"""Conditional paired bootstrap via exact empirical joint-cell compression."""
import numpy as np
from scipy.stats import rankdata


def interval(v):
    v = np.asarray(v); v = v[np.isfinite(v)]
    return tuple(float(z) for z in np.quantile(v, [.025, .975])) if len(v) else (None, None)


def joint_bootstrap(columns, repetitions, seed):
    cells, counts = np.unique(np.column_stack(columns), axis=0, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(int(counts.sum()), counts/counts.sum(), size=repetitions)
    return cells, draws


def ratio(n, d):
    return np.divide(n, d, out=np.full(np.broadcast_shapes(np.shape(n), np.shape(d)), np.nan), where=d!=0)


def policy_intervals(y, g, bins, approvals, repetitions, seed):
    keys = list(approvals)
    cells, draws = joint_bootstrap([y, g, bins] + [approvals[k] for k in keys], repetitions, seed)
    labels, groups, income = cells[:, 0], cells[:, 1], cells[:, 2]
    values = {}; rows = []
    for j, key in enumerate(keys):
        a = cells[:, 3+j]
        v = dict(approval_rate=ratio(draws@a, draws.sum(axis=1)),
                 approved_risk=ratio(draws@(a*labels), draws@a))
        for group, name in [(0, 'thick'), (1, 'thin')]:
            good = (groups==group)&(labels==0)
            v[name+'_frr'] = ratio(draws@(good*(1-a)), draws@good)
            for band in [0, 1, 2]:
                m = good&(income==band)
                v[f'subgroup_{group}_{band}'] = ratio(draws@(m*(1-a)), draws@m)
        v['abs_frr_gap'] = abs(v['thin_frr']-v['thick_frr']); values[key] = v
    for key, v in values.items():
        for metric, draws_v in v.items():
            lo, hi = interval(draws_v)
            dlo, dhi = interval(draws_v-values['business'][metric])
            rows.append(dict(policy=key, metric=metric, lo=lo, hi=hi,
                             difference=float(np.nanmean(draws_v-values['business'][metric])),
                             difference_lo=dlo, difference_hi=dhi,
                             method='paired empirical joint-cell bootstrap', repetitions=repetitions))
    return rows


def auc_interval(y, p):
    """DeLong influence variance for a single fixed scoring model, ties midrank."""
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y==1], p[y==0]; m, n = len(pos), len(neg)
    rank = rankdata(np.r_[pos, neg], method='average')
    v10 = (rank[:m]-rankdata(pos))/n
    v01 = 1-(rank[m:]-rankdata(neg))/m
    auc = float(v10.mean()); se = np.sqrt(np.var(v10, ddof=1)/m+np.var(v01, ddof=1)/n)
    return dict(auc=auc, lo=max(0., auc-1.959963984540054*se), hi=min(1., auc+1.959963984540054*se), method='DeLong conditional asymptotic')


def replay_intervals(ledger, oracle_y, repetitions, seed):
    # Oracle enters only this final offline error evaluator, never estimate().
    cells, draws = joint_bootstrap([ledger.thin, oracle_y, ledger.visible,
        ledger.proposed_approval, ledger.inclusion_probability], repetitions, seed)
    g, y, visible, proposed, pi = cells.T
    w = np.divide(1., pi, out=np.zeros(len(pi)), where=pi>0)
    out = []
    for group in [0, 1]:
        good = (g==group)&(y==0); obs = good*(visible==1); reject = 1-proposed
        naive = ratio(draws@(obs*reject), draws@obs)
        oracle = ratio(draws@(good*reject), draws@good)
        ipw = ratio(draws@(obs*w*reject), draws@(obs*w))
        r = dict(group=group)
        for name, v in [('naive', naive), ('oracle', oracle), ('ipw', ipw)]:
            if name=='ipw' and (pi[g==group]==0).any(): v = np.full(repetitions, np.nan)
            r[name+'_lo'], r[name+'_hi'] = interval(v)
            r[name+'_error_lo'], r[name+'_error_hi'] = interval(v-oracle)
        out.append(r)
    # Business risk and exploration risk use only visible (mature) labels.
    for name, mask in [('regular', proposed==1), ('exploration', (visible==1)&(proposed==0)), ('actual', visible==1)]:
        v = ratio(draws@(mask*y), draws@mask)
        lo, hi = interval(v)
        for r in out: r[name+'_risk_lo'], r[name+'_risk_hi'] = lo, hi
    return out
