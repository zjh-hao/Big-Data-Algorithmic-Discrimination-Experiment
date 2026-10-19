"""One finite decision space, deterministic optimization, validation inputs only."""
import hashlib
import numpy as np
from research_extension_v2.common import metrics
from extensions import threshold_points, apply_rule
from experiment import choose_threshold


def candidate_space(p, y, g, budget, minimum, grid):
    p, y, g = np.asarray(p, dtype=float), np.asarray(y), np.asarray(g)
    if len(p) == 0 or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all() or not np.isin(g, [0, 1]).all():
        raise ValueError('Invalid validation inputs')
    common = choose_threshold(p, y, budget, minimum)['threshold']
    data = []
    for group in [0, 1]:
        m = g == group
        if not (y[m] == 0).any(): raise ValueError('No normal-label support in group')
        t, n, b, good = threshold_points(p[m], y[m], grid, common)
        data.append((t, n, b, 1 - good / (y[m] == 0).sum()))
    t0, n0, b0, f0 = data[0]; t1, n1, b1, f1 = data[1]
    n, bad = n0[:, None] + n1[None, :], b0[:, None] + b1[None, :]
    gap = abs(f0[:, None] - f1[None, :])
    valid = (n >= max(1, minimum)) & (bad <= budget * n + 1e-12)
    digest = hashlib.sha256(t0.astype('<f8').tobytes() + t1.astype('<f8').tobytes()).hexdigest()
    return dict(t0=t0, t1=t1, f0=f0, f1=f1, n=n, bad=bad, gap=gap, valid=valid, fingerprint=digest)


def solve(space, epsilon=None, baseline=None, delta=None):
    s = space; ok = s['valid'].copy()
    risk_candidates = int(ok.sum())
    if epsilon is not None: ok &= s['gap'] <= epsilon + 1e-12
    gap_candidates = int(ok.sum())
    if baseline is not None:
        if baseline['status'] != 'PASS':
            return dict(status='INFEASIBLE', feasible=False, thresholds=None, reason='Business baseline infeasible')
        ok &= (s['f0'][:, None] <= baseline['validation_thick_frr'] + delta + 1e-12)
        ok &= (s['f1'][None, :] <= baseline['validation_thin_frr'] + delta + 1e-12)
    result = dict(status='PASS' if ok.any() else 'INFEASIBLE', feasible=bool(ok.any()),
                  epsilon=epsilon, delta=delta, candidate_fingerprint=s['fingerprint'],
                  grid_counts=[len(s['t0']), len(s['t1'])], risk_feasible_candidates=risk_candidates,
                  gap_feasible_candidates=gap_candidates, final_feasible_candidates=int(ok.sum()),
                  noharm_removed_candidates=gap_candidates-int(ok.sum()) if baseline is not None else 0)
    if not ok.any():
        return dict(result, thresholds=None, reason='No candidate meets every constraint and minimum approvals')
    i, j = np.where(ok)
    # Shared with legacy fair_rule: approvals, risk, gap, then threshold indices.
    order = np.lexsort((j, i, s['gap'][i, j], s['bad'][i, j] / s['n'][i, j], -s['n'][i, j]))
    a, b = i[order[0]], j[order[0]]
    result.update(thresholds=[float(s['t0'][a]), float(s['t1'][b])], indices=[int(a), int(b)],
                  validation_approvals=int(s['n'][a, b]), validation_risk=float(s['bad'][a, b] / s['n'][a, b]),
                  validation_thick_frr=float(s['f0'][a]), validation_thin_frr=float(s['f1'][b]),
                  validation_abs_frr_gap=float(s['gap'][a, b]))
    if epsilon is not None: result['gap_slack'] = epsilon - result['validation_abs_frr_gap']
    if baseline is not None:
        for group in ['thin', 'thick']:
            result[group + '_noharm_slack'] = baseline['validation_' + group + '_frr'] + delta - result['validation_' + group + '_frr']
    return result


def approve(p, g, rule):
    if not rule.get('feasible', False): raise ValueError('INFEASIBLE policy has no approval decisions')
    return apply_rule(np.asarray(p), np.asarray(g, dtype=int), rule)


def fit_policies(validation, config):
    s = candidate_space(validation.probability, validation.TARGET, validation.thin,
                        config['risk_budget'], config['minimum_approvals'], config['grid'])
    business = solve(s)
    out = {'business': business}
    for e in config['epsilons']:
        gap = solve(s, epsilon=e)
        gap['gap_changes_business'] = gap.get('thresholds') != business.get('thresholds')
        out[f'gap_e{e:.2f}'] = gap
        for d in config['deltas']:
            r = solve(s, epsilon=e, baseline=business, delta=d)
            r['noharm_changes_gap_only'] = r.get('thresholds') != gap.get('thresholds')
            out[f'noharm_e{e:.2f}_d{d:.2f}'] = r
    return out
