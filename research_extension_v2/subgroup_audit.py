import numpy as np
from research_extension_v2.common import metrics


def fit_income_boundaries(training_income, quantiles):
    x = np.asarray(training_income, dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Missing/nonfinite training income requires explicit protocol')
    cuts = np.quantile(x, quantiles)
    if cuts[0] >= cuts[1]: raise ValueError('Degenerate income quantiles')
    return cuts


def income_groups(income, cuts):
    x = np.asarray(income, dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Missing/nonfinite audit income')
    return np.searchsorted(cuts, x, side='left')


def wilson(k, n):
    if n == 0: return (None, None)
    z = 1.959963984540054; p = k / n; den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return float(center-half), float(center+half)


def audit(y, g, p, approvals, baseline, income, cuts, minimum_n, minimum_good):
    y, g, baseline = np.asarray(y), np.asarray(g), np.asarray(baseline)
    bins = income_groups(income, cuts); rows = []
    for group, name in [(0, 'comparison'), (1, 'thin')]:
        for band, label in enumerate(['low', 'middle', 'high']):
            mask = (g == group) & (bins == band); good = mask & (y == 0)
            n, n0 = int(mask.sum()), int(good.sum())
            for policy, a in approvals.items():
                a = np.asarray(a, dtype=bool); rejected = int((~a[good]).sum()); approved = mask & a
                lo, hi = wilson(rejected, n0)
                delta = ((~a[good]).astype(float) - (~baseline[good]).astype(float))
                rows.append(dict(policy=policy, group=name, income=label, n=n, normal_count=n0,
                    rejected_good=rejected, frr=rejected/n0 if n0 else None, frr_lo=lo, frr_hi=hi,
                    frr_change=float(delta.mean()) if n0 else None,
                    approval_rate=float(a[mask].mean()) if n else None, approved_count=int(approved.sum()),
                    approved_risk=float(y[approved].mean()) if approved.any() else None,
                    status='PASS' if n>=minimum_n and n0>=minimum_good else 'INSUFFICIENT_SUPPORT'))
    return rows
