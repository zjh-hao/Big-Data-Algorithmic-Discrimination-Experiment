"""Simultaneous DKW audit of threshold mixtures and baseline-relative harm.

This is an audit, not another optimization constraint.  Under the assumptions
in ``ASSUMPTIONS``, one CDF event covers every threshold and every convex
mixture, including knots/weights selected on this validation sample.  There is
therefore no union penalty for the number of policies.  There *is* a union
penalty for the declared fixed scorer family and the eight normal-label cells.

For cell j, DKW gives sup_t |F_j(t)-Fhat_j(t)| <= r_j.  For weights w and
baseline v aligned on a common union of knots,
    |(FRR_j(w)-FRR_j(v)) - (FRRhat_j(w)-FRRhat_j(v))|
        <= r_j * ||w-v||_1.
This follows by writing the paired difference as sum_k (v_k-w_k) F_j(t_k).
It is exactly zero for identical mixtures.  Absolute FRR uses radius r_j;
macro parity uses the sum of the two macro radii.  Bounds are clipped only to
their known ranges, never to a requested fairness/protection target.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pandas as pd


SOURCE = "https://doi.org/10.1214/aop/1176990746"
ASSUMPTIONS = (
    "Applicants are iid from the target population; conditional on a positive "
    "normal-label cell count, scores in that cell are iid from its conditional CDF. "
    "Every scorer and probability calibrator, and the group definitions, are fixed "
    "independently of this audit sample. The declared finite scorer family contains "
    "every scorer compared. Threshold knots and mixture weights may be selected "
    "using this validation sample because the CDF event is uniform over all thresholds."
)
LIMITATIONS = (
    "Assumptions are required, not established by this function. This is conditional "
    "normal-label population inference under iid sampling, not a new blind test, "
    "a guarantee under distribution shift, or a bound for every realized randomized "
    "loan book. TARGET is the historical repayment-difficulty label. Income is used "
    "only to audit cells; applying a policy uses score and thin group only."
)
CELL_COUNT = 8
_TOL = 1e-8


def _mixture(policy: dict, group: int) -> tuple[np.ndarray, np.ndarray]:
    if not policy.get("feasible", False) or not isinstance(policy.get("mixture"), dict):
        raise ValueError("A feasible frozen threshold mixture is required")
    entry = policy["mixture"].get(str(group), policy["mixture"].get(group))
    if entry is None:
        raise ValueError("Both thin groups must have frozen mixtures")
    thresholds = np.asarray(entry["thresholds"], dtype=float)
    weights = np.asarray(entry["weights"], dtype=float)
    if (thresholds.ndim != 1 or thresholds.size == 0 or
            weights.shape != thresholds.shape or not np.isfinite(thresholds).all() or
            not np.isfinite(weights).all() or np.any(weights < 0) or
            abs(float(weights.sum()) - 1.0) > _TOL):
        raise ValueError("Invalid convex threshold mixture")
    grid, inverse = np.unique(thresholds, return_inverse=True)
    merged = np.zeros(len(grid), dtype=float)
    np.add.at(merged, inverse, weights)
    # Remove only floating-point sum error, not a substantive policy defect.
    merged /= merged.sum()
    return grid, merged


def aligned_weights(policy: dict, baseline: dict, group: int):
    """Preserve both policies exactly while aligning their full knot dictionaries."""
    t, w = _mixture(policy, group)
    t0, v = _mixture(baseline, group)
    grid = np.union1d(t, t0)
    ww, vv = np.zeros(len(grid)), np.zeros(len(grid))
    ww[np.searchsorted(grid, t)] = w
    vv[np.searchsorted(grid, t0)] = v
    return grid, ww, vv


def policy_approval_probabilities(frame: pd.DataFrame, policy: dict) -> np.ndarray:
    """Apply a frozen mixture using only probability and thin, never outcomes."""
    if not {"probability", "thin"}.issubset(frame.columns):
        raise ValueError("Applying a policy requires probability and thin only")
    scores = frame["probability"].to_numpy(dtype=float)
    groups = frame["thin"].to_numpy()
    if (not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)) or
            not np.isin(groups, [0, 1]).all()):
        raise ValueError("Scores must be probabilities and thin must be 0/1")
    result = np.zeros(len(frame), dtype=float)
    for group in (0, 1):
        thresholds, weights = _mixture(policy, group)
        mask = groups == group
        tail = np.r_[np.cumsum(weights[::-1])[::-1], 0.0]
        result[mask] = tail[np.searchsorted(thresholds, scores[mask], side="left")]
    return np.clip(result, 0.0, 1.0)


def _number_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def certificate_rows(validation: pd.DataFrame, frozen: dict, alpha: float = .05,
                     number_of_scorers: int = 9) -> list[dict]:
    """Return eight cell rows and one macro-parity row per feasible policy.

    ``frozen['policies']`` follows protected_policy.py.  A policy's explicit
    ``baseline_policy_id`` is honored; otherwise the unique unmodified LP
    profit baseline is used.  Infeasible policies have no decisions and are
    skipped. Missing/ambiguous baselines and malformed weights raise errors.

    The caller must pass validation predictions only.  If a split column is
    supplied it is checked, but absence of that column cannot prove provenance.
    M=number_of_scorers is the full declared scorer family, not the number of
    rows or the number of policies in this one call.  With J=8*M, each event is
    allocated alpha/J and r=sqrt(log(2*J/alpha)/(2*n_normal)).
    """
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if (isinstance(number_of_scorers, bool) or not isinstance(number_of_scorers, Integral)
            or number_of_scorers < 1):
        raise ValueError("number_of_scorers must be a positive integer")
    required = {"probability", "thin", "income_band", "TARGET"}
    if not required.issubset(validation.columns):
        raise ValueError("Validation needs score, thin, income_band, and TARGET")
    if "split" in validation and not validation["split"].eq("validation").all():
        raise ValueError("This certificate API accepts validation rows only")
    y = validation["TARGET"].to_numpy()
    g = validation["thin"].to_numpy()
    b = validation["income_band"].to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.isin(b, [0, 1, 2]).all():
        raise ValueError("TARGET must be 0/1 and income_band must be training-defined 0/1/2")
    policies = frozen.get("policies")
    if not isinstance(policies, list):
        raise ValueError("frozen must contain a policies list")
    by_id = {p["policy_id"]: p for p in policies}
    if len(by_id) != len(policies):
        raise ValueError("Frozen policy identifiers must be unique")
    # A run can also contain a *robust* LP profit comparator marked as a
    # baseline. Its explicit parent identifier must not replace the primary.
    defaults = [p for p in policies if p.get("policy_id") == "lp_profit_baseline"]
    if not defaults:
        defaults = [p for p in policies if p.get("is_lp_profit_baseline") and
                    not p.get("baseline_policy_id")]
    if len(defaults) != 1:
        raise ValueError("Exactly one LP profit baseline must be declared")
    default_baseline = defaults[0]
    if not default_baseline.get("feasible"):
        raise ValueError("The LP profit baseline must be feasible")
    family_size = CELL_COUNT * int(number_of_scorers)
    log_term = math.log(2.0 * family_size / float(alpha))
    cells = [(f"macro_{h}", "macro", h, None) for h in (0, 1)]
    cells += [(f"subgroup_{h}_{band}", "subgroup", h, band)
              for h in (0, 1) for band in (0, 1, 2)]
    rows = []
    for policy in policies:
        if not policy.get("feasible"):
            continue
        base_id = policy.get("baseline_policy_id") or default_baseline["policy_id"]
        if base_id not in by_id:
            raise ValueError(f"Missing baseline policy: {base_id}")
        baseline = by_id[base_id]
        q = policy_approval_probabilities(validation, policy)
        q0 = policy_approval_probabilities(validation, baseline)
        alignment = {h: aligned_weights(policy, baseline, h) for h in (0, 1)}
        eps = _number_or_none(policy.get("epsilon"))
        delta = _number_or_none(policy.get("effective_delta", policy.get("delta")))
        shared = {
            "policy_id": policy["policy_id"], "baseline_policy_id": base_id,
            "audit_split": "validation", "alpha": float(alpha),
            "joint_confidence_level": 1.0 - float(alpha),
            "number_of_scorers": int(number_of_scorers),
            "normal_cells_per_scorer": CELL_COUNT, "simultaneous_cdf_events": family_size,
            "per_cdf_failure_probability": float(alpha) / family_size,
            "statistical_statement": "simultaneous_conditional_DKW_under_stated_assumptions",
            "decision_semantics": "stochastic_expectation",
            "assumptions": ASSUMPTIONS, "limitations": LIMITATIONS, "source_url": SOURCE,
            "certificate_is_optimization_constraint": False,
        }
        macro = {}
        for cell, kind, group, band in cells:
            mask = (y == 0) & (g == group)
            if band is not None:
                mask &= b == band
            n = int(mask.sum())
            grid, w, v = alignment[group]
            l1 = float(np.abs(w - v).sum())
            row = dict(shared, row_type="normal_cell", cell=cell, group_kind=kind,
                       thin=group, income_band=band, normal_records=n,
                       common_grid_size=len(grid), l1_weight_distance=l1,
                       effective_protection_target=delta)
            if not n:
                row.update(status="EMPTY_NORMAL_STRATUM", dkw_radius=None,
                           empirical_approval=None, empirical_frr=None, baseline_frr=None,
                           empirical_harm=None, frr_lower=None, frr_upper=None,
                           paired_halfwidth=None, harm_lower=None, harm_upper=None,
                           protection_certified=None)
            else:
                r = math.sqrt(log_term / (2.0 * n))
                approval = float(q[mask].mean())
                frr = 1.0 - approval
                frr0 = 1.0 - float(q0[mask].mean())
                # The paired average also eliminates cancellation of two large FRRs.
                harm = float(np.mean(q0[mask] - q[mask]))
                margin = r * l1
                lo, hi = max(-1.0, harm - margin), min(1.0, harm + margin)
                row.update(status="COMPUTED_ASSUMPTIONS_REQUIRED", dkw_radius=r,
                           empirical_approval=approval, empirical_frr=frr,
                           baseline_frr=frr0, empirical_harm=harm,
                           frr_lower=max(0.0, frr-r), frr_upper=min(1.0, frr+r),
                           baseline_frr_lower=max(0.0, frr0-r),
                           baseline_frr_upper=min(1.0, frr0+r),
                           paired_halfwidth=margin, harm_lower=lo, harm_upper=hi,
                           separate_absolute_halfwidth=2.0*r,
                           protection_certified=(hi <= delta) if delta is not None else None)
            rows.append(row)
            if kind == "macro":
                macro[group] = row
        parity = dict(shared, row_type="macro_parity", cell="macro_gap",
                      group_kind="macro_pair", thin=None, income_band=None,
                      normal_records=sum(macro[h]["normal_records"] for h in (0, 1)),
                      fairness_target=eps)
        if any(macro[h]["normal_records"] == 0 for h in (0, 1)):
            parity.update(status="EMPTY_NORMAL_STRATUM", empirical_signed_gap=None,
                          empirical_abs_gap=None, gap_halfwidth=None, abs_gap_lower=None,
                          abs_gap_upper=None, parity_certified=None)
        else:
            gap = macro[1]["empirical_frr"] - macro[0]["empirical_frr"]
            margin = macro[0]["dkw_radius"] + macro[1]["dkw_radius"]
            lo, hi = max(-1.0, gap-margin), min(1.0, gap+margin)
            abslo = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
            abshi = max(abs(lo), abs(hi))
            parity.update(status="COMPUTED_ASSUMPTIONS_REQUIRED", empirical_signed_gap=gap,
                          empirical_abs_gap=abs(gap), gap_halfwidth=margin,
                          signed_gap_lower=lo, signed_gap_upper=hi,
                          abs_gap_lower=abslo, abs_gap_upper=abshi,
                          parity_certified=(abshi <= eps) if eps is not None else None)
        rows.append(parity)
    return rows
