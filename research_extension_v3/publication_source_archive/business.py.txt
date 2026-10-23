"""Validation-frozen approval policies and transparent credit business scenarios.

TARGET=1 means repayment difficulty, so both economic payoffs below are scenarios,
not observed bank profit.  Fitting uses validation labels only; evaluation never
changes thresholds.  One shared finite grid makes all policy comparisons nested.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("SK_ID_CURR", "TARGET", "thin", "probability", "AMT_CREDIT")
TOLERANCE = 1e-10


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) == 0:
        raise ValueError("Approval evaluation needs at least one record")
    if frame.SK_ID_CURR.isna().any() or frame.SK_ID_CURR.duplicated().any():
        raise ValueError("SK_ID_CURR must be nonmissing and unique")
    out = {"y": frame.TARGET.to_numpy(dtype=float),
           "g": frame.thin.to_numpy(dtype=float),
           "p": frame.probability.to_numpy(dtype=float),
           "amount": frame.AMT_CREDIT.to_numpy(dtype=float)}
    if any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError("Labels, groups, probabilities, and amounts must be finite")
    if not np.isin(out["y"], [0, 1]).all() or not np.isin(out["g"], [0, 1]).all():
        raise ValueError("TARGET and thin must be binary")
    if ((out["p"] < 0) | (out["p"] > 1)).any():
        raise ValueError("probability must lie in [0, 1]")
    if (out["amount"] <= 0).any():
        raise ValueError("AMT_CREDIT must be positive")
    out["g"] = out["g"].astype(int)
    return out


def _scenario(loss_ratio: float, config: dict | None) -> tuple[float, float, float]:
    cfg = config or {}
    lam = float(loss_ratio)
    interest = float(cfg.get("scenario_interest", 0.12))
    cost = float(cfg.get("scenario_cost", 0.03))
    if not np.isfinite([lam, interest, cost]).all() or lam <= 0 or cost < 0 or interest <= cost:
        raise ValueError("Scenario needs positive loss_ratio and interest > cost >= 0")
    loss = lam * (interest - cost) - cost
    if loss < 0:
        raise ValueError("Scenario implies a negative difficulty loss rate")
    return interest, cost, loss


def metrics_for_decisions(frame: pd.DataFrame, approvals, loss_ratio: float = 5,
                          config: dict | None = None) -> dict[str, Any]:
    """Metrics for hard decisions or fractional randomized approval probabilities.

    For fractional decisions every count/payoff is its expectation; ``risk`` is
    the ratio of expected difficulty approvals to expected total approvals.
    It is not a guarantee on any single random realization.
    """
    data = _arrays(frame)
    a = np.asarray(approvals, dtype=float)
    if a.shape != (len(frame),) or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError("Approvals must be a finite vector in [0, 1]")
    y, p, g, amount = (data[k] for k in ("y", "p", "g", "amount"))
    interest, cost, loss = _scenario(loss_ratio, config)
    n = float(a.sum())
    bad = float(np.dot(a, y))
    good = float(np.dot(a, 1 - y))
    approved_amount = float(np.dot(a, amount))
    bad_amount = float(np.dot(a, amount * y))
    expected_bad = float(np.dot(a, p))
    utility = good - float(loss_ratio) * bad
    expected_utility = float(np.dot(a, (1 - p) - float(loss_ratio) * p))
    amount_utility = float(np.dot(a, amount * ((1 - y) * interest - y * loss - cost)))
    expected_amount_utility = float(np.dot(a, amount * ((1 - p) * interest - p * loss - cost)))
    result: dict[str, Any] = {
        "records": int(len(frame)), "approvals": n, "approval_rate": n / len(frame),
        "good_approved": good, "difficulty_count": bad,
        "risk": bad / n if n else None,
        "expected_risk": expected_bad / n if n else None,
        "approved_amount": approved_amount,
        "capital_fraction_used": approved_amount / float(amount.sum()),
        "amount_weighted_risk": bad_amount / approved_amount if approved_amount else None,
        "utility": utility, "expected_utility": expected_utility,
        "utility_per_1000_applicants": utility * 1000 / len(frame),
        "amount_utility": amount_utility, "expected_amount_utility": expected_amount_utility,
        "amount_utility_per_1000_applicants": amount_utility * 1000 / len(frame),
        "net_contribution_per_funded_unit": amount_utility / approved_amount if approved_amount else None,
        "loss_ratio": float(loss_ratio), "scenario_interest": interest,
        "scenario_cost": cost, "scenario_difficulty_loss": loss,
        "decision_type": "deterministic" if np.isin(a, [0, 1]).all() else "stochastic_expectation",
    }
    for group, name in ((0, "thick"), (1, "thin")):
        mask = g == group
        normal_count = int(np.sum(mask & (y == 0)))
        approved_good = float(np.dot(a[mask], 1 - y[mask]))
        group_n = float(a[mask].sum())
        group_bad = float(np.dot(a[mask], y[mask]))
        result.update({f"{name}_records": int(mask.sum()),
                       f"{name}_normal_records": normal_count,
                       f"{name}_approvals": group_n,
                       f"{name}_good_approved": approved_good,
                       f"{name}_difficulty_approved": group_bad,
                       f"{name}_risk": group_bad / group_n if group_n else None,
                       f"{name}_frr": 1 - approved_good / normal_count if normal_count else None})
    f0, f1 = result["thick_frr"], result["thin_frr"]
    result["abs_frr_gap"] = abs(f0 - f1) if f0 is not None and f1 is not None else None
    return result


def apply_policy(frame: pd.DataFrame, policy: dict) -> np.ndarray:
    """Apply a frozen threshold pair ordered [thick=0, thin=1]."""
    if not policy.get("feasible", False) or policy.get("thresholds") is None:
        raise ValueError("INFEASIBLE policy has no approval decisions")
    data = _arrays(frame)
    thresholds = np.asarray(policy["thresholds"], dtype=float)
    if thresholds.shape != (2,) or not np.isfinite(thresholds).all():
        raise ValueError("Frozen thresholds must be a finite pair")
    return data["p"] <= thresholds[data["g"]]


def _candidate_space(frame: pd.DataFrame, config: dict) -> dict:
    data = _arrays(frame)
    grid_size = int(config.get("grid", 256))
    minimum = max(1, int(config.get("minimum_approvals", 500)))
    if grid_size < 2:
        raise ValueError("grid must be at least 2")
    p, y, g, amount = (data[k] for k in ("p", "y", "g", "amount"))
    unique = np.unique(p)
    interior = unique if len(unique) <= grid_size else np.quantile(p, np.linspace(0, 1, grid_size))
    # -1 rejects even a zero score; 1 accepts even a unit score, and both are JSON-safe.
    thresholds = np.unique(np.r_[-1.0, interior, 1.0])
    groups = []
    for group in (0, 1):
        mask = g == group
        normal = int(np.sum(y[mask] == 0))
        if normal == 0:
            raise ValueError("Validation needs normal-label support in each group")
        order = np.argsort(p[mask], kind="stable")
        pg, yg, lg = p[mask][order], y[mask][order], amount[mask][order]
        positions = np.searchsorted(pg, thresholds, side="right")
        def cumulative(values):
            return np.r_[0.0, np.cumsum(values, dtype=float)][positions]
        n = positions.astype(float)
        bad = cumulative(yg)
        groups.append({"n": n, "bad": bad, "good": n - bad,
                       "amount": cumulative(lg), "bad_amount": cumulative(lg * yg),
                       "expected_bad": cumulative(pg),
                       "expected_bad_amount": cumulative(lg * pg),
                       "frr": 1 - (n - bad) / normal})
    first, second = groups
    space = {"thresholds": thresholds, "groups": groups, "minimum": minimum,
             "total_amount": float(amount.sum()), "records": len(frame)}
    for name in ("n", "bad", "good", "amount", "bad_amount", "expected_bad", "expected_bad_amount"):
        space[name] = first[name][:, None] + second[name][None, :]
    space["gap"] = abs(first["frr"][:, None] - second["frr"][None, :])
    space["risk"] = np.divide(space["bad"], space["n"], out=np.zeros_like(space["n"]), where=space["n"] > 0)
    space["fingerprint"] = hashlib.sha256(thresholds.astype("<f8").tobytes()).hexdigest()
    return space


def _identifier(family, basis, loss_ratio, risk_budget, capital_fraction, epsilon, delta):
    def number(x):
        return format(float(x), ".10g").replace("-", "m").replace(".", "p")
    key = f"{basis}_{family}_l{number(loss_ratio)}_r{number(risk_budget)}_c{number(capital_fraction)}"
    if epsilon is not None:
        key += f"_e{number(epsilon)}"
    if delta is not None:
        key += f"_d{number(delta)}"
    return key


def _solve(space: dict, frame: pd.DataFrame, config: dict, *, family: str,
           basis: str, loss_ratio: float, risk_budget: float, capital_fraction: float,
           epsilon: float | None = None, delta: float | None = None,
           baseline: dict | None = None, tags: list[str] | None = None) -> dict:
    if not 0 <= risk_budget <= 1 or not 0 < capital_fraction <= 1:
        raise ValueError("Risk budget must be in [0,1], capital fraction in (0,1]")
    if epsilon is not None and not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    if delta is not None and not 0 <= delta <= 1:
        raise ValueError("noharm delta must be in [0,1]")
    interest, cost, loss = _scenario(loss_ratio, config)
    n, bad = space["n"], space["bad"]
    utility = space["good"] - loss_ratio * bad
    amount_utility = ((interest - cost) * space["amount"]
                      - (interest + loss) * space["bad_amount"])
    objective_values = utility if basis == "unit_utility" else amount_utility
    objective = "approvals" if family == "group_max_approval" else basis
    ok = (n >= space["minimum"]) & (bad <= risk_budget * n + TOLERANCE)
    if family == "common_profit":
        ok &= np.eye(len(space["thresholds"]), dtype=bool)
    counts = {"risk_feasible_candidates": int(ok.sum())}
    ok &= space["amount"] <= capital_fraction * space["total_amount"] + max(1.0, space["total_amount"]) * 1e-12
    counts["capital_feasible_candidates"] = int(ok.sum())
    if epsilon is not None:
        ok &= space["gap"] <= epsilon + TOLERANCE
    counts["gap_feasible_candidates"] = int(ok.sum())
    reason = "No threshold candidate meets every constraint and minimum approvals"
    if delta is not None:
        if baseline is None or not baseline.get("feasible", False):
            ok[:] = False
            reason = "The matching unconstrained profit baseline is infeasible"
        else:
            bm = baseline["validation_metrics"]
            ok &= space["groups"][0]["frr"][:, None] <= bm["thick_frr"] + delta + TOLERANCE
            ok &= space["groups"][1]["frr"][None, :] <= bm["thin_frr"] + delta + TOLERANCE
    counts["final_feasible_candidates"] = int(ok.sum())
    counts["noharm_removed_candidates"] = counts["gap_feasible_candidates"] - int(ok.sum()) if delta is not None else 0
    policy = {
        "policy_id": _identifier(family, basis, loss_ratio, risk_budget, capital_fraction, epsilon, delta),
        "family": family, "objective": objective, "utility_basis": basis,
        "loss_ratio": float(loss_ratio), "risk_budget": float(risk_budget),
        "capital_fraction": float(capital_fraction), "epsilon": epsilon,
        "noharm_delta": delta, "baseline_policy_id": baseline["policy_id"] if baseline else None,
        "experiment_tags": tags or [], "status": "PASS" if ok.any() else "INFEASIBLE",
        "feasible": bool(ok.any()), "thresholds": None,
        "candidate_fingerprint": space["fingerprint"], "grid_counts": [len(space["thresholds"])] * 2,
        **counts,
    }
    if not ok.any():
        return {**policy, "reason": reason, "validation_metrics": None}
    i, j = np.where(ok)
    score, nn, risk, gap = (x[i, j] for x in (objective_values, n, space["risk"], space["gap"]))
    # Largest primary objective, then approvals (or utility), lower risk, smaller
    # gap, and finally lower threshold indices.  All families share tie rules.
    keys = (j, i, gap, risk, -score, -nn) if objective == "approvals" else (j, i, gap, risk, -nn, -score)
    best = np.lexsort(keys)[0]
    ii, jj = int(i[best]), int(j[best])
    policy.update({"thresholds": [float(space["thresholds"][ii]), float(space["thresholds"][jj])],
                   "indices": [ii, jj], "validation_objective": float(nn[best] if objective == "approvals" else score[best])})
    metrics = metrics_for_decisions(frame, apply_policy(frame, policy), loss_ratio=loss_ratio, config=config)
    policy["validation_metrics"] = metrics
    policy["validation_risk_slack"] = risk_budget - metrics["risk"]
    policy["validation_capital_slack"] = capital_fraction - metrics["capital_fraction_used"]
    if epsilon is not None:
        policy["validation_gap_slack"] = epsilon - metrics["abs_frr_gap"]
    if delta is not None and baseline is not None:
        for group in ("thin", "thick"):
            policy[f"validation_{group}_noharm_slack"] = baseline["validation_metrics"][f"{group}_frr"] + delta - metrics[f"{group}_frr"]
    return policy


def fit_policies(validation: pd.DataFrame, config: dict) -> dict:
    """Select thresholds on validation only; return a JSON-serializable ledger.

    Sensitivities vary one business parameter at a time: all risk budgets at the
    primary loss ratio, all loss ratios at the primary risk budget, and capital
    fractions for the amount objective.  This avoids an uninformative full
    Cartesian product of every setting.
    """
    space = _candidate_space(validation, config)
    primary_loss = float(config.get("primary_loss_ratio", 5))
    primary_risk = float(config.get("primary_risk_budget", 0.05))
    primary_epsilon = float(config.get("primary_epsilon", 0.02))
    epsilons = sorted(set(map(float, config.get("epsilons", [primary_epsilon]))) | {primary_epsilon}, reverse=True)
    settings: dict[tuple, list[str]] = {}
    def setting(basis, lam, risk, capital, tag):
        key = (basis, float(lam), float(risk), float(capital))
        settings.setdefault(key, [])
        if tag not in settings[key]:
            settings[key].append(tag)
    setting("unit_utility", primary_loss, primary_risk, 1.0, "primary")
    for risk in config.get("risk_budgets", [primary_risk]):
        setting("unit_utility", primary_loss, risk, 1.0, "risk_sensitivity")
    for lam in config.get("loss_ratios", [primary_loss]):
        setting("unit_utility", lam, primary_risk, 1.0, "loss_sensitivity")
    for capital in config.get("capital_fractions", [1.0, 0.6]):
        setting("amount_utility", primary_loss, primary_risk, capital, "capital_sensitivity")
    policies = []
    for (basis, lam, risk, capital), tags in settings.items():
        common_kwargs = {"basis": basis, "loss_ratio": lam, "risk_budget": risk,
                         "capital_fraction": capital, "tags": tags}
        common = _solve(space, validation, config, family="common_profit", **common_kwargs)
        maximum = _solve(space, validation, config, family="group_max_approval", **common_kwargs)
        profit = _solve(space, validation, config, family="group_profit", **common_kwargs)
        policies.extend([common, maximum, profit])
        for epsilon in epsilons:
            policies.append(_solve(space, validation, config, family="group_profit_gap", epsilon=epsilon, **common_kwargs))
        if lam == primary_loss and risk == primary_risk:
            for delta in sorted(set(map(float, config.get("noharm_deltas", [0.01])))):
                policies.append(_solve(space, validation, config, family="group_profit_gap_noharm",
                                       epsilon=primary_epsilon, delta=delta, baseline=profit, **common_kwargs))
    ids = validation.SK_ID_CURR.astype(str).sort_values().to_list()
    metadata = {
        "version": "3.0.0", "fit_split": "validation", "records": len(validation),
        "validation_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "candidate_fingerprint": space["fingerprint"], "threshold_grid": space["thresholds"].tolist(),
        "grid_requested": int(config.get("grid", 256)), "grid_actual": len(space["thresholds"]),
        "minimum_approvals": space["minimum"], "objective_data": "observed_validation_labels",
        "threshold_order": ["thick=0", "thin=1"], "decision_rule": "probability <= threshold[group]",
        "tie_break": "objective descending, approvals/utility descending, risk ascending, gap ascending, indices ascending",
        "economic_interpretation": "Scenario net contribution; TARGET denotes repayment difficulty, not observed monetary loss",
        "candidate_space": "Shared global quantiles plus reject-all/accept-all; same pair space across group policy families",
        "policy_count": len(policies), "feasible_count": sum(p["feasible"] for p in policies),
    }
    return {"metadata": metadata, "policies": policies}


def evaluate_policies(test: pd.DataFrame, frozen: dict, config: dict) -> list[dict]:
    """Replay validation-frozen policies; report test violations without repairs."""
    _arrays(test)
    policies = frozen["policies"]
    by_id = {p["policy_id"]: p for p in policies}
    metrics_cache = {p["policy_id"]: metrics_for_decisions(test, apply_policy(test, p), p["loss_ratio"], config)
                     for p in policies if p.get("feasible", False)}
    rows = []
    for policy in policies:
        row = {k: v for k, v in policy.items() if k != "validation_metrics"}
        row["evaluation_split"] = "test"
        if not policy.get("feasible", False):
            rows.append(row)
            continue
        metrics = metrics_cache[policy["policy_id"]]
        row.update(metrics)
        row["test_risk_constraint_met"] = metrics["risk"] is not None and metrics["risk"] <= policy["risk_budget"] + TOLERANCE
        row["test_capital_constraint_met"] = metrics["capital_fraction_used"] <= policy["capital_fraction"] + TOLERANCE
        row["test_gap_constraint_met"] = (None if policy["epsilon"] is None else
                                           metrics["abs_frr_gap"] is not None and metrics["abs_frr_gap"] <= policy["epsilon"] + TOLERANCE)
        row["test_noharm_constraint_met"] = None
        if policy["noharm_delta"] is not None:
            baseline = by_id[policy["baseline_policy_id"]]
            bm = metrics_cache.get(baseline["policy_id"])
            row["test_noharm_constraint_met"] = bool(bm and all(
                metrics[f"{group}_frr"] is not None and bm[f"{group}_frr"] is not None
                and metrics[f"{group}_frr"] <= bm[f"{group}_frr"] + policy["noharm_delta"] + TOLERANCE
                for group in ("thin", "thick")))
        rows.append(row)
    return rows
