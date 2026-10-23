"""Validation-only inverse pricing of a prespecified weakest-group opportunity gain.

The target is A_s >= W0 + eta for every one of the six normal-label cells.
It is NOT an eta improvement relative to each cell's own baseline rate.
No test prediction path is constructed or read by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..common import HERE, clean, dump, now, sha
from ..protected_policy import _Space, _fit_one, apply_protected_policy, protected_metrics
from .allocation_policy import _Program, _base_program, KEYS


ETAS = (.0005, .001, .002)
NUMERIC_TOL = 1e-8
CHECK_TOL = 3e-8


def _hash_object(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _minimum_harm(program, rates, reference, label):
    work = program.copy()
    old = len(work.bub)
    for rate, value in zip(rates, reference):
        work.upper(-rate, -value)
    result = work.solve(np.r_[np.zeros(program.dim), 1.], label,
                        extra_column=np.r_[np.zeros(old), -np.ones(len(rates))])
    return result, work.attempts


def price_program(program, rates, reference, profit, etas=ETAS, numeric_tol=NUMERIC_TOL):
    """Generic finite LP pricing oracle, also exercised on analytic toy problems.

    All w coordinates are [0,1], matching threshold-mixture weights. ``profit``
    is only used as a tie breaker; the original business constraints are retained.
    ``minimum_harm`` is an LP optimum. Numerical allowances are separately
    exposed, never relabeled as a substantive protection budget.
    """
    rates, reference, profit = (np.asarray(x, float) for x in (rates, reference, profit))
    etas = tuple(float(v) for v in etas)
    if rates.ndim != 2 or rates.shape[1] != program.dim or reference.shape != (len(rates),):
        raise ValueError("Malformed opportunity vectors")
    if not etas or len(set(etas)) != len(etas) or any(not np.isfinite(v) or v <= 0 for v in etas):
        raise ValueError("etas must be distinct, finite positive target increments")
    if numeric_tol < 0 or not np.isfinite(numeric_tol):
        raise ValueError("numeric_tol must be nonnegative")
    initial, attempts = _minimum_harm(program, rates, reference, "minimum_required_harm")
    if not initial.success:
        return {"status": "INFEASIBLE" if initial.status == 2 else "SOLVER_ERROR",
                "reason": str(initial.message), "solver_attempts": attempts, "rows": []}
    dstar = max(0., float(initial.x[-1]))
    face = program.copy()
    for rate, value in zip(rates, reference):
        face.upper(-rate, -value + dstar + numeric_tol)
    weakest = face.copy()
    old = len(weakest.bub)
    for rate in rates:
        weakest.upper(-rate, 0.)
    absolute = weakest.solve(np.r_[np.zeros(program.dim), -1.], "least_harm_face_best_absolute_minimum",
                              extra_column=np.r_[np.zeros(old), np.ones(len(rates))])
    attempts.extend(weakest.attempts)
    if not absolute.success:
        return {"status": "SOLVER_ERROR", "reason": str(absolute.message),
                "minimum_required_harm": dstar, "solver_attempts": attempts, "rows": []}
    w0 = float(absolute.x[-1])
    rows = []
    for eta in etas:
        target = w0 + eta
        targeted = program.copy()
        target_start = len(targeted.bub)
        for rate in rates:
            targeted.upper(-rate, -target)
        priced, traces = _minimum_harm(targeted, rates, reference, f"price_eta_{eta:g}")
        row = {"eta": eta, "eta_percentage_points": 100*eta,
               "W0": w0, "target_minimum_opportunity": target,
               "target_semantics": "Every six-cell A_s >= W0 + eta, not A_s >= own baseline + eta",
               "minimum_required_harm": dstar, "numerical_allowance": numeric_tol,
               "status": "PASS" if priced.success else ("INFEASIBLE" if priced.status == 2 else "SOLVER_ERROR"),
               "feasible": bool(priced.success), "solver_attempts": traces}
        if not priced.success:
            row.update(reason=str(priced.message), tau_star=None, weights=None)
            rows.append(row)
            continue
        priced_d = max(0., float(priced.x[-1]))
        if priced_d < dstar - CHECK_TOL:
            raise AssertionError("Targeted minimum harm unexpectedly below unrestricted optimum")
        tau = max(0., priced_d-dstar)
        target_marginals = np.asarray(priced.ineqlin.marginals)[target_start:target_start+len(rates)]
        row.update(minimum_harm_at_target=priced_d, tau_star=tau,
                   tau_star_percentage_points=100*tau,
                   target_rhs_marginals=target_marginals.tolist(),
                   local_tau_slope_wrt_eta=float(-target_marginals.sum()),
                   dual_scope="One LP dual subgradient, holding beta, W0 and all other constraints fixed")
        # Price first; then select the highest-contribution policy at that price.
        final = targeted.copy()
        for rate, value in zip(rates, reference):
            final.upper(-rate, -value + priced_d + numeric_tol)
        picked = final.solve(-profit, f"eta_{eta:g}_profit_tiebreak")
        row["solver_attempts"].extend(final.attempts)
        if not picked.success:
            row.update(status="SOLVER_ERROR", feasible=False, reason=str(picked.message), weights=None)
        else:
            approvals = rates @ picked.x
            harm = max(0., float(np.max(reference-approvals)))
            residual = max(0., float(np.max(program.aub @ picked.x-program.bub)),
                           float(np.max(np.abs(program.aeq @ picked.x-program.beq))))
            target_met = bool(np.min(approvals) >= target-CHECK_TOL)
            price_met = bool(harm <= priced_d+numeric_tol+CHECK_TOL)
            if not target_met or not price_met or residual > CHECK_TOL:
                raise AssertionError("Independent LP primal/target/price checks failed")
            row.update(weights=picked.x.tolist(), group_opportunity=approvals.tolist(),
                       achieved_minimum_opportunity=float(np.min(approvals)),
                       actual_maximum_harm=harm, actual_extra_harm=max(0., harm-dstar),
                       actual_profit=float(profit @ picked.x), target_met=target_met,
                       protection_price_met=price_met, base_constraint_max_violation=residual)
        rows.append(row)
    result = {"status": "PASS" if all(r["status"] != "SOLVER_ERROR" for r in rows) else "SOLVER_ERROR",
              "minimum_required_harm": dstar, "W0": w0,
              "W0_weights": absolute.x[:-1].tolist(),
              "W0_group_opportunity": (rates @ absolute.x[:-1]).tolist(),
              "numerical_allowance": numeric_tol, "solver_attempts": attempts, "rows": rows}
    # The primary constraint ordering is provided by _base_program. Generic toy
    # programs can omit these economics; only expose this dual when available.
    if len(initial.ineqlin.marginals) > 3:
        result["dstar_profit_floor_rhs_marginal"] = float(initial.ineqlin.marginals[3])
    return result


def _policy(space, weights, name):
    return {"name": name, "policy_id": name, "status": "PASS", "feasible": True,
            "decision_semantics": "stochastic_expectation", "mixture": {
                str(g): {"thresholds": space.thresholds.tolist(),
                         "weights": list(weights[g*space.k:(g+1)*space.k])}
                for g in (0, 1)}}


def fit_prices(validation, config):
    cfg = {"grid": 256, **config}
    beta, epsilon = float(cfg.get("primary_beta", .01)), float(cfg.get("primary_epsilon", .02))
    if not 0 <= beta <= 1 or not 0 <= epsilon <= 1:
        raise ValueError("Invalid business/fairness budget")
    space = _Space(validation, cfg)
    baseline = _fit_one(space, {"name": "lp_profit_baseline", "mode": "profit"})
    if not baseline.get("feasible") or baseline["validation_metrics"]["utility"] <= 0:
        raise ValueError("A feasible positive-profit matched-space baseline is required")
    program, bw, base_profit = _base_program(space, baseline, beta, epsilon)
    rates = np.asarray([space.whole["good_rates"][key] for key in KEYS])
    reference = rates @ bw
    result = price_program(program, rates, reference, space.whole["utility"] / len(validation),
                           etas=cfg.get("etas", ETAS), numeric_tol=cfg.get("numerical_allowance", NUMERIC_TOL))
    result.update(config=cfg, baseline=baseline, baseline_sha256=_hash_object(baseline),
                  baseline_utility=base_profit, group_order=KEYS,
                  baseline_group_opportunity=reference.tolist(), split="validation",
                  records=len(validation), scope="Finite prespecified opportunity targets; no test data read")
    if "dstar_profit_floor_rhs_marginal" in result:
        result["local_dstar_slope_wrt_beta"] = result["dstar_profit_floor_rhs_marginal"]*base_profit/len(validation)
    if result.get("W0_weights") is not None:
        w0_policy = _policy(space, result["W0_weights"], "least_harm_absolute_opportunity_reference")
        result["W0_policy"] = w0_policy
        result["W0_validation_metrics"] = protected_metrics(validation, apply_protected_policy(validation, w0_policy), space.loss, cfg)
    for row in result["rows"]:
        row.update(beta=beta, epsilon=epsilon, baseline_utility=base_profit,
                   baseline_sha256=result["baseline_sha256"])
        if not row["feasible"]:
            continue
        policy = _policy(space, row["weights"], f"inverse_price_eta_{row['eta']:g}")
        q = apply_protected_policy(validation, policy)
        metrics = protected_metrics(validation, q, space.loss, cfg)
        direct = np.array([1-metrics[key+"_frr"] for key in KEYS])
        if not np.allclose(direct, row["group_opportunity"], rtol=0, atol=CHECK_TOL):
            raise AssertionError("Serialized policy opportunities differ from LP coefficients")
        macro_harm = max(metrics["thick_frr"]-baseline["validation_metrics"]["thick_frr"],
                         metrics["thin_frr"]-baseline["validation_metrics"]["thin_frr"])
        checks = dict(target_met=bool(direct.min() >= row["target_minimum_opportunity"]-CHECK_TOL),
                      risk_met=metrics["risk"] <= space.risk+CHECK_TOL,
                      fairness_met=metrics["abs_frr_gap"] <= epsilon+CHECK_TOL,
                      profit_floor_met=metrics["utility"] >= (1-beta)*base_profit-CHECK_TOL*len(validation),
                      capital_met=metrics["capital_fraction_used"] <= space.capital+CHECK_TOL,
                      minimum_approvals_met=metrics["approvals"] >= space.minimum-CHECK_TOL*len(validation),
                      macro_protection_implied=macro_harm <= row["actual_maximum_harm"]+CHECK_TOL)
        if not all(checks.values()):
            raise AssertionError(f"Validation policy check failed: {checks}")
        row.update(policy=policy, validation_metrics=metrics, independent_checks=checks,
                   utility=metrics["utility"], risk=metrics["risk"], approvals=metrics["approvals"],
                   relative_profit_change=metrics["utility"]/base_profit-1,
                   abs_frr_gap=metrics["abs_frr_gap"])
    return result


def run_one(output, model, seed, cfg):
    started = time.perf_counter()
    job = f"{model}_s{seed}"
    origin = HERE/"results"/("run_001" if model == "xgboost" else "run_002")
    source = origin/"predictions"/f"{job}_validation.parquet"
    job_file = origin/"jobs"/f"{job}.json"
    old = json.loads(job_file.read_text(encoding="utf-8"))
    if old.get("status") != "PASS":
        raise ValueError(f"Frozen scorer is not complete: {job}")
    source_hash = sha(source)
    expected = old.get("prediction_sha256", {}).get("validation")
    if expected is not None and source_hash != expected:
        raise ValueError("Frozen validation predictions changed")
    result = fit_prices(pd.read_parquet(source), cfg)
    result.update(job=job, model=model, seed=seed, source_validation=str(source),
                  source_validation_sha256=source_hash, source_job_sha256=sha(job_file),
                  completed_utc=now(), wall_seconds=time.perf_counter()-started)
    dump(output/"jobs"/f"{job}.json", result)
    print(json.dumps({"job": job, "status": result["status"],
                      "dstar": result.get("minimum_required_harm"), "W0": result.get("W0"),
                      "prices": [{k:r.get(k) for k in ("eta", "status", "tau_star")} for r in result["rows"]]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=[f"{m}_s{s}" for m in ("lightgbm", "tabm", "xgboost") for s in (731, 732, 733)])
    args = parser.parse_args()
    cfg = dict(grid=256, minimum_approvals=500, minimum_subgroup_normal=100,
               primary_loss_ratio=5, primary_risk_budget=.05, capital_fraction=1.,
               primary_epsilon=.02, primary_beta=.01, etas=list(ETAS),
               numerical_allowance=NUMERIC_TOL, scenario_interest=.12, scenario_cost=.03)
    jobs = [(m, s) for m in ("lightgbm", "tabm", "xgboost") for s in (731, 732, 733)]
    dependencies = [Path(__file__), Path(__file__).with_name("allocation_policy.py"), HERE/"protected_policy.py"]
    plan = dict(config=cfg, jobs=[dict(model=m, seed=s) for m,s in jobs],
                source_sha256={p.name:sha(p) for p in dependencies},
                target_semantics="All six A_s >= W0 + eta; W0 is best absolute weakest opportunity on minimum-harm face",
                target_selection="Fixed eta values; no selection on validation or test results; no test input read")
    plan_path = args.output/"pricing_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing inverse-pricing output has a different frozen plan")
    dump(plan_path, plan)
    for model, seed in jobs:
        job = f"{model}_s{seed}"
        if args.only and args.only != job:
            continue
        marker = args.output/"jobs"/f"{job}.json"
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("status") == "PASS":
            continue
        result = run_one(args.output, model, seed, cfg)
        if result["status"] == "SOLVER_ERROR":
            raise RuntimeError(f"Unresolved numerical failure: {job}")
    flat, completed = [], 0
    for model, seed in jobs:
        path = args.output/"jobs"/f"{model}_s{seed}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        completed += int(result["status"] == "PASS")
        for row in result["rows"]:
            item = {k:v for k,v in row.items() if not isinstance(v, (dict, list))}
            item.update(job=result["job"], model=model, seed=seed,
                        source_validation_sha256=result["source_validation_sha256"])
            for key,value in zip(KEYS, row.get("group_opportunity", [])):
                item[key+"_opportunity"] = value
            flat.append(item)
    if flat:
        pd.DataFrame(flat).to_csv(args.output/"opportunity_prices.csv", index=False, encoding="utf-8-sig")
    dump(args.output/"status.json", dict(status="COMPLETE" if completed == len(jobs) else "PARTIAL",
         completed_jobs=completed, expected_jobs=len(jobs), rows=len(flat), updated=now(), split="validation_only"))


if __name__ == "__main__":
    main()
