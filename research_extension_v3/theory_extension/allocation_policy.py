"""Validation-only opportunity allocation on the minimum-deterioration face.

The normalized leximin rule is inspired by fair resource allocation.  No Nash
or Kalai--Smorodinsky axioms are claimed for this six-population construction.
All LPs retain the same risk, capital, profit, gap and deterioration limits.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
from scipy.optimize import OptimizeResult, linprog

from ..protected_policy import (
    CELLS, _Space, _fit_one, apply_protected_policy,
    protected_metrics,
)


FACE_TOL = 1e-8
RANGE_TOL = 1e-7
LEX_TOL = 1e-9
KEYS = [f"subgroup_{g}_{b}" for g, b in CELLS]


class _Program:
    """A bounded LP with an audit trail; all variables are in [0, 1]."""
    def __init__(self, aub, bub, aeq, beq):
        self.aub = np.asarray(aub, float)
        self.bub = np.asarray(bub, float)
        self.aeq = np.asarray(aeq, float)
        self.beq = np.asarray(beq, float)
        self.dim = self.aeq.shape[1]
        self.attempts = []

    def copy(self):
        return _Program(self.aub.copy(), self.bub.copy(), self.aeq.copy(), self.beq.copy())

    def upper(self, row, rhs):
        self.aub = np.vstack([self.aub, np.asarray(row, float)])
        self.bub = np.r_[self.bub, float(rhs)]

    def equal(self, row, rhs):
        self.aeq = np.vstack([self.aeq, np.asarray(row, float)])
        self.beq = np.r_[self.beq, float(rhs)]

    def solve(self, objective, label, extra_column=None):
        if extra_column is None:
            aub, aeq = self.aub, self.aeq
        else:
            aub = np.column_stack([self.aub, np.asarray(extra_column, float)])
            aeq = np.column_stack([self.aeq, np.zeros(len(self.beq))])
        # Tight faces require the smallest tolerance supported by HiGHS. An
        # infeasible presolve status is retried on the unchanged matrices.
        attempts=[]
        for method,presolve in (("highs",True),("highs-ds",False),("highs-ipm",True),("highs-ipm",False)):
            result=linprog(np.asarray(objective,float),A_ub=aub,b_ub=self.bub,
                           A_eq=aeq,b_eq=self.beq,bounds=[(0.,1.)]*len(objective),
                           method=method,options={"presolve":presolve,
                             "dual_feasibility_tolerance":1e-10,"primal_feasibility_tolerance":1e-10,
                             "ipm_optimality_tolerance":1e-12})
            attempts.append({"method":method,"presolve":presolve,"status":int(result.status),"message":str(result.message)})
            if result.success:
                violation=max(0.,float(np.max(aub@result.x-self.bub)),
                              float(np.max(np.abs(aeq@result.x-self.beq))),float(-np.min(result.x)),float(np.max(result.x)-1))
                if violation<=2e-8:
                    break
                result=OptimizeResult(success=False,status=4,message="LP result violates unchanged primal constraints")
        result.solver_attempts=attempts
        self.attempts.append({"stage": label, "attempts": deepcopy(result.solver_attempts),
                              "success": bool(result.success), "status": int(result.status)})
        return result


def _ranges(program, rates):
    lower, upper = [], []
    for i, row in enumerate(rates):
        lo = program.solve(row, f"range_{i}_min")
        if not lo.success:
            return None, None, lo
        hi = program.solve(-row, f"range_{i}_max")
        if not hi.success:
            return None, None, hi
        lower.append(float(row @ lo.x))
        upper.append(float(row @ hi.x))
    return np.asarray(lower), np.asarray(upper), None


def _allocate(program, rates, lower, upper, profit, *, mode, lex_tol=LEX_TOL):
    """Allocate affine utilities and retain every stage for independent checks.

    Normalization is encoded as -a_i w + span_i t <= -lower_i, avoiding
    division by a small span in the LP matrix.  Full leximin freezes only
    coordinates proven unable to exceed the current water level on its face.
    """
    rates, lower, upper = map(lambda x: np.asarray(x, float), (rates, lower, upper))
    spans = upper - lower
    active = np.flatnonzero(spans > RANGE_TOL).tolist()
    if mode == "absolute_maxmin":
        remaining = list(range(len(rates)))
        offsets, scales = np.zeros(len(rates)), np.ones(len(rates))
    elif mode in ("normalized_maxmin", "normalized_leximin"):
        remaining = active.copy()
        offsets, scales = lower, spans
    else:
        raise ValueError(f"Unknown allocation mode: {mode}")
    work = program.copy()
    stages, frozen = [], {}
    while remaining:
        step = work.copy()
        old_count = len(step.bub)
        for i in remaining:
            step.upper(-rates[i], -offsets[i])
        column = np.r_[np.zeros(old_count), scales[remaining]]
        objective = np.r_[np.zeros(work.dim), -1.]
        fill = step.solve(objective, f"waterfill_{len(stages)}", extra_column=column)
        program.attempts.extend(step.attempts)
        if not fill.success:
            return fill, stages, frozen
        level = float(fill.x[-1])
        # Use exact optimizer value; the 1e-8 allowance is only on the original
        # minimum-deterioration face, not an economic leximin relaxation.
        face = work.copy()
        for i in remaining:
            face.upper(-rates[i], -(offsets[i] + scales[i] * level))
        stage = {"level": level, "remaining": remaining.copy(), "frozen": [],
                 "coordinate_maxima_on_optimal_face": {}}
        if mode != "normalized_leximin":
            # A one-stage maxmin ablation deliberately does not claim leximin.
            final = face.solve(-profit, "maxmin_profit_tiebreak")
            program.attempts.extend(face.attempts)
            stages.append(stage)
            return final, stages, frozen
        forced = []
        for i in remaining:
            check = face.solve(-rates[i], f"forced_coordinate_{len(stages)}_{i}")
            if not check.success:
                program.attempts.extend(face.attempts)
                return check, stages, frozen
            maximum = float((float(rates[i] @ check.x) - offsets[i]) / scales[i])
            stage["coordinate_maxima_on_optimal_face"][str(i)] = maximum
            if maximum <= level + lex_tol:
                forced.append(i)
        program.attempts.extend(face.attempts)
        if not forced:
            return OptimizeResult(success=False, status=4,
                message="No forced coordinate on the maxmin face; numerical leximin certification failed"), stages, frozen
        stage["frozen"] = forced
        stages.append(stage)
        # Preserve the whole optimal face, not just an arbitrary maximizing
        # point, and freeze every coordinate proven forced on that face.
        work = face
        work.attempts = []
        for i in forced:
            value = offsets[i] + scales[i] * level
            work.equal(rates[i], value)
            frozen[str(i)] = {"normalized_value": level, "acceptance_rate": float(value)}
        remaining = [i for i in remaining if i not in forced]
    final = work.solve(-profit, "all_coordinates_fixed_profit_tiebreak")
    program.attempts.extend(work.attempts)
    return final, stages, frozen


def _base_program(space, baseline, beta, epsilon):
    stats, n = space.whole, len(space.frame)
    bw = np.concatenate([np.asarray(baseline["mixture"][str(g)]["weights"]) for g in (0, 1)])
    utility = float(stats["utility"] @ bw)
    difference = stats["good_rates"]["macro_0"] - stats["good_rates"]["macro_1"]
    aub = [-stats["n"] / n, stats["amount"] / space.total_amount,
           (stats["bad"] - space.risk * stats["n"]) / n,
           -stats["utility"] / n, difference, -difference]
    bub = [-space.minimum / n, space.capital, 0., -(1-beta) * utility / n,
           epsilon, epsilon]
    aeq = np.zeros((2, space.dim))
    for g in (0, 1):
        aeq[g, g*space.k:(g+1)*space.k] = 1.
    return _Program(aub, bub, aeq, np.ones(2)), bw, utility


def _failure(policy, result, program, reason=None, known_feasible=False):
    return {**policy, "status": "INFEASIBLE" if result.status == 2 and not known_feasible else "SOLVER_ERROR",
            "feasible": False, "mixture": None, "validation_metrics": None,
            "solver_status": int(result.status), "reason": reason or str(result.message),
            "solver_attempts": deepcopy(program.attempts)}


def fit_allocation_policies(validation, config):
    """Fit a shared baseline and three allocation rules per predeclared beta."""
    cfg = {"grid": 256, **config}
    betas = [float(v) for v in cfg.get("betas", [cfg.get("primary_beta", .01)])]
    epsilon = float(cfg.get("primary_epsilon", .02))
    modes = cfg.get("allocation_modes", ["absolute_maxmin", "normalized_maxmin", "normalized_leximin"])
    if not betas or len(betas) != len(set(betas)) or any(not 0 <= b <= 1 for b in betas):
        raise ValueError("betas must be distinct values in [0,1]")
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("primary_epsilon must lie in [0,1]")
    if len(modes) != len(set(modes)) or any(m not in ("absolute_maxmin", "normalized_maxmin", "normalized_leximin") for m in modes):
        raise ValueError("allocation_modes contains duplicate or unknown modes")
    space = _Space(validation, cfg)
    baseline = _fit_one(space, {"name": "lp_profit_baseline", "mode": "profit"})
    policies = [baseline]
    rates = np.asarray([space.whole["good_rates"][key] for key in KEYS])
    for beta in betas:
        templates = [{"name": f"allocation_{mode}_b{beta:g}",
                      "policy_id": f"allocation_{mode}_b{beta:g}", "family": "opportunity_allocation_lp",
                      "mode": mode, "protection": "six", "minimax_scope": "six", "robust": False,
                      "epsilon": epsilon, "beta": beta, "noharm_delta": None, "delta": None,
                      "baseline_policy_id": "lp_profit_baseline", "is_lp_profit_baseline": False,
                      "loss_ratio": space.loss, "risk_budget": space.risk,
                      "capital_fraction": space.capital, "utility_basis": "unit_utility",
                      "objective": mode+"_then_profit", "decision_semantics": "stochastic_expectation",
                      "face_numerical_tolerance": FACE_TOL,
                      "range_fixed_tolerance": RANGE_TOL, "leximin_identification_tolerance": LEX_TOL}
                     for mode in modes]
        if not baseline.get("feasible") or baseline["validation_metrics"]["utility"] <= 0:
            for p in templates:
                policies.append({**p, "status": baseline["status"] if not baseline.get("feasible") else "INFEASIBLE",
                                 "feasible": False, "mixture": None, "validation_metrics": None,
                                 "reason": "A feasible positive-profit LP baseline is required"})
            continue
        program, bw, base_profit = _base_program(space, baseline, beta, epsilon)
        base_rates = rates @ bw
        first = program.copy()
        old_count = len(first.bub)
        for row, value in zip(rates, base_rates):
            first.upper(-row, -value)
        stage1 = first.solve(np.r_[np.zeros(space.dim), 1.], "minimum_deterioration",
                             extra_column=np.r_[np.zeros(old_count), -np.ones(len(rates))])
        program.attempts.extend(first.attempts)
        if not stage1.success:
            policies.extend(_failure(p, stage1, program) for p in templates)
            continue
        dstar = max(0., float(stage1.x[-1]))
        effective = dstar + FACE_TOL
        marginal = float(stage1.ineqlin.marginals[3])
        sensitivity = {"normalized_profit_floor_rhs_marginal": marginal,
                       "profit_floor_multiplier": -marginal,
                       "local_dstar_slope_wrt_beta": marginal*base_profit/len(validation),
                       "interpretation": "One LP dual sensitivity/subgradient; local linear slope only while the optimal basis remains valid"}
        for row, value in zip(rates, base_rates):
            program.upper(-row, -value + effective)
        lower, upper, failed = _ranges(program, rates)
        diagnostics = None if failed is not None else [
            {"group": key, "index": i, "minimum_acceptance": float(lower[i]),
             "maximum_acceptance": float(upper[i]), "range": float(upper[i]-lower[i]),
             "fixed": bool(upper[i]-lower[i] <= RANGE_TOL),
             "baseline_acceptance": float(base_rates[i])}
            for i, key in enumerate(KEYS)]
        shared = {"minimum_required_delta": dstar, "effective_delta": effective,
                  "minimum_deterioration_dual": sensitivity,
                  "noharm_delta": effective, "delta": effective,
                  "validation_profit_floor": (1-beta)*base_profit,
                  "validation_baseline_profit": base_profit, "range_diagnostics": diagnostics,
                  "fixed_groups": [] if diagnostics is None else [r["group"] for r in diagnostics if r["fixed"]]}
        if failed is not None:
            policies.extend(_failure({**p, **shared}, failed, program, known_feasible=True) for p in templates)
            continue
        for p in templates:
            work = program.copy()
            work.attempts = deepcopy(program.attempts)
            result, stages, fixed = _allocate(work, rates, lower, upper,
                                               space.whole["utility"]/len(validation), mode=p["mode"])
            policy = {**p, **shared, "leximin_stages": stages, "leximin_fixed_coordinates": fixed}
            if not result.success:
                policies.append(_failure(policy, result, work, known_feasible=True))
                continue
            weights = np.maximum(np.asarray(result.x[:space.dim]), 0).reshape(2, space.k)
            weights /= weights.sum(axis=1, keepdims=True)
            w = weights.ravel()
            violation = max(0., float(np.max(program.aub@w-program.bub)),
                            float(np.max(np.abs(program.aeq@w-program.beq))))
            # Tighter than the legacy common solver acceptance: do not turn the
            # intended 1e-8 face allowance into a hidden economic relaxation.
            if violation > 2e-8:
                err = OptimizeResult(success=False, status=4, message="Final policy violates the minimum-deterioration face")
                policies.append(_failure(policy, err, work))
                continue
            accepts = rates @ w
            active = upper-lower > RANGE_TOL
            normalized = [float((accepts[i]-lower[i])/(upper[i]-lower[i])) if active[i] else None
                          for i in range(len(rates))]
            policy.update(status="PASS", feasible=True, solver_status=int(result.status),
                          solver_attempts=deepcopy(work.attempts),
                          mixture={str(g): {"thresholds": space.thresholds.tolist(), "weights": weights[g].tolist()} for g in (0,1)},
                          active_mixture={str(g): {"thresholds": space.thresholds[weights[g]>1e-12].tolist(),
                                                  "weights": weights[g][weights[g]>1e-12].tolist()} for g in (0,1)},
                          nonzero_thresholds=[int(np.count_nonzero(weights[g]>1e-12)) for g in (0,1)],
                          validation_max_constraint_violation=violation,
                          validation_subgroup_acceptance=accepts.tolist(),
                          validation_normalized_opportunity=normalized,
                          validation_sorted_normalized_opportunity=sorted(v for v in normalized if v is not None),
                          validation_max_deterioration=float(max(0., np.max(base_rates-accepts))))
            policy["validation_metrics"] = protected_metrics(validation, apply_protected_policy(validation, policy), space.loss, cfg)
            policies.append(policy)
    ids = validation.SK_ID_CURR.astype(str).sort_values().tolist()
    return {"metadata": {"version": "opportunity-allocation-lp-1.0", "fit_split": "validation",
                         "records": len(validation), "validation_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                         "threshold_grid": space.thresholds.tolist(), "grid_requested": cfg["grid"], "grid_actual": space.k,
                         "betas": betas, "modes": list(modes), "group_order": KEYS,
                         "objective_data": "observed_validation_labels", "approval_features": ["probability", "thin"],
                         "audit_only_feature": "income_band", "decision_semantics": "stochastic_expectation",
                         "range_definition": "Each coordinate minimum and maximum on the SAME minimum-deterioration face",
                         "policy_count": len(policies), "feasible_count": sum(p.get("feasible",False) for p in policies)},
            "policies": policies}
