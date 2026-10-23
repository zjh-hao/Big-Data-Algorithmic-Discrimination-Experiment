"""Profit-budgeted subgroup protection using mixtures of monotone thresholds.

All selection uses validation labels.  Policies use probability and ``thin``
only at application time: income bands are audit/objective cells, never a new
approval feature.  Counts, profit and constraints are expectations under the
frozen randomized policy; risk is a ratio of expected counts, not a guarantee
for one realization of lending.  TARGET=1 remains a repayment-difficulty proxy.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, linprog

from .business import _arrays, metrics_for_decisions


TOL = 2e-7
CELLS = [(g, b) for g in (0, 1) for b in (0, 1, 2)]


def _linear_solve(objective, *, A_ub, b_ub, A_eq, b_eq, bounds):
    """Retry numerical solver failures without changing economic constraints.

    HiGHS presolve can return an unknown status at a tight lexicographic face.
    Switching algorithm/presolve addresses that numerical failure; every
    accepted solution is checked against the original, unchanged LP matrices.
    """
    attempts=[]
    options={"dual_feasibility_tolerance":1e-9,"primal_feasibility_tolerance":1e-9}
    for method,presolve in (("highs",True),("highs-ds",False),("highs-ipm",True),("highs-ipm",False)):
        result=linprog(objective,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,
                       bounds=bounds,method=method,options={**options,"presolve":presolve})
        if result.success:
            x=np.asarray(result.x)
            violation=max(0.,float(np.max(np.asarray(A_ub)@x-np.asarray(b_ub))),
                          float(np.max(np.abs(np.asarray(A_eq)@x-np.asarray(b_eq)))),
                          float(-np.min(x)),float(np.max(x)-1))
            if not np.isfinite(x).all() or violation>TOL:
                result=OptimizeResult(success=False,status=4,message="Returned solution violates original LP primal constraints",x=x)
        attempts.append({"method":method,"presolve":presolve,"status":int(result.status),"message":str(result.message)})
        result.solver_attempts=attempts
        if result.success:
            return result
        # The ordinary initial solver has an explicit infeasibility certificate.
        # If it first failed numerically, continue all retries before judging.
        if len(attempts)==1 and result.status==2:
            return result
    return OptimizeResult(success=False,status=4,message="Numerical LP failure after algorithm/presolve retries",solver_attempts=attempts)


def _band(frame):
    if "income_band" not in frame or not np.isin(frame.income_band.to_numpy(), [0, 1, 2]).all():
        raise ValueError("income_band must contain training-defined terciles 0/1/2")
    return frame.income_band.to_numpy(dtype=int)


def protected_metrics(frame, decisions, loss_ratio=5, config=None):
    """Expected business metrics plus six explicit normal-label denominators."""
    result = metrics_for_decisions(frame, decisions, loss_ratio, config)
    band, q = _band(frame), np.asarray(decisions, dtype=float)
    g, y = frame.thin.to_numpy(), frame.TARGET.to_numpy()
    frrs = []
    for group, income in CELLS:
        key = f"subgroup_{group}_{income}"
        mask = (g == group) & (band == income)
        normal = mask & (y == 0)
        count = int(normal.sum())
        accepted = float(q[mask].sum())
        bad = float(np.dot(q[mask], y[mask]))
        frr = 1 - float(q[normal].sum()) / count if count else None
        result.update({key + "_normal_records": count, key + "_records": int(mask.sum()),
                       key + "_frr": frr, key + "_approvals": accepted,
                       key + "_risk": bad / accepted if accepted else None})
        if frr is not None:
            frrs.append(frr)
    result["worst_subgroup_frr"] = max(frrs) if len(frrs) == 6 else None
    macro = [result["thick_frr"], result["thin_frr"]]
    result["worst_macro_frr"] = max(macro) if all(v is not None for v in macro) else None
    return result


def apply_protected_policy(frame: pd.DataFrame, policy: dict) -> np.ndarray:
    """Return approval probabilities; applying a policy never reads labels/income."""
    if not policy.get("feasible") or policy.get("mixture") is None:
        raise ValueError("INFEASIBLE policy has no approval decisions")
    required = {"probability", "thin"}
    if not required.issubset(frame):
        raise ValueError("Applying a policy requires probability and thin only")
    p, g = frame.probability.to_numpy(dtype=float), frame.thin.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isin(g, [0, 1]).all():
        raise ValueError("Invalid probabilities or groups")
    result = np.zeros(len(frame), dtype=float)
    for group in (0, 1):
        entry = policy["mixture"][str(group)]
        thresholds = np.asarray(entry["thresholds"], dtype=float)
        weights = np.asarray(entry["weights"], dtype=float)
        if (thresholds.ndim != 1 or weights.shape != thresholds.shape or
                not np.isfinite(thresholds).all() or not np.isfinite(weights).all() or
                (weights < -TOL).any() or abs(float(weights.sum()) - 1) > TOL or
                (np.diff(thresholds) < 0).any()):
            raise ValueError("Malformed threshold mixture")
        mask = g == group
        # The tail sum handles <= thresholds, including scores exactly on a knot.
        tail = np.r_[np.cumsum(weights[::-1])[::-1], 0.0]
        result[mask] = tail[np.searchsorted(thresholds, p[mask], side="left")]
    return np.clip(result, 0, 1)


class _Space:
    def __init__(self, frame, config):
        data = _arrays(frame)
        band = _band(frame)
        grid = int(config.get("grid", 257))
        if grid < 2:
            raise ValueError("grid must be at least 2")
        unique = np.unique(data["p"])
        knots = unique if len(unique) <= grid else np.quantile(data["p"], np.linspace(0, 1, grid))
        self.thresholds = np.unique(np.r_[-1., knots, 1.])
        self.k = len(self.thresholds)
        self.dim = self.k * 2
        self.frame = frame
        self.config = config
        self.data = data
        self.band = band
        self.minimum = max(1, int(config.get("minimum_approvals", 500)))
        self.minimum_normal = max(1, int(config.get("minimum_subgroup_normal", 100)))
        self.loss = float(config.get("primary_loss_ratio", 5))
        self.risk = float(config.get("primary_risk_budget", .05))
        self.capital = float(config.get("capital_fraction", 1))
        if not np.isfinite([self.loss, self.risk, self.capital]).all() or self.loss <= 0 or not 0 <= self.risk <= 1 or not 0 < self.capital <= 1:
            raise ValueError("Invalid loss, risk, or capital setting")
        self.total_amount = float(data["amount"].sum())
        self.whole = self.statistics(np.ones(len(frame), bool), require_support=True)
        # Hash parity is fixed by record ID and never depends on outcomes.
        ids = pd.to_numeric(frame.SK_ID_CURR, errors="raise").to_numpy(dtype=np.int64)
        self.halves = [self.statistics(ids % 2 == half, require_support=False) for half in (0, 1)]

    def vector(self, values, mask):
        out = np.zeros(self.dim)
        for group in (0, 1):
            take = mask & (self.data["g"] == group)
            order = np.argsort(self.data["p"][take], kind="stable")
            pg = self.data["p"][take][order]
            vg = np.asarray(values)[take][order]
            positions = np.searchsorted(pg, self.thresholds, side="right")
            out[group*self.k:(group+1)*self.k] = np.r_[0., np.cumsum(vg, dtype=float)][positions]
        return out

    def statistics(self, mask, require_support):
        y, g = self.data["y"], self.data["g"]
        ones = np.ones(len(y))
        result = {"records": int(mask.sum()), "n": self.vector(ones, mask),
                  "bad": self.vector(y, mask), "utility": self.vector(1-(1+self.loss)*y, mask),
                  "amount": self.vector(self.data["amount"], mask), "good_rates": {}}
        for group in (0, 1):
            take = mask & (g == group) & (y == 0)
            normal = int(take.sum())
            if normal:
                result["good_rates"][f"macro_{group}"] = self.vector(ones, take) / normal
            elif require_support:
                raise ValueError("Validation needs normal-label support in both macro groups")
            for income in (0, 1, 2):
                cell = take & (self.band == income)
                count = int(cell.sum())
                if require_support and count < self.minimum_normal:
                    raise ValueError(f"Subgroup {group}/{income} has {count} normal records; need {self.minimum_normal}")
                if count:
                    result["good_rates"][f"subgroup_{group}_{income}"] = self.vector(ones, cell) / count
        return result


def _reference_value(stats, key, weights):
    return 1 - float(stats["good_rates"][key] @ weights)


def _fit_adaptive(space, spec, baseline):
    """First minimize required harm, then minimize worst-cell FRR and maximize profit.

    This is a predeclared validation-only fallback when the requested one-point
    no-harm goal is infeasible.  The effective deterioration cap is reported
    explicitly, rather than claiming that the requested cap was satisfied.
    """
    beta = float(spec.get("beta", space.config.get("primary_beta", .01)))
    epsilon = float(spec.get("epsilon", space.config.get("primary_epsilon", .02)))
    requested = float(spec.get("delta", space.config.get("primary_delta", .01)))
    slack = float(spec.get("deterioration_slack", .001))
    robust = bool(spec.get("robust", False))
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in (beta,epsilon,requested,slack)):
        raise ValueError("Adaptive beta,epsilon,requested delta,slack must lie in [0,1]")
    protocol = "Validation-only LP: minimize required shared macro/six-group FRR deterioration; add predeclared slack; minimize worst-six FRR; maximize unit scenario profit"
    adaptation = {"requested_delta": requested, "minimum_required_delta": None,
                  "deterioration_slack": slack, "effective_delta": None,
                  "adaptation_protocol": protocol, "validation_requested_goal_met": False}
    final_spec = {**spec, "name": str(spec.get("name", "adaptive_protected_minimax")),
                  "mode": "budgeted_minimax", "protection": "six", "minimax_scope": "six"}
    if not baseline or not baseline.get("feasible") or baseline["validation_metrics"]["utility"] <= 0:
        result = _fit_one(space, final_spec, baseline)
        return {**result, **adaptation, "mode": "adaptive_protected_minimax"}
    base_weights = np.concatenate([np.asarray(baseline["mixture"][str(g)]["weights"]) for g in (0,1)])
    dim = space.dim+1
    rows, limits = [],[]
    def add(vector, limit, dcoef=0):
        rows.append(np.r_[vector,dcoef]);limits.append(float(limit))
    whole = space.whole
    records = len(space.frame)
    add(-whole["n"]/records,-space.minimum/records)
    add(whole["amount"]/space.total_amount,space.capital)
    profit_floor = (1-beta)*float(whole["utility"]@base_weights)
    add(-whole["utility"]/records,-profit_floor/records)
    scenarios = [("whole",whole)]
    if robust:
        scenarios += [(f"half_{half}",values) for half,values in enumerate(space.halves)]
    missing = None
    for scenario_name,scenario in scenarios:
        add((scenario["bad"]-space.risk*scenario["n"])/max(1,scenario["records"]),0)
        rates = scenario["good_rates"]
        if not {"macro_0","macro_1"}.issubset(rates):
            missing = f"Missing macro normal support in {scenario_name}"
            break
        difference = rates["macro_0"]-rates["macro_1"]
        add(difference,epsilon);add(-difference,epsilon)
        for key in ["macro_0","macro_1"]+[f"subgroup_{g}_{b}" for g,b in CELLS]:
            if key not in rates:
                missing = f"Missing {key} normal support in {scenario_name}"
                break
            add(-rates[key],_reference_value(scenario,key,base_weights)-1,dcoef=-1)
        if missing:
            break
    aeq = np.zeros((2,dim))
    for group in (0,1):
        aeq[group,group*space.k:(group+1)*space.k]=1
    objective=np.zeros(dim);objective[-1]=1
    stage1 = None if missing else _linear_solve(objective,A_ub=np.asarray(rows),b_ub=np.asarray(limits),
                     A_eq=aeq,b_eq=np.ones(2),bounds=[(0.,1.)]*dim)
    if missing or not stage1.success:
        # Reuse the standard infeasibility serialization, then retain stage-one
        # failure explicitly.  A d=1 cap is maximally permissive on FRR.
        result = _fit_one(space,{**final_spec,"delta":1.},baseline)
        result.update(status="INFEASIBLE" if missing or stage1.status==2 else "SOLVER_ERROR",feasible=False,mixture=None,validation_metrics=None,
                      reason=missing or "Adaptive minimum-deterioration LP failed: "+stage1.message)
        result.pop("active_mixture",None)
        return {**result, **adaptation, "mode":"adaptive_protected_minimax",
                "adaptation_solver_status":None if stage1 is None else int(stage1.status),
                "adaptation_solver_attempts":[] if stage1 is None else stage1.solver_attempts}
    required = max(0.,float(stage1.x[-1]))
    effective = min(1.,required+slack)
    adaptation.update(minimum_required_delta=required,effective_delta=effective,
                      adaptation_solver_status=int(stage1.status),
                      adaptation_solver_attempts=stage1.solver_attempts,
                      adaptation_max_constraint_violation=float(max(0,np.max(np.asarray(rows)@stage1.x-np.asarray(limits)))))
    result = _fit_one(space,{**final_spec,"delta":effective},baseline)
    if result.get("feasible"):
        # A requested goal is considered attained only if every relevant
        # actual validation FRR meets it, not merely d* <= requested.
        q = apply_protected_policy(space.frame,result)
        reference = apply_protected_policy(space.frame,baseline)
        checks=[]
        masks=[np.ones(len(space.frame),bool)]
        if robust:
            ids=pd.to_numeric(space.frame.SK_ID_CURR).to_numpy(dtype=np.int64)
            masks += [ids%2==half for half in (0,1)]
        for mask in masks:
            current=protected_metrics(space.frame.loc[mask],q[mask],space.loss,space.config)
            previous=protected_metrics(space.frame.loc[mask],reference[mask],space.loss,space.config)
            keys=["thin_frr","thick_frr"]+[f"subgroup_{g}_{b}_frr" for g,b in CELLS]
            checks.extend(current[k] is not None and previous[k] is not None and current[k]<=previous[k]+requested+TOL for k in keys)
        adaptation["validation_requested_goal_met"]=bool(all(checks))
    return {**result, **adaptation, "mode":"adaptive_protected_minimax"}


def _fit_one(space, spec, baseline=None):
    mode = spec.get("mode", "budgeted_minimax")
    if mode == "adaptive_protected_minimax":
        return _fit_adaptive(space,spec,baseline)
    if mode not in ("profit", "profit_gap", "profit_gap_protected", "budgeted_minimax"):
        raise ValueError(f"Unknown mode {mode}")
    protected = mode in ("profit_gap_protected", "budgeted_minimax")
    protection = spec.get("protection", "six" if mode == "budgeted_minimax" else "macro") if protected else "none"
    scope = spec.get("minimax_scope", "six")
    if protection not in ("none", "macro", "six") or scope not in ("macro", "six"):
        raise ValueError("protection/minimax_scope must be macro or six")
    epsilon = None if mode == "profit" else float(spec.get("epsilon", space.config.get("primary_epsilon", .02)))
    beta = float(spec.get("beta", space.config.get("primary_beta", .01))) if mode == "budgeted_minimax" else None
    delta = float(spec.get("delta", space.config.get("primary_delta", .01))) if protected else None
    if any(value is not None and (not np.isfinite(value) or not 0 <= value <= 1) for value in (epsilon, beta, delta)):
        raise ValueError("epsilon, beta and delta must lie in [0,1]")
    minimax = mode == "budgeted_minimax"
    robust = bool(spec.get("robust", False))
    dim = space.dim + int(minimax)
    rowlist, rhs, labels = [], [], []
    def constraint(vector, limit, name, zcoef=0):
        row = np.zeros(dim)
        row[:space.dim] = vector
        if minimax:
            row[-1] = zcoef
        rowlist.append(row)
        rhs.append(float(limit))
        labels.append(name)
    stats = space.whole
    constraint(-stats["n"] / len(space.frame), -space.minimum / len(space.frame), "minimum_approvals")
    constraint(stats["amount"] / space.total_amount, space.capital, "capital")
    base_weights = None
    if baseline and baseline.get("feasible"):
        base_weights = np.concatenate([np.asarray(baseline["mixture"][str(g)]["weights"]) for g in (0, 1)])
    policy = {"policy_id": str(spec.get("name", mode)), "name": str(spec.get("name", mode)), "family": "protected_lp",
              "mode": mode, "protection": protection, "minimax_scope": scope, "robust": robust,
              "epsilon": epsilon, "beta": beta, "noharm_delta": delta, "delta": delta,
              "is_lp_profit_baseline": mode == "profit",
              "loss_ratio": space.loss, "risk_budget": space.risk, "capital_fraction": space.capital,
              "baseline_policy_id": baseline["policy_id"] if baseline else None,
              "utility_basis": "unit_utility", "objective": "minimax_frr_then_profit" if minimax else "unit_utility",
              "status": "INFEASIBLE", "feasible": False, "mixture": None,
              "validation_metrics": None, "decision_semantics": "stochastic_expectation"}
    if (protected or minimax) and base_weights is None:
        numerical=baseline is not None and baseline.get("status")=="SOLVER_ERROR"
        return {**policy, "status":"SOLVER_ERROR" if numerical else "INFEASIBLE",
                "reason": "Matching LP profit baseline failed numerically" if numerical else "Matching LP profit baseline is infeasible"}
    if minimax:
        base_profit = float(stats["utility"] @ base_weights)
        if base_profit <= 0:
            return {**policy, "reason": "Relative profit floor requires positive baseline scenario profit"}
        floor = (1-beta) * base_profit
        constraint(-stats["utility"] / len(space.frame), -floor / len(space.frame), "profit_floor")
        policy["validation_profit_floor"] = floor
    scenarios = [("whole", stats)]
    if robust:
        scenarios += [(f"half_{half}", values) for half, values in enumerate(space.halves)]
    for scenario_name, scenario in scenarios:
        denom = max(1, scenario["records"])
        constraint((scenario["bad"] - space.risk * scenario["n"]) / denom, 0, scenario_name+"_risk")
        rates = scenario["good_rates"]
        if epsilon is not None:
            if not {"macro_0", "macro_1"}.issubset(rates):
                return {**policy, "reason": f"Insufficient normal-label support in {scenario_name}"}
            difference = rates["macro_0"] - rates["macro_1"]
            constraint(difference, epsilon, scenario_name+"_gap_plus")
            constraint(-difference, epsilon, scenario_name+"_gap_minus")
        if protected:
            keys = ["macro_0", "macro_1"]
            if protection == "six":
                keys += [f"subgroup_{g}_{b}" for g,b in CELLS]
            for key in keys:
                if key not in rates:
                    return {**policy, "reason": f"Missing {key} normal support in {scenario_name}"}
                cap = _reference_value(scenario, key, base_weights) + delta
                constraint(-rates[key], cap-1, scenario_name+"_noharm_"+key)
    if minimax:
        keys = ["macro_0", "macro_1"] if scope == "macro" else [f"subgroup_{g}_{b}" for g,b in CELLS]
        for key in keys:
            constraint(-stats["good_rates"][key], -1, "minimax_"+key, zcoef=-1)
    aeq = np.zeros((2, dim))
    for g in (0, 1):
        aeq[g, g*space.k:(g+1)*space.k] = 1
    objective = np.zeros(dim)
    if minimax:
        objective[-1] = 1
    else:
        objective[:space.dim] = -stats["utility"] / len(space.frame)
    aub, bub = np.asarray(rowlist), np.asarray(rhs)
    bounds = [(0., 1.)] * dim
    solution = _linear_solve(objective, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=np.ones(2), bounds=bounds)
    policy.update(solver_status=int(solution.status), solver_message=solution.message,
                  solver_attempts=solution.solver_attempts,
                  constraint_count=len(rhs), constraint_names=labels)
    if not solution.success:
        return {**policy, "status":"INFEASIBLE" if solution.status==2 else "SOLVER_ERROR", "reason": solution.message}
    if minimax:
        zstar = float(solution.x[-1])
        objective[:space.dim] = -stats["utility"] / len(space.frame)
        objective[-1] = 0
        extra = np.zeros(dim)
        extra[-1] = 1
        refined = _linear_solve(objective, A_ub=np.vstack([aub, extra]), b_ub=np.r_[bub,zstar+1e-9],
                          A_eq=aeq, b_eq=np.ones(2), bounds=bounds)
        if not refined.success:
            return {**policy, "reason": "Lexicographic refinement failed: "+refined.message,
                    "status":"SOLVER_ERROR", "solver_status": int(refined.status),
                    "refinement_solver_attempts":refined.solver_attempts}
        solution = refined
        policy["refinement_solver_attempts"]=refined.solver_attempts
        policy["validation_minimax_optimum"] = zstar
    weights = solution.x[:space.dim].reshape(2, space.k).copy()
    # Normalize only solver roundoff; retain all knots for baseline comparisons.
    weights[abs(weights) < 1e-12] = 0
    weights = np.maximum(weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    x = np.r_[weights.ravel(), solution.x[-1]] if minimax else weights.ravel()
    if np.max(aub @ x - bub) > TOL or np.max(abs(aeq @ x - 1)) > TOL:
        return {**policy, "status":"SOLVER_ERROR", "reason": "Post-normalization LP feasibility check failed", "solver_status": 4}
    policy.update(status="PASS", feasible=True,
                  mixture={str(g): {"thresholds": space.thresholds.tolist(), "weights": weights[g].tolist()} for g in (0,1)},
                  active_mixture={str(g): {"thresholds": space.thresholds[weights[g]>0].tolist(), "weights": weights[g][weights[g]>0].tolist()} for g in (0,1)},
                  nonzero_thresholds=[int(np.count_nonzero(weights[g])) for g in (0,1)],
                  validation_max_constraint_violation=float(max(0, np.max(aub @ x-bub))))
    policy["validation_metrics"] = protected_metrics(space.frame, apply_protected_policy(space.frame, policy), space.loss, space.config)
    policy["validation_objective"] = (policy["validation_metrics"]["worst_macro_frr" if scope=="macro" else "worst_subgroup_frr"] if minimax
                                      else policy["validation_metrics"]["utility"])
    return policy


def fit_protected_policies(validation: pd.DataFrame, config: dict) -> dict:
    """Fit one common LP profit reference and a finite, explicit protocol."""
    space = _Space(validation, config)
    specs = config.get("policy_specs", [
        {"name": "lp_profit_gap", "mode": "profit_gap"},
        {"name": "lp_profit_gap_protected", "mode": "profit_gap_protected"},
        {"name": "lp_budgeted_minimax", "mode": "budgeted_minimax"},
    ])
    names = [str(s.get("name", s.get("mode", "budgeted_minimax"))) for s in specs]
    if len(set(names)) != len(names) or "lp_profit_baseline" in names:
        raise ValueError("Policy names must be unique and not lp_profit_baseline")
    baseline = _fit_one(space, {"name": "lp_profit_baseline", "mode": "profit"})
    policies = [baseline] + [_fit_one(space, spec, baseline) for spec in specs]
    ids = validation.SK_ID_CURR.astype(str).sort_values().tolist()
    metadata = {"version": "protected-lp-1.0", "fit_split": "validation", "objective_data": "observed_validation_labels",
                "records": len(validation), "validation_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "threshold_grid": space.thresholds.tolist(), "grid_requested": int(config.get("grid",257)),
                "grid_actual": space.k, "minimum_approvals": space.minimum,
                "minimum_subgroup_normal": space.minimum_normal,
                "candidate_space": "Separate mixtures over a shared threshold grid for thick/thin groups",
                "approval_features": ["probability", "thin"], "audit_only_feature": "income_band",
                "decision_semantics": "Expected metrics under stochastic monotone threshold policy; no single-realization risk guarantee",
                "robust_scenarios": "Validation whole and two predeclared SK_ID_CURR parity halves",
                "economic_interpretation": "Scenario net contribution using repayment-difficulty labels; not observed bank profit",
                "solver": "scipy.optimize.linprog(method=highs)", "policy_count": len(policies),
                "feasible_count": sum(p["feasible"] for p in policies)}
    return {"metadata": metadata, "policies": policies}


def evaluate_protected_policies(test: pd.DataFrame, frozen: dict, config: dict) -> list[dict]:
    """Replay frozen mixtures and expose test violations without refitting."""
    _arrays(test)
    _band(test)
    policies = frozen["policies"]
    cache = {p["policy_id"]: protected_metrics(test, apply_protected_policy(test,p), p["loss_ratio"],config)
             for p in policies if p.get("feasible")}
    rows = []
    for p in policies:
        row = {k: deepcopy(v) for k,v in p.items() if k not in ("validation_metrics", "mixture", "constraint_names")}
        row["evaluation_split"] = "test"
        if not p.get("feasible"):
            rows.append(row)
            continue
        m = cache[p["policy_id"]]
        row.update(m)
        base = cache.get(p["baseline_policy_id"])
        row.update(test_risk_constraint_met=m["risk"] is not None and m["risk"] <= p["risk_budget"]+TOL,
                   test_capital_constraint_met=m["capital_fraction_used"] <= p["capital_fraction"]+TOL,
                   test_gap_constraint_met=None if p["epsilon"] is None else m["abs_frr_gap"] <= p["epsilon"]+TOL,
                   test_noharm_constraint_met=None, test_profit_floor_met=None)
        if p["noharm_delta"] is not None:
            keys = ["thick_frr", "thin_frr"]
            if p["protection"] == "six":
                keys += [f"subgroup_{g}_{b}_frr" for g,b in CELLS]
            row["test_noharm_constraint_met"] = bool(base and all(m[k] is not None and base[k] is not None and m[k] <= base[k]+p["noharm_delta"]+TOL for k in keys))
        if p["beta"] is not None:
            row["test_profit_floor_met"] = bool(base and m["utility"] >= (1-p["beta"])*base["utility"]-TOL)
        if base:
            row["utility_delta_vs_lp_baseline"] = m["utility"]-base["utility"]
            row["profit_cost_fraction_vs_lp_baseline"] = 1-m["utility"]/base["utility"] if base["utility"]>0 else None
            row["worst_subgroup_frr_delta_vs_lp_baseline"] = m["worst_subgroup_frr"]-base["worst_subgroup_frr"] if m["worst_subgroup_frr"] is not None and base["worst_subgroup_frr"] is not None else None
        rows.append(row)
    return rows
