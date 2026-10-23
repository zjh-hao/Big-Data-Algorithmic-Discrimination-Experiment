"""Opportunity allocation with an explicit, predeclared extra harm budget.

``tau`` is an economic parameter: the permitted increase beyond the smallest
possible worst-group deterioration. It is distinct from solver roundoff.
This module leaves the exact-face experiment and implementation unchanged.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np
from scipy.optimize import OptimizeResult

from ..protected_policy import _Space, _fit_one, apply_protected_policy, protected_metrics
from .allocation_policy import (
    FACE_TOL, RANGE_TOL, LEX_TOL, KEYS, _base_program, _ranges, _allocate, _failure,
)


CERTIFICATE_TOL = 1e-6


def _certify_stages(stages, values):
    """Check the reported lex face in its actual normalized coordinate units.

    Tiny original rate residuals do not certify normalized lexicographic
    decisions when their attainable ranges are tiny.
    """
    values=np.asarray(values,float)
    monotonic=lower=freezing=forced=0.
    previous=None
    for stage in stages:
        level=float(stage["level"])
        if previous is not None:
            monotonic=max(monotonic,previous-level)
        previous=level
        for i in stage["remaining"]:
            lower=max(lower,level-float(values[i]))
        for i in stage["frozen"]:
            freezing=max(freezing,abs(float(values[i])-level))
            maximum=stage["coordinate_maxima_on_optimal_face"].get(str(i))
            if maximum is None:
                forced=float("inf")
            else:
                forced=max(forced,float(maximum)-level)
    maximum=max(0.,monotonic,lower,freezing,forced)
    return {"stage_level_monotonicity_violation":max(0.,monotonic),
            "stage_lower_bound_violation":max(0.,lower),
            "stage_frozen_value_violation":max(0.,freezing),
            "stage_forced_coordinate_violation":max(0.,forced),
            "max_normalized_stage_violation":maximum,
            "normalized_certificate_tolerance":CERTIFICATE_TOL,
            "normalized_stage_certificate_pass":bool(np.isfinite(maximum) and maximum<=CERTIFICATE_TOL)}


def _budget_face(program, rates, baseline_rates, minimum_required_delta, tau):
    """Return the unchanged business domain intersected with a harm budget."""
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("additional_harm_budget tau must be finite and in [0,1]")
    result = program.copy()
    result.attempts = deepcopy(program.attempts)
    cap = minimum_required_delta + tau + FACE_TOL
    for row, value in zip(rates, baseline_rates):
        result.upper(-row, -value+cap)
    return result


def _templates(space, beta, tau, epsilon, modes):
    return [{"name": f"budget_allocation_{mode}_b{beta:g}_tau{tau:g}",
             "policy_id": f"budget_allocation_{mode}_b{beta:g}_tau{tau:g}",
             "family": "dual_budget_opportunity_allocation_lp", "mode": mode,
             "protection": "six", "minimax_scope": "six", "robust": False,
             "epsilon": epsilon, "beta": beta, "tau": tau,
             "additional_harm_budget": tau, "noharm_delta": None, "delta": None,
             "minimum_required_delta": None, "total_harm_cap": None,
             "baseline_policy_id": "lp_profit_baseline", "is_lp_profit_baseline": False,
             "loss_ratio": space.loss, "risk_budget": space.risk,
             "capital_fraction": space.capital, "utility_basis": "unit_utility",
             "objective": mode+"_then_profit", "decision_semantics": "stochastic_expectation",
             "face_numerical_tolerance": FACE_TOL, "range_fixed_tolerance": RANGE_TOL,
             "leximin_identification_tolerance": LEX_TOL}
            for mode in modes]


def fit_allocation_policies(validation, config):
    """Fit a common LP baseline and explicit finite beta/tau allocation rules.

    Config keys: ``betas`` (default [.01]), ``additional_harm_budgets``
    (default [.001]), and ``allocation_modes`` (the same three as run 005).
    Decisions still use probability and thin only; no outcome is read at replay.
    """
    cfg={"grid":256, **config}
    betas=[float(v) for v in cfg.get("betas", [cfg.get("primary_beta", .01)])]
    taus=[float(v) for v in cfg.get("additional_harm_budgets", [cfg.get("additional_harm_budget", .001)])]
    epsilon=float(cfg.get("primary_epsilon", .02))
    modes=cfg.get("allocation_modes", ["absolute_maxmin", "normalized_maxmin", "normalized_leximin"])
    for name, values in (("betas",betas),("additional_harm_budgets",taus)):
        if not values or len(values)!=len(set(values)) or any(not np.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError(name+" must contain distinct finite values in [0,1]")
    if not np.isfinite(epsilon) or not 0<=epsilon<=1:
        raise ValueError("primary_epsilon must lie in [0,1]")
    if len(modes)!=len(set(modes)) or any(m not in ("absolute_maxmin","normalized_maxmin","normalized_leximin") for m in modes):
        raise ValueError("allocation_modes contains duplicate or unknown modes")
    space=_Space(validation,cfg)
    baseline=_fit_one(space,{"name":"lp_profit_baseline","mode":"profit"})
    baseline.update(tau=None,additional_harm_budget=None,minimum_required_delta=None,total_harm_cap=None)
    policies=[baseline]
    rates=np.asarray([space.whole["good_rates"][key] for key in KEYS])
    for beta in betas:
        all_templates=[p for tau in taus for p in _templates(space,beta,tau,epsilon,modes)]
        if not baseline.get("feasible") or baseline["validation_metrics"]["utility"]<=0:
            for p in all_templates:
                policies.append({**p,"status":baseline["status"] if not baseline.get("feasible") else "INFEASIBLE",
                                 "feasible":False,"mixture":None,"validation_metrics":None,
                                 "reason":"A feasible positive-profit LP baseline is required"})
            continue
        business,bw,base_profit=_base_program(space,baseline,beta,epsilon)
        base_rates=rates@bw
        first=business.copy()
        old_count=len(first.bub)
        for row,value in zip(rates,base_rates):
            first.upper(-row,-value)
        stage1=first.solve(np.r_[np.zeros(space.dim),1.],"minimum_deterioration",
                           extra_column=np.r_[np.zeros(old_count),-np.ones(len(rates))])
        business.attempts.extend(first.attempts)
        if not stage1.success:
            policies.extend(_failure(p,stage1,business) for p in all_templates)
            continue
        dstar=max(0.,float(stage1.x[-1]))
        marginal=float(stage1.ineqlin.marginals[3])
        sensitivity={"normalized_profit_floor_rhs_marginal":marginal,"profit_floor_multiplier":-marginal,
                     "local_dstar_slope_wrt_beta":marginal*base_profit/len(validation),
                     "interpretation":"One LP dual sensitivity/subgradient; local linear slope only while the optimal basis remains valid"}
        for tau in taus:
            program=_budget_face(business,rates,base_rates,dstar,tau)
            effective=dstar+tau+FACE_TOL
            lower,upper,failed=_ranges(program,rates)
            diagnostics=None if failed is not None else [
                {"group":key,"index":i,"minimum_acceptance":float(lower[i]),
                 "maximum_acceptance":float(upper[i]),"range":float(upper[i]-lower[i]),
                 "fixed":bool(upper[i]-lower[i]<=RANGE_TOL),"baseline_acceptance":float(base_rates[i])}
                for i,key in enumerate(KEYS)]
            shared={"minimum_required_delta":dstar,"effective_delta":effective,
                    "total_harm_cap":effective,"economic_total_harm_cap":dstar+tau,
                    "minimum_deterioration_dual":sensitivity,"noharm_delta":effective,"delta":effective,
                    "validation_profit_floor":(1-beta)*base_profit,"validation_baseline_profit":base_profit,
                    "range_diagnostics":diagnostics,
                    "fixed_groups":[] if diagnostics is None else [r["group"] for r in diagnostics if r["fixed"]]}
            templates=_templates(space,beta,tau,epsilon,modes)
            if failed is not None:
                policies.extend(_failure({**p,**shared},failed,program,known_feasible=True) for p in templates)
                continue
            for p in templates:
                work=program.copy()
                work.attempts=deepcopy(program.attempts)
                result,stages,fixed=_allocate(work,rates,lower,upper,
                                              space.whole["utility"]/len(validation),mode=p["mode"])
                policy={**p,**shared,"leximin_stages":stages,"leximin_fixed_coordinates":fixed}
                if not result.success:
                    policies.append(_failure(policy,result,work,known_feasible=True))
                    continue
                weights=np.maximum(np.asarray(result.x[:space.dim]),0).reshape(2,space.k)
                weights/=weights.sum(axis=1,keepdims=True)
                w=weights.ravel()
                violation=max(0.,float(np.max(program.aub@w-program.bub)),float(np.max(np.abs(program.aeq@w-program.beq))))
                if violation>2e-8:
                    err=OptimizeResult(success=False,status=4,message="Final policy violates the explicit beta/tau business domain")
                    policies.append(_failure(policy,err,work))
                    continue
                accepts=rates@w
                active=upper-lower>RANGE_TOL
                normalized=[float((accepts[i]-lower[i])/(upper[i]-lower[i])) if active[i] else None for i in range(len(rates))]
                policy.update(status="PASS",feasible=True,solver_status=int(result.status),solver_attempts=deepcopy(work.attempts),
                              mixture={str(g):{"thresholds":space.thresholds.tolist(),"weights":weights[g].tolist()} for g in (0,1)},
                              active_mixture={str(g):{"thresholds":space.thresholds[weights[g]>1e-12].tolist(),
                                                       "weights":weights[g][weights[g]>1e-12].tolist()} for g in (0,1)},
                              nonzero_thresholds=[int(np.count_nonzero(weights[g]>1e-12)) for g in (0,1)],
                              validation_max_constraint_violation=violation,
                              validation_subgroup_acceptance=accepts.tolist(),
                              validation_normalized_opportunity=normalized,
                              validation_sorted_normalized_opportunity=sorted(v for v in normalized if v is not None),
                              validation_max_deterioration=float(max(0.,np.max(base_rates-accepts))))
                policy["validation_metrics"]=protected_metrics(validation,apply_protected_policy(validation,policy),space.loss,cfg)
                if p["mode"] in ("normalized_maxmin","normalized_leximin"):
                    certificate=_certify_stages(stages,[v if v is not None else 0. for v in normalized])
                    policy.update(certificate)
                    if not certificate["normalized_stage_certificate_pass"]:
                        policy.update(status="NUMERICAL_DEGENERATE",feasible=False,
                                      reason="Normalized stage residual exceeds the declared numerical certificate tolerance",
                                      candidate_mixture=policy.pop("mixture"),
                                      candidate_validation_metrics=policy.pop("validation_metrics"),
                                      mixture=None,validation_metrics=None)
                policies.append(policy)
    ids=validation.SK_ID_CURR.astype(str).sort_values().tolist()
    return {"metadata":{"version":"dual-budget-opportunity-allocation-lp-1.0","fit_split":"validation",
                        "records":len(validation),"validation_id_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                        "threshold_grid":space.thresholds.tolist(),"grid_requested":cfg["grid"],"grid_actual":space.k,
                        "betas":betas,"additional_harm_budgets":taus,"tau_values":taus,
                        "primary_tau":float(cfg.get("primary_tau",.001)),"modes":list(modes),"group_order":KEYS,
                        "objective_data":"observed_validation_labels","approval_features":["probability","thin"],
                        "audit_only_feature":"income_band","decision_semantics":"stochastic_expectation",
                        "tau_semantics":"Predeclared additional worst-subgroup harm budget beyond minimum required deterioration, in absolute probability units",
                        "face_numerical_tolerance":FACE_TOL,
                        "normalized_certificate_tolerance":CERTIFICATE_TOL,
                        "range_definition":"Each coordinate minimum and maximum on the SAME explicit beta/tau feasible set",
                        "policy_count":len(policies),"feasible_count":sum(p.get("feasible",False) for p in policies)},
            "policies":policies}
