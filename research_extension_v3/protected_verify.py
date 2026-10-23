"""Independent arithmetic audit of the finite protected-approval experiment.

This verifier deliberately imports neither business.py nor protected_policy.py.
It reconstructs approval probabilities directly from serialized threshold
mixtures, audits their validation constraints, and treats test breaches as
out-of-sample observations. It checks feasibility, never certifies LP optimality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TOL = 5e-7
CELLS = [(g, b) for g in (0, 1) for b in (0, 1, 2)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def numeric_equal(a, b):
    if a is None or pd.isna(a):
        return b is None or pd.isna(b)
    if b is None or pd.isna(b):
        return False
    return bool(np.isclose(float(a), float(b), rtol=2e-10, atol=2e-6))


def replay(frame, policy, deterministic=False):
    """Use only risk scores and macro group; labels/income are not accessed."""
    p = frame["probability"].to_numpy(float)
    g = frame["thin"].to_numpy(int)
    if deterministic:
        thresholds = np.asarray(policy["thresholds"], float)
        if thresholds.shape != (2,) or not np.isfinite(thresholds).all():
            raise ValueError("Invalid deterministic threshold pair")
        return (p <= thresholds[g]).astype(float)
    q = np.zeros(len(frame), float)
    if set(policy["mixture"]) != {"0", "1"}:
        raise ValueError("Mixture must have exactly groups 0/1")
    for group in (0, 1):
        mix = policy["mixture"][str(group)]
        knots, weights = np.asarray(mix["thresholds"], float), np.asarray(mix["weights"], float)
        if (knots.ndim != 1 or len(knots) == 0 or weights.shape != knots.shape
                or not np.isfinite(knots).all() or not np.isfinite(weights).all()
                or np.any(np.diff(knots) < 0) or np.any(weights < -TOL)
                or abs(weights.sum()-1) > TOL):
            raise ValueError(f"Invalid mixture weights/grid for group {group}")
        take = g == group
        scores = p[take]
        result = np.zeros(len(scores))
        # Direct weighted indicator sums independently reproduce the optimized
        # implementation's tail-sum/searchsorted evaluation.
        for threshold, weight in zip(knots, weights):
            if weight != 0:
                result += weight * (scores <= threshold)
        q[take] = result
    if np.any(q < -TOL) or np.any(q > 1+TOL):
        raise ValueError("Replayed probabilities outside [0,1]")
    return np.clip(q, 0, 1)


def independent_metrics(frame, q, cfg):
    y = frame["TARGET"].to_numpy(float)
    g = frame["thin"].to_numpy(int)
    income = frame["income_band"].to_numpy(int)
    amount = frame["AMT_CREDIT"].to_numpy(float)
    p = frame["probability"].to_numpy(float)
    lam = float(cfg["primary_loss_ratio"])
    interest = float(cfg.get("scenario_interest", .12))
    cost = float(cfg.get("scenario_cost", .03))
    loss = lam*(interest-cost)-cost
    n, bad, good = float(np.sum(q)), float(np.sum(q*y)), float(np.sum(q*(1-y)))
    funded = float(np.sum(q*amount))
    utility = float(np.sum(q*np.where(y == 0, 1., -lam)))
    expected_utility = float(np.sum(q*(1-(1+lam)*p)))
    amount_utility = float(np.sum(q*amount*np.where(y == 0, interest-cost, -loss-cost)))
    expected_amount = float(np.sum(q*amount*((1-p)*interest-p*loss-cost)))
    m = dict(records=len(frame), approvals=n, approval_rate=n/len(frame),
             good_approved=good, difficulty_count=bad, risk=bad/n if n else None,
             expected_risk=float(np.sum(q*p))/n if n else None,
             approved_amount=funded, capital_fraction_used=funded/float(np.sum(amount)),
             amount_weighted_risk=float(np.sum(q*amount*y))/funded if funded else None,
             utility=utility, expected_utility=expected_utility,
             utility_per_1000_applicants=utility*1000/len(frame),
             amount_utility=amount_utility, expected_amount_utility=expected_amount,
             amount_utility_per_1000_applicants=amount_utility*1000/len(frame),
             net_contribution_per_funded_unit=amount_utility/funded if funded else None,
             loss_ratio=lam, scenario_interest=interest, scenario_cost=cost,
             scenario_difficulty_loss=loss,
             fractional_records=int(((q > 1e-9)&(q < 1-1e-9)).sum()))
    frrs = []
    for group, name in ((0, "thick"), (1, "thin")):
        mask = g == group
        normal = mask&(y == 0)
        count = int(normal.sum())
        approved = float(np.sum(q[mask]))
        group_bad = float(np.sum(q[mask]*y[mask]))
        group_good = float(np.sum(q[normal]))
        m.update({f"{name}_records": int(mask.sum()), f"{name}_normal_records": count,
                  f"{name}_approvals": approved, f"{name}_good_approved": group_good,
                  f"{name}_difficulty_approved": group_bad,
                  f"{name}_risk": group_bad/approved if approved else None,
                  f"{name}_frr": 1-group_good/count if count else None})
        for band in (0, 1, 2):
            cell = normal&(income == band)
            frr = 1-float(np.sum(q[cell]))/int(cell.sum()) if cell.any() else None
            m[f"g{group}_income{band}_frr"] = frr
            m[f"g{group}_income{band}_normal_records"] = int(cell.sum())
            frrs.append(frr)
    m["abs_frr_gap"] = abs(m["thin_frr"]-m["thick_frr"]) if None not in (m["thin_frr"], m["thick_frr"]) else None
    available = [v for v in frrs if v is not None]
    m["worst_subgroup_frr"] = max(available) if available else None
    m["six_group_frr_range"] = max(available)-min(available) if available else None
    return m


def constraint_results(m, base, policy, cfg, half=False):
    """Yield actual constraints only; reporting flags can describe other goals."""
    yield "risk", m["risk"], float(cfg["primary_risk_budget"]), "le"
    if not half:
        yield "capital", m["capital_fraction_used"], float(cfg["capital_fraction"]), "le"
        yield "minimum_approvals", m["approvals"], int(cfg["minimum_approvals"]), "ge"
    if policy.get("epsilon") is not None:
        yield "gap", m["abs_frr_gap"], float(policy["epsilon"]), "le"
    delta = policy.get("delta", policy.get("noharm_delta"))
    if delta is not None:
        for group in ("thin", "thick"):
            yield f"protect_{group}", m[f"{group}_frr"]-base[f"{group}_frr"], float(delta), "le"
        if policy.get("protection") == "six":
            for g, b in CELLS:
                key = f"g{g}_income{b}_frr"
                yield f"protect_g{g}_income{b}", m[key]-base[key], float(delta), "le"
    if not half and policy.get("beta") is not None:
        yield "profit_floor", m["utility"], (1-float(policy["beta"]))*base["utility"], "ge"


def deterministic_from_source(source, job):
    content = read(source/"policies"/f"{job}.json")
    result = {}
    names = {"group_profit": "det_profit", "group_profit_gap": "det_gap",
             "group_profit_gap_noharm": "det_gap_macro_protected"}
    for p in content["policies"]:
        if p.get("family") not in names or p.get("utility_basis") not in (None, "unit", "unit_utility"):
            continue
        if p.get("risk_budget") != .05 or p.get("loss_ratio") != 5 or p.get("capital_fraction") != 1:
            continue
        if p["family"] != "group_profit" and p.get("epsilon") != .02:
            continue
        if p["family"] == "group_profit_gap_noharm" and p.get("noharm_delta") != .01:
            continue
        result[names[p["family"]]] = p
    return result


def verify(run):
    run = Path(run).resolve()
    audit, test_observations, metrics_rows, failures = [], [], [], []
    def check(name, passed, job=None, policy=None, split=None, detail=None):
        row = dict(check=name, passed=bool(passed), job=job, policy=policy, split=split, detail=detail)
        audit.append(row)
        if not passed:
            failures.append(row)
    plan_path = run/"experiments_config.json"
    plan = read(plan_path)
    cfg, jobs = plan["config"], plan["jobs"]
    expected_names = [j["job"] for j in jobs]
    check("finite_job_count_24", len(jobs) == 24 and len(set(expected_names)) == 24)
    check("expected_job_manifest", expected_names == plan["report_expected_jobs"])
    plan_cfg = read(run/"config_frozen.json")
    check("frozen_config_matches_plan", all(plan_cfg.get(k) == v for k,v in cfg.items()))
    aggregate = pd.read_csv(run/"policy_metrics.csv") if (run/"policy_metrics.csv").exists() else pd.DataFrame()
    check("aggregate_metrics_present", not aggregate.empty)
    completed = 0
    feasible_count = infeasible_count = 0
    numerical_error_count = 0
    expected_rows = 0
    population = {}
    split_manifests = {}
    for job_cfg in jobs:
        job, source = job_cfg["job"], Path(job_cfg["source"])
        job_marker = run/"jobs"/f"{job}.json"
        expected_specs = job_cfg.get("policy_specs", cfg["policy_specs"] if job_cfg["experiment"] == "main" else cfg["policy_specs"][:5])
        expected_policy_names = ["lp_profit_baseline"]+[s["name"] for s in expected_specs]+[
            "det_profit", "det_gap", "det_gap_macro_protected"]
        expected_rows += len(expected_policy_names)
        if not job_marker.exists():
            check("completed_job_exists", False, job, detail=str(job_marker))
            continue
        try:
            marker = read(job_marker)
            check("job_status_pass", marker.get("status") == "PASS", job)
            check("job_manifest_identity", all(marker.get(k) == v for k,v in job_cfg.items()), job)
            frozen_path = run/"policies"/f"{job}.json"
            frozen = read(frozen_path)
            check("frozen_policy_hash", marker.get("frozen_sha256") == digest(frozen_path), job)
            check("validation_only_fit", frozen["metadata"].get("fit_split") == "validation", job)
            check("approval_features_only_score_group", frozen["metadata"].get("approval_features") == ["probability", "thin"], job)
            source_marker = read(source/"jobs"/f"{job}.json")
            check("source_job_pass", source_marker.get("status") == "PASS", job)
            manifest_path = source/"split_manifest.parquet"
            if not manifest_path.exists():
                manifest_path = source.parent/"split_manifest.parquet"
            if str(manifest_path) not in split_manifests:
                manifest = pd.read_parquet(manifest_path)
                check("all_partition_ids_unique",manifest.SK_ID_CURR.is_unique,job,detail=str(manifest_path))
                check("all_expected_partitions",set(manifest.split) == {"train","reference","calibration","validation","test"},job)
                split_manifests[str(manifest_path)] = manifest
            manifest = split_manifests[str(manifest_path)]
            frames, decision_frames = {}, {}
            for split in ("validation", "test"):
                path = source/"predictions"/f"{job}_{split}.parquet"
                current_sha = digest(path)
                check("source_prediction_frozen_hash", marker.get("source_prediction_sha256",{}).get(split) == current_sha, job, split=split)
                registered = source_marker.get("prediction_sha256",{}).get(split)
                if registered:
                    check("original_source_prediction_hash", registered == current_sha, job, split=split)
                frame = pd.read_parquet(path)
                ids = frame.SK_ID_CURR.to_numpy()
                check("unique_source_ids", frame.SK_ID_CURR.notna().all() and frame.SK_ID_CURR.is_unique, job, split=split)
                registered_split = manifest[manifest.split.eq(split)][["SK_ID_CURR","TARGET","thin"]].sort_values("SK_ID_CURR").reset_index(drop=True)
                actual_split = frame[["SK_ID_CURR","TARGET","thin"]].sort_values("SK_ID_CURR").reset_index(drop=True)
                check("source_partition_labels_and_ids",registered_split.equals(actual_split),job,split=split)
                check("valid_source_values", np.isin(frame.TARGET,[0,1]).all() and np.isin(frame.thin,[0,1]).all()
                      and np.isin(frame.income_band,[0,1,2]).all() and np.isfinite(frame.probability).all()
                      and frame.probability.between(0,1).all() and np.isfinite(frame.AMT_CREDIT).all()
                      and (frame.AMT_CREDIT > 0).all(), job, split=split)
                fixed_cols = ["SK_ID_CURR", "TARGET", "thin", "income_band", "AMT_CREDIT"]
                if split not in population:
                    population[split] = frame[fixed_cols].sort_values("SK_ID_CURR").reset_index(drop=True)
                else:
                    check("shared_evaluation_population", population[split].equals(frame[fixed_cols].sort_values("SK_ID_CURR").reset_index(drop=True)), job, split=split)
                decision_path = run/"decisions"/f"{job}_{split}.parquet"
                decisions = pd.read_parquet(decision_path)
                check("decision_id_order", np.array_equal(ids, decisions.SK_ID_CURR.to_numpy()), job, split=split)
                frames[split], decision_frames[split] = frame, decisions
            check("validation_test_disjoint", not set(frames["validation"].SK_ID_CURR)&set(frames["test"].SK_ID_CURR), job)
            id_hash = hashlib.sha256("\n".join(frames["validation"].SK_ID_CURR.astype(str).sort_values().tolist()).encode()).hexdigest()
            check("validation_ids_registered", frozen["metadata"].get("validation_id_sha256") == id_hash, job)
            row_csv = pd.read_csv(run/"job_metrics"/f"{job}.csv")
            check("finite_policy_rows", len(row_csv) == len(expected_policy_names) and row_csv.policy.is_unique
                  and set(row_csv.policy) == set(expected_policy_names), job)
            subset = aggregate[aggregate.job.eq(job)] if not aggregate.empty else pd.DataFrame()
            check("aggregate_job_rows", len(subset) == len(row_csv), job)
            if len(subset) == len(row_csv):
                a = subset.sort_values("policy").reset_index(drop=True)
                b = row_csv.sort_values("policy").reset_index(drop=True)
                same = list(a.columns) == list(b.columns)
                if same:
                    for column in a:
                        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
                            same &= bool(np.allclose(a[column],b[column],rtol=1e-10,atol=1e-7,equal_nan=True))
                        else:
                            same &= bool(a[column].fillna("__NA__").astype(str).equals(b[column].fillna("__NA__").astype(str)))
                check("aggregate_matches_job_metrics", same, job)
            policies = {p["name"]: p for p in frozen["policies"]}
            check("frozen_lp_names", set(policies) == set(expected_policy_names[:len(expected_specs)+1]), job)
            for spec in expected_specs:
                p = policies.get(spec["name"],{})
                for key in ("mode","protection","robust","epsilon","beta","minimax_scope"):
                    applicable = key != "beta" or p.get("mode") in ("budgeted_minimax","adaptive_protected_minimax")
                    if key in spec and applicable:
                        check("frozen_specification_"+key,p.get(key) == spec[key],job,spec["name"])
                if "delta" in spec and p.get("mode") in ("profit_gap_protected","budgeted_minimax"):
                    check("frozen_specification_delta",p.get("delta") == spec["delta"],job,spec["name"])
            deterministic = deterministic_from_source(source,job)
            check("three_source_deterministic_policies", len(deterministic) == 3, job)
            policies.update(deterministic)
            baseline_policy = policies["lp_profit_baseline"]
            if not baseline_policy.get("feasible"):
                raise ValueError("LP baseline infeasible; cannot audit relative policies")
            base_q = {s: replay(f,baseline_policy) for s,f in frames.items()}
            base_metrics = {s: independent_metrics(f,base_q[s],cfg) for s,f in frames.items()}
            for name,p in policies.items():
                deterministic = name.startswith("det_")
                rows = row_csv[row_csv.policy.eq(name)]
                if len(rows) != 1:
                    check("one_metric_row_per_policy",False,job,name)
                    continue
                saved = rows.iloc[0]
                check("feasibility_status_matches_frozen", bool(saved.feasible) == bool(p.get("feasible")),job,name)
                if not p.get("feasible"):
                    if p.get("status") == "SOLVER_ERROR":
                        numerical_error_count += 1
                        check("unresolved_numerical_solver_error",False,job,name,detail=p.get("reason"))
                    else:
                        infeasible_count += 1
                        check("unsolved_status_explicitly_infeasible",p.get("status") == "INFEASIBLE",job,name)
                    check("infeasible_has_no_decisions", all(name not in d.columns for d in decision_frames.values()),job,name)
                    check("infeasible_has_no_reported_metrics", pd.isna(saved.get("test_utility")) and pd.isna(saved.get("validation_utility")),job,name)
                    continue
                feasible_count += 1
                check("feasible_policy_success_status",p.get("status") == "PASS",job,name)
                if not deterministic:
                    check("successful_lp_solver_status",p.get("solver_status") == 0,job,name)
                    check("common_lp_baseline_reference",name == "lp_profit_baseline" or p.get("baseline_policy_id") == "lp_profit_baseline",job,name)
                    check("mixture_shared_grid", all(p["mixture"][str(g)]["thresholds"] == frozen["metadata"]["threshold_grid"] for g in (0,1)),job,name)
                    if p.get("mode") == "adaptive_protected_minimax":
                        check("successful_adaptation_solver_status",p.get("adaptation_solver_status") == 0,job,name)
                        required_delta = p.get("minimum_required_delta")
                        slack = p.get("deterioration_slack")
                        requested = p.get("requested_delta")
                        effective = p.get("effective_delta")
                        valid_values = all(v is not None and np.isfinite(v) for v in (required_delta,slack,requested,effective))
                        check("adaptive_delta_metadata",valid_values and 0 <= required_delta <= 1 and slack >= 0
                              and numeric_equal(effective,min(1.,required_delta+slack))
                              and numeric_equal(effective,p.get("delta")) and numeric_equal(effective,p.get("noharm_delta")),job,name)
                    if p.get("active_mixture"):
                        for g in (0,1):
                            entry = p["mixture"][str(g)]
                            knots,weights = np.asarray(entry["thresholds"]),np.asarray(entry["weights"])
                            active = p["active_mixture"][str(g)]
                            check("active_mixture_serialization",np.array_equal(np.asarray(active["thresholds"]),knots[weights>0])
                                  and np.array_equal(np.asarray(active["weights"]),weights[weights>0]),job,name)
                for split, frame in frames.items():
                    # Crucially the replay input excludes both outcome and income.
                    q = replay(frame[["probability","thin"]],p,deterministic)
                    stored = decision_frames[split][name].to_numpy(float)
                    check("independent_decision_replay", np.isfinite(stored).all() and np.allclose(q,stored,rtol=0,atol=2e-9),job,name,split)
                    monotone = all(np.all(np.diff(q[(frame.thin.to_numpy()==g)][np.argsort(frame.probability.to_numpy()[frame.thin.to_numpy()==g],kind="stable")]) <= TOL) for g in (0,1))
                    check("monotone_within_macro_group",monotone,job,name,split)
                    m = independent_metrics(frame,q,cfg)
                    metrics_rows.append(dict(job=job,policy=name,split=split,**m))
                    for key,value in m.items():
                        if f"{split}_{key}" in saved:
                            check("metric_"+key,numeric_equal(value,saved[f"{split}_{key}"]),job,name,split)
                    # These descriptive flags are deliberately referenced to
                    # the LP baseline even for an older deterministic policy.
                    # They are distinct from that old policy's own constraints.
                    bm = base_metrics[split]
                    eps = p.get("epsilon") if p.get("epsilon") is not None else cfg["primary_epsilon"]
                    delta = p.get("delta") if p.get("delta") is not None else cfg["primary_delta"]
                    beta = p.get("beta") if p.get("beta") is not None else cfg["primary_beta"]
                    flags = dict(risk_met=m["risk"] <= cfg["primary_risk_budget"]+1e-7,
                                 capital_met=m["capital_fraction_used"] <= cfg["capital_fraction"]+1e-7,
                                 min_approvals_met=m["approvals"] >= cfg["minimum_approvals"]-1e-7,
                                 gap_met=m["abs_frr_gap"] <= eps+1e-7,
                                 macro_protection_met=max(m[f"{g}_frr"]-bm[f"{g}_frr"] for g in ("thin","thick")) <= delta+1e-7,
                                 six_protection_met=max(m[f"g{g}_income{b}_frr"]-bm[f"g{g}_income{b}_frr"] for g,b in CELLS) <= delta+1e-7,
                                 profit_floor_met=m["utility"] >= (1-beta)*bm["utility"]-1e-7,
                                 relative_profit_change=(m["utility"]-bm["utility"])/abs(bm["utility"]) if bm["utility"] else None)
                    for key,value in flags.items():
                        if f"{split}_{key}" in saved:
                            check("reported_"+key,numeric_equal(value,saved[f"{split}_{key}"]),job,name,split)
                    if not deterministic and split == "validation":
                        recorded = p.get("validation_metrics") or {}
                        check("frozen_validation_utility",numeric_equal(m["utility"],recorded.get("utility")),job,name,split)
                        for g,b in CELLS:
                            check(f"frozen_subgroup_{g}_{b}_frr",numeric_equal(m[f"g{g}_income{b}_frr"],recorded.get(f"subgroup_{g}_{b}_frr")),job,name,split)
                        if p.get("validation_minimax_optimum") is not None:
                            objective_value = max(m["thin_frr"],m["thick_frr"]) if p.get("minimax_scope") == "macro" else m["worst_subgroup_frr"]
                            check("minimax_objective_replay",objective_value <= p["validation_minimax_optimum"]+TOL,job,name,split)
                    if deterministic:
                        # Old deterministic policies protect their own deterministic
                        # baseline, rather than the expanded LP-space baseline.
                        reference = independent_metrics(frame,replay(frame,policies["det_profit"],True),cfg)
                    else:
                        reference = base_metrics[split]
                    scenario_frames = [("whole",frame,q,reference)]
                    if not deterministic and p.get("robust"):
                        for half in (0,1):
                            take = frame.SK_ID_CURR.to_numpy(dtype=np.int64)%2 == half
                            subframe = frame.loc[take]
                            scenario_frames.append((f"half_{half}",subframe,q[take],independent_metrics(subframe,base_q[split][take],cfg)))
                    for scenario,subframe,subq,ref in scenario_frames:
                        mm = m if scenario == "whole" else independent_metrics(subframe,subq,cfg)
                        for key,value,limit,sense in constraint_results(mm,ref,p,cfg,scenario!="whole"):
                            passed = value is not None and np.isfinite(value) and ((value <= limit+TOL) if sense=="le" else (value >= limit-TOL))
                            detail = dict(value=value,limit=limit,sense=sense,scenario=scenario)
                            if split == "validation":
                                check("validation_constraint_"+key,passed,job,name,split,detail)
                            else:
                                test_observations.append(dict(job=job,policy=name,scenario=scenario,constraint=key,
                                    met=bool(passed),value=value,limit=limit,sense=sense,
                                    interpretation="Out-of-sample observation, not a validation-fitting failure"))
                    if split == "validation" and p.get("mode") == "adaptive_protected_minimax":
                        actual_goal_met = True
                        for scenario,subframe,subq,ref in scenario_frames:
                            mm = m if scenario == "whole" else independent_metrics(subframe,subq,cfg)
                            keys = ["thin_frr","thick_frr"]+[f"g{g}_income{b}_frr" for g,b in CELLS]
                            actual_goal_met &= all(mm[key] is not None and ref[key] is not None
                                                   and mm[key] <= ref[key]+p["requested_delta"]+2e-7 for key in keys)
                        check("adaptive_requested_goal_flag",bool(p.get("validation_requested_goal_met")) == bool(actual_goal_met),job,name)
            completed += 1
        except Exception as exc:
            check("job_audit_exception",False,job,detail=f"{type(exc).__name__}: {exc}")
    check("all_24_jobs_audited",completed == 24,detail=dict(completed=completed,expected=24))
    check("aggregate_expected_policy_records",len(aggregate) == expected_rows,detail=dict(actual=len(aggregate),expected=expected_rows))
    status_path = run/"status.json"
    if status_path.exists():
        status = read(status_path)
        check("run_status_complete",status.get("status") == "COMPLETE")
        for name,value in status.get("source_sha256",{}).items():
            snapshot = run/"source_snapshot"/name
            selected = snapshot if snapshot.exists() else HERE/name
            check("frozen_runner_source_"+name,digest(selected) == value,detail=str(selected))
    else:
        check("run_status_exists",False)
    for name,rows in (("protected_verification_audit.csv",audit),("protected_test_constraint_observations.csv",test_observations),
                      ("protected_independent_metrics.csv",metrics_rows)):
        pd.DataFrame(rows).to_csv(run/name,index=False,encoding="utf-8-sig")
    observations = pd.DataFrame(test_observations)
    violations = observations[observations.met.eq(False)] if not observations.empty else observations
    summary = dict(status="PASS" if not failures else "FAIL",generated_utc=datetime.now(timezone.utc).isoformat(),
        verifier_source_sha256=digest(__file__),experiment_plan_sha256=digest(plan_path),
        expected_jobs=24,audited_jobs=completed,expected_policy_records=expected_rows,
        actual_policy_records=len(aggregate),feasible_policies=feasible_count,infeasible_policies=infeasible_count,
        numerical_solver_error_policies=numerical_error_count,
        split_manifest_sha256={path:digest(path) for path in split_manifests},
        independent_checks=len(audit),failed_checks=len(failures),failures=failures,
        out_of_sample_constraint_observations=len(observations),out_of_sample_constraint_violations=len(violations),
        test_violation_counts_by_constraint=violations.groupby("constraint").size().to_dict() if len(violations) else {},
        scope="Independent decision replay, metric arithmetic, actual validation constraints, hashes and finite-job completeness; no LP optimality certificate or infeasibility proof",
        decision_semantics="Approval probabilities and expected counts; test breaches retained as observations, not hidden or refitted",
        files={name:digest(run/name) for name in ("protected_verification_audit.csv","protected_test_constraint_observations.csv","protected_independent_metrics.csv")})
    (run/"protected_verification.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("status","audited_jobs","expected_policy_records","independent_checks","failed_checks","out_of_sample_constraint_violations")},ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=HERE/"results/run_003")
    args = parser.parse_args()
    result = verify(args.run)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
