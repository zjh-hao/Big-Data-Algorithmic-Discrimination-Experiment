"""Read-only audit of frozen allocation evidence, independent of its fitter.

Reads validation predictions only. Replays each threshold sum directly and
checks every recorded water level/frozen coordinate against its final policy.
Recorded coordinate-maximization evidence is checked for internal consistency;
this is not a new exact-arithmetic proof of LP optimality. Original run sources,
policies, and metrics are never changed. Only new audit artifacts are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


NORMALIZED_CERT_TOL = 1e-6
BUSINESS_RATE_TOL = 2e-7
UTILITY_TOL = 2e-6
HERE = Path(__file__).resolve().parents[1]
GROUPS = [(g, b) for g in (0, 1) for b in (0, 1, 2)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _replay(frame, policy):
    """Direct sums, no dependency on protected_policy or allocation_policy."""
    score = frame.probability.to_numpy(float)
    group = frame.thin.to_numpy()
    q = np.zeros(len(frame))
    simplex_error, negative_weight, threshold_disorder = 0., 0., 0.
    for g in (0, 1):
        entry = policy["mixture"][str(g)]
        ts, ws = np.asarray(entry["thresholds"], float), np.asarray(entry["weights"], float)
        if ts.shape != ws.shape or not np.isfinite(ts).all() or not np.isfinite(ws).all():
            raise ValueError("Malformed frozen policy")
        simplex_error = max(simplex_error, abs(ws.sum()-1))
        negative_weight = max(negative_weight, float(-np.min(ws)))
        threshold_disorder = max(threshold_disorder, float(np.max(-np.diff(ts))) if len(ts)>1 else 0.)
        mask = group == g
        for t, w in zip(ts, ws):
            if w:
                q[mask] += w*(score[mask] <= t)
    monotonicity_error = 0.
    for g in (0, 1):
        mask = group == g
        ordered = q[mask][np.argsort(score[mask], kind="stable")]
        monotonicity_error = max(monotonicity_error, float(np.max(np.diff(ordered))) if len(ordered)>1 else 0.)
    return q, dict(simplex_error=float(simplex_error), negative_weight=max(0., negative_weight),
                   threshold_disorder=max(0., threshold_disorder),
                   monotonicity_error=max(0., monotonicity_error))


def _metrics(frame, q, loss):
    y = frame.TARGET.to_numpy(float)
    g = frame.thin.to_numpy()
    b = frame.income_band.to_numpy()
    amount = frame.AMT_CREDIT.to_numpy(float)
    approvals = float(q.sum())
    bad = float(q@y)
    masks = [(y==0)&(g==h)&(b==band) for h, band in GROUPS]
    if any(not mask.any() for mask in masks):
        raise ValueError("An audited normal-label cell is empty")
    accepts = np.asarray([q[mask].mean() for mask in masks])
    macro = [1-float(q[(y==0)&(g==h)].mean()) for h in (0, 1)]
    return dict(approvals=approvals, risk=bad/approvals if approvals else float("inf"),
                capital_fraction_used=float(q@amount/amount.sum()),
                utility=float(q@(1-(1+loss)*y)), gap=abs(macro[1]-macro[0]),
                accepts=accepts, counts=[int(mask.sum()) for mask in masks])


def audit_run(run):
    run = Path(run)
    cfg = read(run/"config_frozen.json")
    plan_path = run/"experiments_config.json"
    plan_cfg = read(plan_path).get("config", {}) if plan_path.exists() else {}
    primary_beta = cfg.get("primary_beta", plan_cfg.get("primary_beta", .01))
    primary_tau = cfg.get("primary_tau", plan_cfg.get("primary_tau"))
    rows, detail, range_rows = [], [], []
    for path in sorted((run/"policies").glob("*.json")):
        frozen = read(path)
        job = path.stem
        model, seed_text = job.rsplit("_s", 1)
        origin = "run_001" if model == "xgboost" else "run_002"
        source = HERE/"results"/origin/"predictions"/f"{job}_validation.parquet"
        frame = pd.read_parquet(source)
        ids = frame.SK_ID_CURR.astype(str).sort_values().tolist()
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        id_match = digest == frozen["metadata"]["validation_id_sha256"]
        baseline = next(p for p in frozen["policies"] if p["policy_id"] == "lp_profit_baseline")
        bq, _ = _replay(frame, baseline)
        bm = _metrics(frame, bq, cfg["primary_loss_ratio"])
        for p in frozen["policies"]:
            pid = p["policy_id"]
            common = dict(job=job, model=model, seed=int(seed_text), policy=pid,
                          mode=p.get("mode"), beta=p.get("beta"), stored_status=p.get("status"),
                          tau=p.get("tau", p.get("additional_harm_budget")),
                          feasible=bool(p.get("feasible")), normalized_certificate_tolerance=NORMALIZED_CERT_TOL)
            if not p.get("feasible"):
                rows.append({**common, "business_constraints_pass": None,
                             "normalized_leximin_certificate_pass": None, "reason": p.get("reason")})
                continue
            q, replay = _replay(frame, p)
            m = _metrics(frame, q, cfg["primary_loss_ratio"])
            checks = {
                "validation_ids": id_match,
                "risk": m["risk"] <= cfg["primary_risk_budget"]+BUSINESS_RATE_TOL,
                "capital": m["capital_fraction_used"] <= cfg["capital_fraction"]+BUSINESS_RATE_TOL,
                "minimum_approvals": m["approvals"] >= cfg["minimum_approvals"]-UTILITY_TOL,
                "valid_simplex": replay["simplex_error"] <= 1e-8 and replay["negative_weight"] <= 1e-12,
                "monotone_approval": replay["monotonicity_error"] <= 1e-12,
            }
            beta = p.get("beta")
            deterioration = float(max(0., np.max(bm["accepts"]-m["accepts"])))
            if beta is not None:
                checks.update(fairness=m["gap"] <= p["epsilon"]+BUSINESS_RATE_TOL,
                              profit_floor=m["utility"] >= (1-beta)*bm["utility"]-UTILITY_TOL,
                              minimum_harm_face=deterioration <= p["noharm_delta"]+BUSINESS_RATE_TOL)
            stored = p.get("validation_metrics", {})
            stored_metric_errors = {
                "utility": abs(m["utility"]-stored.get("utility", m["utility"])),
                "risk": abs(m["risk"]-stored.get("risk", m["risk"])),
                "gap": abs(m["gap"]-stored.get("abs_frr_gap", m["gap"])),
                "subgroup_acceptance": max(abs((1-stored.get(f"subgroup_{g}_{b}_frr", 1-m["accepts"][i]))-m["accepts"][i])
                                           for i,(g,b) in enumerate(GROUPS)),
            }
            checks["serialized_metrics"] = (stored_metric_errors["utility"] <= UTILITY_TOL and
                                             all(v <= 1e-10 for k,v in stored_metric_errors.items() if k!="utility"))
            row = {**common, **replay, "normal_records": sum(m["counts"]),
                   "business_constraints_pass": all(checks.values()),
                   "business_failed_checks": ";".join(k for k,v in checks.items() if not v),
                   "maximum_deterioration": deterioration,
                   "minimum_required_delta": p.get("minimum_required_delta"),
                   "effective_delta": p.get("noharm_delta"),
                   "normalized_leximin_certificate_pass": None}
            diagnostics = p.get("range_diagnostics")
            stages = p.get("leximin_stages", [])
            per_policy = dict(**common, replay=replay, business_checks=checks,
                              serialized_metric_errors=stored_metric_errors, stage_checks=[])
            if diagnostics:
                lower = np.array([d["minimum_acceptance"] for d in diagnostics])
                upper = np.array([d["maximum_acceptance"] for d in diagnostics])
                span = upper-lower
                threshold = p.get("range_fixed_tolerance", 1e-7)
                active = span>threshold
                z = np.full(6, np.nan)
                z[active] = (m["accepts"][active]-lower[active])/span[active]
                max_stage_violation, max_decrease, max_frozen_error, forced_evidence_excess = 0., 0., 0., 0.
                maximum_below_level, missing_evidence_count = 0., 0
                frozen_indices = []
                previous_level = None
                for k, stage in enumerate(stages):
                    level = float(stage["level"])
                    remaining = stage["remaining"]
                    values = m["accepts"] if p["mode"]=="absolute_maxmin" else z
                    stage_violation = max([0.]+[float(level-values[i]) for i in remaining])
                    decrease = max(0., previous_level-level) if previous_level is not None else 0.
                    frozen_error = max([0.]+[abs(float(values[i])-level) for i in stage["frozen"]])
                    maxima = stage["coordinate_maxima_on_optimal_face"]
                    missing_evidence = [i for i in stage["frozen"] if str(i) not in maxima]
                    forced_excess = max([0.]+[float(maxima[str(i)])-level for i in stage["frozen"] if str(i) in maxima])
                    maximum_below_level = max([maximum_below_level]+[level-float(v) for v in maxima.values()])
                    missing_evidence_count += len(missing_evidence)
                    frozen_indices.extend(stage["frozen"])
                    max_stage_violation = max(max_stage_violation, stage_violation)
                    max_decrease = max(max_decrease, decrease)
                    max_frozen_error = max(max_frozen_error, frozen_error)
                    forced_evidence_excess = max(forced_evidence_excess, forced_excess)
                    per_policy["stage_checks"].append(dict(stage=k, level=level,
                        stage_requirement_violation=stage_violation, decrease_from_previous=decrease,
                        frozen_equality_error=frozen_error, recorded_forced_maximum_excess=forced_excess,
                        missing_forced_evidence=missing_evidence, remaining=remaining, frozen=stage["frozen"]))
                    previous_level = level
                frozen_metadata_error = 0.
                for key, item in p.get("leximin_fixed_coordinates", {}).items():
                    i = int(key)
                    if active[i]:
                        frozen_metadata_error = max(frozen_metadata_error, abs(float(z[i])-item["normalized_value"]))
                range_violation = max([0.]+[-float(v) for v in z[active]]+[float(v)-1 for v in z[active]])
                coverage = set(frozen_indices)==set(np.flatnonzero(active).tolist())
                lex = p["mode"]=="normalized_leximin"
                norm_errors = [max_stage_violation, max_decrease, max_frozen_error,
                               forced_evidence_excess, frozen_metadata_error, range_violation,
                               maximum_below_level]
                cert = bool(coverage and missing_evidence_count==0 and max(norm_errors)<=NORMALIZED_CERT_TOL) if lex else None
                row.update(minimum_opportunity_range=float(span.min()),
                           median_opportunity_range=float(np.median(span)),
                           maximum_opportunity_range=float(span.max()),
                           max_range_per_10000_normal=float(span.max()*10000),
                           active_coordinates=int(active.sum()), effectively_fixed_coordinates=int((~active).sum()),
                           stage_count=len(stages), max_stage_requirement_violation=max_stage_violation,
                           max_level_decrease=max_decrease, max_frozen_equality_error=max_frozen_error,
                           max_recorded_forced_maximum_excess=forced_evidence_excess,
                           max_recorded_maximum_below_level=maximum_below_level,
                           missing_forced_evidence_count=missing_evidence_count,
                           max_frozen_metadata_error=frozen_metadata_error,
                           normalized_range_violation=range_violation,
                           active_coordinates_frozen=coverage, normalized_leximin_certificate_pass=cert,
                           leximin_vacuous_due_to_fixed_ranges=bool(lex and not active.any()))
                per_policy.update(normalized_opportunity=[float(v) if np.isfinite(v) else None for v in z],
                                  normalized_leximin_certificate_pass=cert)
                # Once per beta, not once per allocation ablation, to avoid duplication.
                if lex:
                    for i,d in enumerate(diagnostics):
                        range_rows.append(dict(job=job, beta=beta,
                                               tau=p.get("tau", p.get("additional_harm_budget")), group=d["group"],
                                               normal_records=m["counts"][i], range=float(span[i]),
                                               range_pp=float(span[i]*100), fixed=bool(not active[i])))
            rows.append(row)
            detail.append(per_policy)
    table = pd.DataFrame(rows)
    ranges = pd.DataFrame(range_rows)
    lexrows = [r for r in rows if r.get("mode")=="normalized_leximin" and r.get("feasible")]
    primary_ranges = ranges[ranges.beta.eq(primary_beta)] if not ranges.empty else ranges
    if primary_tau is not None and not primary_ranges.empty:
        primary_ranges = primary_ranges[primary_ranges.tau.eq(primary_tau)]
    summary = dict(
        run=str(run.resolve()), data_used="validation only; no test predictions read",
        method="Direct threshold replay and recomputation; consistency audit of recorded range/optimal-face evidence",
        limitation="This audit does not independently prove the LP extrema with exact arithmetic. Business feasibility and normalized leximin precision are reported separately.",
        normalized_certificate_tolerance=NORMALIZED_CERT_TOL,
        primary_beta=primary_beta, primary_tau=primary_tau,
        policy_rows=len(rows), feasible_policies=sum(r["feasible"] for r in rows),
        business_constraint_failures=sum(r.get("business_constraints_pass") is False for r in rows),
        feasible_leximin_policies=len(lexrows),
        leximin_numerical_certificate_failures=sum(r["normalized_leximin_certificate_pass"] is False for r in lexrows),
        leximin_fixed_range_cases=sum(r.get("leximin_vacuous_due_to_fixed_ranges",False) for r in lexrows),
        primary_group_range_count=len(primary_ranges),
        primary_range_max=float(primary_ranges["range"].max()) if len(primary_ranges) else None,
        primary_range_median=float(primary_ranges["range"].median()) if len(primary_ranges) else None,
        policies=detail,
    )
    run.mkdir(parents=True,exist_ok=True)
    table.to_csv(run/"leximin_numerical_audit.csv", index=False, encoding="utf-8-sig")
    ranges.to_csv(run/"leximin_opportunity_ranges.csv", index=False, encoding="utf-8-sig")
    (run/"leximin_numerical_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2), encoding="utf-8")
    return {k:v for k,v in summary.items() if k!="policies"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=HERE/"results/run_005_theory")
    args=parser.parse_args()
    print(json.dumps(audit_run(args.run),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
