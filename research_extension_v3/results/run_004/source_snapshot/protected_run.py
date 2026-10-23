"""Finite, validation-frozen experiment for profit-budgeted subgroup protection.

Reuses completed score models; does not train or tune on test outcomes.  The
primary specification and all one-factor sensitivity choices are saved before
the first test frame is read.  Fractional decisions are expected randomized
outcomes, with separate implementation-randomness simulations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .business import apply_policy, metrics_for_decisions
from .common import HERE, dump, now, sha
from .protected_policy import fit_protected_policies, apply_protected_policy


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def specifications(full=True):
    def spec(name, mode="budgeted_minimax", protection="six", robust=False,
             beta=.01, delta=.01, epsilon=.02):
        return dict(name=name, mode=mode, protection=protection, robust=robust,
                    beta=beta, delta=delta, epsilon=epsilon, minimax_scope="six")
    values = [spec("lp_gap", mode="profit_gap", protection="none"),
              spec("lp_gap_macro_protected", mode="profit_gap_protected", protection="macro"),
              spec("minimax_macro", protection="macro"), spec("minimax_six"),
              spec("minimax_six_robust", robust=True)]
    if full:
        values += [spec(f"minimax_six_beta_{b:g}", beta=b) for b in (0, .005, .02, .03)]
        values += [spec(f"minimax_six_delta_{d:g}", delta=d) for d in (0, .02)]
        values += [spec("minimax_six_epsilon_0.01", epsilon=.01)]
    return values


def jobs_config():
    jobs = []
    for model in ("lightgbm", "tabm", "xgboost"):
        origin = "run_001" if model == "xgboost" else "run_002"
        for seed in (731, 732, 733):
            jobs.append(dict(job=f"{model}_s{seed}", model=model, seed=seed,
                             experiment="main", source=str(HERE / "results" / origin)))
    for arm in ("biased", "corrected", "opportunity1"):
        for seed in (17, 29, 43, 61, 89):
            model = f"lightgbm_{arm}"
            jobs.append(dict(job=f"{model}_s{seed}", model=model, seed=seed,
                             experiment="stress", source=str(HERE / "results/run_002/bias_stress")))
    return jobs


def enrich(frame, q, cfg):
    result = metrics_for_decisions(frame, q, cfg["primary_loss_ratio"], cfg)
    y = frame.TARGET.to_numpy()
    groups = frame.thin.to_numpy()
    inc = frame.income_band.to_numpy()
    values = []
    for g in (0, 1):
        for b in (0, 1, 2):
            mask = (groups == g) & (inc == b) & (y == 0)
            frr = float(1 - q[mask].mean()) if mask.any() else np.nan
            result[f"g{g}_income{b}_frr"] = frr
            result[f"g{g}_income{b}_normal_records"] = int(mask.sum())
            result[f"subgroup_{g}_{b}_frr"] = frr
            result[f"subgroup_{g}_{b}_normal_records"] = int(mask.sum())
            values.append(frr)
    result["worst_subgroup_frr"] = float(np.nanmax(values))
    result["six_group_frr_range"] = float(np.nanmax(values) - np.nanmin(values))
    result["fractional_records"] = int(((q > 1e-9) & (q < 1 - 1e-9)).sum())
    return result


def violation_flags(m, base, spec, cfg):
    tol = 1e-7
    eps = spec.get("epsilon") if spec.get("epsilon") is not None else cfg["primary_epsilon"]
    delta = spec.get("delta") if spec.get("delta") is not None else cfg["primary_delta"]
    beta = spec.get("beta") if spec.get("beta") is not None else cfg["primary_beta"]
    return dict(risk_met=m["risk"] <= cfg["primary_risk_budget"] + tol,
                capital_met=m["capital_fraction_used"] <= cfg["capital_fraction"] + tol,
                min_approvals_met=m["approvals"] >= cfg["minimum_approvals"] - tol,
                gap_met=m["abs_frr_gap"] <= eps + tol,
                macro_protection_met=max(m[f"{g}_frr"]-base[f"{g}_frr"] for g in ("thin", "thick")) <= delta+tol,
                six_protection_met=max(m[f"g{g}_income{b}_frr"]-base[f"g{g}_income{b}_frr"] for g in (0, 1) for b in (0, 1, 2)) <= delta+tol,
                profit_floor_met=m["utility"] >= (1-beta)*base["utility"] - tol,
                relative_profit_change=(m["utility"]-base["utility"])/abs(base["utility"]) if base["utility"] else np.nan)


def deterministic_policies(source, job):
    path = source / "policies" / f"{job}.json"
    frozen = read(path)
    rows = []
    names = {"group_profit": "det_profit", "group_profit_gap": "det_gap", "group_profit_gap_noharm": "det_gap_macro_protected"}
    for p in frozen["policies"]:
        if p.get("family") not in names or p.get("utility_basis") not in (None, "unit", "unit_utility"):
            continue
        if p.get("risk_budget") != .05 or p.get("loss_ratio") != 5 or p.get("capital_fraction") != 1:
            continue
        if p["family"] != "group_profit" and p.get("epsilon") != .02:
            continue
        if p["family"] == "group_profit_gap_noharm" and p.get("noharm_delta") != .01:
            continue
        rows.append((names[p["family"]], p))
    if len(rows) != 3:
        raise ValueError(f"Expected three frozen deterministic policies: {path}: {len(rows)}")
    return rows


def sample_implementation(frame, q, base, cfg, job, name, repeats=100):
    """Independent per-applicant Bernoulli replay of a fractional policy."""
    rng = np.random.default_rng(20261022)
    y = frame.TARGET.to_numpy(float)
    amount = frame.AMT_CREDIT.to_numpy(float)
    g = frame.thin.to_numpy()
    inc = frame.income_band.to_numpy()
    rows = []
    for repetition in range(repeats):
        a = (rng.random(len(q)) < q).astype(float)
        n = a.sum()
        good = np.sum(a * (1-y))
        bad = np.sum(a * y)
        frr = {k: 1-a[(g == v)&(y == 0)].mean() for v, k in ((0, "thick"), (1, "thin"))}
        subgroup = [1-a[(g == v)&(inc == b)&(y == 0)].mean() for v in (0, 1) for b in (0, 1, 2)]
        u = good-cfg["primary_loss_ratio"]*bad
        rows.append(dict(job=job, policy=name, repetition=repetition, approvals=n,
                         risk=bad/n, utility=u, abs_frr_gap=abs(frr["thin"]-frr["thick"]),
                         worst_subgroup_frr=max(subgroup),
                         capital_fraction_used=np.sum(a*amount)/np.sum(amount),
                         risk_met=bad/n <= cfg["primary_risk_budget"],
                         gap_met=abs(frr["thin"]-frr["thick"]) <= cfg["primary_epsilon"],
                         profit_floor_met=u >= (1-cfg["primary_beta"])*base["utility"]))
    return rows


def paired_intervals(frame, before, after, job, policy, repeats=1000):
    """Paired applicant bootstrap of expected policy outcomes, via joint cells.

    Multinomial cell counts are exactly equivalent to resampling applicant rows
    for these unit/count statistics.  This does not represent training-seed or
    approval-randomization uncertainty; those are reported separately.
    """
    values = pd.DataFrame(dict(a=before, b=after, y=frame.TARGET.to_numpy(),
                               g=frame.thin.to_numpy(), inc=frame.income_band.to_numpy()))
    cells = values.groupby(list(values.columns), dropna=False).size().reset_index(name="n")
    rng = np.random.default_rng(20261022)
    counts = rng.multinomial(len(frame), cells.n.to_numpy()/len(frame), size=repeats)
    y, g, inc = [cells[c].to_numpy() for c in ("y", "g", "inc")]
    def stats(q):
        unit = counts @ (q*(1-6*y))*1000/len(frame)
        frrs = []
        macro = []
        for gv in (0, 1):
            normal = (g == gv)&(y == 0)
            macro.append(1-(counts @ (q*normal))/(counts @ normal))
            for b in (0, 1, 2):
                mask = normal&(inc == b)
                frrs.append(1-(counts @ (q*mask))/(counts @ mask))
        return dict(utility_per_1000_applicants=unit,
                    abs_frr_gap=np.abs(macro[1]-macro[0]),
                    worst_subgroup_frr=np.max(frrs, axis=0))
    a, b = stats(cells.a.to_numpy()), stats(cells.b.to_numpy())
    result = []
    for metric in a:
        difference = b[metric]-a[metric]
        result.append(dict(job=job, policy=policy, baseline="lp_profit_baseline",metric=metric,
                           lower=float(np.quantile(difference,.025)), upper=float(np.quantile(difference,.975)),
                           bootstrap_mean=float(difference.mean()), repetitions=repeats,
                           uncertainty="paired applicant bootstrap of expected decisions"))
    return result


def run_one(run, job_cfg, cfg):
    started = time.perf_counter()
    job = job_cfg["job"]
    source = Path(job_cfg["source"])
    paths = {s: source / "predictions" / f"{job}_{s}.parquet" for s in ("validation", "test")}
    old_job = read(source / "jobs" / f"{job}.json")
    if old_job.get("status") != "PASS":
        raise ValueError(f"Source job not PASS: {job}")
    fingerprints = {s: sha(p) for s,p in paths.items()}
    for split, expected in old_job.get("prediction_sha256", {}).items():
        if split in fingerprints and expected != fingerprints[split]:
            raise ValueError(f"Source prediction hash mismatch: {job}/{split}")
    v = pd.read_parquet(paths["validation"])
    job_config = dict(cfg, policy_specs=job_cfg.get("policy_specs", specifications(job_cfg["experiment"] == "main")))
    frozen = fit_protected_policies(v, job_config)
    # Save fitted rule before touching its test labels or reporting test metrics.
    frozen_path = run / "policies" / f"{job}.json"
    dump(frozen_path, frozen)
    frozen = read(frozen_path)
    t = pd.read_parquet(paths["test"])
    matrices = {"validation": v, "test": t}
    lp_policies = [(p.get("name", p.get("policy_id")), p) for p in frozen["policies"]]
    det_policies = deterministic_policies(source, job)
    rows, draws, intervals = [], [], []
    decisions = {s: pd.DataFrame({"SK_ID_CURR": frame.SK_ID_CURR}) for s,frame in matrices.items()}
    baseline_metrics, baseline_q = {}, {}
    base_policy = next(p for name,p in lp_policies if name == "lp_profit_baseline")
    for split, frame in matrices.items():
        baseline_q[split] = apply_protected_policy(frame, base_policy)
        baseline_metrics[split] = enrich(frame, baseline_q[split], cfg)
    for name,p in lp_policies + det_policies:
        is_det = name.startswith("det_")
        row = dict(job=job,model=job_cfg["model"],seed=job_cfg["seed"],experiment=job_cfg["experiment"],
                   policy=name,policy_id=name,feasible=bool(p.get("feasible",False)),
                   policy_space="deterministic_threshold_pair" if is_det else "monotone_threshold_mixture",
                   decision_family="deterministic_two_thresholds" if is_det else "group_threshold_mixture",
                   source_policy_id=p.get("policy_id"),
                   mode="deterministic" if is_det else p.get("mode"),protection=p.get("protection"),
                   minimax_scope=p.get("minimax_scope"),
                   robust=p.get("robust",False), beta=p.get("beta"),delta=p.get("delta",p.get("noharm_delta")),
                   epsilon=p.get("epsilon"), source_run=str(source),
                   status=p.get("status"), primary=name == "minimax_six")
        for key in ("requested_delta", "minimum_required_delta", "effective_delta", "deterioration_slack", "validation_requested_goal_met", "adaptation_protocol"):
            row[key] = p.get(key)
        if row["feasible"]:
            for split, frame in matrices.items():
                q = apply_policy(frame,p).astype(float) if is_det else apply_protected_policy(frame,p)
                decisions[split][name] = q
                m = enrich(frame,q,cfg)
                m.update(violation_flags(m,baseline_metrics[split],p,cfg))
                row.update({f"{split}_{k}": value for k,value in m.items()})
                if split == "test" and job_cfg["experiment"] == "main" and name in ("minimax_six", "minimax_six_robust", "lp_gap", "adaptive_primary", "adaptive_robust"):
                    draws += sample_implementation(frame,q,baseline_metrics[split],cfg,job,name)
                    intervals += paired_intervals(frame,baseline_q[split],q,job,name)
        rows.append(row)
    for split,frame in decisions.items():
        path = run / "decisions" / f"{job}_{split}.parquet"
        path.parent.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(path,index=False)
    save_csv(run / "job_metrics" / f"{job}.csv", rows)
    save_csv(run / "randomization" / f"{job}.csv", draws)
    save_csv(run / "paired_intervals" / f"{job}.csv", intervals)
    dump(run / "jobs" / f"{job}.json", dict(**job_cfg,status="PASS",finished=now(),
         seconds=time.perf_counter()-started,source_prediction_sha256=fingerprints,
         frozen_sha256=sha(frozen_path), policies=len(rows),feasible=sum(r["feasible"] for r in rows)))
    print(json.dumps(dict(job=job,status="PASS",policies=len(rows),feasible=sum(r["feasible"] for r in rows),
                          seconds=round(time.perf_counter()-started,2))),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE/"results/run_003")
    parser.add_argument("--only", help="Comma separated job names, for a bounded smoke run")
    args = parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    cfg = dict(version="3.2.0",grid=256,minimum_approvals=500,minimum_subgroup_normal=100,
               primary_loss_ratio=5,primary_risk_budget=.05,capital_fraction=1.,
               primary_epsilon=.02,primary_beta=.01,primary_delta=.01,
               scenario_interest=.12,scenario_cost=.03,policy_specs=specifications())
    jobs = jobs_config()
    plan = dict(config=cfg,jobs=jobs,primary_policy="minimax_six",report_expected_jobs=[j["job"] for j in jobs],
                claim="Course-project integration of existing LP randomized thresholds, profit floor and subgroup protection; no SOTA novelty claim",
                protocol="Validation-only policy fit; fixed specification; historical test previously inspected; no fresh blind-test claim",
                economic_interpretation="TARGET repayment difficulty; scenario contribution, not actual bank profit",
                uncertainty="Expected randomized outcomes; applicant bootstrap and randomization replay are distinct")
    frozen_plan = args.run/"experiments_config.json"
    if frozen_plan.exists() and read(frozen_plan) != plan:
        raise ValueError("Existing run plan differs; use a new run directory")
    dump(frozen_plan,plan)
    dump(args.run/"config_frozen.json",dict(cfg,report_expected_jobs=plan["report_expected_jobs"]))
    dump(args.run/"status.json",dict(status="RUNNING",updated=now()))
    selected = set(args.only.split(",")) if args.only else None
    for j in jobs:
        if selected is not None and j["job"] not in selected:
            continue
        marker = args.run / "jobs" / f"{j['job']}.json"
        if marker.exists() and read(marker).get("status") == "PASS":
            continue
        run_one(args.run,j,cfg)
    for folder,out in (("job_metrics","policy_metrics.csv"),("randomization","randomization_replays.csv"),("paired_intervals","paired_intervals.csv")):
        files = sorted((args.run/folder).glob("*.csv"))
        frames = []
        for p in files:
            try:
                frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                pass
        if frames:
            pd.concat(frames,ignore_index=True).to_csv(args.run/out,index=False,encoding="utf-8-sig")
    complete = all((args.run/"jobs"/f"{j['job']}.json").exists() and read(args.run/"jobs"/f"{j['job']}.json").get("status") == "PASS" for j in jobs)
    dump(args.run/"status.json",dict(status="COMPLETE" if complete else "PARTIAL",updated=now(),
         completed_jobs=sum((args.run/"jobs"/f"{j['job']}.json").exists() for j in jobs),expected_jobs=len(jobs),
         source_sha256={name:sha(HERE/name) for name in ("protected_run.py","protected_policy.py","business.py")}))


if __name__ == "__main__":
    main()
