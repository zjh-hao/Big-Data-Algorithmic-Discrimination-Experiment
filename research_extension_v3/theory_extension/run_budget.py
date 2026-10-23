"""Finite validation-frozen pilot of normalized opportunity allocation.

Same score models, threshold family and business constraints; only the welfare
allocation criterion changes. Primary beta=.01 is declared before test replay.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ..common import HERE, dump, now, sha
from ..protected_policy import apply_protected_policy
from ..protected_run import read, enrich, save_csv, paired_intervals
from ..verify import recompute
from .budget_allocation import fit_allocation_policies
from .paired_certificate import certificate_rows


def independent_q(frame, policy):
    result = np.zeros(len(frame))
    for g in (0, 1):
        mask = frame.thin.to_numpy() == g
        entry = policy["mixture"][str(g)]
        for t,weight in zip(entry["thresholds"],entry["weights"]):
            result[mask] += weight*(frame.probability.to_numpy()[mask] <= t)
    return result


def run_one(output,job,cfg):
    start = time.perf_counter()
    model,seed = job["model"],job["seed"]
    name = f"{model}_s{seed}"
    origin = HERE/"results"/("run_001" if model=="xgboost" else "run_002")
    paths = {split:origin/"predictions"/f"{name}_{split}.parquet" for split in ("validation","test")}
    old_job = read(origin/"jobs"/f"{name}.json")
    if old_job["status"] != "PASS":
        raise ValueError("Required score model not complete")
    hashes = {split:sha(path) for split,path in paths.items()}
    for split,h in old_job.get("prediction_sha256",{}).items():
        if split in hashes and h != hashes[split]:
            raise ValueError("Prediction fingerprint differs")
    validation = pd.read_parquet(paths["validation"])
    frozen = fit_allocation_policies(validation,cfg)
    frozen_path = output/"policies"/f"{name}.json"
    dump(frozen_path,frozen)
    frozen = read(frozen_path)
    certificates = certificate_rows(validation,frozen,alpha=.05,number_of_scorers=9)
    for r in certificates:
        r.update(job=name,model=model,seed=seed)
    # All policy fitting is complete and serialized before test labels are read.
    test = pd.read_parquet(paths["test"])
    if set(validation.SK_ID_CURR)&set(test.SK_ID_CURR):
        raise ValueError("Split overlap")
    rows,checks,ci = [],[],[]
    frames = {"validation":validation,"test":test}
    decisions = {split:pd.DataFrame({"SK_ID_CURR":frame.SK_ID_CURR}) for split,frame in frames.items()}
    baseline = next(p for p in frozen["policies"] if p.get("policy_id",p.get("name"))=="lp_profit_baseline")
    base_metrics = {split:enrich(frame,apply_protected_policy(frame,baseline),cfg) for split,frame in frames.items()}
    old_root = HERE/"results/run_004"
    old_json = read(old_root/"policies"/f"{name}.json")
    old_method = next(p for p in old_json["policies"] if p["name"]=="adaptive_primary")
    policies = [(p.get("policy_id",p.get("name")),p,False) for p in frozen["policies"]]
    policies += [("previous_adaptive_primary",old_method,True)]
    for pid,p,is_old in policies:
        row = dict(job=name,model=model,seed=seed,policy=pid,mode=p.get("mode"),beta=p.get("beta"),
                   feasible=bool(p.get("feasible")),status=p.get("status"),source="run004" if is_old else "theory_pilot",
                   minimum_required_delta=p.get("minimum_required_delta"), tau=p.get("tau"),
                   delta=p.get("noharm_delta"),epsilon=p.get("epsilon"),
                   primary=p.get("mode")=="normalized_leximin" and p.get("beta")==.01 and p.get("tau")==.001)
        if row["feasible"]:
            for split,frame in frames.items():
                q = apply_protected_policy(frame,p)
                independent = independent_q(frame,p)
                checks.append(dict(job=name,policy=pid,split=split,check="independent_serialization_replay",passed=bool(np.allclose(q,independent,atol=2e-8,rtol=0))))
                m = enrich(frame,q,cfg)
                direct = recompute(frame,independent,cfg["primary_loss_ratio"],cfg)
                for key,value in direct.items():
                    equal = (value is None and m.get(key) is None) or np.isclose(value,m[key],atol=2e-7,rtol=1e-8)
                    checks.append(dict(job=name,policy=pid,split=split,check=f"metric_{key}",passed=bool(equal)))
                base = base_metrics[split]
                harmful = [m[f"subgroup_{g}_{b}_frr"]-base[f"subgroup_{g}_{b}_frr"] for g in (0,1) for b in (0,1,2)]
                m.update(max_subgroup_deterioration=max(harmful),utility_change_vs_baseline=m["utility"]-base["utility"],
                         relative_profit_change=m["utility"]/base["utility"]-1,
                         risk_met=m["risk"]<=cfg["primary_risk_budget"]+2e-7,
                         gap_met=m["abs_frr_gap"]<=cfg["primary_epsilon"]+2e-7,
                         profit_floor_met=p.get("beta") is None or m["utility"] >= (1-p["beta"])*base["utility"]-2e-6)
                row.update({f"{split}_{k}":v for k,v in m.items()})
                decisions[split][pid] = q
                if split=="validation" and not is_old:
                    checks.append(dict(job=name,policy=pid,split=split,check="risk_constraint",passed=m["risk_met"]))
                    if p.get("beta") is not None:
                        checks += [dict(job=name,policy=pid,split=split,check="fairness_constraint",passed=m["gap_met"]),
                                   dict(job=name,policy=pid,split=split,check="profit_floor",passed=m["profit_floor_met"]),
                                   dict(job=name,policy=pid,split=split,check="minimum_harm_face",passed=max(harmful)<=float(p["noharm_delta"])+2e-7)]
                if split=="test" and row["primary"]:
                    q_old = apply_protected_policy(frame,old_method)
                    for r in paired_intervals(frame,q_old,q,name,pid):
                        r["baseline"] = "previous_adaptive_primary"
                        ci.append(r)
        rows.append(row)
    failures = [r for r in checks if not r["passed"]]
    if failures:
        dump(output/"failures"/f"{name}.json",failures)
        raise AssertionError(f"Independent verification failed: {failures[:3]}")
    for split,frame in decisions.items():
        target = output/"decisions"/f"{name}_{split}.parquet"
        target.parent.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(target,index=False)
    save_csv(output/"job_metrics"/f"{name}.csv",rows)
    save_csv(output/"certificates"/f"{name}.csv",certificates)
    save_csv(output/"checks"/f"{name}.csv",checks)
    save_csv(output/"intervals"/f"{name}.csv",ci)
    dump(output/"jobs"/f"{name}.json",dict(**job,status="PASS",finished=now(),source_prediction_sha256=hashes,
         frozen_policy_sha256=sha(frozen_path),policy_rows=len(rows),checks=len(checks),seconds=time.perf_counter()-start))
    print(json.dumps(dict(job=name,status="PASS",policies=len(rows),feasible=sum(r["feasible"] for r in rows),seconds=round(time.perf_counter()-start,2))),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=HERE/"results/run_006_budget")
    parser.add_argument("--only")
    args = parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    cfg = dict(version="budget-theory-1",grid=256,minimum_approvals=500,minimum_subgroup_normal=100,
               primary_loss_ratio=5,primary_risk_budget=.05,capital_fraction=1.,primary_epsilon=.02,
               primary_beta=.01,betas=[.01],additional_harm_budgets=[.001,.003],primary_tau=.001,allocation_modes=["absolute_maxmin","normalized_maxmin","normalized_leximin"],
               face_tolerance=1e-8,range_tolerance=1e-7,scenario_interest=.12,scenario_cost=.03)
    jobs = [dict(model=m,seed=s) for m in ("lightgbm","tabm","xgboost") for s in (731,732,733)]
    plan = dict(config=cfg,jobs=jobs,primary="normalized_leximin beta=.01 tau=.001",created_context="fixed before pilot test replay",
                hypothesis="Predeclared explicit protection budget restores meaningful allocation space; full normalized leximin is compared with same-budget absolute maxmin and one-stage normalized maxmin",
                selection="No test-based hyperparameter choice; original historical test already inspected",
                theoretical_scope="Exact empirical polytope theorems; numerical LP uses explicit tolerances; DKW population certificates separate")
    planpath = args.run/"experiments_config.json"
    if planpath.exists() and read(planpath)!=plan:
        raise ValueError("Frozen pilot plan differs")
    dump(planpath,plan)
    dump(args.run/"config_frozen.json",cfg)
    snapshot = args.run/"source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("allocation_policy.py","budget_allocation.py","paired_certificate.py","run_budget.py"):
        source = Path(__file__).parent/name
        if (snapshot/name).exists() and sha(source)!=sha(snapshot/name):
            raise ValueError(f"Pilot code changed: {name}; archive failed preflight before retry")
        shutil.copy2(source,snapshot/name)
    dump(args.run/"status.json",dict(status="RUNNING",updated=now(),expected_jobs=9))
    for job in jobs:
        name = f"{job['model']}_s{job['seed']}"
        if args.only and name!=args.only:
            continue
        marker = args.run/"jobs"/f"{name}.json"
        if marker.exists() and read(marker).get("status")=="PASS":
            continue
        run_one(args.run,job,cfg)
    for folder,target in (("job_metrics","policy_metrics.csv"),("certificates","paired_certificates.csv"),("checks","independent_checks.csv"),("intervals","paired_intervals.csv")):
        frames = []
        for path in sorted((args.run/folder).glob("*.csv")):
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                pass
        if frames:
            pd.concat(frames,ignore_index=True).to_csv(args.run/target,index=False,encoding="utf-8-sig")
    count = sum((args.run/"jobs"/f"{j['model']}_s{j['seed']}.json").exists() for j in jobs)
    dump(args.run/"status.json",dict(status="COMPLETE" if count==9 else "PARTIAL",updated=now(),completed_jobs=count,expected_jobs=9,
         source_sha256={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()}))


if __name__=="__main__":
    main()
