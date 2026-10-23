"""Predeclared finite minimum-necessary-harm extension after hard infeasibility.

Selects required protection relaxation from validation LP feasibility, never by
test performance.  This is a practical fallback, not a claim to satisfy the
original 1pp protection goal when that goal is impossible in this policy space.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .common import HERE, dump, now, sha
from .protected_run import jobs_config, read, run_one


def specs(full):
    def adaptive(name, beta=.01, robust=False):
        return dict(name=name,mode="adaptive_protected_minimax",protection="six",minimax_scope="six",
                    beta=beta,delta=.01,epsilon=.02,robust=robust,deterioration_slack=.001)
    rows = [adaptive("adaptive_primary"), adaptive("adaptive_robust",robust=True),
            dict(name="lp_gap",mode="profit_gap",epsilon=.02),
            dict(name="minimax_six",mode="budgeted_minimax",beta=.01,delta=.01,
                 epsilon=.02,protection="six",minimax_scope="six")]
    if full:
        rows += [adaptive(f"adaptive_beta_{b:g}",beta=b) for b in (0,.005,.02,.03,.05)]
        rows += [adaptive(f"adaptive_robust_beta_{b:g}",beta=b,robust=True) for b in (.02,.03,.05)]
        rows += [dict(name="lp_profit_risk_robust",mode="profit",robust=True),
                 dict(name="lp_gap_robust",mode="profit_gap",epsilon=.02,robust=True)]
    else:
        rows += [adaptive("adaptive_robust_beta_0.03",beta=.03,robust=True)]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=HERE/"results/run_004")
    args = parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    cfg = dict(version="3.3.0",grid=256,minimum_approvals=500,minimum_subgroup_normal=100,
               primary_loss_ratio=5,primary_risk_budget=.05,capital_fraction=1.,
               primary_epsilon=.02,primary_beta=.01,primary_delta=.01,deterioration_slack=.001,
               scenario_interest=.12,scenario_cost=.03,policy_specs=specs(True))
    jobs = jobs_config()
    for j in jobs:
        j["policy_specs"] = specs(j["experiment"] == "main")
    plan = dict(config=cfg,jobs=jobs,primary_policy="adaptive_primary",
                report_expected_jobs=[j["job"] for j in jobs],
                trigger="run003 validation hard subgroup-protection infeasibility",
                protocol="Fixed LP lexicographic stages: min required deterioration, then min worst6 FRR within dstar+.001, then max contribution; validation only",
                fallback_semantics="Original 1pp goal remains recorded, effective protection may exceed it; no test-based choice of slack or budgets",
                budget_sensitivity=[0,.005,.01,.02,.03,.05],
                economic_interpretation="TARGET repayment difficulty; scenario contribution, not actual bank profit",
                uncertainty="Expected randomized outcomes; paired applicant bootstrap and randomization replay reported separately")
    plan_path = args.run/"experiments_config.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("Existing run plan differs; use a new output directory")
    dump(plan_path,plan)
    dump(args.run/"config_frozen.json",dict(cfg,report_expected_jobs=plan["report_expected_jobs"],primary_policy="adaptive_primary"))
    sources = ("adaptive_run.py","protected_run.py","protected_policy.py","business.py")
    snapshot = args.run/"source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in sources:
        target = snapshot/name
        if target.exists() and sha(target) != sha(HERE/name):
            raise ValueError(f"Run source changed during resumption: {name}")
        shutil.copy2(HERE/name,target)
    dump(args.run/"status.json",dict(status="RUNNING",updated=now(),source_sha256={n:sha(snapshot/n) for n in sources}))
    for j in jobs:
        marker = args.run/"jobs"/f"{j['job']}.json"
        if marker.exists() and read(marker).get("status") == "PASS":
            continue
        run_one(args.run,j,cfg)
    for folder,out in (("job_metrics","policy_metrics.csv"),("randomization","randomization_replays.csv"),("paired_intervals","paired_intervals.csv")):
        frames = []
        for path in sorted((args.run/folder).glob("*.csv")):
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                pass
        if frames:
            pd.concat(frames,ignore_index=True).to_csv(args.run/out,index=False,encoding="utf-8-sig")
    dump(args.run/"status.json",dict(status="COMPLETE",updated=now(),completed_jobs=len(jobs),expected_jobs=len(jobs),
         primary_policy="adaptive_primary",source_sha256={n:sha(snapshot/n) for n in sources}))


if __name__ == "__main__":
    main()
