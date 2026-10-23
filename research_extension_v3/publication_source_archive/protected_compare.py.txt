"""Matched existing standard Fairlearn versus new frozen approval methods.

No fitting or best-test selection: uses predeclared 2pp Fairlearn tolerance and
the same six LightGBM/TabM score files.  Different policy objectives/spaces are
kept explicit; this is a practical method comparison, not an optimality claim.
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from .common import HERE, dump, now, sha
from .protected_run import enrich, paired_intervals, read, save_csv


def compare(run):
    cfg = read(run/"config_frozen.json")
    source = HERE/"results/run_002"
    fair_root = source/"fairlearn_business"
    fair_metrics = pd.read_csv(fair_root/"metrics.csv")
    ours = pd.read_csv(run/"policy_metrics.csv")
    rows, intervals, hashes = [], [], {}
    for model in ("lightgbm", "tabm"):
        for seed in (731,732,733):
            job = f"{model}_s{seed}"
            pred_path = source/"predictions"/f"{job}_test.parquet"
            frame = pd.read_parquet(pred_path)
            selected = fair_metrics[(fair_metrics.model==model)&(fair_metrics.seed==seed)&np.isclose(fair_metrics.epsilon,.02)]
            if len(selected) != 1 or not bool(selected.iloc[0].feasible):
                raise ValueError(f"Missing fixed Fairlearn comparator {job}")
            rule_id = selected.iloc[0].policy_id
            path = fair_root/"decisions"/f"{job}_{rule_id}.parquet"
            decisions = pd.read_parquet(path)
            aligned = frame[["SK_ID_CURR"]].merge(decisions[["SK_ID_CURR","approval_probability"]],on="SK_ID_CURR",validate="one_to_one",how="left")
            if aligned.approval_probability.isna().any():
                raise ValueError("Fairlearn missing applicant decisions")
            qf = aligned.approval_probability.to_numpy()
            fm = enrich(frame,qf,cfg)
            # Independent published-row reconciliation prevents mixing score runs.
            for key in ("utility","risk","thin_frr","thick_frr","abs_frr_gap"):
                if not np.isclose(fm[key],selected.iloc[0][key],atol=1e-8,rtol=1e-8):
                    raise ValueError(f"Stored Fairlearn metric mismatch {job}/{key}")
            rows.append(dict(job=job,model=model,seed=seed,method="cost_sensitive_fairlearn",epsilon=.02,**fm))
            own_dec_path = run/"decisions"/f"{job}_test.parquet"
            own_dec = pd.read_parquet(own_dec_path).set_index("SK_ID_CURR").loc[frame.SK_ID_CURR]
            for name in ("det_profit","det_gap","lp_profit_baseline","lp_gap","adaptive_primary","adaptive_robust_beta_0.03"):
                selected_ours = ours[(ours.job==job)&(ours.policy==name)]
                if len(selected_ours) != 1:
                    raise ValueError(f"Missing registered method: {job}/{name}")
                if not bool(selected_ours.iloc[0].feasible):
                    rows.append(dict(job=job,model=model,seed=seed,method=name,feasible=False))
                    continue
                q = own_dec[name].to_numpy()
                m = enrich(frame,q,cfg)
                rows.append(dict(job=job,model=model,seed=seed,method=name,feasible=True,
                                 utility_delta_vs_fairlearn=m["utility"]-fm["utility"],
                                 worst_frr_delta_vs_fairlearn=m["worst_subgroup_frr"]-fm["worst_subgroup_frr"],**m))
                if name in ("lp_gap","adaptive_primary"):
                    ci = paired_intervals(frame,qf,q,job,name)
                    for row in ci:
                        row["baseline"] = "cost_sensitive_fairlearn"
                    intervals.extend(ci)
            hashes[job] = dict(prediction=sha(pred_path),fairlearn_decisions=sha(path),our_decisions=sha(own_dec_path))
    out = pd.DataFrame(rows)
    out.loc[out.method.eq("cost_sensitive_fairlearn"),"feasible"] = True
    save_csv(run/"standard_method_comparison.csv",out)
    save_csv(run/"standard_method_paired_intervals.csv",intervals)
    fields = ["utility","approval_rate","risk","thin_frr","thick_frr","abs_frr_gap","worst_subgroup_frr","utility_delta_vs_fairlearn","worst_frr_delta_vs_fairlearn"]
    summary = out.groupby(["model","method"],dropna=False)[fields].mean().reset_index()
    save_csv(run/"standard_method_summary.csv",summary)
    dump(run/"standard_method_comparison_manifest.json",dict(generated=now(),source_sha256=sha(__file__),inputs=hashes,
         rows=len(out),protocol="Fixed Fairlearn tol=.02 from run002, frozen main run004 methods, same model/seed/applicant outcomes",
         interpretation="Objectives and policy spaces differ. Differences quantify practical matched-score behavior, not universal superiority. Expected randomized metrics."))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=HERE/"results/run_004")
    compare(parser.parse_args().run)
