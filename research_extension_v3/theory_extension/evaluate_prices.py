"""Replay all 27 pre-frozen opportunity-price policies on the existing test set.

No fitting, optimization or eta selection occurs here. The validation target
W0+eta is distinguished from the realized test improvement over the frozen
W0 reference policy. All approvals are expectations under frozen mixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import HERE, clean, dump, now, sha
from ..protected_policy import apply_protected_policy, protected_metrics
from ..verify import recompute


CELLS=[(g,b) for g in (0,1) for b in (0,1,2)]
CHECK_TOL=3e-8


def _object_sha(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()


def _independent_q(frame,policy):
    q=np.zeros(len(frame),dtype=float)
    probability=frame.probability.to_numpy(dtype=float)
    group=frame.thin.to_numpy(dtype=int)
    for g in (0,1):
        mask=group==g
        entry=policy["mixture"][str(g)]
        for threshold,weight in zip(entry["thresholds"],entry["weights"]):
            q[mask]+=float(weight)*(probability[mask]<=float(threshold))
    return q


def _checked_metrics(frame,policy,cfg):
    q=apply_protected_policy(frame,policy)
    explicit=_independent_q(frame,policy)
    replay_error=float(np.max(np.abs(q-explicit)))
    if replay_error>CHECK_TOL:
        raise AssertionError("Explicit threshold sum differs from frozen-policy replay")
    metrics=protected_metrics(frame,q,cfg["primary_loss_ratio"],cfg)
    independently=recompute(frame,explicit,cfg["primary_loss_ratio"],cfg)
    checked=1
    max_metric_error=0.
    for key,value in independently.items():
        other=metrics[key]
        equal=(value is None and other is None) or (value is not None and other is not None and np.isclose(value,other,atol=2e-7,rtol=1e-8))
        if not equal:
            raise AssertionError(f"Independent direct metric mismatch: {key}")
        if value is not None:
            max_metric_error=max(max_metric_error,abs(float(value)-float(other)))
        checked+=1
    # Explicit six-cell denominators do not call the protected metric helper.
    y=frame.TARGET.to_numpy()
    g=frame.thin.to_numpy()
    income=frame.income_band.to_numpy()
    frrs=[]
    for group,band in CELLS:
        mask=(g==group)&(income==band)
        normal=mask&(y==0)
        if not normal.any():
            raise ValueError("No test normal-label support in a required group")
        normal_count=int(normal.sum())
        frr=1-float(explicit[normal].sum())/normal_count
        if normal_count!=metrics[f"subgroup_{group}_{band}_normal_records"] or not np.isclose(frr,metrics[f"subgroup_{group}_{band}_frr"],atol=CHECK_TOL,rtol=0):
            raise AssertionError("Independent six-cell denominator / rejection metric mismatch")
        frrs.append(frr)
        checked+=2
    if not np.isclose(max(frrs),metrics["worst_subgroup_frr"],atol=CHECK_TOL,rtol=0):
        raise AssertionError("Independent weakest subgroup mismatch")
    checked+=1
    for group in (0,1):
        mask=g==group
        order=np.argsort(frame.probability.to_numpy()[mask],kind="stable")
        if np.any(np.diff(q[mask][order])>CHECK_TOL):
            raise AssertionError("Approval probability is not monotone in score")
        checked+=1
    digest=hashlib.sha256()
    digest.update(frame.SK_ID_CURR.to_numpy(dtype="<i8").tobytes())
    digest.update(np.asarray(q,dtype="<f8").tobytes())
    return metrics,{"checks":checked,"explicit_q_max_abs_error":replay_error,
                    "direct_metric_max_abs_error":max_metric_error,
                    "record_order_approval_sha256":digest.hexdigest()}


def evaluate(output):
    plan_path=output/"pricing_plan.json"
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    cfg=plan["config"]
    expected_etas=[float(v) for v in cfg["etas"]]
    if expected_etas!=[.0005,.001,.002] or len(plan["jobs"])!=9:
        raise ValueError("Expected the previously frozen 9-job, 3-eta pricing plan")
    rows=[]
    checks=0
    inputs=[]
    for spec in plan["jobs"]:
        model,seed=spec["model"],int(spec["seed"])
        job=f"{model}_s{seed}"
        frozen_path=output/"jobs"/f"{job}.json"
        frozen=json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen_hash=sha(frozen_path)
        if frozen["status"]!="PASS" or frozen["config"]!=cfg:
            raise ValueError("Frozen pricing job incomplete or config differs")
        if [float(r["eta"]) for r in frozen["rows"]]!=expected_etas:
            raise ValueError("Frozen eta sequence differs; no target can be selected or skipped")
        origin=HERE/"results"/("run_001" if model=="xgboost" else "run_002")
        test_path=origin/"predictions"/f"{job}_test.parquet"
        source_marker=origin/"jobs"/f"{job}.json"
        source=json.loads(source_marker.read_text(encoding="utf-8"))
        if source.get("status")!="PASS" or sha(source_marker)!=frozen["source_job_sha256"]:
            raise ValueError("Original frozen scorer marker changed")
        test_sha=sha(test_path)
        if source.get("prediction_sha256",{}).get("test")!=test_sha:
            raise ValueError("Original frozen test predictions changed")
        # The stored validation path is hashed only; no labels or scores are read.
        if sha(Path(frozen["source_validation"]))!=frozen["source_validation_sha256"]:
            raise ValueError("Validation prediction fingerprint changed since policy fitting")
        frame=pd.read_parquet(test_path)
        if frame.SK_ID_CURR.duplicated().any():
            raise ValueError("Duplicate test IDs")
        baseline=frozen["baseline"]
        w0_policy=frozen["W0_policy"]
        if _object_sha(baseline)!=frozen["baseline_sha256"]:
            raise ValueError("Frozen baseline object fingerprint differs")
        base,base_audit=_checked_metrics(frame,baseline,cfg)
        w0,w0_audit=_checked_metrics(frame,w0_policy,cfg)
        checks+=base_audit["checks"]+w0_audit["checks"]
        baseline_hash=_object_sha(baseline)
        w0_hash=_object_sha(w0_policy)
        for priced in frozen["rows"]:
            if not priced.get("feasible") or priced.get("status")!="PASS":
                raise ValueError("All 27 frozen policies must be feasible before this replay")
            policy=priced["policy"]
            m,audit=_checked_metrics(frame,policy,cfg)
            checks+=audit["checks"]
            eta=float(priced["eta"])
            minimum=1-m["worst_subgroup_frr"]
            w0_minimum=1-w0["worst_subgroup_frr"]
            harm=max(0.,max(m[f"subgroup_{g}_{b}_frr"]-base[f"subgroup_{g}_{b}_frr"] for g,b in CELLS))
            w0_harm=max(0.,max(w0[f"subgroup_{g}_{b}_frr"]-base[f"subgroup_{g}_{b}_frr"] for g,b in CELLS))
            cap=float(priced["minimum_harm_at_target"])+float(priced["numerical_allowance"])
            input_hash=_object_sha({"pricing_plan_sha256":sha(plan_path),"frozen_job_sha256":frozen_hash,
                                    "policy_sha256":_object_sha(policy),"baseline_sha256":baseline_hash,
                                    "W0_policy_sha256":w0_hash,"test_prediction_sha256":test_sha})
            row={"job":job,"model":model,"seed":seed,"policy":policy["policy_id"],
                 "status":"PASS","evaluation_split":"test","decision_semantics":"stochastic_expectation",
                 "eta":eta,"eta_percentage_points":100*eta,"beta":float(priced["beta"]),
                 "epsilon":float(priced["epsilon"]),"validation_W0":float(priced["W0"]),
                 "validation_target_minimum_opportunity":float(priced["target_minimum_opportunity"]),
                 "validation_achieved_minimum_opportunity":float(priced["achieved_minimum_opportunity"]),
                 "validation_minimum_required_harm":float(priced["minimum_required_harm"]),
                 "validation_minimum_harm_at_target":float(priced["minimum_harm_at_target"]),
                 "tau_star":float(priced["tau_star"]),"frozen_total_harm_cap":cap,
                 "test_minimum_opportunity":minimum,"W0_test_minimum_opportunity":w0_minimum,
                 "test_minimum_opportunity_gain_vs_W0_policy":minimum-w0_minimum,
                 "test_eta_gain_met_vs_W0_policy":minimum-w0_minimum>=eta-CHECK_TOL,
                 "test_validation_absolute_target_met":minimum>=float(priced["target_minimum_opportunity"])-CHECK_TOL,
                 "test_max_subgroup_deterioration_vs_baseline":harm,
                 "W0_test_max_subgroup_deterioration_vs_baseline":w0_harm,
                 "test_max_harm_change_vs_W0_policy":harm-w0_harm,
                 "test_risk_met":m["risk"]<=cfg["primary_risk_budget"]+CHECK_TOL,
                 "test_gap_met":m["abs_frr_gap"]<=float(priced["epsilon"])+CHECK_TOL,
                 "test_profit_floor_met":m["utility"]>=(1-float(priced["beta"]))*base["utility"]-CHECK_TOL*len(frame),
                 "test_harm_cap_met":harm<=cap+CHECK_TOL,
                 "test_capital_met":m["capital_fraction_used"]<=cfg["capital_fraction"]+CHECK_TOL,
                 "test_minimum_approvals_met":m["approvals"]>=cfg["minimum_approvals"]-CHECK_TOL*len(frame),
                 "test_utility_delta_vs_baseline":m["utility"]-base["utility"],
                 "test_relative_profit_change_vs_baseline":m["utility"]/base["utility"]-1,
                 "test_utility_delta_vs_W0_policy":m["utility"]-w0["utility"],
                 "test_approval_delta_vs_W0_policy":m["approvals"]-w0["approvals"],
                 "test_risk_delta_vs_W0_policy":m["risk"]-w0["risk"],
                 "test_gap_delta_vs_W0_policy":m["abs_frr_gap"]-w0["abs_frr_gap"],
                 "pricing_job_sha256":frozen_hash,"policy_sha256":_object_sha(policy),
                 "baseline_policy_sha256":baseline_hash,"W0_policy_sha256":w0_hash,
                 "source_test_sha256":test_sha,"input_sha256":input_hash,
                 **audit}
            for label,metrics in (("test",m),("baseline_test",base),("W0_test",w0)):
                row.update({f"{label}_{key}":value for key,value in metrics.items()})
            for g,b in CELLS:
                key=f"subgroup_{g}_{b}"
                row[f"test_{key}_opportunity"]=1-m[key+"_frr"]
                row[f"test_{key}_opportunity_gain_vs_W0_policy"]=w0[key+"_frr"]-m[key+"_frr"]
                row[f"test_{key}_deterioration_vs_baseline"]=m[key+"_frr"]-base[key+"_frr"]
            row["output_sha256"]=_object_sha(row)
            rows.append(row)
        inputs.append(dict(job=job,pricing_job_sha256=frozen_hash,test_prediction_sha256=test_sha,
                           baseline_approval_sha256=base_audit["record_order_approval_sha256"],
                           W0_approval_sha256=w0_audit["record_order_approval_sha256"]))
        print(json.dumps({"job":job,"evaluated":len(frozen["rows"]),"checks":"PASS"}),flush=True)
    if len(rows)!=27:
        raise AssertionError("Not all 27 policies were evaluated")
    csv_path=output/"test_evaluation.csv"
    pd.DataFrame(rows).to_csv(csv_path,index=False,encoding="utf-8-sig")
    # Round-trip output fingerprints are bound to semantic objects, while the
    # file fingerprint covers CSV representation and ordering.
    dump(output/"evaluation_status.json",dict(status="COMPLETE",completed_utc=now(),
         jobs=9,policies=27,independent_checks=checks,failures=0,
         fit_calls=0,target_selection="None; all three pre-frozen eta values replayed for every scorer",
         target_interpretation="Validation target is W0_validation+eta. Test gain is minimum test opportunity minus the frozen W0 policy's minimum test opportunity; it is measured, not imposed.",
         sampling_interpretation="Expected metrics under frozen stochastic mixtures; historical test has already been inspected",
         pricing_plan_sha256=sha(plan_path),evaluator_sha256=sha(Path(__file__)),
         output_csv_sha256=sha(csv_path),output_csv=str(csv_path),input_fingerprints=inputs,
         test_gap_met=sum(r["test_gap_met"] for r in rows),
         test_risk_met=sum(r["test_risk_met"] for r in rows),
         test_profit_floor_met=sum(r["test_profit_floor_met"] for r in rows),
         test_eta_gain_met_vs_W0_policy=sum(r["test_eta_gain_met_vs_W0_policy"] for r in rows)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results/run_006_budget/opportunity_prices")
    args=parser.parse_args()
    evaluate(args.output)


if __name__=="__main__":
    main()
