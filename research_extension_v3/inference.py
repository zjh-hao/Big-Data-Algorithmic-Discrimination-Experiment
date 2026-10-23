"""Paired uncertainty and six-group auditing for validation-frozen policies.

For unit contribution, counts and FRR, resampling joint outcome/decision cells
is exactly equivalent to an ordinary paired bootstrap of the test records.
Models and rules remain fixed: these intervals do not include training or
validation-selection uncertainty.  Monetary contribution is reported as a point
estimate, because arbitrary loan amounts cannot be collapsed into these cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .business import apply_policy, metrics_for_decisions
from .common import dump, now, sha


SCOPE = "Paired test-record bootstrap conditional on fixed trained models and validation-selected policies"
METRICS = ("utility", "utility_per_1000_applicants", "approvals", "approval_rate",
           "good_approved", "difficulty_count", "risk", "thin_frr", "thick_frr", "abs_frr_gap")
SUBGROUP_METRICS = ("utility", "utility_per_1000_applicants", "approvals", "approval_rate",
                   "good_approved", "difficulty_count", "risk", "frr")


def _ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0)


def _cells(frame, baseline, candidate):
    required = {"TARGET", "thin", "income_band"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Bootstrap frame missing {sorted(required - set(frame.columns))}")
    if len(frame) == 0:
        raise ValueError("Bootstrap needs nonempty records")
    y, g, band = (frame[c].to_numpy(dtype=float) for c in ("TARGET", "thin", "income_band"))
    base, cand = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if base.shape != (len(frame),) or cand.shape != base.shape:
        raise ValueError("Paired decisions must match the test frame")
    if not all(np.isfinite(x).all() for x in [y, g, band, base, cand]):
        raise ValueError("Bootstrap values must be finite")
    if not all(np.isin(x, [0, 1]).all() for x in [y, g, base, cand]):
        raise ValueError("Joint-cell bootstrap requires binary labels, groups and hard decisions")
    if not np.isin(band, [0, 1, 2]).all():
        raise ValueError("income_band must be training-defined terciles 0/1/2")
    cells, counts = np.unique(np.column_stack([g, y, base, cand, band]).astype(np.int8),
                              axis=0, return_counts=True)
    return cells, counts


def _cell_metrics(draws, cells, decision_column, loss_ratio, group=None, income_band=None):
    # draws is (resamples, cells); the first row can be the actual cell counts.
    count = np.asarray(draws, dtype=float)
    take = np.ones(len(cells), dtype=bool)
    if group is not None:
        take &= cells[:, 0] == group
    if income_band is not None:
        take &= cells[:, 4] == income_band
    count = count[:, take]
    selected = cells[take]
    y, a, g = selected[:, 1], selected[:, decision_column], selected[:, 0]
    n = count.sum(axis=1)
    approved = count @ a
    good = count @ (a * (1 - y))
    bad = count @ (a * y)
    utility = good - loss_ratio * bad
    normal = count @ (1 - y)
    result = {"records": n, "approvals": approved, "approval_rate": _ratio(approved, n),
              "good_approved": good, "difficulty_count": bad, "risk": _ratio(bad, approved),
              "utility": utility, "utility_per_1000_applicants": _ratio(utility * 1000, n),
              "frr": 1 - _ratio(good, normal)}
    for h, name in [(0, "thick"), (1, "thin")]:
        normal_h = count @ ((g == h) * (1 - y))
        accepted_good_h = count @ ((g == h) * (1 - y) * a)
        result[name + "_frr"] = 1 - _ratio(accepted_good_h, normal_h)
    result["abs_frr_gap"] = abs(result["thin_frr"] - result["thick_frr"])
    return result


def _interval_rows(base, cand, metrics, repetitions, confidence, identical_decisions=False):
    rows = []
    alpha = (1 - confidence) / 2
    for name in metrics:
        b, c = base[name], cand[name]
        point = c[0] - b[0]
        deltas = c[1:] - b[1:]
        valid = np.isfinite(deltas)
        exact_zero = identical_decisions and np.isfinite(point)
        sufficient = np.isfinite(point) and (exact_zero or int(valid.sum()) >= np.ceil(.95 * repetitions))
        low, high = ([0.0, 0.0] if exact_zero else
                     np.quantile(deltas[valid], [alpha, 1 - alpha]) if sufficient else [np.nan, np.nan])
        rows.append({"metric": name, "baseline": float(b[0]), "candidate": float(c[0]),
                     "delta": float(point), "ci_low": float(low), "ci_high": float(high),
                     "confidence": confidence, "bootstrap_repetitions": repetitions,
                     "valid_repetitions": int(valid.sum()),
                     "identical_decisions": identical_decisions,
                     "interval_status": "PASS" if sufficient else "INSUFFICIENT_BOOTSTRAP_SUPPORT",
                     "direction": "candidate_minus_baseline", "uncertainty_scope": SCOPE})
    return rows


def paired_bootstrap(frame: pd.DataFrame, baseline, candidate, loss_ratio: float = 5,
                     repetitions: int = 1000, seed: int = 20261022,
                     confidence: float = .95) -> tuple[list[dict], list[dict]]:
    """Exact paired multinomial bootstrap for unit metrics and six subgroups."""
    if int(repetitions) != repetitions or repetitions < 2:
        raise ValueError("Need at least two bootstrap repetitions")
    if not 0 < confidence < 1 or not np.isfinite(loss_ratio) or loss_ratio <= 0:
        raise ValueError("Invalid confidence or loss ratio")
    repetitions = int(repetitions)
    cells, counts = _cells(frame, baseline, candidate)
    rng = np.random.default_rng(seed)
    resamples = rng.multinomial(len(frame), counts / counts.sum(), size=repetitions)
    draws = np.vstack([counts, resamples])
    base = _cell_metrics(draws, cells, 2, loss_ratio)
    cand = _cell_metrics(draws, cells, 3, loss_ratio)
    identical = bool(np.array_equal(np.asarray(baseline), np.asarray(candidate)))
    overall = _interval_rows(base, cand, METRICS, repetitions, confidence, identical)
    subgroups = []
    for group in (0, 1):
        for band in (0, 1, 2):
            b = _cell_metrics(draws, cells, 2, loss_ratio, group, band)
            c = _cell_metrics(draws, cells, 3, loss_ratio, group, band)
            for row in _interval_rows(b, c, SUBGROUP_METRICS, repetitions, confidence, identical):
                row.update(thin=group, income_band=band, records=int(b["records"][0]))
                if b["records"][0] == 0:
                    row.update(interval_status="NO_SUPPORT", ci_low=np.nan, ci_high=np.nan)
                subgroups.append(row)
    return overall, subgroups


def subgroup_metrics(frame, decisions, loss_ratio=5, config=None):
    """Six training-defined bureau-history × income groups, with amount points."""
    _cells(frame, decisions, decisions)
    decisions = np.asarray(decisions)
    rows = []
    for group in (0, 1):
        for band in (0, 1, 2):
            mask = (frame.thin.to_numpy() == group) & (frame.income_band.to_numpy() == band)
            item = {"thin": group, "credit_group": "thin" if group else "thick",
                    "income_band": band, "income_group": ["low", "middle", "high"][band],
                    "records": int(mask.sum())}
            if not mask.any():
                rows.append({**item, "status": "NO_SUPPORT", "normal_records": 0, "frr": None})
                continue
            values = metrics_for_decisions(frame.loc[mask], decisions[mask], loss_ratio, config)
            normal = int((frame.loc[mask, "TARGET"] == 0).sum())
            item.update({k: values[k] for k in ["approvals", "approval_rate", "good_approved", "difficulty_count",
                        "risk", "utility", "utility_per_1000_applicants", "approved_amount", "amount_utility",
                        "capital_fraction_used", "net_contribution_per_funded_unit"]})
            item.update(normal_records=normal, frr=1 - values["good_approved"] / normal if normal else None,
                        status="PASS" if normal else "NO_NORMAL_SUPPORT")
            rows.append(item)
    return rows


def _near(a, b):
    return a is not None and np.isclose(float(a), float(b), atol=1e-10, rtol=0)


def _primary_policies(frozen, config):
    chosen = []
    for p in frozen["policies"]:
        if not (p.get("utility_basis") == "unit_utility"
                and _near(p.get("loss_ratio"), config["primary_loss_ratio"])
                and _near(p.get("risk_budget"), config["primary_risk_budget"])
                and _near(p.get("capital_fraction"), 1.0)):
            continue
        is_primary_fair = (p["family"] in ("group_profit_gap", "group_profit_gap_noharm")
                           and _near(p.get("epsilon"), config["primary_epsilon"]))
        is_prespecified_one_percent = (p["family"] == "group_profit_gap" and _near(p.get("epsilon"), .01)
                                       and any(_near(e, .01) for e in config.get("epsilons", [])))
        if p["family"] in ("common_profit", "group_max_approval", "group_profit") or is_primary_fair or is_prespecified_one_percent:
            chosen.append(p)
    return chosen


def _aligned(base_frame, candidate_frame):
    for frame in (base_frame, candidate_frame):
        if frame.SK_ID_CURR.duplicated().any():
            raise ValueError("Cannot pair duplicated application IDs")
    base = base_frame.sort_values("SK_ID_CURR").reset_index(drop=True)
    cand = candidate_frame.sort_values("SK_ID_CURR").reset_index(drop=True)
    for column in ["SK_ID_CURR", "TARGET", "thin", "income_band", "AMT_CREDIT"]:
        if not np.array_equal(base[column].to_numpy(), cand[column].to_numpy()):
            raise ValueError(f"Paired model records disagree in {column}")
    return base, cand


def _seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")


def _csv(rows, path):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, path)


@threadpool_limits.wrap(limits=1)
def build_inference(run_dir, repetitions=None, seed=20261022, confidence=.95):
    """Derive intervals from completed saved jobs, without training or selecting."""
    run = Path(run_dir).resolve()
    config = json.loads((run / "config_frozen.json").read_text(encoding="utf-8"))
    repetitions = int(repetitions if repetitions is not None else config.get("bootstrap_repetitions", 1000))
    completed, skipped, sources = {}, [], []
    for job_path in sorted((run / "jobs").glob("*.json")):
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job["status"] != "PASS":
            skipped.append({"job": job_path.stem, "reason": f"Job status {job['status']}"})
            continue
        name = f"{job['model']}_s{job['seed']}"
        frozen_path = run / "policies" / f"{name}.json"
        test_path = run / "predictions" / f"{name}_test.parquet"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(test_path).sort_values("SK_ID_CURR").reset_index(drop=True)
        completed[name] = {"model": job["model"], "seed": job["seed"], "frame": frame,
                           "policies": _primary_policies(frozen, config)}
        sources.append({"job": name, "test_predictions_sha256": sha(test_path),
                        "frozen_policies_sha256": sha(frozen_path), "job_sha256": sha(job_path)})
    intervals, subgroup_intervals, subgroups, comparisons = [], [], [], []
    def compare(base_job, base_policy, candidate_job, candidate_policy, comparison):
        bj, cj = completed[base_job], completed[candidate_job]
        metadata = {"model": cj["model"], "seed": cj["seed"], "comparison": comparison,
                    "baseline_model": bj["model"], "baseline_seed": bj["seed"],
                    "baseline_policy_id": base_policy["policy_id"], "baseline_family": base_policy["family"],
                    "candidate_policy_id": candidate_policy["policy_id"], "candidate_family": candidate_policy["family"],
                    "candidate_noharm_delta": candidate_policy.get("noharm_delta"),
                    "loss_ratio": config["primary_loss_ratio"], "risk_budget": config["primary_risk_budget"],
                    "epsilon": candidate_policy.get("epsilon"), "primary_epsilon": config["primary_epsilon"]}
        if not base_policy["feasible"] or not candidate_policy["feasible"]:
            skipped.append({**metadata, "reason": "One validation-frozen policy is INFEASIBLE"})
            return
        bframe, cframe = _aligned(bj["frame"], cj["frame"])
        base_decision, cand_decision = apply_policy(bframe, base_policy), apply_policy(cframe, candidate_policy)
        pair_seed = _seed(seed, f"{base_job}:{base_policy['policy_id']}:{candidate_job}:{candidate_policy['policy_id']}")
        whole, parts = paired_bootstrap(cframe, base_decision, cand_decision,
                                       config["primary_loss_ratio"], repetitions, pair_seed, confidence)
        intervals.extend([{**metadata, **row} for row in whole])
        subgroup_intervals.extend([{**metadata, **row} for row in parts])
        comparisons.append({**metadata, "bootstrap_seed": pair_seed,
                            "identical_decisions": bool(np.array_equal(base_decision, cand_decision))})
    for name, data in completed.items():
        selected = data["policies"]
        families = {p["family"]: p for p in selected if p["family"] != "group_profit_gap_noharm"
                    and (p["family"] != "group_profit_gap" or _near(p.get("epsilon"), config["primary_epsilon"]))}
        for policy in selected:
            if not policy["feasible"]:
                skipped.append({"job": name, "policy_id": policy["policy_id"], "reason": "INFEASIBLE; no subgroup decisions"})
                continue
            points = subgroup_metrics(data["frame"], apply_policy(data["frame"], policy), config["primary_loss_ratio"], config)
            subgroups.extend([{**row, "model": data["model"], "seed": data["seed"],
                              "policy_id": policy["policy_id"], "family": policy["family"],
                              "epsilon": policy["epsilon"], "noharm_delta": policy["noharm_delta"]} for row in points])
        if "group_profit" in families and "common_profit" in families:
            compare(name, families["common_profit"], name, families["group_profit"], "group_threshold_gain")
        if "group_profit" in families and "group_profit_gap" in families:
            compare(name, families["group_profit"], name, families["group_profit_gap"], "fairness_cost")
            for policy in selected:
                if policy["family"] == "group_profit_gap_noharm":
                    compare(name, families["group_profit_gap"], name, policy, "protection_cost")
                elif (policy["family"] == "group_profit_gap" and _near(policy.get("epsilon"), .01)
                      and not _near(policy.get("epsilon"), config["primary_epsilon"])):
                    compare(name, families["group_profit"], name, policy, "fairness_cost_epsilon_1pct")
    reference_name = "xgboost_s731"
    if reference_name in completed:
        reference_policies = completed[reference_name]["policies"]
        base = next((p for p in reference_policies if p["family"] == "group_profit"), None)
        if base is not None:
            for name, data in completed.items():
                fair = next((p for p in data["policies"] if p["family"] == "group_profit_gap"
                             and _near(p.get("epsilon"), config["primary_epsilon"])), None)
                if fair is not None:
                    compare(reference_name, base, name, fair, "model_gain_after_fairness_vs_xgboost_s731")
    else:
        skipped.append({"reason": "Reference XGBoost seed731 incomplete; no cross-model paired comparisons"})
    _csv(intervals, run / "paired_intervals.csv")
    _csv(subgroups, run / "subgroups.csv")
    _csv(subgroup_intervals, run / "paired_subgroup_intervals.csv")
    manifest = {"created_utc": now(), "bootstrap_repetitions": repetitions, "confidence": confidence,
                "base_seed": seed, "uncertainty_scope": SCOPE,
                "method": "Ordinary paired bootstrap via exact multinomial joint-cell resampling; percentile intervals",
                "cells": "thin × TARGET × baseline approval × candidate approval × training-defined income tercile",
                "money_intervals": "Not estimated; arbitrary amount-weighted business metrics are point estimates in subgroups.csv",
                "no_test_selection": "All completed jobs and configuration-defined primary policies are included; no test-based winner selection",
                "completed_jobs": list(completed), "comparisons": comparisons, "skipped": skipped,
                "sources": sources, "config_sha256": sha(run / "config_frozen.json"),
                "inference_code_sha256": sha(__file__), "interval_rows": len(intervals),
                "subgroup_rows": len(subgroups), "subgroup_interval_rows": len(subgroup_intervals)}
    dump(run / "inference_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="research_extension_v3/results/run_001")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--confidence", type=float, default=.95)
    args = parser.parse_args()
    result = build_inference(args.run, args.repetitions, args.seed, args.confidence)
    print(json.dumps({k: result[k] for k in ["completed_jobs", "interval_rows", "subgroup_rows", "subgroup_interval_rows"]}))


if __name__ == "__main__":
    main()
