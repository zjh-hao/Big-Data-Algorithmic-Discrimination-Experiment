"""Independent saved-record audit; no fitting and no imports from business.py.

Checks include partition membership, AUC/Brier, threshold replay, frozen-grid
optimality and infeasibility, plus expected outcomes for randomized decisions.
Partial runs are explicitly PARTIAL rather than reported as complete passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

HERE = Path(__file__).resolve().parent


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or pd.isna(b)
    if b is None or pd.isna(b):
        return False
    return bool(np.isclose(float(a), float(b), rtol=1e-8, atol=1e-8))


def recompute(frame, approvals, loss_ratio, config):
    """Independent direct arithmetic for deterministic or fractional approvals."""
    a = np.asarray(approvals, dtype=float)
    y, p, g, amount = [frame[column].to_numpy(dtype=float) for column in ["TARGET", "probability", "thin", "AMT_CREDIT"]]
    count = a.sum()
    bad = (a * y).sum()
    good = (a * (1 - y)).sum()
    funded = (a * amount).sum()
    bad_amount = (a * amount * y).sum()
    spread = config.get("scenario_interest", .12) - config.get("scenario_cost", .03)
    utility = good - loss_ratio * bad
    amount_utility = spread * (funded - (1 + loss_ratio) * bad_amount)
    metrics = {
        "records": len(frame), "approvals": count, "approval_rate": count / len(frame),
        "difficulty_count": bad, "good_approved": good, "risk": bad / count if count else None,
        "expected_risk": (a * p).sum() / count if count else None,
        "approved_amount": funded, "capital_fraction_used": funded / amount.sum(),
        "amount_weighted_risk": bad_amount / funded if funded else None,
        "utility": utility, "expected_utility": (a * (1 - (1 + loss_ratio) * p)).sum(),
        "utility_per_1000_applicants": utility * 1000 / len(frame),
        "amount_utility": amount_utility,
        "expected_amount_utility": spread * (a * amount * (1 - (1 + loss_ratio) * p)).sum(),
        "amount_utility_per_1000_applicants": amount_utility * 1000 / len(frame),
        "net_contribution_per_funded_unit": amount_utility / funded if funded else None,
    }
    for value, name in [(0, "thick"), (1, "thin")]:
        mask = g == value
        normal = int((y[mask] == 0).sum())
        n = a[mask].sum()
        b = (a[mask] * y[mask]).sum()
        approved_good = (a[mask] * (1 - y[mask])).sum()
        metrics.update({f"{name}_records": int(mask.sum()), f"{name}_normal_records": normal,
                        f"{name}_approvals": n, f"{name}_good_approved": approved_good,
                        f"{name}_difficulty_approved": b, f"{name}_risk": b / n if n else None,
                        f"{name}_frr": 1 - approved_good / normal if normal else None})
    metrics["abs_frr_gap"] = abs(metrics["thin_frr"] - metrics["thick_frr"])
    return metrics


def prefixes(frame, thresholds):
    """Record-derived cumulative values on the frozen global threshold grid."""
    result = []
    for group in [0, 1]:
        part = frame[frame.thin.eq(group)].sort_values("probability", kind="stable")
        p = part.probability.to_numpy(dtype=float)
        y = part.TARGET.to_numpy(dtype=float)
        amount = part.AMT_CREDIT.to_numpy(dtype=float)
        index = np.searchsorted(p, thresholds, side="right")
        cumulative = lambda values: np.r_[0., np.cumsum(values, dtype=float)][index]
        n = index.astype(float)
        bad = cumulative(y)
        result.append({"n": n, "bad": bad, "amount": cumulative(amount),
                       "bad_amount": cumulative(amount * y), "good": n - bad,
                       "frr": 1 - (n - bad) / (y == 0).sum()})
    a, b = result
    out = {key: a[key][:, None] + b[key][None, :] for key in ["n", "bad", "amount", "bad_amount", "good"]}
    out["gap"] = np.abs(a["frr"][:, None] - b["frr"][None, :])
    out["frr0"], out["frr1"] = a["frr"], b["frr"]
    return out


def replay_interpolation(frame, interpolation):
    """Replay serialized Fairlearn threshold mixtures without importing Fairlearn."""
    score = 1 - frame.probability.to_numpy(dtype=float)
    group = frame.thin.to_numpy(dtype=int)
    output = np.zeros(len(frame), dtype=float)
    for value, rule in interpolation.items():
        mask = group == int(value)
        for index in [0, 1]:
            match = re.fullmatch(r"\[(>=|<=|>|<)(.+)\]", rule[f"operation{index}"])
            if not match:
                raise ValueError("Unsupported frozen threshold operation")
            operator, threshold = match.group(1), float(match.group(2))
            if operator == ">":
                positive = score[mask] > threshold
            elif operator == "<":
                positive = score[mask] < threshold
            elif operator == ">=":
                positive = score[mask] >= threshold
            else:
                positive = score[mask] <= threshold
            output[mask] += float(rule[f"p{index}"]) * positive
    return output


def validate_grid_policy(policy, frozen, frame, space, config):
    """Check selected score is globally optimal on the *saved finite grid*."""
    loss = float(policy["loss_ratio"])
    n, bad = space["n"], space["bad"]
    feasible = (n >= int(frozen["metadata"].get("minimum_approvals", config.get("minimum_approvals", 500))))
    feasible &= bad <= float(policy["risk_budget"]) * n + 1e-10
    if policy["family"] == "common_profit":
        feasible &= np.eye(len(n), dtype=bool)
    funded = space["amount"]
    total = float(frame.AMT_CREDIT.sum())
    feasible &= funded <= float(policy["capital_fraction"]) * total + max(1., total) * 1e-12
    if policy.get("epsilon") is not None:
        feasible &= space["gap"] <= float(policy["epsilon"]) + 1e-10
    if policy.get("noharm_delta") is not None:
        by_id = {entry["policy_id"]: entry for entry in frozen["policies"]}
        baseline = by_id.get(policy.get("baseline_policy_id"))
        if not baseline or not baseline.get("feasible"):
            feasible[:] = False
        else:
            thresholds = np.asarray(baseline["thresholds"])
            approved = frame.probability.to_numpy() <= thresholds[frame.thin.to_numpy(dtype=int)]
            metrics = recompute(frame, approved, loss, config)
            delta = float(policy["noharm_delta"])
            feasible &= space["frr0"][:, None] <= metrics["thick_frr"] + delta + 1e-10
            feasible &= space["frr1"][None, :] <= metrics["thin_frr"] + delta + 1e-10
    if not policy.get("feasible"):
        return not feasible.any(), "INFEASIBLE contains a feasible grid candidate" if feasible.any() else None
    if not feasible.any():
        return False, "PASS has no feasible grid candidate"
    grid = np.asarray(frozen["metadata"]["threshold_grid"])
    chosen = [np.flatnonzero(grid == float(value)) for value in policy["thresholds"]]
    if any(len(index) != 1 for index in chosen):
        return False, "Threshold not in frozen shared grid"
    i, j = int(chosen[0][0]), int(chosen[1][0])
    if not feasible[i, j]:
        return False, "Selected thresholds violate validation constraints"
    if policy["objective"] == "approvals":
        objective = n
    elif policy.get("utility_basis", policy["objective"]) == "amount_utility":
        spread = config.get("scenario_interest", .12) - config.get("scenario_cost", .03)
        objective = spread * (space["amount"] - (1 + loss) * space["bad_amount"])
    else:
        objective = space["good"] - loss * bad
    best = float(np.max(objective[feasible]))
    actual = float(objective[i, j])
    if not same(best, actual):
        return False, f"Selected objective {actual} below grid optimum {best}"
    if "validation_objective" in policy and not same(actual, policy["validation_objective"]):
        return False, "Saved validation objective disagrees with records"
    return True, None


def main(run_dir, require_complete=False):
    run = Path(run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    config = read_json(run / "config_frozen.json") or read_json(run / "config.json") or read_json(HERE / "config.json", {})
    checks, skipped, jobs = [], [], []

    def check(ok, name, detail=None):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    manifest_path = run / "split_manifest.parquet"
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        check(manifest.SK_ID_CURR.notna().all() and manifest.SK_ID_CURR.is_unique, "Partitions have unique nonmissing IDs")
        check(set(manifest.TARGET.unique()) <= {0, 1} and set(manifest.thin.unique()) <= {0, 1}, "Partition labels/groups are binary")
        check({"train", "validation", "test"}.issubset(set(manifest.split)), "Train/validation/test partitions present")
        audit = read_json(run / "data_audit.json", {})
        expected_hash = audit.get("split_manifest_sha256")
        if expected_hash:
            check(sha(manifest_path) == expected_hash, "Split manifest matches recorded SHA-256")
        else:
            skipped.append("No split_manifest_sha256 recorded in data_audit.json")
    else:
        skipped.append("split_manifest.parquet not yet available")

    table_path = run / "policy_metrics.csv"
    table = pd.read_csv(table_path, float_precision="round_trip") if table_path.exists() and table_path.stat().st_size else pd.DataFrame()
    model_path = run / "model_metrics.csv"
    model_table = pd.read_csv(model_path, float_precision="round_trip") if model_path.exists() and model_path.stat().st_size else pd.DataFrame()
    if not table.empty:
        check(not table.duplicated(["model", "seed", "policy_id"]).any(), "Each model/seed/policy has one saved result")
    policy_paths = sorted((run / "policies").glob("*_s*.json"))
    completed_names = set()
    for frozen_path in policy_paths:
        name = frozen_path.stem
        model, seed_string = name.rsplit("_s", 1)
        if not seed_string.isdigit():
            # Individual Fairlearn rules share the model_sSEED prefix.
            continue
        seed = int(seed_string)
        job_path = run / "jobs" / f"{name}.json"
        job = read_json(job_path, {})
        if job and str(job.get("status", "PASS")).upper() not in {"PASS", "SUCCESS", "COMPLETED"}:
            skipped.append(f"{name}: job not complete")
            continue
        paths = {split: run / "predictions" / f"{name}_{split}.parquet" for split in ["validation", "test"]}
        if any(not path.exists() for path in paths.values()):
            skipped.append(f"{name}: saved validation/test predictions incomplete")
            continue
        frozen = read_json(frozen_path)
        if job.get("frozen_policy_sha256"):
            check(sha(frozen_path) == job["frozen_policy_sha256"], f"{name}: completed job frozen-policy SHA-256")
        frames = {split: pd.read_parquet(path) for split, path in paths.items()}
        identity_errors = []
        for split, frame in frames.items():
            if not frame.SK_ID_CURR.is_unique:
                identity_errors.append(f"duplicate {split} IDs")
            required = ["SK_ID_CURR", "TARGET", "thin", "AMT_CREDIT", "probability"]
            if not set(required).issubset(frame):
                identity_errors.append(f"missing required {split} columns")
                continue
            if not np.isfinite(frame[["TARGET", "thin", "AMT_CREDIT", "probability"]].to_numpy()).all():
                identity_errors.append(f"nonfinite {split} records")
            if not frame.probability.between(0, 1).all():
                identity_errors.append(f"out-of-range {split} probabilities")
            if not frame.AMT_CREDIT.gt(0).all():
                identity_errors.append(f"nonpositive {split} amounts")
            if manifest is not None:
                expected = manifest[manifest.split.eq(split)].set_index("SK_ID_CURR")
                if set(expected.index) != set(frame.SK_ID_CURR):
                    identity_errors.append(f"{split} membership disagrees with partition")
                else:
                    aligned = expected.loc[frame.SK_ID_CURR]
                    for column in ["TARGET", "thin"]:
                        if not np.array_equal(aligned[column].to_numpy(), frame[column].to_numpy()):
                            identity_errors.append(f"{split} {column} disagrees with partition")
        check(not identity_errors, f"{name}: saved record identities and values", identity_errors[:8])
        if identity_errors:
            continue
        val, test = frames["validation"], frames["test"]
        check(not (set(val.SK_ID_CURR) & set(test.SK_ID_CURR)), f"{name}: validation and test disjoint")
        calibration = frozen.get("calibration")
        if calibration and all("raw_probability" in frame for frame in frames.values()):
            calibration_errors = []
            for split, frame in frames.items():
                raw = np.clip(frame.raw_probability.to_numpy(dtype=float), 1e-7, 1 - 1e-7)
                linear = float(calibration["slope"]) * np.log(raw / (1 - raw)) + float(calibration["intercept"])
                expected = 1 / (1 + np.exp(-linear))
                if not np.allclose(expected, frame.probability, rtol=1e-10, atol=1e-10):
                    calibration_errors.append(split)
            check(not calibration_errors and float(calibration["slope"]) > 0,
                  f"{name}: saved calibration replay", calibration_errors)
        expected_id_sha = hashlib.sha256("\n".join(val.SK_ID_CURR.astype(str).sort_values().to_list()).encode()).hexdigest()
        meta = frozen.get("metadata", {})
        check(meta.get("fit_split") == "validation" and meta.get("validation_id_sha256") == expected_id_sha,
              f"{name}: policy fit split and validation IDs")
        grid = np.asarray(meta.get("threshold_grid", []), dtype=float)
        unique = np.unique(val.probability.to_numpy())
        count = int(meta.get("grid_requested", config.get("grid", 256)))
        expected_grid = np.unique(np.r_[-1., unique if len(unique) <= count else np.quantile(val.probability, np.linspace(0, 1, count)), 1.])
        grid_ok = np.array_equal(grid, expected_grid)
        fingerprint = hashlib.sha256(grid.astype("<f8").tobytes()).hexdigest()
        check(grid_ok and meta.get("candidate_fingerprint") == fingerprint, f"{name}: independent frozen candidate grid")
        if not grid_ok:
            continue
        record_rows = table[(table.model == model) & (table.seed == seed)] if not table.empty else pd.DataFrame()
        stored = {row.policy_id: row for row in record_rows.itertuples(index=False)}
        errors, optimality_errors, constraint_flag_errors = [], [], []
        metrics_count = 0
        space = prefixes(val, grid)
        for policy in frozen["policies"]:
            pid = policy["policy_id"]
            optimal, reason = validate_grid_policy(policy, frozen, val, space, config)
            if not optimal:
                optimality_errors.append(f"{pid}: {reason}")
            if pid not in stored:
                errors.append(f"{pid}: missing policy CSV row")
                continue
            row = stored[pid]
            if not policy.get("feasible"):
                if getattr(row, "status", None) != "INFEASIBLE":
                    errors.append(f"{pid}: infeasible policy not marked INFEASIBLE")
                continue
            thresholds = np.asarray(policy["thresholds"])
            recomputed = {}
            for split, frame in frames.items():
                a = frame.probability.to_numpy() <= thresholds[frame.thin.to_numpy(dtype=int)]
                actual = recompute(frame, a, float(policy["loss_ratio"]), config)
                recomputed[split] = actual
                expected = policy["validation_metrics"] if split == "validation" else row._asdict()
                for key, value in actual.items():
                    if key in expected:
                        metrics_count += 1
                        if not same(value, expected[key]):
                            errors.append(f"{pid}/{split}/{key}: records={value}, saved={expected[key]}")
            actual = recomputed["test"]
            flags = {"test_risk_constraint_met": actual["risk"] <= policy["risk_budget"] + 1e-10,
                     "test_capital_constraint_met": actual["capital_fraction_used"] <= policy["capital_fraction"] + 1e-10}
            if policy.get("epsilon") is not None:
                flags["test_gap_constraint_met"] = actual["abs_frr_gap"] <= policy["epsilon"] + 1e-10
            if policy.get("noharm_delta") is not None:
                by_id = {item["policy_id"]: item for item in frozen["policies"]}
                baseline = by_id[policy["baseline_policy_id"]]
                bt = np.asarray(baseline["thresholds"])
                ba = test.probability.to_numpy() <= bt[test.thin.to_numpy(dtype=int)]
                bm = recompute(test, ba, policy["loss_ratio"], config)
                flags["test_noharm_constraint_met"] = all(actual[f"{g}_frr"] <= bm[f"{g}_frr"] + policy["noharm_delta"] + 1e-10 for g in ["thin", "thick"])
            for flag, value in flags.items():
                if hasattr(row, flag) and str(getattr(row, flag)).lower() in {"true", "false"} and str(getattr(row, flag)).lower() != str(value).lower():
                    constraint_flag_errors.append(f"{pid}/{flag}")
        check(not errors, f"{name}: deterministic record-level validation/test metric replay", {"metric_values": metrics_count, "errors": errors[:10]})
        check(not optimality_errors, f"{name}: finite-grid optimality and infeasibility", {"policies": len(frozen["policies"]), "errors": optimality_errors[:10]})
        check(not constraint_flag_errors, f"{name}: reported test constraint flags", constraint_flag_errors[:10])
        mm = model_table[(model_table.model == model) & (model_table.seed == seed)] if not model_table.empty else pd.DataFrame()
        check(len(mm) == 1, f"{name}: one model metric row")
        if len(mm) == 1:
            check(same(roc_auc_score(test.TARGET, test.probability), mm.iloc[0].auc) and
                  same(brier_score_loss(test.TARGET, test.probability), mm.iloc[0].brier), f"{name}: independent AUC and Brier")
        # Randomized policies are evaluated by their stored per-record approval probabilities.
        fair_path = run / "predictions" / f"{name}_fairlearn.parquet"
        fair_errors, fair_count = [], 0
        if fair_path.exists():
            fair = pd.read_parquet(fair_path)
            if set(fair.SK_ID_CURR) != set(test.SK_ID_CURR) or not fair.SK_ID_CURR.is_unique:
                fair_errors.append("Randomized decision IDs disagree with test partition")
            else:
                fair = fair.set_index("SK_ID_CURR").loc[test.SK_ID_CURR].reset_index()
                non_policy = {"SK_ID_CURR", "TARGET", "thin", "AMT_CREDIT", "probability", "income_band"}
                for column in [key for key in fair if key not in non_policy and key in stored]:
                    a = fair[column].to_numpy(dtype=float)
                    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
                        fair_errors.append(f"{column}: invalid approval probability")
                        continue
                    row = stored[column]
                    actual = recompute(test, a, float(row.loss_ratio), config)
                    for key, value in actual.items():
                        if hasattr(row, key) and not same(value, getattr(row, key)):
                            fair_errors.append(f"{column}/{key}")
                    fair_count += 1
            check(not fair_errors, f"{name}: randomized expected-outcome replay", {"policies": fair_count, "errors": fair_errors[:10]})
        else:
            extra = set(stored) - {item["policy_id"] for item in frozen["policies"]}
            for pid in sorted(extra):
                decision_path = run / "decisions" / f"{name}_{pid}.parquet"
                if not decision_path.exists():
                    fair_errors.append(f"{pid}: missing per-record approval probabilities")
                    continue
                decisions = pd.read_parquet(decision_path)
                if not decisions.SK_ID_CURR.is_unique or set(decisions.SK_ID_CURR) != set(test.SK_ID_CURR):
                    fair_errors.append(f"{pid}: decision IDs disagree with test partition")
                    continue
                decisions = decisions.set_index("SK_ID_CURR").loc[test.SK_ID_CURR]
                a = decisions.approval_probability.to_numpy(dtype=float)
                if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
                    fair_errors.append(f"{pid}: invalid approval probabilities")
                    continue
                row = stored[pid]
                actual = recompute(test, a, float(row.loss_ratio), config)
                for key, value in actual.items():
                    if hasattr(row, key) and not same(value, getattr(row, key)):
                        fair_errors.append(f"{pid}/{key}")
                fair_count += 1
            if extra:
                check(not fair_errors, f"{name}: randomized expected-outcome replay", {"policies": fair_count, "errors": fair_errors[:10]})
        completed_names.add((model, seed))
        jobs.append({"model": model, "seed": seed, "frozen_policies": len(frozen["policies"]),
                     "metric_values_reconstructed": metrics_count, "randomized_policies_reconstructed": fair_count})

    # Audit the separately stored, business-aligned standard Fairlearn baseline.
    business_root = run / "fairlearn_business"
    business_complete = set()
    business_table_path = business_root / "metrics.csv"
    business_table = pd.read_csv(business_table_path, float_precision="round_trip") if business_table_path.exists() else pd.DataFrame()
    candidates_path = business_root / "candidate_validation.csv"
    candidates = pd.read_csv(candidates_path, float_precision="round_trip") if candidates_path.exists() else pd.DataFrame()
    for path in sorted((business_root / "jobs").glob("*.json")):
        job = read_json(path)
        if job.get("status") != "PASS":
            continue
        model, seed = job["model"], int(job["seed"])
        name = f"{model}_s{seed}"
        frames = {split: pd.read_parquet(run / "predictions" / f"{name}_{split}.parquet") for split in ["validation", "test"]}
        source_ok = all(job.get(f"{split}_sha256") == sha(run / "predictions" / f"{name}_{split}.parquet") for split in frames)
        errors, selection_errors, replayed = [], [], 0
        rows = business_table[(business_table.model == model) & (business_table.seed == seed)] if not business_table.empty else pd.DataFrame()
        pool = candidates[(candidates.model == model) & (candidates.seed == seed)] if not candidates.empty else pd.DataFrame()
        for row in rows.itertuples(index=False):
            pid = row.policy_id
            rule = read_json(business_root / "policies" / f"{name}_{pid}.json", {})
            eligible = pool[pool.policy_id.eq(pid)].copy()
            if eligible.empty or set(eligible.training_cost.astype(int)) != set(rule.get("cost_grid", [])):
                selection_errors.append(f"{pid}: incomplete candidate-cost grid")
                continue
            ok = (eligible.approvals >= config.get("minimum_approvals", 500))
            ok &= eligible.risk <= config["primary_risk_budget"] + 1e-10
            ok &= eligible.abs_frr_gap <= float(row.epsilon) + 1e-8
            selected = eligible[ok].sort_values(["utility", "approvals", "risk", "training_cost"], ascending=[False, False, True, True])
            if not rule.get("feasible"):
                if len(selected):
                    selection_errors.append(f"{pid}: marked INFEASIBLE despite eligible cost")
                continue
            if selected.empty or int(selected.iloc[0].training_cost) != int(rule["selected_cost"]):
                selection_errors.append(f"{pid}: selected cost differs from validation ranking")
            if rule.get("validation_sha256") != job.get("validation_sha256") or rule.get("fit_split") != "validation":
                selection_errors.append(f"{pid}: validation provenance disagreement")
            try:
                for split, frame in frames.items():
                    q = replay_interpolation(frame, rule["interpolation"])
                    actual = recompute(frame, q, float(row.loss_ratio), config)
                    expected = rule["validation_metrics"] if split == "validation" else row._asdict()
                    for metric, value in actual.items():
                        if metric in expected and not same(value, expected[metric]):
                            errors.append(f"{pid}/{split}/{metric}")
                    if split == "test":
                        decisions = pd.read_parquet(business_root / "decisions" / f"{name}_{pid}.parquet")
                        if not decisions.SK_ID_CURR.is_unique or set(decisions.SK_ID_CURR) != set(frame.SK_ID_CURR):
                            errors.append(f"{pid}: decision ID mismatch")
                        else:
                            saved_q = decisions.set_index("SK_ID_CURR").loc[frame.SK_ID_CURR].approval_probability
                            if not np.allclose(q, saved_q, atol=1e-10, rtol=1e-10):
                                errors.append(f"{pid}: frozen mixture disagrees with saved decisions")
                replayed += 1
            except (ValueError, KeyError, FileNotFoundError) as error:
                errors.append(f"{pid}: {error}")
        check(source_ok, f"{name}: business Fairlearn source SHA-256")
        check(not selection_errors and len(rows) == len(config.get("fairlearn_tolerances", [])),
              f"{name}: business Fairlearn validation-only cost selection", selection_errors[:8])
        check(not errors, f"{name}: business Fairlearn frozen mixture validation/test replay", {"policies": replayed, "errors": errors[:10]})
        business_complete.add((model, seed))

    expected_jobs = {(str(model), int(seed)) for model in config.get("models", []) for seed in config.get("seeds", [])}
    missing = sorted(expected_jobs - completed_names)
    business_missing = sorted(completed_names - business_complete) if (business_root / "protocol.json").exists() else []
    complete = not missing and not business_missing and manifest is not None and bool(completed_names)
    if require_complete:
        check(complete, "All configured model/seed jobs and split manifest available", {"missing": missing})
    failed = [item for item in checks if not item["passed"]]
    status = "FAIL" if failed else "PASS" if complete else "PARTIAL"
    result = {"status": status, "complete": complete, "run_directory": str(run), "audit_categories": len(checks),
              "failed_categories": len(failed), "checks": checks, "jobs": jobs,
              "missing_configured_jobs": [{"model": model, "seed": seed} for model, seed in missing], "skipped": skipped,
              "business_fairlearn_jobs_audited": len(business_complete),
              "missing_business_fairlearn_jobs": [{"model": model, "seed": seed} for model, seed in business_missing],
              "scope": "Independent arithmetic and optimum on saved finite grids. Does not prove absence of all leakage, causal fairness, or future constraint guarantees."}
    (run / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [f"Verification status: {status}", f"Completed audited model-seed jobs: {len(jobs)}",
             f"Audit categories: {len(checks)}; failed: {len(failed)}", f"Missing configured jobs: {len(missing)}"]
    lines.extend(f"FAIL: {item['name']}: {item['detail']}" for item in failed)
    lines.extend(f"SKIP: {item}" for item in skipped)
    (run / "verification.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "results" / "run_001")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    result = main(arguments.run_dir, arguments.require_complete)
    print(json.dumps({key: result[key] for key in ["status", "complete", "audit_categories", "failed_categories"]}))
    sys.exit(1 if result["status"] == "FAIL" else 0)
