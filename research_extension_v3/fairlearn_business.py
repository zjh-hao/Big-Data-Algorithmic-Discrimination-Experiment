"""A business-aligned standard Fairlearn baseline selected on validation only.

Repeating TARGET=1 records w times makes expected classification accuracy
proportional to good_approved - w * difficulty_approved plus a policy-independent
constant. TARGET=0 records are not repeated, so favorable-outcome TPR and its
group parity constraint are unchanged. This uses unmodified ThresholdOptimizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .business import metrics_for_decisions
from .common import HERE, BASELINE_SOURCES, dump, sha, now

SOURCE = BASELINE_SOURCES / 'fairlearn'
COST_GRID = (3, 5, 7, 10, 15, 25)


class FrozenFavorableScore(ClassifierMixin, BaseEstimator):
    """Expose fixed favorable scores to the standard sklearn/Fairlearn API."""
    def fit(self, X=None, y=None):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 1
        return self

    def predict_proba(self, X):
        favorable = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([1 - favorable, favorable])

    def predict(self, X):
        return (np.asarray(X).reshape(-1) >= .5).astype(int)


def replication_indices(target, cost):
    if int(cost) != cost or cost <= 0:
        raise ValueError("Replication costs must be positive integers")
    return np.repeat(np.arange(len(target)), np.where(np.asarray(target) == 1, int(cost), 1))


def equivalence_checks():
    """Executable check of expected weighted accuracy and favorable TPR identity."""
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    q = np.array([.9, .3, .2, .1, .7, .4, .6, .05])
    rows = []
    for cost in COST_GRID:
        repeat = replication_indices(target, cost)
        weighted_accuracy = np.mean(np.where(target[repeat] == 0, q[repeat], 1 - q[repeat]))
        utility = np.dot(q, 1 - target) - cost * np.dot(q, target)
        expected = (cost * target.sum() + utility) / len(repeat)
        if not np.isclose(weighted_accuracy, expected, atol=1e-12, rtol=0):
            raise AssertionError("Weighted expected accuracy is not affine in scenario utility")
        for value in [0, 1]:
            before = q[(target == 0) & (group == value)].mean()
            after = q[repeat][(target[repeat] == 0) & (group[repeat] == value)].mean()
            if not np.isclose(before, after, atol=1e-12, rtol=0):
                raise AssertionError("Favorable-outcome TPR changed under negative replication")
        rows.append({"cost": cost, "weighted_accuracy": weighted_accuracy, "affine_utility_accuracy": expected,
                     "positive_tpr_unchanged": True})
    return rows


def serialize_rule(value):
    if isinstance(value, dict):
        return {str(key): serialize_rule(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_rule(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    return str(value)


def atomic_csv(frame, path):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def aggregate(output):
    metrics, candidates, states = [], [], []
    for path in sorted((output / "jobs").glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        states.append({key: job.get(key) for key in ["model", "seed", "status", "error"]})
        if job["status"] == "PASS":
            metrics.extend(job["metrics"])
            candidates.extend(job["candidate_validation"])
    if metrics:
        atomic_csv(pd.DataFrame(metrics), output / "metrics.csv")
    if candidates:
        atomic_csv(pd.DataFrame(candidates), output / "candidate_validation.csv")
    dump(output / "status.json", {"jobs": states, "completed": sum(row["status"] == "PASS" for row in states),
                                  "failed": sum(row["status"] == "FAIL" for row in states), "updated_utc": now()})


def execute(run, base_job_path, config, output):
    sys.path.insert(0, str(SOURCE))
    from fairlearn import __version__
    from fairlearn.postprocessing import ThresholdOptimizer
    base_job = json.loads(base_job_path.read_text(encoding="utf-8"))
    model, seed = base_job["model"], int(base_job["seed"])
    name = f"{model}_s{seed}"
    path = output / "jobs" / f"{name}.json"
    validation_path = run / "predictions" / f"{name}_validation.parquet"
    test_path = run / "predictions" / f"{name}_test.parquet"
    code_hash = sha(Path(__file__))
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if (old.get("status") == "PASS" and old.get("code_sha256") == code_hash
            and old.get("validation_sha256") == sha(validation_path) and old.get("test_sha256") == sha(test_path)):
        print(f"Resume business Fairlearn: {name}", flush=True)
        return
    started = time.perf_counter()
    dump(path, {"status": "RUNNING", "model": model, "seed": seed, "started_utc": now()})
    try:
        validation = pd.read_parquet(validation_path)
        if not validation.SK_ID_CURR.is_unique:
            raise ValueError("Duplicate validation IDs")
        y = validation.TARGET.to_numpy(dtype=int)
        groups = validation.thin.to_numpy(dtype=int)
        favorable = (1 - validation.probability.to_numpy(dtype=float)).reshape(-1, 1)
        candidates, selected, processors = [], [], {}
        for tol in config.get("fairlearn_tolerances", [0, .01, .02, .05]):
            pid = f"fairlearn_business_tol{float(tol):g}".replace(".", "p")
            best = None
            for cost in COST_GRID:
                repeats = replication_indices(y, cost)
                processor = ThresholdOptimizer(estimator=FrozenFavorableScore().fit(), prefit=True,
                         predict_method="predict_proba", constraints="true_positive_rate_parity",
                         objective="accuracy_score", tol=float(tol), grid_size=1000, flip=False)
                processor.fit(favorable[repeats], 1 - y[repeats], sensitive_features=groups[repeats])
                q = processor._pmf_predict(favorable, sensitive_features=groups)[:, 1]
                metric = metrics_for_decisions(validation, q, config["primary_loss_ratio"], config)
                feasible = (metric["approvals"] >= config.get("minimum_approvals", 500)
                            and metric["risk"] is not None and metric["risk"] <= config["primary_risk_budget"] + 1e-10
                            and metric["abs_frr_gap"] <= float(tol) + 1e-8)
                candidate = {"model": model, "seed": seed, "policy_id": pid, "epsilon": float(tol),
                             "training_cost": cost, "replicated_records": len(repeats),
                             "business_feasible": feasible, **metric}
                candidates.append(candidate)
                rank = (metric["utility"], metric["approvals"], -(metric["risk"] if metric["risk"] is not None else 1.), -cost)
                if feasible and (best is None or rank > best[0]):
                    best = (rank, cost, metric, processor)
            rule = {"model": model, "seed": seed, "policy_id": pid, "family": "fairlearn_business",
                    "objective": "unit_utility", "utility_basis": "unit_utility",
                    "optimizer_objective": "accuracy_score_on_negative_replicated_validation",
                    "constraints": "true_positive_rate_parity", "epsilon": float(tol),
                    "risk_budget": config["primary_risk_budget"], "capital_fraction": 1.0,
                    "loss_ratio": config["primary_loss_ratio"], "noharm_delta": None,
                    "fit_split": "validation", "selected_cost": best[1] if best else None,
                    "status": "PASS" if best else "INFEASIBLE", "feasible": best is not None,
                    "library_version": __version__, "cost_grid": list(COST_GRID),
                    "validation_sha256": sha(validation_path), "frozen_utc": now(),
                    "validation_metrics": best[2] if best else None,
                    "selection_rule": "Max original-population validation utility at primary loss ratio; expected risk and minimum approvals constrained; tie approvals, lower risk, lower cost",
                    "decision_type": "stochastic_expectation",
                    "risk_interpretation": "Ratio of expected bad approvals to expected total approvals; not a per-randomization guarantee",
                    "interpolation": serialize_rule(best[3].interpolated_thresholder_.interpolation_dict) if best else None}
            dump(output / "policies" / f"{name}_{pid}.json", rule)
            selected.append(rule)
            if best:
                processors[pid] = best[3]
        # Every selection and interpolation above is frozen before loading test records.
        test = pd.read_parquet(test_path)
        if not test.SK_ID_CURR.is_unique or set(test.SK_ID_CURR) & set(validation.SK_ID_CURR):
            raise ValueError("Test IDs duplicated or overlapping validation IDs")
        rows = []
        for rule in selected:
            row = {key: value for key, value in rule.items() if key not in {"validation_metrics", "interpolation"}}
            row.update(method="fairlearn_business", evaluation_split="test", fit_status=rule["status"])
            if rule["feasible"]:
                q = processors[rule["policy_id"]]._pmf_predict((1 - test.probability.to_numpy()).reshape(-1, 1),
                                                              sensitive_features=test.thin.to_numpy())[:, 1]
                values = metrics_for_decisions(test, q, rule["loss_ratio"], config)
                row.update(values)
                row["test_risk_constraint_met"] = values["risk"] <= rule["risk_budget"] + 1e-10
                row["test_gap_constraint_met"] = values["abs_frr_gap"] <= rule["epsilon"] + 1e-8
                row["validation_risk"] = rule["validation_metrics"]["risk"]
                decisions = test[["SK_ID_CURR", "TARGET", "thin", "AMT_CREDIT", "probability"]].copy()
                decisions["approval_probability"] = q
                decisions.to_parquet(output / "decisions" / f"{name}_{rule['policy_id']}.parquet", index=False)
            rows.append(row)
        dump(path, {"model": model, "seed": seed, "status": "PASS", "completed_utc": now(),
                    "code_sha256": code_hash, "validation_sha256": sha(validation_path), "test_sha256": sha(test_path),
                    "wall_seconds": time.perf_counter() - started, "metrics": rows, "candidate_validation": candidates})
        count = sum(row["feasible"] for row in rows)
        print(f"PASS business Fairlearn {name}: {count}/{len(rows)} feasible tolerances; {time.perf_counter() - started:.1f}s", flush=True)
    except Exception as error:
        dump(path, {"model": model, "seed": seed, "status": "FAIL", "error": str(error),
                    "traceback": traceback.format_exc(), "completed_utc": now()})
        raise


def main(run, models=None, seeds=None):
    run = Path(run).resolve()
    config = json.loads((run / "config_frozen.json").read_text(encoding="utf-8"))
    output = run / "fairlearn_business"
    for folder in ["jobs", "policies", "decisions"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    dump(output / "replication_equivalence_checks.json", {"status": "PASS", "checks": equivalence_checks()})
    dump(output / "protocol.json", {"primary_loss_ratio": config["primary_loss_ratio"], "cost_grid": list(COST_GRID),
        "fairness_tolerances": config["fairlearn_tolerances"], "risk_budget": config["primary_risk_budget"],
        "validation_selection_only": True, "standard_algorithm": "Unmodified Fairlearn ThresholdOptimizer",
        "weighting_identity": "Expected weighted accuracy = (w*N_difficulty + good_approved - w*difficulty_approved) / (N_good+w*N_difficulty)",
        "conditional_fairness_identity": "Only TARGET=1 is replicated; favorable TARGET=0 TPR within each group is unchanged",
        "scope": "Cost-grid tuning under risk and minimum approvals; candidate-grid infeasibility is not global impossibility",
        "favorable_label": "1-TARGET", "config_sha256": sha(run / "config_frozen.json"), "code_sha256": sha(Path(__file__))})
    for path in sorted((run / "jobs").glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job["status"] != "PASS" or models and job["model"] not in models or seeds and int(job["seed"]) not in seeds:
            continue
        execute(run, path, config, output)
        aggregate(output)
    aggregate(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "results" / "run_001")
    parser.add_argument("--models", help="Comma-separated model filter")
    parser.add_argument("--seeds", help="Comma-separated seed filter")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({"status": "PASS", "checks": equivalence_checks()}))
    else:
        main(args.run, args.models.split(",") if args.models else None,
             [int(value) for value in args.seeds.split(",")] if args.seeds else None)
