"""Controlled sampling experiments: sample count versus within-group label bias.

Run: .venv-v3/Scripts/python.exe -m research_extension_v3.mechanisms
Every arm uses the same frozen split, features, XGBoost parameters, calibration,
and validation-only common-threshold selection. Test labels never select a model
or a threshold. TARGET is repayment difficulty; utility is a scenario measure.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .business import metrics_for_decisions
from .common import HERE, ROOT, dump, load_config, now, sha
from .prepare import load_prepared
from src.experiment import biased_indices, correction_weights, choose_threshold


CONDITIONS = (
    "full",
    "random_same_total",
    "group_count_matched_preserve_label_mix",
    "biased_drop_thin_good",
    "corrected_referenceweights",
)


def sample_arms(y, group, ref_y, ref_group, seed, retention=0.5):
    """Return training-relative indexes and weights for all controlled arms."""
    y, group = np.asarray(y), np.asarray(group)
    biased = biased_indices(y, group, retention, seed)
    random = np.sort(np.random.default_rng(seed + 10000).choice(len(y), len(biased), replace=False))
    matched_parts, matched_cells = [], []
    rng = np.random.default_rng(seed + 20000)
    for g in (0, 1):
        original = np.flatnonzero(group == g)
        n = int(np.sum(group[biased] == g))
        n_bad = int(np.rint(n * np.mean(y[original])))
        n_bad = max(n - int(np.sum(y[original] == 0)), min(n_bad, int(np.sum(y[original] == 1))))
        for label, count in ((0, n - n_bad), (1, n_bad)):
            eligible = np.flatnonzero((group == g) & (y == label))
            chosen = eligible if count == len(eligible) else rng.choice(eligible, count, replace=False)
            matched_parts.append(chosen)
            matched_cells.append(dict(group=g, label=label, count=count))
    matched = np.sort(np.concatenate(matched_parts))
    weights, cell_weights = correction_weights(y[biased], group[biased], ref_y, ref_group)
    arms = {
        "full": (np.arange(len(y)), np.ones(len(y))),
        "random_same_total": (random, np.ones(len(random))),
        "group_count_matched_preserve_label_mix": (matched, np.ones(len(matched))),
        "biased_drop_thin_good": (biased, np.ones(len(biased))),
        "corrected_referenceweights": (biased.copy(), weights),
    }
    # These assertions encode the experiment, rather than its expected outcome.
    assert len(random) == len(matched) == len(biased)
    assert np.array_equal(np.bincount(group[matched], minlength=2), np.bincount(group[biased], minlength=2))
    for g in (0, 1):
        actual = np.sum((group[matched] == g) & (y[matched] == 1))
        expected = np.sum(group[matched] == g) * np.mean(y[group == g])
        assert abs(actual - expected) <= 0.5000001
    return arms, {"matched_cells": matched_cells, "reference_cell_weights_thick_good_bad_thin_good_bad": cell_weights.tolist()}


def model_parameters(threads=4):
    return dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                tree_method="hist", device="cpu", max_bin=128,
                n_jobs=threads, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=5, objective="binary:logistic", eval_metric="auc",
                random_state=731)


def execute(condition, seed, indexes, weights, sample_notes, data, cfg, output, main_run):
    X, _, records, split, info = data
    name = f"{condition}_sampling{seed}"
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    job_path = folder / "status.json"
    if job_path.exists():
        previous = json.loads(job_path.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS":
            if previous.get("code_sha256") != sha(__file__):
                raise ValueError(f"Code changed after completed job {name}; choose a new output folder")
            print("Resume", name, flush=True)
            return previous["metrics"]
    started = time.perf_counter()
    code_sha = sha(__file__)
    dump(job_path, dict(status="RUNNING", name=name, started_utc=now(), code_sha256=code_sha))
    try:
        train = split["train"]
        selected = train[indexes]
        y, g = records.TARGET.to_numpy(), records.thin.to_numpy()
        samples = records.iloc[selected][["SK_ID_CURR", "TARGET", "thin"]].copy()
        samples["row_index"] = selected
        samples["sample_weight"] = weights
        samples.to_parquet(folder / "training_samples.parquet", index=False)
        cells = []
        for group in (0, 1):
            for label in (0, 1):
                mask = (g[selected] == group) & (y[selected] == label)
                cells.append(dict(group=group, TARGET=label, count=int(mask.sum()), weight_sum=float(weights[mask].sum())))
        params = model_parameters(min(4, int(cfg["threads"])))
        checkpoint = folder / "model.ubj"
        reused = None
        main_checkpoint = main_run / "checkpoints" / "xgboost_s731" / "basic.joblib"
        if checkpoint.exists():
            model = XGBClassifier(**params)
            model.load_model(checkpoint)
            reused = str(checkpoint)
        elif condition == "full" and main_checkpoint.exists():
            saved = joblib.load(main_checkpoint)
            model = saved["model"]
            expected = params
            actual = model.get_params()
            mismatch = {key: [actual.get(key), value] for key, value in expected.items() if actual.get(key) != value}
            if mismatch:
                raise ValueError("Full checkpoint parameters differ: " + str(mismatch))
            reused = str(main_checkpoint)
            model.save_model(checkpoint)
        else:
            model = XGBClassifier(**params)
            model.fit(X[selected], y[selected], sample_weight=weights)
            model.save_model(checkpoint)

        calibration = split["calibration"]
        raw_calibration = model.predict_proba(X[calibration])[:, 1]
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(
            logit(np.clip(raw_calibration, 1e-7, 1 - 1e-7)).reshape(-1, 1), y[calibration])
        slope, intercept = float(platt.coef_[0, 0]), float(platt.intercept_[0])
        if slope <= 0:
            raise ValueError("Nonpositive Platt slope")

        def make_scores(which):
            idx = split[which]
            frame = records.iloc[idx].reset_index(drop=True).copy()
            frame["raw_probability"] = model.predict_proba(X[idx])[:, 1]
            frame["probability"] = expit(slope * logit(np.clip(frame.raw_probability.to_numpy(), 1e-7, 1 - 1e-7)) + intercept)
            return frame

        val = make_scores("validation")
        frozen = choose_threshold(val.probability.to_numpy(), val.TARGET.to_numpy(),
                                  cfg["primary_risk_budget"], cfg["minimum_approvals"])
        if not frozen["feasible"]:
            raise ValueError("No validation threshold meets risk/minimum-approval conditions")
        frozen.update(dict(condition=condition, sampling_seed=seed, model_seed=731,
                           calibration=dict(slope=slope, intercept=intercept),
                           selected_on="validation only", frozen_utc=now(), code_sha256=code_sha,
                           objective="maximum approval count at common risk budget; ties kept complete",
                           risk_budget=cfg["primary_risk_budget"], minimum_approvals=cfg["minimum_approvals"]))
        dump(folder / "frozen_policy.json", frozen)
        # The policy is persisted before the first test prediction/evaluation.
        test = make_scores("test")
        val["approved"] = val.probability <= frozen["threshold"]
        test["approved"] = test.probability <= frozen["threshold"]
        val.to_parquet(folder / "validation_predictions_decisions.parquet", index=False)
        test.to_parquet(folder / "test_predictions_decisions.parquet", index=False)
        calibration_frame = records.iloc[calibration][["SK_ID_CURR", "TARGET", "thin"]].copy()
        calibration_frame["raw_probability"] = raw_calibration
        calibration_frame["probability"] = expit(slope * logit(np.clip(raw_calibration, 1e-7, 1 - 1e-7)) + intercept)
        calibration_frame.to_parquet(folder / "calibration_predictions.parquet", index=False)
        business = metrics_for_decisions(test, test.approved.to_numpy(), cfg["primary_loss_ratio"], cfg)
        metrics = dict(condition=condition, sampling_seed=seed, model_seed=731,
                       training_records=len(selected), train_thin_count=int(np.sum(g[selected] == 1)),
                       train_thick_count=int(np.sum(g[selected] == 0)),
                       train_thin_difficulty_rate=float(np.mean(y[selected][g[selected] == 1])),
                       train_thick_difficulty_rate=float(np.mean(y[selected][g[selected] == 0])),
                       auc=float(roc_auc_score(test.TARGET, test.probability)),
                       average_precision=float(average_precision_score(test.TARGET, test.probability)),
                       brier=float(brier_score_loss(test.TARGET, test.probability)),
                       validation_threshold=frozen["threshold"],
                       validation_approvals=frozen["validation_approvals"],
                       validation_risk=frozen["validation_risk"], **business)
        metrics["signed_frr_gap"] = metrics["thin_frr"] - metrics["thick_frr"]
        metrics["utility5"] = metrics["good_approved"] - 5 * metrics["difficulty_count"]
        metadata = dict(status="PASS", name=name, completed_utc=now(),
                        wall_seconds=time.perf_counter() - started, code_sha256=code_sha,
                        source_helper_sha256=sha(ROOT / "src" / "experiment.py"),
                        business_code_sha256=sha(HERE / "business.py"),
                        split_manifest_sha256=info["cache_hashes"]["split_manifest.parquet"],
                        model_parameters=params, reference_weights_assumption="independent representative labeled reference split",
                        reused_model=reused, training_cells=cells, sample_design=sample_notes,
                        metrics=metrics,
                        artifact_sha256={path.name: sha(path) for path in folder.iterdir()
                                         if path.is_file() and path.name != "status.json"})
        dump(job_path, metadata)
        print(f"PASS {name}: thin FRR={metrics['thin_frr']:.4f}, thick FRR={metrics['thick_frr']:.4f}, AUC={metrics['auc']:.5f}, utility5={metrics['utility5']:.0f}", flush=True)
        return metrics
    except Exception as exc:
        dump(job_path, dict(status="FAIL", name=name, error=str(exc), traceback=traceback.format_exc(),
                            completed_utc=now(), code_sha256=code_sha))
        raise


def summarize(output):
    rows = []
    for path in output.glob("*/status.json"):
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("status") == "PASS":
            rows.append(status["metrics"])
    if not rows:
        return
    metrics = pd.DataFrame(rows).sort_values(["condition", "sampling_seed"])
    metrics.to_csv(output / "mechanisms_metrics.csv", index=False)
    columns = ["training_records", "train_thin_count", "train_thin_difficulty_rate", "auc",
               "thin_frr", "thick_frr", "abs_frr_gap", "signed_frr_gap", "approval_rate",
               "approvals", "good_approved", "difficulty_count", "risk", "utility5", "amount_utility"]
    summary = metrics.groupby("condition")[columns].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reindex([c for c in CONDITIONS if c in summary.index])
    summary.to_csv(output / "summary.csv")
    full = metrics.loc[metrics.condition == "full"].iloc[0]
    comparisons = []
    for condition in CONDITIONS:
        arm = metrics.loc[metrics.condition == condition]
        if len(arm):
            comparisons.append(dict(condition=condition, n_runs=len(arm),
                thin_frr_change_pp=float((arm.thin_frr.mean() - full.thin_frr) * 100),
                thick_frr_change_pp=float((arm.thick_frr.mean() - full.thick_frr) * 100),
                gap_change_pp=float((arm.abs_frr_gap.mean() - full.abs_frr_gap) * 100),
                utility5_change=float(arm.utility5.mean() - full.utility5),
                auc_change=float(arm.auc.mean() - full.auc)))
    pd.DataFrame(comparisons).to_csv(output / "changes_vs_full.csv", index=False)
    make_figure(summary, output)


def make_figure(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"full": "Full data", "random_same_total": "Random same N",
              "group_count_matched_preserve_label_mix": "Matched group N\nOriginal label mix",
              "biased_drop_thin_good": "Drop thin good",
              "corrected_referenceweights": "Reference\nreweighting"}
    names = list(summary.index)
    locations = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    for label, key, offset, color in [("Thin", "thin_frr", -0.18, "#e07a38"),
                                     ("Thick", "thick_frr", 0.18, "#3574a8")]:
        ax.bar(locations + offset, summary[f"{key}_mean"] * 100, 0.36,
               yerr=summary[f"{key}_std"].fillna(0) * 100, label=label, color=color, capsize=3)
    ax.set_ylabel("Normal-label false rejection (%)")
    ax.legend(frameon=False)
    for ax, key, title, factor in [(axes[0, 1], "abs_frr_gap", "Absolute FRR gap (percentage points)", 100),
                                  (axes[1, 0], "utility5", "Scenario utility: good approvals - 5 x difficulty approvals", 1),
                                  (axes[1, 1], "auc", "Test ROC AUC", 1)]:
        ax.bar(locations, summary[f"{key}_mean"] * factor,
               yerr=summary[f"{key}_std"].fillna(0) * factor,
               color=["#708090", "#7897b6", "#5c9d8b", "#c65e57", "#986cac"][:len(names)], capsize=3)
        ax.set_ylabel(title)
        if key == "auc":
            ax.set_ylim(max(0.5, float(summary[f"{key}_mean"].min()) - 0.015),
                        min(1.0, float(summary[f"{key}_mean"].max()) + 0.01))
    for ax in axes.flat:
        ax.set_xticks(locations, [labels[n] for n in names], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle("Controlled sampling: data volume, label selection, and reference correction", fontsize=15)
    fig.savefig(output / "controlled_sampling_mechanisms.png", dpi=180)
    fig.savefig(output / "controlled_sampling_mechanisms.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "results" / "run_001" / "mechanisms"))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        summarize(output)
        return
    cfg = load_config()
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else cfg["sampling_seeds"]
    conditions = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    if set(conditions).difference(CONDITIONS):
        raise ValueError("Unknown sampling condition")
    data = load_prepared(cfg)
    records, split = data[2], data[3]
    train, reference = split["train"], split["reference"]
    y, g = records.TARGET.to_numpy(), records.thin.to_numpy()
    dump(output / "design.json", dict(created_utc=now(), code_sha256=sha(__file__),
         conditions=conditions, sampling_seeds=seeds, model_seed=731,
         model_parameters=model_parameters(min(4, cfg["threads"])),
         retention=cfg["sampling_retention"], risk_budget=cfg["primary_risk_budget"],
         favorable_label="TARGET=0", mechanism="only training membership/weights change; shared feature encoder and splits",
         provenance="All preprocessing fitted on original training split; test holdout previously inspected historically; no fresh-blind claim",
         versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost"]}))
    with threadpool_limits(limits=min(4, cfg["threads"])):
        if "full" in conditions:
            execute("full", 0, np.arange(len(train)), np.ones(len(train)), {}, data, cfg, output, output.parent)
            summarize(output)
        for seed in seeds:
            arms, notes = sample_arms(y[train], g[train], y[reference], g[reference], seed, cfg["sampling_retention"])
            for condition in conditions:
                if condition == "full":
                    continue
                selected, weights = arms[condition]
                print("Starting", condition, "sampling seed", seed, flush=True)
                execute(condition, seed, selected, weights, notes, data, cfg, output, output.parent)
                summarize(output)
    dump(output / "completion.json", dict(status="PASS", completed_utc=now(),
         expected_jobs=int("full" in conditions) + len(seeds) * len([c for c in conditions if c != "full"]),
         completed_jobs=len(pd.read_csv(output / "mechanisms_metrics.csv")), code_sha256=sha(__file__)))


if __name__ == "__main__":
    main()
