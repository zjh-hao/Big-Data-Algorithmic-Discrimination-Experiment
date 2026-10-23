"""Durable V3.1 follow-up searches, fixed-model comparisons, and bias stress.

Every search uses only the original training and reference partitions.  The
winner is written to disk before calibration, validation policy selection, or
test evaluation.  Completed trials and model checkpoints survive queue retries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .common import HERE, ROOT, dump, load_config, now, sha
from .prepare import load_prepared


SEARCH_VERSION = "v3.1_reference_search_20261022_v1"
PHASES = ("search_lightgbm", "search_tabm", "fit_models", "bias_stress")
BIAS_ARMS = ("biased", "corrected", "opportunity1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lightgbm_candidates():
    return [dict(num_leaves=leaves, min_child_samples=child, learning_rate=rate,
                 n_estimators=2000, early_stopping_rounds=60)
            for leaves in (15, 31, 63) for child in (50, 150) for rate in (0.03, 0.05)]


def tabm_candidates(base):
    # Explicit, predeclared candidates rather than a test-guided search.
    definitions = [(128, 2, 16, .001), (128, 3, 16, .002),
                   (128, 4, 32, .001), (192, 2, 16, .002),
                   (192, 3, 32, .001), (192, 4, 16, .001),
                   (256, 2, 32, .002), (256, 3, 16, .001)]
    return [{**base, "d_block": width, "n_blocks": blocks, "k": k,
             "learning_rate": rate, "max_epochs": 120, "patience": 12,
             "batch_size": 512, "eval_batch_size": 1024}
            for width, blocks, k, rate in definitions]


def followup_config():
    cfg = copy.deepcopy(load_config())
    cfg.update(version="3.1.0", models=["lightgbm", "tabm"], seeds=[731, 732, 733], threads=4)
    cfg["followup_search"] = dict(
        version=SEARCH_VERSION, selection_split="reference", selection_metric="ROC AUC",
        search_seed=731, lightgbm=lightgbm_candidates(), tabm=tabm_candidates(cfg["tabm"]),
        bias_arms=list(BIAS_ARMS), bias_sampling_seeds=[17, 29, 43, 61, 89],
        bias_model_seed=731,
        note="Historical holdout has been inspected; all follow-up selection remains reference-only.")
    return cfg


def initialize(run):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    cfg = followup_config()
    identity = run / "config_frozen.json"
    if identity.exists():
        if read_json(identity) != cfg:
            raise ValueError("Frozen follow-up configuration differs; select a new run directory")
    else:
        dump(identity, cfg)
    for directory in ("jobs", "predictions", "policies", "checkpoints", "decisions", "tuning"):
        (run / directory).mkdir(exist_ok=True)
    data = load_prepared(cfg)
    split_copy = run / "split_manifest.parquet"
    if split_copy.exists():
        if sha(split_copy) != sha(HERE / "cache" / "split_manifest.parquet"):
            raise ValueError("Follow-up split manifest changed")
    else:
        shutil.copyfile(HERE / "cache" / "split_manifest.parquet", split_copy)
    audit = dict(**data[-1], split_manifest_sha256=sha(split_copy))
    if (run / "data_audit.json").exists():
        if read_json(run / "data_audit.json") != audit:
            raise ValueError("Follow-up prepared data audit changed")
    else:
        dump(run / "data_audit.json", audit)
    versions = {package: importlib.metadata.version(package) for package in
                ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch", "pyarrow")}
    if not (run / "runtime_versions.json").exists():
        dump(run / "runtime_versions.json", versions)
    return run, cfg, data


def fingerprint(name, cfg, run):
    from .advanced_models import SOURCE_ROOT
    source = SOURCE_ROOT / "tabm" / "tabm.py"
    inputs = dict(model=name, config=cfg, source_sha256=sha(__file__),
                  adapter_sha256=sha(HERE / "advanced_models.py"),
                  data_audit_sha256=sha(Path(run) / "data_audit.json"),
                  tabm_source_sha256=sha(source))
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def lgb_parameters(candidate, seed, threads):
    keys = {key: value for key, value in candidate.items() if key != "early_stopping_rounds"}
    return dict(**keys, max_depth=-1, subsample=.9, subsample_freq=1,
                colsample_bytree=.9, reg_lambda=5, n_jobs=min(4, int(threads)),
                verbosity=-1, random_state=int(seed), deterministic=True, force_col_wise=True)


def _fit_lgb(matrix, y, train, reference, candidate, cfg, seed, folder, weights=None):
    import lightgbm as lgb
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = lgb.LGBMClassifier(**lgb_parameters(candidate, seed, cfg["threads"]))
    model.fit(matrix[train], y[train], sample_weight=weights,
              eval_set=[(matrix[reference], y[reference])], eval_metric="auc",
              callbacks=[lgb.early_stopping(int(candidate["early_stopping_rounds"]), verbose=False)])
    reference_auc = float(roc_auc_score(y[reference], model.predict_proba(matrix[reference])[:, 1]))
    model.booster_.save_model(str(folder / "model.txt"))
    metadata = dict(model="lightgbm", seed=int(seed), train_seconds=time.perf_counter() - started,
                    training_seconds=time.perf_counter() - started, reference_auc=reference_auc,
                    parameters=candidate, best_iteration=int(model.best_iteration_),
                    input_features=int(matrix.shape[1]), train_n=len(train), selection_split="reference",
                    library_version=importlib.metadata.version("lightgbm"), model_sha256=sha(folder / "model.txt"))
    dump(folder / "metadata.json", metadata)
    return model, lambda value: model.predict_proba(value)[:, 1], metadata


def _load_lgb(folder):
    import lightgbm as lgb
    folder = Path(folder)
    metadata = read_json(folder / "metadata.json")
    if metadata["model_sha256"] != sha(folder / "model.txt"):
        raise ValueError("LightGBM checkpoint checksum mismatch")
    model = lgb.Booster(model_file=str(folder / "model.txt"), params={"num_threads": 4})
    return model, lambda value: model.predict(value, num_threads=4), metadata


def checkpoint_hashes(name, folder):
    folder = Path(folder)
    names = ["model.txt", "metadata.json"] if name == "lightgbm" else [
        "best_checkpoint.pt", "preprocessor.joblib", "metadata.json"]
    return {item: sha(folder / item) for item in names}


def verify_checkpoint(name, folder, expected):
    if checkpoint_hashes(name, folder) != expected:
        raise ValueError(f"{name} persisted checkpoint differs from completed trial")


def search(name, data, cfg, run):
    X, dense, records, split, _ = data
    matrix = dense if name == "tabm" else X
    y = records.TARGET.to_numpy()
    seed = cfg["followup_search"]["search_seed"]
    identity = fingerprint(name, cfg, run)
    selected_path = run / "tuning" / f"{name}_selection.json"
    candidates = cfg["followup_search"][name]
    trials = []
    if selected_path.exists():
        frozen = read_json(selected_path)
        if frozen["input_fingerprint"] != identity:
            raise ValueError("Frozen reference selection input changed")
        if len(frozen["trials"]) != len(candidates):
            raise ValueError("Incomplete frozen reference search")
        for trial in frozen["trials"]:
            verify_checkpoint(name, trial["artifact"], trial["checkpoint_sha256"])
        print("Resume completed reference search", name, flush=True)
        return frozen
    failures = []
    for index, candidate in enumerate(candidates):
        folder = run / "tuning" / name / f"trial_{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "status.json"
        state = read_json(path) if path.exists() else {}
        if state.get("status") == "PASS":
            if state.get("input_fingerprint") != identity or state.get("parameters") != candidate:
                raise ValueError("Completed trial fingerprint differs")
            verify_checkpoint(name, folder, state["checkpoint_sha256"])
            print("Resume completed", name, index, flush=True)
            trials.append(state)
            continue
        dump(path, dict(status="RUNNING", input_fingerprint=identity, parameters=candidate,
                        started_utc=now(), artifact=str(folder), seed=seed))
        try:
            print(f"Reference-only search {name} {index+1}/{len(candidates)}: {candidate}", flush=True)
            if name == "lightgbm":
                model, predict, metadata = _fit_lgb(matrix, y, split["train"], split["reference"],
                                                    candidate, cfg, seed, folder)
            else:
                from .advanced_models import train_tabm
                model, predict, metadata = train_tabm(matrix[split["train"]], y[split["train"]],
                    matrix[split["reference"]], y[split["reference"]],
                    {**cfg, "tabm": candidate}, seed, folder)
            score = float(metadata.get("reference_auc", metadata.get("best_stop_auc")))
            state = dict(status="PASS", index=index, model=name, seed=seed,
                         parameters=candidate, reference_auc=score, artifact=str(folder),
                         input_fingerprint=identity, completed_utc=now(),
                         checkpoint_sha256=checkpoint_hashes(name, folder), metadata=metadata)
            dump(path, state)
            trials.append(state)
            del model, predict
            if name == "tabm":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            state = dict(status="FAIL", index=index, model=name, seed=seed, parameters=candidate,
                         input_fingerprint=identity, artifact=str(folder), error=str(exc),
                         traceback=traceback.format_exc(), completed_utc=now())
            dump(path, state)
            failures.append(state)
            traceback.print_exc()
    if failures:
        raise RuntimeError(f"{len(failures)} {name} search trials failed; completed trials retained for retry")
    winner = max(trials, key=lambda row: (row["reference_auc"], -row["index"]))
    selected = dict(model=name, seed=seed, selection_split="reference", selection_metric="ROC AUC",
                    parameters=winner["parameters"], best_reference_auc=winner["reference_auc"],
                    selected_artifact=winner["artifact"], selected_index=winner["index"],
                    trials=trials, frozen_utc=now(), input_fingerprint=identity,
                    followup_source_sha256=sha(__file__), adapter_sha256=sha(HERE / "advanced_models.py"),
                    test_used_for_selection=False)
    dump(selected_path, selected)
    print(f"FROZEN {name}: reference AUC={winner['reference_auc']:.6f}", flush=True)
    return selected


def selected_configuration(name, cfg, run):
    path = Path(run) / "tuning" / f"{name}_selection.json"
    if not path.exists():
        raise ValueError(f"Run --phase search_{name} before model evaluation")
    selected = read_json(path)
    if selected["input_fingerprint"] != fingerprint(name, cfg, run):
        raise ValueError("Reference search configuration or source changed")
    return selected


def training_override(run):
    """Return the callback accepted by run.execute(train_override=...)."""
    run = Path(run)
    def train(model_name, seed, data, cfg, artifact):
        X, dense, records, split, _ = data
        selected = selected_configuration(model_name, cfg, run)
        matrix = dense if model_name == "tabm" else X
        y = records.TARGET.to_numpy()
        artifact = Path(artifact)
        local = artifact / "selected_training"
        local.mkdir(parents=True, exist_ok=True)
        guard = dict(model=model_name, seed=int(seed),
                     selection_sha256=sha(run / "tuning" / f"{model_name}_selection.json"),
                     input_fingerprint=selected["input_fingerprint"])
        status_path = local / "completed.json"
        if status_path.exists():
            status = read_json(status_path)
            if status["guard"] != guard:
                raise ValueError("Selected training checkpoint guard differs")
            source = Path(status["artifact"])
            verify_checkpoint(model_name, source, status["checkpoint_sha256"])
        else:
            if seed == selected["seed"]:
                source = Path(selected["selected_artifact"])
            else:
                source = local
                if model_name == "lightgbm":
                    obj, fn, _ = _fit_lgb(matrix, y, split["train"], split["reference"],
                                         selected["parameters"], cfg, seed, local)
                else:
                    from .advanced_models import train_tabm
                    obj, fn, _ = train_tabm(matrix[split["train"]], y[split["train"]],
                        matrix[split["reference"]], y[split["reference"]],
                        {**cfg, "tabm": selected["parameters"]}, seed, local)
                del obj, fn
            status = dict(status="PASS", guard=guard, artifact=str(source),
                          checkpoint_sha256=checkpoint_hashes(model_name, source), completed_utc=now())
            dump(status_path, status)
        if model_name == "lightgbm":
            obj, fn, metadata = _load_lgb(source)
        else:
            from .advanced_search import _load_tabm
            obj, fn, metadata = _load_tabm(source)
        metadata = dict(metadata, model=model_name, seed=int(seed),
                        selected_configuration=selected, checkpoint_artifact=str(source),
                        selection_sha256=guard["selection_sha256"], input_features=int(matrix.shape[1]))
        metadata.setdefault("reference_auc", metadata.get("best_stop_auc"))
        metadata.setdefault("train_seconds", metadata.get("training_seconds"))
        return obj, fn, metadata, matrix
    return train


def fit_models(data, cfg, run):
    from .run import execute, aggregate
    callback = training_override(run)
    # Force both winners to exist before any test score is evaluated.
    for name in cfg["models"]:
        selected_configuration(name, cfg, run)
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            execute(name, seed, data, cfg, run, train_override=callback)
    aggregate(run)
    expected = len(cfg["models"]) * len(cfg["seeds"])
    completed = [read_json(path) for path in (run / "jobs").glob("*.json")]
    passed = sum(row.get("status") == "PASS" for row in completed)
    if passed != expected:
        raise ValueError(f"Expected {expected} model jobs, found {passed} passes")
    dump(run / "fit_models_completion.json", dict(status="PASS", expected=expected,
         completed=passed, completed_utc=now(), followup_source_sha256=sha(__file__)))


def bias_stress(data, cfg, run):
    from .run import execute, aggregate
    from .mechanisms import sample_arms
    from .training_objective import load_original_loss_weights, validate_loss_equivalence
    selected = selected_configuration("lightgbm", cfg, run)
    loss_weights, loss_source = load_original_loss_weights()
    validate_loss_equivalence(loss_weights)
    output = run / "bias_stress"
    for sub in ("jobs", "predictions", "policies", "checkpoints", "decisions"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    if not (output / "config_frozen.json").exists():
        dump(output / "config_frozen.json", cfg)
    elif read_json(output / "config_frozen.json") != cfg:
        raise ValueError("Bias stress configuration differs")
    for name in ("split_manifest.parquet", "data_audit.json"):
        if not (output / name).exists():
            shutil.copyfile(run / name, output / name)
    X, _, records, split, _ = data
    y, group = records.TARGET.to_numpy(), records.thin.to_numpy()
    tr, ref = split["train"], split["reference"]
    design = dict(selection_sha256=sha(run / "tuning" / "lightgbm_selection.json"),
        model_seed=cfg["followup_search"]["bias_model_seed"],
        sampling_seeds=cfg["followup_search"]["bias_sampling_seeds"], arms=list(BIAS_ARMS),
        loss_source_sha256=sha(loss_source), mechanisms_sha256=sha(HERE / "mechanisms.py"),
        followup_sha256=sha(__file__), retention=cfg["sampling_retention"],
        notes="All arms share selected LightGBM hyperparameters, calibration split, validation policies and test rows; seed field denotes sampling seed, not model initialization.",
        opportunity_formula="Exact audit_extension loss_weights(y,group,1,1), normalized to mean one")
    if (output / "design.json").exists() and read_json(output / "design.json") != design:
        raise ValueError("Bias stress design changed")
    dump(output / "design.json", design)

    def callback(model_name, sampling_seed, data_, cfg_, artifact):
        condition = model_name.removeprefix("lightgbm_")
        arms, notes = sample_arms(y[tr], group[tr], y[ref], group[ref],
                                 sampling_seed, cfg_["sampling_retention"])
        original_arm = "corrected_referenceweights" if condition == "corrected" else "biased_drop_thin_good"
        relative, weights = arms[original_arm]
        if condition == "opportunity1":
            weights = loss_weights(y[tr][relative], group[tr][relative], 1, 1.0)
        actual = tr[relative]
        artifact = Path(artifact)
        folder = artifact / "training"
        folder.mkdir(parents=True, exist_ok=True)
        guard = dict(design=design, arm=condition, sampling_seed=int(sampling_seed))
        saved = folder / "completed.json"
        if saved.exists():
            prior = read_json(saved)
            if prior["guard"] != guard:
                raise ValueError("Bias checkpoint design differs")
            verify_checkpoint("lightgbm", folder, prior["checkpoint_sha256"])
            obj, predict, meta = _load_lgb(folder)
        else:
            samples = records.iloc[actual][["SK_ID_CURR", "TARGET", "thin"]].copy()
            samples["row_index"] = actual
            samples["sample_weight"] = weights
            samples.to_parquet(folder / "training_samples.parquet", index=False)
            obj, predict, meta = _fit_lgb(X, y, actual, ref, selected["parameters"], cfg_,
                                        design["model_seed"], folder, weights=weights)
            dump(saved, dict(status="PASS", guard=guard,
                             checkpoint_sha256=checkpoint_hashes("lightgbm", folder), completed_utc=now()))
        cells = []
        for g in (0, 1):
            for label in (0, 1):
                mask = (group[actual] == g) & (y[actual] == label)
                cells.append(dict(group=g, TARGET=label, count=int(mask.sum()), weight_sum=float(weights[mask].sum())))
        meta.update(model=model_name, model_seed=design["model_seed"], sampling_seed=sampling_seed,
                    condition=condition, training_cells=cells, sample_design=notes,
                    training_samples_sha256=sha(folder / "training_samples.parquet"),
                    selected_parameters=selected["parameters"], loss_source_sha256=sha(loss_source))
        return obj, predict, meta, X

    for seed in design["sampling_seeds"]:
        for condition in BIAS_ARMS:
            execute(f"lightgbm_{condition}", seed, data, cfg, output, train_override=callback)
    aggregate(output)
    expected = len(design["sampling_seeds"]) * len(BIAS_ARMS)
    jobs = [read_json(path) for path in (output / "jobs").glob("*.json")]
    passed = sum(row.get("status") == "PASS" for row in jobs)
    if passed != expected:
        raise ValueError(f"Expected {expected} bias stress jobs, found {passed} passes")
    if (output / "policy_metrics.csv").exists():
        metrics = pd.read_csv(output / "policy_metrics.csv")
        numeric = [key for key in ("thin_frr", "thick_frr", "abs_frr_gap", "utility", "amount_utility", "risk", "approval_rate") if key in metrics]
        summary = metrics.groupby(["model", "policy_id"])[numeric].agg(["mean", "std", "count"])
        summary.columns = ["_".join(pair) for pair in summary.columns]
        summary.to_csv(output / "summary.csv")
    dump(output / "completion.json", dict(status="PASS", expected=expected, completed=passed,
         completed_utc=now(), design_sha256=sha(output / "design.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--run", default=str(HERE / "results" / "run_002"))
    args = parser.parse_args()
    run, cfg, data = initialize(args.run)
    with threadpool_limits(limits=4):
        if args.phase.startswith("search_"):
            search(args.phase.removeprefix("search_"), data, cfg, run)
        elif args.phase == "fit_models":
            fit_models(data, cfg, run)
        else:
            bias_stress(data, cfg, run)
    dump(run / f"phase_{args.phase}_completion.json", dict(status="PASS", phase=args.phase,
         completed_utc=now(), followup_sha256=sha(__file__)))


if __name__ == "__main__":
    main()
