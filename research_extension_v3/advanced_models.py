"""Official TabM and FAccT 2025 M2FGB adapters for the V3 experiments.

All public prediction functions return P(TARGET=1), i.e. repayment difficulty.
The official source snapshots are read directly; their algorithms are not replaced
by similarly named local implementations. See each run's provenance metadata.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import random
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


from .common import BASELINE_SOURCES

SOURCE_ROOT = BASELINE_SOURCES


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _section(config: dict, name: str) -> dict:
    return dict(config.get(name, config))


def _source_metadata(name: str) -> dict:
    manifest = SOURCE_ROOT / "sources.json"
    if manifest.is_file():
        for item in json.loads(manifest.read_text(encoding="utf-8"))["sources"]:
            if item["name"] == name:
                return {key: item[key] for key in ("repository", "branch", "commit")}
    return {"source_name": name}


def _load_tabm():
    """Import the official single-file package, rather than a package lookalike."""
    source = SOURCE_ROOT / "tabm" / "tabm.py"
    name = "_credit_fairness_official_tabm"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name], source


def train_tabm(
    X_train, y_train, X_stop, y_stop, config: dict, seed: int, artifact_dir
) -> tuple[Any, Callable, dict]:
    """Train official TabM; reference/stop data alone select its checkpoint.

    Dense arrays must already be imputed by the caller. Scaling is fitted on the
    training matrix only. Each ensemble member receives its own BCE loss; only
    inference averages member probabilities, following the official instructions.
    """
    import torch
    import torch.nn.functional as F
    import joblib
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    cfg = _section(config, "tabm")
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("threads", cfg.get("threads", 4))))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_name = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    # TF32 is an arithmetic speed setting, not mixed-precision loss scaling.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_stop = np.asarray(X_stop, dtype=np.float32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_stop).all():
        raise ValueError("TabM input must be finite; use training-fitted imputation first")
    y_train = np.asarray(y_train, dtype=np.int64)
    y_stop = np.asarray(y_stop, dtype=np.int64)
    transform_name = cfg.get("normalization", "quantile")
    if transform_name == "quantile":
        # Rank-normalize continuous variables as in the official end-to-end
        # example; preserve one-hot and missing-indicator columns as binary.
        minima, maxima = X_train.min(axis=0), X_train.max(axis=0)
        continuous = np.array([
            not np.all((X_train[:, j] == minima[j]) | (X_train[:, j] == maxima[j]))
            for j in range(X_train.shape[1])
        ], dtype=bool)
        scaler = QuantileTransformer(
            n_quantiles=min(int(cfg.get("n_quantiles", 1000)), len(X_train)),
            output_distribution="normal", random_state=seed,
            subsample=min(int(cfg.get("quantile_subsample", 100000)), len(X_train)),
        )
        if continuous.any():
            scaler.fit(X_train[:, continuous])
        clip = None

        def transform(X):
            value = np.array(X, dtype=np.float32, copy=True)
            if continuous.any():
                value[:, continuous] = scaler.transform(value[:, continuous])
            return value

        normalization_description = "train-only QuantileTransformer(normal), continuous columns only; one-hot/binary indicators preserved"
    elif transform_name == "standard":
        continuous = np.ones(X_train.shape[1], dtype=bool)
        scaler = StandardScaler().fit(X_train)
        clip = float(cfg.get("standardized_clip", 10.0))

        def transform(X):
            value = scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32, copy=False)
            return np.clip(value, -clip, clip)

        normalization_description = "train-only StandardScaler; standardized values clipped to +/-" + str(clip)
    else:
        raise ValueError("TabM normalization must be quantile or standard")
    x_train, x_stop = transform(X_train), transform(X_stop)
    preprocessor = {"normalization": transform_name, "transformer": scaler,
                    "continuous_columns": continuous, "clip": clip}
    joblib.dump(preprocessor, out / "preprocessor.joblib")

    official, source = _load_tabm()
    model_params = {
        "n_num_features": x_train.shape[1],
        "d_out": 1,
        "k": int(cfg.get("k", 16)),
        "n_blocks": int(cfg.get("n_blocks", 2)),
        "d_block": int(cfg.get("d_block", 128)),
        "dropout": float(cfg.get("dropout", 0.1)),
        "arch_type": cfg.get("arch_type", "tabm"),
    }
    model = official.TabM.make(**model_params).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 0.002)),
        weight_decay=float(cfg.get("weight_decay", 0.0003)),
    )
    batch_size = int(cfg.get("batch_size", 512))
    eval_batch_size = int(cfg.get("eval_batch_size", 2048))
    # Keep the modest Home Credit design matrices on the GPU to avoid repeated
    # host-to-device copies. A 4060 can hold the matrices and this configuration.
    train_tensor = torch.as_tensor(x_train, device=device)
    stop_tensor = torch.as_tensor(x_stop, device=device)
    target_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    def predict_tensor(tensor):
        model.eval()
        values = []
        with torch.inference_mode():
            for batch in tensor.split(eval_batch_size):
                # Mean of probabilities, rather than sigmoid(mean logits).
                probability = model(batch).squeeze(-1).sigmoid().mean(dim=1)
                values.append(probability.detach().cpu().numpy())
        return np.concatenate(values) if values else np.empty(0, dtype=np.float32)

    history = []
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    patience = int(cfg.get("patience", 8))
    epochs = int(cfg.get("max_epochs", 60))
    started = time.perf_counter()
    history_file = out / "training_history.jsonl"
    with history_file.open("w", encoding="utf-8") as stream:
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            permutation = torch.randperm(len(train_tensor), device=device)
            loss_sum = 0.0
            for indices in permutation.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_tensor[indices]).squeeze(-1)
                targets = target_tensor[indices, None].expand_as(logits)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
            probability = predict_tensor(stop_tensor)
            auc = float(roc_auc_score(y_stop, probability))
            record = {
                "epoch": epoch,
                "train_member_bce": loss_sum / len(train_tensor),
                "stop_auc": auc,
                "stop_logloss": float(log_loss(y_stop, probability)),
                "seconds": time.perf_counter() - epoch_started,
            }
            history.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"TabM seed={seed} epoch={epoch} stop_auc={auc:.6f}", flush=True)
            if auc > best_auc + float(cfg.get("min_delta", 1e-5)):
                best_auc, best_epoch = auc, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({"state_dict": best_state, "model_params": model_params,
                            "seed": seed, "epoch": epoch}, out / "best_checkpoint.pt")
            if epoch - best_epoch >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict_fn(X):
        transformed = transform(X)
        # Predict in chunks so very large user matrices need not fit on the GPU.
        output = []
        for begin in range(0, len(transformed), eval_batch_size):
            tensor = torch.as_tensor(transformed[begin:begin + eval_batch_size], device=device)
            output.append(predict_tensor(tensor))
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    metadata = {
        "algorithm": "TabM", "official_source": _source_metadata("tabm"),
        "source_file": str(source), "source_sha256": _sha256(source),
        "package_version": official.__version__, "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "seed": seed, "parameters": model_params, "training_config": cfg,
        "preprocessing": "caller-imputed dense inputs; " + normalization_description,
        "rank_normalized_columns": int(continuous.sum()) if transform_name == "quantile" else 0,
        "label": "P(TARGET=1): repayment difficulty",
        "loss": "mean binary cross-entropy over all objects and all ensemble members independently",
        "inference": "mean of member sigmoid probabilities",
        "selection": "reference/stop ROC AUC; no calibration/validation/test data",
        "best_epoch": best_epoch, "epochs_run": len(history), "best_stop_auc": best_auc,
        "training_seconds": time.perf_counter() - started,
        "model_artifact": str(out / "best_checkpoint.pt"),
        "preprocessor_artifact": str(out / "preprocessor.joblib"),
    }
    _json_write(out / "metadata.json", metadata)
    # Attach preprocessing for interactive reuse; persistent checkpoint stays portable.
    model.credit_preprocessor_ = preprocessor
    return model, predict_fn, metadata


_M2_NODES = {
    "projection_to_simplex", "logloss_grad", "logloss_hessian", "logloss_group",
    "logloss_group_grad", "logloss_group_hess", "get_subgroup_indicator",
    "dual_obj_cls", "M2FGBClassifier",
}


def _load_m2fgb(threads: int):
    """Extract exact official definitions while skipping unrelated optional models.

    The authors' models.py imports FairGBM, MinMaxFair, and PyTorch MMPF at
    module level even when only M2FGB is used. AST extraction leaves every
    selected function/class body intact and supplies their direct dependencies.
    A LightGBM facade adds only the requested CPU thread limit to train params.
    """
    import lightgbm as lgb
    import pandas as pd
    from scipy.sparse import csr_matrix
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

    source = SOURCE_ROOT / "m2fgb-facct25" / "scripts" / "models.py"
    utils_source = source.with_name("utils.py")
    source_tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in source_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _M2_NODES]
    if {node.name for node in nodes} != _M2_NODES:
        raise RuntimeError("Official M2FGB snapshot definitions changed unexpectedly")
    utils_tree = ast.parse(utils_source.read_text(encoding="utf-8"))
    utils_nodes = [node for node in utils_tree.body if isinstance(node, ast.FunctionDef) and node.name in {"logloss_group", "max_logloss_score"}]
    utils_namespace = {"np": np}
    exec(compile(ast.Module(body=utils_nodes, type_ignores=[]), str(utils_source), "exec"), utils_namespace)
    utils = types.SimpleNamespace(max_logloss_score=utils_namespace["max_logloss_score"])

    def train_with_threads(params, *args, **kwargs):
        return lgb.train({**params, "num_threads": int(threads)}, *args, **kwargs)

    facade = types.SimpleNamespace(Dataset=lgb.Dataset, train=train_with_threads,
                                  early_stopping=lgb.early_stopping)
    namespace = {
        "np": np, "pd": pd, "BaseEstimator": BaseEstimator,
        "ClassifierMixin": ClassifierMixin, "check_X_y": check_X_y,
        "check_array": check_array, "check_is_fitted": check_is_fitted,
        "log_loss": log_loss, "csr_matrix": csr_matrix, "lgb": facade,
        "utils": utils,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["M2FGBClassifier"], source, utils_source


def train_m2fgb(
    X_train, y_train, X_stop, y_stop, config: dict, seed: int, artifact_dir,
    *, group_train, group_stop=None,
) -> tuple[Any, Callable, dict]:
    """Train exact facct25 M2FGB with favorable-label TPR fairness.

    The source's TPR fairness acts on y=1. We therefore train on 1-TARGET
    (historically normal repayment), then flip its predictions back to TARGET.
    Train-only median imputation accommodates the source's finite-input checks.
    """
    from sklearn.impute import SimpleImputer

    cfg = _section(config, "m2fgb")
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)  # official objective uses NumPy to break worst-group ties
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(np.asarray(X_train, dtype=np.float32)).astype(np.float32, copy=False)
    x_stop = imputer.transform(np.asarray(X_stop, dtype=np.float32)).astype(np.float32, copy=False)
    np.savez(out / "imputer.npz", statistics=imputer.statistics_)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_stop = np.asarray(y_stop, dtype=np.int64)
    groups = np.asarray(group_train)
    for group in np.unique(groups):
        if not np.any((groups == group) & (y_train == 0)):
            raise ValueError("M2FGB favorable-label fairness requires normal records in every training group")
    cls, source, utils_source = _load_m2fgb(int(config.get("threads", cfg.get("threads", 4))))
    parameters = {
        "fairness_constraint": "true_positive_rate",
        "fair_weight": float(cfg.get("fair_weight", 0.5)),
        "dual_learning": cfg.get("dual_learning", "gradient_norm"),
        "multiplier_learning_rate": float(cfg.get("multiplier_learning_rate", 0.1)),
        "n_estimators": int(cfg.get("n_estimators", 400)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "num_leaves": int(cfg.get("num_leaves", 31)),
        "max_depth": int(cfg.get("max_depth", 5)),
        "min_child_samples": int(cfg.get("min_child_samples", 20)),
        "min_child_weight": float(cfg.get("min_child_weight", 1e-3)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 1.0)),
        "reg_alpha": float(cfg.get("reg_alpha", 0.0)),
        "reg_lambda": float(cfg.get("reg_lambda", 0.0)),
        "random_state": seed,
    }
    estimator = cls(**parameters)
    started = time.perf_counter()
    if group_stop is not None:
        estimator.fit(x_train, 1 - y_train, groups, X_val=x_stop,
                      Y_val=1 - y_stop, sensitive_attribute_val=np.asarray(group_stop))
        selection = "official reference early stopping: global BCE + maximum group BCE, patience=10"
    else:
        estimator.fit(x_train, 1 - y_train, groups)
        selection = "fixed n_estimators; stop data used for reporting only"
    # The exact source estimator includes dynamically extracted functions, so its
    # native LightGBM text model is the portable persistent artifact.
    estimator.model_.save_model(str(out / "model.txt"))

    def predict_fn(X):
        transformed = imputer.transform(np.asarray(X, dtype=np.float32)).astype(np.float32, copy=False)
        return 1.0 - estimator.predict_proba(transformed)[:, 1]

    probability = predict_fn(X_stop)
    _json_write(out / "dual_history.json", [{key: value.tolist() if isinstance(value, np.ndarray) else float(value)
                                           for key, value in item.items()} for item in estimator.info])
    metadata = {
        "algorithm": "M2FGB", "official_source": _source_metadata("m2fgb-facct25"),
        "source_file": str(source), "source_sha256": _sha256(source),
        "utils_source_sha256": _sha256(utils_source),
        "integration": "AST-extracted exact official definitions; excluded unrelated optional-model imports; LightGBM num_threads injection only",
        "selected_definitions": sorted(_M2_NODES),
        "preprocessing": "train-only median imputation, preserve empty features; no scaling",
        "training_label": "1-TARGET; favorable outcome is historically normal repayment",
        "prediction_label": "P(TARGET=1), obtained as 1-P(favorable)",
        "fairness_meaning": "min-max favorable-label logloss surrogate for true-positive-rate fairness; no hard test-gap guarantee",
        "selection": selection,
        "parameters": parameters, "seed": seed,
        "best_iteration": int(estimator.model_.best_iteration or estimator.model_.current_iteration()),
        "stop_auc": float(roc_auc_score(y_stop, probability)),
        "stop_logloss": float(log_loss(y_stop, probability)),
        "training_seconds": time.perf_counter() - started,
        "model_artifact": str(out / "model.txt"),
        "imputer_artifact": str(out / "imputer.npz"),
    }
    _json_write(out / "metadata.json", metadata)
    estimator.credit_imputer_ = imputer
    return estimator, predict_fn, metadata
