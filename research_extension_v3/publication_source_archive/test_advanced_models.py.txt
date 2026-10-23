"""Small integration checks for official-source advanced-model adapters."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from research_extension_v3.advanced_models import train_m2fgb, train_tabm


def _data():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(1000, 8)).astype(np.float32)
    x[:, 7] = (x[:, 7] > 0).astype(np.float32)
    risk = 1 / (1 + np.exp(-2.5 * x[:, 0]))
    y = (rng.random(len(x)) < risk).astype(np.int64)
    group = (x[:, 1] > 0).astype(np.int64)
    return x[:800], y[:800], group[:800], x[800:], y[800:], group[800:]


def test_m2fgb_official_source_native_lightgbm_and_label_direction(tmp_path):
    x, y, g, stop_x, stop_y, stop_g = _data()
    x[0, 2] = np.nan
    stop_x[0, 2] = np.nan
    config = {"threads": 2, "m2fgb": {"n_estimators": 40,
              "max_depth": 3, "num_leaves": 7, "learning_rate": 0.1,
              "fair_weight": 0.2}}
    estimator, predict, metadata = train_m2fgb(
        x, y, stop_x, stop_y, config, 44, tmp_path,
        group_train=g, group_stop=stop_g,
    )
    probability = predict(stop_x)
    assert probability.shape == stop_y.shape
    assert np.isfinite(probability).all()
    assert ((0 <= probability) & (probability <= 1)).all()
    assert roc_auc_score(stop_y, probability) > 0.75
    # The published model is trained with good=1; external probabilities must
    # consistently be difficulty=1 for all V3 metrics and approval thresholds.
    good_probability = estimator.predict_proba(estimator.credit_imputer_.transform(stop_x))[:, 1]
    np.testing.assert_allclose(probability, 1 - good_probability)
    assert metadata["official_source"]["branch"] == "facct25"
    assert Path(metadata["model_artifact"]).is_file()
    assert (tmp_path / "dual_history.json").is_file()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="PyTorch CUDA runtime is still being installed")
def test_tabm_official_source_train_predict_and_checkpoint(tmp_path):
    x, y, _, stop_x, stop_y, _ = _data()
    config = {"threads": 2, "tabm": {"max_epochs": 4, "patience": 2,
              "batch_size": 128, "eval_batch_size": 128, "k": 4,
              "d_block": 32, "n_blocks": 2, "learning_rate": 0.01}}
    model, predict, metadata = train_tabm(x, y, stop_x, stop_y, config, 44, tmp_path)
    probability = predict(stop_x)
    assert probability.shape == stop_y.shape
    assert np.isfinite(probability).all()
    assert ((0 <= probability) & (probability <= 1)).all()
    assert roc_auc_score(stop_y, probability) > 0.75
    assert model.k == 4
    assert Path(metadata["model_artifact"]).is_file()
    assert Path(metadata["preprocessor_artifact"]).is_file()
    # One-hot/binary indicator is excluded from continuous rank normalization.
    assert metadata["rank_normalized_columns"] == 7
