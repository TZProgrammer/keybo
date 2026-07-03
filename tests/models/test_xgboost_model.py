"""Tests for the XGBoost TypingModel implementation.

Covers the two bugs the old code had here:
  - #12: modern XGBoost params (device=), not the removed gpu_hist/gpu_id.
  - #13: portable JSON persistence, not __main__-dependent pickles.
"""

import numpy as np
import pytest

from keybo.models.base import FeatureVersionMismatch, ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel


def make_meta(feature_version="v1"):
    return ModelMetadata(
        feature_version=feature_version,
        feature_names=[f"f{i}" for i in range(4)] + ["wpm"],
        wpm_range=(60, 100),
        ngram="bigram",
    )


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    X = rng.random((60, 5))
    # A learnable target so the model isn't degenerate.
    y = X[:, 0] * 10 + X[:, 1] * 3 + 1
    return X, y


def test_fit_and_predict_shape(tiny_data):
    X, y = tiny_data
    model = XGBoostTypingModel(make_meta(), n_estimators=10, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_predict_is_reasonable_after_fit(tiny_data):
    X, y = tiny_data
    model = XGBoostTypingModel(make_meta(), n_estimators=50, max_depth=3)
    model.fit(X, y)
    mae = np.mean(np.abs(model.predict(X) - y))
    assert mae < 2.0  # loose sanity bound; target range is ~1..14


def test_save_load_roundtrip_is_bit_identical(tmp_path, tiny_data):
    X, y = tiny_data
    model = XGBoostTypingModel(make_meta(), n_estimators=10, max_depth=3)
    model.fit(X, y)
    path = tmp_path / "bg.json"
    model.save(str(path))

    loaded = XGBoostTypingModel.load(str(path), expected_feature_version="v1")
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_persistence_is_json_not_pickle(tmp_path, tiny_data):
    X, y = tiny_data
    model = XGBoostTypingModel(make_meta(), n_estimators=5)
    model.fit(X, y)
    path = tmp_path / "bg.json"
    model.save(str(path))
    # Artifact is valid UTF-8 JSON text (a pickle would not be).
    text = path.read_text()
    assert text.lstrip().startswith("{")


def test_load_enforces_feature_version(tmp_path, tiny_data):
    X, y = tiny_data
    model = XGBoostTypingModel(make_meta(feature_version="OLD"), n_estimators=5)
    model.fit(X, y)
    path = tmp_path / "bg.json"
    model.save(str(path))
    with pytest.raises(FeatureVersionMismatch):
        XGBoostTypingModel.load(str(path), expected_feature_version="NEW")


def test_predict_before_fit_raises():
    model = XGBoostTypingModel(make_meta(), n_estimators=5)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 5)))
