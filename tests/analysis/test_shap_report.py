"""Tests for the SHAP report: TreeSHAP correctness invariants + the CLI end-to-end.

The correctness anchor is additivity (contributions sum to the prediction) and a
known-answer check: a model trained where only ONE feature drives the target must rank
that feature first by mean |SHAP|.
"""

import json

import numpy as np
import pytest

from keybo.analysis.shap_report import compute_shap, render_report
from keybo.cli.__main__ import main
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel


def _make_model(rng_seed=0, n=400):
    """Train a tiny model where `distance` (plus noise) is the only real signal."""
    rng = np.random.default_rng(rng_seed)
    names = BIGRAM_FEATURE_NAMES
    X = rng.random((n, len(names)))
    dist_col = names.index("distance")
    y = 100.0 + 80.0 * X[:, dist_col] + rng.normal(0, 1.0, n)
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=list(names),
        wpm_range=(0, 300),
        ngram="bigram",
    )
    model = XGBoostTypingModel(meta, n_estimators=50, max_depth=3)
    model.fit(X, y)
    return model, X


def test_shap_additivity_and_known_answer():
    model, X = _make_model()
    report = compute_shap(model, X)

    # additivity is asserted inside compute_shap; re-check the exposed arrays agree
    total = report.shap_values.sum(axis=1) + report.base_value
    assert np.allclose(total, model.predict(X), atol=0.5)

    # the only real signal must dominate the ranking
    assert report.ranking()[0][0] == "distance"
    # and its signed mean direction must be ~0 (symmetric around base), while its
    # dependence is monotone: high-distance rows get positive SHAP
    dist_col = report.feature_names.index("distance")
    hi = X[:, dist_col] > 0.8
    lo = X[:, dist_col] < 0.2
    assert report.shap_values[hi, dist_col].mean() > report.shap_values[lo, dist_col].mean()


def test_interaction_pairs_present_and_sorted():
    model, X = _make_model()
    report = compute_shap(model, X)
    vals = [v for _, _, v in report.interaction_pairs]
    assert vals == sorted(vals, reverse=True)


def test_render_report_writes_figures(tmp_path):
    model, X = _make_model()
    report = compute_shap(model, X)
    paths = render_report(report, str(tmp_path / "shap"), top_k=5)
    assert len(paths) >= 3
    for p in paths:
        assert (tmp_path / p.split("/")[-1]).exists()


@pytest.mark.slow
def test_cli_grid_end_to_end(tmp_path):
    model, _X = _make_model()
    model_path = tmp_path / "model.json"
    model.save(str(model_path))

    rc = main(
        [
            "shap-report",
            "--model",
            str(model_path),
            "--on",
            "grid",
            "--out-prefix",
            str(tmp_path / "shap"),
            "--top-k",
            "5",
        ]
    )
    assert rc == 0
    out = json.loads((tmp_path / "shap.json").read_text())
    assert out["ranking"][0]["feature"] == "distance"
    assert (tmp_path / "shap_ranking.png").exists()
    assert (tmp_path / "shap_beeswarm.png").exists()
    assert (tmp_path / "shap_dependence.png").exists()
