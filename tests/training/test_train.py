"""Tests for model training.

The essential property: training builds its feature matrix with the SAME feature pipeline
that scoring uses, so a model trained here is valid for the scorer (no train/serve skew).
The trained model's metadata records the current FEATURE_VERSION.
"""

import numpy as np

from keybo.data.strokes import StrokeRow
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.training.train import build_training_matrix, train_bigram_model


def _synthetic_bigram_rows(n=80):
    """Stroke rows whose durations depend on distance, so the model can learn something."""
    rng = np.random.default_rng(0)
    bigrams = ["th", "he", "an", "in", "er", "re", "on", "at", "en", "nd"]
    rows = []
    for i in range(n):
        bg = bigrams[i % len(bigrams)]
        # a couple of (wpm, duration) samples per row
        samples = [(90, 100 + rng.integers(0, 40)), (85, 110 + rng.integers(0, 40))]
        rows.append(StrokeRow(positions=((-1, 3), (1, 2)), ngram=bg, frequency=5, samples=samples))
    return rows


def test_build_training_matrix_uses_shared_feature_width():
    rows = _synthetic_bigram_rows()
    X, y = build_training_matrix(rows, ngram="bigram", target_wpm=90)
    assert X.shape[1] == len(BIGRAM_FEATURE_NAMES)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0


def test_train_bigram_model_returns_model_with_current_feature_version():
    rows = _synthetic_bigram_rows()
    model = train_bigram_model(rows, target_wpm=90, n_estimators=10, max_depth=3)
    assert isinstance(model, XGBoostTypingModel)
    assert model.metadata.feature_version == FEATURE_VERSION
    assert model.metadata.ngram == "bigram"


def test_trained_model_can_score_a_layout(tmp_path):
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.layout import Layout
    from keybo.scoring.model_scorer import BigramModelScorer

    rows = _synthetic_bigram_rows()
    model = train_bigram_model(rows, target_wpm=90, n_estimators=10, max_depth=3)
    scorer = BigramModelScorer(model, bigram_freqs={"th": 100, "he": 90}, target_wpm=90)
    fitness = scorer.fitness(Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", ROW_STAGGERED_30))
    assert np.isfinite(fitness)


def test_trained_model_saves_and_reloads(tmp_path):
    rows = _synthetic_bigram_rows()
    model = train_bigram_model(rows, target_wpm=90, n_estimators=10, max_depth=3)
    path = tmp_path / "bg.json"
    model.save(str(path))
    # Reloads under the CURRENT feature version (default) -> no mismatch.
    reloaded = XGBoostTypingModel.load(str(path))
    assert reloaded.metadata.feature_version == FEATURE_VERSION
