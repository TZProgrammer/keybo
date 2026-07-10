"""Tests for model training.

The essential property: training builds its feature matrix with the SAME feature pipeline
that scoring uses, so a model trained here is valid for the scorer (no train/serve skew).
The trained model's metadata records the current FEATURE_VERSION.
"""

import numpy as np
import pytest

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
        # a couple of (wpm, duration, pid, hold) samples per row
        samples = [
            (90, 100 + rng.integers(0, 40), i, 50),
            (85, 110 + rng.integers(0, 40), i, 55),
        ]
        rows.append(
            StrokeRow(
                layout="qwerty",
                positions=((-1, 3), (1, 2)),
                ngram=bg,
                frequency=5,
                samples=samples,
            )
        )
    return rows


def test_regression_training_handles_space_bigrams():
    """Real bistroke tables are dominated by space bigrams (positions include (0,0)).
    build_training_matrix must not crash on them."""
    space = (0, 0)

    def row(positions, ngram, freq, dur):
        return StrokeRow(
            layout="qwerty",
            positions=positions,
            ngram=ngram,
            frequency=freq,
            samples=[(90, dur, 1, 50)],
        )

    rows = [
        row(((-3, 3), space), "e ", 100, 120),
        row((space, (-1, 3)), " t", 90, 130),
        row(((-1, 3), (1, 2)), "th", 80, 110),
    ]
    X, y = build_training_matrix(rows, ngram="bigram", target_wpm=90)
    assert X.shape[0] == 3
    assert np.all(np.isfinite(X))


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


# --- practice term + layout weights (the R1W recipe, productionized) --------------------


def test_practice_term_recovers_a_planted_per_bigram_offset():
    """Plant a large per-bigram additive offset on top of a flat geometry world; the
    fitted practice term must recover it (sign and rough magnitude), and the offset must
    NOT leak into g's predictions (which are geometry-only).

    Pinned to the MS space so the planted ±60ms offsets stay additive; the LOGRAT
    analogue (log-scale b) is asserted separately below.
    """
    fast, slow = "th", "qz"
    rows = []
    for ngram, offset in ((fast, -60), (slow, +60)):
        # Same positions for both -> geometry identical; only the offset differs.
        samples = [(90, 150 + offset + (i % 5), i, 50) for i in range(200)]
        rows.append(
            StrokeRow(
                layout="qwerty",
                positions=((-1, 3), (1, 2)),
                ngram=ngram,
                frequency=200,
                samples=samples,
            )
        )
    model = train_bigram_model(rows, target_wpm=90, n_estimators=20, max_depth=2, target_space="MS")
    b = model.metadata.extra["training"]["practice_term"]["values"]
    assert b[fast] < -20, b  # planted -60, shrunk toward 0 is fine; sign + size must show
    assert b[slow] > +20, b
    # b(fast) + b(slow) ~ 0 (g absorbs the shared mean).
    assert abs(b[fast] + b[slow]) < 20


def test_practice_term_can_be_disabled():
    rows = _synthetic_bigram_rows()
    model = train_bigram_model(
        rows, target_wpm=90, n_estimators=5, max_depth=2, practice_term=False
    )
    assert model.metadata.extra["training"]["practice_term"] is None


def test_layout_weights_flag_recorded():
    rows = _synthetic_bigram_rows()
    on = train_bigram_model(rows, target_wpm=90, n_estimators=5, max_depth=2)
    off = train_bigram_model(rows, target_wpm=90, n_estimators=5, max_depth=2, layout_weights=False)
    assert on.metadata.extra["training"]["layout_weights"] is True
    assert off.metadata.extra["training"]["layout_weights"] is False


def test_layout_balance_weights_equalize_shares():
    import numpy as np

    from keybo.training.train import layout_balance_weights

    layouts = np.array(["qwerty"] * 90 + ["dvorak"] * 10, dtype=object)
    w = layout_balance_weights(layouts)
    assert w.mean() == pytest.approx(1.0)
    # Each layout's total weight is ~equal after balancing.
    assert w[:90].sum() == pytest.approx(w[90:].sum(), rel=0.01)


def test_practice_shrinkage_suppresses_rare_ngrams():
    import numpy as np

    from keybo.training.train import fit_practice_term

    ngrams = np.array(["aa"] * 3 + ["bb"] * 300, dtype=object)
    resid = np.array([-50.0] * 3 + [-50.0] * 300)
    counts = np.ones(303)
    b = fit_practice_term(ngrams, resid, counts, k=100.0)
    # Same residual, but 'aa' has 3 samples vs 'bb' 300: shrinkage must bite 'aa' hard.
    assert abs(b["aa"]) < 5
    assert b["bb"] < -35


# --- target space: LOGRAT (pace-normalized log-ratio, T-REL 2026-07-10) ------------------


def test_bigram_training_defaults_to_lograt_space():
    """LOGRAT is the adopted bigram recipe (T-REL: -37% cross-layout wmae); like the
    practice term and layout weights, the shipped default IS the validated recipe."""
    rows = _synthetic_bigram_rows()
    model = train_bigram_model(rows, target_wpm=90, n_estimators=5, max_depth=2)
    assert model.metadata.extra["training"]["target_space"] == "LOGRAT"
    assert model.target_space == "LOGRAT"


def test_trigram_training_defaults_to_lograt_space():
    """The conditioned-trigram LOGRAT A/B adopted (2026-07-10: wmae -30.7%, all guards
    improved), so the trigram default matches the bigram one."""
    from keybo.training.train import train_trigram_model

    rows = [
        StrokeRow(
            layout="qwerty",
            positions=((-1, 3), (1, 2), (-2, 2)),
            ngram="the",
            frequency=5,
            samples=[(90, 200 + i, i, 50) for i in range(4)],
        )
        for _ in range(30)
    ]
    model = train_trigram_model(rows, target_wpm=90, n_estimators=5, max_depth=2)
    assert model.metadata.extra["training"]["target_space"] == "LOGRAT"
    assert model.target_space == "LOGRAT"


def test_lograt_model_predict_ms_recovers_duration_scale():
    """A LOGRAT-trained model's raw predict() is in log space (small numbers), but
    predict_ms must come back in the data's duration scale."""
    rows = _synthetic_bigram_rows(n=120)
    model = train_bigram_model(
        rows, target_wpm=90, n_estimators=30, max_depth=2, target_space="LOGRAT"
    )
    X, y = build_training_matrix(rows, ngram="bigram", target_wpm=90)
    raw = model.predict(X)
    ms = model.predict_ms(X)
    assert np.all(np.abs(raw) < 5)  # log-ratio scale
    assert 50 < np.median(ms) < 400  # duration scale (data is ~100-150ms)


def test_lograt_practice_term_lives_in_log_space():
    """In LOGRAT space the backfit residuals are log-ratios, so stored b values must be
    log-scale (|b| << 1), not the ±60ms of the planted offset."""
    fast, slow = "th", "qz"
    rows = []
    for ngram, offset in ((fast, -60), (slow, +60)):
        samples = [(90, 150 + offset + (i % 5), i, 50) for i in range(200)]
        rows.append(
            StrokeRow(
                layout="qwerty",
                positions=((-1, 3), (1, 2)),
                ngram=ngram,
                frequency=200,
                samples=samples,
            )
        )
    model = train_bigram_model(
        rows, target_wpm=90, n_estimators=20, max_depth=2, target_space="LOGRAT"
    )
    b = model.metadata.extra["training"]["practice_term"]["values"]
    # log(90/150) ~ -0.51, log(210/150) ~ +0.34: log-scale, correct signs.
    assert -1.0 < b[fast] < -0.1, b
    assert 0.1 < b[slow] < 1.0, b


def test_train_rejects_unknown_target_space():
    rows = _synthetic_bigram_rows()
    with pytest.raises(ValueError, match="target_space"):
        train_bigram_model(rows, target_wpm=90, n_estimators=5, target_space="RATIO")
