"""Tests for ModelScorer.

Includes the end-to-end regression for bug #2 (the objective must see every key) with a
negative control that reproduces the old 28-character-restricted behavior and shows it
would NOT have detected the change.
"""

import numpy as np
import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.model_scorer import BigramModelScorer

LAYOUT = "qwertyuiopasdfghjkl'zxcvbnm,.-"


class StubModel:
    """Returns a fixed per-row value so total fitness is hand-checkable."""

    def __init__(self, value=2.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class DistanceModel:
    """Predicts the 'distance' feature (index 16 in BIGRAM_FEATURE_NAMES).

    This makes fitness sensitive to physical placement, so moving any key changes the score.
    """

    def predict(self, X):
        from keybo.features.schema import BIGRAM_FEATURE_NAMES

        idx = BIGRAM_FEATURE_NAMES.index("distance")
        return X[:, idx]


def test_fitness_is_sum_of_prediction_times_frequency():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    # Two bigrams, freqs 3 and 5, model returns 2.0 each -> 2*3 + 2*5 = 16.
    scorer = BigramModelScorer(StubModel(2.0), bigram_freqs={"th": 3, "he": 5})
    assert scorer.fitness(lay) == 16.0


def test_fitness_is_deterministic():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    scorer = BigramModelScorer(DistanceModel(), bigram_freqs={"th": 1, "he": 1, "an": 1})
    assert scorer.fitness(lay) == scorer.fitness(lay)


def test_regression_bug2_moving_punctuation_keys_changes_fitness():
    """Every key must affect fitness -- including the two punctuation keys the old
    28-char scorer ignored. Use a corpus of bigrams *involving* those keys."""
    base = Layout(LAYOUT, ROW_STAGGERED_30)
    swapped = Layout(LAYOUT, ROW_STAGGERED_30)
    swapped.swap("'", "-")

    # A corpus that actually uses the punctuation keys.
    freqs = {"'a": 100, "-e": 100, "n'": 50, "m-": 50}
    scorer = BigramModelScorer(DistanceModel(), bigram_freqs=freqs)

    assert scorer.fitness(base) != scorer.fitness(swapped)


def test_negative_control_28char_restricted_scorer_is_blind():
    """Negative control proving the regression above is meaningful.

    Reproduce the OLD behavior: only score bigrams over the hardcoded 28-char set (letters
    + comma + period). Under that rule, a corpus of punctuation bigrams contributes nothing,
    so the fitness is identical before/after moving the punctuation keys -- the exact bug.
    """
    old_charset = set("qwertyuiopasdfghjklzxcvbnm,.")
    base = Layout(LAYOUT, ROW_STAGGERED_30)
    swapped = Layout(LAYOUT, ROW_STAGGERED_30)
    swapped.swap("'", "-")

    freqs = {"'a": 100, "-e": 100, "n'": 50, "m-": 50}

    def old_fitness(layout):
        from keybo.features import bigram_features

        total = 0.0
        model = DistanceModel()
        for bg, f in freqs.items():
            if not all(c in old_charset for c in bg):
                continue  # old scorer skipped anything outside its 28 chars
            total += float(model.predict(bigram_features(layout, bg).reshape(1, -1))[0]) * f
        return total

    # Every corpus bigram touches ' or -, so the restricted scorer sees nothing: both 0.
    assert old_fitness(base) == old_fitness(swapped) == 0.0


def test_scorer_covers_all_provided_corpus_bigrams():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    scorer = BigramModelScorer(StubModel(1.0), bigram_freqs={"th": 1, "'-": 1, ",.": 1})
    # 3 bigrams * value 1.0 * freq 1 = 3.0 -> proves none were dropped.
    assert scorer.fitness(lay) == 3.0


def test_scorer_skips_bigrams_with_characters_not_on_the_layout():
    # This layout has ' and - but NOT ; or /. A corpus bigram using ';' is not typable here
    # and must be skipped, not mapped to a phantom position.
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    scorer = BigramModelScorer(StubModel(1.0), bigram_freqs={"th": 1, "a;": 1, "e/": 1})
    # Only 'th' is fully on the layout -> fitness = 1 * 1.0.
    assert scorer.fitness(lay) == 1.0


def test_regression_bug_scorer_includes_space_bigrams():
    """Space is a fixed board key at (0,0); the training pipeline emits space bigrams, so
    the scorer MUST score them too (train/serve parity). Dropping them silently discarded
    ~37% of the real corpus weight, including the single most common bigram 'e '.
    """
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    # A corpus of space bigrams only. If space were dropped, fitness would be 0.
    scorer = BigramModelScorer(StubModel(1.0), bigram_freqs={"e ": 5, " t": 3, "s ": 2})
    assert scorer.fitness(lay) == 10.0  # (5 + 3 + 2) * 1.0 — none dropped


def test_regression_space_bigram_train_serve_parity():
    """The feature vector the SCORER builds for a space bigram must equal the vector the
    TRAINING path builds for the same bigram+positions. (This is the identical-features
    guarantee that space previously violated.)"""
    import numpy as np

    from keybo.features import bigram_features_from_positions

    lay = Layout(LAYOUT, ROW_STAGGERED_30)

    # Capture what the scorer feeds the model for "e " by using a model that echoes its input.
    class CaptureModel:
        def __init__(self):
            self.seen = None

        def predict(self, X):
            self.seen = X
            return np.zeros(len(X))

    m = CaptureModel()
    BigramModelScorer(m, bigram_freqs={"e ": 1}, target_wpm=90).fitness(lay)
    scorer_vec = m.seen[0]

    # The training path builds features from positions; space is (0,0).
    e_pos = lay.pos("e")
    train_vec = bigram_features_from_positions(ROW_STAGGERED_30, (e_pos, (0, 0)), wpm=90)
    np.testing.assert_array_equal(scorer_vec, train_vec)


def test_scorer_still_skips_genuinely_off_layout_chars_but_keeps_space():
    # ';' is not on this layout (has ' and -), so 'a;' is skipped; but 'e ' (space) is kept.
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    scorer = BigramModelScorer(StubModel(1.0), bigram_freqs={"th": 1, "a;": 1, "e ": 1})
    assert scorer.fitness(lay) == 2.0  # 'th' and 'e ' kept; 'a;' dropped


# --- LOGRAT target space (T-REL, 2026-07-10) ----------------------------------------------


class LogratStubModel:
    """A stub in the LOGRAT space: raw predict() returns log(ms*wpm/12000) for a fixed
    ms, carrying the sidecar shape a real saved LOGRAT model has."""

    def __init__(self, ms=100.0, calibration=None):
        from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
        from keybo.models.base import ModelMetadata

        self.ms = ms
        self.metadata = ModelMetadata(
            feature_version=FEATURE_VERSION,
            feature_names=list(BIGRAM_FEATURE_NAMES),
            wpm_range=(40, 140),
            ngram="bigram",
            extra={
                "training": {
                    "target_space": "LOGRAT",
                    "calibration": calibration,
                }
            },
        )

    def predict(self, X):
        from keybo.features.schema import BIGRAM_FEATURE_NAMES

        wpm = np.asarray(X)[:, BIGRAM_FEATURE_NAMES.index("wpm")]
        return np.log(self.ms * wpm / 12000.0)

    # borrow the conversion seam from the real base class
    from keybo.models.base import TypingModel

    target_space = TypingModel.target_space
    to_ms = TypingModel.to_ms
    predict_ms = TypingModel.predict_ms
    predict_ms_at = TypingModel.predict_ms_at


def test_lograt_model_scorer_sums_ms_not_log_values():
    """fitness must be Σ freq·ms — for a constant-100ms LOGRAT model with freqs 3+5
    that is 800, not the (near-zero) sum of raw log predictions."""
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    scorer = BigramModelScorer(
        LogratStubModel(ms=100.0), bigram_freqs={"th": 3, "he": 5}, target_wpm=90.0
    )
    assert scorer.fitness(lay) == pytest.approx(800.0, rel=1e-9)


def test_model_scorer_applies_first_finger_calibration_by_position():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    model = LogratStubModel(
        ms=138.0,
        calibration={"deltas_ms": {"pinky_first": 62.0}},
    )

    scorer = BigramModelScorer(model, bigram_freqs={"qw": 3}, target_wpm=100.0)

    # q->w is same-row pinky-first. At 100 WPM the calibration reference is
    # 138 ms, so the 62 ms sidecar delta makes each occurrence exactly 200 ms.
    assert scorer.fitness(lay) == pytest.approx(600.0)
