"""Tests for ModelScorer.

Includes the end-to-end regression for bug #2 (the objective must see every key) with a
negative control that reproduces the old 28-character-restricted behavior and shows it
would NOT have detected the change.
"""

import numpy as np

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
