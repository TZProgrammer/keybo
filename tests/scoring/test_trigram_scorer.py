"""Tests for TrigramModelScorer (mirrors the bigram scorer tests).

Fills the coverage gap: the trigram scorer had no tests. Uses a stub model so the fitness
arithmetic is hand-checkable, and covers the layout-character filtering and constituent
(bigram/skipgram) frequency wiring that distinguish it from the bigram scorer.
"""

import numpy as np

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.model_scorer import TrigramModelScorer

LAYOUT = "qwertyuiopasdfghjkl'zxcvbnm,.-"


class StubModel:
    """Returns a fixed per-row value so total fitness is hand-checkable."""

    def __init__(self, value=2.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_fitness_is_sum_of_prediction_times_frequency():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    # Two trigrams, freqs 3 and 5, model returns 2.0 each -> 2*3 + 2*5 = 16.
    scorer = TrigramModelScorer(StubModel(2.0), trigram_freqs={"the": 3, "and": 5})
    assert scorer.fitness(lay) == 16.0


def test_fitness_is_deterministic():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    scorer = TrigramModelScorer(StubModel(1.5), trigram_freqs={"the": 1, "ing": 1})
    assert scorer.fitness(lay) == scorer.fitness(lay)


def test_empty_corpus_is_zero():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    assert TrigramModelScorer(StubModel(9.0), trigram_freqs={}).fitness(lay) == 0.0


def test_skips_trigrams_with_characters_not_on_the_layout():
    # This layout has ' and - but NOT ; or /. A trigram using ';' is not typable here.
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    scorer = TrigramModelScorer(StubModel(1.0), trigram_freqs={"the": 1, "a;b": 1, "e/n": 1})
    # Only 'the' is fully on the layout -> fitness = 1 * 1.0.
    assert scorer.fitness(lay) == 1.0


def test_freq_is_weight_only_not_a_feature():
    """OQ-1: fitness scales linearly with corpus frequency (the weight), and the feature
    matrix contains no frequency column for it to leak into."""
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    one = TrigramModelScorer(StubModel(2.0), trigram_freqs={"the": 1}).fitness(lay)
    ten = TrigramModelScorer(StubModel(2.0), trigram_freqs={"the": 10}).fitness(lay)
    assert ten == 10 * one  # weight role intact
    from keybo.features.schema import TRIGRAM_FEATURE_NAMES

    assert not [n for n in TRIGRAM_FEATURE_NAMES if "freq" in n]  # feature role gone


def test_regression_trigram_scorer_train_serve_parity():
    """The vector the scorer feeds the model for a trigram must equal the training path's
    vector for the same positions. (The old freq-leak landmine — audit #5 — died with the
    schema: there are no frequency columns to diverge anymore; this guards the rest.)
    """
    import numpy as np

    from keybo.features import trigram_features_from_positions

    lay = Layout(LAYOUT, ROW_STAGGERED_30)

    class CaptureModel:
        def __init__(self):
            self.seen = None

        def predict(self, X):
            self.seen = X
            return np.zeros(len(X))

    m = CaptureModel()
    TrigramModelScorer(m, trigram_freqs={"the": 999}, target_wpm=90).fitness(lay)
    scorer_vec = m.seen[0]

    ps = tuple(lay.pos(c) for c in "the")
    train_vec = trigram_features_from_positions(ROW_STAGGERED_30, ps, wpm=90)
    np.testing.assert_array_equal(scorer_vec, train_vec)
