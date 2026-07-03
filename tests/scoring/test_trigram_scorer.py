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


class FreqEchoModel:
    """Predicts the tg_freq feature (column 0) back, so fitness = sum(freq^2)."""

    def predict(self, X):
        from keybo.features.schema import TRIGRAM_FEATURE_NAMES

        idx = TRIGRAM_FEATURE_NAMES.index("tg_freq")
        return X[:, idx]


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


def test_tg_freq_feature_is_passed_through():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    # FreqEchoModel returns tg_freq per row, so fitness = sum(freq * freq).
    scorer = TrigramModelScorer(FreqEchoModel(), trigram_freqs={"the": 4, "and": 3})
    assert scorer.fitness(lay) == 4 * 4 + 3 * 3


def test_constituent_bigram_and_skipgram_freqs_are_used():
    # With explicit bg/sg freqs, the feature vector's bg1_freq/bg2_freq/sg_freq differ from
    # the default of 1, so a model reading those columns produces a different score.
    lay = Layout(LAYOUT, ROW_STAGGERED_30)

    class SkipFreqModel:
        def predict(self, X):
            from keybo.features.schema import TRIGRAM_FEATURE_NAMES

            return X[:, TRIGRAM_FEATURE_NAMES.index("sg_freq")]

    with_sg = TrigramModelScorer(
        SkipFreqModel(), trigram_freqs={"the": 1}, skipgram_freqs={"te": 50}
    )
    without_sg = TrigramModelScorer(SkipFreqModel(), trigram_freqs={"the": 1})
    # 'the' skipgram is 't'+'e' = 'te'; with sg_freq=50 the score reflects it, default is 1.
    assert with_sg.fitness(lay) == 50.0
    assert without_sg.fitness(lay) == 1.0
