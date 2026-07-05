"""TableTrigramScorer: the trigram objective's fast path, exact-parity-tested.

Same standard as the bigram table (test_table_scorer.py): with freq-inert features and a
fixed wpm, a trigram's predicted time depends only on its position TRIPLE — 31^3 = 29,791
entries, one batch predict at construction, fitness = fancy-indexed sum over the corpus.
The speedup is only legitimate if it is bit-for-bit the same objective as
TrigramModelScorer, so exact parity on random permutations is the core test.
"""

import numpy as np
import pytest

from keybo.data.strokes import StrokeRow
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.model_scorer import TrigramModelScorer
from keybo.scoring.table_trigram import TableTrigramScorer
from keybo.training.train import train_trigram_model

QWERTY = NAMED_LAYOUTS["qwerty"]


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    tris = ["the", "and", "ing", "her", "ent", "he ", " th", "was"]
    rows = []
    for i in range(100):
        tg = tris[i % len(tris)]
        samples = [(90, 200 + int(rng.integers(0, 80)), i % 7 + 1, 50)]
        rows.append(
            StrokeRow(
                layout="qwerty",
                positions=((-1, 3), (1, 2), (-3, 3)),
                ngram=tg,
                frequency=5,
                samples=samples,
            )
        )
    return train_trigram_model(rows, target_wpm=90, n_estimators=6, max_depth=2)


@pytest.fixture(scope="module")
def freqs():
    return {"the": 100, "and": 80, "he ": 60, " th": 50, "qzx": 5, "'ab": 9}


def test_parity_with_model_scorer_on_permutations(model, freqs):
    ref = TrigramModelScorer(model, freqs, target_wpm=90.0)
    table = TableTrigramScorer(model, freqs, target_wpm=90.0, chars=QWERTY)
    rng = np.random.default_rng(11)
    chars = list(QWERTY)
    for trial in range(3):
        perm = "".join(rng.permutation(chars)) if trial else QWERTY
        lay = Layout(perm, ROW_STAGGERED_30)
        assert table.fitness(lay) == pytest.approx(ref.fitness(lay), rel=1e-9), perm


def test_untypable_trigrams_excluded(model):
    table = TableTrigramScorer(model, {"'ab": 9}, target_wpm=90.0, chars=QWERTY)
    assert table.fitness(Layout(QWERTY, ROW_STAGGERED_30)) == 0.0


def test_charset_mismatch_rejected(model, freqs):
    table = TableTrigramScorer(model, freqs, target_wpm=90.0, chars=QWERTY)
    with pytest.raises(ValueError, match="charset"):
        table.fitness(Layout(NAMED_LAYOUTS["dvorak"], ROW_STAGGERED_30))


def test_fitness_of_permutation_fast_path_matches(model, freqs):
    table = TableTrigramScorer(model, freqs, target_wpm=90.0, chars=QWERTY)
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    perm = table.permutation(lay)
    assert table.fitness_of_permutation(perm) == pytest.approx(table.fitness(lay), rel=1e-12)
