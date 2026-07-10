"""TableBigramScorer: the QAP-table fast path must be EXACTLY the model scorer.

With the freq feature pinned and wpm fixed, a bigram's predicted time depends only on its
(position, position) pair — so a layout's fitness is a sum over a precomputed 31x31 table
(30 slots + the fixed space key). That reduces scoring from ~25 ms (a model predict over
the whole corpus) to microseconds (a fancy-indexed sum), which is what makes deep search
feasible. The table path earns that speed only if it is bit-for-bit the same objective, so
the core test here is exact parity against BigramModelScorer on random permutations.
"""

import numpy as np
import pytest

from keybo.data.strokes import StrokeRow
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.model_scorer import BigramModelScorer
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.training.train import train_bigram_model

QWERTY = NAMED_LAYOUTS["qwerty"]


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    bigrams = ["th", "he", "an", "in", "er", "re", "on", "at", "e ", " t", "d ", "s "]
    rows = []
    for i in range(120):
        bg = bigrams[i % len(bigrams)]
        samples = [
            (90, 100 + int(rng.integers(0, 60)), i, 50),
            (85, 110 + int(rng.integers(0, 60)), i, 55),
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
    return train_bigram_model(rows, target_wpm=90, n_estimators=10, max_depth=3)


@pytest.fixture(scope="module")
def freqs():
    # Letter and space bigrams (typable on qwerty) + one with a char qwerty lacks.
    return {"th": 100, "he": 90, "e ": 80, " t": 70, "an": 60, "qz": 5, "'a": 40}


def test_parity_with_model_scorer_on_qwerty_and_permutations(model, freqs):
    ref = BigramModelScorer(model, freqs, target_wpm=90.0)
    table = TableBigramScorer(model, freqs, target_wpm=90.0, chars=QWERTY)
    rng = np.random.default_rng(7)
    chars = list(QWERTY)
    for trial in range(4):
        perm = "".join(rng.permutation(chars)) if trial else QWERTY
        lay = Layout(perm, ROW_STAGGERED_30)
        assert table.fitness(lay) == pytest.approx(ref.fitness(lay), rel=1e-9), perm


def test_untypable_bigrams_are_excluded_like_the_model_scorer(model, freqs):
    # "'a" is typable on dvorak but not qwerty; the qwerty-charset table must ignore it,
    # exactly as BigramModelScorer's has_key filter does (covered by parity above, but
    # assert the direct consequence: a freqs dict of ONLY untypable bigrams scores 0).
    table = TableBigramScorer(model, {"'a": 40, "qz'": 5}, target_wpm=90.0, chars=QWERTY)
    assert table.fitness(Layout(QWERTY, ROW_STAGGERED_30)) == 0.0


def test_swapping_two_keys_changes_fitness_consistently(model, freqs):
    table = TableBigramScorer(model, freqs, target_wpm=90.0, chars=QWERTY)
    ref = BigramModelScorer(model, freqs, target_wpm=90.0)
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    lay.swap("t", "q")
    assert table.fitness(lay) == pytest.approx(ref.fitness(lay), rel=1e-9)


def test_charset_mismatch_is_rejected(model, freqs):
    table = TableBigramScorer(model, freqs, target_wpm=90.0, chars=QWERTY)
    dvorak = Layout(NAMED_LAYOUTS["dvorak"], ROW_STAGGERED_30)  # has ' — not in the table
    with pytest.raises(ValueError, match="charset"):
        table.fitness(dvorak)


def test_table_is_much_faster_than_model_scorer(model, freqs):
    import time

    ref = BigramModelScorer(model, freqs, target_wpm=90.0)
    table = TableBigramScorer(model, freqs, target_wpm=90.0, chars=QWERTY)
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    t0 = time.perf_counter()
    for _ in range(3):
        ref.fitness(lay)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(3):
        table.fitness(lay)
    t_table = time.perf_counter() - t0
    assert t_table < t_ref  # conservatively "faster"; in practice ~1000x


# --- LOGRAT target space (T-REL, 2026-07-10) ----------------------------------------------
# The module `model` fixture trains in the LOGRAT default now, so every parity test above
# already exercises the predict_ms conversion end-to-end against BigramModelScorer. The
# tests below pin the semantics directly.


def _reencoded_lograt(base_model):
    """A second LOGRAT model encoding the SAME millisecond times as ``base_model``, built
    by hand-applying the transform to base_model's ms output (the saved-artifact shape)."""
    import copy

    m2 = copy.deepcopy(base_model)
    m2.metadata.extra.setdefault("training", {})["target_space"] = "LOGRAT"
    ms_of = base_model.predict_ms

    def lograt_predict(X):
        from keybo.features.schema import BIGRAM_FEATURE_NAMES

        wpm = np.asarray(X)[:, BIGRAM_FEATURE_NAMES.index("wpm")]
        return np.log(np.maximum(ms_of(X), 1.0) * wpm / 12000.0)

    m2.predict = lograt_predict
    return m2


def test_lograt_reencoding_builds_the_identical_table(model, freqs):
    """Two models encoding the SAME times — one via its own booster, one via an explicit
    log(ms*wpm/12000) re-encoding — must build the same table: the scorer converts every
    prediction to ms before storing."""
    ref = TableBigramScorer(model, freqs, target_wpm=90.0, chars=QWERTY)
    re_enc = TableBigramScorer(_reencoded_lograt(model), freqs, target_wpm=90.0, chars=QWERTY)
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    assert re_enc.fitness(lay) == pytest.approx(ref.fitness(lay), rel=1e-6)


def test_lograt_table_rejects_zero_target_wpm(model, freqs):
    """target_wpm=0 makes the LOGRAT->ms conversion divide by zero; it must raise at
    construction, not emit inf into the table. (The module model IS LOGRAT-space.)"""
    assert model.target_space == "LOGRAT"
    with pytest.raises(ValueError, match="wpm"):
        TableBigramScorer(model, freqs, target_wpm=0.0, chars=QWERTY)
