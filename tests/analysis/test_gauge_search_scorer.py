"""`TimeSurface.triple_ms_table` + `gauge_search_scorer`: the gauge as a searchable objective.

WHAT DEFECT THIS PREVENTS: the campaign's reported ruler (``analyze``'s ms/char) lived only
inside ``TimeSurface.card()``, a ~50 ms-per-layout Python loop over 114,920 corpus rows. Any
attempt to SEARCH that ruler therefore had to re-implement it, and two agent drivers did —
each re-deriving the corpus indexing, the pinned-space slot and the coverage denominator by
hand. A re-implementation that is 1.5e-2 off (the naive reading: ``bigrams.txt`` weights and a
single model seed instead of the trigram marginal and the 3-seed mean) still looks entirely
plausible, ranks boards in nearly the right order, and is ~11 resolution floors wrong.

So the gauge gets ONE definition that both roles read:

* ``TimeSurface.triple_ms_table()`` exposes ``T2[a,b] + Tcond[a,b,c]`` as a single
  (31,31,31) millisecond table — the exact quantity ``card()`` accumulates;
* ``gauge_search_scorer()`` hands that table to the SHIPPED ``TableTrigramScorer`` evaluator,
  so the fast path is the reviewed, bit-exact-parity-tested code and not a fourth copy.

These tests pin the equivalence at the table level, where a mis-indexed axis or a dropped
seed is a direct assertion rather than a downstream ranking difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring import model_norm as MN
from keybo.scoring.table_trigram import TableTrigramScorer

C30M = MN.S.C30M


@pytest.fixture(scope="module")
def surface():
    from keybo.analysis.timecard import default_surface

    return default_surface(90.0)


@pytest.mark.slow
def test_triple_ms_table_is_the_bigram_table_broadcast_over_the_trigram_increment(surface):
    """``triple[a,b,c] == T2[a,b] + Tc[a,b,c]`` for every one of the 29,791 triples.

    The broadcast axis is the defect surface: ``T2`` is indexed by the FIRST TWO keys, so it
    must be added along the third axis. Broadcasting over the wrong axis produces a table of
    the right shape and the right magnitude that silently charges the ``a->b`` transition to
    the wrong pair — a bug no ranking check would localize.
    """
    triple = surface.triple_ms_table()
    assert triple.shape == (31, 31, 31)
    expected = surface._T2[:, :, None] + surface._Tc
    assert np.array_equal(triple, expected)
    # Spot-check one entry against the scalar arithmetic, independent of the broadcast idiom.
    assert triple[3, 7, 11] == pytest.approx(surface._T2[3, 7] + surface._Tc[3, 7, 11])


@pytest.mark.slow
def test_triple_ms_table_is_a_copy_so_a_caller_cannot_corrupt_the_cached_surface(surface):
    """Mutating the returned table must not poison the ``lru_cache``d surface.

    ``default_surface`` is cached for the process, so handing out a view of ``_T2``/``_Tc``
    would let one caller's in-place normalization change every later ``analyze`` number in the
    same run — a cross-contamination that no single test of either caller could catch.
    """
    first = surface.triple_ms_table()
    first[0, 0, 0] = -12345.0
    second = surface.triple_ms_table()
    assert second[0, 0, 0] != -12345.0


@pytest.mark.slow
def test_gauge_search_scorer_is_the_shipped_trigram_table_evaluator(surface):
    """The factory reuses ``TableTrigramScorer`` rather than defining a fourth gauge copy.

    Pinned by type because the VALUE of reusing it is the reviewed evaluator: its
    charset guard, its pinned-space slot and its bit-exact ``TrigramModelScorer`` parity are
    already tested, and a bespoke class would silently drop all three.
    """
    from keybo.analysis.timecard import gauge_search_scorer

    scorer = gauge_search_scorer(chars=C30M, target_wpm=90.0)
    assert isinstance(scorer, TableTrigramScorer)


@pytest.mark.slow
def test_gauge_scorer_total_equals_the_analyzer_total_on_every_permutation_board(surface):
    """Total ms matches ``card().total_ms`` on the reference board and both C30M named boards.

    ``graphite`` and ``semimak`` are permutations of C30M, so the table keeps exactly the same
    corpus rows and the two paths must agree to floating-point noise. The tolerance (1e-12) is
    ~100x looser than the measured worst case (1.2e-14) and 10 orders tighter than the naive
    re-implementation this replaces (1.5e-2).
    """
    from keybo.analysis.timecard import gauge_search_scorer

    for board in (C30M, NAMED_LAYOUTS["graphite"], NAMED_LAYOUTS["semimak"]):
        scorer = gauge_search_scorer(chars=board, target_wpm=90.0)
        layout = Layout(board, ROW_STAGGERED_30)
        assert scorer.fitness(layout) == pytest.approx(surface.card(board).total_ms, rel=1e-12), (
            board
        )


@pytest.mark.slow
def test_gauge_scorer_ms_per_char_uses_the_analyzers_covered_mass_denominator(surface):
    """ms/char divides by COVERED corpus mass, matching ``card()``'s denominator.

    ``card()`` divides by the mass it actually summed, not by the corpus total (coverage is
    88.7% on C30M). Dividing by ``total_mass`` instead would scale every reported number by
    ~1.13 — a uniform factor that preserves RANKING, so no comparison test would catch it, and
    the published ms/char would simply be wrong.
    """
    from keybo.analysis.timecard import gauge_search_scorer

    scorer = gauge_search_scorer(chars=C30M, target_wpm=90.0)
    card = surface.card(C30M)
    layout = Layout(C30M, ROW_STAGGERED_30)

    assert scorer.ms_per_char(layout) == pytest.approx(card.ms_per_char, rel=1e-12)
    # And the denominator really is the covered mass, not the corpus total.
    covered = card.total_ms / card.ms_per_char
    assert covered == pytest.approx(surface.total_mass * card.coverage_pct / 100.0, rel=1e-9)
    assert covered < surface.total_mass


@pytest.mark.slow
def test_gauge_scorer_ranks_a_known_faster_board_below_qwerty(surface):
    """A direction check: the objective must be MINIMIZED, matching ``IScorer``'s contract.

    A sign or reciprocal error would still parity-match a single board's magnitude while
    turning the search into a pessimizer. ``semimak`` is 257.39 ms/char vs C30M's 264.14, so
    lower-is-better must put it first.
    """
    from keybo.analysis.timecard import gauge_search_scorer

    semimak = NAMED_LAYOUTS["semimak"]
    scorer = gauge_search_scorer(chars=C30M, target_wpm=90.0)
    faster = scorer.ms_per_char(Layout(semimak, ROW_STAGGERED_30))
    slower = scorer.ms_per_char(Layout(C30M, ROW_STAGGERED_30))
    assert faster < slower


@pytest.mark.slow
def test_gauge_scorer_refuses_a_layout_whose_charset_is_not_its_table(surface):
    """A foreign-charset board raises instead of being scored against the wrong kept rows.

    Inherited from ``TableTrigramScorer.permutation``; asserted here because the factory is a
    new entry point to it, and silently scoring a ``qwerty``-charset board against a
    C30M-charset table would produce a number that looks like ms/char and is not comparable to
    any published one.
    """
    from keybo.analysis.timecard import gauge_search_scorer

    scorer = gauge_search_scorer(chars=C30M, target_wpm=90.0)
    with pytest.raises(ValueError, match="charset"):
        scorer.fitness(Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30))


@pytest.mark.slow
def test_gauge_scorer_target_wpm_moves_the_objective(surface):
    """``target_wpm`` reaches the table, so the flag is effective and not merely accepted.

    The surface is evaluated at a WPM; a factory that dropped the argument would silently
    score every run at 90 while the result file recorded the user's value — the
    ``present != effective`` shape again.
    """
    from keybo.analysis.timecard import gauge_search_scorer

    layout = Layout(C30M, ROW_STAGGERED_30)
    at_90 = gauge_search_scorer(chars=C30M, target_wpm=90.0).ms_per_char(layout)
    at_60 = gauge_search_scorer(chars=C30M, target_wpm=60.0).ms_per_char(layout)
    assert at_90 != at_60


@pytest.mark.slow
def test_from_table_preserves_the_evaluators_corpus_row_filtering():
    """``TableTrigramScorer.from_table`` keeps exactly the rows the normal constructor keeps.

    The alternate constructor exists only to inject a precomputed table; if it diverged in
    which corpus rows it accumulates (length-3 rows whose every character is on the board,
    space included) the two construction paths would be different objectives wearing one class.
    """
    from keybo.analysis.timecard import _load_gz_model

    freqs = {"the": 100, "th": 7, "quux": 3, "a b": 11, "ZZZ": 5}
    model = _load_gz_model("trigram_cond31_seed0")
    normal = TableTrigramScorer(model, freqs, target_wpm=90.0, chars=C30M)
    injected = TableTrigramScorer.from_table(
        normal._T3, freqs, chars=C30M, geometry=ROW_STAGGERED_30
    )

    assert np.array_equal(injected._i, normal._i)
    assert np.array_equal(injected._j, normal._j)
    assert np.array_equal(injected._l, normal._l)
    assert np.array_equal(injected._f, normal._f)
    layout = Layout(C30M, ROW_STAGGERED_30)
    assert injected.fitness(layout) == pytest.approx(normal.fitness(layout), rel=1e-15)
    # "the" and "a b" are on-board length-3 rows; "th"/"quux"/"ZZZ" are not.
    assert injected._f.tolist() == [100.0, 11.0]
