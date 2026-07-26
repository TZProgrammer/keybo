"""Tests for the optimizing-the-ruler GUARD itself.

The guard is the deliverable. If it cannot fire, a "GENUINE" verdict is worthless — so these
tests drive it with SYNTHETIC head-to-head data covering each branch, and check that the verbatim
phrase the task requires appears exactly when the win exists only on the trained gauge.

Also pinned: the constrained-champion selection rule, because it is the thing that stops MY
selection from deciding the verdict instead of the search.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus_eval as CE  # noqa: E402
import wscissor_eval as WE  # noqa: E402
import wscissor_score as WS  # noqa: E402

RULER_PHRASE = "OPTIMIZING THE RULER"


# -- the verdict function ---------------------------------------------------------------------
def test_wins_only_on_trained_gauge_is_named_optimizing_the_ruler():
    verdict = WS.guard_verdict(beats_wide=True, beats_narrow=False, holds_others=True)
    assert RULER_PHRASE in verdict
    assert "wide gauge it was trained on" in verdict
    # and it says so even when everything else holds — winning the trained gauge alone is the
    # exact condition, not a tiebreak.
    assert RULER_PHRASE in WS.guard_verdict(True, False, False)


def test_wins_on_both_and_holds_is_genuine():
    verdict = WS.guard_verdict(beats_wide=True, beats_narrow=True, holds_others=True)
    assert verdict.startswith("GENUINE")
    assert RULER_PHRASE not in verdict


def test_wins_on_both_but_regresses_elsewhere_is_partial_not_genuine():
    verdict = WS.guard_verdict(beats_wide=True, beats_narrow=True, holds_others=False)
    assert verdict.startswith("PARTIAL")
    assert "GENUINE" not in verdict
    assert RULER_PHRASE not in verdict


def test_failing_the_trained_gauge_is_no_win_not_ruler():
    """Losing even on wide is NOT optimizing the ruler — it is a plain negative. Conflating the
    two would mislabel a search that simply failed."""
    for holds in (True, False):
        verdict = WS.guard_verdict(beats_wide=False, beats_narrow=False, holds_others=holds)
        assert verdict.startswith("NO WIN")
        assert RULER_PHRASE not in verdict


def test_every_branch_is_reachable_and_distinct():
    verdicts = {
        WS.guard_verdict(w, n, o)
        for w in (True, False)
        for n in (True, False)
        for o in (True, False)
    }
    assert "UNCLASSIFIED" not in verdicts, "an unreachable branch means an unhandled case"
    assert len(verdicts) == 4


# -- the constrained champion -----------------------------------------------------------------
@pytest.fixture(scope="module")
def board() -> WE.WScissorBoard:
    ceilings = CE.SixSurface("iweb").ceiling_map
    return WE.WScissorBoard(corpus="iweb", arm="A", ceilings=ceilings, objective="wide")


@pytest.fixture(scope="module")
def incumbents(board) -> dict:
    return {
        name: board.axes12(string) | board.severity_axes(string)
        for name, string in CE.INCUMBENTS.items()
    }


def test_every_incumbent_is_feasible_against_the_weakest_incumbent_floor(board, incumbents):
    """Sanity floor: the constraint is 'no worse than the WEAKEST incumbent', so every incumbent
    must itself satisfy it. If one did not, the floor would be mis-derived (a sign error)."""
    result = WS.best_on_constrained(board, list(CE.INCUMBENTS.values()), "wscissor_P", incumbents)
    assert result["n_feasible_in_archive"] == len(CE.INCUMBENTS)
    assert result["layout"] is not None


def test_constrained_selection_rejects_a_ruler_corner(board, incumbents):
    """A layout that is superb on wide and awful on the board's own axes must be EXCLUDED.

    qwerty is the canonical such corner on the board axes. Adding it to a candidate pool must
    not make it the constrained champion even if its wide score were attractive.
    """
    pool = [*CE.INCUMBENTS.values(), "qwertyuiopasdfghjkl'zxcvbnm,.-"]
    result = WS.best_on_constrained(board, pool, "wscissor_P", incumbents)
    assert result["layout"] != "qwertyuiopasdfghjkl'zxcvbnm,.-"
    assert result["n_feasible_in_archive"] == len(CE.INCUMBENTS)


def test_constrained_returns_none_when_nothing_is_feasible(board, incumbents):
    result = WS.best_on_constrained(
        board, ["qwertyuiopasdfghjkl'zxcvbnm,.-"], "wscissor_P", incumbents
    )
    assert result["layout"] is None
    assert result["n_feasible_in_archive"] == 0


def test_constraint_floors_are_recorded_in_oriented_units(board, incumbents):
    result = WS.best_on_constrained(board, list(CE.INCUMBENTS.values()), "wscissor_P", incumbents)
    floors = result["constraint"]["floors"]
    for axis, floor in floors.items():
        oriented = [WE.SIGN12[axis] * inc[axis] for inc in incumbents.values()]
        assert floor == min(oriented), f"{axis} floor is not the weakest incumbent"


# -- the cheap prefilter (an EXACT optimization, so it must lose nothing) ----------------------
def test_cheap_axes_recovered_from_stored_objectives_match_the_board(board):
    """The prefilter reads the six cheap axes out of stored EA objective vectors instead of
    recomputing them. If that recovery were wrong (a sign slip on the two negated maxima) the
    filter would silently drop real dominators."""
    import numpy as np

    movables = np.array([[CE.C30M.index(c) for c in CE.C30M]])
    for name, string in CE.INCUMBENTS.items():
        movables = np.array([[string.index(c) for c in CE.C30M]])
        objs = board.evaluate_batch(movables)[0].tolist()
        recovered = WS.cheap_axes_from_objs(objs)
        actual = board.axes(string)
        for axis, value in recovered.items():
            assert value == pytest.approx(actual[axis], rel=0, abs=1e-9), (
                f"{name}/{axis}: recovered {value} != board {actual[axis]}"
            )


def test_prefilter_keeps_every_dominator(board, incumbents):
    """Exhaustive equivalence on a real candidate pool: the set of 12-axis dominators found
    among prefilter survivors equals the set found by scanning everything.

    This is what licenses the 500x speedup. The prefilter is only sound because the six cheap
    axes are a SUBSET of the 12-axis frame, so failing one already precludes dominance.
    """
    import numpy as np

    rng = np.random.default_rng(31337)
    pool = {}
    for string in [*CE.INCUMBENTS.values(), "qwertyuiopasdfghjkl'zxcvbnm,.-"]:
        movables = np.array([[string.index(c) for c in CE.C30M]])
        pool[string] = board.evaluate_batch(movables)[0].tolist()
    for _ in range(120):  # random layouts, to populate the "fails the filter" side
        string = "".join(rng.permutation(list(CE.C30M)))
        movables = np.array([[string.index(c) for c in CE.C30M]])
        pool[string] = board.evaluate_batch(movables)[0].tolist()

    for name, inc in incumbents.items():
        brute = {
            layout
            for layout in pool
            if WE.dominates12(board.axes12(layout) | board.severity_axes(layout), inc)[0]
        }
        survivors = WS.cheap_prefilter(pool, inc)
        filtered = {
            layout
            for layout in survivors
            if WE.dominates12(board.axes12(layout) | board.severity_axes(layout), inc)[0]
        }
        assert filtered == brute, f"prefilter changed the dominator set for {name}"
        assert set(survivors) >= brute, f"prefilter dropped a dominator for {name}"


def test_prefilter_actually_filters(board, incumbents):
    """A filter that keeps everything is sound but useless; assert it really discriminates."""
    import numpy as np

    rng = np.random.default_rng(99)
    pool = {}
    for _ in range(200):
        string = "".join(rng.permutation(list(CE.C30M)))
        movables = np.array([[string.index(c) for c in CE.C30M]])
        pool[string] = board.evaluate_batch(movables)[0].tolist()
    survivors = WS.cheap_prefilter(pool, incumbents["keybo-lsb"])
    assert len(survivors) < len(pool)


# -- head-to-head plumbing --------------------------------------------------------------------
def test_head_to_head_reads_the_same_layout_on_every_gauge(board, incumbents):
    """The guard's core discipline: one champion, read on all four severity gauges. If the
    champion differed per gauge the tradeoff the guard hunts for would be invisible."""
    champion = CE.INCUMBENTS["archive-1843"]
    h2h = WS.head_to_head_vs(board, champion, incumbents)
    assert set(h2h) == set(CE.INCUMBENTS)
    for row in h2h.values():
        assert row["champion_layout"] == champion
    # the champion's own reading must be identical across every incumbent row
    for key in WS.SEVERITY_KEYS:
        values = {row[key]["champion"] for row in h2h.values()}
        assert len(values) == 1


def test_a_layout_never_beats_itself(board, incumbents):
    """archive-1843 vs archive-1843 must tie on all four gauges and dominate nothing."""
    h2h = WS.head_to_head_vs(board, CE.INCUMBENTS["archive-1843"], incumbents)
    row = h2h["archive-1843"]
    for key in WS.SEVERITY_KEYS:
        assert row[key]["beats"] is False
        assert row[key]["champion"] == row[key]["incumbent"]
    assert row["dominance_12axis"]["dominates"] is False
    assert row["dominance_12axis"]["n_strictly_better"] == 0


def test_archive_1843_is_recorded_as_beating_others_on_wide_where_it_does(board, incumbents):
    """Directional smoke check against the step-1 reproduction: on iWeb at P, archive-1843 has
    the lowest wide share among the five incumbents, so it must beat all four others on wide."""
    h2h = WS.head_to_head_vs(board, CE.INCUMBENTS["archive-1843"], incumbents)
    others = [n for n in CE.INCUMBENTS if n != "archive-1843"]
    assert all(h2h[n]["wscissor_P"]["beats"] for n in others)
