"""Tests for 2-opt and 3-opt local search.

The key correctness property (which the old 3-opt violated via broken multi-swap undo) is
that every rejected move is fully undone, so the layout always stays a valid permutation
and the returned layout's score is never worse than the start.
"""

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.optimize.local_search import three_opt, two_opt

from .conftest import CharPlacementScorer

LAYOUT = "qwertyuiopasdfghjkl'zxcvbnm,.-"

# Shared real assignment landscape (see conftest).
PositionSumScorer = CharPlacementScorer


def test_two_opt_strictly_improves_a_suboptimal_start():
    scorer = PositionSumScorer()
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    start = scorer.fitness(lay)
    best = two_opt(lay, scorer)
    # QWERTY is not optimal for this scorer, so 2-opt must strictly improve it.
    assert scorer.fitness(best) < start


def test_two_opt_reaches_a_local_optimum():
    # At the returned layout, no single swap improves fitness.
    scorer = PositionSumScorer()
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    best = two_opt(lay, scorer)
    base = scorer.fitness(best)
    chars = list(best.chars)
    for i in range(len(chars)):
        for j in range(i + 1, len(chars)):
            best.swap(chars[i], chars[j])
            assert scorer.fitness(best) >= base
            best.undo()


def test_two_opt_keeps_layout_a_valid_permutation():
    scorer = PositionSumScorer()
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    best = two_opt(lay, scorer)
    assert sorted(best.chars) == sorted(LAYOUT)


def test_regression_three_opt_keeps_valid_permutation():
    """Bug #1 downstream: 3-opt applies 2-swap moves; a rejected one must fully undo.

    The old undo left q/w/e rotated on rejection. Here, after a full 3-opt pass the layout
    must still contain exactly the original characters with no duplicates or losses.
    """
    scorer = PositionSumScorer()
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    best = three_opt(lay, scorer)
    assert sorted(best.chars) == sorted(LAYOUT)
    # And the mapping is internally consistent.
    assert len({best.pos(c) for c in best.chars}) == 30


def test_three_opt_never_worsens_fitness():
    scorer = PositionSumScorer()
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    start = scorer.fitness(lay)
    best = three_opt(lay, scorer)
    assert scorer.fitness(best) <= start
