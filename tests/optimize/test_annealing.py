"""Tests for simulated annealing.

The SA math (initial-temperature estimation, geometric cooling, coupon-collector stopping
criterion) is preserved from the original and pinned with golden values. Reproducibility
under a fixed seed is the regression guard for bug #11 (the old code never seeded its RNG).
"""

from math import ceil, log

import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.optimize.annealing import SimulatedAnnealing, stopping_point
from keybo.scoring.base import IScorer

from .conftest import CharPlacementScorer

LAYOUT = "qwertyuiopasdfghjkl'zxcvbnm,.-"

# Real char->position landscape lives in conftest so both optimizer test modules share it.
PositionSumScorer = CharPlacementScorer


def test_stopping_point_matches_coupon_collector_formula():
    # k=30 keys -> C(30,2)=435 pairs. ceil(435*(ln435 + gamma) + 0.5).
    n = 435
    gamma = 0.5772156649
    expected = ceil(n * (log(n) + gamma) + 0.5)
    assert stopping_point(30) == expected


def test_stopping_point_is_2895_for_30_keys():
    # Golden value cited in the paper's derivation.
    assert stopping_point(30) == 2895


def test_initial_temperature_is_positive_and_finite():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=0)
    t0 = sa.estimate_initial_temperature(lay, PositionSumScorer(), acceptance=0.8)
    assert t0 > 0
    assert t0 != float("inf")


def test_cooling_multiplies_temperature_by_alpha():
    sa = SimulatedAnnealing(seed=0, alpha=0.9)
    assert sa.cool(100.0) == pytest.approx(90.0)


def test_optimize_returns_a_valid_layout():
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=0, alpha=0.9)
    best = sa.optimize(lay, PositionSumScorer())
    assert sorted(best.chars) == sorted(LAYOUT)  # still a permutation


def test_optimize_improves_on_a_bad_start():
    scorer = PositionSumScorer()
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    start = scorer.fitness(lay)
    sa = SimulatedAnnealing(seed=0, alpha=0.9)
    best = sa.optimize(lay, scorer)
    # QWERTY is far from optimal for this scorer; SA must strictly improve it.
    assert scorer.fitness(best) < start


def test_regression_bug11_same_seed_is_reproducible():
    scorer = PositionSumScorer()
    results = []
    for _ in range(2):
        lay = Layout(LAYOUT, ROW_STAGGERED_30)
        sa = SimulatedAnnealing(seed=123, alpha=0.9)
        results.append("".join(sa.optimize(lay, scorer).chars))
    assert results[0] == results[1]


def test_different_seeds_explore_different_paths():
    # The seed must actually steer the search: across several seeds we expect more than one
    # distinct resulting layout (this landscape has many optima). Combined with the
    # reproducibility test above, this shows the RNG is both seeded and effective.
    scorer = PositionSumScorer()
    outs = set()
    for seed in range(8):
        lay = Layout(LAYOUT, ROW_STAGGERED_30)
        sa = SimulatedAnnealing(seed=seed, alpha=0.9)
        outs.add("".join(sa.optimize(lay, scorer).chars))
    assert len(outs) > 1


class ConstantScorer(IScorer):
    """Every layout scores the same -> every swap has delta == 0 (a total plateau)."""

    def fitness(self, layout: Layout) -> float:
        return 42.0


def test_regression_terminates_on_a_plateau():
    """A landscape of all-equal fitness must still terminate.

    Bug: delta == 0 was accepted via the Metropolis branch, which decremented the
    convergence counter, so it could never reach the stopping point. A max_outer cap plus
    treating delta == 0 as non-improving guarantees termination.
    """
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=0, alpha=0.9, max_outer=500)
    best = sa.optimize(lay, ConstantScorer())  # must return, not hang
    assert sorted(best.chars) == sorted(LAYOUT)


def test_max_outer_bounds_the_run():
    # With a tiny cap the run returns quickly regardless of convergence.
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=0, alpha=0.9, max_outer=5)
    best = sa.optimize(lay, ConstantScorer())
    assert sorted(best.chars) == sorted(LAYOUT)


def test_optimize_does_not_leave_layout_dirty():
    # After optimize, the returned best must be internally consistent (pos<->key inverse).
    lay = Layout(LAYOUT, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=7, alpha=0.9)
    best = sa.optimize(lay, PositionSumScorer())
    for ch in best.chars:
        assert best.key_at(*best.pos(ch)) == ch
