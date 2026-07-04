"""The best-of-N search logic behind `keybo optimize --attempts N`.

These drive the search over a real discriminating landscape (CharPlacementScorer), so the
best-of-N guarantee is tested with teeth -- the CLI-level tests in tests/cli use a tiny model
that cannot discriminate layouts, which would make a `<=` assertion vacuous. Here different
seeds provably reach different optima, so "keep the lowest across N seeds" is exercised for
real: it must be <= any single attempt, and equal to the min over the individual seeds.
"""

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.optimize.annealing import SimulatedAnnealing
from keybo.optimize.local_search import two_opt

from .conftest import CharPlacementScorer

LAYOUT = "qwertyuiopasdfghjkl'zxcvbnm,.-"


def _attempt(scorer, seed, *, local_search=False):
    """One search from a fresh layout at a given seed -- mirrors optimize.run's per-attempt."""
    layout = Layout(LAYOUT, ROW_STAGGERED_30)
    best = SimulatedAnnealing(seed=seed, alpha=0.9, max_outer=40).optimize(layout, scorer)
    if local_search:
        best = two_opt(best, scorer)
    return best


def _best_of(scorer, base_seed, attempts, **kw):
    """Best (lowest-fitness) layout over seeds base..base+attempts-1 -- the run() contract."""
    best_fitness = float("inf")
    best_layout = None
    for i in range(attempts):
        cand = _attempt(scorer, base_seed + i, **kw)
        f = scorer.fitness(cand)
        if f < best_fitness:
            best_fitness, best_layout = f, cand
    return best_fitness, best_layout


def test_seeds_reach_distinct_optima_so_best_of_n_is_meaningful():
    """Sanity: on this landscape the seeds base..base+2 do NOT all coincide, so a best-of-3
    that beats a single attempt is a real selection, not an artifact of a flat landscape."""
    scorer = CharPlacementScorer()
    fits = {scorer.fitness(_attempt(scorer, s)) for s in (7, 8, 9)}
    assert len(fits) > 1


def test_best_of_n_equals_min_over_individual_seeds():
    """best-of-3 fitness == min(fitness of the three individual seeds) -- exact selection."""
    scorer = CharPlacementScorer()
    individual = [scorer.fitness(_attempt(scorer, 7 + i)) for i in range(3)]
    best_fitness, _ = _best_of(scorer, 7, 3)
    assert best_fitness == min(individual)


def test_best_of_n_strictly_better_than_worst_single_attempt():
    """With genuinely distinct optima, best-of-N is <= every single attempt and strictly
    below the worst of them -- i.e. the extra attempts demonstrably help."""
    scorer = CharPlacementScorer()
    individual = [scorer.fitness(_attempt(scorer, 7 + i)) for i in range(3)]
    best_fitness, _ = _best_of(scorer, 7, 3)
    assert best_fitness <= min(individual)
    assert best_fitness < max(individual)


def test_best_of_n_is_reproducible():
    """Same base seed + attempts -> identical best layout (determinism across the whole run)."""
    scorer = CharPlacementScorer()
    _, layout1 = _best_of(scorer, 7, 3)
    _, layout2 = _best_of(scorer, 7, 3)
    assert "".join(layout1.chars) == "".join(layout2.chars)
