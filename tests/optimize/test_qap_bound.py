"""Gilmore–Lawler lower bound for the QAP objective (gaps-audit 5.1).

The certificate that turns "best we found" into "within X% of the best possible": a
valid lower bound L on min-over-permutations fitness means a found layout with fitness F
is certifiably within (F - L)/L of optimal. GL is classic: for each (char i, slot k),
compute the cheapest possible interaction cost given i sits at k (a sorted-dot-product
minimization over the remaining assignment), then solve one linear assignment over those
per-pair floors. Validity is testable: the bound must NEVER exceed the true optimum
(checked exhaustively on tiny instances) and must be exact for separable cost structures.
"""

import itertools

import numpy as np
import pytest

from keybo.optimize.qap_bound import gilmore_lawler_bound, qap_fitness


def brute_force_min(F, T):
    n = F.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(n)):
        p = np.array(perm)
        best = min(best, qap_fitness(F, T, p))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bound_never_exceeds_true_optimum_random_small(seed):
    rng = np.random.default_rng(seed)
    n = 6
    F = rng.uniform(0, 10, (n, n))
    T = rng.uniform(50, 250, (n, n))
    lb = gilmore_lawler_bound(F, T)
    opt = brute_force_min(F, T)
    assert lb <= opt + 1e-9
    # And it should be a USEFUL bound, not trivially zero.
    assert lb > 0


def test_bound_is_tight_for_uniform_costs():
    # If T is constant, every permutation has identical fitness -> bound == optimum.
    n = 5
    F = np.arange(n * n, dtype=float).reshape(n, n)
    T = np.full((n, n), 3.0)
    lb = gilmore_lawler_bound(F, T)
    assert lb == pytest.approx(brute_force_min(F, T))


def test_qap_fitness_matches_direct_sum():
    rng = np.random.default_rng(7)
    n = 4
    F = rng.uniform(0, 5, (n, n))
    T = rng.uniform(1, 9, (n, n))
    perm = np.array([2, 0, 3, 1])
    direct = sum(F[i, j] * T[perm[i], perm[j]] for i in range(n) for j in range(n))
    assert qap_fitness(F, T, perm) == pytest.approx(direct)


def test_certificate_gap_reported():
    from keybo.optimize.qap_bound import certificate

    rng = np.random.default_rng(3)
    n = 6
    F = rng.uniform(0, 10, (n, n))
    T = rng.uniform(50, 250, (n, n))
    opt = brute_force_min(F, T)
    cert = certificate(F, T, found_fitness=opt * 1.02)
    assert cert["lower_bound"] <= opt + 1e-9
    assert cert["gap_pct"] >= 2.0 - 1e-6  # found is 2% above true opt, bound <= opt
