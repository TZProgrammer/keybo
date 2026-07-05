"""Gilmore–Lawler lower bound for the layout QAP (gaps-audit 5.1).

fitness(perm) = Σ_{i,j} F[i,j] · T[perm[i], perm[j]] over character pairs (i,j) and their
assigned slots. The GL bound composes two relaxations, each individually valid:

1. For char i at slot k, the interaction cost Σ_{j≠i} F[i,j]·T[k, perm[j]] is bounded
   below by the sorted dot product: sort the i-th row of F descending against the k-th
   row of T ascending (rearrangement inequality) — the cheapest conceivable completion,
   ignoring consistency between rows.
2. Those per-(i,k) floors form a linear assignment problem; its optimum (Hungarian
   method, scipy) is a valid lower bound on the full quadratic objective.

The diagonal (i==j: a char with itself — 'ee'-style repeats at T[k,k]) is exact per
assignment and rides in the LAP cost directly.

Classic reference: Gilmore (1962), Lawler (1963). The bound is loose on hard instances
but perfectly valid — a certificate "found layout is within (F−L)/L of optimal" that no
amount of restarts can fake.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def qap_fitness(F: np.ndarray, T: np.ndarray, perm: np.ndarray) -> float:
    """Σ F[i,j] · T[perm[i], perm[j]] — the exact objective for a full permutation."""
    p = np.asarray(perm)
    return float((F * T[np.ix_(p, p)]).sum())


def _sorted_dot_min(f_row: np.ndarray, t_row: np.ndarray) -> float:
    """Min over pairings of Σ f·t: largest f against smallest t (rearrangement ineq.)."""
    return float(np.sort(f_row)[::-1] @ np.sort(t_row))


def gilmore_lawler_bound(F: np.ndarray, T: np.ndarray) -> float:
    """A valid lower bound on min_perm qap_fitness(F, T, perm)."""
    n = F.shape[0]
    if F.shape != (n, n) or T.shape != (n, n):
        raise ValueError("F and T must be square and same-shaped")
    off = ~np.eye(n, dtype=bool)
    cost = np.empty((n, n))
    # Each F[i,j]·T[·,·] term appears exactly once as outgoing-of-i and once as
    # incoming-of-j across the whole objective, so bounding outgoing and incoming
    # separately per (i,k) and HALVING their sum yields a valid floor on the total
    # (each side is independently a rearrangement-inequality minimum).
    for i in range(n):
        f_out = F[i][off[i]]
        f_in = F[:, i][off[:, i]]
        for k in range(n):
            t_out = T[k][off[k]]
            t_in = T[:, k][off[:, k]]
            cost[i, k] = F[i, i] * T[k, k] + 0.5 * (
                _sorted_dot_min(f_out, t_out) + _sorted_dot_min(f_in, t_in)
            )
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def certificate(F: np.ndarray, T: np.ndarray, found_fitness: float) -> dict:
    """Optimality certificate: how far can the found layout possibly be from optimal?"""
    lb = gilmore_lawler_bound(F, T)
    gap = (found_fitness - lb) / lb * 100 if lb > 0 else float("inf")
    return {
        "lower_bound": lb,
        "found_fitness": float(found_fitness),
        "gap_pct": float(gap),
        "statement": (
            f"the found layout is within {gap:.2f}% of the best possible layout "
            "(Gilmore-Lawler certificate; the true optimum lies between the bound and the found value)"
        ),
    }
