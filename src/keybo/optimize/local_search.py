"""Exhaustive local search: 2-opt (swap pairs) and 3-opt (reorder triples).

Both repeatedly scan for an improving move and apply the first one found, restarting until
no move improves (a local optimum). They rely on ``Layout.undo`` to reverse rejected moves;
because undo is stack-based, the multi-swap moves 3-opt uses are always fully reversed —
the property the original 3-opt got wrong (it reversed only the last swap of a pair,
silently corrupting the board on rejection).
"""

from __future__ import annotations

from itertools import combinations, permutations

from keybo.layout import Layout
from keybo.scoring.base import IScorer


def two_opt(layout: Layout, scorer: IScorer) -> Layout:
    """Greedy 2-opt: accept any single swap that improves fitness, until none does."""
    improved = True
    while improved:
        improved = False
        chars = list(layout.chars)
        for i, j in combinations(range(len(chars)), 2):
            current = scorer.fitness(layout)
            layout.swap(chars[i], chars[j])
            if scorer.fitness(layout) < current:
                improved = True
                break  # restart the scan from a fresh snapshot
            layout.undo()
        # loop continues if we improved; otherwise we've hit a local optimum
    return layout


# Target orderings of a triple (a, b, c) at positions (i, j, k) -> the swap sequence (on
# positions) that produces them. Verified: applying these swaps to [a,b,c] yields the key.
_TRIPLE_MOVES = {
    ("b", "a", "c"): [("i", "j")],
    ("a", "c", "b"): [("j", "k")],
    ("c", "b", "a"): [("i", "k")],
    ("b", "c", "a"): [("i", "j"), ("j", "k")],
    ("c", "a", "b"): [("i", "k"), ("j", "k")],
}


def three_opt(layout: Layout, scorer: IScorer) -> Layout:
    """Greedy 3-opt: try every non-identity reordering of each triple of keys."""
    improved = True
    while improved:
        improved = False
        n = len(layout.chars)
        for i, j, k in combinations(range(n), 3):
            if _try_improve_triple(layout, scorer, i, j, k):
                improved = True
                break
    return layout


def _try_improve_triple(layout: Layout, scorer: IScorer, i: int, j: int, k: int) -> bool:
    """Try each reordering of the triple at (i, j, k); apply the first that improves."""
    current = scorer.fitness(layout)
    orig = list(layout.chars)
    pos_of = {"i": i, "j": j, "k": k}

    for target in permutations(("a", "b", "c")):
        if target == ("a", "b", "c"):
            continue
        moves = _TRIPLE_MOVES[target]

        applied = 0
        for p1, p2 in moves:
            chars = list(layout.chars)
            layout.swap(chars[pos_of[p1]], chars[pos_of[p2]])
            applied += 1

        if scorer.fitness(layout) < current:
            return True
        for _ in range(applied):
            layout.undo()

    # After trying and rejecting every reordering, the layout must be exactly as it started.
    assert list(layout.chars) == orig
    return False
