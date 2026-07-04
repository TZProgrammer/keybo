"""QAP-table scorer: the exact model objective, reduced to a position-pair table.

When the model's ``freq`` feature is inert (trained on a constant — the OQ-1 outcome) and
the scoring WPM is fixed, a bigram's predicted time depends ONLY on its two key positions.
There are just 31 positions (30 slots + the fixed space key), so the whole objective
collapses to::

    fitness(layout) = sum_{c1,c2} F[c1, c2] * T[slot(c1), slot(c2)]

with ``F`` the corpus bigram-frequency matrix over the charset and ``T`` the model's
961-entry position-pair time table. That is a Quadratic Assignment Problem: evaluating a
layout is a fancy-indexed sum (microseconds) instead of a full model predict over the
corpus (~25 ms), which is what makes deep search (millions of evaluations) feasible.

The reduction is only valid for freq-inert models, so the constructor PROBES the model
(predict at freq=1 vs freq=1e7) and refuses frequency-sensitive ones rather than silently
optimizing a different objective than :class:`BigramModelScorer` scores.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from keybo.features import bigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.layout import Layout
from keybo.scoring.base import IScorer


class TableBigramScorer(IScorer):
    """Bit-for-bit :class:`BigramModelScorer`, ~1000x faster, for a FIXED charset.

    ``chars`` fixes the assignable character set up front (the corpus rows outside it are
    excluded exactly as the model scorer's ``has_key`` filter would). Every scored layout
    must carry exactly that charset — permutations of it, which is what an optimizer
    explores — else scoring raises instead of silently mis-scoring.
    """

    def __init__(
        self,
        model,
        bigram_freqs: Mapping[str, int],
        target_wpm: float = 0.0,
        chars: str | None = None,
        geometry: Geometry = ROW_STAGGERED_30,
    ) -> None:
        if chars is None:
            raise ValueError("TableBigramScorer requires the fixed charset (chars=...)")
        self._chars = tuple(chars)
        self._geometry = geometry
        positions = [*geometry.slots, geometry.space_position]
        n_pos = len(positions)

        # Guard: the table reduction assumes predictions are invariant to the freq feature.
        pa, pb = positions[0], positions[1]
        probe = np.vstack(
            [
                bigram_features_from_positions(geometry, (pa, pb), freq=f, wpm=target_wpm)
                for f in (1.0, 1e7)
            ]
        )
        lo, hi = model.predict(probe)
        if abs(float(lo) - float(hi)) > 1e-9:
            raise ValueError(
                "model predictions depend on the freq feature; the position-pair table "
                "reduction is invalid — use BigramModelScorer (or retrain freq-inert)"
            )

        # T: predicted time for every ordered position pair, at the scoring WPM.
        vectors = np.vstack(
            [
                bigram_features_from_positions(geometry, (a, b), freq=1.0, wpm=target_wpm)
                for a in positions
                for b in positions
            ]
        )
        self._T = model.predict(vectors).reshape(n_pos, n_pos)

        # F: corpus frequency between character indices (charset + space at index n-1).
        charset = set(self._chars) | {" "}
        self._index = {c: i for i, c in enumerate(self._chars)}
        self._index[" "] = len(self._chars)
        F = np.zeros((n_pos, n_pos))
        for bg, freq in bigram_freqs.items():
            if len(bg) == 2 and all(c in charset for c in bg):
                F[self._index[bg[0]], self._index[bg[1]]] += freq
        self._F = F
        self._slot_index = {pos: i for i, pos in enumerate(geometry.slots)}
        self._space_slot = len(geometry.slots)

    def permutation(self, layout: Layout) -> np.ndarray:
        """char-index -> position-index vector for ``layout`` (space pinned last)."""
        if set(layout.chars) != set(self._chars):
            raise ValueError(
                "layout charset differs from the table's charset; the table excludes "
                "different corpus rows than this layout would — rebuild the scorer"
            )
        perm = np.empty(len(self._chars) + 1, dtype=np.intp)
        for i, c in enumerate(self._chars):
            perm[i] = self._slot_index[layout.pos(c)]
        perm[len(self._chars)] = self._space_slot
        return perm

    def fitness_of_permutation(self, perm: np.ndarray) -> float:
        """Fitness from a raw permutation vector (the search-loop fast path)."""
        return float((self._F * self._T[np.ix_(perm, perm)]).sum())

    def fitness(self, layout: Layout) -> float:
        return self.fitness_of_permutation(self.permutation(layout))
