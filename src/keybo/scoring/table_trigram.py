"""Trigram QAP-table scorer: the exact trigram objective at table speed.

Features are pure geometry + wpm (post-OQ-1 schema), so at a fixed scoring WPM a
trigram's predicted time depends ONLY on its three key positions. With 31 positions
(30 slots + fixed space) that is a 31^3 = 29,791-entry table — one batch predict at
construction, then::

    fitness(layout) = sum_k f_k * T3[perm[i_k], perm[j_k], perm[l_k]]

over the corpus trigrams k with character indices (i, j, l). Evaluation is a fancy-index
sum (~microseconds), which is what makes multi-restart trigram search feasible; the
LOLO-validated trigram model (PREREGISTRATIONS.md, 2026-07-05) is what makes it licensed.

Unlike the bigram case, the cubic objective admits no Gilmore-Lawler-style certificate
here (that machinery is quadratic); certificates remain bigram-only for now.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.layout import Layout
from keybo.scoring.base import IScorer


class TableTrigramScorer(IScorer):
    """Bit-for-bit :class:`TrigramModelScorer`, table-speed, for a FIXED charset."""

    def __init__(
        self,
        model,
        trigram_freqs: Mapping[str, int],
        target_wpm: float = 0.0,
        chars: str | None = None,
        geometry: Geometry = ROW_STAGGERED_30,
    ) -> None:
        if chars is None:
            raise ValueError("TableTrigramScorer requires the fixed charset (chars=...)")
        self._chars = tuple(chars)
        self._geometry = geometry
        positions = [*geometry.slots, geometry.space_position]
        n = len(positions)

        # T3: predicted time for every ordered position triple, one batch predict. As with
        # the bigram table, entries are MILLISECONDS (target-space-aware conversion).
        vectors = np.vstack(
            [
                trigram_features_from_positions(geometry, (a, b, c), wpm=target_wpm)
                for a in positions
                for b in positions
                for c in positions
            ]
        )
        from keybo.scoring.model_scorer import predict_ms

        self._T3 = predict_ms(model, vectors).reshape(n, n, n)

        charset = set(self._chars) | {" "}
        char_idx = {c: i for i, c in enumerate(self._chars)}
        char_idx[" "] = len(self._chars)
        ks_i, ks_j, ks_l, fs = [], [], [], []
        for tg, freq in trigram_freqs.items():
            if len(tg) == 3 and all(c in charset for c in tg):
                ks_i.append(char_idx[tg[0]])
                ks_j.append(char_idx[tg[1]])
                ks_l.append(char_idx[tg[2]])
                fs.append(freq)
        self._i = np.array(ks_i, dtype=np.intp)
        self._j = np.array(ks_j, dtype=np.intp)
        self._l = np.array(ks_l, dtype=np.intp)
        self._f = np.array(fs, dtype=np.float64)
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
        if not len(self._f):
            return 0.0
        return float((self._f * self._T3[perm[self._i], perm[self._j], perm[self._l]]).sum())

    def fitness(self, layout: Layout) -> float:
        return self.fitness_of_permutation(self.permutation(layout))
