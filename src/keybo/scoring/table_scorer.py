"""QAP-table scorer: the exact model objective, reduced to a position-pair table.

Features are pure geometry + wpm (frequency was removed from the schema per OQ-1), so at a
fixed scoring WPM a bigram's predicted time depends ONLY on its two key positions. There
are just 31 positions (30 slots + the fixed space key), so the whole objective collapses
to::

    fitness(layout) = sum_{c1,c2} F[c1, c2] * T[slot(c1), slot(c2)]

with ``F`` the corpus bigram-frequency matrix over the charset and ``T`` the model's
961-entry position-pair time table. That is a Quadratic Assignment Problem: evaluating a
layout is a fancy-indexed sum (microseconds) instead of a full model predict over the
corpus (~25 ms), which is what makes deep search (millions of evaluations) feasible.
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
        if len(self._chars) != len(geometry.slots):
            raise ValueError(
                f"charset has {len(self._chars)} characters but geometry has "
                f"{len(geometry.slots)} slots"
            )
        positions = [*geometry.slots, geometry.space_position]
        n_pos = len(positions)

        # T: predicted time for every ordered position pair, at the scoring WPM. The
        # reduction is valid by construction since the 2026-07-05 schema: features are pure
        # geometry + wpm (frequency was removed per OQ-1), so a bigram's prediction depends
        # only on its two positions. The table stores MILLISECONDS: a LOGRAT-space model's
        # raw output is log(ms*wpm/12000), and only ms entries sum to a corpus time.
        vectors = np.vstack(
            [
                bigram_features_from_positions(geometry, (a, b), wpm=target_wpm)
                for a in positions
                for b in positions
            ]
        )
        from keybo.scoring.model_scorer import predict_ms

        self._T = predict_ms(model, vectors).reshape(n_pos, n_pos)

        # First-finger calibration (PINKY-FIT): the calibrated classes' vectors collide
        # with their inner-first mirrors, so the offset must be applied by POSITION —
        # exactly what a table indexed by position pair can do and a feature path cannot.
        # Serve restores exactly what training subtracted (the model's own sidecar
        # deltas): a log-space offset, i.e. a multiplicative factor on the ms entry.
        training = (getattr(model, "metadata", None) and model.metadata.extra.get("training")) or {}
        cal = training.get("calibration")
        if cal and cal.get("deltas_ms"):
            from keybo.training.calibration import delta_log, finger_class

            for i, a in enumerate(positions):
                for j, b in enumerate(positions):
                    cls = finger_class(geometry, a, b)
                    if cls is not None:
                        self._T[i, j] *= float(np.exp(delta_log(cls, target_wpm, cal["deltas_ms"])))

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
