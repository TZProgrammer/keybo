"""Range objectives: aggregate the model's time surface over a BAND of typing paces.

The shipped speed objective is a single point — ``total_ms(layout; target_wpm)`` at one
``--target-wpm`` (default 90). This module aggregates that same objective over several paces,
so a layout can be optimized for a band of typists rather than one.

Two aggregations, which are DIFFERENT DECISIONS, not variants:

``mean``
    ``mean_w total_ms(L; w)`` — equal weight across the band.

``minimax``
    worst case over the band. This one has a trap, and it is the reason ``reference`` exists.

    A LOGRAT model's ms is ``exp(pred) * 12000 / wpm``, so the ``1 / wpm`` factor is baked
    into every prediction as pure arithmetic. Empirically (MULTIWPM-1, 62/62 layouts tested)
    ``total_ms(L; w)`` is therefore MONOTONE DECREASING in ``w``, so a max over the band
    always lands on the band's LOWEST pace. A raw ``max_w total_ms`` objective is not a
    worst-case-over-the-band at all: it is EXACTLY the single-point objective at ``min(band)``,
    wearing a wider objective's clothes.

    ``reference`` fixes that by dividing each pace's total by the SAME fixed board's total at
    that pace, which removes the per-pace scale and leaves only the layout's relative standing.
    ``max_w total_ms(L; w) / total_ms(ref; w)`` is a genuine minimax: it asks at which pace the
    layout is furthest behind the reference, and minimizes that. Since the divisor is constant
    within a pace, it cannot reorder layouts at a fixed ``w`` — it only reweights ACROSS paces.

Every per-pace evaluation is a :class:`~keybo.scoring.table_scorer.TableBigramScorer`, i.e. the
exact model objective the shipped search uses, just built once per band point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.layout import Layout
from keybo.scoring.base import IScorer
from keybo.scoring.table_scorer import TableBigramScorer

AGGREGATIONS = ("mean", "minimax", "endpoint")


class RangeBigramScorer(IScorer):
    """The bigram speed objective aggregated over a band of target WPMs.

    ``wpms`` is the sampled band (any length >= 1; a length-1 band reproduces the shipped
    single-point objective exactly, which is what makes the control arm run through this same
    code path). ``aggregation`` is ``"mean"``, ``"minimax"`` or ``"endpoint"``.

    ``reference`` is a 30-char layout string used as the per-pace divisor. It is REQUIRED for
    ``minimax`` and refused for the others: without it a minimax silently collapses to the
    band's lowest pace (see the module docstring), and with it a mean would no longer be in
    milliseconds, so the aggregation and the normalization are not independently free choices.
    """

    def __init__(
        self,
        model,
        bigram_freqs: Mapping[str, int],
        wpms: Sequence[float],
        aggregation: str = "mean",
        chars: str | None = None,
        reference: str | None = None,
        geometry: Geometry = ROW_STAGGERED_30,
    ) -> None:
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {aggregation!r} (known: {AGGREGATIONS})")
        if not len(wpms):
            raise ValueError("wpms must contain at least one pace")
        if any(w <= 0 for w in wpms):
            raise ValueError(f"every pace must be > 0 (LOGRAT->ms divides by wpm); got {list(wpms)}")
        if aggregation == "minimax" and reference is None:
            raise ValueError(
                "minimax requires reference=<30-char layout>: total_ms is monotone decreasing "
                "in wpm, so an un-normalized max over the band collapses to the band's lowest "
                "pace and silently reproduces the single-point objective there"
            )
        if aggregation != "minimax" and reference is not None:
            raise ValueError(
                f"reference is only meaningful for minimax; a {aggregation!r} of "
                "reference-normalized ratios is not a time in milliseconds"
            )
        if aggregation == "endpoint" and len(wpms) != 1:
            raise ValueError(f"endpoint takes exactly one pace, got {list(wpms)}")

        self.wpms = tuple(float(w) for w in wpms)
        self.aggregation = aggregation
        self.reference = reference
        self._scorers = [
            TableBigramScorer(model, bigram_freqs, target_wpm=w, chars=chars, geometry=geometry)
            for w in self.wpms
        ]
        # Per-pace divisors, computed ONCE: the reference board is fixed, so these are constants
        # of the objective, not of the layout being scored.
        if reference is not None:
            ref_layout = Layout(reference, geometry)
            self._divisors = np.array([s.fitness(ref_layout) for s in self._scorers])
            if np.any(self._divisors <= 0):
                raise ValueError(f"reference layout {reference!r} scored <= 0 at some pace")
        else:
            self._divisors = None

    def per_wpm(self, layout: Layout) -> np.ndarray:
        """The band's raw per-pace totals (ms) for ``layout`` — the curve behind the scalar."""
        return np.array([s.fitness(layout) for s in self._scorers])

    def fitness(self, layout: Layout) -> float:
        # The search-loop hot path: score through each pace's permutation fast path rather than
        # re-deriving the permutation per scorer (they share a charset, so it is the same vector).
        perm = self._scorers[0].permutation(layout)
        totals = np.array([s.fitness_of_permutation(perm) for s in self._scorers])
        if self._divisors is not None:
            return float(np.max(totals / self._divisors))
        if self.aggregation == "mean":
            return float(np.mean(totals))
        return float(totals[0])  # endpoint (single pace, validated in __init__)

    def describe(self) -> str:
        """One line naming the objective, for a result file that must be reproducible."""
        band = "/".join(f"{w:g}" for w in self.wpms)
        if self.aggregation == "endpoint":
            return f"single-point total_ms at wpm={band}"
        if self.aggregation == "mean":
            return f"mean of total_ms over wpm in {{{band}}}"
        return f"minimax of total_ms/total_ms(reference={self.reference!r}) over wpm in {{{band}}}"
