"""Comfort objective (OQ-4) — explicit, user-owned PREFERENCES, separate from speed.

The speed model is measured; comfort weights are opinions. This module keeps that line
bright: `ComfortBigramScorer` scores nothing but the documented penalty table below, and
`CompositeScorer` combines it with the speed scorer through one explicit knob
(``comfort_weight``, exposed as ``optimize --comfort-weight``; 0 = pure measured speed).

Measured license for the first default (OQ-14, `agent-artifacts/
OQ14-row-practice-identifiability.md`): top-vs-home is a SPEED NEAR-TIE (+1/+6/+3 ms
across skill bands), so a home-row preference costs ~nothing in predicted time — it is a
free tie-break, which is exactly what a comfort term should spend its budget on first.
The other defaults follow community consensus (SFB/scissor/LSB aversion beyond their time
cost; lag-2 finger reuse measured speed-neutral but disliked; utilization balance is
fatigue doctrine — see the lag-2 probe) — all overridable per run via a JSON file.

Units: penalties are expressed in MILLISECOND-EQUIVALENTS per corpus occurrence, so a
composite fitness stays interpretable ("comfort_weight=1.0 trades 1 ms of predicted time
for 1 ms-equivalent of discomfort").
"""

from __future__ import annotations

from collections.abc import Mapping

from keybo.features import classify as C
from keybo.layout import Layout
from keybo.scoring.base import IScorer

#: name -> (default weight in ms-equivalents per occurrence, why — an OPINION, documented)
DEFAULT_COMFORT: dict[str, tuple[float, str]] = {
    "off_home": (
        8.0,
        "keystrokes off the home row cost reach/return effort even where speed ties "
        "(OQ-14 measured the tie; this breaks it toward home)",
    ),
    "bottom_row": (
        10.0,
        "bottom-row keystrokes are additionally awkward beyond their (real, ~50ms) "
        "time penalty — curl + underreach",
    ),
    "sfb": (
        25.0,
        "same-finger bigrams feel bad beyond their time cost and correlate with errors "
        "(measured 1.22x error rate; below the objective bar but real)",
    ),
    "scissor": (15.0, "adjacent-finger two-row reaches strain the hand"),
    "lsb": (10.0, "lateral index stretches pull the hand off its anchor"),
    "lag2_reuse": (
        5.0,
        "same finger at distance 2 is speed-neutral (measured -13ms, lag-2 probe) but "
        "breaks flow; mild aversion",
    ),
}


class ComfortBigramScorer(IScorer):
    """Corpus-weighted comfort penalty, in ms-equivalents (lower = more comfortable)."""

    def __init__(
        self,
        bigram_freqs: Mapping[str, int],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self._freqs = dict(bigram_freqs)
        self._w = {name: w for name, (w, _why) in DEFAULT_COMFORT.items()}
        if weights:
            unknown = set(weights) - set(self._w)
            if unknown:
                raise ValueError(f"unknown comfort weight(s): {sorted(unknown)}")
            self._w.update(weights)

    def fitness(self, layout: Layout) -> float:
        g = layout.geometry
        total = 0.0
        for bg, freq in self._freqs.items():
            if len(bg) != 2 or not all(layout.has_key(c) for c in bg):
                continue
            a, b = layout.pos(bg[0]), layout.pos(bg[1])
            pen = 0.0
            for pos in (a, b):
                if pos[1] != 2 and pos[1] != 0:  # off home (space row exempt: thumb)
                    pen += self._w["off_home"] / 2
                if pos[1] == 1:
                    pen += self._w["bottom_row"] / 2
            cls = C.classify_positions(g, a, b)
            if cls is C.BigramClass.SAME_FINGER and a != b:
                pen += self._w["sfb"]
            if C.is_scissor(g, a, b):
                pen += self._w["scissor"]
            if C.is_lsb(g, a, b):
                pen += self._w["lsb"]
            total += freq * pen
        return total


class CompositeScorer(IScorer):
    """speed + comfort_weight * comfort — the two axes, one explicit knob."""

    def __init__(self, speed: IScorer, comfort: IScorer, comfort_weight: float = 0.0) -> None:
        self.speed = speed
        self.comfort = comfort
        self.comfort_weight = float(comfort_weight)

    def fitness(self, layout: Layout) -> float:
        value = self.speed.fitness(layout)
        if self.comfort_weight:
            value += self.comfort_weight * self.comfort.fitness(layout)
        return value
