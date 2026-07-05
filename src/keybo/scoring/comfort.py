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


#: Per-finger capacity multipliers for the load penalty — PREFERENCES informed by
#: community consensus (pinkies weakest, index strongest) and directionally consistent
#: with our (too-thin) per-finger service-rate probe. NOT calibrated measurements: the
#: lag-2 probe showed utilization has no speed mechanism to calibrate against, and
#: fatigue is unmeasurable in 15-minute sessions — which is exactly why this term lives
#: in the comfort axis with user-owned weights.
DEFAULT_FINGER_CAPACITY: dict[str, float] = {
    "L-pinky": 0.6,
    "L-ring": 0.85,
    "L-middle": 1.0,
    "L-index": 1.0,
    "R-index": 1.0,
    "R-middle": 1.0,
    "R-ring": 0.85,
    "R-pinky": 0.6,
    "thumb": 1.5,  # space; effectively unconstrained
}


class FingerLoadScorer(IScorer):
    """Utilization-balancing penalty: Σ_f load_f² / capacity_f (ms-equivalents scale).

    The semimak principle ("use fingers proportionally to their strength") as an explicit
    convex objective term. h(load) = load² makes concentration on one finger cost more
    than the same weight spread out (Jensen), and dividing by capacity makes pinky
    concentration cost more than index concentration. Linear-in-assignment structure:
    load_f is a sum over keys of per-key corpus weight, so swap deltas are O(1).

    Measured context (lag-2 probe, PREREGISTRATIONS 2026-07-05): finger reuse has NO
    speed penalty beyond lag 1, so this term is fatigue/comfort doctrine, deliberately
    NOT part of the measured speed model. Scale: penalty is normalized so a perfectly
    qwerty-like imbalance contributes O(10 ms-equivalents) per keystroke at weight 1.
    """

    _SCALE = 1000.0  # maps squared load shares into the ms-equivalent regime

    def __init__(
        self,
        bigram_freqs: Mapping[str, int] | None = None,
        multipliers: Mapping[str, float] | None = None,
    ) -> None:
        self._freqs = dict(bigram_freqs) if bigram_freqs else None
        self._capacity = dict(DEFAULT_FINGER_CAPACITY)
        if multipliers:
            unknown = set(multipliers) - set(self._capacity)
            if unknown:
                raise ValueError(f"unknown finger name(s): {sorted(unknown)}")
            self._capacity.update(multipliers)

    def penalty(self, layout: Layout, bigram_freqs: Mapping[str, int]) -> float:
        from keybo.scoring.inspect import layout_diagnostics

        loads = layout_diagnostics(layout, bigram_freqs)["finger_load"]
        return self._SCALE * sum(
            (share * share) / self._capacity[finger] for finger, share in loads.items()
        )

    def fitness(self, layout: Layout) -> float:
        if self._freqs is None:
            raise ValueError("FingerLoadScorer needs bigram_freqs (at init or via penalty())")
        return self.penalty(layout, self._freqs)
