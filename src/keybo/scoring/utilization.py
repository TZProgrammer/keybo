"""Finger-utilization preference scorers (FU round, 2026-07-11).

STATUS: DOCUMENTED PREFERENCES, not measured physics. The lag-3 probe
(runs/lag3_probe.json: 2.48M 4-gram windows, matched controls) found NO time cost for
displaced-finger reuse at lag 3 once lag-1/lag-2 collisions are excluded (-0.06ms,
52% positive — a null). The speed objective already prices everything the timing data
can see; these scorers encode the COMFORT/BALANCE intuition that concentrated load on
weak fingers is undesirable, for users who want it — exactly like the oxey term, via
CompositeScorer weights.

Two forms:

- :class:`DislocationScorer` — the owner's formulation: what matters is not how OFTEN
  a finger is used but how much it must TRAVEL, weighted by how slow the finger is.
  ``cost(position) = distance(position, finger_home) * slowness(finger)``; the layout
  penalty is the corpus-frequency-weighted sum over letters. A pinky resting on its
  home key costs nothing however often it types. LINEAR in the assignment, so it
  composes into the QAP objective exactly (per-position cost table).
- :class:`FingerSpeedScorer` — a genkey-style "fingerspeed" approximation: per-finger
  (usage x mean-travel) / strength, summed. Approximated from genkey's documented
  behavior, like OxeyStyleScorer approximates oxeylyzer; exact-tool parity is a
  registered follow-up.

Slowness weights are OUR measurements where we have them: the fitted first-finger
calibration measured pinky-initiated rolls +43ms and ring +21ms over the index/middle
reference (runs/pinky_cal.json / pinkyfit_gates); index and middle showed no
separable penalty. Normalized to index=1.0: pinky 1.43, ring 1.21, middle 1.0,
index 1.0. (Provenance: measured on same-row adjacent rolls; extrapolating the
ORDERING to general travel is the preference part.)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from keybo.geometry import ROW_STAGGERED_30, Geometry, Position
from keybo.layout import Layout
from keybo.scoring.base import IScorer

#: slowness multiplier per finger class (index=1.0 reference). Pinky/ring from the
#: fitted first-finger calibration deltas (+43/+21ms on ~150ms rolls); middle/index
#: unseparated by the data => 1.0.
DEFAULT_SLOWNESS = {"pinky": 1.43, "ring": 1.21, "middle": 1.0, "index": 1.0}

#: home position per finger (column, home row).
_FINGER_HOME: dict[tuple[int, str], Position] = {
    (-1, "pinky"): (-5, 2),
    (-1, "ring"): (-4, 2),
    (-1, "middle"): (-3, 2),
    (-1, "index"): (-2, 2),
    (1, "index"): (2, 2),
    (1, "middle"): (3, 2),
    (1, "ring"): (4, 2),
    (1, "pinky"): (5, 2),
}


def finger_name(p: Position) -> str:
    return {5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}[abs(p[0])]


def finger_key(p: Position) -> tuple[int, str]:
    return (1 if p[0] > 0 else -1, finger_name(p))


class DislocationScorer(IScorer):
    """Corpus-weighted finger-travel penalty, slowness-weighted (owner's form).

    ``fitness = sum over letters freq(l) * distance(pos_l, home(finger)) * slowness``.
    Lower is better. Space (thumb) contributes nothing.
    """

    def __init__(
        self,
        bigram_freqs: Mapping[str, int],
        slowness: Mapping[str, float] | None = None,
        geometry: Geometry = ROW_STAGGERED_30,
    ) -> None:
        self._geometry = geometry
        self._slowness = dict(slowness or DEFAULT_SLOWNESS)
        # letter frequencies from the bigram corpus (both chars)
        lf: dict[str, float] = {}
        for bg, f in bigram_freqs.items():
            for c in bg:
                if c != " ":
                    lf[c] = lf.get(c, 0.0) + float(f)
        self._letter_freqs = lf

    def position_cost(self, p: Position) -> float:
        """distance-from-finger-home x finger slowness for one position."""
        if p[0] == 0:
            return 0.0  # thumb/space
        home = _FINGER_HOME[finger_key(p)]
        d = float(np.hypot(p[0] - home[0], p[1] - home[1]))
        return d * self._slowness[finger_name(p)]

    def per_finger_dislocation(self, layout: Layout) -> dict[str, float]:
        """Diagnostic: corpus-weighted dislocation per finger (unnormalized)."""
        out: dict[str, float] = {}
        for c, f in self._letter_freqs.items():
            if not layout.has_key(c):
                continue
            p = layout.pos(c)
            if p[0] == 0:
                continue
            k = f"{'R' if p[0] > 0 else 'L'}-{finger_name(p)}"
            out[k] = out.get(k, 0.0) + f * self.position_cost(p)
        return out

    def fitness(self, layout: Layout) -> float:
        total = 0.0
        for c, f in self._letter_freqs.items():
            if layout.has_key(c):
                total += f * self.position_cost(layout.pos(c))
        return total


class FingerSpeedScorer(IScorer):
    """genkey-style fingerspeed approximation: per-finger (usage x travel)/strength.

    For each finger: usage = corpus letter mass assigned to it; mean_travel = its
    usage-weighted mean distance-from-home; penalty_f = usage * (1 + mean_travel) /
    strength, where strength = 1/slowness. Sum over fingers, superlinear in
    concentration via the usage x travel product. Lower is better.
    """

    def __init__(
        self,
        bigram_freqs: Mapping[str, int],
        slowness: Mapping[str, float] | None = None,
        geometry: Geometry = ROW_STAGGERED_30,
    ) -> None:
        self._d = DislocationScorer(bigram_freqs, slowness, geometry)
        self._slowness = self._d._slowness

    def fitness(self, layout: Layout) -> float:
        use: dict[str, float] = {}
        trav: dict[str, float] = {}
        for c, f in self._d._letter_freqs.items():
            if not layout.has_key(c):
                continue
            p = layout.pos(c)
            if p[0] == 0:
                continue
            k = f"{'R' if p[0] > 0 else 'L'}-{finger_name(p)}"
            use[k] = use.get(k, 0.0) + f
            home = _FINGER_HOME[finger_key(p)]
            trav[k] = trav.get(k, 0.0) + f * float(np.hypot(p[0] - home[0], p[1] - home[1]))
        total = 0.0
        for k, u in use.items():
            mean_travel = trav.get(k, 0.0) / u if u else 0.0
            name = k.split("-")[1]
            total += u * (1.0 + mean_travel) * self._slowness[name]
        return total
