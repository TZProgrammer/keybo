"""First-finger calibration: measured physics the free fit cannot attribute.

The probe (PINKY-GAP, 2026-07-10): same-hand, same-row, adjacent-finger bigrams are
slower when initiated by the OUTER finger — pinky-first +42ms, ring-first +21ms,
practice-controlled on qwerty matched pairs (as/ds, po/io, we/re, oi/ui; 8/8 cells
positive). The schema cannot represent this (the origin key's finger is not encoded,
and for same-row pairs every relational feature is origin-symmetric — the 184-collision
class), and adding learnable columns FAILS: within one layout the identity effect is
collinear with the per-ngram practice term, and the backfit hands the physics to b,
leaving the columns to fit noise with an inverted sign (PREREGISTRATIONS 485be17).

So the effect enters as a FIXED, probe-calibrated offset:

- training subtracts ``delta_log`` from calibrated classes' targets (so g fits the
  inner-first level of the class),
- serving adds it back per position pair (``TypingModel.predict_ms_at`` and the table
  scorers, which know positions — the plain feature-vector path cannot, because the
  colliding pairs are byte-identical by construction).

Scope is exactly what the probe measured: pinky->ring and ring->middle, same row,
adjacent fingers. middle->index was NOT measured and is NOT extrapolated. Cross-row
pairs carry origin information through dy/angle/roll features, so the collision this
repairs does not arise there.
"""

from __future__ import annotations

import numpy as np

from keybo.features import classify as C
from keybo.geometry import Geometry, Position

#: provenance tag recorded in model sidecars; bump when the deltas are re-measured.
CALIBRATION_VERSION = "2026-07-10.pinky1"

#: measured deltas (ms), count-weighted over the probe's matched-pair cells
#: (runs/pinky_gap_probe.json via runs/pinky_cal.json).
DELTA_MS = {
    "pinky_first": 42.107603423799965,
    "ring_first": 21.28909329785162,
}

#: the class-typical time used to convert a ms delta into LOGRAT units: same-row
#: adjacent-finger bigrams run ~15% over the session mean keystroke (12000/wpm).
_CLASS_PACE_FACTOR = 1.15


def finger_class(geometry: Geometry, a: Position, b: Position) -> str | None:
    """The calibrated-class membership of a position pair, or None.

    Exactly the probe's measured configuration: same hand, same row, adjacent
    (distinct) fingers, initiated by the outer finger — pinky(5)->ring(4) or
    ring(4)->middle(3).
    """
    if a[1] != b[1]:
        return None
    if not C.same_hand(geometry, a, b) or C.same_finger(geometry, a, b):
        return None
    if not C.is_adjacent(geometry, a, b):
        return None
    fa, fb = abs(a[0]), abs(b[0])
    if fa == 5 and fb == 4:
        return "pinky_first"
    if fa == 4 and fb == 3:
        return "ring_first"
    return None


def delta_log(cls: str | None, wpm: float) -> float:
    """The calibrated offset in LOGRAT units at this pace (0.0 for uncalibrated).

    Constructed so that serving ``exp(pred + delta_log) * 12000/wpm`` adds exactly
    ``DELTA_MS[cls]`` at the class-typical time for that pace.
    """
    if cls is None:
        return 0.0
    t_typ = 12000.0 / max(wpm, 1.0) * _CLASS_PACE_FACTOR
    return float(np.log((t_typ + DELTA_MS[cls]) / t_typ))
