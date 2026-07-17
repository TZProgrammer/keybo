"""First-finger calibration: identified physics the free fit cannot attribute.

The measured effect (PINKY-GAP probe, 2026-07-10): same-hand, same-row,
adjacent-finger bigrams are slower when initiated by the OUTER finger (pinky-first
~+42ms, ring-first ~+21ms on the full data). The schema cannot represent it — the
origin key's finger is not encoded, and for same-row pairs every relational feature is
origin-symmetric, so outer-first and inner-first pairs into the same key are
byte-identical (the 184-collision family). Nor can learnable columns recover it:
within one layout, first-finger class is a deterministic function of bigram identity,
so the class effect is COLLINEAR with the per-ngram practice term — a free fit chooses
an arbitrary decomposition, and in practice hands the physics to the practice term and
the noise to the columns (sign inversion; PREREGISTRATIONS 485be17). Identification
requires a restriction: practice is a global smooth curve in log(sample count), and
matched cells (outer-first vs inner-first into the SAME key, same row, same wpm
bucket) difference it out.

That estimator lives here, IN the pipeline (PINKY-FIT, a735b70 — nothing hardcoded):

- :func:`fit_first_finger_deltas` fits the per-class deltas from whatever rows the
  trainer receives (so LOLO folds fit on their own training data — leakage-clean, and
  new data yields new deltas),
- training subtracts :func:`delta_log` from calibrated classes' targets (g fits the
  class's inner-first level), and records the fitted deltas in the model sidecar,
- serving adds the SIDECAR's deltas back per position pair
  (``TypingModel.predict_ms_at``, ``TableBigramScorer``) — position-aware paths only,
  because the colliding pairs are inseparable by feature vector. BIGRAMS ONLY:
  every trigram serving site uses the plain feature path and REJECTS a trigram
  model carrying deltas (``reject_calibrated_trigram_model``) — the single-pair
  delta API cannot express a trigram's increment/full-span pairs, and the trainer
  gates delta fitting on ``ngram == "bigram"`` so none exist to restore.

Scope: the two classes the estimator can match cleanly (pinky->ring, ring->middle;
same row, adjacent fingers). middle->index has no matched inner control (columns 1-2
are the same finger), so it is not calibrated. Cross-row pairs carry origin
information through dy/angle/roll features and need no repair.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from keybo.features import classify as C
from keybo.geometry import Geometry, Position

#: provenance tag recorded in model sidecars; bump when the ESTIMATOR changes
#: (the deltas themselves are data-fitted, not versioned constants).
CALIBRATION_VERSION = "2026-07-10.pinky2-fitted"

#: the class-typical time used to convert a ms delta into LOGRAT units: same-row
#: adjacent-finger bigrams run ~15% over the session mean keystroke (12000/wpm).
_CLASS_PACE_FACTOR = 1.15

#: matched-cell estimator knobs (probe-inherited; registered in a735b70).
_MIN_CELL_N = 50
_BUCKETS = [(lo, lo + 20) for lo in range(40, 140, 20)]

#: outer-first classes and their inner-first control column (same landing key).
#: landing |x|=4 (ring): outer 5 (pinky) vs inner 3 (middle);
#: landing |x|=3 (middle): outer 4 (ring) vs inner 2 (index).
_CLASS_SPEC = {
    "pinky_first": {"land": 4, "outer": 5, "inner": 3},
    "ring_first": {"land": 3, "outer": 4, "inner": 2},
}


def finger_class(geometry: Geometry, a: Position, b: Position) -> str | None:
    """The calibrated-class membership of a position pair, or None.

    Same hand, same row, adjacent (distinct) fingers, initiated by the outer finger —
    pinky(5)->ring(4) or ring(4)->middle(3).
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


def fit_first_finger_deltas(rows, geometry: Geometry) -> dict[str, float]:
    """Fit the per-class first-finger deltas (ms) from stroke rows.

    The probe's matched-cell design, generalized to every layout and landing key in
    ``rows``: for each (layout, landing position, wpm bucket), compare the IQR-mean
    time of the OUTER-first cell against the INNER-first control cell into the same
    key (identical feature vectors by construction), adjusting for practice via the
    per-(layout, bucket) regression of cell time on log10(cell sample count) over all
    same-row same-hand non-same-finger cells. Count-weighted pooling; classes with no
    qualifying matched cells are absent from the result (=> uncalibrated).
    """
    from keybo.data.strokes import iqr_average

    # per (layout, positions): the row (position pair <-> ngram is 1:1 within a layout)
    per_bucket: dict = defaultdict(dict)  # (layout, bucket_idx) -> {positions: (t, n, logn)}
    for r in rows:
        a, b = r.positions
        if a[1] != b[1]:
            continue
        if not C.same_hand(geometry, a, b) or C.same_finger(geometry, a, b):
            continue
        total = len(r.samples)
        by_bucket = defaultdict(list)
        for w, d, _p, _h in r.samples:
            for bi, (lo, hi) in enumerate(_BUCKETS):
                if lo <= w < hi:
                    by_bucket[bi].append(d)
                    break
        for bi, ds in by_bucket.items():
            if len(ds) < _MIN_CELL_N:
                continue
            per_bucket[(r.layout, bi)][(a, b)] = (
                iqr_average(ds),
                len(ds),
                float(np.log10(max(total, 1))),
            )

    num = defaultdict(float)
    den = defaultdict(float)
    for (_layout, _bi), cells in per_bucket.items():
        if len(cells) < 6:
            continue
        # practice slope for this (layout, bucket): time ~ 1 + log10(count)
        x = np.array([v[2] for v in cells.values()])
        y = np.array([v[0] for v in cells.values()])
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope = float(coef[1])
        for (a, b), (t_out, n_out, logn_out) in cells.items():
            cls = finger_class(geometry, a, b)
            if cls is None:
                continue
            spec = _CLASS_SPEC[cls]
            sign = 1 if a[0] > 0 else -1
            inner = (sign * spec["inner"], a[1])
            ctrl = cells.get((inner, b))
            if ctrl is None:
                continue
            t_in, n_in, logn_in = ctrl
            d_adj = (t_out - t_in) - slope * (logn_out - logn_in)
            w = float(min(n_out, n_in))
            num[cls] += w * d_adj
            den[cls] += w
    return {cls: num[cls] / den[cls] for cls in num if den[cls] > 0}


def delta_log(cls: str | None, wpm: float, deltas_ms: dict[str, float]) -> float:
    """The calibrated offset in LOGRAT units at this pace (0.0 when uncalibrated).

    Constructed so that serving ``exp(pred + delta_log) * 12000/wpm`` adds exactly
    ``deltas_ms[cls]`` at the class-typical time for that pace. ``deltas_ms`` comes
    from :func:`fit_first_finger_deltas` (training) or the model sidecar (serving).
    """
    if cls is None or cls not in deltas_ms:
        return 0.0
    t_typ = 12000.0 / max(wpm, 1.0) * _CLASS_PACE_FACTOR
    return float(np.log((t_typ + deltas_ms[cls]) / t_typ))
