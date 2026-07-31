"""KITCHEN-SINK candidate audit — what is GENUINELY missing, measured before implementing.

The rule this enforces (rule 7b, code-citation before expensive work): a candidate feature is
only worth a training round if it is (a) not already computed by the served frame and (b) not
LINEARLY RECOVERABLE from the columns we already serve. KEYCRAFT-1 killed `2RL-IN + 2RL-OUT`
this way (R2 = 1.0000 against our roll), and that refutation cost one regression, not a
12-cell LOLO run.

So for every candidate this reports, over the FULL position-pair / triple enumeration of
``ROW_STAGGERED_30`` (never a corpus sample, which would confound coverage with definition):

* ``fires``        — how many pairs/triples the predicate fires on (0 = dead, n = invariant)
* ``swap_asym``    — pairs whose value CHANGES under reversal. 0 means the column cannot see
                     stroke order, which is the exact defect CYANO-1/KEYCRAFT-1 found in the
                     shipped ``inwards``/``outwards``.
* ``r2_vs_served`` — OLS R2 regressing the candidate on the served frame. >= 0.99 means the
                     model can already synthesize it and a new column buys nothing.
* ``r2_vs_wide``   — same, against the widened (direction) frame we build on.

Run:
    PYTHONPATH=src python agent-artifacts/kitchensink_audit.py --out agent-artifacts/kitchensink_audit.json
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations

import numpy as np

from keybo.features import classify as C
from keybo.features.ngram import (
    _placement_row_from_positions,
    _trigram_row_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30, Geometry, Position

G = ROW_STAGGERED_30


# --- candidate predicates -----------------------------------------------------------------
#
# Reimplemented from the DEFINITIONS in keycraft (BSD-3, github.com/rbscholtus/keycraft) and
# cyanophage's keyboard_svg.js, read via the KEYCRAFT-1 / CYANO-1 audits. No code vendored;
# each is expressed in OUR geometry vocabulary (signed columns, Geometry.finger).


def _fkind(g: Geometry, x: int) -> int:
    """Finger dexterity rank: pinky 0, ring 1, middle 2, index 3 (keycraft's FingerKind)."""
    return {5: 0, 6: 0, 4: 1, 3: 2, 2: 3, 1: 3}[abs(x)]


def half_scissor(g: Geometry, a: Position, b: Position) -> bool:
    """HSB — a ONE-row adjacent-finger reach. Our ``is_scissor`` gates on ``dy == 2`` only, so
    every one-row scissor is invisible to the served frame. keycraft splits FSB (2 rows) from
    HSB (1 row) and prices them separately."""
    if not C.is_adjacent(g, a, b):
        return False
    return abs(a[1] - b[1]) == 1


def sf_skip_eligible(g: Geometry, a: Position, c: Position) -> bool:
    """SFS at the skipgram level: first and third key on the same finger, different key.

    The trigram frame HAS ``sg_same_finger``, so this is the control that should come back
    fully recoverable — it is in the audit precisely to prove the instrument can detect
    redundancy, not only novelty.
    """
    return C.same_finger(g, a, c) and a != c


def pinky_off_home(g: Geometry, a: Position, b: Position) -> float:
    """POH — keycraft's weighted pinky-off-home penalty, as a per-bigram landing indicator.
    Ours has a ``pinky`` one-hot and a ``home`` one-hot but no INTERACTION term, and a tree
    can only build one by spending depth."""
    return float(_fkind(g, b[0]) == 0 and b[1] != 2)


def lateral_stretch_magnitude(g: Geometry, a: Position, b: Position) -> float:
    """LSB-dist — keycraft prices the lateral stretch by its horizontal MAGNITUDE, we ship a
    binary ``lsb``. The magnitude is the graded version of a flag we already have."""
    if not C.is_lsb(g, a, b):
        return 0.0
    return float(g.stagger_adjusted_dx(a, b))


def row_skip(g: Geometry, a: Position, b: Position) -> float:
    """A two-row jump on the same hand REGARDLESS of finger adjacency. ``scissor`` requires
    adjacent fingers, so a pinky->index two-row jump is unflagged today."""
    return float(C.same_hand(g, a, b) and abs(a[1] - b[1]) == 2)


def weak_finger_pair(g: Geometry, a: Position, b: Position) -> float:
    """Both keys on the two least-dextrous fingers (pinky/ring) — keycraft's RED-WEAK gate
    applied at bigram level. Our finger one-hot describes only the LANDING key, so a
    pinky->ring bigram and an index->ring bigram are featurewise identical in the finger
    block."""
    return float(C.same_hand(g, a, b) and _fkind(g, a[0]) <= 1 and _fkind(g, b[0]) <= 1)


def finger_dist_ordered(g: Geometry, a: Position, b: Position) -> float:
    """SIGNED finger-rank step: +ve toward index, -ve toward pinky. The magnitude version of
    ``inwards_ordered``, which is binary. keycraft's IN/OUT is also binary, so this is ours."""
    if not C.same_hand(g, a, b) or C.same_finger(g, a, b):
        return 0.0
    return float(_fkind(g, b[0]) - _fkind(g, a[0]))


BIGRAM_CANDIDATES = {
    "half_scissor": lambda g, a, b: float(half_scissor(g, a, b)),
    "row_skip_anyfinger": row_skip,
    "pinky_off_home": pinky_off_home,
    "lsb_magnitude": lateral_stretch_magnitude,
    "weak_finger_pair": weak_finger_pair,
    "finger_dist_ordered": finger_dist_ordered,
}


# --- trigram-level candidates -------------------------------------------------------------


def _tri_hands(g: Geometry, a, b, c):
    return g.hand(a[0]), g.hand(b[0]), g.hand(c[0])


def onehand_monotonic(g: Geometry, a, b, c) -> float:
    """3RL — keycraft's three-key one-hand MONOTONIC roll (an "onehand"). Our trigram frame has
    ``redirect`` (the non-monotonic case) but names no column for the monotonic one, so the
    smoothest trigram class is only representable as NOT-redirect AND same_hand."""
    ha, hb, hc = _tri_hands(g, a, b, c)
    if not (ha != 0 and ha == hb == hc):
        return 0.0
    fa, fb, fc = _fkind(g, a[0]), _fkind(g, b[0]), _fkind(g, c[0])
    if fa == fb or fb == fc:
        return 0.0
    return float((fa < fb) == (fb < fc))


def onehand_in(g: Geometry, a, b, c) -> float:
    """3RL-IN — a monotonic one-hand roll travelling toward the index. DIRECTIONAL."""
    if not onehand_monotonic(g, a, b, c):
        return 0.0
    return float(_fkind(g, c[0]) > _fkind(g, a[0]))


def red_weak(g: Geometry, a, b, c) -> float:
    """RED-WEAK — a redirect with NO index finger involved (keycraft's own separate weight
    class). We ship ``bad_redirect``, which is the same idea; this is the control for it."""
    ha, hb, hc = _tri_hands(g, a, b, c)
    if not (ha != 0 and ha == hb == hc):
        return 0.0
    fa, fb, fc = _fkind(g, a[0]), _fkind(g, b[0]), _fkind(g, c[0])
    if fa == fb or fb == fc:
        return 0.0
    monotonic = (fa < fb) == (fb < fc)
    if monotonic:
        return 0.0
    return float(fa != 3 and fb != 3 and fc != 3)


def red_sfs(g: Geometry, a, b, c) -> float:
    """RED-SFS — a redirect whose first and third key are the SAME FINGER but different keys.
    keycraft prices this apart from a plain redirect; we have neither the split nor a
    same-finger-gated redirect until REDIRGATE-1's pair (which gates the OTHER way)."""
    ha, hb, hc = _tri_hands(g, a, b, c)
    if not (ha != 0 and ha == hb == hc):
        return 0.0
    fa, fb, fc = _fkind(g, a[0]), _fkind(g, b[0]), _fkind(g, c[0])
    if fa == fb or fb == fc:
        return 0.0
    if (fa < fb) == (fb < fc):
        return 0.0
    return float(C.same_finger(g, a, c) and a != c)


def alt_sfs(g: Geometry, a, b, c) -> float:
    """ALT-SFS — a hand ALTERNATION whose outer two keys land on the same finger (the hidden
    same-finger skip inside an alternation). keycraft splits ALT into ALT-NML and ALT-SFS."""
    ha, hb, hc = _tri_hands(g, a, b, c)
    if not (ha != 0 and hc != 0 and ha == hc and ha != hb):
        return 0.0
    return float(C.same_finger(g, a, c) and a != c)


def full_scissor_skip(g: Geometry, a, b, c) -> float:
    """FSS — a full (two-row, adjacent-finger) scissor across the SKIPGRAM (keys 1 and 3).
    Our trigram frame carries sg_dx/sg_dy/sg_distance/sg_same_finger but no sg_scissor."""
    return float(C.is_adjacent(g, a, c) and abs(a[1] - c[1]) == 2)


def half_scissor_skip(g: Geometry, a, b, c) -> float:
    """HSS — the one-row scissor across the skipgram."""
    return float(C.is_adjacent(g, a, c) and abs(a[1] - c[1]) == 1)


def lsb_skip(g: Geometry, a, b, c) -> float:
    """LSS — lateral stretch across the skipgram."""
    return float(C.is_lsb(g, a, c))


TRIGRAM_CANDIDATES = {
    "onehand": onehand_monotonic,
    "onehand_in": onehand_in,
    "red_weak": red_weak,
    "red_sfs": red_sfs,
    "alt_sfs": alt_sfs,
    "sg_full_scissor": full_scissor_skip,
    "sg_half_scissor": half_scissor_skip,
    "sg_lsb": lsb_skip,
}


# --- audit machinery ----------------------------------------------------------------------


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R2 of y on [1, X]. 1.0 = the frame already determines the candidate exactly."""
    var = float(((y - y.mean()) ** 2).sum())
    if var <= 1e-12:
        return float("nan")  # constant candidate: R2 undefined, `fires` tells the story
    A = np.hstack([np.ones((len(y), 1)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(1.0 - (resid**2).sum() / var)


def audit_bigrams() -> list[dict]:
    pairs = [(a, b) for a, b in permutations(G.slots, 2)]
    narrow = np.array(
        [list(_placement_row_from_positions(G, a, b, direction=False).values()) for a, b in pairs]
    )
    wide = np.array(
        [list(_placement_row_from_positions(G, a, b, direction=True).values()) for a, b in pairs]
    )
    out = []
    for name, fn in BIGRAM_CANDIDATES.items():
        y = np.array([float(fn(G, a, b)) for a, b in pairs])
        y_rev = np.array([float(fn(G, b, a)) for a, b in pairs])
        out.append(
            {
                "feature": name,
                "level": "bigram",
                "n_pairs": len(pairs),
                "fires": int((y != 0).sum()),
                "n_distinct_values": int(len(np.unique(y))),
                "swap_asym": int((y != y_rev).sum()),
                "r2_vs_served": _r2(y, narrow),
                "r2_vs_wide": _r2(y, wide),
            }
        )
    return out


def audit_trigrams(max_triples: int | None = None) -> list[dict]:
    triples = [t for t in permutations(G.slots, 3)]
    if max_triples:
        triples = triples[:max_triples]
    narrow = np.array(
        [
            list(_trigram_row_from_positions(G, a, b, c, 90.0, direction=False).values())
            for a, b, c in triples
        ]
    )
    wide = np.array(
        [
            list(_trigram_row_from_positions(G, a, b, c, 90.0, direction=True).values())
            for a, b, c in triples
        ]
    )
    out = []
    for name, fn in TRIGRAM_CANDIDATES.items():
        y = np.array([float(fn(G, a, b, c)) for a, b, c in triples])
        y_rev = np.array([float(fn(G, c, b, a)) for a, b, c in triples])
        out.append(
            {
                "feature": name,
                "level": "trigram",
                "n_triples": len(triples),
                "fires": int((y != 0).sum()),
                "n_distinct_values": int(len(np.unique(y))),
                "swap_asym": int((y != y_rev).sum()),
                "r2_vs_served": _r2(y, narrow),
                "r2_vs_wide": _r2(y, wide),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-triples", type=int, default=None)
    args = ap.parse_args()

    rows = audit_bigrams() + audit_trigrams(args.max_triples)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, default=float)

    hdr = f"{'feature':22} {'lvl':8} {'fires':>7} {'nval':>5} {'swapasym':>9} {'R2narrow':>9} {'R2wide':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        n = r.get("n_pairs") or r.get("n_triples")
        print(
            f"{r['feature']:22} {r['level']:8} {r['fires']:>7} {r['n_distinct_values']:>5} "
            f"{r['swap_asym']:>9} {r['r2_vs_served']:>9.4f} {r['r2_vs_wide']:>9.4f}   (of {n})"
        )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
