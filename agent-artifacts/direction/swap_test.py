"""CHEAPEST DECISIVE TEST: which candidate origin/direction features are order-dependent?

Exhaustive over all ordered distinct position pairs on ROW_STAGGERED_30 (30*29 = 870) and
ROW_STAGGERED_31 (31*30 = 930).  Also reports the "900" frame (30*30 incl. a==b) that the
brief mentions, so the count convention is unambiguous.

For each candidate feature f, report n_pairs where f(a,b) != f(b,a) and the max |diff|.
A feature with 0 differing pairs is swap-invariant and therefore WORTHLESS as a direction
channel — that is the whole point of running this before any fit.
"""

from __future__ import annotations

import sys
from math import atan2, degrees

import numpy as np

sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")

from keybo.features import classify as C  # noqa: E402
from keybo.features.ngram import _placement_row_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31, Geometry, Position  # noqa: E402


# ---------------------------------------------------------------- candidates
def signed_dx(g: Geometry, a: Position, b: Position) -> float:
    """Stagger-adjusted horizontal displacement WITH SIGN (b relative to a)."""
    ax, ay = a
    bx, by = b
    return (bx + g.row_offsets.get(by, 0.0)) - (ax + g.row_offsets.get(ay, 0.0))


def signed_dx_handed(g: Geometry, a: Position, b: Position) -> float:
    """Signed dx expressed as inward(+)/outward(-) in the hand's own frame.

    Left hand's inward direction is +x; right hand's inward direction is -x.  For a
    cross-hand pair the notion is undefined -> 0.0.
    """
    if not C.same_hand(g, a, b):
        return 0.0
    hand = g.hand(a[0]) or 1
    return signed_dx(g, a, b) * hand * -1.0 if hand > 0 else signed_dx(g, a, b) * 1.0


def signed_dx_inward(g: Geometry, a: Position, b: Position) -> float:
    """Signed dx in |column| space: positive = moving toward the index finger (inward)."""
    if not C.same_hand(g, a, b):
        return 0.0
    return float(abs(a[0]) - abs(b[0]))


def signed_dy(g: Geometry, a: Position, b: Position) -> float:
    """Row displacement WITH SIGN: positive = moving up (toward the top row)."""
    return float(b[1] - a[1])


def signed_angle(g: Geometry, a: Position, b: Position) -> float:
    """rotation_angle, but measured from a->b instead of outer->inner (so it flips sign)."""
    if not C.same_hand(g, a, b) or C.same_finger(g, a, b):
        return 0.0
    if abs(a[0]) == abs(b[0]):
        return 0.0
    ax, ay = a
    bx, by = b
    off_a = g.row_offsets.get(ay, 0.0)
    off_b = g.row_offsets.get(by, 0.0)
    hand = g.hand(a[0]) or 1
    return round(degrees(atan2((by - ay), ((bx + off_b) - (ax + off_a)) * hand)), 2)


def origin_row_onehot(g: Geometry, a: Position, b: Position) -> dict[str, float]:
    return {
        "o_bottom": float(a[1] == 1),
        "o_home": float(a[1] == 2),
        "o_top": float(a[1] == 3),
    }


def origin_finger_onehot(g: Geometry, a: Position, b: Position) -> dict[str, float]:
    ax = abs(a[0])
    return {
        "o_pinky": float(ax in (5, 6)),
        "o_ring": float(ax == 4),
        "o_middle": float(ax == 3),
        "o_index": float(ax in (1, 2)),
        "o_lateral": float(C.is_lateral(a[0])),
    }


def true_inwards(g: Geometry, a: Position, b: Position) -> float:
    """TRUE inward roll: same hand, two fingers, second key nearer the index finger."""
    if not C.same_hand(g, a, b) or C.same_finger(g, a, b):
        return 0.0
    if abs(a[0]) == abs(b[0]):
        return 0.0
    return float(abs(b[0]) < abs(a[0]))


def true_outwards(g: Geometry, a: Position, b: Position) -> float:
    if not C.same_hand(g, a, b) or C.same_finger(g, a, b):
        return 0.0
    if abs(a[0]) == abs(b[0]):
        return 0.0
    return float(abs(b[0]) > abs(a[0]))


SCALARS = {
    "signed_dx (raw x, unsigned-hand)": signed_dx,
    "signed_dx_inward (|col| a - |col| b)": signed_dx_inward,
    "signed_dy (b.y - a.y, up positive)": signed_dy,
    "signed_angle (a->b atan2)": signed_angle,
    "true_inwards": true_inwards,
    "true_outwards": true_outwards,
}
DICTS = {
    "origin row one-hot": origin_row_onehot,
    "origin finger one-hot": origin_finger_onehot,
}


def report(g: Geometry, name: str, include_equal: bool) -> None:
    slots = list(g.slots)
    pairs = [
        (a, b) for a in slots for b in slots if (include_equal or a != b)
    ]
    print(f"\n=== {name}: {len(pairs)} ordered pairs "
          f"({'incl' if include_equal else 'excl'} a==b) ===")

    # --- positive control: reproduce the THEORY-1 result on the SHIPPED 20 features
    shipped_diff: dict[str, tuple[int, float]] = {}
    for a, b in pairs:
        r1 = _placement_row_from_positions(g, a, b)
        r2 = _placement_row_from_positions(g, b, a)
        for k in r1:
            d = abs(r1[k] - r2[k])
            n, m = shipped_diff.get(k, (0, 0.0))
            shipped_diff[k] = (n + (1 if d > 0 else 0), max(m, d))
    landing = {"bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral"}
    nonlanding_max = max(m for k, (n, m) in shipped_diff.items() if k not in landing)
    nonlanding_n = sum(n for k, (n, m) in shipped_diff.items() if k not in landing)
    print(f"  [positive control] SHIPPED non-landing features: "
          f"{nonlanding_n} differing pairs, max|diff| = {nonlanding_max:.3e}")
    print(f"  [positive control] SHIPPED landing one-hots: "
          + ", ".join(f"{k}={shipped_diff[k][0]}" for k in
                      ["bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral"]))

    print("  candidate                              n_differing / n_pairs   max|diff|   nonzero_rows")
    for fname, fn in SCALARS.items():
        n = 0
        mx = 0.0
        nz = 0
        for a, b in pairs:
            v1 = fn(g, a, b)
            v2 = fn(g, b, a)
            if v1 != 0.0:
                nz += 1
            d = abs(v1 - v2)
            if d > 0:
                n += 1
            mx = max(mx, d)
        flag = "ORDER-DEPENDENT" if n else "!! SWAP-INVARIANT (worthless)"
        print(f"  {fname:38s} {n:5d} / {len(pairs):5d}        {mx:9.3f}   {nz:5d}   {flag}")
    for gname, fn in DICTS.items():
        keys = list(fn(g, pairs[0][0], pairs[0][1]).keys())
        agg = {k: [0, 0.0] for k in keys}
        any_n = 0
        for a, b in pairs:
            d1, d2 = fn(g, a, b), fn(g, b, a)
            row_diff = False
            for k in keys:
                d = abs(d1[k] - d2[k])
                if d > 0:
                    agg[k][0] += 1
                    row_diff = True
                agg[k][1] = max(agg[k][1], d)
            if row_diff:
                any_n += 1
        print(f"  {gname:38s} {any_n:5d} / {len(pairs):5d}   (any column)")
        for k in keys:
            print(f"      {k:34s} {agg[k][0]:5d} / {len(pairs):5d}        {agg[k][1]:9.3f}")


if __name__ == "__main__":
    report(ROW_STAGGERED_30, "ROW_STAGGERED_30 distinct", include_equal=False)
    report(ROW_STAGGERED_30, "ROW_STAGGERED_30 with a==b", include_equal=True)
    report(ROW_STAGGERED_31, "ROW_STAGGERED_31 distinct", include_equal=False)
