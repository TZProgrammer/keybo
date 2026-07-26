"""WHY is the origin ROW already determined by the shipped 20 features?

Hypothesis: `dx` is STAGGER-ADJUSTED — dx = |(ax+off[ay]) - (bx+off[by])| with distinct
per-row offsets {1:+0.5, 2:0.0, 3:-0.25}.  So dx leaks a's ROW.  Combined with the landing
row one-hot (which gives b's row) and `dy` (the |row span|), the origin row is recoverable.

Test: rebuild the 20-vector with an UNSTAGGERED dx and re-run the determination check.
If origin row becomes NON-determined, the stagger is the channel.  Also test dropping dy,
and report how many ORDERED pairs the whole vector identifies uniquely.
"""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")

from keybo.features import classify as C  # noqa: E402
from keybo.features.ngram import _placement_row_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

G = ROW_STAGGERED_30
SLOTS = list(G.slots)
PAIRS = [(a, b) for a in SLOTS for b in SLOTS if a != b]


def vec(a, b, unstagger=False, drop=()):
    r = dict(_placement_row_from_positions(G, a, b))
    if unstagger:
        r["dx"] = float(abs(a[0] - b[0]))
    for k in drop:
        r.pop(k)
    return tuple(round(v, 9) for v in r.values())


def determination(unstagger=False, drop=(), label=""):
    groups = defaultdict(list)
    for a, b in PAIRS:
        groups[vec(a, b, unstagger, drop)].append((a, b))
    orow_varies = sum(1 for v in groups.values() if len({a[1] for a, b in v}) > 1)
    orow_pairs = sum(len(v) for v in groups.values() if len({a[1] for a, b in v}) > 1)
    ofing_varies = sum(1 for v in groups.values() if len({G.finger(a[0]) for a, b in v}) > 1)
    okey_varies = sum(1 for v in groups.values() if len({a for a, b in v}) > 1)
    unique = sum(len(v) for v in groups.values() if len(v) == 1)
    print(f"  {label:44s} groups={len(groups):4d}  uniquely-identified pairs={unique:4d}"
          f"  origin-row-ambiguous groups={orow_varies:3d} ({orow_pairs:3d} pairs)"
          f"  origin-finger-amb={ofing_varies:3d}  origin-key-amb={okey_varies:3d}")


print(f"ROW_STAGGERED_30: {len(PAIRS)} ordered distinct pairs")
print(f"row_offsets = {G.row_offsets}")
print()
determination(label="SHIPPED 20 (stagger-adjusted dx)")
determination(unstagger=True, label="20 with UNSTAGGERED dx")
determination(drop=("dx",), label="20 minus dx entirely")
determination(unstagger=True, drop=("dy",), label="unstaggered dx, minus dy")
determination(drop=("angle",), label="SHIPPED minus angle")
determination(unstagger=True, drop=("angle",), label="unstaggered dx, minus angle")

# --- explicit demonstration on one concrete pair family ---------------------------------
print("\nCONCRETE DEMONSTRATION — same landing key, same |row span|, different origin row:")
b = (5, 2)          # right pinky, home row
for a in [(-5, 1), (-5, 3)]:
    r = _placement_row_from_positions(G, a, b)
    print(f"  a={a} b={b}:  dx(stagger-adj)={r['dx']:.4f}  dy={r['dy']:.1f} "
          f"distance={r['distance']:.4f}  angle={r['angle']:.2f}  "
          f"unstaggered|ax-bx|={abs(a[0]-b[0])}")
print("  => the two origin rows give DIFFERENT stagger-adjusted dx, so the row is readable.")

print("\nAnd the pairs whose reverse is featurewise IDENTICAL (structurally un-orderable):")
seen = set()
n = 0
for a, b in PAIRS:
    if vec(a, b) == vec(b, a):
        n += 1
        if (b, a) not in seen and len(seen) < 12:
            seen.add((a, b))
print(f"  {n} of {len(PAIRS)} ordered pairs; {n//2} unordered pairs. Examples:")
for a, b in sorted(seen)[:10]:
    print(f"    {a} <-> {b}   same_hand={C.same_hand(G,a,b)}  "
          f"landing sig a=({a[1]},{G.finger(a[0]).value}) b=({b[1]},{G.finger(b[0]).value})")
