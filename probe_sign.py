"""Verify the DIRECTION of bad_scissor's flag, and the same-row mechanism claim, directly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import bad_scissor, bad_scissor_cell  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402

LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
LSB_LM = "pyuo,vgdnmhiea.cstrlkj-z'fwbxq"

print("=== DIRECTION CHECK: pinky HOME + middle TOP  vs  pinky TOP + middle HOME ===")
pinky_home, middle_top = (5, 2), (3, 3)
pinky_top, middle_home = (5, 3), (3, 2)
print(f"  pinky@HOME(5,2) + middle@TOP(3,3)  -> flagged={bad_scissor(G, pinky_home, middle_top)}"
      f"  cell={bad_scissor_cell(G, pinky_home, middle_top)}")
print(f"  pinky@TOP(5,3)  + middle@HOME(3,2) -> flagged={bad_scissor(G, pinky_top, middle_home)}"
      f"  cell={bad_scissor_cell(G, pinky_top, middle_home)}")
print(f"  pinky@BOTTOM(5,1)+middle@HOME(3,2) -> flagged={bad_scissor(G, (5,1), (3,2))}"
      f"  cell={bad_scissor_cell(G, (5,1), (3,2))}")
print("  => flagged iff the WEAKER finger holds the LOWER key. Weak-on-TOP is EXCLUDED.")

print("\n=== the 108/72/36 census, re-derived from the geometry (docstring table) ===")
from keybo.features import classify as C  # noqa: E402

slots = G.slots
pairs = [(a, b) for a in slots for b in slots if a != b]
print(f"  ordered position pairs: {len(pairs) + len(slots)} incl. self-pairs; distinct a!=b: {len(pairs)}")
bad = [(a, b) for a, b in pairs if bad_scissor(G, a, b)]
narrow = [(a, b) for a, b in pairs if C.is_scissor(G, a, b)]
wide = [
    (a, b)
    for a, b in pairs
    if C.same_hand(G, a, b) and not C.same_finger(G, a, b) and abs(a[1] - b[1]) == 2
]
print(f"  bad-scissor: {len(bad)}  (dy1 {sum(1 for a,b in bad if abs(a[1]-b[1])==1)}, "
      f"dy2 {sum(1 for a,b in bad if abs(a[1]-b[1])==2)})")
print(f"  narrow is_scissor: {len(narrow)}   wide (same-hand,distinct-finger,dy2): {len(wide)}")
print(f"  narrow \\ bad: {len(set(narrow) - set(bad))}   wide \\ bad: {len(set(wide) - set(bad))}")
mp = [(a, b) for a, b in bad if {"middle", "pinky"} == {G.finger(a[0]).value.split('-')[1], G.finger(b[0]).value.split('-')[1]}]
print(f"  bad-scissor middle-pinky pairs: {len(mp)}")
mp_narrow = [(a, b) for a, b in narrow if {"middle", "pinky"} == {G.finger(a[0]).value.split('-')[1], G.finger(b[0]).value.split('-')[1]}]
print(f"  NARROW middle-pinky pairs: {len(mp_narrow)}  <-- BADSCISSOR-1's claim: 0 of 24")
mp_wide = [(a, b) for a, b in wide if {"middle", "pinky"} == {G.finger(a[0]).value.split('-')[1], G.finger(b[0]).value.split('-')[1]}]
print(f"  WIDE   middle-pinky pairs: {len(mp_wide)}")

print("\n=== the MECHANISM: where l's and m's right-hand partners sit, per layout ===")
for label, spec in (("keybo-lsb", LSB), ("keybo-lsb+lm", LSB_LM)):
    lay = Layout(spec, G)
    print(f"\n  {label}")
    for ch in "lm":
        p = lay.pos(ch)
        print(f"    {ch!r} at {p} ({G.finger(p[0]).value}, row {p[1]})")
    for partner in "dngv":
        pp = lay.pos(partner)
        for ch in "lm":
            p = lay.pos(ch)
            same_row = p[1] == pp[1]
            fl = bad_scissor(G, p, pp)
            print(
                f"    {ch!r}{partner!r}: {p} vs {pp}  same_row={same_row}  "
                f"dy={abs(p[1]-pp[1])}  flagged={fl}"
            )
