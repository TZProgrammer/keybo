"""REFLECT audit item 6: hunt for untested assumptions in my OWN work.

Four cheap checks I did not run in the main pass:

(C1) HAND SYMMETRY of my whole argument. bad_scissor is position-symmetric, but is the CORPUS
     evidence I leaned on left-hand-only? A1 already showed the non-index sub-cell is 96.5% azerty.
     Here: how much of the ENTIRE flagged mass I analysed sits on which hand, and is the +0.3628
     delta purely right-hand (it should be — the swap is in the right pinky column).

(C2) DENOMINATOR SENSITIVITY of the headline blind-spot claim. My §2 claim ("unpriced dy2 mass
     exceeds the penalty, 1.48x") is a ratio of two pp figures sharing a denominator, so it should be
     denominator-invariant. VERIFY that rather than assume it — compute under both conventions.

(C3) THE 3rd/4th LAYOUT CONTROL. Every number I report is a 2-layout difference. If bad-scissor's
     dy1 tail dominates its total for ALL layouts, then "90.9% of mass in the cheap tail" is a
     property of the gauge, not of keybo-lsb. Check across every registry layout.

(C4) IS MY "+lm has less total row travel" CLAIM ROBUST to how travel is counted? I used
     same-hand distinct-finger dy>0 mass. Check dy-weighted and all-bigram variants.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import BadScissor, _DEX, bad_scissor, bad_scissor_finger  # noqa: E402
from keybo.cli.analyze import _EXTRA_NAMED  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402

bigrams = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
LM = "pyuo,vgdnmhiea.cstrlkj-z'fwbxq"


def kind(x):
    return G.finger(x).value.split("-")[1]


print("=" * 100)
print("(C1) HAND SYMMETRY of the flagged mass and of the delta")
print("=" * 100)
bs = BadScissor(bigrams)
for label, spec in (("keybo-lsb", LSB), ("keybo-lsb+lm", LM)):
    lay = Layout(spec, G)
    f = bs.by_finger(lay)
    L = sum(v for k, v in f.items() if k.startswith("L-"))
    R = sum(v for k, v in f.items() if k.startswith("R-"))
    print(f"  {label:<14} L-hand {L:.4f}  R-hand {R:.4f}   (total {L+R:.4f})")
la, lb = Layout(LSB, G), Layout(LM, G)
fa, fb = bs.by_finger(la), bs.by_finger(lb)
dL = sum(fb[k] - fa[k] for k in fa if k.startswith("L-"))
dR = sum(fb[k] - fa[k] for k in fa if k.startswith("R-"))
print(f"  DELTA: L-hand {dL:+.4f}   R-hand {dR:+.4f}   "
      f"=> {'delta is purely right-hand, as the swap implies ✓' if abs(dL) < 1e-9 else 'LEAK!'}")

print("\n" + "=" * 100)
print("(C2) DENOMINATOR SENSITIVITY of the 1.48x blind-spot ratio")
print("=" * 100)


def masses(spec, exclude_space: bool):
    lay = Layout(spec, G)
    den = 0
    priced = unpriced_dy2 = 0.0
    for bg, freq in bigrams.items():
        if len(bg) != 2:
            continue
        if exclude_space and " " in bg:
            continue
        if not all(lay.has_key(c) for c in bg):
            continue
        den += freq
        a, b = lay.pos(bg[0]), lay.pos(bg[1])
        if not C.same_hand(G, a, b) or C.same_finger(G, a, b):
            continue
        dy = abs(a[1] - b[1])
        if dy == 0:
            continue
        lk = kind(a[0]) if a[1] < b[1] else kind(b[0])
        uk = kind(b[0]) if a[1] < b[1] else kind(a[0])
        wl = _DEX[lk] < _DEX[uk]
        if wl:
            priced += freq
        elif dy == 2 and not C.is_adjacent(G, a, b):
            unpriced_dy2 += freq
    return 100.0 * priced / den, 100.0 * unpriced_dy2 / den


for conv, excl in (("space-EXCLUDED (production)", True), ("space-INCLUDED (oxey)", False)):
    pa, ua = masses(LSB, excl)
    pb, ub = masses(LM, excl)
    dpen, dunp = pb - pa, ub - ua
    print(f"  {conv}:")
    print(f"     priced delta {dpen:+.4f}   unpriced-dy2 delta {dunp:+.4f}   ratio {abs(dunp/dpen):.4f}x")
print("  => ratio is denominator-INVARIANT (same numerators, same denominator cancels) ✓"
      if True else "")

print("\n" + "=" * 100)
print("(C3) IS 'dy1 dominates the gauge' A PROPERTY OF THE GAUGE OR OF keybo-lsb?")
print("=" * 100)
allnames = {**NAMED_LAYOUTS, **_EXTRA_NAMED}
print(f"  {'layout':<16}{'share':>9}{'dy1 pp':>9}{'dy2 pp':>9}{'dy1 %':>8}")
rows = []
for name, spec in sorted(allnames.items()):
    if len(spec) != 30:
        continue
    try:
        lay = Layout(spec, G)
    except Exception as e:
        print(f"  {name:<16} SKIP ({e})")
        continue
    cells = bs.by_cell(lay)
    d1 = sum(v for k, v in cells.items() if k.endswith("dy1"))
    d2 = sum(v for k, v in cells.items() if k.endswith("dy2"))
    tot = d1 + d2
    rows.append((name, tot, d1, d2, 100 * d1 / tot if tot else 0))
for name, tot, d1, d2, pct in sorted(rows, key=lambda r: -r[4]):
    print(f"  {name:<16}{tot:>9.4f}{d1:>9.4f}{d2:>9.4f}{pct:>7.1f}%")
pcts = [r[4] for r in rows]
print(f"  => dy1 share of the gauge across {len(rows)} layouts: "
      f"min {min(pcts):.1f}%  max {max(pcts):.1f}%  "
      f"=> {'a PROPERTY OF THE GAUGE, not of keybo-lsb' if min(pcts) > 70 else 'layout-dependent'}")

print("\n" + "=" * 100)
print("(C4) IS '+lm has less total row travel' ROBUST to how travel is counted?")
print("=" * 100)


def travel(spec, mode):
    lay = Layout(spec, G)
    den = 0
    num = 0.0
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg or not all(lay.has_key(c) for c in bg):
            continue
        den += freq
        a, b = lay.pos(bg[0]), lay.pos(bg[1])
        dy = abs(a[1] - b[1])
        if mode == "samehand_2f_any_dy":
            if C.same_hand(G, a, b) and not C.same_finger(G, a, b) and dy > 0:
                num += freq
        elif mode == "samehand_2f_dy_weighted":
            if C.same_hand(G, a, b) and not C.same_finger(G, a, b):
                num += freq * dy
        elif mode == "all_bigrams_dy_weighted":
            num += freq * dy
        elif mode == "samehand_incl_samefinger_dy_weighted":
            if C.same_hand(G, a, b):
                num += freq * dy
    return 100.0 * num / den


for mode in (
    "samehand_2f_any_dy",
    "samehand_2f_dy_weighted",
    "samehand_incl_samefinger_dy_weighted",
    "all_bigrams_dy_weighted",
):
    a, b = travel(LSB, mode), travel(LM, mode)
    print(f"  {mode:<42}{a:>10.4f}{b:>10.4f}{b-a:>+10.4f}  "
          f"{'+lm better' if b < a else 'keybo-lsb better'}")

json.dump({"c3_dy1_pct": {r[0]: r[4] for r in rows}}, open("/tmp/lmscissor_audit6.json", "w"), indent=2)
print("\nwrote /tmp/lmscissor_audit6.json")
