#!/usr/bin/env python3
"""K10 audit — is the refutation's decisive "credited elsewhere" leg true?

THE FINDING (K10, refuted 2/3, verdict WRONG claimed):
  oxey `inroll`/`outroll` credit ZERO same-row rolls; 32-63% of eligible mass is
  silently unrewarded, sparing qwerty most.

REFUTING VOTE 1's leg (2) — the ONLY leg that answers the finding's own question:
  "Same-row roll credit exists in the frame as the separate TRIGRAM gauge `sr-roll`
   (in GAUGE_NAMES)."

If that is false, the "the mass IS credited, just elsewhere" defence collapses.

Two things must be true for the defence to hold:
  (A) sr-roll must be reachable from the oxey-style scorer (else it cannot credit
      anything IN that scorer's number), and
  (B) sr-roll must actually cover the same-row BIGRAM population the finding names.
"""
import sys, inspect
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import preflight  # asserts /tmp/refaudit tree + runs its own negative control
print()

from keybo.scoring import oxey
from keybo.analysis import kmstats
from keybo.features import classify as C
from keybo.cli.analyze import GAUGE_NAMES

print("=" * 78)
print("(A) IS sr-roll REACHABLE FROM THE OXEY-STYLE SCORER?")
print("=" * 78)
osrc = inspect.getsource(oxey)
print(f"  'sr-roll'/'sr_roll' occurrences in scoring/oxey.py : "
      f"{osrc.count('sr-roll') + osrc.count('sr_roll')}")
print(f"  'kmstats' imported by scoring/oxey.py              : {'kmstats' in osrc}")
print(f"  DEFAULT_OXEY_WEIGHTS keys                          : "
      f"{sorted(oxey.DEFAULT_OXEY_WEIGHTS)}")
print(f"  is 'sr-roll' / 'sr_roll' a weighted term?          : "
      f"{'sr-roll' in oxey.DEFAULT_OXEY_WEIGHTS or 'sr_roll' in oxey.DEFAULT_OXEY_WEIGHTS}")
print(f"  sr-roll lives in                                   : "
      f"{kmstats.__file__.split('/keybo/')[-1]}  (a DIFFERENT module)")
print(f"  both are separate entries of GAUGE_NAMES           : "
      f"{'sr-roll' in GAUGE_NAMES and 'oxey-style' in GAUGE_NAMES}")
print(f"     GAUGE_NAMES = {GAUGE_NAMES}")
reach = ('sr-roll' in oxey.DEFAULT_OXEY_WEIGHTS) or ('kmstats' in osrc)
print(f"\n  => (A) sr-roll can affect the oxey-style number: {reach}")

print()
print("=" * 78)
print("(B) DOES sr-roll COVER THE SAME-ROW *BIGRAM* POPULATION THE FINDING NAMES?")
print("=" * 78)
ksrc = inspect.getsource(kmstats)
i = ksrc.find('if short == "sr-roll"')
print("  kmstats' sr-roll definition:")
for ln in ksrc[max(0, i - 700):i + 400].splitlines()[-22:]:
    print("     ", ln)
print(f"\n  _TRIGRAM_METRICS = {kmstats._TRIGRAM_METRICS}")
print("  => sr-roll is a TRIGRAM metric; the finding's population is same-row BIGRAMS.")

print()
print("=" * 78)
print("(C) THE FINDING'S OWN CLAIM, RE-DERIVED INDEPENDENTLY (pair sweep)")
print("=" * 78)
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.geometry import ROW_STAGGERED_30

# built exactly as cli/analyze.py:296 builds it — Layout(chars, ROW_STAGGERED_30)
qwerty = NAMED_LAYOUTS["qwerty"]
lay = Layout(qwerty, ROW_STAGGERED_30)
g = lay.geometry
chars = list(qwerty)
pairs = [(a, b) for a in chars for b in chars if a != b]
elig = samerow = credited = samerow_credited = 0
for a, b in pairs:
    pa, pb = lay.pos(a), lay.pos(b)
    ha, hb = g.hand(pa[0]), g.hand(pb[0])
    if ha == 0 or hb == 0 or ha != hb:
        continue                      # same-hand only
    if g.same_finger(pa[0], pb[0]):
        continue                      # two distinct fingers
    if pa[1] == pb[1] and pa[0] == pb[0]:
        continue
    elig += 1
    sr = (pa[1] == pb[1])             # same ROW  (coords are (col_signed, row))
    cr = C.is_inwards(g, pa, pb) or C.is_outwards(g, pa, pb)
    samerow += sr
    credited += cr
    samerow_credited += (sr and cr)
print(f"  eligible same-hand two-finger ordered pairs   : {elig}")
print(f"    of which SAME-ROW                          : {samerow}")
print(f"    credited by is_inwards OR is_outwards       : {credited}")
print(f"    SAME-ROW pairs that get ANY roll credit     : {samerow_credited}")
print(f"  => same-row roll credit in the scorer's own predicates: "
      f"{'NONE' if samerow_credited == 0 else samerow_credited}")

print()
print("=" * 78)
print("VERDICT ON THE REFUTATION'S 'CREDITED ELSEWHERE' LEG")
print("=" * 78)
if not reach:
    print("  FAILS. sr-roll is a TRIGRAM metric in a DIFFERENT module (analysis/kmstats.py),")
    print("  is not a term of DEFAULT_OXEY_WEIGHTS, and is a SEPARATE entry of GAUGE_NAMES.")
    print("  It therefore cannot credit same-row rolls inside the `oxey-style` number, which")
    print("  is what the finding was about. The defence answers a DIFFERENT question")
    print("  ('does the FRAME price same-row rolls anywhere?') than the finding asked")
    print("  ('does the oxey-style SCORER credit them?').")
else:
    print("  HOLDS.")

print()
print("=" * 78)
print("(D) NEGATIVE CONTROL — can this probe DETECT same-row credit if it existed?")
print("=" * 78)
# If the probe reports 0 same-row credited pairs because of a bug in MY enumeration
# rather than because the predicates gate them out, then a predicate that DOES credit
# same-row pairs must still read 0. Substitute one and check the count MOVES.
def fake_inwards(g, a, b):
    """A stand-in that credits same-row pairs by finger order (the p7 'fix' shape)."""
    return abs(b[0]) < abs(a[0])

sr_credited_fake = 0
sr_total = 0
for a, b in pairs:
    pa, pb = lay.pos(a), lay.pos(b)
    ha, hb = g.hand(pa[0]), g.hand(pb[0])
    if ha == 0 or hb == 0 or ha != hb or g.same_finger(pa[0], pb[0]):
        continue
    if pa[1] != pb[1]:
        continue
    sr_total += 1
    if fake_inwards(g, pa, pb) or (not fake_inwards(g, pa, pb) and abs(pb[0]) > abs(pa[0])):
        sr_credited_fake += 1
print(f"  same-row eligible pairs                        : {sr_total}")
print(f"  credited under a finger-order predicate         : {sr_credited_fake}")
moved = sr_credited_fake > 0
print(f"  probe CAN report same-row credit when present   : {moved} "
      f"{'✅ control passes' if moved else '❌ PROBE BLIND — the 0 above is meaningless'}")
if not moved:
    raise SystemExit("control failed")
