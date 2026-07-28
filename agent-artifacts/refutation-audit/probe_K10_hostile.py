#!/usr/bin/env python3
"""HOSTILE SELF-REVIEW of my own K10 resurrection.

I resurrected K10 as UNSUPPORTED. Now I attack that. Four attacks, strongest first:

A1. "GROUND A was never load-bearing." A 2/3 kill needs only 2 refuting votes. If vote 1's
    leg (1) — the invented-should argument — is SUFFICIENT on its own, then killing GROUND A
    changes nothing: the kill stands on leg (1) + vote 3. Test: is leg (1) sufficient?

A2. "The scorer DOES disclose it, one call away." classify.is_inwards' docstring states the
    row test. If the reader of DEFAULT_OXEY_WEIGHTS is expected to follow the predicate, the
    disclosure exists. Test: is there any in-repo text that ties the scorer's roll TERMS to
    the cross-row-only predicate?

A3. "My own check shares a component with its target." I used C.is_inwards/is_outwards to
    show same-row pairs get no credit — the very predicates under accusation. If they were
    wrong in the other direction my probe would inherit the error. Test: re-derive the
    same-row exclusion WITHOUT calling those predicates at all.

A4. "The population figure is a qwerty-flattering artifact / the mass claim is unsupported."
    The finder's 32-63% is corpus mass, not pair count. Test whether the PAIR-count claim
    (108 of 324) is what I actually verified, and mark the mass claim's status honestly.
"""
import sys, subprocess, inspect
from pathlib import Path
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import preflight  # noqa
print()
ROOT = Path("/tmp/refaudit")
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring import oxey

print("=" * 78)
print("A3 — re-derive the same-row exclusion WITHOUT using is_inwards/is_outwards")
print("=" * 78)
print("  The predicates' own SOURCE, read as text (not called):")
for fn in (C.is_inwards, C.is_outwards):
    src = inspect.getsource(fn)
    body = [l.strip() for l in src.splitlines() if "outer[1]" in l or "abs(a[0]) == abs(b[0])" in l]
    print(f"    {fn.__name__}: {body}")
print("\n  Both return `outer[1] > inner[1]` / `outer[1] < inner[1]`, where index [1] is the")
print("  ROW. For a SAME-ROW pair outer[1] == inner[1], so BOTH strict comparisons are False.")
print("  This is a pure algebraic fact about the returned expression -- it does not depend on")
print("  trusting the predicates' correctness, only on reading them.")
# independent structural derivation: enumerate rows directly
lay = Layout(NAMED_LAYOUTS["qwerty"], G)
chars = list(NAMED_LAYOUTS["qwerty"])
elig = same_row = 0
for a in chars:
    for b in chars:
        if a == b: continue
        pa, pb = lay.pos(a), lay.pos(b)
        ha, hb = G.hand(pa[0]), G.hand(pb[0])
        if ha == 0 or hb == 0 or ha != hb: continue
        if G.same_finger(pa[0], pb[0]): continue
        elig += 1
        if pa[1] == pb[1]: same_row += 1
print(f"\n  Structural count (row equality only, NO predicate call):")
print(f"    eligible same-hand two-finger ordered pairs : {elig}")
print(f"    of which same-row (pa[1] == pb[1])          : {same_row}")
print(f"  => the excluded population is {same_row} of {elig} = {100*same_row/elig:.1f}% of pairs,")
print(f"     established without calling the accused predicates. A3 does NOT overturn it.")

print()
print("=" * 78)
print("A2 — is the cross-row-only gating disclosed ANYWHERE at the scorer?")
print("=" * 78)
osrc = (ROOT / "src/keybo/scoring/oxey.py").read_text()
for pat in ("row", "same-row", "cross-row", "higher row", "lower row"):
    n = osrc.lower().count(pat)
    print(f"    occurrences of {pat!r:14s} in scoring/oxey.py : {n}")
hits = [l.strip() for l in osrc.splitlines() if "row" in l.lower()]
print(f"    every line containing 'row': {hits}")
print(f"\n  And the two weight labels, verbatim:")
for k in ("inroll", "outroll"):
    print(f"    {k:8s}: {oxey.DEFAULT_OXEY_WEIGHTS[k][1]!r}")
print(f"\n  => neither label, nor the module docstring's 'rolls (rewarded)', qualifies the")
print(f"     population. A2 does NOT overturn the disclosure gap AT THIS SITE.")
print(f"     (Honest concession: classify.is_inwards' own docstring DOES disclose it, so a")
print(f"      reader who follows the call chain can find it. That is why the correct label is")
print(f"      UNSUPPORTED/rank-4, not WRONG -- and why this is a weaker finding than the")
print(f"      finder claimed.)")

print()
print("=" * 78)
print("A1 — was GROUND A load-bearing? Could the kill stand on the other legs alone?")
print("=" * 78)
print("  The kill needed 2 of 3 refuting votes. It had vote 1 and vote 3.")
print("  If EITHER vote is independently sufficient, removing GROUND A (vote 1's leg 2)")
print("  does not resurrect the finding. So I must show BOTH votes' remaining legs fail.")
print()
print("  VOTE 1's legs: (1) invented should [partially holds -- at the PREDICATE, not the")
print("     scorer]; (2) sr-roll credits it elsewhere [FAILS -- trigram metric, other module];")
print("     (3) the 50% figure is a tautology [HOLDS -- and it kills a SUPPORTING leg of the")
print("     finding, not the core population claim]; (4) effect_curves bucketed them [HOLDS,")
print("     but effect_curves is not the accused site]; (5) the exemption is a strawman")
print("     [HOLDS -- the finder misquoted the docstring, as vote 2 also found].")
print("  VOTE 3's legs: (1) already registered [FAILS as applied -- all cites are schema/")
print("     driver/effect_curves]; (2) already one of the four known-defective terms [VERIFIES")
print("     but registers the WEIGHT RATIO, not the population gap]; (3) the finder falsified")
print("     a reason the code never gave [HOLDS].")
print()
print("  => Both votes' SURVIVING legs attack the finder's FRAMING and its supporting")
print("     arguments (the invented 'should', the tautological 50%, the misquote). NEITHER")
print("     surviving leg addresses the core factual claim: the scorer's rewarded-roll terms")
print("     price 216 of 324 eligible pairs and nothing at that site says so.")
print("  => GROUND A and vote 3's leg (1) were the only legs aimed at the CORE claim, and")
print("     both fail. A1 is the strongest attack and it does NOT save the kill -- but it DOES")
print("     mean the resurrection is of a NARROWER finding than the finder wrote.")

print()
print("=" * 78)
print("A4 — what did I actually verify, and what remains UNVERIFIED?")
print("=" * 78)
print("  VERIFIED by me: the PAIR-COUNT claim (108 of 324 eligible pairs uncredited),")
print("    the sr-roll unreachability, the absence of disclosure at the scorer, the absence")
print("    of a ledger registration for this population.")
print("  NOT VERIFIED by me: the finder's '32-63% of eligible MASS' (corpus-weighted) and")
print("    the 'sparing qwerty most' asymmetry. Vote 2 (the non-refuter) reproduced those")
print("    numbers, but vote 2 ALSO showed the finder's counterfactual sizes a FULL")
print("    finger-order redefinition, not same-row-mass-added -- so the magnitude half of")
print("    the finding is UNSUPPORTED-as-stated by its own supporter.")
print("  => I resurrect the POPULATION/DISCLOSURE claim only. The magnitude claim stays dead.")
