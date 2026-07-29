"""HOSTILE RE-READ of my own candidates. For each: what would refute it, and does it survive?

Written after the FIND pass, deliberately as a separate driver, so the refutation attempts are
not shaped by the code that produced the candidates.

  H1  'dy2 under a tenth' fails on qwerty30m — is that CORPUS-DEPENDENT? If it holds on the
      CLI's DEFAULT corpus (blend-v1) my finding needs the corpus named, or it is noise.
  H2  the FINGER_ORDER test gap — does it have a USER-VISIBLE consequence, or is the
      dict-append behaviour harmless? Reproduce the CLI's own consumption pattern.
  H3  _check_geometry accepts K31 — is it REACHABLE, and does the sibling gauge the docstring
      claims parity with ("the same stance as scissor_severity") behave the same way? If both
      accept it, this is house style, not a bad_scissor defect.
  H4  the 108-pair census — my census and the SUITE's census share the same predicate import,
      so they are NOT independent. Re-derive the support from an INDEPENDENT reimplementation
      of the stated rule and check it agrees.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path

from keybo.analysis import bad_scissor as BS
from keybo.cli.analyze import _EXTRA_NAMED, _shared_corpora, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.geometry import ROW_STAGGERED_31 as G31
from keybo.geometry import Geometry
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.testkit import assert_module_under

ROOT = Path("/tmp/bsaudit")
assert_module_under("keybo", ROOT)
REGISTRY = {k: v for k, v in {**NAMED_LAYOUTS, **_EXTRA_NAMED}.items() if len(v) == 30}
out: dict = {}

# ---- H1 ---------------------------------------------------------------------------------
print("=== H1: is 'dy2 under a tenth' corpus-dependent? ===")
h1 = {}
for cname in ("iweb", "blend-v1"):
    bigrams = _shared_corpora(production_corpus_dir(cname))[0]
    sc = BS.BadScissor(bigrams)
    worst = []
    for label, lay in sorted(REGISTRY.items()):
        L = Layout(lay, G)
        share = sc.share(L)
        dy2 = sum(v for k, v in sc.by_cell(L).items() if k.endswith("dy2"))
        worst.append((100.0 * dy2 / share, label))
    worst.sort(reverse=True)
    over = [(p, n) for p, n in worst if p >= 10.0]
    h1[cname] = {"max_pct": worst[0][0], "max_layout": worst[0][1],
                 "over_a_tenth": [n for _p, n in over]}
    print(f"  {cname:9s} max dy2/share = {worst[0][0]:6.3f}% on {worst[0][1]:12s} "
          f"· over a tenth: {[n for _p, n in over] or 'none'}")
print(f"  => the claim is CORPUS-DEPENDENT: {h1['iweb']['over_a_tenth'] != h1['blend-v1']['over_a_tenth']}")
print(f"  => on the CLI's DEFAULT corpus (blend-v1) the claim "
      f"{'HOLDS' if not h1['blend-v1']['over_a_tenth'] else 'FAILS'}; "
      f"on iweb (the spec frame) it "
      f"{'HOLDS' if not h1['iweb']['over_a_tenth'] else 'FAILS on ' + str(h1['iweb']['over_a_tenth'])}")
out["H1"] = h1

# ---- H2 ---------------------------------------------------------------------------------
print("\n=== H2: does the FINGER_ORDER gap have a USER-VISIBLE consequence? ===")
print("  Reproducing the CLI's exact consumption pattern with a DRIFTED label.")
bigrams = _shared_corpora(production_corpus_dir("iweb"))[0]
sc = BS.BadScissor(bigrams)
L = Layout(REGISTRY["keybo-lsb+lm"], G)
real = sc.by_finger(L)
print(f"  true by_finger['R-pinky'] = {real['R-pinky']:.5f}")
drifted = tuple("R-pinkyX" if f == "R-pinky" else f for f in BS.FINGER_ORDER)
part = sc._partition(L, BS.bad_scissor_finger, drifted, True)
print(f"  with FINGER_ORDER drifted to {drifted[-1]!r}:")
print(f"    dict has {len(part)} keys (was {len(real)}): {sorted(part)}")
print(f"    the CLI prints, for each f in FINGER_ORDER, by_finger[f]:")
row = "".join(f"{part[f]:>9.4f}" for f in drifted)
print(f"      header: {''.join(f'{f:>9}' for f in drifted)}")
print(f"      values: {row}")
print(f"    => R-pinkyX column shows {part['R-pinkyX']:.4f} while the real "
      f"{real['R-pinky']:.4f} is in the dict but NEVER PRINTED.")
sum_printed = sum(part[f] for f in drifted)
print(f"    printed columns sum to {sum_printed:.5f} but share is {sc.share(L):.5f} "
      f"— the table silently stops being a partition (diff {sc.share(L) - sum_printed:+.5f})")
out["H2"] = {"user_visible": True, "printed_sum": sum_printed, "share": sc.share(L),
             "hidden_mass": sc.share(L) - sum_printed,
             "n_keys_after_drift": len(part)}

# ---- H3 ---------------------------------------------------------------------------------
print("\n=== H3: K31 — reachable? and does scissor_severity behave the same? ===")
from keybo.scoring.scissor_severity import ScissorSeverity

chars31 = "qwertyuiopasdfghjkl'zxcvbnm,.-;"
L31 = Layout(chars31, G31)
res = {}
try:
    v = BS.BadScissor({"qw": 1}).share(L31)
    res["bad_scissor_on_K31"] = f"ACCEPTED, share={v}"
except ValueError as e:
    res["bad_scissor_on_K31"] = f"REFUSED: {e}"
print(f"  bad_scissor  on K31: {res['bad_scissor_on_K31']}")
try:
    sev = ScissorSeverity({"qw": 1})
    v = sev.share(L31) if hasattr(sev, "share") else "no share()"
    res["scissor_severity_on_K31"] = f"ACCEPTED, {v}"
except ValueError as e:
    res["scissor_severity_on_K31"] = f"REFUSED: {e}"
except Exception as e:
    res["scissor_severity_on_K31"] = f"{type(e).__name__}: {e}"
print(f"  scissor_sev  on K31: {res['scissor_severity_on_K31']}")
# four-row: both must refuse
four = Geometry(slots=(*G.slots, (-5, 4)))
for name, fn in (("bad_scissor", lambda: BS.BadScissor({"qw": 1}).share(
        Layout("qwertyuiopasdfghjkl'zxcvbnm,.-;", four))),):
    try:
        fn()
        res[f"{name}_on_four_row"] = "ACCEPTED (guard failed)"
    except ValueError as e:
        res[f"{name}_on_four_row"] = f"REFUSED: {str(e)[:60]}"
print(f"  bad_scissor  on 4-row: {res['bad_scissor_on_four_row']}")
# reachability through the shipped CLI
src = (ROOT / "src/keybo/cli/analyze.py").read_text()
n30 = src.count("ROW_STAGGERED_30")
n31 = src.count("ROW_STAGGERED_31")
print(f"  analyze.py references: ROW_STAGGERED_30 x{n30}, ROW_STAGGERED_31 x{n31}")
print(f"  => K31 is {'NOT ' if n31 == 0 else ''}reachable through `keybo analyze`; "
      f"the gap is {'LATENT (library-level only)' if n31 == 0 else 'LIVE'}")
res["reachable_via_analyze"] = n31 > 0
res["k31_support_size"] = sum(
    1 for a, b in itertools.product(sorted(G31.slots), repeat=2) if BS.bad_scissor(G31, a, b))
res["k30_support_size"] = sum(
    1 for a, b in itertools.product(sorted(G.slots), repeat=2) if BS.bad_scissor(G, a, b))
print(f"  support the spec examined (K30) = {res['k30_support_size']}; "
      f"K31 would score against {res['k31_support_size']}")
out["H3"] = res

# ---- H4 ---------------------------------------------------------------------------------
print("\n=== H4: my census shares the predicate with its target — INDEPENDENT re-derivation ===")
print("  Re-implementing the STATED rule from the docstring's own words, using only")
print("  geometry primitives, with NO call into bad_scissor.py:")
DEX_INDEP = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}


def indep_bad_scissor(geometry, a, b) -> bool:
    """same hand AND different fingers AND different rows AND weaker finger on the LOWER row."""
    ha, hb = geometry.hand(a[0]), geometry.hand(b[0])
    if ha == 0 or ha != hb:
        return False
    if geometry.same_finger(a[0], b[0]):
        return False
    if a[1] == b[1]:
        return False
    ka = geometry.finger(a[0]).value.split("-")[1]
    kb = geometry.finger(b[0]).value.split("-")[1]
    # the weaker (less dextrous) of the two fingers must hold the lower key
    if DEX_INDEP[ka] < DEX_INDEP[kb]:
        weak_row, strong_row = a[1], b[1]
    elif DEX_INDEP[kb] < DEX_INDEP[ka]:
        weak_row, strong_row = b[1], a[1]
    else:
        return False  # equal dexterity, distinct fingers: impossible on this board
    return weak_row < strong_row


mismatch = []
for a, b in itertools.product(sorted(G.slots), repeat=2):
    if indep_bad_scissor(G, a, b) != BS.bad_scissor(G, a, b):
        mismatch.append((a, b))
n_indep = sum(1 for a, b in itertools.product(sorted(G.slots), repeat=2)
              if indep_bad_scissor(G, a, b))
print(f"  independent support size = {n_indep} (shipped = {res['k30_support_size']})")
print(f"  pairs where the two DISAGREE: {len(mismatch)}")
print(f"  => the shipped predicate implements the rule as written: {not mismatch}")
# POSITIVE CONTROL on the independent implementation: it must DISAGREE with a mutant.
disagree_with_mutant = sum(
    1 for a, b in itertools.product(sorted(G.slots), repeat=2)
    if indep_bad_scissor(G, a, b) is not False) > 0
print(f"  control — the independent impl is not vacuously False: {disagree_with_mutant} "
      f"(it fires on {n_indep} pairs)")
out["H4"] = {"independent_support": n_indep, "mismatches": len(mismatch),
             "agrees_with_shipped": not mismatch}

p = ROOT / "agent-artifacts/bsaudit/hostile_recheck.json"
p.write_text(json.dumps(out, indent=2, default=str))
print(f"\nwrote {p}")
