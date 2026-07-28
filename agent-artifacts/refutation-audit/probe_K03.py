#!/usr/bin/env python3
"""K3 audit — the same_hand_other "defining SHAP feature" kill.

FINDING (killed 2/3): same_hand_other's "defining SHAP feature" (`same_hand`) fires on 420
of 870 pairs while the class is 108, and is the reference class's own column.

The kill rests on FOUR checkable legs. Two matter:
  L1 (vote 1 leg 1): "the docstring promise is met as written — 'over the class's pairs' —
      so the 312 over-covered pairs are NEVER averaged."
  L3 (vote 1 leg 3): "the exactness sweep MIXES TWO POLARITIES. Under one consistent
      convention `alternate` is the worse offender (870 mismatches vs 312)."

L3 is the decisive one, because if true the finder's own metric indicts the class it uses as
the clean baseline, i.e. the metric is an artifact of convention.
"""
import sys, inspect
import numpy as np
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import preflight  # noqa
print()

from keybo.analysis import effect_curves as EC
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30
from keybo.features import bigram_model_row  # noqa: F401  (schema source)

g = ROW_STAGGERED_30
positions = list(g.slots)
n = len(positions)
pairs = [(i, j) for i in range(n) for j in range(n)]

def mask_of(pred):
    m = np.zeros(n * n, dtype=bool)
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            if i != j and a[0] != 0 and b[0] != 0 and pred(g, a, b):
                m[i * n + j] = True
    return m

masks = {cls: mask_of(pred) for cls, (pred, _f) in EC.PATTERN_CLASSES.items()}
same_hand_fires = mask_of(lambda gg, a, b: (i_ := None) or C.same_hand(gg, a, b))
# eligible grid the compute path uses (i!=j, no thumb column)
eligible = np.zeros(n * n, dtype=bool)
for i, a in enumerate(positions):
    for j, b in enumerate(positions):
        if i != j and a[0] != 0 and b[0] != 0:
            eligible[i * n + j] = True

print("=" * 78)
print("L1 — is the class mask a SUBSET of the named feature's firing set?")
print("=" * 78)
src = inspect.getsource(EC)
print("  compute line (verbatim):")
for ln in src.splitlines():
    if "shap_ms_matrix[mask]" in ln:
        print("     ", ln.strip())
print(f"  docstring standard: 'mean SHAP contribution of the class's *defining* feature "
      f"column(s)\\n                      OVER THE CLASS'S PAIRS'")
sho = masks["same_hand_other"]
sh = same_hand_fires & eligible
print(f"  |same_hand_other class|            : {sho.sum()}")
print(f"  |same_hand fires (eligible grid)|  : {sh.sum()}")
print(f"  class rows NOT in same_hand fires  : {(sho & ~sh).sum()}  "
      f"(0 => class SUBSET-OF fires => the 312 extras are never averaged)")
print(f"  same_hand constant on the class?   : "
      f"{len(set(sh[sho].tolist())) == 1} (n_distinct={len(set(sh[sho].tolist()))})")
L1 = (sho & ~sh).sum() == 0
print(f"\n  => L1 (docstring met as written): {'VERIFIES' if L1 else 'FAILS'}")

print()
print("=" * 78)
print("L3 — does the finder's exactness metric MIX POLARITIES?")
print("=" * 78)
alt = masks["alternate"]
not_sh = (~same_hand_fires) & eligible
def cmp(name, cls_mask, feat_mask):
    extra = (feat_mask & ~cls_mask).sum()
    missing = (cls_mask & ~feat_mask).sum()
    return f"class={cls_mask.sum():4d} extra={extra:4d} missing={missing:4d} " \
           f"exact={extra == 0 and missing == 0} total_mismatch={extra + missing}"
print(f"  alternate       vs same_hand==0 : {cmp('alt', alt, not_sh)}")
print(f"  alternate       vs same_hand==1 : {cmp('alt', alt, sh)}")
print(f"  same_hand_other vs same_hand==1 : {cmp('sho', sho, sh)}")
print(f"  same_hand_other vs same_hand==0 : {cmp('sho', sho, not_sh)}")
alt_1 = (sh & ~alt).sum() + (alt & ~sh).sum()
sho_1 = (sh & ~sho).sum() + (sho & ~sh).sum()
print(f"\n  Under ONE consistent convention (feature==1, which is what 'the named column")
print(f"  fires' MEANS for a 0/1 indicator):")
print(f"     alternate       total mismatch = {alt_1}")
print(f"     same_hand_other total mismatch = {sho_1}")
L3 = alt_1 > sho_1
print(f"  => alternate is the WORSE offender: {L3} "
      f"({alt_1} vs {sho_1})  => L3 {'VERIFIES' if L3 else 'FAILS'}")

print()
print("=" * 78)
print("L4 — does ANY exact single-column indicator of same_hand_other exist?")
print("=" * 78)
from keybo.features import BIGRAM_FEATURE_NAMES
import keybo.features as F
# build the feature matrix over the eligible grid for one wpm, then test each column
try:
    rows = []
    from keybo.layout import Layout
    from keybo.layouts import NAMED_LAYOUTS
    lay = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    print(f"  BIGRAM_FEATURE_NAMES ({len(BIGRAM_FEATURE_NAMES)}): {list(BIGRAM_FEATURE_NAMES)}")
except Exception as e:
    print("  (schema introspection unavailable:", e, ")")
print("  => vote 1 claims NONE exists; the 20-col schema has no same_hand_other column,")
print("     which is visible from the name list above (no 'same_hand_other' / equivalent).")

print()
print("=" * 78)
print("NEGATIVE CONTROL — can this probe report a NON-exact class as exact, or vice versa?")
print("=" * 78)
for cls in ("sfb", "outer_high", "outer_low", "scissor", "lsb"):
    m = masks[cls]
    feat = mask_of({"sfb": lambda gg,a,b: C.same_finger(gg,a,b),
                    "outer_high": C.is_inwards, "outer_low": C.is_outwards,
                    "scissor": C.is_scissor, "lsb": C.is_lsb}[cls]) & eligible
    if cls == "sfb":
        feat = feat & eligible
    extra = (feat & ~m).sum(); missing = (m & ~feat).sum()
    print(f"  {cls:12s} class={m.sum():4d} feat={feat.sum():4d} extra={extra} missing={missing}")
print("  (these SHOULD be exact-or-near; a probe that called them all 420 would be broken)")
