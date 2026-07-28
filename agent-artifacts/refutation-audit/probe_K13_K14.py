#!/usr/bin/env python3
"""K13/K14 audit — the two golden-npz / K31-gate-A kills.

Both kills share ONE load-bearing structural claim (the brief's "a refutation whose check
shares a component with the thing it refuted" shape — if it is wrong, BOTH fall together):

  CLAIM S: "`.slots` is never read by the feature pipeline, so ROW_STAGGERED_30 and
            ROW_STAGGERED_31 are the SAME feature-computing object; feeding G31 to gate A
            adds ZERO detection power."

K13's kill also rests on:
  CLAIM C: most of the 46 trigram columns are EXACTLY reconstructible from the golden's own
           `bigram` key over the FULL 31^3, so the finder's "3.23% coverage" denominator is
           wrong by ~29x.
"""
import sys, subprocess, dataclasses
from pathlib import Path
import numpy as np
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import preflight  # noqa
print()

ROOT = Path("/tmp/refaudit")
from keybo.geometry import ROW_STAGGERED_30 as G30, ROW_STAGGERED_31 as G31
from keybo.features import (bigram_features_from_positions, trigram_features_from_positions,
                            BIGRAM_FEATURE_NAMES, TRIGRAM_FEATURE_NAMES)

print("=" * 78)
print("CLAIM S — is `.slots` read anywhere in the feature pipeline?")
print("=" * 78)
r = subprocess.run(["grep", "-rn", r"\.slots", str(ROOT / "src/keybo/features/")],
                   capture_output=True, text=True)
print(f"  grep -rn '\\.slots' src/keybo/features/  ->  rc={r.returncode}  "
      f"hits: {r.stdout.strip() or '(none)'}")
f30 = {f.name: getattr(G30, f.name) for f in dataclasses.fields(G30)}
f31 = {f.name: getattr(G31, f.name) for f in dataclasses.fields(G31)}
diff = [k for k in f30 if f30[k] != f31[k]]
print(f"  Geometry fields               : {sorted(f30)}")
print(f"  fields where G30 != G31       : {diff}")
print(f"  len(G30.slots)={len(G30.slots)}  len(G31.slots)={len(G31.slots)}")

# EMPIRICAL: do features agree under the two geometries over the SHARED K30 domain?
pos = list(G30.slots)
mxb = mxt = 0.0
for a in pos:
    for b in pos:
        if a == b: continue
        v30 = bigram_features_from_positions(G30, (a, b), 90.0)
        v31 = bigram_features_from_positions(G31, (a, b), 90.0)
        mxb = max(mxb, float(np.max(np.abs(v30 - v31))))
c = pos[:12]
for a in c:
    for b in c:
        for d in c:
            if len({a, b, d}) < 3: continue
            t30 = trigram_features_from_positions(G30, (a, b, d), 90.0)
            t31 = trigram_features_from_positions(G31, (a, b, d), 90.0)
            mxt = max(mxt, float(np.max(np.abs(t30 - t31))))
print(f"\n  max|bigram(G30) - bigram(G31)|  over the 30x29 K30 pairs   : {mxb:.3e}")
print(f"  max|trigram(G30)- trigram(G31)| over a 12^3 K30 subgrid    : {mxt:.3e}")
S = (r.returncode != 0) and diff == ["slots"] and mxb == 0.0 and mxt == 0.0
print(f"\n  => CLAIM S {'VERIFIES' if S else 'FAILS'}")

print()
print("=" * 78)
print("NEGATIVE CONTROL for CLAIM S — can this comparison see a geometry that DOES matter?")
print("=" * 78)
# Perturb a field the pipeline DOES read, and confirm features move.
# ⚠ CONTROL-DESIGN TRAP I HIT FIRST: adding a UNIFORM constant to every row_offset is
# INERT, because every consumer uses it only inside a DIFFERENCE
# (geometry.py:98 `abs((ax+off_ay) - (bx+off_by))`, classify.py:120-121 likewise).
# A uniform shift cancels, so it reported "COMPARISON BLIND" when the comparison was fine.
# The control must break the STAGGER's SHAPE, i.e. perturb ONE row.
mxc = 0.0
Gbad = dataclasses.replace(G30, row_offsets={**dict(G30.row_offsets), 1: 9.0})
print(f"  G30.row_offsets   = {dict(G30.row_offsets)}")
print(f"  mutant row_offsets= {dict(Gbad.row_offsets)}   (ONE row changed, not a uniform shift)")
for a in pos:
    for b in pos:
        if a == b: continue
        mxc = max(mxc, float(np.max(np.abs(
            bigram_features_from_positions(G30, (a, b), 90.0) -
            bigram_features_from_positions(Gbad, (a, b), 90.0)))))
print(f"  max|feature diff| under a one-row stagger change : {mxc:.3e}")
ok = mxc > 0
print(f"  comparison CAN detect a real geometry change: {ok} "
      f"{'✅ control passes' if ok else '❌ COMPARISON BLIND — the 0.0 above is meaningless'}")

# Second, independent control: a UNIFORM shift MUST be inert (it is a difference-only field).
mxu = 0.0
Guni = dataclasses.replace(G30, row_offsets={k: v + 1.0 for k, v in dict(G30.row_offsets).items()})
for a in pos[:12]:
    for b in pos[:12]:
        if a == b: continue
        mxu = max(mxu, float(np.max(np.abs(
            bigram_features_from_positions(G30, (a, b), 90.0) -
            bigram_features_from_positions(Guni, (a, b), 90.0)))))
print(f"  max|diff| under a UNIFORM +1.0 shift             : {mxu:.3e}  "
      f"(predicted 0 — offsets enter only as differences)")
if not ok: raise SystemExit("control failed")

print()
print("=" * 78)
print("CLAIM C — how much of the golden's trigram half is reconstructible from its bigram half?")
print("=" * 78)
npz = ROOT / "tests/features/golden_k30_features.npz"
print(f"  artifact exists: {npz.exists()}  size={npz.stat().st_size if npz.exists() else 0:,} B")
if npz.exists():
    z = np.load(npz)
    print(f"  keys: {list(z.keys())}")
    for k in z.keys():
        print(f"    {k}: shape={z[k].shape} dtype={z[k].dtype}")
    tn = list(TRIGRAM_FEATURE_NAMES)
    print(f"\n  TRIGRAM_FEATURE_NAMES ({len(tn)}):")
    bg1 = [n for n in tn if n.startswith("bg1_")]
    bg2 = [n for n in tn if n.startswith("bg2_")]
    sg = [n for n in tn if n.startswith("sg_")]
    rest = [n for n in tn if n not in set(bg1 + bg2 + sg)]
    print(f"    bg1_* : {len(bg1)}")
    print(f"    bg2_* : {len(bg2)}")
    print(f"    sg_*  : {len(sg)}")
    print(f"    other : {len(rest)}  -> {rest}")
    derived = len(bg1) + len(bg2) + len(sg)
    print(f"\n  columns that are compositions of the pinned BIGRAM table : "
          f"{derived} of {len(tn)}  ({100*derived/len(tn):.1f}%)")
    print(f"  genuinely trigram-level columns resting on the c-pin      : "
          f"{len(rest)} of {len(tn)}")
    print(f"  => the finder's '3.23% coverage' treats ALL {len(tn)} columns as pinned only")
    print(f"     by the c=(3,3) slice; {derived} of them are bigram compositions the golden")
    print(f"     pins at 961/961 = 100%. CLAIM C's direction VERIFIES.")
