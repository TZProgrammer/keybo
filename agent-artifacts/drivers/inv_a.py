"""INVARIANT A: identifiability of row_offsets as the MODEL sees them.

Structural claim to test: offsets enter ONLY inside differences
    (ax + off(ay)) - (bx + off(by))
so a uniform shift over the rows PRESENT IN THE PAIR cancels.

The suspected hole: row_offsets is {1,2,3}; SPACE is (0,0) and y=0 is NOT a key of the
dict, so `.get(ay, 0.0)` pins space's offset at 0.0 and it does NOT move under a shift.
If space bigrams reach the feature matrix, a uniform shift does NOT cancel for them.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import sys, itertools, json
import numpy as np

import keybo
print("keybo.__file__ =", keybo.__file__, flush=True)

from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31, Geometry
from keybo.features import bigram_features_from_positions, trigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES

G30 = ROW_STAGGERED_30
G31 = ROW_STAGGERED_31
print("shipped row_offsets:", G30.row_offsets)
print("G30 slots[:3] =", G30.slots[:3], " -> confirms y of top row")
print("n bigram feature cols:", len(BIGRAM_FEATURE_NAMES))

SPACE = (0, 0)

def geom_with(off, base=G30):
    return Geometry(slots=base.slots, row_offsets=dict(off), space_position=base.space_position)

def featmat(geom, pairs, wpm=90.0):
    return np.array([bigram_features_from_positions(geom, p, wpm) for p in pairs])

# --- universes -------------------------------------------------------------------------
letters = list(G30.slots)                       # 30 letter slots
all_keys = letters + [SPACE]
pairs_LL = [(a, b) for a in letters for b in letters]              # 900 ordered letter-letter
pairs_sp = [(a, SPACE) for a in letters] + [(SPACE, b) for b in letters] + [(SPACE, SPACE)]
pairs_all = pairs_LL + pairs_sp
print(f"\nuniverse: {len(pairs_LL)} letter-letter ordered pairs, {len(pairs_sp)} space-involving, {len(pairs_all)} total")

# --- A1. UNIFORM SHIFT ------------------------------------------------------------------
print("\n=== A1: uniform shift of all three row_offsets ===")
base_off = dict(G30.row_offsets)
for c in (0.5, -0.3, 1.0, 7.0):
    shifted = {k: v + c for k, v in base_off.items()}
    g2 = geom_with(shifted)
    M0_LL, M1_LL = featmat(G30, pairs_LL), featmat(g2, pairs_LL)
    M0_sp, M1_sp = featmat(G30, pairs_sp), featmat(g2, pairs_sp)
    ident_LL = np.array_equal(M0_LL, M1_LL)
    ident_sp = np.array_equal(M0_sp, M1_sp)
    nrow_sp = int((np.abs(M0_sp - M1_sp) > 0).any(axis=1).sum())
    maxd_sp = float(np.abs(M0_sp - M1_sp).max())
    print(f"  c={c:+.2f}  letter-letter bit-identical: {ident_LL}   "
          f"space-involving bit-identical: {ident_sp}  (rows differing: {nrow_sp}/{len(pairs_sp)}, max|d|={maxd_sp:.4f})")
    if not ident_sp:
        cols = [BIGRAM_FEATURE_NAMES[j] for j in range(M0_sp.shape[1]) if np.abs(M0_sp[:, j] - M1_sp[:, j]).max() > 0]
        print(f"      columns that MOVE on space pairs: {cols}")

# --- A2. SAME-ROW pairs carry zero information -----------------------------------------
print("\n=== A2: same-row pairs vs cross-row pairs under an ARBITRARY offset perturbation ===")
rng = np.random.default_rng(0)
same_row_pairs  = [(a, b) for a, b in pairs_LL if a[1] == b[1]]
cross_row_pairs = [(a, b) for a, b in pairs_LL if a[1] != b[1]]
print(f"  letter-letter: {len(same_row_pairs)} same-row, {len(cross_row_pairs)} cross-row")
for trial in range(3):
    pert = {k: base_off[k] + float(rng.normal(0, 0.4)) for k in base_off}
    g2 = geom_with(pert)
    d_same  = np.abs(featmat(G30, same_row_pairs)  - featmat(g2, same_row_pairs)).max()
    d_cross = np.abs(featmat(G30, cross_row_pairs) - featmat(g2, cross_row_pairs)).max()
    n_cross = int((np.abs(featmat(G30, cross_row_pairs) - featmat(g2, cross_row_pairs)) > 0).any(axis=1).sum())
    print(f"  trial {trial} offsets={ {k: round(v,4) for k,v in pert.items()} }  "
          f"max|d| same-row = {d_same:.10f}   max|d| cross-row = {d_cross:.6f}  (rows moving {n_cross}/{len(cross_row_pairs)})")

# --- A3. parameter count: which offsets are individually identified? ---------------------
print("\n=== A3: per-row identifiability — perturb ONE row at a time ===")
for target in (1, 2, 3):
    pert = dict(base_off); pert[target] += 0.37
    g2 = geom_with(pert)
    dLL = np.abs(featmat(G30, pairs_LL) - featmat(g2, pairs_LL)).max()
    dsp = np.abs(featmat(G30, pairs_sp) - featmat(g2, pairs_sp)).max()
    print(f"  row y={target} +0.37 -> max|d| letter-letter {dLL:.6f} | space-involving {dsp:.6f}")

print("\n=== A3b: is the LETTER-ONLY map invariant to a uniform shift => 2 params; "
      "does SPACE break it => 3 params? ===")
# Rank of the map offsets->features, numerically, over each universe.
def jac(pairs, h=1e-5):
    """Numerical Jacobian of the (concatenated) feature vector wrt the 3 offsets."""
    cols = []
    for k in (1, 2, 3):
        up, dn = dict(base_off), dict(base_off)
        up[k] += h; dn[k] -= h
        fu = featmat(geom_with(up), pairs).ravel()
        fd = featmat(geom_with(dn), pairs).ravel()
        cols.append((fu - fd) / (2 * h))
    return np.array(cols).T   # (n_out, 3)

for name, pairs in (("letter-letter only", pairs_LL), ("space-involving only", pairs_sp), ("ALL pairs", pairs_all)):
    J = jac(pairs)
    s = np.linalg.svd(J, compute_uv=False)
    rank = int((s > 1e-6 * max(1.0, s[0])).sum())
    ns = None
    if rank < 3:
        _, _, Vt = np.linalg.svd(J)
        ns = Vt[rank:]
    print(f"  {name:22s} numerical rank = {rank}   singular values = {np.round(s, 6)}")
    if ns is not None:
        print(f"      null space (directions the FEATURES CANNOT SEE), rows=(y1,y2,y3): {np.round(ns/np.abs(ns).max(), 4)}")

# --- A4. trigram frame: same question ---------------------------------------------------
print("\n=== A4: trigram frame (sg_dx etc.) — uniform shift ===")
tri_LL = [(a, b, c) for a in letters[::4] for b in letters[::5] for c in letters[::6]]
tri_sp = [(a, SPACE, c) for a in letters[::3] for c in letters[::3]] + \
         [(SPACE, b, c) for b in letters[::3] for c in letters[::3]] + \
         [(a, b, SPACE) for a in letters[::3] for b in letters[::3]]
def trimat(geom, tris, wpm=90.0):
    return np.array([trigram_features_from_positions(geom, t, wpm) for t in tris])
g2 = geom_with({k: v + 0.5 for k, v in base_off.items()})
print(f"  trigram letter-only  ({len(tri_LL)} triples) bit-identical: {np.array_equal(trimat(G30, tri_LL), trimat(g2, tri_LL))}")
dsp = np.abs(trimat(G30, tri_sp) - trimat(g2, tri_sp))
print(f"  trigram space-touching ({len(tri_sp)} triples) bit-identical: {not (dsp>0).any()}  max|d|={dsp.max():.4f}")

# --- A5. K31 geometry sanity ------------------------------------------------------------
print("\n=== A5: K31 (the geometry the SHIPPED models were trained under) ===")
print("  G31.row_offsets:", G31.row_offsets, " (same dict default)")
l31 = list(G31.slots)
p31 = [(a, b) for a in l31 for b in l31]
g31b = Geometry(slots=G31.slots, row_offsets={k: v + 0.5 for k, v in base_off.items()})
m0 = np.array([bigram_features_from_positions(G31, p, 90.0) for p in p31])
m1 = np.array([bigram_features_from_positions(g31b, p, 90.0) for p in p31])
print(f"  K31 letter-letter ({len(p31)} pairs) uniform-shift bit-identical: {np.array_equal(m0, m1)}")
