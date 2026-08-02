"""A1 follow-up: is the c=-0.30 letter-letter non-identity STRUCTURAL or FLOATING-POINT?"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.features import bigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES

G30 = ROW_STAGGERED_30
base = dict(G30.row_offsets)
letters = list(G30.slots)
pairs_LL = [(a,b) for a in letters for b in letters]
def gw(off): return Geometry(slots=G30.slots, row_offsets=dict(off))
def fm(g, ps): return np.array([bigram_features_from_positions(g, p, 90.0) for p in ps])
M0 = fm(G30, pairs_LL)

print("=== magnitude of the letter-letter difference under a uniform shift ===")
for c in (0.5, 1.0, 7.0, -0.25, 0.25, 0.125, -0.30, 0.1, 1.0/3.0, 0.7):
    M1 = fm(gw({k: v+c for k,v in base.items()}), pairs_LL)
    d = np.abs(M0-M1)
    exact = np.array_equal(M0, M1)
    # is c exactly representable as a dyadic rational (binary fraction)?
    dyadic = (c * 2**20) == int(c * 2**20) and abs(c*2**20) < 2**52
    print(f"  c={c:+.6f} dyadic={str(dyadic):5s} exact-equal={str(exact):5s} max|d|={d.max():.3e} "
          f"n_cells_differing={(d>0).sum()}  cols={[BIGRAM_FEATURE_NAMES[j] for j in range(d.shape[1]) if d[:,j].max()>0]}")

print("\n=== does any THRESHOLDED feature (is_lsb dx>1.5, lateral_span, angle) flip under the float noise? ===")
for c in (-0.30, 0.1, 0.7, 1.0/3.0):
    M1 = fm(gw({k: v+c for k,v in base.items()}), pairs_LL)
    d = np.abs(M0-M1)
    for j,name in enumerate(BIGRAM_FEATURE_NAMES):
        if d[:,j].max() > 1e-9:
            print(f"  c={c:+.4f} col {name!r} moves by {d[:,j].max():.3e}  <-- LARGER THAN FLOAT NOISE")
    binary_cols = [j for j,n in enumerate(BIGRAM_FEATURE_NAMES) if set(np.unique(M0[:,j]))<= {0.0,1.0}]
    flips = sum(int((M0[:,j]!=M1[:,j]).sum()) for j in binary_cols)
    print(f"  c={c:+.6f}: binary/indicator column flips = {flips}  (0 => no threshold crossed)")

print("\n=== the SPACE offset: is it a real free parameter? which cols does it reach ===")
SPACE=(0,0)
sp = [(a,SPACE) for a in letters]+[(SPACE,b) for b in letters]+[(SPACE,SPACE)]
# perturb ONLY the notion of space's offset -- can't via row_offsets (y=0 not a key). Prove it:
g_with_y0 = gw({**base, 0: 0.9})
d = np.abs(fm(G30, sp) - fm(g_with_y0, sp))
print(f"  adding row_offsets[0]=0.9 -> space pairs move: max|d|={d.max():.4f} on cols "
      f"{[BIGRAM_FEATURE_NAMES[j] for j in range(d.shape[1]) if d[:,j].max()>0]}")
print("  => y=0 IS a settable key; the shipped dict simply omits it, pinning space at 0.0 via .get default")
dLL = np.abs(fm(G30,pairs_LL)-fm(g_with_y0,pairs_LL)).max()
print(f"  and it touches NO letter-letter pair (max|d|={dLL:.1e}) => it is an ORTHOGONAL 4th knob")
