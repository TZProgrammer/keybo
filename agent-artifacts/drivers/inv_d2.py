"""D follow-up: PROVE the mirror-asymmetry statistic is BLIND to the SIGN of the stagger.

Claim (algebraic): dx(a,b)      = |Dx + Doff|   where Dx = ax-bx, Doff = off(ay)-off(by)
                   dx(mir a,mir b) = |-(Dx) + Doff| = |Dx - Doff|
Negating ALL offsets (Doff -> -Doff) SWAPS those two values. So negating the offsets maps the
pair (a,b) onto the pair (mir a, mir b) featurewise, and |T2[a,b] - T2[mir a,mir b]| is an
ABSOLUTE difference => the MULTISET of 870 asymmetries is invariant. Hence mean/median/p90/max
are identical, and NO mirror-asymmetry statistic can identify the SIGN of the stagger.
If true, minimizing mirror asymmetry drives you to ZERO stagger (a flat ortho board), which is
physically wrong for ANSI -- so mirror asymmetry is NOT a valid fitting objective for stagger.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.features import bigram_features_from_positions
from keybo.features.ngram import _placement_row_from_positions
from keybo.analysis.timecard import _load_gz_model
from keybo.scoring.model_scorer import predict_ms

G=ROW_STAGGERED_30; WPM=90.0
slots=list(G.slots); idx={p:i for i,p in enumerate(slots)}
mir=lambda p:(-p[0],p[1])
pairs=[(a,b) for a in slots for b in slots if a!=b]
def gm(ot,ob,home=0.0): return Geometry(slots=G.slots,row_offsets={1:ob,2:home,3:ot})

print("=== [1] FEATURE-LEVEL proof: negating offsets maps (a,b) -> (mir a, mir b) exactly ===")
for (ot,ob) in [(-0.25,0.5), (-0.5,0.25), (0.375,-0.125), (-1.0,1.0)]:
    gP, gN = gm(ot,ob), gm(-ot,-ob)
    worst=0.0; nbad=0
    for a,b in pairs:
        rP = _placement_row_from_positions(gP, a, b)          # features of (a,b) under +offsets
        rN = _placement_row_from_positions(gN, mir(a), mir(b)) # features of (mir a,mir b) under -offsets
        d = max(abs(rP[k]-rN[k]) for k in rP)
        worst=max(worst,d); nbad += (d>1e-12)
    print(f"  offsets ({ot:+.3f},{ob:+.3f}): max |feat(a,b | +off) - feat(mir a,mir b | -off)| = {worst:.3e}  "
          f"pairs differing: {nbad}/{len(pairs)}")

print("\n=== [2] MODEL-LEVEL: the 870-asymmetry MULTISET under +offsets vs -offsets ===")
models=[_load_gz_model(f"bigram_reg31_seed{s}") for s in (0,1,2)]
def asym_vec(g):
    vecs=np.vstack([bigram_features_from_positions(g,(a,b),wpm=WPM) for a in slots for b in slots])
    T2=np.mean([predict_ms(m,vecs).reshape(30,30) for m in models],axis=0)
    return np.array([abs(T2[idx[a],idx[b]]-T2[idx[mir(a)],idx[mir(b)]]) for a,b in pairs])
for (ot,ob) in [(-0.25,0.5), (-0.5,0.25), (0.375,-0.125)]:
    vP, vN = asym_vec(gm(ot,ob)), asym_vec(gm(-ot,-ob))
    print(f"  ({ot:+.3f},{ob:+.3f}) vs negated: sorted-multiset identical? "
          f"{np.allclose(np.sort(vP),np.sort(vN),atol=1e-12)}  "
          f"mean {vP.mean():.6f} vs {vN.mean():.6f}   max {vP.max():.6f} vs {vN.max():.6f}  "
          f"elementwise identical? {np.allclose(vP,vN,atol=1e-12)}")

print("\n=== [3] CONSEQUENCE: what does MINIMIZING mirror asymmetry choose? ===")
best=None
for ot in np.arange(-0.5,0.501,0.125):
    row=[]
    for ob in np.arange(-0.5,0.501,0.125):
        m=asym_vec(gm(round(ot,4),round(ob,4))).mean()
        row.append(f"{m:6.3f}")
        if best is None or m<best[0]: best=(m,round(ot,4),round(ob,4))
    print(f"  top={ot:+.3f} | " + " ".join(row))
print(f"  (columns are bottom offset -0.5 .. +0.5 step 0.125)")
print(f"  ARGMIN of mean mirror asymmetry: mean={best[0]:.6f} at top={best[1]:+.3f} bottom={best[2]:+.3f}")
print("  => minimizing mirror asymmetry selects ZERO STAGGER (a flat ortho board), which is")
print("     PHYSICALLY WRONG for a row-staggered ANSI keyboard. Mirror asymmetry is therefore")
print("     NOT a usable objective for fitting the stagger: it is a magnitude penalty, blind to sign.")
