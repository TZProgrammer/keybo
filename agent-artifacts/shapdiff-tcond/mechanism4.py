"""Two things mechanism3 left open, both SHAP-free.

(A) IS THE MECHANISM CONCENTRATED OR DISTRIBUTED? mechanism3's mass-shift term recovered only
    28.7% of the gap at its best stratification, i.e. the coarse strata do NOT carry it. If
    REFINING the strata monotonically drives the mass-shift term toward 100%, the mechanism is
    a mass-shift after all and the coarse strata were simply too coarse. If it does not, the
    Tcond gap is genuinely distributed re-pricing and no single "graphite puts N% more mass on
    X" sentence can carry it -- which would be a finding, and one that CONTRADICTS the shape of
    SHAPDIFF-1's bottom-row answer.
    The limit is exact and checkable: stratifying by the CELL recovers the gap's mass-shift part
    exactly when priced at F, so the sequence has a known endpoint.

(B) THE REDIRECT FAMILY, since the frame's `redirect` column attributes AGAINST the direction
    of its own corpus mass share. Measured per class, with the same-hand gate made explicit.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import subprocess, pathlib, sys
import numpy as np, keybo
print("keybo.__file__ =", keybo.__file__)
root = pathlib.Path(keybo.__file__).resolve().parents[2]
print("checkout =", root, "branch =", subprocess.run(["git","-C",str(root),"rev-parse","--abbrev-ref","HEAD"],capture_output=True,text=True).stdout.strip())

from keybo.analysis.timecard import default_surface
from keybo.analysis.shap_diff import _char_weight_tables
from keybo.features import trigram_features_from_positions
from keybo.features.schema import TRIGRAM_FEATURE_NAMES as TN
from keybo.verdicts import require_finite

F="pyou'vgdnmheai.cstrlkjz,-wfbxq"; G="bldwz'foujnrtsgyhaeixqmcvkp,.-"
corpus = sys.argv[1] if len(sys.argv)>1 else None
s = default_surface(90.0, corpus); g = s.geometry
pos=[*g.slots,g.space_position]; n=len(pos); Tc=s._Tc
X = np.vstack([trigram_features_from_positions(g,(a,b,c),wpm=90.0) for a in pos for b in pos for c in pos])
for k in ("bg1_distance","bg2_distance","redirect","bad_redirect","same_hand_trigram"):
    assert k in TN, f"KEY MISSING: {k}"
w3,_,cov = _char_weight_tables(s,F); wn=w3/cov
sa,sb=s._slot_of(F),s._slot_of(G)
pa=np.array([sa[c] for c in F]+[sa[" "]]); pb=np.array([sb[c] for c in F]+[sb[" "]])
require_finite([float(cov)],"covered mass")
def cv(c): return X[:,TN.index(c)].reshape(n,n,n)
TcF=Tc[np.ix_(pa,pa,pa)]; TcG=Tc[np.ix_(pb,pb,pb)]
gap=float((wn*TcG).sum()-(wn*TcF).sum())
print(f"\ncorpus={corpus or 'blend-v1(default)'}  gap_Tcond={gap:+.6f}")

def mass_shift(codes, k):
    """Exact shift-share on an integer stratum code array. Returns (mass_term, within_term)."""
    cF = codes[np.ix_(pa,pa,pa)].ravel(); cG = codes[np.ix_(pb,pb,pb)].ravel()
    wf = wn.ravel()
    mF = np.bincount(cF, weights=wf, minlength=k); mG = np.bincount(cG, weights=wf, minlength=k)
    sF = np.bincount(cF, weights=wf*TcF.ravel(), minlength=k)
    sG = np.bincount(cG, weights=wf*TcG.ravel(), minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        pF = np.where(mF>0, sF/np.where(mF>0,mF,1), 0.0)
        pG = np.where(mG>0, sG/np.where(mG>0,mG,1), 0.0)
    return float(((mG-mF)*pF).sum()), float((mG*(pG-pF)).sum())

# --- (A) refinement ladder -------------------------------------------------------------
print("\n=== (A) does the MASS-SHIFT share grow as strata refine? ===")
row_of = np.array([p[1] for p in pos])                    # 0=space,1=bottom,2=home,3=top
fin = np.array([min(abs(p[0]),5) for p in pos])            # coarse finger id 0..5
i_idx = np.arange(n)[:,None,None]*np.ones((1,n,n),dtype=int)
j_idx = np.ones((n,1,n),dtype=int)*np.arange(n)[None,:,None]
k_idx = np.ones((n,n,1),dtype=int)*np.arange(n)[None,None,:]
travel = cv("bg1_distance")+cv("bg2_distance")
tbin = np.digitize(travel, [4,6,8,10,12])                  # 6 bins
ladder = [
    ("3rd-key row (4)",                 row_of[k_idx], 4),
    ("mid-key row (4)",                 row_of[j_idx], 4),
    ("mid x 3rd row (16)",              row_of[j_idx]*4+row_of[k_idx], 16),
    ("all-three rows (64)",             row_of[i_idx]*16+row_of[j_idx]*4+row_of[k_idx], 64),
    ("3rd row x travel (24)",           row_of[k_idx]*6+tbin, 24),
    ("all rows x travel (384)",         (row_of[i_idx]*16+row_of[j_idx]*4+row_of[k_idx])*6+tbin, 384),
    ("3rd key IDENTITY (31)",           k_idx, n),
    ("mid x 3rd key identity (961)",    j_idx*n+k_idx, n*n),
    ("3rd key x finger-pair (~1116)",   (k_idx*6+fin[j_idx])*6+fin[i_idx], n*36),
    ("FULL CELL (29791) [limit]",       (i_idx*n+j_idx)*n+k_idx, n**3),
]
print(f"{'stratification':<32} {'strata':>7} {'mass-shift ms':>14} {'% of gap':>9} {'within ms':>11}")
for label, codes, k in ladder:
    m, w = mass_shift(np.ascontiguousarray(codes).astype(np.int64), k)
    assert abs(m+w-gap) < 1e-9, f"shift-share identity broke on {label}: resid {abs(m+w-gap):.3e}"
    print(f"{label:<32} {k:>7} {m:>+14.4f} {100*m/gap:>8.1f}% {w:>+11.4f}")

# --- (B) the redirect family ------------------------------------------------------------
print("\n=== (B) the redirect family, per class (frame's OWN columns) ===")
red=cv("redirect"); bad=cv("bad_redirect"); sht=cv("same_hand_trigram")
print("gate check: bad_redirect => redirect:", bool(((bad>0.5)<=(red>0.5)).all()),
      " redirect => same_hand_trigram:", bool(((red>0.5)<=(sht>0.5)).all()))
print(f"\n{'class':<26} {'mass_F%':>8} {'mass_G%':>8} {'G/F':>7} {'price_on':>9} {'price_off':>10} {'d_price':>8}")
for nm, m in (("same_hand_trigram", sht>0.5),
              ("  redirect (any)", red>0.5),
              ("    bad_redirect", bad>0.5),
              ("    redirect, not bad", (red>0.5)&(bad<0.5)),
              ("  same-hand, no redirect", (sht>0.5)&(red<0.5))):
    mF=float((wn*m[np.ix_(pa,pa,pa)]).sum()); mG=float((wn*m[np.ix_(pb,pb,pb)]).sum())
    on=float(Tc[m].mean()); off=float(Tc[~m].mean())
    print(f"{nm:<26} {100*mF:>8.4f} {100*mG:>8.4f} {(mG/mF if mF else float('nan')):>7.4f} {on:>9.3f} {off:>10.3f} {on-off:>+8.3f}")
print("\nBoth boards' redirect mass priced at the SHIPPED table, mass-weighted (own board):")
for nm, m in (("redirect (any)", red>0.5), ("bad_redirect", bad>0.5)):
    mF=float((wn*m[np.ix_(pa,pa,pa)]).sum()); mG=float((wn*m[np.ix_(pb,pb,pb)]).sum())
    pF=float((wn*m[np.ix_(pa,pa,pa)]*TcF).sum()/mF); pG=float((wn*m[np.ix_(pb,pb,pb)]*TcG).sum()/mG)
    print(f"  {nm:<16} F {100*mF:.4f}% @ {pF:.3f} ms   G {100*mG:.4f}% @ {pG:.3f} ms")
