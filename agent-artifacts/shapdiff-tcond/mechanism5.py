"""mechanism4's ladder was WRONG, and this fixes it. Plus the travel confound. No SHAP.

BUG IN mechanism4 (found by reading its own residual list, not by a bar): I priced the mass
shift at board F's own within-stratum price pF, and set pF := 0 for strata where F has no
mass. At fine strata that is most of them, so mG*Tc leaked out of the mass term into the
within term -- which is why the ladder overshot to 113% and then read -118.4% at the FULL-CELL
limit, where the true within term is EXACTLY 0 by construction (a position cell's price is
board-independent). A decomposition whose known endpoint comes out negative is broken.

THE FIX: price the shift at a BOARD-FREE reference p(S), the mass-POOLED price. For any such
p the decomposition is still exact --
    gap = SUM_S (mG-mF)*p(S)          <- pure COMPOSITION (mass moved onto pricier strata)
        + SUM_S [mG*(pG-p) - mF*(pF-p)]  <- WITHIN-stratum residual
-- and because p does not depend on either board's mass, an empty stratum contributes
mG*pG correctly instead of leaking. At the FULL-CELL limit the within term is then exactly 0
and the composition term is exactly the gap, which is the endpoint the ladder must hit.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
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
s=default_surface(90.0,corpus); g=s.geometry
pos=[*g.slots,g.space_position]; n=len(pos); Tc=s._Tc
X=np.vstack([trigram_features_from_positions(g,(a,b,c),wpm=90.0) for a in pos for b in pos for c in pos])
for k in ("bg1_distance","bg2_distance","bg1_same_finger","bg2_same_finger"): assert k in TN, f"KEY MISSING: {k}"
w3,_,cov=_char_weight_tables(s,F); wn=w3/cov
sa,sb=s._slot_of(F),s._slot_of(G)
pa=np.array([sa[c] for c in F]+[sa[" "]]); pb=np.array([sb[c] for c in F]+[sb[" "]])
require_finite([float(cov)],"covered mass")
def cv(c): return X[:,TN.index(c)].reshape(n,n,n)
TcF=Tc[np.ix_(pa,pa,pa)]; TcG=Tc[np.ix_(pb,pb,pb)]
gap=float((wn*TcG).sum()-(wn*TcF).sum())
print(f"\ncorpus={corpus or 'blend-v1(default)'}  gap_Tcond={gap:+.6f}")

def pooled_shift(codes,k):
    cF=codes[np.ix_(pa,pa,pa)].ravel(); cG=codes[np.ix_(pb,pb,pb)].ravel(); wf=wn.ravel()
    mF=np.bincount(cF,weights=wf,minlength=k); mG=np.bincount(cG,weights=wf,minlength=k)
    sF=np.bincount(cF,weights=wf*TcF.ravel(),minlength=k); sG=np.bincount(cG,weights=wf*TcG.ravel(),minlength=k)
    tot=mF+mG
    p=np.where(tot>0,(sF+sG)/np.where(tot>0,tot,1),0.0)      # BOARD-FREE pooled price
    comp=float(((mG-mF)*p).sum())
    within=float((sG-mG*p).sum()-(sF-mF*p).sum())
    return comp,within

row_of=np.array([q[1] for q in pos]); fin=np.array([min(abs(q[0]),5) for q in pos])
i_idx=np.arange(n)[:,None,None]*np.ones((1,n,n),dtype=int)
j_idx=np.ones((n,1,n),dtype=int)*np.arange(n)[None,:,None]
k_idx=np.ones((n,n,1),dtype=int)*np.arange(n)[None,None,:]
travel=cv("bg1_distance")+cv("bg2_distance"); tbin=np.digitize(travel,[4,6,8,10,12])
ladder=[("3rd-key row (4)",row_of[k_idx],4),
        ("mid-key row (4)",row_of[j_idx],4),
        ("first-key row (4)",row_of[i_idx],4),
        ("mid x 3rd row (16)",row_of[j_idx]*4+row_of[k_idx],16),
        ("all-three rows (64)",row_of[i_idx]*16+row_of[j_idx]*4+row_of[k_idx],64),
        ("travel only (6)",tbin,6),
        ("3rd row x travel (24)",row_of[k_idx]*6+tbin,24),
        ("all rows x travel (384)",(row_of[i_idx]*16+row_of[j_idx]*4+row_of[k_idx])*6+tbin,384),
        ("3rd key identity (31)",k_idx,n),
        ("mid x 3rd key (961)",j_idx*n+k_idx,n*n),
        ("FULL CELL (29791) [limit=100%]",(i_idx*n+j_idx)*n+k_idx,n**3)]
print("\n=== (A-fixed) COMPOSITION share vs stratum refinement, pooled board-free price ===")
print(f"{'stratification':<34} {'strata':>7} {'composition':>12} {'% of gap':>9} {'within':>10}")
for label,codes,k in ladder:
    c,w=pooled_shift(np.ascontiguousarray(codes).astype(np.int64),k)
    assert abs(c+w-gap)<1e-9, f"identity broke on {label}: {abs(c+w-gap):.3e}"
    print(f"{label:<34} {k:>7} {c:>+12.4f} {100*c/gap:>8.1f}% {w:>+10.4f}")

print("\n=== (B) the TRAVEL confound: is short travel cheap or is it same-finger? ===")
sf = (cv("bg1_same_finger")>0.5)|(cv("bg2_same_finger")>0.5)
print(f"{'travel bin':<14} {'cells':>7} {'mean ms ALL':>12} {'mean ms no-SF':>14} {'SF share of cells':>18}")
for lo,hi in ((0,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,100)):
    m=(travel>=lo)&(travel<hi)
    if not m.any(): continue
    nosf=m&~sf
    ms_all=float(Tc[m].mean()); ms_no=float(Tc[nosf].mean()) if nosf.any() else float("nan")
    print(f"{f'[{lo},{hi})':<14} {int(m.sum()):>7} {ms_all:>12.3f} {ms_no:>14.3f} {100*float(sf[m].mean()):>17.1f}%")
print("\nSAME-FINGER-FREE cells only: composition share by travel bin")
c,w = pooled_shift(np.ascontiguousarray(np.where(sf, 6+tbin, tbin)).astype(np.int64), 13)
print(f"  travel x same-finger (13 strata): composition {c:+.4f} ({100*c/gap:.1f}% of gap)  within {w:+.4f}")
mF=float((wn*travel[np.ix_(pa,pa,pa)]).sum()); mG=float((wn*travel[np.ix_(pb,pb,pb)]).sum())
print(f"\nmass-weighted mean TOTAL TRAVEL: F {mF:.4f}  G {mG:.4f}  (+{mG-mF:.4f} = {100*(mG-mF)/mF:+.2f}%)")
for nm,msk in (("all cells",np.ones_like(sf,dtype=bool)),("no same-finger",~sf)):
    lo=msk&(travel<=4); hi=msk&(travel>=10)
    print(f"  price[{nm}]  travel<=4 {float(Tc[lo].mean()):.3f} ms   travel>=10 {float(Tc[hi].mean()):.3f} ms   delta {float(Tc[hi].mean()-Tc[lo].mean()):+.3f}")
