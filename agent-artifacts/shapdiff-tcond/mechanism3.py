"""The gap itself, decomposed WITHOUT SHAP -- an independent check on the block story.

The SHAP block table says the Tcond gap is ~89% the two constituent-bigram blocks, with the
(b,c) block (BG2) largest. This script tests that claim with machinery SHAP never touches:
exact shift-share (Oaxaca) accounting on the SHIPPED _Tc table.

For any partition of the 31^3 cells into strata S, the gap decomposes EXACTLY as

    gap = sum_S [ (m_G(S) - m_F(S)) * price(S) ]  +  sum_S [ m_F(S) * (price_G(S) - price_F(S)) ]
          \_________ MASS-SHIFT term __________/     \______ WITHIN-STRATUM term ______/

where m_X(S) is board X's corpus mass share in S and price_X(S) its mass-weighted mean ms.
The MASS-SHIFT term is a pure "the board moved corpus mass onto pricier cells" statement --
no model attribution, no Shapley value, no LMDI. If stratifying by the THIRD KEY'S ROW
recovers a large share of +2.1953, the SHAP story is corroborated by independent means.
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

F = "pyou'vgdnmheai.cstrlkjz,-wfbxq"; G = "bldwz'foujnrtsgyhaeixqmcvkp,.-"
corpus = sys.argv[1] if len(sys.argv) > 1 else None
s = default_surface(90.0, corpus); g = s.geometry
pos = [*g.slots, g.space_position]; n = len(pos)
Tc = s._Tc
X = np.vstack([trigram_features_from_positions(g,(a,b,c),wpm=90.0) for a in pos for b in pos for c in pos])
for k in ("bg1_distance","bg2_distance","redirect","sg_distance"): assert k in TN, f"KEY MISSING: {k}"
w3, _, cov = _char_weight_tables(s, F)
sa, sb = s._slot_of(F), s._slot_of(G)
pa = np.array([sa[c] for c in F]+[sa[" "]]); pb = np.array([sb[c] for c in F]+[sb[" "]])
require_finite([float(cov)], "covered mass")
def cv(col): return X[:, TN.index(col)].reshape(n,n,n)

TcF = Tc[np.ix_(pa,pa,pa)]; TcG = Tc[np.ix_(pb,pb,pb)]
wn  = w3/cov
gap = float((wn*TcG).sum() - (wn*TcF).sum())
print(f"\ncorpus={corpus or 'blend-v1(default)'}   gap_Tcond (shipped table, no SHAP) = {gap:+.6f} ms/char")

def shift_share(label, strata):
    """strata: list of (name, bool mask over (n,n,n) POSITION cells)."""
    mass_term = within_term = 0.0; rows=[]
    for name, m in strata:
        mF = float((wn*m[np.ix_(pa,pa,pa)]).sum()); mG = float((wn*m[np.ix_(pb,pb,pb)]).sum())
        # per-board price INSIDE the stratum (mass-weighted mean ms), each on its own board
        pF = float((wn*m[np.ix_(pa,pa,pa)]*TcF).sum()/mF) if mF>0 else 0.0
        pG = float((wn*m[np.ix_(pb,pb,pb)]*TcG).sum()/mG) if mG>0 else 0.0
        mass_term += (mG-mF)*pF          # mass shift priced at F's own price
        within_term += mG*(pG-pF)        # residual: price change inside the stratum
        rows.append((name, mF, mG, pF, pG))
    print(f"\n--- stratify by {label} ---")
    print(f"{'stratum':<16} {'mass_F%':>8} {'mass_G%':>8} {'d_mass pp':>10} {'price_F':>9} {'price_G':>9} {'mass-shift ms':>14}")
    for name,mF,mG,pF,pG in rows:
        print(f"{name:<16} {100*mF:>8.4f} {100*mG:>8.4f} {100*(mG-mF):>+10.4f} {pF:>9.3f} {pG:>9.3f} {(mG-mF)*pF:>+14.4f}")
    print(f"{'MASS-SHIFT total':<16} {'':>8} {'':>8} {'':>10} {'':>9} {'':>9} {mass_term:>+14.4f}   ({100*mass_term/gap:.1f}% of the gap)")
    print(f"{'WITHIN total':<16} {'':>8} {'':>8} {'':>10} {'':>9} {'':>9} {within_term:>+14.4f}   ({100*within_term/gap:.1f}%)")
    print(f"{'SUM (must == gap)':<16} {'':>8} {'':>8} {'':>10} {'':>9} {'':>9} {mass_term+within_term:>+14.4f}   resid {abs(mass_term+within_term-gap):.3e}")
    return mass_term, within_term

# 1) by the THIRD key's row -- what BG2's `row` sub-block claims
third_row = [(nm, np.array([[[float(pos[k][1]==r) for k in range(n)] for j in range(n)] for i in range(n)]))
             for r,nm in ((3,"3rd=top"),(2,"3rd=home"),(1,"3rd=bottom"),(0,"3rd=space"))]
shift_share("THIRD key's ROW (BG2 row sub-block)", third_row)

# 2) by the MIDDLE key's row -- what BG1's `row` sub-block claims
mid_row = [(nm, np.array([[[float(pos[j][1]==r) for k in range(n)] for j in range(n)] for i in range(n)]))
           for r,nm in ((3,"mid=top"),(2,"mid=home"),(1,"mid=bottom"),(0,"mid=space"))]
shift_share("MIDDLE key's ROW (BG1 row sub-block)", mid_row)

# 3) by TOTAL TRAVEL -- what the two geometry sub-blocks claim
travel = cv("bg1_distance") + cv("bg2_distance")
edges = [0,4,6,8,10,12,100]
trav = [(f"travel[{lo},{hi})", (travel>=lo)&(travel<hi)) for lo,hi in zip(edges[:-1],edges[1:])]
shift_share("TOTAL TRAVEL (geometry sub-blocks)", trav)

# 4) by the frame's own redirect column -- the DIRECT gauge test
red = cv("redirect"); bad = cv("bad_redirect")
rstrata = [("bad_redirect", bad>0.5), ("redirect_only", (red>0.5)&(bad<0.5)), ("no_redirect", red<0.5)]
shift_share("the FRAME's redirect column", rstrata)

# 5) JOINT: third-key row x travel, the two dominant claims together
joint = []
for r,nm in ((3,"top"),(2,"home"),(1,"bot"),(0,"spc")):
    rm = np.array([[[float(pos[k][1]==r) for k in range(n)] for j in range(n)] for i in range(n)])
    for lo,hi in ((0,8),(8,100)):
        joint.append((f"3rd={nm},t[{lo},{hi})", (rm>0.5)&(travel>=lo)&(travel<hi)))
shift_share("THIRD-key row x TRAVEL (joint)", joint)
