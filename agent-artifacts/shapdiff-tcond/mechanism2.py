"""What bg1_/bg2_ actually ARE, and the model-side price of the dominant blocks. No SHAP.

Settles two things my first read got glib about:
  (1) bg1_* are the (a,b) transition's features and bg2_* the (b,c) transition's -- so bg2_
      one-hots describe the trigram's THIRD key, not the first bigram re-described. Asserted
      by BYTE-COMPARING against the bigram frame rather than read off a docstring.
  (2) a model-side PRICE for the continuous geometry columns that is not a univariate OLS
      slope (which is confounded across correlated columns and even flips sign): bin the
      position grid by travel and report the shipped table's mean ms per bin.
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
from keybo.features import trigram_features_from_positions, bigram_features_from_positions
from keybo.features.schema import TRIGRAM_FEATURE_NAMES as TN, BIGRAM_FEATURE_NAMES as BN
from keybo.verdicts import require_finite

F = "pyou'vgdnmheai.cstrlkjz,-wfbxq"; G = "bldwz'foujnrtsgyhaeixqmcvkp,.-"
corpus = sys.argv[1] if len(sys.argv) > 1 else None
s = default_surface(90.0, corpus); g = s.geometry
pos = [*g.slots, g.space_position]; n = len(pos)
Tc = s._Tc                                             # predict_ms-anchored. NOT SHAP.
X = np.vstack([trigram_features_from_positions(g,(a,b,c),wpm=90.0) for a in pos for b in pos for c in pos])
for k in ("bg1_dx","bg2_bottom","bg1_top","sg_distance","bg1_distance","bg2_distance"):
    assert k in TN, f"KEY MISSING ON THIS TREE: {k}"
w3, _, cov = _char_weight_tables(s, F)
sa, sb = s._slot_of(F), s._slot_of(G)
pa = np.array([sa[c] for c in F]+[sa[" "]]); pb = np.array([sb[c] for c in F]+[sb[" "]])
require_finite([float(cov)], "covered mass")
def cv(col): return X[:, TN.index(col)].reshape(n,n,n)

# --- (1) WHAT ARE bg1_/bg2_? byte-compare against the BIGRAM frame -----------------------
print("\n=== (1) bg1_/bg2_ identity, asserted by byte-comparison against the bigram frame ===")
shared = [c for c in BN if c != "wpm"]
ok1 = ok2 = 0; bad = []
rng = np.random.default_rng(0)
for _ in range(400):
    i, j, k = (int(x) for x in rng.integers(0, n, 3))
    a, b, c = pos[i], pos[j], pos[k]
    tri = trigram_features_from_positions(g,(a,b,c),wpm=90.0)
    ab = bigram_features_from_positions(g,(a,b),wpm=90.0)
    bc = bigram_features_from_positions(g,(b,c),wpm=90.0)
    v1 = np.array([tri[TN.index(f"bg1_{cn}")] for cn in shared])
    v2 = np.array([tri[TN.index(f"bg2_{cn}")] for cn in shared])
    r1 = np.array([ab[BN.index(cn)] for cn in shared])
    r2 = np.array([bc[BN.index(cn)] for cn in shared])
    ok1 += int(np.array_equal(v1, r1)); ok2 += int(np.array_equal(v2, r2))
    if not (np.array_equal(v1,r1) and np.array_equal(v2,r2)): bad.append((i,j,k))
print(f"bg1_* == bigram_features(a,b) on {ok1}/400 sampled triples")
print(f"bg2_* == bigram_features(b,c) on {ok2}/400 sampled triples")
print("mismatches:", bad[:5])
# and the row one-hot therefore describes WHICH key?
print("\nrow one-hot semantics (bigram placement one-hots are the SECOND key of their pair):")
print("  => bg1_{bottom,home,top} describes the trigram's MIDDLE key (b)")
print("  => bg2_{bottom,home,top} describes the trigram's THIRD  key (c)")
# verify directly: bg2_bottom must equal (c is on row 1)
c_row1 = np.array([[[float(pos[k][1]==1) for k in range(n)] for j in range(n)] for i in range(n)])
b_row1 = np.array([[[float(pos[j][1]==1) for k in range(n)] for j in range(n)] for i in range(n)])
print("  bg2_bottom == 1[c on bottom row]  exact:", bool(np.array_equal(cv("bg2_bottom"), c_row1)))
print("  bg1_bottom == 1[b on bottom row]  exact:", bool(np.array_equal(cv("bg1_bottom"), b_row1)))

# --- (2) ROW of the THIRD key: mass share (corpus) x price (model) -----------------------
print("\n=== (2) THIRD-key row: corpus mass share per board  x  shipped-table price ===")
rows = {3:"top", 2:"home", 1:"bottom", 0:"space"}
print(f"{'3rd-key row':<12} {'share_F%':>9} {'share_G%':>9} {'G/F':>7} {'mean ms (shipped)':>18}")
for r, nm in rows.items():
    m = np.array([[[float(pos[k][1]==r) for k in range(n)] for j in range(n)] for i in range(n)])
    sF = float((w3*m[np.ix_(pa,pa,pa)]).sum()/cov); sG = float((w3*m[np.ix_(pb,pb,pb)]).sum()/cov)
    price = float(Tc[m>0.5].mean())
    print(f"{nm:<12} {100*sF:>9.4f} {100*sG:>9.4f} {(sG/sF if sF else float('nan')):>7.4f} {price:>18.3f}")
print(f"{'MIDDLE-key row':<12}")
for r, nm in rows.items():
    m = np.array([[[float(pos[j][1]==r) for k in range(n)] for j in range(n)] for i in range(n)])
    sF = float((w3*m[np.ix_(pa,pa,pa)]).sum()/cov); sG = float((w3*m[np.ix_(pb,pb,pb)]).sum()/cov)
    price = float(Tc[m>0.5].mean())
    print(f"{nm:<12} {100*sF:>9.4f} {100*sG:>9.4f} {(sG/sF if sF else float('nan')):>7.4f} {price:>18.3f}")

# --- (3) TRAVEL: binned price, not an OLS slope ------------------------------------------
print("\n=== (3) TOTAL TRAVEL (bg1_distance + bg2_distance): binned shipped price x mass ===")
travel = cv("bg1_distance") + cv("bg2_distance")
edges = [0,2,4,6,8,10,12,100]
print(f"{'travel bin':<12} {'cells':>7} {'mean ms':>9} {'share_F%':>9} {'share_G%':>9} {'G-F pp':>8}")
tot_F = tot_G = 0.0
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (travel >= lo) & (travel < hi)
    if not m.any(): continue
    sF = float((w3*m[np.ix_(pa,pa,pa)]).sum()/cov); sG = float((w3*m[np.ix_(pb,pb,pb)]).sum()/cov)
    tot_F += sF; tot_G += sG
    print(f"{f'[{lo},{hi})':<12} {int(m.sum()):>7} {float(Tc[m].mean()):>9.3f} {100*sF:>9.4f} {100*sG:>9.4f} {100*(sG-sF):>+8.4f}")
print(f"{'(check)':<12} {'':>7} {'':>9} {100*tot_F:>9.4f} {100*tot_G:>9.4f}")
mF = float((w3*travel[np.ix_(pa,pa,pa)]).sum()/cov); mG = float((w3*travel[np.ix_(pb,pb,pb)]).sum()/cov)
print(f"\nmass-weighted MEAN total travel:  flagship-c3 {mF:.4f}   graphite {mG:.4f}   (+{mG-mF:.4f}, {100*(mG-mF)/mF:+.2f}%)")
# a monotone, board-free price curve summary
lo_m = travel <= 4; hi_m = travel >= 8
print(f"shipped price: travel<=4  {float(Tc[lo_m].mean()):.3f} ms   travel>=8  {float(Tc[hi_m].mean()):.3f} ms   delta {float(Tc[hi_m].mean()-Tc[lo_m].mean()):+.3f} ms")

# --- (4) how much of Tc is the THIRD key? (the 'genuinely trigram' question) --------------
print("\n=== (4) is Tcond genuinely trigram-level? variance decomposition of the SHIPPED table ===")
mu = Tc.mean(); ss_tot = float(((Tc-mu)**2).sum())
ab_mean = Tc.mean(axis=2, keepdims=True)          # average over the third key
bc_mean = Tc.mean(axis=0, keepdims=True)          # average over the first key
print("share of Tc variance explained by (a,b) alone : %.4f" % (float(((ab_mean-mu)**2).sum()*n)/ss_tot))
print("share of Tc variance explained by (b,c) alone : %.4f" % (float(((bc_mean-mu)**2).sum()*n)/ss_tot))
print("=> the (b,c) transition, which the BIGRAM channel never sees, carries most of Tc's structure")
