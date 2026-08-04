"""MECHANISM for the dominant Tcond blocks, measured INDEPENDENTLY of SHAP.

Two halves per claim, from separate machinery and neither computed from the other:
  (a) a CORPUS-SIDE share  -- what fraction of trigram mass has the property, per board;
  (b) a MODEL-SIDE price   -- what the SHIPPED _Tc table charges for it (predict_ms path,
      NOT pred_contribs), as a mean over cells with/without the property.
Nothing here reads a SHAP number or the shap_diff module's attribution.
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
from keybo.features.schema import TRIGRAM_FEATURE_NAMES as N
from keybo.verdicts import require_finite

F = "pyou'vgdnmheai.cstrlkjz,-wfbxq"; G = "bldwz'foujnrtsgyhaeixqmcvkp,.-"
corpus = sys.argv[1] if len(sys.argv) > 1 else None
s = default_surface(90.0, corpus)
g = s.geometry
pos = [*g.slots, g.space_position]
n = len(pos)
# The FEATURE table over position triples (geometry only, no model) and the SHIPPED ms table.
X = np.vstack([trigram_features_from_positions(g,(a,b,c),wpm=90.0) for a in pos for b in pos for c in pos])
Tc = s._Tc                                    # predict_ms-anchored, NOT SHAP
assert Tc.shape == (n,n,n)
for k in ("bg1_dx","bg2_bottom","bg1_top","sg_distance","redirect","bad_redirect","sg_dx","bg1_distance"):
    assert k in N, f"KEY MISSING ON THIS TREE: {k}"    # key-presence assert (rc=0+all-None guard)

w3, _, cov = _char_weight_tables(s, F)
sa, sb = s._slot_of(F), s._slot_of(G)
pa = np.array([sa[c] for c in F]+[sa[" "]]); pb = np.array([sb[c] for c in F]+[sb[" "]])
require_finite([float(cov)], "covered mass")

def cellview(col):
    """The (n,n,n) view of one feature column over position triples."""
    return X[:, N.index(col)].reshape(n,n,n)

def mass_share(col, perm, thresh=0.5):
    """(a) CORPUS SIDE: share of this board's trigram mass whose cell has feature>thresh."""
    v = cellview(col)[np.ix_(perm,perm,perm)]
    return float((w3 * (v > thresh)).sum() / cov)

def mass_mean(col, perm):
    """(a') CORPUS SIDE for a CONTINUOUS column: mass-weighted mean of the feature value."""
    v = cellview(col)[np.ix_(perm,perm,perm)]
    return float((w3 * v).sum() / cov)

def price_onoff(col, thresh=0.5):
    """(b) MODEL SIDE: shipped-table mean ms on cells WITH vs WITHOUT the property.

    Averaged over the 31^3 POSITION grid, unweighted by corpus and unweighted by board, so it
    is a property of the fitted surface alone -- it cannot inherit either board's mass profile.
    """
    v = cellview(col); on = v > thresh
    return float(Tc[on].mean()), float(Tc[~on].mean()), int(on.sum())

def price_slope(col):
    """(b') MODEL SIDE for a CONTINUOUS column: OLS slope of shipped ms on the column, over
    the whole position grid. Again board-independent and SHAP-independent."""
    v = cellview(col).ravel(); y = Tc.ravel()
    A = np.vstack([v, np.ones_like(v)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)

print("\ncorpus =", corpus or "blend-v1(default)", " covered mass =", cov)
print("\n=== BINARY columns: corpus share (a) x model price (b) ===")
print(f"{'column':<20} {'share_F%':>9} {'share_G%':>9} {'ratio G/F':>10} {'ms_on':>9} {'ms_off':>9} {'delta_ms':>9} {'cells_on':>9}")
for col in ("bg1_bottom","bg2_bottom","bg1_top","bg2_top","bg1_home","bg2_home",
            "bg1_index","bg1_pinky","bg1_middle","bg2_lateral","bg2_ring","bg2_pinky",
            "redirect","bad_redirect","same_hand_trigram","sg_same_finger",
            "bg1_same_finger","bg2_same_finger","bg1_scissor","bg2_scissor","bg2_adjacent"):
    sF, sG = mass_share(col,pa), mass_share(col,pb)
    on, off, ncell = price_onoff(col)
    ratio = sG/sF if sF > 0 else float("nan")
    print(f"{col:<20} {100*sF:>9.4f} {100*sG:>9.4f} {ratio:>10.4f} {on:>9.3f} {off:>9.3f} {on-off:>+9.3f} {ncell:>9}")

print("\n=== CONTINUOUS columns: corpus mean (a') x model slope (b') ===")
print(f"{'column':<20} {'mean_F':>9} {'mean_G':>9} {'G-F':>9} {'slope ms/unit':>14} {'pred delta ms':>14}")
for col in ("bg1_dx","bg2_dx","bg1_dy","bg2_dy","bg1_distance","bg2_distance",
            "bg1_angle","bg2_angle","sg_dx","sg_dy","sg_distance"):
    mF, mG = mass_mean(col,pa), mass_mean(col,pb)
    slope, _ = price_slope(col)
    print(f"{col:<20} {mF:>9.4f} {mG:>9.4f} {mG-mF:>+9.4f} {slope:>14.4f} {slope*(mG-mF):>+14.4f}")

# --- the BG1-vs-BG2 asymmetry: is the trigram model just re-pricing bigrams? -------------
# Independent of SHAP: correlate the shipped Tc table against the shipped T2 table over the
# same position grid, and check how much of Tc's variance the (a,b) pair alone explains.
print("\n=== is Tcond a genuinely TRIGRAM-level quantity? (no SHAP) ===")
T2 = s._T2
tc_flat = Tc.reshape(n*n, n)
by_ab = tc_flat.mean(axis=1)                       # Tc averaged over the third key
ss_tot = float(((Tc - Tc.mean())**2).sum())
ss_ab  = float((((by_ab[:,None] - Tc.mean())*np.ones((1,n)))**2).sum())
print("var share of Tc explained by the (a,b) pair alone: %.4f" % (ss_ab/ss_tot))
print("corr(Tc.mean_over_c, T2) over position pairs: %.4f" % float(np.corrcoef(by_ab, T2.ravel())[0,1]))
# how much does the THIRD key move Tc, per (a,b)?
spread = tc_flat.max(axis=1) - tc_flat.min(axis=1)
print("Tc spread over the third key: mean %.3f ms  median %.3f  max %.3f" % (spread.mean(), np.median(spread), spread.max()))
print("Tc overall: mean %.3f  sd %.3f" % (Tc.mean(), Tc.std()))
