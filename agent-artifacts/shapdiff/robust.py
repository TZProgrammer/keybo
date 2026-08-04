"""Registered SHAPDIFF-1 §4 rule: corpus-robust iff (i) top-5 signs agree AND (ii) rho>=0.90.
Plus the §5 gauge cross-check E1-E4, and a same_finger/lsb deep dive."""
import os, json, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="2"
import keybo, numpy as np
from scipy.stats import spearmanr
print("keybo.__file__:",keybo.__file__)
A=os.environ["A"]
b=json.load(open(f"{A}/shapdiff_blend-v1.json")); w=json.load(open(f"{A}/shapdiff_iweb.json"))
# ASSERT the keys exist (rc=0 + all-None is a key-not-present bug)
for tag,d in (("blend",b),("iweb",w)):
    assert d["residuals"]["reconciles"] is True, f"{tag} did not reconcile"
    assert len(d["contributions"])==20, f"{tag} has {len(d['contributions'])} features"
B={c["feature"]:c["ms_per_char"] for c in b["contributions"]}
W={c["feature"]:c["ms_per_char"] for c in w["contributions"]}
names=list(B)
print(f"\ngap_total blend {b['gap']['total']:+.4f}  iweb {w['gap']['total']:+.4f}")
print(f"gap_t2    blend {b['gap']['t2_bigram_channel']:+.4f} ({b['gap']['decomposed_share_pct']:.1f}%)"
      f"  iweb {w['gap']['t2_bigram_channel']:+.4f} ({w['gap']['decomposed_share_pct']:.1f}%)")
print(f"gap_tcond blend {b['gap']['tcond_trigram_channel']:+.4f}  iweb {w['gap']['tcond_trigram_channel']:+.4f}")
# (i) top-5 sign agreement
top5b=[c["feature"] for c in b["contributions"]][:5]; top5w=[c["feature"] for c in w["contributions"]][:5]
print(f"\ntop5 blend: {top5b}\ntop5 iweb : {top5w}")
sign_ok=all(np.sign(B[f])==np.sign(W[f]) for f in top5b)
flips=[f for f in names if np.sign(B[f])!=np.sign(W[f])]
print(f"(i) top-5 sign agreement (on blend's top5): {sign_ok}")
print(f"    ALL features that FLIP sign across corpora: {flips}")
for f in flips: print(f"      {f}: blend {B[f]:+.5f}  iweb {W[f]:+.5f}")
# (ii) spearman over signed contributions
rho=spearmanr([B[f] for f in names],[W[f] for f in names]).statistic
print(f"(ii) Spearman rho over all 20 signed contributions: {rho:.4f}  (bar >= 0.90) -> {rho>=0.90}")
print(f"VERDICT corpus-robust = {bool(sign_ok and rho>=0.90)}")
# side-by-side
print(f"\n{'feature':<14}{'blend ms':>11}{'iweb ms':>11}{'blend%':>9}{'iweb%':>9}")
for f in names:
    print(f"{f:<14}{B[f]:>+11.4f}{W[f]:>+11.4f}"
          f"{100*B[f]/b['gap']['t2_bigram_channel']:>8.1f}%{100*W[f]/w['gap']['t2_bigram_channel']:>8.1f}%")
# §5 gauge cross-check
print("\n=== §5 GAUGE CROSS-CHECK (registered expectations) ===")
gauges={"sfb":("same_finger",1.654,1.526),"lsb":("lsb",0.797,0.559),"scissor":("scissor",0.089,0.517)}
for g,(feat,fv,gv) in gauges.items():
    better="flagship" if fv<gv else "graphite"
    for tag,D in (("blend",B),("iweb",W)):
        fav="flagship" if D[feat]>0 else ("graphite" if D[feat]<0 else "tie")
        verdict="AGREE" if fav.startswith(better[:4]) else "DISAGREE"
        print(f"  {g:<8} gauge favours {better:<9} | SHAP {feat:<12} {tag:<6} {D[feat]:+.5f} favours {fav:<9} {verdict}")
