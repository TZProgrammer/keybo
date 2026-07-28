"""AUDIT claim 2: is my additive 'share of win' decomposition order-dependent / invalid under
collinearity? Test directly.

My method (state it plainly): the evidence objective is a SUM of 14 independent univariate
curves, price_m(x_m). I computed delta_m = price_m(champ) - price_m(reference) per gauge. That
is EXACT and order-independent BY CONSTRUCTION: sum_m delta_m == total delta, no residual, no
ordering choice. Collinearity affects the FIT (which gauge got which coefficient) but NOT the
arithmetic of decomposing a sum whose terms are already separated.

The real vulnerabilities are different, and I test all three:
  (i)  exactness: does sum(delta_m) equal the total to float precision?
  (ii) SIGNED offsets: shares of a NET quantity can exceed 100% when terms cancel.
  (iii) causal misattribution: because gauges are collinear, you cannot move one and hold the
       others fixed, so a per-gauge share is NOT a counterfactual. Test with a Shapley value
       over gauge GROUPS (correlation clusters) using a real ablation: re-SEARCH is the
       sibling's job, so instead ablate by re-scoring the champion with gauge m's price frozen
       at the reference level, which is the honest static analogue.
"""
import sys, json, itertools
from pathlib import Path
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
import numpy as np, evobj as EV
from keybo.analysis.evidence_scorer import LIVE_GAUGES
S=Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARMJ="/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
J=json.load(open(S/"judgement.json")); pl=J["per_layout"]
fe=EV.FastEval(None,ARMJ,with_surface=True)
A="champ-evidence"; R="archive-1846"
perms=np.stack([EV.perm_of(pl[A]["layout"]),EV.perm_of(pl[R]["layout"])])
g=fe.gauges(perms)
price={c.metric:(float(c.price(g[c.metric])[0]),float(c.price(g[c.metric])[1])) for c in fe.curves}
tot_A=sum(v[0] for v in price.values()); tot_R=sum(v[1] for v in price.values())
d={m:price[m][0]-price[m][1] for m in LIVE_GAUGES}
print("=== (i) EXACTNESS of the additive decomposition ===")
print("  sum of per-gauge deltas = %.12f"%sum(d.values()))
print("  total(A) - total(R)     = %.12f"%(tot_A-tot_R))
print("  residual                = %.3e  -> %s"%(abs(sum(d.values())-(tot_A-tot_R)),
      "EXACT, no residual, order-independent" if abs(sum(d.values())-(tot_A-tot_R))<1e-9 else "NOT EXACT"))
print("  (the objective is a SUM of 14 univariate curves, so this is an identity, not a model)")
gross_imp=sum(v for v in d.values() if v<0); off=sum(v for v in d.values() if v>0); net=sum(d.values())
print("\n=== (ii) SIGNED-share inflation: shares of a NET quantity ===")
print("  gross improvement %.4f | offsetting %+.4f | net %.4f"%(gross_imp,off,net))
print("  comfort+sr-roll as %% of NET   = %.1f%%   <- what I reported"%(100*(d['comfort']+d['sr-roll'])/net))
print("  comfort+sr-roll as %% of GROSS = %.1f%%   <- the non-inflated denominator"%(100*(d['comfort']+d['sr-roll'])/gross_imp))
print("  offsets are %.1f%% of net, so the NET denominator inflates every share by %.3fx"%(100*off/net, gross_imp/net))
# ---- (iii) Shapley over correlation clusters, with a REAL static ablation ----
W=json.load(open(ARMJ))["weights"]
clusters={k:(v["members"] if isinstance(v,dict) and "members" in v else v) for k,v in W["clusters"].items()}
print("\n=== (iii) SHAPLEY value over the %d correlation CLUSTERS (order-independent by definition) ==="%len(clusters))
print("  clusters:", {k:len(v) for k,v in clusters.items()})
keys=list(clusters)
def val(subset):
    """value = improvement obtained when only clusters in `subset` take the champion's levels,
    all others frozen at the reference's levels."""
    t=0.0
    for k in keys:
        for m in clusters[k]:
            t += price[m][0] if k in subset else price[m][1]
    return t - tot_R
n=len(keys); shap={k:0.0 for k in keys}
import math
fact=math.factorial
for k in keys:
    others=[x for x in keys if x!=k]
    for r in range(len(others)+1):
        w=fact(r)*fact(n-r-1)/fact(n)
        for comb in itertools.combinations(others,r):
            shap[k]+= w*(val(set(comb)|{k})-val(set(comb)))
print("  sum of Shapley values = %.12f  (must equal net %.12f) residual %.3e"%(sum(shap.values()),net,abs(sum(shap.values())-net)))
print("\n  %-22s %12s %12s %10s"%("cluster","shapley","additive","identical?"))
add={k:sum(d[m] for m in clusters[k]) for k in keys}
for k in sorted(keys,key=lambda k:shap[k]):
    print("  %-22s %12.6f %12.6f %10s"%(k,shap[k],add[k],"YES" if abs(shap[k]-add[k])<1e-9 else "NO"))
print("\n  VERDICT: for an ADDITIVE objective the Shapley value EQUALS the additive delta exactly,")
print("  so the decomposition is NOT order-dependent. Collinearity is a FIT problem, not a")
print("  decomposition problem — but see (iii-b).")
print("\n=== (iii-b) what the decomposition CANNOT tell you ===")
print("  The share answers: 'of the score difference, how much is booked to gauge m?'")
print("  It does NOT answer: 'if gauge m were removed from the objective, how much of the")
print("  ms/char deficit would go away?' — because gauges are collinear (VIF 12.8-119), you")
print("  cannot move one gauge and hold the rest fixed on a real layout. That counterfactual")
print("  needs a RE-SEARCH per ablation (arm-D-shaped), which is the sibling's job.")
json.dump({"per_gauge_delta":d,"tot_A":tot_A,"tot_R":tot_R,"net":net,"gross":gross_imp,"offset":off,
           "pct_of_net":100*(d['comfort']+d['sr-roll'])/net,
           "pct_of_gross":100*(d['comfort']+d['sr-roll'])/gross_imp,
           "inflation_factor":gross_imp/net,
           "shapley_clusters":shap,"additive_clusters":add,
           "shapley_equals_additive":all(abs(shap[k]-add[k])<1e-9 for k in keys)},
          open(S/"decomposition-audit.json","w"),indent=1)
print("\nWROTE decomposition-audit.json")
