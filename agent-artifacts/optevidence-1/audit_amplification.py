"""AUDIT claim 3: 'sr-roll delivers 39.2% of the win from 4.90% of attribution = 8x
amplification'. Amplification relative to WHAT, and is 8x a property of sr-roll or of hinge
geometry generally?

Baseline for the ratio: shap_share_pct is mean|SHAP| over the FITTING POOL (400 random
permutations), normalized to 100. So the ratio is (share of THIS win) / (share of attribution
ON A DIFFERENT POOL). It is a cross-pool ratio — that is the thing to scrutinize.

Test: for every gauge, regress its realized win-share on candidate geometric predictors:
  - |far slope| (slope beyond the knot)
  - distance travelled outside valid_domain
  - |far slope| x distance outside  (the mechanical product = extrapolated price)
If the product explains the amplification across ALL gauges, 8x is HINGE GEOMETRY. If sr-roll
is an outlier even after conditioning on it, it is sr-roll-specific.
"""
import sys, json
from pathlib import Path
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
import numpy as np, evobj as EV
import scipy.stats as st
from keybo.analysis.evidence_scorer import LIVE_GAUGES
S=Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARMJ="/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
J=json.load(open(S/"judgement.json")); pl=J["per_layout"]
W=json.load(open(ARMJ))["weights"]; shap={x["metric"]:x["shap_share_pct"] for x in W["weights"]}
fe=EV.FastEval(None,ARMJ,with_surface=True)
A="champ-evidence"; R="archive-1846"
perms=np.stack([EV.perm_of(pl[A]["layout"]),EV.perm_of(pl[R]["layout"])]); g=fe.gauges(perms)
rows=[]
for c in fe.curves:
    m=c.metric
    xa,xr=float(g[m][0]),float(g[m][1])
    pa,pr=float(c.price(np.array([xa]))[0]),float(c.price(np.array([xr]))[0])
    d=pa-pr
    far = c.coeffs[1]+c.coeffs[2] if c.form=="hinge" else (c.coeffs[1] if c.form=="linear" else np.nan)
    near= c.coeffs[1]
    out_a = max(c.domain[0]-xa, xa-c.domain[1], 0.0)
    out_r = max(c.domain[0]-xr, xr-c.domain[1], 0.0)
    # price attributable purely to being outside the domain (champ vs ref)
    clamp_a=float(c.price(np.array([np.clip(xa,*c.domain)]))[0])
    clamp_r=float(c.price(np.array([np.clip(xr,*c.domain)]))[0])
    d_clamped = clamp_a-clamp_r
    d_extrap  = d - d_clamped
    rows.append(dict(metric=m,form=c.form,delta=d,shap=shap[m],near=near,far=far,
                     out_a=out_a,out_r=out_r,d_clamped=d_clamped,d_extrap=d_extrap,
                     level=xa,dom=list(c.domain)))
net=sum(r["delta"] for r in rows)
for r in rows: r["pct_net"]=100*r["delta"]/net; r["amp"]=r["pct_net"]/r["shap"]
print("=== claim 3: decomposing each gauge's delta into IN-DOMAIN vs EXTRAPOLATION parts ===")
print("%-11s %8s %8s %9s %9s %9s %9s %8s"%("gauge","delta","%net","amp","in-dom","EXTRAP","%extrap","out_by"))
for r in sorted(rows,key=lambda r:r["delta"]):
    pe = 100*r["d_extrap"]/r["delta"] if abs(r["delta"])>1e-12 else 0.0
    print("%-11s %+8.4f %+8.1f %9.2f %+9.4f %+9.4f %8.1f %8.4f"%(
        r["metric"],r["delta"],r["pct_net"],r["amp"],r["d_clamped"],r["d_extrap"],pe,r["out_a"]))
tot_ex=sum(r["d_extrap"] for r in rows); tot_in=sum(r["d_clamped"] for r in rows)
print("\nTOTAL: in-domain %+.4f | EXTRAPOLATION %+.4f | net %+.4f"%(tot_in,tot_ex,net))
print("  -> extrapolation supplies %.1f%% of arm A's score advantage over %s"%(100*tot_ex/net,R))
print("  -> comfort's extrapolation part %+.4f (%.1f%% of net); sr-roll's %+.4f (%.1f%%)"%(
    [r for r in rows if r['metric']=='comfort'][0]['d_extrap'],
    100*[r for r in rows if r['metric']=='comfort'][0]['d_extrap']/net,
    [r for r in rows if r['metric']=='sr-roll'][0]['d_extrap'],
    100*[r for r in rows if r['metric']=='sr-roll'][0]['d_extrap']/net))
print("\n=== is amplification predicted by HINGE GEOMETRY across all 14 gauges? ===")
amp=np.array([r["amp"] for r in rows]); 
pred={"|far slope|":np.array([abs(r["far"]) for r in rows]),
      "distance outside domain":np.array([r["out_a"] for r in rows]),
      "|far slope| x distance":np.array([abs(r["far"])*r["out_a"] for r in rows]),
      "shap share":np.array([r["shap"] for r in rows])}
for k,v in pred.items():
    rho=st.spearmanr(v,amp).statistic
    print("   Spearman(%-24s, amplification) = %+.4f"%(k,rho))
print("\n   sr-roll's |far slope| x distance = %.4f ; rank among 14 = %d"%(
    abs([r for r in rows if r['metric']=='sr-roll'][0]['far'])*[r for r in rows if r['metric']=='sr-roll'][0]['out_a'],
    1+sorted([abs(r['far'])*r['out_a'] for r in rows],reverse=True).index(
        abs([r for r in rows if r['metric']=='sr-roll'][0]['far'])*[r for r in rows if r['metric']=='sr-roll'][0]['out_a'])))
print("   comfort's |far slope| x distance = %.4f ; rank = %d"%(
    abs([r for r in rows if r['metric']=='comfort'][0]['far'])*[r for r in rows if r['metric']=='comfort'][0]['out_a'],
    1+sorted([abs(r['far'])*r['out_a'] for r in rows],reverse=True).index(
        abs([r for r in rows if r['metric']=='comfort'][0]['far'])*[r for r in rows if r['metric']=='comfort'][0]['out_a'])))
json.dump({"rows":rows,"net":net,"extrap_total":tot_ex,"indomain_total":tot_in,
           "extrap_pct_of_net":100*tot_ex/net,
           "spearman":{k:float(st.spearmanr(v,amp).statistic) for k,v in pred.items()}},
          open(S/"amplification-audit.json","w"),indent=1)
print("\nWROTE amplification-audit.json")
