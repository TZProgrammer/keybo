import sys, json
from pathlib import Path
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
import numpy as np, evobj as EV
from keybo.analysis.evidence_scorer import LIVE_GAUGES
S=Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARMD="/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
J=json.load(open(S/"judgement.json")); pl=J["per_layout"]
W=json.load(open(ARMD))["weights"]
shap={g["metric"]:g["shap_share_pct"] for g in W["weights"]}
fe=EV.FastEval(None,ARMD,with_surface=True)
CH=["champ-evidence","champ-baseline","champ-constrained"]
INC=["keybo-lsb","lsb-sib","archive-1843","archive-1846","keybo-lsb+lm"]
names=CH+INC
perms=np.stack([EV.perm_of(pl[n]["layout"]) for n in names]); g=fe.gauges(perms)

# ---------- (A) CLAMPED re-scoring: price at the clamped level (SEARCH_DOMAIN_POLICY=CLAMP) ----------
def score(gv, clamp):
    tot=np.zeros(len(names))
    per={}
    for c in fe.curves:
        x=gv[c.metric].copy()
        if clamp: x=np.clip(x, c.domain[0], c.domain[1])
        p=c.price(x); per[c.metric]=p; tot=tot+p
    return tot, per
raw,per_raw = score(g,False)
cl ,per_cl  = score(g,True)
print("=== (A) does CLAMPing valid_domain reverse the RANKING? (re-scoring only; arm D is the sibling's search) ===")
print("%-18s %11s %11s %11s | %10s"%("layout","raw","CLAMPED","delta","ms/char"))
for i,n in enumerate(names):
    print("%-18s %11.4f %11.4f %+11.4f | %10.4f"%(n,raw[i],cl[i],cl[i]-raw[i],pl[n]["ms_per_char"]))
oc=np.argsort(cl); orr=np.argsort(raw)
print("\nRAW     ranking (best first):", [names[i] for i in orr])
print("CLAMPED ranking (best first):", [names[i] for i in oc])
best_cl=names[int(np.argmin(cl))]; best_raw=names[int(np.argmin(raw))]
print("\nBEST under raw = %s ; BEST under CLAMP = %s"%(best_raw,best_cl))
import scipy.stats as st
rho_raw=st.spearmanr(raw,[pl[n]["ms_per_char"] for n in names]).statistic
rho_cl =st.spearmanr(cl ,[pl[n]["ms_per_char"] for n in names]).statistic
print("Spearman(objective, ms/char) over these 8: raw %+.4f  CLAMPED %+.4f  (positive = objective agrees with speed)"%(rho_raw,rho_cl))

# ---------- (B) amplification ratio for EVERY gauge, not just sr-roll ----------
iA=names.index("champ-evidence"); iR=names.index("archive-1846")
d={m: float(per_raw[m][iA]-per_raw[m][iR]) for m in LIVE_GAUGES}
gain=sum(d.values()); pos=sum(v for v in d.values() if v<0); neg=sum(v for v in d.values() if v>0)
print("\n=== (B) is '8x amplification' a property of sr-roll, or of hinge geometry? ===")
print("net gain %.4f | gross improvement %.4f (%.1f%% of net) | offsetting %.4f (%.1f%%)"%(gain,pos,100*pos/gain,neg,100*neg/gain))
print("%-11s %9s %9s %8s %8s %9s %9s"%("gauge","delta","%of_NET","%of_GROSS","shap%","amplif","far_slope"))
rows=[]
for m in sorted(LIVE_GAUGES,key=lambda m:d[m]):
    c=[cc for cc in fe.curves if cc.metric==m][0]
    fs = c.coeffs[1]+c.coeffs[2] if c.form=="hinge" else (c.coeffs[1] if c.form=="linear" else float("nan"))
    pn=100*d[m]/gain; pg=100*d[m]/pos if d[m]<0 else 0.0
    amp=pn/shap[m]
    rows.append((m,d[m],pn,pg,shap[m],amp,fs))
    print("%-11s %+9.4f %+9.1f %8.1f %8.2f %9.2f %+9.4f"%(m,d[m],pn,pg,shap[m],amp,fs))
print("\ngauges with amplification > 1 (contributing MORE than their fitted importance):")
for m,dd,pn,pg,sh,amp,fs in sorted(rows,key=lambda r:-r[5]):
    if amp>1: print("   %-11s amp=%.2f (delta %+.4f, shap %.2f%%, far_slope %+.4f)"%(m,amp,dd,sh,fs))

# ---------- (C) reference sensitivity of the decomposition ----------
print("\n=== (C) is the decomposition reference-dependent? comfort/sr-roll share vs each incumbent ===")
print("%-14s %9s %9s %9s %9s"%("reference","net_gain","comfort%","sr-roll%","sum%"))
for r in INC:
    iRx=names.index(r)
    dd={m: float(per_raw[m][iA]-per_raw[m][iRx]) for m in LIVE_GAUGES}
    gg=sum(dd.values())
    print("%-14s %9.4f %9.1f %9.1f %9.1f"%(r,gg,100*dd["comfort"]/gg,100*dd["sr-roll"]/gg,100*(dd["comfort"]+dd["sr-roll"])/gg))

# ---------- (D) does the ARCHIVE-fitted arm's domain cover the near-optimal band? ----------
print("\n=== (D) UNTESTED ALTERNATIVE: the archive400-fitted weights' domains vs the near-optimal band ===")
A2=json.load(open("/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-archive400-native.json"))["weights"]
dom2={x["metric"]:x["valid_domain"] for x in A2["weights"]}
nood=0
for m in LIVE_GAUGES:
    lo,hi=dom2[m]; v=float(g[m][names.index("keybo-lsb")]); inside=lo<=v<=hi
    nood += 0 if inside else 1
    print("   %-11s keybo-lsb=%9.4f archive400 domain=[%9.4f,%9.4f] %s"%(m,v,lo,hi,"in" if inside else "OUT"))
print("   -> keybo-lsb is out-of-domain on %d of 14 under the ARCHIVE-fitted weights (vs 9 of 14 under random400)"%nood)
json.dump({"clamped":{n:float(cl[i]) for i,n in enumerate(names)},
           "raw":{n:float(raw[i]) for i,n in enumerate(names)},
           "best_raw":best_raw,"best_clamped":best_cl,
           "spearman_obj_vs_mschar":{"raw":float(rho_raw),"clamped":float(rho_cl)},
           "amplification":{m:{"delta":dd,"pct_net":pn,"pct_gross":pg,"shap_pct":sh,"amplification":amp,"far_slope":fs} for m,dd,pn,pg,sh,amp,fs in rows},
           "gross_improvement":float(pos),"offsetting":float(neg),"net":float(gain),
           "archive400_ood_keybo_lsb":nood},
          open(S/"self-audit.json","w"),indent=1)
print("\nWROTE self-audit.json")
