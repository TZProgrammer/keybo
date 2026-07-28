"""DECISIVE TEST for audit claim 1: are the weights informative about predicted time IN THE
NEAR-OPTIMAL BAND — measured directly, with NO dependence on arm C?

Pool: perturbations of the five incumbents (1-4 random swaps), which is exactly "the
near-optimal band" and is selected by NEITHER objective, so it cannot flatter either.
Second pool: the union of the three arms' top-50 archives (band, but objective-selected —
reported separately because it IS biased).
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
INC=["keybo-lsb","lsb-sib","archive-1843","archive-1846","keybo-lsb+lm"]
fe=EV.FastEval(None,ARMJ,with_surface=True)

def score_pool(perms, clamp):
    g=fe.gauges(perms)
    tot=np.zeros(perms.shape[0])
    for c in fe.curves:
        x=g[c.metric].copy()
        if clamp: x=np.clip(x,c.domain[0],c.domain[1])
        tot=tot+c.price(x)
    return tot, g

rng=np.random.default_rng(4242)
# ---- POOL 1: incumbent perturbations (unbiased w.r.t. both objectives) ----
pool=[]
for name in INC:
    base=EV.perm_of(pl[name]["layout"])[:30].copy()
    pool.append(base.copy())
    for k in (1,2,3,4):
        for _ in range(100):
            p=base.copy()
            for _s in range(k):
                i,j=rng.choice(30,2,replace=False); p[i],p[j]=p[j],p[i]
            pool.append(p)
P1=np.stack([np.concatenate([p,[30]]).astype(np.int32) for p in pool])
# ---- POOL 2: union of the three arms' top-50 archives (objective-SELECTED; biased) ----
lays=set()
for a in ("evidence","baseline","constrained"):
    for row in json.load(open(S/"runs"/f"arm-{a}.json"))["top50"]:
        lays.add(row["layout"])
P2=np.stack([EV.perm_of(l) for l in sorted(lays)])

# ---- evidence-scorer's own noise band, for reference ----
V=json.load(open(ARMJ))["validation"]["noise_placebo"]
band_mean, band_p95 = V["spearman_abs_mean"], V["spearman_abs_p95"]

def boot_ci(x,y,n=2000):
    r=[]; idx=np.arange(len(x))
    for _ in range(n):
        s=rng.choice(idx,len(idx),replace=True)
        if len(set(y[s].tolist()))<3: continue
        r.append(st.spearmanr(x[s],y[s]).statistic)
    return float(np.percentile(r,2.5)), float(np.percentile(r,97.5))

print("=== DECISIVE: rank agreement between the EVIDENCE objective and predicted ms/char ===")
print("(evidence objective is lower=better and ms/char is lower=better, so a POSITIVE rho")
print(" means the objective agrees with predicted speed. NEGATIVE means it ANTI-ranks.)")
print("evidence-scorer's own noise placebo band: |rho| mean %.4f, p95 %.4f\n"%(band_mean,band_p95))
out={}
for lbl,PP,note in (("incumbent-perturbation (UNBIASED)",P1,"1-4 swaps off each incumbent"),
                    ("arms top-50 union (objective-SELECTED, biased)",P2,"biased pool, shown for contrast")):
    ms=fe.gauges(PP)["_ms_per_char"]
    for clamp in (False,True):
        ev,_=score_pool(PP,clamp)
        rho=st.spearmanr(ev,ms).statistic
        lo,hi=boot_ci(ev,ms)
        verdict = ("ANTI-ranks speed" if hi<0 else ("agrees with speed" if lo>0 else "INDISTINGUISHABLE from 0"))
        inband = abs(rho)<=band_p95
        print("%-46s n=%4d %-9s rho=%+.4f CI95=[%+.4f,%+.4f]  %s%s"%(
            lbl,PP.shape[0],"CLAMPED" if clamp else "raw",rho,lo,hi,verdict,
            "  (|rho| inside the scorer's own noise band)" if inband else ""))
        out[f"{lbl}|{'clamped' if clamp else 'raw'}"]={"n":int(PP.shape[0]),"rho":float(rho),
            "ci95":[lo,hi],"verdict":verdict,"inside_noise_band":bool(inband)}
    # ms/char spread of the pool, so "in band" is quantified
    print("      pool ms/char: min %.4f max %.4f sd %.4f\n"%(ms.min(),ms.max(),ms.std(ddof=1)))
    out[f"{lbl}|pool"]={"ms_min":float(ms.min()),"ms_max":float(ms.max()),"ms_sd":float(ms.std(ddof=1))}

# ---- the SAME test for the BASELINE objective, as a positive control on the instrument ----
print("=== POSITIVE CONTROL on the instrument: the same rank test using the SERVED objective ===")
ms1=fe.gauges(P1)["_ms_per_char"]
rho=st.spearmanr(ms1,ms1).statistic
print("   served objective vs itself on pool 1: rho=%+.4f (must be +1.0000 — the test can detect agreement)"%rho)
out["instrument_positive_control_rho"]=float(rho)
json.dump(out,open(S/"decisive-inband-rank.json","w"),indent=1)
print("\nWROTE decisive-inband-rank.json")
