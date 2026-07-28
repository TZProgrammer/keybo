"""Band-stratified version of the decisive test. My headline claim was about the NEAR-OPTIMAL
band specifically; the perturbation pool spans 254.6-280.2 ms/char, which is far wider. If the
rank agreement is carried by the DEGRADED tail, the headline survives; if it holds in the tight
band, my headline was wrong."""
import sys, json
from pathlib import Path
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
import numpy as np, evobj as EV
import scipy.stats as st
S=Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARMJ="/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
J=json.load(open(S/"judgement.json")); pl=J["per_layout"]
INC=["keybo-lsb","lsb-sib","archive-1843","archive-1846","keybo-lsb+lm"]
fe=EV.FastEval(None,ARMJ,with_surface=True)
rng=np.random.default_rng(4242)

# Bigger perturbation pool so the tight bands have enough n
pool=[]
for name in INC:
    base=EV.perm_of(pl[name]["layout"])[:30].copy()
    pool.append(base.copy())
    for k in (1,2,3,4,5,6):
        for _ in range(1200):
            p=base.copy()
            for _s in range(k):
                i,j=rng.choice(30,2,replace=False); p[i],p[j]=p[j],p[i]
            pool.append(p)
P=np.stack([np.concatenate([p,[30]]).astype(np.int32) for p in pool])
g=fe.gauges(P); ms=g["_ms_per_char"]
ev=np.zeros(P.shape[0]); evc=np.zeros(P.shape[0])
for c in fe.curves:
    x=g[c.metric]; ev=ev+c.price(x); evc=evc+c.price(np.clip(x,c.domain[0],c.domain[1]))

V=json.load(open(ARMJ))["validation"]["noise_placebo"]
p95=V["spearman_abs_p95"]
def boot(x,y,n=1500):
    r=[];idx=np.arange(len(x))
    for _ in range(n):
        s=rng.choice(idx,len(idx),replace=True)
        r.append(st.spearmanr(x[s],y[s]).statistic)
    return float(np.nanpercentile(r,2.5)),float(np.nanpercentile(r,97.5))

print("pool n=%d, ms/char range [%.4f, %.4f]"%(P.shape[0],ms.min(),ms.max()))
print("incumbents span 254.6307-254.8436; the three arm champions span 253.9006-256.8466")
print("evidence-scorer's own noise band p95 = %.4f\n"%p95)
print("=== rank agreement (POSITIVE = objective agrees with predicted speed), stratified by band ===")
print("%-28s %6s %10s %9s %-22s %-22s"%("band (ms/char)","n","sd(ms)","rho_raw","CI95_raw","verdict_raw"))
out={}
BANDS=[("<= 255.0",255.0),("<= 255.5",255.5),("<= 256.0",256.0),("<= 257.0",257.0),
       ("<= 260.0",260.0),("<= 270.0",270.0),("all",1e9)]
for lbl,cap in BANDS:
    m=ms<=cap
    n=int(m.sum())
    if n<30: print("%-28s %6d   (too few)"%(lbl,n)); continue
    r=st.spearmanr(ev[m],ms[m]).statistic; lo,hi=boot(ev[m],ms[m])
    v=("agrees" if lo>0 else ("ANTI-ranks" if hi<0 else "INDISTINGUISHABLE from 0"))
    if abs(r)<=p95: v+=" [inside noise band]"
    print("%-28s %6d %10.4f %+9.4f [%+.4f,%+.4f] %s"%(lbl,n,ms[m].std(ddof=1),r,lo,hi,v))
    rc=st.spearmanr(evc[m],ms[m]).statistic; loc,hic=boot(evc[m],ms[m])
    vc=("agrees" if loc>0 else ("ANTI-ranks" if hic<0 else "INDISTINGUISHABLE from 0"))
    print("%-28s %6s %10s %+9.4f [%+.4f,%+.4f] %s   <- CLAMPED"%("",  "", "",rc,loc,hic,vc))
    out[lbl]={"n":n,"sd_ms":float(ms[m].std(ddof=1)),"rho_raw":float(r),"ci_raw":[lo,hi],
              "rho_clamped":float(rc),"ci_clamped":[loc,hic],"verdict_raw":v,"verdict_clamped":vc}
json.dump(out,open(S/"banded-rank.json","w"),indent=1)
print("\nWROTE banded-rank.json")
