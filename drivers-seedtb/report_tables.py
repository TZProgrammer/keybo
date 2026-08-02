"""Emit the report's tables GENERATED (never hand-typed — a prior round shipped 56 wrong cells)."""
import json, sys
MY_WT="/local/home/zegertho/agent/workspaces/seedtb/wt"; sys.path.insert(0, MY_WT+"/src")
import keybo; assert keybo.__file__.startswith(MY_WT)
import numpy as np
from scipy import stats
ART="/local/home/zegertho/agent/state/seedtb/artifacts"
n_target=int(sys.argv[1]) if len(sys.argv)>1 else 25
d=json.load(open(f"{ART}/margins_n{n_target}.json"))
mspc=d["mspc"]; NAMES=["arm-B","F(2.5)","BALL-1","F(2.0)","candidate"]
PAIRS=[(NAMES[i],NAMES[j]) for i in range(5) for j in range(i+1,5)]
PRI=[p for p in PAIRS if "arm-B" in p]; SEC=[p for p in PAIRS if "arm-B" not in p]
def st(x,y,k=None):
    a=np.array(mspc[x][:k]); b=np.array(mspc[y][:k]); dd=a-b
    m=dd.mean(); s=dd.std(ddof=1); nn=len(dd); sem=s/np.sqrt(nn); t=m/sem
    p=float(2*stats.t.sf(abs(t),nn-1)); tc=stats.t.ppf(0.975,nn-1)
    return dict(n=nn,mean=m,sd=s,t=t,p=p,lo=m-tc*sem,hi=m+tc*sem,
                pos=int((dd>0).sum()),dsd=abs(m)/s)
def holm(pairs,res,alpha=0.05):
    o=sorted(pairs,key=lambda q:res[q]["p"]); m=len(o); out={}; ok=True
    for i,pr in enumerate(o):
        thr=alpha/(m-i); r=ok and res[pr]["p"]<thr
        if not r: ok=False
        out[pr]=(thr,r)
    return out
def projn(dsd,target=0.80):
    if dsd<=0: return None
    for k in range(3,20001):
        nc=stats.t.ppf(0.975,k-1)
        pw=stats.nct.sf(nc,k-1,dsd*np.sqrt(k))+stats.nct.cdf(-nc,k-1,dsd*np.sqrt(k))
        if pw>=target: return k
    return ">20000"

for stage_n in ([15,n_target] if n_target!=15 else [15]):
    res={pr:st(*pr,k=stage_n) for pr in PAIRS}
    hp=holm(PRI,res); hs=holm(SEC,res)
    alpha = 0.05 if stage_n==15 else 0.02
    print(f"\n### ALL 10 PAIRS at n={stage_n}   (margin = FIRST minus SECOND; negative = first faster)\n")
    print("| pair | family | mean Δ ms/char | sd | t | p (raw) | Holm thr | resolved? | signs +/− | d/sd | 95% CI | n for 80% |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pr in PAIRS:
        r=res[pr]; fam="**primary**" if pr in PRI else "secondary"
        thr,rej=(hp.get(pr) or hs.get(pr))
        print(f"| {pr[0]} vs {pr[1]} | {fam} | {r['mean']:+.4f} | {r['sd']:.4f} | {r['t']:+.3f} | "
              f"{r['p']:.2e} | {thr:.4f} | {'**YES**' if rej else 'no'} | "
              f"{r['pos']}/{r['n']-r['pos']} | {r['dsd']:.2f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
              f"{projn(r['dsd'])} |")
    print(f"\n**Primary family (Holm @0.05), ordered by p:**\n")
    for pr in sorted(PRI,key=lambda q:res[q]["p"]):
        thr,rej=hp[pr]
        print(f"- {pr[0]} vs {pr[1]}: p={res[pr]['p']:.2e}, thr={thr:.4f} → "
              f"{'**RESOLVED**' if rej else 'not resolved'}")

# board means per n
print(f"\n### BOARD MEANS — rank stability\n")
print("| board | mean n=3 | mean n=15 | mean n=%d | rank n=3 | rank n=15 | rank n=%d |" % (n_target,n_target))
print("|---|---|---|---|---|---|---|")
ms={k:{nm:float(np.mean(mspc[nm][:k])) for nm in NAMES} for k in (3,15,n_target)}
rk={k:{nm:i+1 for i,nm in enumerate(sorted(NAMES,key=lambda q:ms[k][q]))} for k in (3,15,n_target)}
for nm in sorted(NAMES,key=lambda q:ms[n_target][q]):
    print(f"| {nm} | {ms[3][nm]:.4f} | {ms[15][nm]:.4f} | {ms[n_target][nm]:.4f} | "
          f"{rk[3][nm]} | {rk[15][nm]} | {rk[n_target][nm]} |")

# P(arm-B faster)
print(f"\n### P(arm-B faster | data), flat prior\n")
print("| pair | n=3 | n=15 | n=%d |" % n_target)
print("|---|---|---|---|")
for other in NAMES[1:]:
    row=[]
    for k in (3,15,n_target):
        r=st("arm-B",other,k=k)
        row.append(float(stats.t.cdf(0,df=r['n']-1,loc=r['mean'],scale=r['sd']/np.sqrt(r['n']))))
    print(f"| arm-B vs {other} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} |")

# per-seed margins, primary
print(f"\n### PER-SEED MARGINS — the 4 primary pairs (all {n_target} seeds)\n")
print("| seed | " + " | ".join(f"arm-B − {o}" for o in NAMES[1:]) + " |")
print("|---|" + "---|"*4)
for s in range(n_target):
    print(f"| {s} | " + " | ".join(f"{mspc['arm-B'][s]-mspc[o][s]:+.4f}" for o in NAMES[1:]) + " |")
print("| **mean** | " + " | ".join(f"**{st('arm-B',o,k=n_target)['mean']:+.4f}**" for o in NAMES[1:]) + " |")
print("| **sd** | " + " | ".join(f"{st('arm-B',o,k=n_target)['sd']:.4f}" for o in NAMES[1:]) + " |")

# per-seed board totals
print(f"\n### PER-SEED ms/char, all 5 boards (the raw data)\n")
print("| seed | " + " | ".join(NAMES) + " |")
print("|---|" + "---|"*5)
for s in range(n_target):
    print(f"| {s} | " + " | ".join(f"{mspc[nm][s]:.6f}" for nm in NAMES) + " |")

# trace
print(f"\n### SEQUENTIAL TRACE (descriptive; NOT a decision surface) — raw p per pair\n")
print("| pair | " + " | ".join(f"n={k}" for k in range(3,n_target+1,2)) + " |")
print("|---|" + "---|"*len(range(3,n_target+1,2)))
for pr in PAIRS:
    print(f"| {pr[0]} vs {pr[1]} | " + " | ".join(f"{st(*pr,k=k)['p']:.3f}" for k in range(3,n_target+1,2)) + " |")
