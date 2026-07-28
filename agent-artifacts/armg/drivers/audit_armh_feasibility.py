"""(d) TRY TO BREAK ARM H. Which of the 13 hard constraints bind, and can a hard-constraint
search report a FALSE empty? Evidence: my own 273-layout archive."""
import json, glob, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, search as S, evobj as EV
RUNS=Path("/local/home/zegertho/agent/state/armg/artifacts/runs")
pool=set()
for f in sorted(glob.glob(str(RUNS/"*-r?.json"))):
    b=json.load(open(f)); pool |= {e["layout"] for e in b["top50"]}
pl=sorted(pool)
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
g=fe.gauges(np.stack([EV.perm_of(x) for x in pl]))
ms=g["_ms_per_char"]
G=[k for k in S.ARMG_DIR]
OX="oxey-style"
others=[k for k in G if k!=OX]
print(f"archive n={len(pl)}\n")
print("PER-CONSTRAINT SATISFACTION over my 273 archived layouts (arm H's 13 non-oxey caps,")
print("g <= g_armB, plus the speed cap). 'n_ok' = how many of 273 satisfy that ONE constraint:")
print(f"{'constraint':<14} {'armB cap':>10} {'n_ok/273':>10} {'%':>7} {'archive best':>13}")
sat={}
for k in others:
    ok = (S.ARMG_DIR[k]*(g[k]-S.ARMG_REF[k]) <= 1e-12)
    sat[k]=ok
    best = float(g[k].min()) if S.ARMG_DIR[k]>0 else float(g[k].max())
    print(f"{k:<14} {S.ARMG_REF[k]:>10.4f} {int(ok.sum()):>10} {ok.mean():>6.1%} {best:>13.4f}")
# speed cap at the MEASURED band (what arm H will use)
sd=0.04917079026480171
cap=S.ARMG_REF_MS+2*sd
spd = ms<=cap
print(f"{'ms/char':<14} {cap:>10.4f} {int(spd.sum()):>10} {spd.mean():>6.1%} {float(ms.min()):>13.4f}")
print()
# joint feasibility, and WHICH constraint kills it
allok=np.ones(len(pl),bool)
for k in others: allok &= sat[k]
print(f"JOINT (13 gauge caps only, no speed): {int(allok.sum())} of {len(pl)} feasible")
print(f"JOINT (13 caps + speed cap):          {int((allok&spd).sum())} of {len(pl)} feasible")
print()
# leave-one-out: which single constraint is most binding?
print("LEAVE-ONE-CONSTRAINT-OUT: how many become feasible if we DROP just that one cap?")
res=[]
for k in others:
    m=np.ones(len(pl),bool)
    for j in others:
        if j!=k: m&=sat[j]
    res.append((int((m&spd).sum()),k))
res.sort(reverse=True)
for n,k in res[:8]:
    print(f"   drop {k:<12} -> {n:>3} feasible of 273   (binding rank)")
print()
# how many caps does the BEST archive layout violate?
nviol=np.zeros(len(pl),int)
for k in others: nviol += (~sat[k]).astype(int)
print(f"distribution of #caps VIOLATED (of 13) across the 273 archive layouts:")
import collections
for v,c in sorted(collections.Counter(nviol.tolist()).items()): print(f"   {v:>2} violated : {c:>3} layouts")
i=int(np.argmin(nviol))
print(f"\nclosest archive layout to feasible: {pl[i]} violates {nviol[i]} caps, ms {float(ms[i]):.4f}")
viol=[k for k in others if not sat[k][i]]
print(f"   its violations: {viol}")
print(f"   its oxey-style: {float(g[OX][i]):.4f}  (armB {S.ARMG_REF[OX]:.4f})")
# and among those with FEWEST violations, what is the oxey range?
mn=nviol.min()
sel=np.where(nviol==mn)[0]
print(f"\namong the {len(sel)} layouts violating only {mn} caps: oxey range "
      f"{float(g[OX][sel].min()):.4f}..{float(g[OX][sel].max()):.4f}, ms range "
      f"{float(ms[sel].min()):.4f}..{float(ms[sel].max()):.4f}")
json.dump({"n_archive":len(pl),"per_constraint_n_ok":{k:int(sat[k].sum()) for k in others},
  "speed_cap":cap,"n_speed_ok":int(spd.sum()),
  "joint_13caps":int(allok.sum()),"joint_13caps_plus_speed":int((allok&spd).sum()),
  "leave_one_out":[{"dropped":k,"n_feasible":n} for n,k in res],
  "min_caps_violated":int(mn),"n_at_min":int(len(sel)),
  "closest":{"layout":pl[i],"n_violated":int(nviol[i]),"violations":viol,"ms":float(ms[i]),
             "oxey":float(g[OX][i])}},
  open("/local/home/zegertho/agent/state/armg/artifacts/armh-feasibility-warning.json","w"),indent=1)
print("\nWROTE armh-feasibility-warning.json")
