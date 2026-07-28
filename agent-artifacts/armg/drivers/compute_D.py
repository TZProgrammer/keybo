import json, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, evobj as EV
G=("sfb","sfs","sfb-dist","sfs-dist","lsb","lsb-dist","alt","roll","sr-roll","redir","scissor","imbalance","oxey-style","comfort")
DIR={"sfb":1,"sfs":1,"sfb-dist":1,"sfs-dist":1,"lsb":1,"lsb-dist":1,"alt":-1,"roll":-1,"sr-roll":-1,"redir":1,"scissor":1,"imbalance":1,"oxey-style":1,"comfort":1}
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
P=json.load(open('/local/home/zegertho/agent/state/optevidence/artifacts/search-noise-placebo.json'))
six={f"s{r['seed']}":r['layout'] for r in P['runs']['baseline']}
INC={"arm-A":"udy.,fgpmliheaocsntr-k'qjwzbvx","keybo-lsb":"pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
 "keybo-lsb+lm":"pyuo,vgdnmhiea.cstrlkj-z'fwbxq","flagship-c3":"pyou'vgdnmheai.cstrlkjz,-wfbxq",
 "graphite":"bldwz'foujnrtsgyhaeixqmcvkp,.-"}
ARMB="flmpg-yuo,sntdcireahkxbwv'.jzq"
allL={"arm-B":ARMB,**six,**INC}
names=list(allL)
g=fe.gauges(np.stack([EV.perm_of(allL[n]) for n in names]))
ref={gg:float(g[gg][names.index("arm-B")]) for gg in G}
# SCALE: range across the SIX frozen 1M champions (pool-matched near-optimal scale)
sixi=[names.index(k) for k in six]
scale={gg: float(g[gg][sixi].max()-g[gg][sixi].min()) for gg in G}
print("=== SCALE s_i = range over the six frozen 1M champions (near-optimal, pool-matched) ===")
for gg in G: print(f"  {gg:<12} armB={ref[gg]:>9.4f}  s={scale[gg]:>9.4f}")
print()
print(f"{'name':<13} {'ms/char':>10} {'D (sum pos excess)':>19} {'worse':>6} {'better':>7} {'tie':>4}")
res={}
for n in names:
    i=names.index(n); D=0.0; w=b=t=0
    per={}
    for gg in G:
        d=(float(g[gg][i])-ref[gg])*DIR[gg]/scale[gg]
        per[gg]=d
        if abs(float(g[gg][i])-ref[gg])<1e-12: t+=1
        elif d>0: w+=1
        else: b+=1
        D+=max(0.0,d)
    res[n]={"layout":allL[n],"ms":float(g['_ms_per_char'][i]),"D":D,"worse":w,"better":b,"tie":t,"per_gauge_norm_excess":per}
    print(f"{n:<13} {float(g['_ms_per_char'][i]):>10.4f} {D:>19.4f} {w:>6} {b:>7} {t:>4}")
print("\nD(arm-B) must be EXACTLY 0.0 by construction -> positive control:", res['arm-B']['D']==0.0)
best=min((v['D'],k) for k,v in res.items() if k!='arm-B')
print(f"lowest D among all existing layouts (excl arm B): {best[1]} at D={best[0]:.4f}")
json.dump({"reference":"arm-B","ref_gauges":ref,"scale_six_champion_range":scale,"D_of_existing":res},
  open('/tmp/armg/agent-artifacts/armg/D-prereg-input.json','w'),indent=1)
