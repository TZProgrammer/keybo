"""Is FastEval.gauges BATCH-SHAPE DEPENDENT? Measured, not asserted.
Hypothesis: `VT = W @ self.KT` at (B,29791)@(29791,n_t) is UNPADDED, so BLAS picks its
kernel from B and the summation order changes -- the MODELNORM-1 CORRECTION class. If so,
one layout's gauge value depends on how many OTHER layouts share its batch."""
import json, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, evobj as EV
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
ARMB="flmpg-yuo,sntdcireahkxbwv'.jzq"
G=("sfb","sfs","sfb-dist","sfs-dist","lsb","lsb-dist","alt","roll","sr-roll","redir",
   "scissor","imbalance","oxey-style","comfort","_ms_per_char")
rng=np.random.default_rng(7)
target=EV.perm_of(ARMB)
base=None; worst={g:0.0 for g in G}; nd={g:0 for g in G}; N=0
for B in range(1,61):
    # target row FIXED at position 0; the other B-1 rows are arbitrary fillers
    rows=[target]+[np.concatenate([rng.permutation(30).astype(np.int32),[30]]) for _ in range(B-1)]
    g=fe.gauges(np.stack(rows))
    v={k:float(g[k][0]) for k in G}
    if base is None: base=v; continue
    N+=1
    for k in G:
        d=abs(v[k]-base[k])
        if d>0: nd[k]+=1
        worst[k]=max(worst[k],d)
print(f"arm B's OWN gauge values across {N+1} batch lengths (B=1..60), its row held FIXED:")
print(f"{'gauge':<13} {'worst |abs diff|':>17} {'worst |rel|':>12} {'lengths differing':>18}")
for k in G:
    print(f"{k:<13} {worst[k]:>17.3e} {worst[k]/max(abs(base[k]),1e-12):>12.3e} {nd[k]:>13}/{N}")
res={"n_lengths":N+1,"worst_abs":worst,"n_differing":nd,"base_B1":base}
# Consequence bound: what does this do to D, and to my decision margins?
SC=json.load(open('/tmp/armg/agent-artifacts/armg/D-prereg-input.json'))["scale_six_champion_range"]
dmax=sum(worst[k]/SC[k] for k in G if k!="_ms_per_char")
print(f"\nIMPLIED WORST-CASE PERTURBATION ON D (sum over 14 axes of worst/scale) = {dmax:.3e}")
print(f"  vs my registered decision thresholds: D=1.4878 (failure bar), gaps between existing layouts >= 0.2")
print(f"  => ratio to the TIGHTEST threshold I use ({0.2:.1f}) = {dmax/0.2:.3e}")
print(f"IMPLIED WORST-CASE ON ms/char = {worst['_ms_per_char']:.3e} vs EPS={0.1234} => ratio {worst['_ms_per_char']/0.1234:.3e}")
res["implied_D_perturbation"]=dmax
res["implied_ms_perturbation"]=worst["_ms_per_char"]
json.dump(res,open('/tmp/armg/agent-artifacts/armg/shape-dependence.json','w'),indent=1)
