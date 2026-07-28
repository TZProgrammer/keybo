"""How much oxey-style headroom sits inside a speed band, and where must the band be set?
Pre-registration input ONLY -- no search has run.
"""
import json, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, evobj as EV
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
P=json.load(open('/local/home/zegertho/agent/state/optevidence/artifacts/search-noise-placebo.json'))
six=[r['layout'] for r in P['runs']['baseline']]
g6=fe.gauges(np.stack([EV.perm_of(l) for l in six]))
ARMB=253.90057910352604
sd=P['bands']['baseline']['ms_per_char']['sd']
print(f"arm B          = {ARMB:.4f}")
print(f"borrowed sd    = {sd:.4f}  (baseline obj, n=6, 1M budget, sd stat -- NOT mine)")
print(f"2x sd band     = [{ARMB:.4f}, {ARMB+2*sd:.4f}]")
print()
ms6=g6['_ms_per_char']; ox6=g6['oxey-style']
inband=ms6<=ARMB+2*sd
print(f"of the six frozen champions, {inband.sum()}/6 sit within arm B + 2x sd")
print(f"  their oxey-style: {np.sort(ox6[inband])}")
print(f"  ALL six oxey-style range: {ox6.min():.4f} .. {ox6.max():.4f}  ratio {ox6.max()/ox6.min():.2f}x")
print(f"  arm B's own oxey-style: {ox6[0]:.4f}  (rank {1+int((ox6<ox6[0]).sum())} of 6, lower=better)")
print()
# what does the epsilon-constrained frontier look like on RANDOM near-optimal draws?
# cheap probe: how tight is the ms/char <= ARMB+eps constraint for a random search?
rng=np.random.default_rng(4242)
perms=np.stack([np.concatenate([rng.permutation(30).astype(np.int32),[30]]) for _ in range(20000)])
gr=fe.gauges(perms)
print(f"20000 random perms: ms/char min {gr['_ms_per_char'].min():.4f} -- so random draws are FAR from the band")
print(f"                    oxey-style min {gr['oxey-style'].min():.4f} max {gr['oxey-style'].max():.4f}")
json.dump({"armB":ARMB,"borrowed_sd":sd,
  "six_oxey":{six[i]:float(ox6[i]) for i in range(6)},
  "six_ms":{six[i]:float(ms6[i]) for i in range(6)},
  "oxey_ratio_six":float(ox6.max()/ox6.min()),
  "random_min_ms":float(gr['_ms_per_char'].min())},
  open('/tmp/armg/agent-artifacts/armg/headroom-probe.json','w'),indent=1)
