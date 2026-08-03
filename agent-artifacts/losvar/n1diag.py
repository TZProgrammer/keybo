import os,sys,json,math
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"48")
WT="/local/home/zegertho/repos/keybo-wt-losvar"; sys.path.insert(0,WT+"/src")
import numpy as np, keybo
assert keybo.__file__.startswith(WT+"/")
from keybo.analysis.los import compute_los
tj=json.load(open('/local/home/zegertho/agent/state/tournament/artifacts/tournament.json'))
B=list(tj['boards']); FL=0.29046
print("ulp of 0.5 =", np.spacing(0.5))
bad={}
for b in B:
    ms=np.array(tj['mspc']['all'][b],float)
    for s in (0.0,0.4900,1.0287,1.0,5.0,50.0):
        r=compute_los(ms,ms.copy(),floor=FL,a_name=b,b_name=b,sigma_diff=s)
        dev=r.los_valid-0.5
        bad.setdefault(s,[]).append(dev)
for s,v in bad.items():
    v=np.array(v)
    print(f"sigma={s:8.4f}: worst |dev| {np.abs(v).max():.3e}  n_nonzero {int((v!=0).sum())}/{v.size}  "
          f"in ulps of 0.5: {np.abs(v).max()/np.spacing(0.5):.2f}")
# also LOS_design (the ORIGINAL bar, sigma-free path) for comparison
d=[]
for b in B:
    ms=np.array(tj['mspc']['all'][b],float)
    d.append(compute_los(ms,ms.copy(),floor=FL,a_name=b,b_name=b).los_design-0.5)
d=np.array(d)
print(f"LOS_design (original bar): worst |dev| {np.abs(d).max():.3e}  n_nonzero {int((d!=0).sum())}/{d.size}")
