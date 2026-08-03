import os,sys,json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"48")
WT="/local/home/zegertho/repos/keybo-wt-losvar"; sys.path.insert(0,WT+"/src")
import numpy as np, keybo
assert keybo.__file__.startswith(WT+"/"), keybo.__file__
print("keybo:",keybo.__file__)
from keybo.analysis.los import compute_los, split_half_floor
tj=json.load(open('/local/home/zegertho/agent/state/tournament/artifacts/tournament.json'))
d1=json.load(open('/local/home/zegertho/agent/state/losvar/artifacts/v01_decompose.json'))
B=list(tj['boards']); A,Bb="candidate","flagship-c3"
res={}
for pr in ("all","observed","common"):
    panel=np.vstack([np.array(tj['mspc'][pr][b],float) for b in B])
    fl=split_half_floor(panel,n_partitions=2000,rng=np.random.default_rng(20260803),pct=90.0)
    ma,mb=np.array(tj['mspc'][pr][A],float),np.array(tj['mspc'][pr][Bb],float)
    row={"floor":fl['floor'],"floor_p50":fl['p50'],"half_n":fl['half_n']}
    r0=compute_los(ma,mb,floor=fl['floor'],a_name=A,b_name=Bb)
    row["margin"]=r0.mean_margin; row["sem"]=r0.sem_margin; row["sd"]=r0.sd_margin
    row["margin_over_floor"]=r0.margin_over_floor
    row["LOS_design"]=r0.los_design; row["LOS_seed"]=r0.los_seed; row["LOS_typist"]=r0.los_typist
    row["signs"]=f"{r0.signs_a_faster}/{r0.signs_b_faster}"
    for vn in ("scoring_bucket_80","all_buckets"):
        s=d1['decomposition']['variants'][vn]['pairs']['candidate vs flagship-c3']['De_delev_rms']
        rv=compute_los(ma,mb,floor=fl['floor'],a_name=A,b_name=Bb,sigma_diff=s)
        row[f"sigma_{vn}"]=s; row[f"LOS_valid_{vn}"]=rv.los_valid; row[f"scale_{vn}"]=rv.scale_valid
    res[pr]=row
    print(f"{pr:7s} floor {fl['floor']:.4f} margin {r0.mean_margin:+.4f} ({r0.margin_over_floor:.2f}x) sem {r0.sem_margin:.4f} "
          f"LOS_design {r0.los_design:.4f} | sig_sb {row['sigma_scoring_bucket_80']:.4f}->LOS_valid {row['LOS_valid_scoring_bucket_80']:.4f} "
          f"| sig_ab {row['sigma_all_buckets']:.4f}->LOS_valid {row['LOS_valid_all_buckets']:.4f}")
# known-big spot check: candidate vs qwerty
print()
for vn in ("scoring_bucket_80","all_buckets"):
    s=d1['decomposition']['variants'][vn]['pairs']['candidate vs qwerty']['De_delev_rms']
    panel=np.vstack([np.array(tj['mspc']['all'][b],float) for b in B])
    fl=split_half_floor(panel,n_partitions=2000,rng=np.random.default_rng(20260803),pct=90.0)['floor']
    rv=compute_los(np.array(tj['mspc']['all']['candidate'],float),np.array(tj['mspc']['all']['qwerty'],float),
                   floor=fl,a_name='candidate',b_name='qwerty',sigma_diff=s)
    print(f"KNOWN-BIG cand-vs-qwerty {vn}: sigma {s:.4f} margin {rv.mean_margin:+.4f} ({rv.margin_over_floor:.1f}x) "
          f"LOS_design {rv.los_design:.4f} LOS_valid {rv.los_valid:.4f}")
json.dump(res,open('/local/home/zegertho/agent/state/losvar/artifacts/v03_quick_live.json','w'),indent=1,default=float)
print("\nwrote v03_quick_live.json")
