import os,sys,json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"48")
WT="/local/home/zegertho/repos/keybo-wt-losvar"; sys.path.insert(0,WT+"/src")
import numpy as np, keybo
assert keybo.__file__.startswith(WT+"/")
from keybo.analysis.los import compute_los, split_half_floor
from scipy import stats
tj=json.load(open('/local/home/zegertho/agent/state/tournament/artifacts/tournament.json'))
d2=json.load(open('/local/home/zegertho/agent/state/losvar/artifacts/v02_sigma_and_flips.json'))
B=list(tj['boards'])
panel=np.vstack([np.array(tj['mspc']['all'][b],float) for b in B])
FL=split_half_floor(panel,n_partitions=2000,rng=np.random.default_rng(20260803),pct=90.0)['floor']
ma=np.array(tj['mspc']['all']['candidate'],float); mb=np.array(tj['mspc']['all']['flagship-c3'],float)
print("LIVE PAIR sensitivity across the FULL range of measured error scales:")
rows=[]
for tag,s in (("sigma_diff primary (scoring bucket)",0.4900),
              ("sigma_diff all-buckets",1.0287),
              ("route-b2 de-levelled err rms (PESSIMISTIC)",3.7314),
              ("route-b2 raw err rms (cohort-confounded)",6.1141),
              ("raw held-out wmae (the NAIVE fix)",9.1224)):
    r=compute_los(ma,mb,floor=FL,a_name='candidate',b_name='flagship-c3',sigma_diff=s)
    kb=compute_los(np.array(tj['mspc']['all']['candidate'],float),
                   np.array(tj['mspc']['all']['qwerty'],float),floor=FL,sigma_diff=s)
    rows.append({"basis":tag,"sigma":s,"LOS_valid_live":r.los_valid,"LOS_valid_knownbig":kb.los_valid,
                 "implied_flip_live":float(stats.norm.cdf(-abs(r.mean_margin)/s))})
    print(f"  {tag:44s} sigma {s:7.4f} -> live {r.los_valid:.4f}  known-big {kb.los_valid:.4f}  "
          f"implied-flip {stats.norm.cdf(-abs(r.mean_margin)/s)*100:5.1f}%")
# CLOSING-2 trajectory
print("\nCLOSING-2 trajectory (single-variable: eval layout FIXED, train set varies):")
c2=d2["closing2_layout_count"]
for vn in ("scoring_bucket_80","all_buckets"):
    print(f"  {vn}:")
    for k in ("1","2","3"):
        b=c2[vn][k]
        print(f"    n_train={k} ({b['n_configs']:2d} cfg): position-resid rms {b['position_resid_rms_mean']:7.3f}  "
              f"sigma_diff live {b['sigma_diff_rms']['candidate|flagship-c3']:.4f}  "
              f"cand-qwerty {b['sigma_diff_rms']['candidate|qwerty']:.4f}  "
              f"armB-cand {b['sigma_diff_rms']['arm-B|candidate']:.4f}")
# all-78 sigma_diff distribution
pp=d2["sigma_diff"]["scoring_bucket_80"]["pairs"]
sd=np.array([v["sigma_diff_rms"] for v in pp.values()])
print(f"\nALL 78 PAIRS sigma_diff (scoring bucket): min {sd.min():.4f} p50 {np.median(sd):.4f} "
      f"max {sd.max():.4f} mean {sd.mean():.4f}")
# how many pairs decided under LOS_valid vs LOS_design
import itertools
dec_d=dec_v=0; rows2=[]
for a,b in itertools.combinations(B,2):
    k=f"{a}|{b}" if f"{a}|{b}" in pp else f"{b}|{a}"
    s=pp[k]["sigma_diff_rms"]
    r=compute_los(np.array(tj['mspc']['all'][a],float),np.array(tj['mspc']['all'][b],float),
                  floor=FL,a_name=a,b_name=b,sigma_diff=s)
    dec_d+=int(r.los_design>=0.95 or r.los_design<=0.05)
    dec_v+=int(r.los_valid>=0.95 or r.los_valid<=0.05)
    rows2.append({"pair":f"{a} vs {b}","margin":r.mean_margin,"m_over_floor":r.margin_over_floor,
                  "sigma":s,"LOS_design":r.los_design,"LOS_valid":r.los_valid})
print(f"DECIDED over 78 pairs: LOS_design {dec_d}/78   LOS_valid {dec_v}/78")
json.dump({"floor_all":FL,"sensitivity":rows,"matrix":rows2,
           "decided_design":dec_d,"decided_valid":dec_v,
           "sigma_all78":{"min":float(sd.min()),"p50":float(np.median(sd)),"max":float(sd.max()),"mean":float(sd.mean())}},
          open('/local/home/zegertho/agent/state/losvar/artifacts/v05_sensitivity_matrix.json','w'),indent=1,default=float)
print("\nwrote v05_sensitivity_matrix.json")
