import os,sys,json,pickle,itertools
from collections import defaultdict
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"48")
WT="/local/home/zegertho/repos/keybo-wt-losvar"; sys.path.insert(0,WT+"/src")
import numpy as np, keybo
assert keybo.__file__.startswith(WT+"/")
CACHE="/local/home/zegertho/agent/workspaces/losvar/cache/v02_folds.pkl"
folds=pickle.loads(open(CACHE,'rb').read())
HOLD=("azerty","dvorak","qwerty","qwertz"); SEEDS=[0,1,2]; SB=80
# per-layout LEVEL on its own held-out support (the cohort/practice offset)
lvl={}
for la in HOLD:
    tr=tuple(sorted(set(HOLD)-{la}))
    for s in SEEDS:
        f=folds[(tr,la,s)]
        sel=f["bucket"]==SB
        w=f["n"][sel]; r=(f["obs"]-f["pred"])[sel]
        lvl[(la,s)]=float((r*w).sum()/w.sum())
print("PER-LAYOUT held-out LEVEL (obs-pred, n-weighted, scoring bucket):")
for la in HOLD:
    print(f"  {la:8s}: " + "  ".join(f"seed{s} {lvl[(la,s)]:+7.3f}" for s in SEEDS))
d2=json.load(open('/local/home/zegertho/agent/state/losvar/artifacts/v02_sigma_and_flips.json'))
print("\nROUTE b2 RAW vs DE-LEVELLED (level = each layout's own held-out offset):")
raw_f=[]; dl_f=[]; raw_e=[]; dl_e=[]
for k,rec in d2["observed_flip_rate_route_b2"]["pairs"].items():
    a,b=k.split("|")
    for r in rec["per_seed"]:
        s=r["seed"]; om=r["obs_margin"]; pm=r["pred_margin"]
        # remove each layout's own level from the OBSERVED side: what remains is structure
        om_dl = om - (lvl[(a,s)]-lvl[(b,s)])
        raw_f.append(np.sign(om)!=np.sign(pm)); dl_f.append(np.sign(om_dl)!=np.sign(pm))
        raw_e.append(abs(om-pm)); dl_e.append(abs(om_dl-pm))
    print(f"  {k:18s} pred {rec['mean_pred_margin']:+7.3f}  obs {rec['mean_obs_margin']:+7.3f}  "
          f"obs_delev {np.mean([r['obs_margin']-(lvl[(a,r['seed'])]-lvl[(b,r['seed'])]) for r in rec['per_seed']]):+7.3f}")
print(f"\nRAW        : flip rate {np.mean(raw_f):.3f} ({sum(raw_f)}/{len(raw_f)})  |err| rms {np.sqrt(np.mean(np.square(raw_e))):.4f}")
print(f"DE-LEVELLED: flip rate {np.mean(dl_f):.3f} ({sum(dl_f)}/{len(dl_f)})  |err| rms {np.sqrt(np.mean(np.square(dl_e))):.4f}")
# how much of the raw error IS the level difference?
lv=[abs(lvl[(k.split('|')[0],r['seed'])]-lvl[(k.split('|')[1],r['seed'])])
    for k,rec in d2["observed_flip_rate_route_b2"]["pairs"].items() for r in rec["per_seed"]]
print(f"\n|level_a - level_b| rms = {np.sqrt(np.mean(np.square(lv))):.4f} ms/char "
      f"(vs raw margin |err| rms {np.sqrt(np.mean(np.square(raw_e))):.4f})")
json.dump({"per_layout_level":{f"{a}/{s}":v for (a,s),v in lvl.items()},
  "raw_flip_rate":float(np.mean(raw_f)),"delevelled_flip_rate":float(np.mean(dl_f)),
  "raw_err_rms":float(np.sqrt(np.mean(np.square(raw_e)))),
  "delevelled_err_rms":float(np.sqrt(np.mean(np.square(dl_e)))),
  "level_diff_rms":float(np.sqrt(np.mean(np.square(lv)))),"n_pair_seed":len(raw_f)},
  open('/local/home/zegertho/agent/state/losvar/artifacts/v04_b2_cohort_diag.json','w'),indent=1,default=float)
print("\nwrote v04_b2_cohort_diag.json")
