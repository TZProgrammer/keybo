"""INVARIANT C — held-out effect of a candidate offset vector, paired per-fold (MOR-FIX-1),
plus the MANDATORY same-width placebo (prereg C4) and the high-wpm gate (prereg C3c).

Runs a small set of NAMED arms at the FULL 4 folds x 3 seeds, honestly (train AND eval under
the same geometry -- validate()'s geometry= is eval-only, VERIFIED).

Arms:
  SHIPPED     (-0.25, 0.0, +0.50)                      the incumbent / baseline
  PLACEBO     (+0.25, +0.5, +1.00) = shipped + 0.5     ZERO new information on letter pairs
                                                        (bit-identical), so its whole effect is
                                                        the space channel + refit noise.
  ZERO        (0, 0, 0)                                 no stagger at all (the D argmin)
  FITTED      whatever the coarse+refine grid selects   filled in from inv_b results
Also: a SEED-ONLY replicate of SHIPPED (seeds 3,4,5) to measure the instrument's own noise, so
"is the candidate's delta bigger than re-running the SAME geometry?" is answerable.
"""
import os, sys, json, pickle, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np
import keybo
assert keybo.__file__.startswith("/local/home/zegertho/repos/keybo/src"), keybo.__file__
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.training.train import train_bigram_model
from keybo.training.validate import (leave_one_layout_out, build_cells, _predict_cells,
                                     weighted_mae, uniform_mae, _centered_spearman,
                                     _per_bucket_rho, split_half_ceiling)
from keybo.verdicts import bucket_regression_report

ROWS=pickle.load(open("/tmp/stagger-work/bi_rows.pkl","rb"))
LAYOUTS=sorted({r.layout for r in ROWS}); CELL_KW=dict(wpm_lo=40,wpm_hi=140,bucket_width=20,min_cell_samples=10)
FOLDS={}
for h in LAYOUTS:
    tr,te=leave_one_layout_out(ROWS,h); FOLDS[h]=(tr,build_cells(te,**CELL_KW))

def geom(d): return Geometry(slots=ROW_STAGGERED_30.slots, row_offsets=dict(d))

def run_arm(offsets, seeds, label):
    g=geom(offsets); out={}
    for h,(tr,test_cells) in FOLDS.items():
        obs=np.array([c.obs for c in test_cells]); per=[]
        for s in seeds:
            m=train_bigram_model(tr,target_wpm=90,geometry=g,random_state=s,n_jobs=1)
            pred=_predict_cells(m,test_cells,g)
            br=_per_bucket_rho(test_cells,pred,obs)
            per.append({"seed":s,"wmae":weighted_mae(test_cells,pred,obs),
                        "umae":uniform_mae(pred,obs),"rho":_centered_spearman(test_cells,pred,obs),
                        "bucket_rhos":{str(k):v for k,v in br.items()}})
        out[h]=per
        print(f"  [{label}] {h}: wmae " + " ".join(f"{p['wmae']:.4f}" for p in per), flush=True)
    return out

def paired_delta(cand, base):
    """MOR-FIX-1: mean of PER-FOLD PER-SEED differences (candidate - baseline), same fold+seed."""
    ds=[]; per_fold={}
    for h in cand:
        pf=[c["wmae"]-b["wmae"] for c,b in zip(cand[h],base[h])]
        per_fold[h]={"per_seed_delta":pf,"mean_delta":float(np.mean(pf))}
        ds += pf
    return {"mean_paired_delta":float(np.mean(ds)),"sd":float(np.std(ds,ddof=1)),
            "n":len(ds),"per_fold":per_fold,
            "folds_improving":int(sum(1 for h in per_fold if per_fold[h]["mean_delta"]<0)),
            "n_folds":len(per_fold)}

if __name__=="__main__":
    which=sys.argv[1]
    t0=time.time(); res={}
    ARMS={
      "SHIPPED":  ({1:0.50,2:0.0,3:-0.25},[0,1,2]),
      "PLACEBO":  ({1:1.00,2:0.5,3:+0.25},[0,1,2]),   # shipped + 0.5 uniform: bit-identical on letters
      "ZERO":     ({1:0.00,2:0.0,3: 0.00},[0,1,2]),
      "SEEDNOISE":({1:0.50,2:0.0,3:-0.25},[3,4,5]),   # SAME geometry, different seeds
    }
    if which=="all":
        for lbl,(off,seeds) in ARMS.items():
            print(f"\n=== arm {lbl} offsets={off} seeds={seeds} ===",flush=True)
            res[lbl]={"offsets":off,"seeds":seeds,"folds":run_arm(off,seeds,lbl)}
        # ceilings once (geometry-independent)
        res["ceilings"]={h:split_half_ceiling(FOLDS[h][0] and [r for r in ROWS if r.layout==h],
                                              n_boot=50,seed=0,**CELL_KW) for h in LAYOUTS}
        base=res["SHIPPED"]["folds"]
        res["deltas"]={k:paired_delta(res[k]["folds"],base) for k in ("PLACEBO","ZERO","SEEDNOISE")}
        json.dump(res,open("/tmp/stagger-work/inv_c_arms.json","w"),indent=1)
        print(f"\n=== PAIRED PER-FOLD DELTAS vs SHIPPED (ms/char wmae; negative = better) ===")
        for k,d in res["deltas"].items():
            print(f"  {k:10s} mean_paired_delta {d['mean_paired_delta']:+.6f}  sd {d['sd']:.6f}  "
                  f"folds improving {d['folds_improving']}/{d['n_folds']}")
            for h,v in d["per_fold"].items():
                print(f"      {h:8s} {v['mean_delta']:+.6f}   per-seed " + " ".join(f"{x:+.4f}" for x in v["per_seed_delta"]))
        print(f"\ntotal {time.time()-t0:.0f}s")
