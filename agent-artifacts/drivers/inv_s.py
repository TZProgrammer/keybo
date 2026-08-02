"""ADDENDUM 1 (space axis) — 1-D LOLO scan of row_offsets[0] at the SHIPPED letter offsets.

Orthogonal to the letter block by A4 (setting row_offsets[0] touches NO letter-letter pair,
max|d| = 0.0e+00), so this is independent of the 7x7 grid.
Trains AND evaluates under the same geometry -- never validate(geometry=) (VALIDATE-GEOM-1).
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
                                     weighted_mae, uniform_mae, _centered_spearman, _per_bucket_rho)

ROWS=pickle.load(open("/tmp/stagger-work/bi_rows.pkl","rb"))
LAYOUTS=sorted({r.layout for r in ROWS})
CELL_KW=dict(wpm_lo=40,wpm_hi=140,bucket_width=20,min_cell_samples=10)
SHIPPED_LETTERS={1:0.50,2:0.0,3:-0.25}
FOLDS={}
for h in LAYOUTS:
    tr,te=leave_one_layout_out(ROWS,h); FOLDS[h]=(tr,build_cells(te,**CELL_KW))
    print(f"fold {h}: {len(tr)} train rows, {len(FOLDS[h][1])} test cells",flush=True)

def run(off_space, seeds):
    d=dict(SHIPPED_LETTERS)
    if off_space is not None: d[0]=off_space
    g=Geometry(slots=ROW_STAGGERED_30.slots, row_offsets=d)
    out={}
    for h,(tr,test_cells) in FOLDS.items():
        obs=np.array([c.obs for c in test_cells]); per=[]
        for s in seeds:
            m=train_bigram_model(tr,target_wpm=90,geometry=g,random_state=s,n_jobs=1)
            pred=_predict_cells(m,test_cells,g)
            per.append({"seed":s,"wmae":weighted_mae(test_cells,pred,obs),
                        "umae":uniform_mae(pred,obs),"rho":_centered_spearman(test_cells,pred,obs),
                        "bucket_rhos":{str(k):v for k,v in _per_bucket_rho(test_cells,pred,obs).items()}})
        out[h]=per
    return out

if __name__=="__main__":
    seeds=[int(x) for x in (sys.argv[1].split(",") if len(sys.argv)>1 else ["0"])]
    t0=time.time(); grid={}
    vals=[round(v,4) for v in np.arange(-1.0,1.0001,0.125)]
    for v in vals:
        r=run(v,seeds)
        pooled=float(np.mean([np.mean([p["wmae"] for p in r[h]]) for h in r]))
        grid[str(v)]={"pooled_wmae":pooled,
                      "per_fold_mean":{h:float(np.mean([p["wmae"] for p in r[h]])) for h in r},
                      "per_fold_seeds":{h:[p["wmae"] for p in r[h]] for h in r},
                      "bucket_rhos":{h:[p["bucket_rhos"] for p in r[h]] for h in r}}
        print(f"  off_space={v:+.4f}  pooled_wmae={pooled:.6f}  [{time.time()-t0:.0f}s]",flush=True)
    json.dump({"stage":"space-1d","seeds":seeds,"cell_kw":CELL_KW,
               "shipped_letters":SHIPPED_LETTERS,"grid":grid},
              open("/tmp/stagger-work/inv_s_scan.json","w"),indent=1)
    best=min(grid,key=lambda k:grid[k]["pooled_wmae"])
    print(f"\nargmin off_space={best} pooled_wmae={grid[best]['pooled_wmae']:.6f}")
    print(f"shipped implicit 0.0: pooled_wmae={grid['0.0']['pooled_wmae']:.6f}")
    print(f"delta(argmin - shipped) = {grid[best]['pooled_wmae']-grid['0.0']['pooled_wmae']:+.6f} ms/char")
    print(f"total {time.time()-t0:.0f}s")
