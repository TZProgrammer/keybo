"""INVARIANT B — fit the 2 identifiable offsets by leave-one-layout-out held-out error.

Registered design (ROWOFFSETS-prereg.md B1-B5):
  home PINNED at 0.0; grid over (off_top=y3, off_bottom=y1), dyadic step 0.125.
  Shipped point is (off_top=-0.25, off_bottom=+0.50) and is ON the grid.
  Per-fold argmin = per-fold estimate; pooled = argmin of summed per-fold held-out wmae.

CRITICAL, and the reason this script does not call validate(geometry=...):
  validate() NEVER forwards `geometry` to train_fn -> it would train on shipped offsets and
  evaluate on the candidate = train/serve SKEW (VERIFIED: 14.13 skew vs 13.41 honest on the
  dvorak fold, a +0.72 artifact = 5x my 0.135 bar). So I train AND evaluate under the same
  candidate geometry, replicating validate()'s fold logic exactly.
"""
import os, sys, json, pickle, time, itertools
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np
import keybo
assert keybo.__file__.startswith("/local/home/zegertho/repos/keybo/src"), keybo.__file__
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.training.train import train_bigram_model
from keybo.training.validate import (leave_one_layout_out, build_cells, _predict_cells,
                                     weighted_mae, uniform_mae, _centered_spearman,
                                     _per_bucket_rho, layout_ranking_tau,
                                     aggregate_layout_table)

ROWS = pickle.load(open("/tmp/stagger-work/bi_rows.pkl","rb"))
LAYOUTS = sorted({r.layout for r in ROWS})
SEEDS = [0,1,2]
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
SHIPPED = (-0.25, 0.50)     # (off_top y=3, off_bottom y=1)

def geom(off_top, off_bot, space=None):
    d = {1: off_bot, 2: 0.0, 3: off_top}
    if space is not None: d[0] = space
    return Geometry(slots=ROW_STAGGERED_30.slots, row_offsets=d)

# pre-split folds + cells once (cells do NOT depend on geometry)
FOLDS = {}
for h in LAYOUTS:
    tr, te = leave_one_layout_out(ROWS, h)
    FOLDS[h] = (tr, build_cells(te, **CELL_KW))
    print(f"fold {h}: {len(tr)} train rows, {len(FOLDS[h][1])} test cells", flush=True)

CACHE = {}
def evaluate(off_top, off_bot, space=None, seeds=SEEDS):
    """Train AND eval under the SAME geometry. Returns per-(fold,seed) metrics."""
    key = (off_top, off_bot, space, tuple(seeds))
    if key in CACHE: return CACHE[key]
    g = geom(off_top, off_bot, space)
    out = {}
    for h, (tr, test_cells) in FOLDS.items():
        obs = np.array([c.obs for c in test_cells])
        per = []
        for s in seeds:
            m = train_bigram_model(tr, target_wpm=(CELL_KW["wpm_lo"]+CELL_KW["wpm_hi"])/2,
                                   geometry=g, random_state=s, n_jobs=1)
            pred = _predict_cells(m, test_cells, g)
            per.append({"seed": s,
                        "wmae": weighted_mae(test_cells, pred, obs),
                        "umae": uniform_mae(pred, obs),
                        "rho":  _centered_spearman(test_cells, pred, obs),
                        "bucket_rhos": {str(k): v for k, v in _per_bucket_rho(test_cells, pred, obs).items()}})
        out[h] = per
    CACHE[key] = out
    return out

def pooled_wmae(res):
    """Mean over folds of the fold's seed-mean wmae (equal fold weight)."""
    return float(np.mean([np.mean([p["wmae"] for p in res[h]]) for h in res]))

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv)>1 else "coarse"
    t0=time.time()
    if stage == "coarse":
        # STAGE 1: coarse dyadic scan at step 0.25 over the registered box, seed 0 only,
        # to locate the basin cheaply. The registered 0.125 grid is refined in stage 2.
        tops = [round(x,4) for x in np.arange(-1.00, 0.5001, 0.25)]
        bots = [round(x,4) for x in np.arange(-0.50, 1.0001, 0.25)]
        grid = {}
        for ot, ob in itertools.product(tops, bots):
            r = evaluate(ot, ob, seeds=[0])
            grid[f"{ot},{ob}"] = {"pooled_wmae": pooled_wmae(r),
                                  "per_fold": {h: r[h][0]["wmae"] for h in r}}
            print(f"  top={ot:+.3f} bot={ob:+.3f}  pooled_wmae={grid[f'{ot},{ob}']['pooled_wmae']:.6f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump({"stage":"coarse","cell_kw":CELL_KW,"seeds":[0],"shipped":SHIPPED,"grid":grid},
                  open("/tmp/stagger-work/inv_b_coarse.json","w"), indent=1)
        best = min(grid, key=lambda k: grid[k]["pooled_wmae"])
        print(f"\ncoarse argmin: {best}  pooled_wmae={grid[best]['pooled_wmae']:.6f}")
        print(f"shipped {SHIPPED}: pooled_wmae={grid[f'{SHIPPED[0]},{SHIPPED[1]}']['pooled_wmae']:.6f}")
    print(f"total {time.time()-t0:.0f}s")
