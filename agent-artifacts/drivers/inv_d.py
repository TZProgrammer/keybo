"""INVARIANT D — mirror asymmetry as a FUNCTION of row_offsets, at SCORING time.

Replicates the sibling `mirror`'s p2 measurement exactly (same shipped k31 models, same 870
ordered distinct pairs, same wpm=90, same seed-mean T2), then re-runs it under alternative
row_offsets. Uses ~/repos/keybo (main) -- I do NOT touch mirror's worktree.

⚠ SCOPE, stated because it changes the reading: the shipped k31 models were TRAINED under the
shipped offsets. Substituting offsets here changes only the FEATURES fed at scoring time, so
this measures the model's SENSITIVITY surface, not a refit. That is exactly the right object
for D's question ("would a mis-specified stagger MANUFACTURE apparent asymmetry?") -- but it
is NOT a claim about a refitted model.
"""
import os, json, itertools
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import numpy as np
import keybo
assert keybo.__file__.startswith("/local/home/zegertho/repos/keybo/src"), keybo.__file__
from keybo.analysis.timecard import _load_gz_model
from keybo.features import bigram_features_from_positions
from keybo.features.ngram import _placement_row_from_positions
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.scoring.model_scorer import predict_ms

G = ROW_STAGGERED_30
WPM = 90.0
slots = list(G.slots); idx = {p:i for i,p in enumerate(slots)}
mir = lambda p: (-p[0], p[1])
pairs = [(a,b) for a in slots for b in slots if a != b]
print(f"{len(pairs)} ordered distinct pairs")

models = [_load_gz_model(f"bigram_reg31_seed{s}") for s in (0,1,2)]
print(f"loaded {len(models)} shipped models; stamp={models[0].metadata.feature_version}")

def geom(off_top, off_bot, home=0.0, space=None):
    d = {1: off_bot, 2: home, 3: off_top}
    if space is not None: d[0]=space
    return Geometry(slots=G.slots, row_offsets=d)

def asym_stats(g, label):
    vecs = np.vstack([bigram_features_from_positions(g,(a,b),wpm=WPM) for a in slots for b in slots])
    T2 = np.mean([predict_ms(m, vecs).reshape(30,30) for m in models], axis=0)
    a = np.array([abs(T2[idx[x],idx[y]] - T2[idx[mir(x)],idx[mir(y)]]) for x,y in pairs])
    # how many feature ROWS change under mirroring at this geometry
    nchg = 0
    for x,y in pairs:
        r0 = _placement_row_from_positions(g,x,y); r1 = _placement_row_from_positions(g,mir(x),mir(y))
        if any(abs(r0[k]-r1[k])>1e-9 for k in r0): nchg += 1
    return {"label":label, "row_offsets":dict(g.row_offsets),
            "mean":float(a.mean()), "median":float(np.median(a)),
            "p90":float(np.percentile(a,90)), "max":float(a.max()),
            "n_gt1":int((a>1).sum()), "n_gt5":int((a>5).sum()),
            "rows_changed_under_mirror":nchg,
            "T2_range":[float(T2.min()),float(T2.max())]}

out=[]
print("\n=== D: reproduce mirror's baseline, then sweep ===")
base = asym_stats(G, "SHIPPED (-0.25/0.0/+0.50)")
out.append(base)
print(f"  SHIPPED           mean {base['mean']:.4f} median {base['median']:.4f} p90 {base['p90']:.4f} "
      f"max {base['max']:.4f} >1ms {base['n_gt1']} >5ms {base['n_gt5']} rows_chg {base['rows_changed_under_mirror']}")
print(f"    (mirror/parent reported mean 1.9624 median 0.0739 p90 5.8208 max 42.2665 >1ms 238 >5ms 112)")

for lbl,(ot,ob) in [("ZERO stagger",(0.0,0.0)),
                    ("half shipped",(-0.125,0.25)),
                    ("double shipped",(-0.5,1.0)),
                    ("uniform+0.5 (PLACEBO)",(0.25,1.0)),
                    ("ANSI true (top -0.25, bot +0.5) == shipped",(-0.25,0.5)),
                    ("sign-flipped",(0.25,-0.5)),
                    ("top only",(-0.25,0.0)),
                    ("bottom only",(0.0,0.5))]:
    home = 0.5 if lbl.startswith("uniform") else 0.0
    s = asym_stats(geom(ot,ob,home=home), lbl)
    out.append(s)
    r = s["mean"]/base["mean"] if base["mean"] else float("nan")
    print(f"  {lbl:42s} mean {s['mean']:.4f} ({r:5.3f}x) max {s['max']:8.4f} rows_chg {s['rows_changed_under_mirror']:4d}")

json.dump(out, open("/tmp/stagger-work/inv_d_sweep.json","w"), indent=1)
print("\nwrote inv_d_sweep.json")

# --- the analytic point: asymmetry vs |stagger magnitude| ------------------------------
print("\n=== D: asymmetry as a function of stagger MAGNITUDE (scale the shipped vector) ===")
scal=[]
for k in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    s = asym_stats(geom(-0.25*k, 0.5*k), f"scale={k}")
    scal.append({"scale":k, **{q:s[q] for q in ("mean","median","p90","max","rows_changed_under_mirror")}})
    print(f"  scale={k:4.3f}  mean {s['mean']:8.4f}  max {s['max']:9.4f}  rows_chg {s['rows_changed_under_mirror']:4d}")
json.dump(scal, open("/tmp/stagger-work/inv_d_scale.json","w"), indent=1)
print("wrote inv_d_scale.json")
