"""D re-scoped (parent instruction): ONE descriptive number at the fitted offsets. NOT a criterion."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, json
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.features import bigram_features_from_positions
from keybo.analysis.timecard import _load_gz_model
from keybo.scoring.model_scorer import predict_ms
G=ROW_STAGGERED_30; slots=list(G.slots); idx={p:i for i,p in enumerate(slots)}
mir=lambda p:(-p[0],p[1]); pairs=[(a,b) for a in slots for b in slots if a!=b]
models=[_load_gz_model(f"bigram_reg31_seed{s}") for s in (0,1,2)]
def stats(off):
    g=Geometry(slots=G.slots,row_offsets=dict(off))
    vecs=np.vstack([bigram_features_from_positions(g,(a,b),wpm=90.0) for a in slots for b in slots])
    T2=np.mean([predict_ms(m,vecs).reshape(30,30) for m in models],axis=0)
    a=np.array([abs(T2[idx[x],idx[y]]-T2[idx[mir(x)],idx[mir(y)]]) for x,y in pairs])
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(a.max()),
                n_gt1=int((a>1).sum()),n_gt5=int((a>5).sum()))
ship=stats({1:0.50,2:0.0,3:-0.25})
fit =stats({1:0.00,2:0.0,3:+0.25})        # B-grid pooled argmin (top +0.25, bottom 0.0)
fits=stats({1:0.50,2:0.0,3:-0.25,0:0.125})# space-axis argmin, letters shipped
print("=== D (DESCRIPTIVE ONLY — explicitly NOT a success criterion, per parent re-scope) ===")
for lbl,s in (("SHIPPED (-0.25/0.0/+0.50)",ship),
              ("B-grid pooled argmin (top +0.25, bottom 0.00)",fit),
              ("space-axis argmin (letters shipped, off_space +0.125)",fits)):
    r=s["mean"]/ship["mean"]
    print(f"  {lbl:52s} mean {s['mean']:.4f} ({r:5.3f}x)  median {s['median']:.4f}  p90 {s['p90']:.4f}  "
          f"max {s['max']:8.4f}  >1ms {s['n_gt1']:3d}  >5ms {s['n_gt5']:3d}")
json.dump({"shipped":ship,"b_argmin":fit,"space_argmin":fits},open("inv_d_fitted.json","w"),indent=1)
print("\n  NOTE: the space-axis row is IDENTICAL to shipped by construction — the 870-pair asymmetry")
print("  universe excludes space, so it is structurally blind to off_space (my A2/D4 result).")
