"""Verify the validate() train/eval geometry SKEW bug, and time one fold."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import time, numpy as np
from keybo.data.strokes import load_strokes
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.training.train import train_bigram_model
from keybo.training.validate import leave_one_layout_out, build_cells, _predict_cells, weighted_mae

BI="/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
t0=time.time(); rows=load_strokes(BI, ngram_len=2, wpm_threshold=0, min_samples=1)
print(f"loaded {len(rows)} rows in {time.time()-t0:.1f}s  (CLI settings: wpm_threshold=0,min_samples=1)")

WILD = Geometry(slots=ROW_STAGGERED_30.slots, row_offsets={1:-3.0, 2:0.0, 3:+3.0})
tr, te = leave_one_layout_out(rows, "dvorak")
kw = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
test_cells = build_cells(te, **kw)
print(f"dvorak fold: {len(tr)} train rows, {len(test_cells)} test cells")

# ---- does geometry reach TRAINING? train two models, one per geometry, compare predictions
t0=time.time()
m_ship = train_bigram_model(tr, target_wpm=90, random_state=0, n_jobs=1)
t_fit = time.time()-t0
print(f"one train_bigram_model fit: {t_fit:.1f}s")
t0=time.time()
m_wild = train_bigram_model(tr, target_wpm=90, geometry=WILD, random_state=0, n_jobs=1)
print(f"second fit: {time.time()-t0:.1f}s")

obs=np.array([c.obs for c in test_cells])
p_ship_ship = _predict_cells(m_ship, test_cells, ROW_STAGGERED_30)
p_ship_wild = _predict_cells(m_ship, test_cells, WILD)      # <-- what validate(geometry=WILD) ACTUALLY does
p_wild_wild = _predict_cells(m_wild, test_cells, WILD)      # <-- what it SHOULD do
print(f"\nwmae  train=ship eval=ship : {weighted_mae(test_cells,p_ship_ship,obs):.6f}")
print(f"wmae  train=ship eval=WILD : {weighted_mae(test_cells,p_ship_wild,obs):.6f}   <-- validate(geometry=WILD) today")
print(f"wmae  train=WILD eval=WILD : {weighted_mae(test_cells,p_wild_wild,obs):.6f}   <-- the honest A/B")
print(f"\nmodels differ? max|pred_ship_model - pred_wild_model| on same eval geom = "
      f"{np.abs(p_ship_wild-p_wild_wild).max():.6f}  => geometry DOES change training when passed")
print("=> CONFIRMED: validate() never forwards `geometry` to train_fn (line ~787). "
      "The documented `geometry=` parameter is EVAL-ONLY.")
# workaround check: does train_params carry it through?
print("\nworkaround: pass geometry inside train_params (forwarded via **params)")
