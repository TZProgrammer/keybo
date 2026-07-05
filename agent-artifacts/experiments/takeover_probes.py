"""Take-over audit probes 3+4 (fable-audit2 died to auth churn; parent runs its brief).

3. W's effective sample size + weight shares — is R1W's edge over R1 (.931 vs .928) even
   supported by a meaningful reweighting, or is W ~a no-op?
4. Shrinkage sensitivity: does R1's transfer edge survive k=10 vs k=1000 (dvorak fold,
   the hardest one, seed 0)? Pre-registered k was 100, pulled from air — check robustness.
"""

import json
import time
from collections import defaultdict

import numpy as np

from keybo.data.strokes import iqr_average, load_strokes
from keybo.features import bigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
from keybo.geometry import ROW_STAGGERED_30
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.training.validate import _centered_spearman, _predict_cells, build_cells

t0 = time.time()
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)

rows = load_strokes("bistrokes_v3.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
geom = ROW_STAGGERED_30
feats, targets, ex_ngram, ex_layout, ex_n = [], [], [], [], []
for row in rows:
    by_wpm = defaultdict(list)
    for wpm, duration, _pid, _hold in row.samples:
        by_wpm[wpm].append(duration)
    for wpm, durations in by_wpm.items():
        feats.append(bigram_features_from_positions(geom, row.positions, freq=1.0, wpm=wpm))
        targets.append(iqr_average(durations))
        ex_ngram.append(row.ngram)
        ex_layout.append(row.layout)
        ex_n.append(len(durations))
X = np.vstack(feats)
y = np.array(targets)
ex_ngram = np.array(ex_ngram, dtype=object)
ex_layout = np.array(ex_layout, dtype=object)
ex_n = np.array(ex_n)
print(f"[{time.time()-t0:.0f}s] {len(y)} examples", flush=True)

# --- probe 3: ESS of the W weighting --------------------------------------------------
share = defaultdict(float)
for la in ex_layout:
    share[la] += 1
total = len(ex_layout)
w = np.array([min(50.0, total / (len(share) * share[la])) for la in ex_layout])
w = w / w.mean()
ess = w.sum() ** 2 / (w * w).sum()
by_lay = defaultdict(float)
for la, wi in zip(ex_layout, w):
    by_lay[la] += wi
print(f"ESS {ess:.0f} / {total} ({ess/total:.1%})")
print("weight share:", {k: f"{v / w.sum():.1%}" for k, v in sorted(by_lay.items())})
raw_share = {k: f"{v / total:.1%}" for k, v in sorted(share.items())}
print("raw example share:", raw_share)

# --- probe 4: k sensitivity on the dvorak fold (seed 0) --------------------------------
def fit(Xm, ym, seed):
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=BIGRAM_FEATURE_NAMES,
        wpm_range=(60, 120),
        ngram="bigram",
    )
    m = XGBoostTypingModel(meta, random_state=seed, n_jobs=0)
    m._regressor.fit(Xm, ym)
    m._fitted = True
    return m


mask = ex_layout != "dvorak"
Xtr, ytr, ngtr, wtr = X[mask], y[mask], ex_ngram[mask], ex_n[mask]
test_cells = [c for c in build_cells(rows, **CELL_KW) if c.layout == "dvorak"]
obs = np.array([c.obs for c in test_cells])

for k in (10.0, 100.0, 1000.0):
    model = fit(Xtr, ytr, 0)
    bmap = {}
    for _ in range(2):
        gpred = model.predict(Xtr)
        num, den = defaultdict(float), defaultdict(float)
        for ng, r, wi in zip(ngtr, ytr - gpred, wtr):
            num[ng] += wi * r
            den[ng] += wi
        bmap = {ng: num[ng] / (den[ng] + k) for ng in num}
        bvec = np.array([bmap.get(ng, 0.0) for ng in ngtr])
        model = fit(Xtr, ytr - bvec, 0)
    pred = _predict_cells(model, test_cells, geom) + np.array(
        [bmap.get(c.ngram, 0.0) for c in test_cells]
    )
    rho = _centered_spearman(test_cells, pred, obs)
    print(f"k={k:6.0f}: dvorak rho {rho:+.3f} (frac of .669 ceiling: {rho/0.669:+.3f})")
print(f"[{time.time()-t0:.0f}s] DONE (reference: R1 k=100 seed0 dvorak rho was +0.562-ish under R1W; B was +0.310)")
