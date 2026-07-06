"""Conditioned-target arm matrix (user challenge #4): the trigram model's real job.

Target = press2->press3 (the context-conditioned second bigram). The full-span target's
LOLO advantage was an artifact — redundant bigram-sum variance — so model selection now
happens HERE: which architecture / feature set / context signal minimizes LOLO error on
the quantity the objective actually needs?

Data construction: tristrokes_v1 (full-span) and tristrokes_last (conditioned) come from
the SAME extraction pass, so their (layout, positions, ngram) rows carry samples in the
same order. Joining them per-sample yields prev = full - last = the ACTUAL t(bg1) of this
very occurrence — the sharpest conceivable local-context signal (one real interval back,
same trigram). The join is verified per-sample ((wpm, pid, hold) must match; abort if
mismatch > 0.1%).

Arms (shipped R1W recipe — practice term + layout weights — conditioned target, shared
folds and cells):
  C-BASE     trigram features, depth 3 (anchor: must reproduce lolo_trigram_last ~1.043)
  C-D2/C-D4  depth sweep
  C-PREV     + prev-duration feature (40ms buckets, capped 400) — teacher-forced at eval
  C-PREV-D2  combo

Metrics: LOLO rho/own-ceiling + tau + beats dist-sum baseline, same construction as the
harness. Rule + consequences preregistered (PREREGISTRATIONS 2026-07-06).
"""

import json
import time
from collections import defaultdict

import numpy as np

from keybo.data.strokes import iqr_average, load_strokes
from keybo.features import trigram_features_from_positions
from keybo.features.schema import FEATURE_VERSION, TRIGRAM_FEATURE_NAMES
from keybo.geometry import ROW_STAGGERED_30
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.training.train import fit_practice_term, layout_balance_weights
from keybo.training.validate import (
    Cell,
    _centered_spearman,
    aggregate_layout_table,
    layout_ranking_tau,
    split_half_ceiling,
)

SEEDS = [0, 1, 2]
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
K = 100.0
geom = ROW_STAGGERED_30
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --- join the two tables per sample -------------------------------------------------------
full_rows = {(r.layout, r.positions, r.ngram): r for r in load_strokes(
    "tristrokes_v1.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)}
last_rows = {(r.layout, r.positions, r.ngram): r for r in load_strokes(
    "tristrokes_last.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)}
log(f"{len(full_rows)} full rows, {len(last_rows)} last rows")

joined = []  # (layout, positions, ngram, [(wpm, dur_cond, prev, pid, hold)])
n_samples = n_mismatch = 0
for key, lr in last_rows.items():
    fr = full_rows.get(key)
    if fr is None or len(fr.samples) != len(lr.samples):
        n_mismatch += len(lr.samples) if fr is None else abs(len(fr.samples) - len(lr.samples))
        continue
    samples = []
    for (wf, df, pf, hf), (wl, dl, pl, hl) in zip(fr.samples, lr.samples):
        n_samples += 1
        if (wf, pf, hf) != (wl, pl, hl):
            n_mismatch += 1
            continue
        prev = df - dl
        if prev < 0 or prev > 5000:
            continue
        samples.append((wl, dl, prev, pl, hl))
    if samples:
        joined.append((key[0], key[1], key[2], samples))
mismatch_rate = n_mismatch / max(n_samples, 1)
log(f"join: {n_samples} samples, mismatch rate {mismatch_rate:.4%}")
assert mismatch_rate < 0.001, f"join integrity failed: {mismatch_rate:.2%}"

# --- training examples: group by (row, wpm) as the shipped recipe does; prev bucketed ----
PREV_CAP = 400.0
PREV_BUCKET = 40.0


def prev_bucketed(p):
    return min(PREV_CAP, (p // PREV_BUCKET) * PREV_BUCKET)


feats, targets, ex_ngram, ex_layout, ex_n, ex_prev = [], [], [], [], [], []
for layout, positions, ngram, samples in joined:
    by_group = defaultdict(list)
    for wpm, dur, prev, pid, hold in samples:
        by_group[(wpm, prev_bucketed(prev))].append((dur, prev))
    for (wpm, pb), items in by_group.items():
        feats.append(trigram_features_from_positions(geom, positions, wpm=wpm))
        targets.append(iqr_average([d for d, _ in items]))
        ex_ngram.append(ngram)
        ex_layout.append(layout)
        ex_n.append(len(items))
        ex_prev.append(float(np.mean([p for _, p in items])))
X_base = np.vstack(feats)
y = np.array(targets)
ex_ngram = np.array(ex_ngram, dtype=object)
ex_layout = np.array(ex_layout, dtype=object)
ex_n = np.array(ex_n)
ex_prev = np.array(ex_prev)
log(f"{len(y)} conditioned training examples")

# --- eval cells: conditioned target, session-wpm buckets (prev enters as feature only) ---
# Build StrokeRow-like cells directly from joined samples.
cells = []
cell_prev = []  # mean actual prev per cell (teacher-forced eval feature)
for layout, positions, ngram, samples in joined:
    by_bucket = defaultdict(list)
    for wpm, dur, prev, pid, hold in samples:
        if not CELL_KW["wpm_lo"] <= wpm < CELL_KW["wpm_hi"]:
            continue
        b = CELL_KW["wpm_lo"] + ((wpm - CELL_KW["wpm_lo"]) // CELL_KW["bucket_width"]) * CELL_KW["bucket_width"]
        by_bucket[b].append((wpm, dur, prev, pid, hold))
    for b, items in by_bucket.items():
        if len(items) < CELL_KW["min_cell_samples"]:
            continue
        cells.append(
            Cell(
                layout=layout, ngram=ngram, positions=positions, frequency=len(samples),
                bucket=b, wpm=b + CELL_KW["bucket_width"] / 2,
                obs=iqr_average([d for _, d, _, _, _ in items]), n=len(items),
                samples=[(w, d, p, h) for w, d, _, p, h in items],
            )
        )
        cell_prev.append(float(np.mean([pv for _, _, pv, _, _ in items])))
cell_prev = np.array(cell_prev)
log(f"{len(cells)} eval cells")
obs_table = aggregate_layout_table(cells)
LAYOUTS = sorted({c.layout for c in cells})

# Ceilings for the conditioned target (per layout, participant split-half).
from keybo.data.strokes import StrokeRow

ceil_rows = defaultdict(list)
for layout, positions, ngram, samples in joined:
    ceil_rows[layout].append(
        StrokeRow(layout=layout, positions=positions, ngram=ngram, frequency=len(samples),
                  samples=[(w, d, p, h) for w, d, _, p, h in samples])
    )
CEILINGS = {}
for lay in LAYOUTS:
    CEILINGS[lay] = split_half_ceiling(ceil_rows[lay], n_boot=30, seed=0, **CELL_KW)
    log(f"ceiling[{lay}] = {CEILINGS[lay]:.3f}")

Xc_base = np.vstack([trigram_features_from_positions(geom, c.positions, wpm=c.wpm) for c in cells])

ARMS = {
    "C-BASE": dict(prev=False, params={}),
    "C-D2": dict(prev=False, params={"max_depth": 2}),
    "C-D4": dict(prev=False, params={"max_depth": 4}),
    "C-PREV": dict(prev=True, params={}),
    "C-PREV-D2": dict(prev=True, params={"max_depth": 2}),
}


def fit_model(Xm, ym, seed, weight, names, params):
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION, feature_names=list(names),
        wpm_range=(60, 120), ngram="trigram",
    )
    m = XGBoostTypingModel(meta, random_state=seed, n_jobs=0, **params)
    m._regressor.fit(Xm, ym, sample_weight=weight)
    m._fitted = True
    return m


results = {}
for arm, spec in ARMS.items():
    if spec["prev"]:
        X = np.column_stack([X_base, ex_prev])
        Xc = np.column_stack([Xc_base, cell_prev])
        names = [*TRIGRAM_FEATURE_NAMES, "prev_duration"]
    else:
        X, Xc, names = X_base, Xc_base, TRIGRAM_FEATURE_NAMES
    pooled = {s: {} for s in SEEDS}
    fracs = []
    for holdout in LAYOUTS:
        mask = ex_layout != holdout
        idx = [i for i, c in enumerate(cells) if c.layout == holdout]
        test_cells = [cells[i] for i in idx]
        obs = np.array([c.obs for c in test_cells])
        for seed in SEEDS:
            w = layout_balance_weights(ex_layout[mask])
            model = fit_model(X[mask], y[mask], seed, w, names, spec["params"])
            bmap = {}
            for _ in range(2):
                bmap = fit_practice_term(ex_ngram[mask], y[mask] - model.predict(X[mask]), ex_n[mask], k=K)
                bvec = np.array([bmap.get(g, 0.0) for g in ex_ngram[mask]])
                model = fit_model(X[mask], y[mask] - bvec, seed, w, names, spec["params"])
            pred = model.predict(Xc[idx]) + np.array([bmap.get(c.ngram, 0.0) for c in test_cells])
            rho = _centered_spearman(test_cells, pred, obs)
            fracs.append(rho / CEILINGS[holdout])
            pooled[seed][holdout] = aggregate_layout_table(test_cells, pred)[holdout]
    taus = [layout_ranking_tau(obs_table, pooled[s]) for s in SEEDS]
    results[arm] = {"taus": taus, "mean_frac": float(np.mean(fracs))}
    log(f"{arm}: tau {[f'{t:+.2f}' for t in taus]} mean rho/ceiling {results[arm]['mean_frac']:+.4f}")

json.dump(results, open("runs/cond_target_arms.json", "w"), indent=2, default=float)
print("\n=== CONDITIONED-TARGET SCOREBOARD ===")
for arm, r in sorted(results.items(), key=lambda kv: -kv[1]["mean_frac"]):
    print(f"{arm:<10} tau {[f'{t:+.2f}' for t in r['taus']]} rho/ceiling {r['mean_frac']:+.4f}")
print("ALL-DONE")
