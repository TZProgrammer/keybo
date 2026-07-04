"""Practice-confound arm matrix, judged by the LOLO harness (pre-registered 2026-07-04).

The user's (correct) causal story: frequent bigrams are fast partly BECAUSE practiced, so a
model with no practice term over-credits their qwerty geometry. freq-as-a-FEATURE failed as
an instrument (98.7%-qwerty data makes freq a position ID), so these arms model practice
EXPLICITLY as an additive per-bigram term b and train geometry g on the residualized target:

  time = g(geometry, wpm) + b(bigram)

b is layout-independent, so it cancels in layout fitness comparisons; its only job is to
stop g from absorbing the practice effect into geometry. Arms:

  B    control: raw targets (must reproduce runs/lolo_v3_nofreq.json — self-check)
  R1   backfit per-ngram b with shrinkage (k=100 raw samples), 2 refit iterations
  R2   b = binned curve of log10(corpus frequency) (global practice-frequency curve)
  W    inverse-layout-share example weights (cap 50, mean-normalized)
  R1W  R1 + W

All arms: freq feature pinned to 1. b-hat per fold computed from TRAIN rows only.
Decision rule (pre-registered): decisive = pooled held-out tau (mean over seeds);
tie-break 1 = mean rho/ceiling; tie-break 2 = beats-baseline count. Winner must be >= B.
Ceilings + baseline reused from runs/lolo_v3.json (arm-independent data properties).
"""

import json
import sys
import time
from collections import defaultdict

import numpy as np

from keybo.data.strokes import iqr_average, load_strokes
from keybo.features import bigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
from keybo.geometry import ROW_STAGGERED_30
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.training.validate import (
    _centered_spearman,
    _predict_cells,
    aggregate_layout_table,
    build_cells,
    layout_ranking_tau,
)

SEEDS = [0, 1, 2]
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
K_SHRINK = 100.0  # raw-sample shrinkage constant for per-ngram b
N_BINS = 8  # quantile bins for the R2 frequency curve
W_CAP = 50.0

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --- load ------------------------------------------------------------------------------

rows = load_strokes("bistrokes_v3.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} stroke rows")

corpus_freq = {}
for line in open("/tmp/keybo_harness1/data/corpus/bigrams.txt"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) == 2:
        corpus_freq[parts[0]] = int(parts[1])

# --- training examples (one pass over all samples; per-example layout tag for masking) --
# Mirrors training._rows_to_examples exactly: one example per (row, exact-integer-wpm),
# target = IQR-mean of that group's durations, features with freq pinned to 1.

feats, targets, ex_ngram, ex_layout, ex_n = [], [], [], [], []
geom = ROW_STAGGERED_30
for row in rows:
    by_wpm = defaultdict(list)
    for wpm, duration, _pid, _hold in row.samples:
        by_wpm[wpm].append(duration)
    for wpm, durations in by_wpm.items():
        feats.append(
            bigram_features_from_positions(geom, row.positions, freq=1.0, wpm=wpm)
        )
        targets.append(iqr_average(durations))
        ex_ngram.append(row.ngram)
        ex_layout.append(row.layout)
        ex_n.append(len(durations))

X_all = np.vstack(feats)
y_all = np.array(targets, dtype=np.float64)
ex_ngram = np.array(ex_ngram, dtype=object)
ex_layout = np.array(ex_layout, dtype=object)
ex_n = np.array(ex_n, dtype=np.float64)
log(f"{len(y_all)} training examples ({X_all.shape[1]} features)")
del feats, targets

all_cells = build_cells(rows, **CELL_KW)
obs_table = aggregate_layout_table(all_cells)
LAYOUTS = sorted({c.layout for c in all_cells})
log(f"{len(all_cells)} cells across {LAYOUTS}")

# Arm-independent data properties, reused from the main run.
main = json.load(open("runs/lolo_v3.json"))
CEILINGS = main["ceilings"]

log10f = np.array(
    [np.log10(corpus_freq.get(str(ng).lower(), 0) + 1.0) for ng in ex_ngram]
)


def fit_xgb(X, y, seed, sample_weight=None):
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=BIGRAM_FEATURE_NAMES,
        wpm_range=(60, 120),
        ngram="bigram",
    )
    m = XGBoostTypingModel(meta, random_state=seed, n_jobs=1)
    m._regressor.fit(X, y, sample_weight=sample_weight)
    m._fitted = True
    return m


def shrunk_ngram_means(ngrams, resid, weights, k):
    num, den = defaultdict(float), defaultdict(float)
    for ng, r, w in zip(ngrams, resid, weights):
        num[ng] += w * r
        den[ng] += w
    return {ng: num[ng] / (den[ng] + k) for ng in num}


def freq_curve(logf, resid, weights, n_bins):
    """Weighted mean residual per quantile bin of log10 corpus freq -> (edges, values)."""
    edges = np.quantile(logf, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, logf, side="right") - 1, 0, n_bins - 1)
    vals = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        vals[b] = np.average(resid[m], weights=weights[m]) if m.any() else 0.0
    return edges, vals


def curve_lookup(edges, vals, logf):
    idx = np.clip(np.searchsorted(edges, logf, side="right") - 1, 0, len(vals) - 1)
    return vals[idx]


def train_arm(arm, mask_train, seed):
    """Fit an arm on the masked examples; return (model, bhat_fn) where bhat_fn maps a
    list of ngrams -> additive b per ngram (0 for arms without one)."""
    X, y, w = X_all[mask_train], y_all[mask_train], ex_n[mask_train]
    ngrams = ex_ngram[mask_train]
    logf = log10f[mask_train]

    sw = None
    if arm in ("W", "R1W"):
        share = defaultdict(float)
        for lay in ex_layout[mask_train]:
            share[lay] += 1.0
        total = sum(share.values())
        wt = np.array(
            [min(W_CAP, total / (len(share) * share[la])) for la in ex_layout[mask_train]]
        )
        sw = wt / wt.mean()

    if arm in ("B", "W"):
        model = fit_xgb(X, y, seed, sample_weight=sw)
        return model, lambda ngs: np.zeros(len(ngs))

    if arm in ("R1", "R1W"):
        model = fit_xgb(X, y, seed, sample_weight=sw)
        bmap = {}
        for _ in range(2):  # backfit iterations
            resid = y - bhat_of(bmap, ngrams) - model.predict(X)
            # b absorbs the residual mean per ngram (shrunk); g refits on y - b.
            bmap = shrunk_ngram_means(ngrams, y - model.predict(X), w, K_SHRINK)
            model = fit_xgb(X, y - bhat_of(bmap, ngrams), seed, sample_weight=sw)
        return model, lambda ngs, _b=bmap: bhat_of(_b, ngs)

    if arm == "R2":
        model = fit_xgb(X, y, seed)
        edges, vals = None, None
        for _ in range(2):
            resid = y - model.predict(X)
            edges, vals = freq_curve(logf, resid, w, N_BINS)
            model = fit_xgb(X, y - curve_lookup(edges, vals, logf), seed)
        lf_of = lambda ngs: np.array(
            [np.log10(corpus_freq.get(str(n).lower(), 0) + 1.0) for n in ngs]
        )
        return model, lambda ngs, _e=edges, _v=vals: curve_lookup(_e, _v, lf_of(ngs))

    raise ValueError(arm)


def bhat_of(bmap, ngrams):
    return np.array([bmap.get(ng, 0.0) for ng in ngrams])


# --- baseline (distance+wpm linear), per fold, from cells — matches the harness ---------


def _distance(positions):
    (x1, y1), (x2, y2) = positions
    return float(np.hypot(x1 - x2, y1 - y2))


def baseline_mae(train_cells, test_cells, obs):
    Xb = np.array([[1.0, _distance(c.positions), c.wpm] for c in train_cells])
    yb = np.array([c.obs for c in train_cells])
    coef, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    Xt = np.array([[1.0, _distance(c.positions), c.wpm] for c in test_cells])
    return float(np.mean(np.abs(Xt @ coef - obs)))


# --- run the matrix ----------------------------------------------------------------------

ARMS = ["B", "R1", "R2", "W", "R1W"]
results = {arm: {"folds": {}, "pooled": {}} for arm in ARMS}

for arm in ARMS:
    pred_heldout = {s: {} for s in SEEDS}
    for holdout in LAYOUTS:
        mask_train = ex_layout != holdout
        test_cells = [c for c in all_cells if c.layout == holdout]
        train_cells = [c for c in all_cells if c.layout != holdout]
        obs = np.array([c.obs for c in test_cells])
        mae_base = baseline_mae(train_cells, test_cells, obs)
        fold = results[arm]["folds"].setdefault(holdout, [])
        for seed in SEEDS:
            model, bhat_fn = train_arm(arm, mask_train, seed)
            g_pred = _predict_cells(model, test_cells, geom)
            pred = g_pred + bhat_fn([c.ngram for c in test_cells])
            rho = _centered_spearman(test_cells, pred, obs)
            mae = float(np.mean(np.abs(pred - obs)))
            g_all = _predict_cells(model, all_cells, geom)
            pred_all = g_all + bhat_fn([c.ngram for c in all_cells])
            tau_all = layout_ranking_tau(
                obs_table, aggregate_layout_table(all_cells, pred_all)
            )
            fold.append(
                {
                    "seed": seed,
                    "rho": rho,
                    "frac": rho / CEILINGS[holdout],
                    "tau_all": tau_all,
                    "mae": mae,
                    "mae_base": mae_base,
                    "beats": mae < mae_base,
                }
            )
            pred_heldout[seed][holdout] = aggregate_layout_table(test_cells, pred)[holdout]
        log(f"{arm}/{holdout}: " + " ".join(f"rho={m['rho']:+.3f}" for m in fold[-3:]))
    for seed in SEEDS:
        results[arm]["pooled"][seed] = layout_ranking_tau(obs_table, pred_heldout[seed])
    log(
        f"{arm} pooled tau: "
        + " ".join(f"{results[arm]['pooled'][s]:+.3f}" for s in SEEDS)
    )

json.dump(results, open("runs/arms_matrix.json", "w"), indent=2, default=float)

# --- scoreboard --------------------------------------------------------------------------

print("\n=== ARM SCOREBOARD (decisive first) ===")
print(f"{'arm':<5} {'pooled tau (3 seeds)':<25} {'mean rho/ceiling':<18} beats-baseline")
for arm in ARMS:
    taus = [results[arm]["pooled"][s] for s in SEEDS]
    fracs = [m["frac"] for f in results[arm]["folds"].values() for m in f]
    beats = sum(m["beats"] for f in results[arm]["folds"].values() for m in f)
    ntotal = sum(len(f) for f in results[arm]["folds"].values())
    print(
        f"{arm:<5} {' '.join(f'{t:+.3f}' for t in taus):<25} "
        f"{np.mean(fracs):+.3f}            {beats}/{ntotal}"
    )

# Self-check: arm B seed 0 must reproduce the earlier no-freq harness run.
ref = json.load(open("runs/lolo_v3_nofreq.json"))
for lay in LAYOUTS:
    mine = results["B"]["folds"][lay][0]["rho"]
    theirs = ref["folds"][lay]["seeds"][0]["rho"]
    status = "OK" if abs(mine - theirs) < 0.02 else "MISMATCH"
    print(f"self-check B/{lay}: {mine:+.3f} vs harness {theirs:+.3f} {status}")
    if status == "MISMATCH":
        sys.exit(2)
print("ALL-DONE")
