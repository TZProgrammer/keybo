"""Red-flag tests (why does dvorak outrank modern layouts?) + the user's bucketed-freq
proposal, run through the LOLO harness. Pre-registered in state memory before results.

Tests:
 1. no-dvorak training: R1W trained with ZERO dvorak rows -> named-layout scoreboard.
    If dvorak's rank drops, its own data was the channel; if unchanged, the bigram
    objective structurally favors alternation (dvorak's design).
 2. R1 (no layout weights) scoreboard: isolates the 800x-dvorak-upweight channel.
 3. Alternation diagnostic: per-layout corpus-weighted motion-class shares + the model's
    predicted class means -> quantifies how much of dvorak's edge is alternation.
 4. Arm F20W (user's proposal): 20 equal-count corpus-frequency buckets as a FEATURE
    (bucket index from quantile edges over the external corpus file - serve-computable),
    practice term OFF, layout weights ON.
 5. Arm R3W (reframe): hierarchical practice term - per-bigram b shrunk toward the R2
    frequency-curve value instead of toward 0.

Decision rule: adopt F20W/R3W over shipped R1W only if pooled held-out tau >= +1.0 AND
mean rho/ceiling > 0.931. Uses the frozen prod checkout (/tmp/keybo_prod1, 61a3d5d).
"""

import json
import time
from collections import defaultdict

import numpy as np

from keybo.cli.score import common_ngrams
from keybo.data.strokes import iqr_average, load_strokes
from keybo.features import bigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
from keybo.features.classify import BigramClass, classify_positions
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.model_scorer import BigramModelScorer
from keybo.training.train import fit_practice_term, layout_balance_weights
from keybo.training.validate import (
    _centered_spearman,
    aggregate_layout_table,
    build_cells,
    layout_ranking_tau,
)

SEEDS = [0, 1, 2]
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
TARGET_WPM = 90.0
K = 100.0
N_BUCKETS = 20
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


geom = ROW_STAGGERED_30
rows = load_strokes("bistrokes_v3.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)

corpus = {}
for line in open("/tmp/keybo_prod1/data/corpus/bigrams.txt"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) == 2:
        corpus[parts[0]] = int(parts[1])

# Bucket edges from the EXTERNAL corpus distribution (serve-computable, not data-derived).
corpus_logf = np.sort(np.log10(np.array(list(corpus.values()), dtype=float) + 1.0))
EDGES = np.quantile(corpus_logf, np.linspace(0, 1, N_BUCKETS + 1))
EDGES[0], EDGES[-1] = -np.inf, np.inf


def bucket_of(ngram: str) -> float:
    lf = np.log10(corpus.get(str(ngram).lower(), 0) + 1.0)
    return float(np.clip(np.searchsorted(EDGES, lf, side="right") - 1, 0, N_BUCKETS - 1))


# --- training examples -------------------------------------------------------------------
feats, targets, ex_ngram, ex_layout, ex_n = [], [], [], [], []
for row in rows:
    by_wpm = defaultdict(list)
    for wpm, duration, _pid, _hold in row.samples:
        by_wpm[wpm].append(duration)
    for wpm, durations in by_wpm.items():
        feats.append(bigram_features_from_positions(geom, row.positions, wpm=wpm))
        targets.append(iqr_average(durations))
        ex_ngram.append(row.ngram)
        ex_layout.append(row.layout)
        ex_n.append(len(durations))
X = np.vstack(feats)
y = np.array(targets)
ex_ngram = np.array(ex_ngram, dtype=object)
ex_layout = np.array(ex_layout, dtype=object)
ex_n = np.array(ex_n)
ex_bucket = np.array([bucket_of(ng) for ng in ex_ngram])
log(f"{len(y)} examples")

all_cells = build_cells(rows, **CELL_KW)
obs_table = aggregate_layout_table(all_cells)
LAYOUTS = sorted({c.layout for c in all_cells})
CEILINGS = json.load(open("runs/lolo_v3.json"))["ceilings"]


def make_meta(names):
    return ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=list(names),
        wpm_range=(60, 120),
        ngram="bigram",
    )


def fit_xgb(Xm, ym, seed, weight=None, names=BIGRAM_FEATURE_NAMES):
    m = XGBoostTypingModel(make_meta(names), random_state=seed, n_jobs=0)
    m._regressor.fit(Xm, ym, sample_weight=weight)
    m._fitted = True
    return m


def freq_curve_map(ngrams, resid, weights):
    """R2-style curve: weighted mean residual per corpus-freq bucket -> per-bucket value."""
    idx = np.array([int(bucket_of(ng)) for ng in ngrams])
    vals = np.zeros(N_BUCKETS)
    for b in range(N_BUCKETS):
        m = idx == b
        vals[b] = np.average(resid[m], weights=weights[m]) if m.any() else 0.0
    return vals


def train_arm(arm, mask, seed):
    """Returns (model, bhat_fn(ngrams)->np.array, uses_bucket_feature)."""
    Xm, ym, ng, la, nn = X[mask], y[mask], ex_ngram[mask], ex_layout[mask], ex_n[mask]
    w = layout_balance_weights(la) if arm.endswith("W") else None

    if arm in ("R1W", "R1"):
        model = fit_xgb(Xm, ym, seed, w)
        bmap = {}
        for _ in range(2):
            bmap = fit_practice_term(ng, ym - model.predict(Xm), nn, k=K)
            bvec = np.array([bmap.get(g, 0.0) for g in ng])
            model = fit_xgb(Xm, ym - bvec, seed, w)
        return model, (lambda gs, _b=bmap: np.array([_b.get(g, 0.0) for g in gs])), False

    if arm == "R3W":
        model = fit_xgb(Xm, ym, seed, w)
        bmap, curve = {}, np.zeros(N_BUCKETS)
        for _ in range(2):
            resid = ym - model.predict(Xm)
            curve = freq_curve_map(ng, resid, nn)
            num, den = defaultdict(float), defaultdict(float)
            for g, r, c in zip(ng, resid, nn):
                num[g] += c * r
                den[g] += c
            bmap = {
                g: (num[g] + K * curve[int(bucket_of(g))]) / (den[g] + K) for g in num
            }
            bvec = np.array([bmap.get(g, 0.0) for g in ng])
            model = fit_xgb(Xm, ym - bvec, seed, w)

        def bhat(gs, _b=bmap, _c=curve):
            return np.array(
                [_b[g] if g in _b else _c[int(bucket_of(g))] for g in gs]
            )

        return model, bhat, False

    if arm == "F20W":
        names = [*BIGRAM_FEATURE_NAMES, "freq_bucket"]
        Xb = np.column_stack([Xm, ex_bucket[mask]])
        model = fit_xgb(Xb, ym, seed, w, names=names)
        return model, (lambda gs: np.zeros(len(gs))), True

    raise ValueError(arm)


def predict_cells_arm(model, bhat_fn, uses_bucket, cells):
    Xc = np.vstack(
        [bigram_features_from_positions(geom, c.positions, wpm=c.wpm) for c in cells]
    )
    if uses_bucket:
        Xc = np.column_stack([Xc, np.array([bucket_of(c.ngram) for c in cells])])
    return model.predict(Xc) + bhat_fn([c.ngram for c in cells])


def lolo(arm):
    pooled = {s: {} for s in SEEDS}
    stats = []
    for holdout in LAYOUTS:
        mask = ex_layout != holdout
        test_cells = [c for c in all_cells if c.layout == holdout]
        obs = np.array([c.obs for c in test_cells])
        for seed in SEEDS:
            model, bhat, ub = train_arm(arm, mask, seed)
            pred = predict_cells_arm(model, bhat, ub, test_cells)
            rho = _centered_spearman(test_cells, pred, obs)
            stats.append((holdout, seed, rho, rho / CEILINGS[holdout]))
            pooled[seed][holdout] = aggregate_layout_table(test_cells, pred)[holdout]
        log(f"  {arm}/{holdout}: rho " + " ".join(f"{s[2]:+.3f}" for s in stats[-3:]))
    taus = [layout_ranking_tau(obs_table, pooled[s]) for s in SEEDS]
    fracs = [s[3] for s in stats]
    log(f"  {arm}: pooled tau {[f'{t:+.3f}' for t in taus]} mean rho/ceiling {np.mean(fracs):+.3f}")
    return {"taus": taus, "mean_frac": float(np.mean(fracs)), "detail": stats}


def scoreboard(models_bhats, label):
    """Common-subset named-layout scoreboard for a list of (model, uses_bucket) per seed."""
    compare = {n: NAMED_LAYOUTS[n] for n in sorted(NAMED_LAYOUTS)}
    lays = {n: Layout(s, geom) for n, s in compare.items()}
    common = common_ngrams(corpus, list(lays.values()))
    out = {}
    for name, lay in lays.items():
        per_seed = []
        for model, uses_bucket in models_bhats:
            if uses_bucket:
                # score via feature append (mirror of the scorer, bucket added)
                vecs, fr = [], []
                for bg, f in common.items():
                    Xr = bigram_features_from_positions(
                        geom, (lay.pos(bg[0]), lay.pos(bg[1])), wpm=TARGET_WPM
                    )
                    vecs.append(np.append(Xr, bucket_of(bg)))
                    fr.append(f)
                per_seed.append(float(np.sum(model.predict(np.vstack(vecs)) * np.array(fr))))
            else:
                per_seed.append(
                    BigramModelScorer(model, common, target_wpm=TARGET_WPM).fitness(lay)
                )
        out[name] = float(np.mean(per_seed))
    base = out["qwerty"]
    log(f"  scoreboard [{label}]:")
    for n, s in sorted(out.items(), key=lambda kv: kv[1]):
        log(f"    {n:<10} {(base - s) / base * 100:+.2f}% vs qwerty")
    return out


results = {}

# --- 1. no-dvorak training scoreboard ---------------------------------------------------
log("TEST 1: R1W trained with ZERO dvorak rows")
mask_nod = ex_layout != "dvorak"
mb = []
for seed in SEEDS:
    model, bhat, ub = train_arm("R1W", mask_nod, seed)
    mb.append((model, ub))
results["scoreboard_no_dvorak"] = scoreboard(mb, "R1W, no dvorak data")

# --- 2. R1 (no layout weights) scoreboard ------------------------------------------------
log("TEST 2: R1 (practice term, NO layout weights), all data")
mask_all = np.ones(len(y), dtype=bool)
mb = []
for seed in SEEDS:
    model, bhat, ub = train_arm("R1", mask_all, seed)
    mb.append((model, ub))
results["scoreboard_r1_noweights"] = scoreboard(mb, "R1 no-weights, all data")

# Reference: shipped R1W all-data scoreboard
log("REFERENCE: R1W all data")
mb = []
for seed in SEEDS:
    model, bhat, ub = train_arm("R1W", mask_all, seed)
    mb.append((model, ub))
results["scoreboard_r1w"] = scoreboard(mb, "R1W, all data (shipped)")

# --- 3. alternation diagnostic -----------------------------------------------------------
log("TEST 3: motion-class diagnostic")
compare = {n: NAMED_LAYOUTS[n] for n in sorted(NAMED_LAYOUTS)}
lays = {n: Layout(s, geom) for n, s in compare.items()}
common = common_ngrams(corpus, list(lays.values()))
wsum = sum(common.values())
diag = {}
for name, lay in lays.items():
    shares = {c: 0.0 for c in ("alt", "shb", "sfb")}
    for bg, f in common.items():
        cls = classify_positions(geom, lay.pos(bg[0]), lay.pos(bg[1]))
        shares[cls.value] += f / wsum
    diag[name] = shares
    log(f"  {name:<10} alt {shares['alt']:.1%}  same-hand {shares['shb']:.1%}  sfb {shares['sfb']:.1%}")
results["motion_shares"] = diag
# model's predicted class means (seed-0 shipped model over all position pairs)
model0 = mb[0][0]
positions = [*geom.slots, geom.space_position]
Xp, cls_of = [], []
for a in positions:
    for b in positions:
        Xp.append(bigram_features_from_positions(geom, (a, b), wpm=TARGET_WPM))
        cls_of.append(classify_positions(geom, a, b).value)
pred = model0.predict(np.vstack(Xp))
cls_of = np.array(cls_of)
class_means = {c: float(pred[cls_of == c].mean()) for c in ("alt", "shb", "sfb")}
log(f"  model class means (ms): {class_means}")
results["model_class_means"] = class_means

# --- 4+5. user's F20W and reframed R3W through LOLO -------------------------------------
log("TEST 4: F20W (20 equal-count freq buckets as a feature)")
results["lolo_f20w"] = lolo("F20W")
log("TEST 5: R3W (hierarchical practice term shrunk toward freq curve)")
results["lolo_r3w"] = lolo("R3W")
log("ANCHOR: R1W via this driver (must match prod numbers)")
results["lolo_r1w_anchor"] = lolo("R1W")

json.dump(results, open("runs/redflag_buckets.json", "w"), indent=2, default=float)
log("wrote runs/redflag_buckets.json")
print("ALL-DONE")
