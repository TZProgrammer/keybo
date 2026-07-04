"""Final layout search: R1W model (arm-matrix winner) + QAP-table deep search.

Pipeline (pre-registered in memory.md before the arm matrix ran):
 1. Train the winning arm — R1W: freq pinned, inverse-layout-share weights, explicit
    per-bigram practice term b backfit with shrinkage, g trained on residualized targets —
    on ALL data, seeds 0/1/2. EXACTLY the code path the harness validated (tau +1.0).
 2. Save the three g models (they are ordinary freq-inert XGBoost models; b is
    layout-independent so it cancels in layout comparisons — g alone is the ranking
    objective).
 3. Build the 31x31 QAP tables via TableBigramScorer (parity-tested vs the model scorer),
    average over seeds, and deep-search: multi-start SA + exhaustive 2-opt + 3-cycle
    polish, millions of table evaluations.
 4. Report the best layout, its per-seed consistency, and comparisons vs named layouts
    (common-subset scoring, same convention as the score CLI).
"""

import json
import time
from collections import defaultdict

import numpy as np

from keybo.data.strokes import iqr_average, load_strokes
from keybo.features import bigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.model_scorer import BigramModelScorer
from keybo.scoring.table_scorer import TableBigramScorer

SEEDS = [0, 1, 2]
K_SHRINK = 100.0
W_CAP = 50.0
TARGET_WPM = 90.0
QWERTY = NAMED_LAYOUTS["qwerty"]
V2_LAYOUT = "vbknl.goiustrhmpcaeydqjxzfw/;,"  # prior best (SA on the freq-live model)

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --- 1. training examples (identical construction to the validated arm matrix) ----------

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
ex_n = np.array(ex_n)
log(f"{len(y)} examples from {len(rows)} rows")

share = defaultdict(float)
for lay in ex_layout:
    share[lay] += 1.0
total = sum(share.values())
sw = np.array([min(W_CAP, total / (len(share) * share[la])) for la in ex_layout])
sw /= sw.mean()


def fit_xgb(Xm, ym, seed, weight):
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=BIGRAM_FEATURE_NAMES,
        wpm_range=(60, 120),
        ngram="bigram",
    )
    m = XGBoostTypingModel(meta, random_state=seed, n_jobs=0)
    m._regressor.fit(Xm, ym, sample_weight=weight)
    m._fitted = True
    return m


corpus_freq = {}
for line in open("/tmp/keybo_harness1/data/corpus/bigrams.txt"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) == 2:
        corpus_freq[parts[0]] = int(parts[1])

# --- 2. train R1W per seed; build parity-tested QAP tables ------------------------------

tables, models = [], []
for seed in SEEDS:
    model = fit_xgb(X, y, seed, sw)
    bmap = {}
    for _ in range(2):  # backfit: b absorbs per-ngram residual mean (shrunk); g refits
        gpred = model.predict(X)
        num, den = defaultdict(float), defaultdict(float)
        for ng, r, w in zip(ex_ngram, y - gpred, ex_n):
            num[ng] += w * r
            den[ng] += w
        bmap = {ng: num[ng] / (den[ng] + K_SHRINK) for ng in num}
        bvec = np.array([bmap.get(ng, 0.0) for ng in ex_ngram])
        model = fit_xgb(X, y - bvec, seed, sw)
    model.metadata.extra["arm"] = "R1W"
    model.metadata.extra["practice_term"] = "per-ngram shrunk backfit (k=100), 2 iters"
    model.save(f"models/bigram_r1w_seed{seed}.json")
    json.dump(
        {k: float(v) for k, v in bmap.items()},
        open(f"models/bigram_r1w_seed{seed}.practice.json", "w"),
    )
    models.append(model)
    sc = TableBigramScorer(model, corpus_freq, target_wpm=TARGET_WPM, chars=QWERTY)
    tables.append(sc)
    log(f"seed {seed}: R1W trained; table built (freq-inert probe passed)")

F = tables[0]._F
T_mean = np.mean([sc._T for sc in tables], axis=0)
N_ASSIGN = 30  # slots 0..29 swappable; index 30 = space, pinned


def fit_of(perm, T=T_mean):
    return float((F * T[np.ix_(perm, perm)]).sum())


def layout_string(perm):
    """perm[i] = slot of char i (chars = QWERTY order) -> chars in slot order."""
    slots = [""] * N_ASSIGN
    for i, s in enumerate(perm[:N_ASSIGN]):
        slots[s] = QWERTY[i]
    return "".join(slots)


def perm_of(layout_str):
    sc = tables[0]
    return sc.permutation(Layout(layout_str, geom))


n_bench = 2000
p0 = perm_of(QWERTY)
tb = time.time()
for _ in range(n_bench):
    fit_of(p0)
per_eval = (time.time() - tb) / n_bench
log(f"table eval: {per_eval * 1e6:.1f} us")

# --- 3. deep search ----------------------------------------------------------------------

rng = np.random.default_rng(12345)


def sa_run(start_perm, iters, t0_temp, alpha=0.9995):
    perm = start_perm.copy()
    cur = fit_of(perm)
    best, best_perm = cur, perm.copy()
    T = t0_temp
    for _ in range(iters):
        i, j = rng.integers(0, N_ASSIGN, 2)
        if i == j:
            continue
        perm[i], perm[j] = perm[j], perm[i]
        cand = fit_of(perm)
        d = cand - cur
        if d <= 0 or rng.random() < np.exp(-d / T):
            cur = cand
            if cur < best:
                best, best_perm = cur, perm.copy()
        else:
            perm[i], perm[j] = perm[j], perm[i]
        T *= alpha
    return best, best_perm


def two_opt(perm):
    perm = perm.copy()
    cur = fit_of(perm)
    improved = True
    while improved:
        improved = False
        for i in range(N_ASSIGN):
            for j in range(i + 1, N_ASSIGN):
                perm[i], perm[j] = perm[j], perm[i]
                cand = fit_of(perm)
                if cand < cur - 1e-9:
                    cur = cand
                    improved = True
                else:
                    perm[i], perm[j] = perm[j], perm[i]
    return cur, perm


def three_cycle(perm):
    """All 3-cycles (two rotation directions); escapes some 2-opt optima."""
    perm = perm.copy()
    cur = fit_of(perm)
    improved = True
    while improved:
        improved = False
        for i in range(N_ASSIGN):
            for j in range(i + 1, N_ASSIGN):
                for k in range(j + 1, N_ASSIGN):
                    a, b, c = perm[i], perm[j], perm[k]
                    for rot in (((b, c, a)), ((c, a, b))):
                        perm[i], perm[j], perm[k] = rot
                        cand = fit_of(perm)
                        if cand < cur - 1e-9:
                            cur = cand
                            improved = True
                            break
                        perm[i], perm[j], perm[k] = a, b, c
                    else:
                        continue
                    break
    return cur, perm


# Calibrate T0: median uphill delta of random swaps from random perms.
deltas = []
for _ in range(400):
    p = np.append(rng.permutation(N_ASSIGN), N_ASSIGN)
    f1 = fit_of(p)
    i, j = rng.integers(0, N_ASSIGN, 2)
    p[i], p[j] = p[j], p[i]
    d = fit_of(p) - f1
    if d > 0:
        deltas.append(d)
T0 = float(np.median(deltas)) / np.log(2)
log(f"SA T0 = {T0:.3e}")

starts = [perm_of(QWERTY), perm_of(V2_LAYOUT)]
starts += [np.append(rng.permutation(N_ASSIGN), N_ASSIGN) for _ in range(26)]

global_best, global_perm = np.inf, None
for idx, sp in enumerate(starts):
    f_sa, p_sa = sa_run(sp, iters=60_000, t0_temp=T0)
    f_2, p_2 = two_opt(p_sa)
    if f_2 < global_best:
        global_best, global_perm = f_2, p_2
        log(f"restart {idx}: new best {global_best:.6e} ({layout_string(p_2)})")

# Intensify around the incumbent: perturb-and-repolish (kick 4 random swaps), then 3-cycle.
for kick in range(30):
    p = global_perm.copy()
    for _ in range(4):
        i, j = rng.integers(0, N_ASSIGN, 2)
        p[i], p[j] = p[j], p[i]
    f_2, p_2 = two_opt(p)
    if f_2 < global_best - 1e-9:
        global_best, global_perm = f_2, p_2
        log(f"kick {kick}: new best {global_best:.6e}")

f_3, p_3 = three_cycle(global_perm)
if f_3 < global_best - 1e-9:
    log(f"3-cycle improved: {global_best:.6e} -> {f_3:.6e}")
    global_best, global_perm = f_3, p_3
    f_2, p_2 = two_opt(global_perm)  # re-polish after 3-cycle
    if f_2 < global_best - 1e-9:
        global_best, global_perm = f_2, p_2

best_layout = layout_string(global_perm)
log(f"FINAL: {global_best:.6e}  {best_layout}")

# --- 4. verification + report ------------------------------------------------------------

lay_best = Layout(best_layout, geom)
per_seed = [sc.fitness(lay_best) for sc in tables]
# Independent path: full model scorer on the final layout only (parity spot-check).
ref_check = [
    BigramModelScorer(m, corpus_freq, target_wpm=TARGET_WPM).fitness(lay_best) for m in models
]
for s, (a, b) in enumerate(zip(per_seed, ref_check)):
    assert abs(a - b) / b < 1e-9, f"seed {s} parity broke: {a} vs {b}"
log("model-scorer parity on final layout: OK (3 seeds)")

# Named-layout comparison on the common typable subset (score-CLI convention).
from keybo.cli.score import common_ngrams

compare = {"best": best_layout, **{n: NAMED_LAYOUTS[n] for n in sorted(NAMED_LAYOUTS)}}
layouts = {n: Layout(s, geom) for n, s in compare.items()}
common = common_ngrams(corpus_freq, list(layouts.values()))
cov = sum(common.values()) / sum(corpus_freq.values())
log(f"common subset: {len(common)} bigrams, {cov:.1%} of corpus weight")

result = {"layout": best_layout, "table_fitness_mean": global_best, "per_seed": per_seed}
scores = {}
for name, lay in layouts.items():
    per_seed_common = [
        BigramModelScorer(m, common, target_wpm=TARGET_WPM).fitness(lay) for m in models
    ]
    scores[name] = float(np.mean(per_seed_common))
    result.setdefault("common_subset_scores", {})[name] = {
        "mean": scores[name],
        "per_seed": [float(v) for v in per_seed_common],
    }
base = scores["qwerty"]
print("\n=== FINAL SCOREBOARD (R1W model, mean of 3 seeds, common subset) ===")
for name, s in sorted(scores.items(), key=lambda kv: kv[1]):
    print(f"  {name:<10} {s:.4e}  ({(base - s) / base * +100:+.2f}% vs qwerty)")

rowlens = (10, 10, 10)
i = 0
print("\nbest layout:")
for rl in rowlens:
    print("  " + " ".join(best_layout[i : i + rl]))
    i += rl

json.dump(result, open("runs/final_layout.json", "w"), indent=2)
log("wrote runs/final_layout.json; models in models/bigram_r1w_seed*.json")
print("ALL-DONE")
