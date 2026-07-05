"""Trigram-objective layout search + rank-stability report (roadmap 1.2 + 4.4).

Trains the shipped trigram recipe (3 seeds) on tristrokes_v1, builds the 31^3 tables
(TableTrigramScorer, parity-tested), and runs a multi-restart search on the MEAN table.
Deliverables:
 - the trigram-optimal layout + named-layout scoreboard under the trigram objective,
 - the RANK-STABILITY report: top-K distinct optima across restarts with fitness spreads
   and per-position consensus (which letters are stably placed — turns the known ~0.5%
   plateau from an embarrassment into a deliverable),
 - E5 structural postflight on the winner,
 - cross-score: trigram-winner vs bigram-winner (d3-best) under BOTH objectives.
"""

import json
import time
from collections import Counter, defaultdict

import numpy as np

from keybo.data.strokes import load_strokes
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.inspect import layout_diagnostics
from keybo.scoring.table_trigram import TableTrigramScorer
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.training.train import train_trigram_model

SEEDS = [0, 1, 2]
TARGET_WPM = 90.0
QWERTY = NAMED_LAYOUTS["qwerty"]
geom = ROW_STAGGERED_30
N = 30
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# --- corpora -----------------------------------------------------------------------------
tri_corpus = {}
for line in open("/tmp/keybo_prod5/data/corpus/trigrams.txt"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2:
        tri_corpus[p[0]] = int(p[1])
bi_corpus = {}
for line in open("/tmp/keybo_prod5/data/corpus/bigrams.txt"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2:
        bi_corpus[p[0]] = int(p[1])
log(f"{len(tri_corpus)} corpus trigrams, {len(bi_corpus)} bigrams")

# --- train trigram models (shipped recipe) + tables --------------------------------------
rows = load_strokes("tristrokes_v1.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} tristroke rows")
tri_tables = []
for seed in SEEDS:
    model = train_trigram_model(rows, target_wpm=TARGET_WPM, random_state=seed, n_jobs=0)
    model.save(f"models/trigram_seed{seed}.json")
    sc = TableTrigramScorer(model, tri_corpus, target_wpm=TARGET_WPM, chars=QWERTY)
    tri_tables.append(sc)
    log(f"seed {seed}: trigram model trained, 31^3 table built")

sc0 = tri_tables[0]
T3 = np.mean([sc._T3 for sc in tri_tables], axis=0)
I, J, L, F = sc0._i, sc0._j, sc0._l, sc0._f


def fit_tri(perm):
    return float((F * T3[perm[I], perm[J], perm[L]]).sum())


t_bench = time.time()
p0 = sc0.permutation(Layout(QWERTY, geom))
for _ in range(500):
    fit_tri(p0)
log(f"trigram table eval: {(time.time() - t_bench) / 500 * 1e6:.0f} us")

# --- multi-restart search + stability ----------------------------------------------------
rng = np.random.default_rng(20260706)


def one_restart(iters=40_000, T0=None):
    perm = np.append(rng.permutation(N), N)
    cur = fit_tri(perm)
    temp = T0
    for _ in range(iters):
        i, j = rng.integers(0, N, 2)
        if i == j:
            continue
        perm[i], perm[j] = perm[j], perm[i]
        cand = fit_tri(perm)
        if cand - cur <= 0 or rng.random() < np.exp(-(cand - cur) / temp):
            cur = cand
        else:
            perm[i], perm[j] = perm[j], perm[i]
        temp *= 0.9995
    improved = True
    while improved:
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                perm[i], perm[j] = perm[j], perm[i]
                cand = fit_tri(perm)
                if cand < cur - 1e-9:
                    cur = cand
                    improved = True
                else:
                    perm[i], perm[j] = perm[j], perm[i]
    return cur, perm


deltas = []
for _ in range(200):
    p = np.append(rng.permutation(N), N)
    f1 = fit_tri(p)
    i, j = rng.integers(0, N, 2)
    p[i], p[j] = p[j], p[i]
    d = fit_tri(p) - f1
    if d > 0:
        deltas.append(d)
T0 = float(np.median(deltas)) / np.log(2)
log(f"SA T0 = {T0:.3e}")

N_RESTARTS = 24
results = []
for r in range(N_RESTARTS):
    fit, perm = one_restart(T0=T0)
    results.append((fit, perm.copy()))
    if fit <= min(f for f, _ in results) + 1e-9:
        log(f"restart {r}: best so far {fit:.6e}")
results.sort(key=lambda t: t[0])


def lay_str(perm):
    slots = [""] * N
    for i, s in enumerate(perm[:N]):
        slots[s] = QWERTY[i]
    return "".join(slots)


best_fit, best_perm = results[0]
best_layout = lay_str(best_perm)
log(f"BEST (trigram objective): {best_fit:.6e}  {best_layout}")

# Rank-stability: distinct top-K layouts within 0.5% of best + per-slot consensus.
top = [(f, lay_str(p)) for f, p in results if f <= best_fit * 1.005]
distinct = []
seen = set()
for f, ls in top:
    if ls not in seen:
        seen.add(ls)
        distinct.append((f, ls))
log(f"{len(distinct)} distinct optima within 0.5% of best (of {N_RESTARTS} restarts)")

consensus = []
for slot in range(N):
    letters = Counter(ls[slot] for _, ls in distinct)
    ch, cnt = letters.most_common(1)[0]
    consensus.append((slot, ch, cnt / len(distinct)))
stable = [(s, c) for s, c, share in consensus if share >= 0.8]
log(f"{len(stable)}/30 slots have >=80% letter consensus across near-optima")

# --- scoreboard under the trigram objective ----------------------------------------------
def tri_fitness_layout(layout_str):
    return np.mean(
        [sc.fitness(Layout(layout_str, geom)) for sc in tri_tables]
    )

d3_best = "dae,yrntscgoipulmfbwq;/.khvxzj"
board = {"tri-best": best_layout, "bigram-best(d3)": d3_best}
board.update({n: s for n, s in NAMED_LAYOUTS.items()})
scores = {}
for name, s in board.items():
    try:
        scores[name] = float(tri_fitness_layout(s))
    except ValueError:
        scores[name] = None
base = scores["qwerty"]
print("\n=== SCOREBOARD under the TRIGRAM objective (mean of 3 seeds) ===")
for name, v in sorted(((n, v) for n, v in scores.items() if v is not None), key=lambda kv: kv[1]):
    print(f"  {name:<16} {v:.4e}  ({(base - v) / base * 100:+.2f}% vs qwerty)")

# Cross: bigram objective on both winners (d3 tables).
bi_models = [XGBoostTypingModel.load(f"models/bigram_d3_seed{s}.json") for s in SEEDS]
bi_tables = [TableBigramScorer(m, bi_corpus, target_wpm=TARGET_WPM, chars=QWERTY) for m in bi_models]
def bi_fitness(s):
    return float(np.mean([sc.fitness(Layout(s, geom)) for sc in bi_tables]))
print("\n=== CROSS-OBJECTIVE ===")
for name, s in [("tri-best", best_layout), ("bigram-best(d3)", d3_best), ("qwerty", QWERTY)]:
    print(f"  {name:<16} bigram-obj {bi_fitness(s):.4e}   trigram-obj {scores[name]:.4e}")

# E5 postflight on the winner.
diag = layout_diagnostics(Layout(best_layout, geom), bi_corpus)
loads = {k: v for k, v in diag["finger_load"].items() if k != "thumb"}
mx = max(loads, key=loads.get)
print(f"\nE5 postflight tri-best: home {diag['row_share']['home']:.1%} | sfb {diag['sfb_share']:.2%} | max load {mx} {loads[mx]:.1%}")

rows_out = {
    "best_layout": best_layout,
    "best_fitness": best_fit,
    "distinct_top": distinct,
    "consensus": consensus,
    "stable_slots": stable,
    "scoreboard": scores,
    "cross": {n: {"bigram": bi_fitness(s), "trigram": scores[n]} for n, s in
              [("tri-best", best_layout), ("bigram-best(d3)", d3_best), ("qwerty", QWERTY)]},
    "e5": diag,
}
json.dump(rows_out, open("runs/trigram_search.json", "w"), indent=2, default=float)

print("\nbest layout (trigram objective):")
for i in range(0, 30, 10):
    print("  " + " ".join(best_layout[i : i + 10]))
print("ALL-DONE")
