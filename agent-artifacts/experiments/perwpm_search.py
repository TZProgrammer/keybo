"""Per-WPM combined bigram+trigram search — the first deliverable-shaped run.

For each target wpm in {60, 90, 120}: build bigram (31^2) and trigram (31^3) tables from
the validated models (mean of 3 seeds each) AT that wpm, search the combined objective
(sum in natural ms units — trigram spans overlap so the trigram term carries ~2x weight
by construction, no arbitrary knob), report:
  - the per-wpm winner + top-K stability (distinct near-optima, slot consensus),
  - E5 structural postflight,
  - named-layout scoreboard under the combined objective per wpm,
  - the CROSS-WPM matrix: is skill-specialization real beyond plateau noise? (The
    skill-strata result predicts the 60-winner should weight alternation, the 120-winner
    should tolerate/чase rolls.)
"""

import json
import time
from collections import Counter

import numpy as np

from keybo.data.strokes import load_strokes
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.inspect import layout_diagnostics
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.scoring.table_trigram import TableTrigramScorer
from keybo.training.train import train_bigram_model

SEEDS = [0, 1, 2]
WPMS = [60.0, 90.0, 120.0]
QWERTY = NAMED_LAYOUTS["qwerty"]
geom = ROW_STAGGERED_30
N = 30
N_RESTARTS = 20
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


bi_corpus = {}
for line in open("/tmp/keybo_prod5/data/corpus/bigrams.txt"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2:
        bi_corpus[p[0]] = int(p[1])
tri_corpus = {}
for line in open("/tmp/keybo_prod5/data/corpus/trigrams.txt"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2:
        tri_corpus[p[0]] = int(p[1])

# Models: bigram d3 seeds exist; trigram seeds exist from the trigram round.
bi_models = [XGBoostTypingModel.load(f"models/bigram_d3_seed{s}.json") for s in SEEDS]
tri_models = [XGBoostTypingModel.load(f"models/trigram_seed{s}.json") for s in SEEDS]
log("models loaded")

rng = np.random.default_rng(20260707)


def search_combined(fit_fn, restarts=N_RESTARTS, iters=35_000):
    deltas = []
    for _ in range(150):
        p = np.append(rng.permutation(N), N)
        f1 = fit_fn(p)
        i, j = rng.integers(0, N, 2)
        p[i], p[j] = p[j], p[i]
        d = fit_fn(p) - f1
        if d > 0:
            deltas.append(d)
    T0 = float(np.median(deltas)) / np.log(2)
    results = []
    for _ in range(restarts):
        perm = np.append(rng.permutation(N), N)
        cur = fit_fn(perm)
        temp = T0
        for _ in range(iters):
            i, j = rng.integers(0, N, 2)
            if i == j:
                continue
            perm[i], perm[j] = perm[j], perm[i]
            cand = fit_fn(perm)
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
                    cand = fit_fn(perm)
                    if cand < cur - 1e-9:
                        cur = cand
                        improved = True
                    else:
                        perm[i], perm[j] = perm[j], perm[i]
        results.append((cur, perm.copy()))
    results.sort(key=lambda t: t[0])
    return results


def lay_str(perm):
    slots = [""] * N
    for i, s in enumerate(perm[:N]):
        slots[s] = QWERTY[i]
    return "".join(slots)


out = {"wpm": {}}
winners = {}
for wpm in WPMS:
    log(f"=== wpm {wpm:.0f}: building tables ===")
    bts = [TableBigramScorer(m, bi_corpus, target_wpm=wpm, chars=QWERTY) for m in bi_models]
    tts = [TableTrigramScorer(m, tri_corpus, target_wpm=wpm, chars=QWERTY) for m in tri_models]
    T2 = np.mean([sc._T for sc in bts], axis=0)
    F2 = bts[0]._F
    T3 = np.mean([sc._T3 for sc in tts], axis=0)
    I, J, L, F3 = tts[0]._i, tts[0]._j, tts[0]._l, tts[0]._f

    def fit(perm, T2=T2, F2=F2, T3=T3, I=I, J=J, L=L, F3=F3):
        return float((F2 * T2[np.ix_(perm, perm)]).sum()) + float(
            (F3 * T3[perm[I], perm[J], perm[L]]).sum()
        )

    results = search_combined(fit)
    best_fit, best_perm = results[0]
    best = lay_str(best_perm)
    winners[wpm] = (best, best_perm, fit)
    top = [(f, lay_str(p)) for f, p in results if f <= best_fit * 1.005]
    distinct = list({ls: f for f, ls in top}.items())
    consensus = []
    for slot in range(N):
        c = Counter(ls[slot] for ls, _ in distinct)
        ch, cnt = c.most_common(1)[0]
        consensus.append(cnt / len(distinct))
    stable = sum(1 for share in consensus if share >= 0.8)
    diag = layout_diagnostics(Layout(best, geom), bi_corpus)
    loads = {k: v for k, v in diag["finger_load"].items() if k != "thumb"}
    mx = max(loads, key=loads.get)
    log(
        f"wpm {wpm:.0f}: best {best} fit {best_fit:.4e} | {len(distinct)} distinct near-optima, "
        f"{stable}/30 stable slots | home {diag['row_share']['home']:.1%} sfb {diag['sfb_share']:.2%} "
        f"maxload {mx} {loads[mx]:.1%} alt {diag['motion_share']['alternate']:.1%}"
    )

    board = {}
    for name, s in {**{f"best@{wpm:.0f}": best}, **NAMED_LAYOUTS}.items():
        try:
            perm = bts[0].permutation(Layout(s, geom))
            board[name] = fit(perm)
        except ValueError:
            board[name] = None
    base = board["qwerty"]
    ranked = sorted(((n, v) for n, v in board.items() if v is not None), key=lambda kv: kv[1])
    log(f"  combined scoreboard @ {wpm:.0f}: " + "  ".join(
        f"{n} {(base - v) / base * 100:+.2f}%" for n, v in ranked
    ))
    out["wpm"][f"{wpm:.0f}"] = {
        "best": best,
        "fitness": best_fit,
        "n_distinct": len(distinct),
        "stable_slots": stable,
        "diag": {k: diag[k] for k in ("row_share", "sfb_share", "motion_share")},
        "scoreboard_pct": {
            n: (base - v) / base * 100 for n, v in board.items() if v is not None
        },
    }

# Cross-wpm matrix: each winner scored under each wpm's combined objective.
log("cross-wpm matrix")
cross = {}
for w_src, (lay, perm, _) in winners.items():
    row = {}
    for w_tab, (_, _, fit_fn) in winners.items():
        own = fit_fn(winners[w_tab][1])
        val = fit_fn(perm)
        row[f"{w_tab:.0f}"] = (val - own) / own * 100
    cross[f"{w_src:.0f}"] = row
    log(f"  winner@{w_src:.0f}: " + "  ".join(f"T@{k}: {v:+.3f}%" for k, v in row.items()))
out["cross_wpm_pct"] = cross
out["alt_share_by_wpm"] = {
    f"{w:.0f}": out["wpm"][f"{w:.0f}"]["diag"]["motion_share"]["alternate"] for w in WPMS
}

json.dump(out, open("runs/perwpm_combined.json", "w"), indent=2, default=float)
for w in WPMS:
    best = out["wpm"][f"{w:.0f}"]["best"]
    print(f"\nbest @ {w:.0f} wpm:")
    for i in range(0, 30, 10):
        print("  " + " ".join(best[i : i + 10]))
print("ALL-DONE")
