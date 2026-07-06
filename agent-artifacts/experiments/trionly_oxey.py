"""Two experiments, one run (user questions 2026-07-06):

A. TRIGRAM-ONLY vs COMBINED objective: is the bigram term redundant? The trigram
   full-span target already embeds bigram physics (t(1->3) ~ bg1+bg2+context), so
   trigram-only is theoretically sufficient; the combined sum's implicit ~3:1
   bigram-physics weighting is unprincipled but lower-variance. Measure: search
   trigram-only, cross-score its winner against the combined winner under BOTH
   objectives. Rule (preregistered): if mutual regret is within plateau noise (0.15%),
   SIMPLIFY to trigram-only as canonical; else keep combined, ensemble-justified.

B. OXEY JOINT-OPTIMIZATION frontier: sweep oxey weight w in {0, 0.5, 1, 2, 4} (w scaled
   so w=1 puts the oxey term at ~1% of speed fitness at qwerty); search speed+w*oxey;
   report speed cost, oxey-score gain, and the pattern stats (sfb%, rolls%, redirects%)
   per w. The deliverable: the measured PRICE of community-approved patterns.
"""

import json
import time

import numpy as np

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.oxey import OxeyStyleScorer
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.scoring.table_trigram import TableTrigramScorer

SEEDS = [0, 1, 2]
QWERTY = NAMED_LAYOUTS["qwerty"]
geom = ROW_STAGGERED_30
N = 30
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def load_freq(path):
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            out[p[0]] = int(p[1])
    return out


ROOT = "/tmp/keybo_prod7/data/corpus"
bi_corpus = load_freq(f"{ROOT}/bigrams.txt")
sg_corpus = load_freq(f"{ROOT}/1-skip.txt")
tri_corpus = load_freq(f"{ROOT}/trigrams.txt")

bi_models = [XGBoostTypingModel.load(f"models/bigram_d3_seed{s}.json") for s in SEEDS]
tri_models = [XGBoostTypingModel.load(f"models/trigram_seed{s}.json") for s in SEEDS]
bts = [TableBigramScorer(m, bi_corpus, target_wpm=90.0, chars=QWERTY) for m in bi_models]
tts = [TableTrigramScorer(m, tri_corpus, target_wpm=90.0, chars=QWERTY) for m in tri_models]
T2 = np.mean([sc._T for sc in bts], axis=0)
F2 = bts[0]._F
T3 = np.mean([sc._T3 for sc in tts], axis=0)
I, J, L, F3 = tts[0]._i, tts[0]._j, tts[0]._l, tts[0]._f
log("tables ready")


def fit_bi(perm):
    return float((F2 * T2[np.ix_(perm, perm)]).sum())


def fit_tri(perm):
    return float((F3 * T3[perm[I], perm[J], perm[L]]).sum())


def fit_combined(perm):
    return fit_bi(perm) + fit_tri(perm)


rng = np.random.default_rng(31337)


def search(fit_fn, restarts=12, iters=30_000):
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
    best, best_p = np.inf, None
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
        if cur < best:
            best, best_p = cur, perm.copy()
    return best, best_p


def lay_str(perm):
    slots = [""] * N
    for i, s in enumerate(perm[:N]):
        slots[s] = QWERTY[i]
    return "".join(slots)


# ==== A. trigram-only vs combined ========================================================
log("A: trigram-only search")
_, p_tri = search(fit_tri)
log("A: combined search")
_, p_comb = search(fit_combined)
regret_tri_under_comb = (fit_combined(p_tri) - fit_combined(p_comb)) / fit_combined(p_comb) * 100
regret_comb_under_tri = (fit_tri(p_comb) - fit_tri(p_tri)) / fit_tri(p_tri) * 100
log(
    f"A: tri-only winner {lay_str(p_tri)} | combined winner {lay_str(p_comb)} | "
    f"regret(tri under comb) {regret_tri_under_comb:+.3f}% | "
    f"regret(comb under tri) {regret_comb_under_tri:+.3f}%"
)

# ==== B. oxey frontier ===================================================================
oxey = OxeyStyleScorer(bi_corpus, sg_corpus, tri_corpus)
lay_cache: dict[bytes, float] = {}


def oxey_of(perm):
    key = perm.tobytes()
    if key not in lay_cache:
        lay_cache[key] = oxey.fitness(Layout(lay_str(perm), geom))
    return lay_cache[key]


p_q = bts[0].permutation(Layout(QWERTY, geom))
speed_q = fit_combined(p_q)
oxey_q = oxey_of(p_q)
UNIT = speed_q / 100.0 / max(abs(oxey_q), 1.0)  # w=1 -> oxey term ~1% of speed at qwerty

results = {}
speed0 = None
for w in (0.0, 0.5, 1.0, 2.0, 4.0):
    # NOTE: oxey.fitness goes through Layout (not a table), so keep iterations modest.
    def fit(perm, w=w):
        v = fit_combined(perm)
        if w:
            v += w * UNIT * oxey_of(perm)
        return v

    _, perm = search(fit, restarts=6, iters=8_000)
    lay = lay_str(perm)
    spd = fit_combined(perm)
    if w == 0.0:
        speed0 = spd
    shares = oxey.pattern_shares(Layout(lay, geom))
    results[f"{w}"] = {
        "layout": lay,
        "speed_loss_pct": (spd - speed0) / speed0 * 100,
        "oxey_score": oxey_of(perm),
        "sfb_pct": shares["sfb"],
        "dsfb_pct": shares["dsfb"],
        "inroll_pct": shares["inroll"],
        "redirect_pct": shares["redirect"],
        "alternate_pct": shares["alternate"],
    }
    log(
        f"B: w={w}: speed loss {(spd - speed0) / speed0 * 100:+.3f}% | oxey {oxey_of(perm):7.1f} | "
        f"sfb {shares['sfb']:.2f}% dsfb {shares['dsfb']:.2f}% inroll {shares['inroll']:.2f}% "
        f"redirect {shares['redirect']:.2f}% alt {shares['alternate']:.1f}%"
    )

json.dump(
    {
        "A": {
            "tri_only_winner": lay_str(p_tri),
            "combined_winner": lay_str(p_comb),
            "regret_tri_under_combined_pct": regret_tri_under_comb,
            "regret_combined_under_tri_pct": regret_comb_under_tri,
        },
        "B": results,
    },
    open("runs/trionly_oxey.json", "w"),
    indent=2,
)
print("ALL-DONE")
