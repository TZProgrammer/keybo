"""Rebuild the trigram objective on the CONDITIONED model + re-run the two verdicts.

The conditioned-target program settled the trigram model (press2->press3 target, shipped
recipe, depth 3, no prev). This run:
 1. trains that model on ALL data (3 seeds) from the joined conditioned table,
 2. builds the corrected trigram objective:
        T3c[a,b,c] = T2[a,b] + Tcond[a,b,c]
    (first-bigram physics from the VALIDATED bigram model + the conditioned increment —
    each term predicted by the model that owns it; corpus trigram fitness then telescopes
    into corpus time with no double-count),
 3. re-runs the trigram-only vs combined A/B on the corrected tables (prereg rule:
    mutual regret <= 0.15% => simplify to trigram-only-corrected as canonical),
 4. final search on the winning objective + stability + E5 + named scoreboard,
 5. GL certificate on the bigram component of the winner (cubic part uncertified).
"""

import json
import time
from collections import Counter, defaultdict

import numpy as np

from keybo.data.strokes import iqr_average, load_strokes
from keybo.features import trigram_features_from_positions
from keybo.features.schema import FEATURE_VERSION, TRIGRAM_FEATURE_NAMES
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.inspect import layout_diagnostics
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.training.train import fit_practice_term, layout_balance_weights

SEEDS = [0, 1, 2]
TARGET_WPM = 90.0
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
tri_corpus = load_freq(f"{ROOT}/trigrams.txt")

# --- 1. conditioned training data (same join as cond_target_arms) ------------------------
full_rows = {(r.layout, r.positions, r.ngram): r for r in load_strokes(
    "tristrokes_v1.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)}
last_rows = {(r.layout, r.positions, r.ngram): r for r in load_strokes(
    "tristrokes_last.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)}
feats, targets, ex_ngram, ex_layout, ex_n = [], [], [], [], []
for key, lr in last_rows.items():
    fr = full_rows.get(key)
    if fr is None or len(fr.samples) != len(lr.samples):
        continue
    by_wpm = defaultdict(list)
    for (wf, df, pf, hf), (wl, dl, pl, hl) in zip(fr.samples, lr.samples):
        if (wf, pf, hf) != (wl, pl, hl) or not (0 <= df - dl <= 5000):
            continue
        by_wpm[wl].append(dl)
    layout, positions, ngram = key
    for wpm, durs in by_wpm.items():
        feats.append(trigram_features_from_positions(geom, positions, wpm=wpm))
        targets.append(iqr_average(durs))
        ex_ngram.append(ngram)
        ex_layout.append(layout)
        ex_n.append(len(durs))
X = np.vstack(feats)
y = np.array(targets)
ex_ngram = np.array(ex_ngram, dtype=object)
ex_layout = np.array(ex_layout, dtype=object)
ex_n = np.array(ex_n)
log(f"{len(y)} conditioned examples")


def fit_model(Xm, ym, seed, weight):
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION, feature_names=list(TRIGRAM_FEATURE_NAMES),
        wpm_range=(60, 120), ngram="trigram",
    )
    m = XGBoostTypingModel(meta, random_state=seed, n_jobs=0)
    m._regressor.fit(Xm, ym, sample_weight=weight)
    m._fitted = True
    return m


cond_models = []
for seed in SEEDS:
    w = layout_balance_weights(ex_layout)
    model = fit_model(X, y, seed, w)
    bmap = {}
    for _ in range(2):
        bmap = fit_practice_term(ex_ngram, y - model.predict(X), ex_n, k=100.0)
        bvec = np.array([bmap.get(g, 0.0) for g in ex_ngram])
        model = fit_model(X, y - bvec, seed, w)
    model.save(f"models/trigram_cond_seed{seed}.json")
    cond_models.append(model)
    log(f"seed {seed}: conditioned trigram model trained + saved")

# --- 2. corrected objective tables --------------------------------------------------------
bi_models = [XGBoostTypingModel.load(f"models/bigram_d3_seed{s}.json") for s in SEEDS]
bts = [TableBigramScorer(m, bi_corpus, target_wpm=TARGET_WPM, chars=QWERTY) for m in bi_models]
T2 = np.mean([sc._T for sc in bts], axis=0)
F2 = bts[0]._F

positions31 = [*geom.slots, geom.space_position]
n31 = len(positions31)
vec_all = np.vstack(
    [
        trigram_features_from_positions(geom, (a, b, c), wpm=TARGET_WPM)
        for a in positions31 for b in positions31 for c in positions31
    ]
)
Tcond = np.mean([m.predict(vec_all).reshape(n31, n31, n31) for m in cond_models], axis=0)
# Corrected trigram time: first-bigram physics + conditioned increment.
T3c = T2[:, :, None] + Tcond
log("corrected T3c built (T2 + Tcond)")

# corpus trigram indices over the qwerty charset
char_idx = {c: i for i, c in enumerate(QWERTY)}
char_idx[" "] = N
charset = set(QWERTY) | {" "}
ks = [(char_idx[t[0]], char_idx[t[1]], char_idx[t[2]], f)
      for t, f in tri_corpus.items() if len(t) == 3 and all(c in charset for c in t)]
I3 = np.array([k[0] for k in ks]); J3 = np.array([k[1] for k in ks])
L3 = np.array([k[2] for k in ks]); F3 = np.array([k[3] for k in ks], dtype=float)


def fit_tri_corrected(perm):
    return float((F3 * T3c[perm[I3], perm[J3], perm[L3]]).sum())


def fit_bi(perm):
    return float((F2 * T2[np.ix_(perm, perm)]).sum())


def fit_combined(perm):
    return fit_bi(perm) + fit_tri_corrected(perm)


rng = np.random.default_rng(60660)


def search(fit_fn, restarts=16, iters=30_000):
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


# --- 3. tri-only-corrected vs combined ----------------------------------------------------
log("searching corrected-trigram-only objective")
res_tri = search(fit_tri_corrected)
log("searching combined objective")
res_comb = search(fit_combined)
p_tri, p_comb = res_tri[0][1], res_comb[0][1]
regret_tri = (fit_combined(p_tri) - fit_combined(p_comb)) / fit_combined(p_comb) * 100
regret_comb = (fit_tri_corrected(p_comb) - fit_tri_corrected(p_tri)) / fit_tri_corrected(p_tri) * 100
log(f"A/B: regret(tri-corrected under combined) {regret_tri:+.3f}% | regret(combined under tri-corrected) {regret_comb:+.3f}%")
simplify = max(regret_tri, regret_comb) <= 0.15
canonical = "tri-corrected" if simplify else "combined"
results = res_tri if simplify else res_comb
fit_fn = fit_tri_corrected if simplify else fit_combined
log(f"canonical objective: {canonical}")

# --- 4. stability + E5 + scoreboard on the canonical objective ---------------------------
best_fit, best_perm = results[0]
best = lay_str(best_perm)
top = [(f, lay_str(p)) for f, p in results if f <= best_fit * 1.005]
distinct = list({ls: f for f, ls in top}.items())
consensus_stable = sum(
    1 for slot in range(N)
    if Counter(ls[slot] for ls, _ in distinct).most_common(1)[0][1] / len(distinct) >= 0.8
)
diag = layout_diagnostics(Layout(best, geom), bi_corpus)
loads = {k: v for k, v in diag["finger_load"].items() if k != "thumb"}
mx = max(loads, key=loads.get)
log(
    f"BEST ({canonical}): {best} | {len(distinct)} distinct near-optima, {consensus_stable}/30 "
    f"stable | home {diag['row_share']['home']:.1%} sfb {diag['sfb_share']:.2%} maxload {mx} {loads[mx]:.1%}"
)

board = {}
for name, s in {"best": best, **NAMED_LAYOUTS}.items():
    try:
        perm = bts[0].permutation(Layout(s, geom))
        board[name] = fit_fn(perm)
    except ValueError:
        board[name] = None
base = board["qwerty"]
print(f"\n=== SCOREBOARD ({canonical} objective) ===")
for name, v in sorted(((n, v) for n, v in board.items() if v is not None), key=lambda kv: kv[1]):
    print(f"  {name:<10} ({(base - v) / base * 100:+.2f}% vs qwerty)")

# --- 5. GL certificate on the bigram component -------------------------------------------
from keybo.optimize.qap_bound import certificate, qap_fitness

cert = certificate(F2, T2, qap_fitness(F2, T2, best_perm))
log(f"bigram-component certificate: within {cert['gap_pct']:.2f}% of optimal")

json.dump(
    {
        "canonical": canonical,
        "regrets": {"tri_under_comb": regret_tri, "comb_under_tri": regret_comb},
        "best": best,
        "n_distinct": len(distinct),
        "stable_slots": consensus_stable,
        "scoreboard_pct": {n: (base - v) / base * 100 for n, v in board.items() if v is not None},
        "bigram_certificate": cert,
    },
    open("runs/cond_rebuild.json", "w"),
    indent=2,
    default=float,
)
print("\nbest layout:")
for i in range(0, 30, 10):
    print("  " + " ".join(best[i : i + 10]))
print("ALL-DONE")
