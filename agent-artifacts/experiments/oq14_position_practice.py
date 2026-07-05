"""OQ-14 resolution round: position-level practice — probe, model arm, sensitivity.

Pre-registered in state memory. Three tests, one run:

PROBE S  Skill-scaling discriminator. Within qwerty, the top-vs-home same-row gap by WPM
         band. Practice accumulates with reps, so a practice-driven gap should GROW with
         skill; a biomechanical gap should be ~flat (or shrink as ceiling effects bite).

ARM PU   Position-usage practice channel. Add u_a, u_b = log10(corpus letter frequency of
         the letters on the two keys) as features. At TRAINING time these encode how
         practiced each position is (its letter's frequency on the participant's own
         layout); at SERVE time the candidate layout's own assignment is used — the
         equilibrium assumption ("the user has practiced their layout"). Algebra: any
         additive dependence on u cancels in layout comparisons (every layout assigns the
         same letter multiset, so sum over corpus of f(bg)*(h(u_a)+h(u_b)) is
         layout-invariant up to the bigram weighting); what differs across layouts — and
         what can drain position-practice out of the geometry estimate — is the
         u x geometry INTERACTION. Judged by LOLO (tau must hold +1.0) AND the E5
         optimizer-side gate (per-bigram table search; home share must move toward sanity).

ARM DW   Dvorak-emphasis sensitivity. Retrain the shipped recipe with dvorak's total
         example weight raised to ~40% (others equalized); re-search. If home-row share of
         the optimum rises materially, the top-row preference was limited by dvorak's thin
         data (practice reading); if flat, the preference is robust in qwerty data itself.

Uses the corrected prod checkout /tmp/keybo_prod3 @ c02c92d (schema 2026-07-05.3, depth 3).
"""

import json
import time
from collections import defaultdict

import numpy as np

from keybo.cli.score import common_ngrams
from keybo.data.strokes import iqr_average, load_strokes
from keybo.features import bigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.training.train import fit_practice_term, layout_balance_weights
from keybo.training.validate import (
    _centered_spearman,
    aggregate_layout_table,
    build_cells,
    layout_ranking_tau,
)

SEEDS = [0, 1, 2]
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
K = 100.0
TARGET_WPM = 90.0
QWERTY = NAMED_LAYOUTS["qwerty"]
geom = ROW_STAGGERED_30
N = 30
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


rows = load_strokes("bistrokes_v3.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
corpus = {}
for line in open("/tmp/keybo_prod3/data/corpus/bigrams.txt"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2:
        corpus[p[0]] = int(p[1])

# Letter marginals from the corpus (the practice proxy: how often each LETTER is typed).
letter_freq = defaultdict(float)
for bg, f in corpus.items():
    for ch in bg:
        letter_freq[ch] += f
U = {ch: float(np.log10(f + 1.0)) for ch, f in letter_freq.items()}
U_DEFAULT = float(np.median(list(U.values())))

# Per-layout letter->position maps for TRAINING u (the participant's own layout).
from keybo.data.keystrokes import build_char_map

char_maps = {name: build_char_map(name) for name in ("qwerty", "azerty", "dvorak", "qwertz")}
pos_to_letter = {name: {pos: ch for ch, pos in cm.items()} for name, cm in char_maps.items()}


def u_of(layout_name, pos):
    ch = pos_to_letter[layout_name].get(tuple(pos))
    return U.get(ch, U_DEFAULT) if ch else U_DEFAULT


# ==== PROBE S: skill-scaling of qwerty's top-vs-home gap =================================
log("PROBE S: top-vs-home gap by wpm band (qwerty same-row bigrams)")
bands = [(40, 70), (70, 100), (100, 130)]
gap_table = {}
for lo, hi in bands:
    per_row = defaultdict(list)
    for row in rows:
        if row.layout != "qwerty":
            continue
        (x1, y1), (x2, y2) = row.positions
        if y1 != y2 or y1 not in (2, 3) or x1 == x2:
            continue
        durs = [d for w, d, _p, _h in row.samples if lo <= w < hi and 0 < d < 1000]
        if len(durs) >= 5:
            per_row[y1].append(float(np.median(durs)))
    top = float(np.median(per_row[3])) if per_row[3] else float("nan")
    home = float(np.median(per_row[2])) if per_row[2] else float("nan")
    gap_table[f"{lo}-{hi}"] = {
        "top": top,
        "home": home,
        "gap_home_minus_top": home - top,
        "n_top": len(per_row[3]),
        "n_home": len(per_row[2]),
    }
    log(f"  wpm {lo}-{hi}: top {top:.0f}ms home {home:.0f}ms gap {home - top:+.0f}ms "
        f"(n={len(per_row[3])}/{len(per_row[2])})")

# ==== shared training scaffolding ========================================================
feats_base, targets, ex_ngram, ex_layout, ex_n, ex_u = [], [], [], [], [], []
for row in rows:
    by_wpm = defaultdict(list)
    for wpm, duration, _pid, _hold in row.samples:
        by_wpm[wpm].append(duration)
    ua = u_of(row.layout, row.positions[0])
    ub = u_of(row.layout, row.positions[1])
    for wpm, durations in by_wpm.items():
        feats_base.append(bigram_features_from_positions(geom, row.positions, wpm=wpm))
        targets.append(iqr_average(durations))
        ex_ngram.append(row.ngram)
        ex_layout.append(row.layout)
        ex_n.append(len(durations))
        ex_u.append((ua, ub))
X_base = np.vstack(feats_base)
y = np.array(targets)
ex_ngram = np.array(ex_ngram, dtype=object)
ex_layout = np.array(ex_layout, dtype=object)
ex_n = np.array(ex_n)
ex_u = np.array(ex_u)
log(f"{len(y)} examples")

all_cells = build_cells(rows, **CELL_KW)
obs_table = aggregate_layout_table(all_cells)
LAYOUTS = sorted({c.layout for c in all_cells})
CEILINGS = json.load(open("runs/lolo_v3.json"))["ceilings"]

PU_NAMES = [*BIGRAM_FEATURE_NAMES, "u_first", "u_second"]


def fit_model(Xm, ym, seed, weight, names):
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=list(names),
        wpm_range=(60, 120),
        ngram="bigram",
    )
    m = XGBoostTypingModel(meta, random_state=seed, n_jobs=0)
    m._regressor.fit(Xm, ym, sample_weight=weight)
    m._fitted = True
    return m


def train_recipe(Xm, ym, ngrams, counts, layouts_, seed, names, weights=None):
    """The shipped R1W recipe on arbitrary feature matrices."""
    w = layout_balance_weights(layouts_) if weights is None else weights
    model = fit_model(Xm, ym, seed, w, names)
    bmap = {}
    for _ in range(2):
        bmap = fit_practice_term(ngrams, ym - model.predict(Xm), counts, k=K)
        bvec = np.array([bmap.get(g, 0.0) for g in ngrams])
        model = fit_model(Xm, ym - bvec, seed, w, names)
    return model, bmap


# u features for eval cells: use the CELL's own layout's letter map (matches training
# semantics: the typist practiced the layout the data came from).
def cell_u(c):
    return (u_of(c.layout, c.positions[0]), u_of(c.layout, c.positions[1]))


# ==== ARM PU: LOLO ========================================================================
log("ARM PU: LOLO with u_first/u_second position-usage features")
X_pu = np.column_stack([X_base, ex_u])
Xc_base = np.vstack(
    [bigram_features_from_positions(geom, c.positions, wpm=c.wpm) for c in all_cells]
)
Xc_pu = np.column_stack([Xc_base, np.array([cell_u(c) for c in all_cells])])

pu_pooled = {s: {} for s in SEEDS}
pu_fracs = []
for holdout in LAYOUTS:
    mask = ex_layout != holdout
    idx = [i for i, c in enumerate(all_cells) if c.layout == holdout]
    test_cells = [all_cells[i] for i in idx]
    obs = np.array([c.obs for c in test_cells])
    for seed in SEEDS:
        model, bmap = train_recipe(
            X_pu[mask], y[mask], ex_ngram[mask], ex_n[mask], ex_layout[mask], seed, PU_NAMES
        )
        pred = model.predict(Xc_pu[idx]) + np.array([bmap.get(c.ngram, 0.0) for c in test_cells])
        rho = _centered_spearman(test_cells, pred, obs)
        pu_fracs.append(rho / CEILINGS[holdout])
        pu_pooled[seed][holdout] = aggregate_layout_table(test_cells, pred)[holdout]
    log(f"  PU/{holdout}: last rho {rho:+.3f}")
pu_taus = [layout_ranking_tau(obs_table, pu_pooled[s]) for s in SEEDS]
log(f"  PU: tau {[f'{t:+.3f}' for t in pu_taus]} mean rho/ceiling {np.mean(pu_fracs):+.4f}")

# ==== ARM PU: E5 search gate (per-bigram generalized table) ==============================
log("ARM PU: training full-data models + per-bigram table search (E5 gate)")
pu_models = []
for seed in SEEDS:
    model, _ = train_recipe(X_pu, y, ex_ngram, ex_n, ex_layout, seed, PU_NAMES)
    pu_models.append(model)

compare_layouts = {n: Layout(s, geom) for n, s in NAMED_LAYOUTS.items()}
common = common_ngrams(corpus, list(compare_layouts.values()))
BGS = sorted(common)
F_vec = np.array([common[bg] for bg in BGS])
positions31 = [*geom.slots, geom.space_position]

# P[k, i, j]: predicted time of bigram k at position pair (i, j) — u features depend on
# the bigram's LETTERS (equilibrium: candidate layout assigns them), geometry on positions.
base_geo = np.vstack(
    [
        bigram_features_from_positions(geom, (a, b), wpm=TARGET_WPM)
        for a in positions31
        for b in positions31
    ]
)  # (961, F)
P_mean = None
for model in pu_models:
    P = np.empty((len(BGS), len(positions31) ** 2))
    for k, bg in enumerate(BGS):
        ua, ub = U.get(bg[0], U_DEFAULT), U.get(bg[1], U_DEFAULT)
        Xk = np.column_stack([base_geo, np.full(len(base_geo), ua), np.full(len(base_geo), ub)])
        P[k] = model.predict(Xk)
    P_mean = P if P_mean is None else P_mean + P
P_mean /= len(pu_models)
P_mean = P_mean.reshape(len(BGS), len(positions31), len(positions31))
log(f"  per-bigram table built: {P_mean.shape}")

# char index for each bigram under a permutation of QWERTY's charset.
char_idx = {c: i for i, c in enumerate(QWERTY)}
char_idx[" "] = N
bg_i = np.array([char_idx[bg[0]] for bg in BGS])
bg_j = np.array([char_idx[bg[1]] for bg in BGS])
K_BG = np.arange(len(BGS))


def fitness_pu(perm):
    return float((F_vec * P_mean[K_BG, perm[bg_i], perm[bg_j]]).sum())


rng = np.random.default_rng(4242)


def search(fit_fn, restarts=8, iters=25_000):
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
            temp *= 0.9993
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


def home_share(layout_str):
    lay = Layout(layout_str, geom)
    tot = hs = 0.0
    for ch, f in letter_freq.items():
        if ch != " " and lay.has_key(ch):
            tot += f
            if lay.pos(ch)[1] == 2:
                hs += f
    return hs / tot


fit_pu, perm_pu = search(fitness_pu)
lay_pu = lay_str(perm_pu)
hs_pu = home_share(lay_pu)
log(f"  PU-best: {lay_pu}  fitness {fit_pu:.4e}  HOME SHARE {hs_pu:.1%} (d3 was 33.7%)")

# ==== ARM DW: dvorak-emphasis sensitivity ================================================
log("ARM DW: dvorak-emphasis retrain (dvorak ~40% of total weight) + search")
share = defaultdict(float)
for la in ex_layout:
    share[la] += 1.0
target = {"dvorak": 0.40, "qwerty": 0.20, "azerty": 0.20, "qwertz": 0.20}
w_dw = np.array([target[la] * len(ex_layout) / share[la] for la in ex_layout])
w_dw = np.clip(w_dw, None, 200.0)
w_dw /= w_dw.mean()

from keybo.scoring.table_scorer import TableBigramScorer

T_dw = None
for seed in SEEDS:
    model, _ = train_recipe(
        X_base, y, ex_ngram, ex_n, ex_layout, seed, BIGRAM_FEATURE_NAMES, weights=w_dw
    )
    sc = TableBigramScorer(model, corpus, target_wpm=TARGET_WPM, chars=QWERTY)
    T_dw = sc._T if T_dw is None else T_dw + sc._T
T_dw /= len(SEEDS)
F_mat = sc._F


def fitness_dw(perm):
    return float((F_mat * T_dw[np.ix_(perm, perm)]).sum())


fit_dw, perm_dw = search(fitness_dw)
lay_dw = lay_str(perm_dw)
hs_dw = home_share(lay_dw)
log(f"  DW-best: {lay_dw}  fitness {fit_dw:.4e}  HOME SHARE {hs_dw:.1%}")

# ==== summary ============================================================================
out = {
    "probe_s": gap_table,
    "arm_pu": {
        "taus": pu_taus,
        "mean_frac": float(np.mean(pu_fracs)),
        "layout": lay_pu,
        "home_share": hs_pu,
    },
    "arm_dw": {"layout": lay_dw, "home_share": hs_dw},
    "reference": {"d3_home_share": 0.337, "qwerty_home_share": 0.320},
}
json.dump(out, open("runs/oq14_position_practice.json", "w"), indent=2, default=float)
print("\n=== OQ-14 SUMMARY ===")
print("probe S (home-minus-top gap by wpm):", {k: round(v["gap_home_minus_top"], 1) for k, v in gap_table.items()})
print(f"arm PU: tau {pu_taus} rho/ceiling {np.mean(pu_fracs):+.4f} home-share {hs_pu:.1%}")
print(f"arm DW: home-share {hs_dw:.1%}")
print("ALL-DONE")
