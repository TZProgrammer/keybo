"""Resume of tune_sweep.py after the tmux crash (P5 bigram + trigram cands 0-3 done).

Replays the SAME rng (seed 90210, same draw order) to regenerate both candidate lists
identically, hardcodes the crash-surviving measurements from tune_sweep.log, runs ONLY
trigram candidates 4-7, applies the P5 rule, then runs P6 unchanged.

Known from the log:
  bigram: incumbent wmae 15.65, best challenger 15.65 -> adopt=False (VERDICT FINAL)
  cond-trigram cands 0-3: wmae 19.09 (incumbent), 19.25, 20.28, 18.79 — all tau +1.0;
  cand 3 (depth-2) currently LEADS by 1.6% relative.
"""

import json
import time
from collections import Counter

import numpy as np

from keybo.data.strokes import StrokeRow, load_strokes
from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.inspect import layout_diagnostics
from keybo.scoring.oxey import OxeyStyleScorer
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.training.train import train_bigram_model, train_trigram_model
from keybo.training.validate import validate

SEEDS = [0, 1]
QWERTY = NAMED_LAYOUTS["qwerty"]
geom = ROW_STAGGERED_30
N = 30
TARGET_WPM = 90.0
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def overall_wmae(report):
    return float(np.mean([m["wmae"] for f in report["folds"].values() for m in f["seeds"]]))


def min_tau(report):
    return float(min(p["tau_heldout"] for p in report["pooled"]))


def load_freq(path):
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            out[p[0]] = int(p[1])
    return out


ROOT = "/tmp/keybo_prod8/data/corpus"
bi_corpus = load_freq(f"{ROOT}/bigrams.txt")
sg_corpus = load_freq(f"{ROOT}/1-skip.txt")
tri_corpus = load_freq(f"{ROOT}/trigrams.txt")

bi_rows = load_strokes("bistrokes_v3.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
full_rows = {(r.layout, r.positions, r.ngram): r for r in load_strokes(
    "tristrokes_v1.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)}
cond_rows = []
for key, lr in {(r.layout, r.positions, r.ngram): r for r in load_strokes(
        "tristrokes_last.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)}.items():
    fr = full_rows.get(key)
    if fr is None or len(fr.samples) != len(lr.samples):
        continue
    samples = [
        (wl, dl, pl, hl)
        for (wf, df, pf, hf), (wl, dl, pl, hl) in zip(fr.samples, lr.samples)
        if (wf, pf, hf) == (wl, pl, hl) and 0 <= df - dl <= 5000
    ]
    if samples:
        cond_rows.append(StrokeRow(layout=key[0], positions=key[1], ngram=key[2],
                                   frequency=len(samples), samples=samples))
log(f"{len(bi_rows)} bigram rows, {len(cond_rows)} cond trigram rows")

# --- replay candidate generation exactly (same rng, same draw order) ---------------------
rng = np.random.default_rng(90210)


def candidates(n, base):
    out = [dict(base)]
    while len(out) < n:
        out.append({
            "n_estimators": int(rng.integers(150, 600)),
            "max_depth": int(rng.integers(2, 6)),
            "learning_rate": float(rng.uniform(0.02, 0.15)),
            "min_child_weight": int(rng.integers(1, 8)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        })
    return out


bi_cands = candidates(16, {})   # consumed to keep rng stream aligned
tri_cands = candidates(8, {})

# --- P5 verdicts: bigram final from log; trigram = 4 known + 4 to run --------------------
bi_best, bi_adopt = {}, False  # from log: incumbent 15.65 == best challenger, adopt=False
known_tri = {0: 19.09, 1: 19.25, 2: 20.28, 3: 18.79}  # all tau +1.0 (log)
board = [(tri_cands[i], w, 1.0) for i, w in known_tri.items()]
for i in range(4, 8):
    rep = validate(cond_rows, seeds=SEEDS, ngram="trigram", train_params=tri_cands[i], n_boot=10)
    w, tau = overall_wmae(rep), min_tau(rep)
    board.append((tri_cands[i], w, tau))
    log(f"P5 cond-trigram cand {i}: wmae {w:.2f} tau {tau:+.2f} {tri_cands[i]}")
best_tau = max(b[2] for b in board)
gated = sorted([(p, w) for p, w, t in board if t >= best_tau - 1e-9], key=lambda pw: pw[1])
incumbent_w = board[0][1]
tri_winner, winner_w = gated[0]
tri_adopt = tri_winner != tri_cands[0] and (incumbent_w - winner_w) / incumbent_w > 0.005
log(f"P5 cond-trigram: incumbent {incumbent_w:.2f}, best {winner_w:.2f}, adopt={tri_adopt} -> {tri_winner}")
json.dump(
    {"bigram": {"adopted": False, "winner": {}, "note": "from pre-crash log: all 16 tau +1.0, none beat 15.65"},
     "trigram": {"board": [(str(p), w, t) for p, w, t in board], "adopted": tri_adopt, "winner": tri_winner}},
    open("runs/p5_tune.json", "w"), indent=2,
)

# --- P6 ----------------------------------------------------------------------------------
log("P6: final models")
bi_models = [train_bigram_model(bi_rows, target_wpm=TARGET_WPM, random_state=s, n_jobs=0, **bi_best) for s in (0, 1, 2)]
tri_params = tri_winner if tri_adopt else {}
tri_models = [train_trigram_model(cond_rows, target_wpm=TARGET_WPM, random_state=s, n_jobs=0, **tri_params) for s in (0, 1, 2)]
for i, m in enumerate(bi_models):
    m.save(f"models/bigram_tuned_seed{i}.json")
for i, m in enumerate(tri_models):
    m.save(f"models/trigram_cond_tuned_seed{i}.json")

bts = [TableBigramScorer(m, bi_corpus, target_wpm=TARGET_WPM, chars=QWERTY) for m in bi_models]
T2 = np.mean([sc._T for sc in bts], axis=0)
positions31 = [*geom.slots, geom.space_position]
n31 = len(positions31)
vec_all = np.vstack(
    [trigram_features_from_positions(geom, (a, b, c), wpm=TARGET_WPM)
     for a in positions31 for b in positions31 for c in positions31]
)
Tcond = np.mean([m.predict(vec_all).reshape(n31, n31, n31) for m in tri_models], axis=0)
T3c = T2[:, :, None] + Tcond
char_idx = {c: i for i, c in enumerate(QWERTY)}
char_idx[" "] = N
charset = set(QWERTY) | {" "}
ks = [(char_idx[t[0]], char_idx[t[1]], char_idx[t[2]], f)
      for t, f in tri_corpus.items() if len(t) == 3 and all(c in charset for c in t)]
I3 = np.array([k[0] for k in ks]); J3 = np.array([k[1] for k in ks])
L3 = np.array([k[2] for k in ks]); F3 = np.array([k[3] for k in ks], dtype=float)
log("P6: corrected tables built")

oxey = OxeyStyleScorer(bi_corpus, sg_corpus, tri_corpus)
oxey_cache: dict[bytes, float] = {}


def lay_str(perm):
    slots = [""] * N
    for i, sl in enumerate(perm[:N]):
        slots[sl] = QWERTY[i]
    return "".join(slots)


def oxey_of(perm):
    key = perm.tobytes()
    if key not in oxey_cache:
        oxey_cache[key] = oxey.fitness(Layout(lay_str(perm), geom))
    return oxey_cache[key]


def fit_speed(perm):
    return float((F3 * T3c[perm[I3], perm[J3], perm[L3]]).sum())


p_q = bts[0].permutation(Layout(QWERTY, geom))
UNIT = fit_speed(p_q) / 100.0 / max(abs(oxey_of(p_q)), 1.0)
rng2 = np.random.default_rng(424243)


def search(w, restarts=12, iters=12_000):
    def fit(perm):
        v = fit_speed(perm)
        if w:
            v += w * UNIT * oxey_of(perm)
        return v

    deltas = []
    for _ in range(120):
        p = np.append(rng2.permutation(N), N)
        f1 = fit(p)
        i, j = rng2.integers(0, N, 2)
        p[i], p[j] = p[j], p[i]
        d = fit(p) - f1
        if d > 0:
            deltas.append(d)
    T0 = float(np.median(deltas)) / np.log(2)
    results = []
    for _ in range(restarts):
        perm = np.append(rng2.permutation(N), N)
        cur = fit(perm)
        temp = T0
        for _ in range(iters):
            i, j = rng2.integers(0, N, 2)
            if i == j:
                continue
            perm[i], perm[j] = perm[j], perm[i]
            cand = fit(perm)
            if cand - cur <= 0 or rng2.random() < np.exp(-(cand - cur) / temp):
                cur = cand
            else:
                perm[i], perm[j] = perm[j], perm[i]
            temp *= 0.9994
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    perm[i], perm[j] = perm[j], perm[i]
                    cand = fit(perm)
                    if cand < cur - 1e-9:
                        cur = cand
                        improved = True
                    else:
                        perm[i], perm[j] = perm[j], perm[i]
        results.append((cur, perm.copy()))
    results.sort(key=lambda t: t[0])
    return results


out = {}
speed0 = None
for w in (0.0, 0.5, 1.0, 2.0):
    results = search(w)
    best_fit, best_perm = results[0]
    lay = lay_str(best_perm)
    spd = fit_speed(best_perm)
    if w == 0.0:
        speed0 = spd
    top = [(f, lay_str(p)) for f, p in results if f <= best_fit * 1.005]
    distinct = list({ls for _, ls in top})
    shares = oxey.pattern_shares(Layout(lay, geom))
    diag = layout_diagnostics(Layout(lay, geom), bi_corpus)
    out[f"{w}"] = {
        "layout": lay,
        "speed_loss_pct": (spd - speed0) / speed0 * 100,
        "oxey_score": oxey_of(best_perm),
        "sfb_pct": shares["sfb"],
        "inroll_pct": shares["inroll"],
        "redirect_pct": shares["redirect"],
        "alternate_pct": shares["alternate"],
        "home_share": diag["row_share"]["home"],
        "n_distinct": len(distinct),
    }
    log(
        f"P6 w={w}: {lay} | speed loss {(spd - speed0) / speed0 * 100:+.3f}% | "
        f"sfb {shares['sfb']:.2f}% inroll {shares['inroll']:.2f}% alt {shares['alternate']:.1f}% "
        f"home {diag['row_share']['home']:.1%} | {len(distinct)} near-optima"
    )

json.dump(out, open("runs/p6_oxey_sweep.json", "w"), indent=2, default=float)
print("\n=== P6 LAYOUT FAMILY (tuned models, wpm 90) ===")
for w, r in out.items():
    print(f"\noxey weight {w}: speed cost {r['speed_loss_pct']:+.2f}%  sfb {r['sfb_pct']:.2f}%")
    for i in range(0, 30, 10):
        print("  " + " ".join(r["layout"][i : i + 10]))
print("ALL-DONE-P56")
