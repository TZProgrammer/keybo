"""Finger-load frontier: what does utilization balancing COST in predicted speed?

Sweep the finger-load weight w over {0, 20, 50, 100, 200}; for each w, search the
combined objective speed_table + w * finger_load (table-speed: the load term is a cheap
per-permutation sum, computed from per-slot letter weights). Report per w: the winner,
its predicted-speed loss vs the w=0 winner, its max/min finger loads, and the E5 profile.
The pre-registered sanity expectations: load spread should shrink monotonically with w;
speed loss should be small at moderate w (plateau logic: many near-optimal layouts exist,
some of them balanced); a large speed cliff would mean balance genuinely fights speed.
"""

import json
import time

import numpy as np

from keybo.geometry import ROW_STAGGERED_30, Finger
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.comfort import DEFAULT_FINGER_CAPACITY
from keybo.scoring.inspect import layout_diagnostics
from keybo.scoring.table_scorer import TableBigramScorer

SEEDS = [0, 1, 2]
QWERTY = NAMED_LAYOUTS["qwerty"]
geom = ROW_STAGGERED_30
N = 30
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


corpus = {}
for line in open("/tmp/keybo_prod6/data/corpus/bigrams.txt"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 2:
        corpus[p[0]] = int(p[1])

models = [XGBoostTypingModel.load(f"models/bigram_d3_seed{s}.json") for s in SEEDS]
scs = [TableBigramScorer(m, corpus, target_wpm=90.0, chars=QWERTY) for m in models]
T = np.mean([sc._T for sc in scs], axis=0)
F = scs[0]._F
log("speed tables ready")

# Per-CHAR corpus weight (both slots of every bigram) and per-slot finger capacity —
# the load term becomes a permutation-indexed sum, as cheap as the speed term.
char_w = np.zeros(N + 1)
total_w = 0.0
for bg, f in corpus.items():
    if len(bg) != 2:
        continue
    for ch in bg:
        idx = QWERTY.find(ch) if ch != " " else N
        if idx >= 0:
            char_w[idx] += f / 2
            total_w += f / 2
char_w /= total_w  # per-char share of keystroke weight

positions = [*geom.slots, geom.space_position]
_FINGER_LABEL = {
    Finger.LP: "L-pinky", Finger.LR: "L-ring", Finger.LM: "L-middle", Finger.LI: "L-index",
    Finger.RI: "R-index", Finger.RM: "R-middle", Finger.RR: "R-ring", Finger.RP: "R-pinky",
    Finger.THUMB: "thumb",
}
slot_finger = [ _FINGER_LABEL[geom.finger(pos[0])] for pos in positions ]
fingers = sorted(set(slot_finger))
f_idx = {f: i for i, f in enumerate(fingers)}
slot_fidx = np.array([f_idx[f] for f in slot_finger])
capacity = np.array([DEFAULT_FINGER_CAPACITY[f] for f in fingers])
SCALE = 1000.0


def load_penalty(perm):
    loads = np.zeros(len(fingers))
    np.add.at(loads, slot_fidx[perm], char_w)
    return SCALE * float((loads * loads / capacity).sum())


# The speed term is ~5e10-scale; normalize the load weight so w is in "percent-of-speed"
# units: w=100 => load term can move fitness by ~1% of the qwerty speed value.
p_q = scs[0].permutation(Layout(QWERTY, geom))
speed_q = float((F * T[np.ix_(p_q, p_q)]).sum())
UNIT = speed_q / 100.0 / load_penalty(p_q)  # w=100 ~ load term ~1% of speed at qwerty-imbalance

rng = np.random.default_rng(778899)


def search(w, restarts=10, iters=30_000):
    def fit(perm):
        v = float((F * T[np.ix_(perm, perm)]).sum())
        if w:
            v += w * UNIT * load_penalty(perm)
        return v

    deltas = []
    for _ in range(150):
        p = np.append(rng.permutation(N), N)
        f1 = fit(p)
        i, j = rng.integers(0, N, 2)
        p[i], p[j] = p[j], p[i]
        d = fit(p) - f1
        if d > 0:
            deltas.append(d)
    T0 = float(np.median(deltas)) / np.log(2)
    best, best_p = np.inf, None
    for _ in range(restarts):
        perm = np.append(rng.permutation(N), N)
        cur = fit(perm)
        temp = T0
        for _ in range(iters):
            i, j = rng.integers(0, N, 2)
            if i == j:
                continue
            perm[i], perm[j] = perm[j], perm[i]
            cand = fit(perm)
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
                    cand = fit(perm)
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


def speed_of(perm):
    return float((F * T[np.ix_(perm, perm)]).sum())


out = {}
speed0 = None
for w in (0.0, 20.0, 50.0, 100.0, 200.0):
    _, perm = search(w)
    lay = lay_str(perm)
    spd = speed_of(perm)
    if w == 0.0:
        speed0 = spd
    diag = layout_diagnostics(Layout(lay, geom), corpus)
    loads = {k: v for k, v in diag["finger_load"].items() if k != "thumb"}
    mx, mn = max(loads.values()), min(loads.values())
    out[f"{w:.0f}"] = {
        "layout": lay,
        "speed_loss_pct": (spd - speed0) / speed0 * 100,
        "max_load": mx,
        "min_load": mn,
        "spread": mx - mn,
        "pinky_load": loads["L-pinky"] + loads["R-pinky"],
    }
    log(
        f"w={w:5.0f}: speed loss {(spd - speed0) / speed0 * 100:+.3f}% | max load {mx:.1%} "
        f"min {mn:.1%} spread {mx - mn:.1%} | pinkies {loads['L-pinky'] + loads['R-pinky']:.1%} | {lay}"
    )

json.dump(out, open("runs/fingerload_frontier.json", "w"), indent=2)
print("ALL-DONE")
