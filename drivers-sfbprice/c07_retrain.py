"""THE RETRAIN ARM + the HIGH-WPM GATE: does the correction survive an actual REFIT?

My INVARIANT-A route is a post-hoc SURFACE EDIT. That is auditable to machine precision (gates
A2/A3/A4 all pass to ~1e-14) but it is not a model, so two things it cannot answer:

  Q1  If the same-finger emphasis is put in the TRAINING instead, does the fitted surface move the
      same way? A surcharge is a claim about what the model SHOULD say; a refit is what it DOES say
      when told to weigh same-finger rows more.
  Q2  Does the corrected model REGRESS ACCURACY at high wpm? The high-wpm gate grades a MODEL's
      per-bucket rho against a baseline, so it is only REACHABLE through a refit -- a surface edit
      has no rho. Per GATESUPPORT-1 I read the `support` dict and treat an azerty-b120-only refusal
      as instability, not a verdict.

ARM-W: retrain the bigram model with SAME-FINGER training rows up-weighted (weight w on rows whose
       position pair is same-finger, 1 elsewhere), which is the "retrain with same_finger
       emphasized" route my brief offered. Then MEASURE the resulting price -- do not assume the
       knob worked ("present != effective" has bitten this repo three times).
ARM-0: the same recipe at w=1, i.e. the shipped recipe reproduced through MY code path. The
       positive control: it must reproduce the shipped model's price.

Writes models ONLY to my own dir. data/models/k31/ is NEVER touched.
"""
import json
import os
import sys
import time

import numpy as np
from _guard import ART, BI, MIN_N, OUT, SERVE, assert_d5

t0 = time.time()
def log(m): print(f"[{time.time() - t0:8.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

import xgboost  # noqa: E402
log(f"xgboost {xgboost.__version__} OMP={os.environ.get('OMP_NUM_THREADS')}")

import surface  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import bigram_features_from_positions, classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31, ROW_STAGGERED_30 as G30  # noqa: E402
from keybo.scoring.table_scorer import TableBigramScorer  # noqa: E402
from keybo.training import train as TR  # noqa: E402
from keybo.training.validate import build_cells  # noqa: E402

WEIGHTS = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "4", "16"])]
MODELS = f"{OUT}/models"
os.makedirs(MODELS, exist_ok=True)

log(f"loading {BI}")
rows = load_strokes(BI, ngram_len=2, wpm_threshold=0, min_samples=1)
assert len(rows) == 2202, f"frame drift: {len(rows)} != 2202"
log(f"  {len(rows)} bigram rows")

# ---- which ROWS are same-finger? (the same predicate the feature uses) ----
G31 = ROW_STAGGERED_31
sf_row = {}
n_sf = 0
for r in rows:
    a, b = r.positions
    is_sf = bool(C.same_finger(G31, a, b) and a != b)
    sf_row[id(r)] = is_sf
    n_sf += is_sf
log(f"  {n_sf} of {len(rows)} rows are same-finger ({100 * n_sf / len(rows):.1f}%)")


def train_weighted(w_sf, seed):
    """The shipped bigram recipe, with same-finger EXAMPLE WEIGHTS multiplied by w_sf.

    Implemented by monkeypatching `layout_balance_weights` for the duration of the call, so
    everything else -- the LOGRAT target, the practice-term backfit, the layout weights, the
    feature frame -- is the SHIPPED code path byte for byte. The row->weight map is built from
    the same `rows` list the trainer walks, in the same order, and I ASSERT the length matches
    so a silent misalignment cannot pass.
    """
    orig = TR.layout_balance_weights
    # the trainer builds examples row-major: one example per (row, wpm-group)
    per_example_sf = []
    for r in rows:
        by_wpm = {}
        for wpm, dur, _pid, _hold in r.samples:
            by_wpm.setdefault(wpm, []).append(dur)
        per_example_sf.extend([sf_row[id(r)]] * len(by_wpm))
    mult = np.where(np.array(per_example_sf), w_sf, 1.0)

    def patched(layouts):
        base = orig(layouts)
        if base is None:
            base = np.ones(len(layouts), dtype=float)
        assert len(base) == len(mult), (
            f"weight misalignment: trainer built {len(base)} examples, my same-finger map has "
            f"{len(mult)} -- refusing to train on a misaligned weight vector")
        return np.asarray(base, float) * mult

    TR.layout_balance_weights = patched
    try:
        return TR.train_bigram_model(rows, target_wpm=surface.WPM, geometry=ROW_STAGGERED_31,
                                     random_state=seed, n_jobs=48)
    finally:
        TR.layout_balance_weights = orig


# ---- the contrast, measured on whatever T2 comes out ----
POS31 = [*G31.slots, G31.space_position]
PIDX = {p: i for i, p in enumerate(POS31)}
log("building the serve-bucket pair aggregation for the price measurement")
cells = build_cells(rows, 40, 140, 20, 1)
agg = {}
for c in cells:
    if c.bucket != SERVE:
        continue
    try:
        a, b = PIDX[tuple(int(v) for v in c.positions[0])], PIDX[tuple(int(v) for v in c.positions[1])]
    except KeyError:
        continue
    agg.setdefault((a, b), []).extend(float(s[1]) for s in c.samples)
SEL = [(a, b) for (a, b), v in agg.items() if len(v) >= MIN_N and a != b and a < 30 and b < 30]
RAWMED = {k: float(np.median(agg[k])) for k in SEL}
SAME = [k for k in SEL if G31.finger(POS31[k[0]][0]) == G31.finger(POS31[k[1]][0])]
OTHER = [k for k in SEL if k not in set(SAME)]
raw_pen = float(np.median([RAWMED[k] for k in SAME]) - np.median([RAWMED[k] for k in OTHER]))
log(f"  {len(SAME)} same / {len(OTHER)} other pairs; RAW penalty {raw_pen:+.4f} ms")


def price_of(model):
    ph = "qwertyuiopasdfghjkl;zxcvbnm,./'"[: len(G30.slots)]
    T2 = np.asarray(TableBigramScorer(model, {}, target_wpm=surface.WPM, chars=ph,
                                     geometry=G30)._T, float)
    ms = float(np.median([T2[a, b] for a, b in SAME]))
    mo = float(np.median([T2[a, b] for a, b in OTHER]))
    return {"model_median_same": ms, "model_median_other": mo, "model_penalty": ms - mo}, T2


R = {"raw_penalty": raw_pen, "n_same_pairs": len(SAME), "n_other_pairs": len(OTHER),
     "n_sf_rows": n_sf, "arms": {}}
log("")
log("=== ARM-W: retrain with same-finger example weights (measure, do not assume) ===")
for w in WEIGHTS:
    arm = {}
    for seed in (0, 1, 2):
        ts = time.time()
        m = train_weighted(w, seed)
        pr, T2 = price_of(m)
        p = f"{MODELS}/bigram_sfw{w:g}_seed{seed}.json"
        m.save(p)
        np.save(f"{MODELS}/T2_sfw{w:g}_seed{seed}.npy", T2)
        arm[f"seed{seed}"] = {**pr, "wall_s": time.time() - ts, "model": p,
                             "practice_n": (m.metadata.extra["training"]["practice_term"] or {})
                             .get("n_ngrams")}
        log(f"  w={w:<5g} seed{seed}: model penalty {pr['model_penalty']:+8.4f} ms  "
            f"(raw target {raw_pen:+.2f})  [{time.time() - ts:.0f}s]")
    pens = [arm[f"seed{s}"]["model_penalty"] for s in (0, 1, 2)]
    T2mean = np.mean([np.load(f"{MODELS}/T2_sfw{w:g}_seed{s}.npy") for s in (0, 1, 2)], axis=0)
    ms = float(np.median([T2mean[a, b] for a, b in SAME]))
    mo = float(np.median([T2mean[a, b] for a, b in OTHER]))
    arm["seed_mean_table_penalty"] = ms - mo
    arm["mean_of_per_seed_penalties"] = float(np.mean(pens))
    arm["ratio_vs_raw"] = (ms - mo) / raw_pen
    R["arms"][f"w={w:g}"] = arm
    log(f"  w={w:<5g} SEED-MEAN TABLE penalty {ms - mo:+8.4f} ms  ratio vs raw "
        f"{(ms - mo) / raw_pen:.4f}   (shipped ratio 0.651)")

# positive control: w=1 must reproduce the SHIPPED model's price
ship_pr, _ = price_of(surface.load_shipped_model("bigram_reg31_seed0"))
w1 = R["arms"].get("w=1", {}).get("seed0", {})
R["positive_control_w1_vs_shipped"] = {
    "shipped_seed0_penalty": ship_pr["model_penalty"],
    "my_w1_seed0_penalty": w1.get("model_penalty"),
    "abs_diff": (abs(w1["model_penalty"] - ship_pr["model_penalty"]) if w1 else None)}
log("")
log(f"POSITIVE CONTROL: shipped seed0 penalty {ship_pr['model_penalty']:+.4f} vs my w=1 seed0 "
    f"{w1.get('model_penalty', float('nan')):+.4f}  |diff| "
    f"{R['positive_control_w1_vs_shipped']['abs_diff']:.4f}")

json.dump(R, open(f"{ART}/c07_retrain.json", "w"), indent=1)
log(f"wrote {ART}/c07_retrain.json")
log("ALL-DONE")
