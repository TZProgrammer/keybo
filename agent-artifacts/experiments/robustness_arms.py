"""Robustness round (user questions #10): the median-beats-mean result, propagated.

Arms over the bigram frame (hardened harness: rho/ceiling + tau + wmae + umae + deciles):
  T-BASE  incumbent: IQR-mean targets, MSE loss, session-mean wpm (anchor)
  T-MED   cell target = median
  T-MAE   loss = reg:absoluteerror
  T-CAP   hesitation filter: drop samples > 3x the session's per-key pace before agg
  P-MED   pace label = session MEDIAN (robust-location, end-to-end)
  P-M5    pace label = full blind blend (log-space: median + participant prior + local)

One extraction pass carries all labels; eval cells rebuilt per arm where the target
statistic changes (ceilings recomputed — the ceiling is target-definition-dependent).
"""

import csv
import json
import os
import time
from collections import defaultdict

import numpy as np

from keybo.data.keystrokes import (
    BANNED_KEYS,
    _letter,
    build_char_map,
    compute_session_wpm,
    group_sessions,
    load_participant_metadata,
    mark_correct_flags,
)
from keybo.data.strokes import StrokeRow, iqr_average
from keybo.features import bigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
from keybo.geometry import ROW_STAGGERED_30
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.training.train import fit_practice_term, layout_balance_weights
from keybo.training.validate import (
    Cell,
    _centered_spearman,
    aggregate_layout_table,
    calibration_slope,
    freq_decile_mae,
    layout_ranking_tau,
    split_half_ceiling,
    uniform_mae,
    weighted_mae,
)

SEEDS = [0, 1]
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
K = 100.0
CAP_MS = 2000.0
SHRINK_SESS = 3.0
geom = ROW_STAGGERED_30
t0 = time.time()
FILES = "dataset/Keystrokes/files"


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


metadata = load_participant_metadata("dataset/Keystrokes/files/metadata_participants.txt")
char_maps = {n: build_char_map(n) for n in ("qwerty", "azerty", "dvorak", "qwertz")}

# pass 1: session medians per pid (for the participant prior)
log("pass 1: session medians for participant priors")
sess_meds = defaultdict(list)
file_list = []
for fname in sorted(os.listdir(FILES)):
    if not fname.endswith("_keystrokes.txt"):
        continue
    pid_s = fname.split("_")[0]
    md = metadata.get(pid_s)
    if not md:
        continue
    file_list.append((fname, pid_s, md["LAYOUT"]))
for fname, pid_s, layout in file_list:
    cmap = char_maps[layout]
    allowed = set(cmap)
    with open(os.path.join(FILES, fname), newline="", encoding="utf-8", errors="replace") as f:
        rows_raw = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    for sess in group_sessions(rows_raw).values():
        if not sess:
            continue
        expected = sess[0].get("SENTENCE") or ""
        single = [(i, r) for i, r in enumerate(sess) if len(_letter(r)) == 1]
        if not single:
            continue
        typed = "".join(_letter(r) for _, r in single)
        flags = mark_correct_flags(typed, expected)
        correct = [(i, r) for (i, r), ok in zip(single, flags, strict=False) if ok]
        ivals = []
        for a, b in zip(correct, correct[1:]):
            try:
                t1, t2 = float(a[1]["PRESS_TIME"]), float(b[1]["PRESS_TIME"])
            except (TypeError, ValueError, KeyError):
                continue
            if b[0] - a[0] == 1 and 0 < t2 - t1 < CAP_MS:
                ivals.append(t2 - t1)
        if len(ivals) >= 8:
            sess_meds[pid_s].append(float(np.median(ivals)))
global_med = float(np.median([m for v in sess_meds.values() for m in v]))
log(f"priors ready ({len(sess_meds)} pids, global med {global_med:.0f}ms)")


def to_wpm(interval_ms):
    return 12000.0 / max(interval_ms, 1.0)


# pass 2: occurrences with all labels
log("pass 2: occurrences with (mean-wpm, med-wpm, m5-wpm, hesitation flag)")
acc = defaultdict(list)  # key -> [(wm, wmed, wm5, dur, pid, hes)]
for fname, pid_s, layout in file_list:
    cmap = char_maps[layout]
    allowed = set(cmap)
    with open(os.path.join(FILES, fname), newline="", encoding="utf-8", errors="replace") as f:
        rows_raw = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    for sess in group_sessions(rows_raw).values():
        if not sess:
            continue
        expected = sess[0].get("SENTENCE") or ""
        single = [(i, r) for i, r in enumerate(sess) if len(_letter(r)) == 1]
        if not single:
            continue
        typed = "".join(_letter(r) for _, r in single)
        flags = mark_correct_flags(typed, expected)
        correct = [(i, r) for (i, r), ok in zip(single, flags, strict=False) if ok]
        if len(correct) < 3:
            continue
        times = []
        for _i, r in correct:
            try:
                times.append(float(r["PRESS_TIME"]))
            except (TypeError, ValueError, KeyError):
                times.append(None)
        if times[0] is None or times[-1] is None:
            continue
        w_mean = compute_session_wpm(times[0], times[-1], len(correct))
        ivals = [
            times[i + 1] - times[i]
            for i in range(len(correct) - 1)
            if times[i] is not None and times[i + 1] is not None
            and correct[i + 1][0] - correct[i][0] == 1 and 0 < times[i + 1] - times[i] < CAP_MS
        ]
        if len(ivals) < 8:
            continue
        s_med = float(np.median(ivals))
        w_med = to_wpm(s_med)
        others = [m for m in sess_meds.get(pid_s, []) if m != s_med]
        prior = (
            (np.sum(others) + SHRINK_SESS * global_med) / (len(others) + SHRINK_SESS)
            if others else global_med
        )
        # M5-style blend in log space (fixed weights from the frontier probe: the blend
        # was ~55/45 median/prior with small local terms — use 0.55/0.45, no local).
        w_m5 = to_wpm(float(np.exp(0.55 * np.log(s_med) + 0.45 * np.log(prior))))
        hes_cut = 3.0 * s_med
        for i in range(len(correct) - 1):
            if correct[i + 1][0] - correct[i][0] != 1:
                continue
            la, lb = _letter(correct[i][1]), _letter(correct[i + 1][1])
            t1, t2 = times[i], times[i + 1]
            if t1 is None or t2 is None or not (0 < t2 - t1 < 5000):
                continue
            if la.upper() in BANNED_KEYS or lb.upper() in BANNED_KEYS:
                continue
            if la.lower() not in allowed or lb.lower() not in allowed:
                continue
            dur = t2 - t1
            acc[(layout, (cmap[la.lower()], cmap[lb.lower()]), la.lower() + lb.lower())].append(
                (int(w_mean), int(round(w_med)), int(round(w_m5)), int(dur), int(pid_s), dur > hes_cut)
            )
log(f"extraction done: {sum(len(v) for v in acc.values())} occurrences, {len(acc)} keys")

ARMS = ["T-BASE", "T-MED", "T-MAE", "T-CAP", "P-MED", "P-M5"]


def arm_rows(arm):
    """StrokeRows with the arm's wpm label and sample filter."""
    wi = {"T-BASE": 0, "T-MED": 0, "T-MAE": 0, "T-CAP": 0, "P-MED": 1, "P-M5": 2}[arm]
    rows = []
    for (layout, positions, ngram), samples in acc.items():
        keep = [
            (s[wi], s[3], s[4], 0)
            for s in samples
            if not (arm == "T-CAP" and s[5])
        ]
        if keep:
            rows.append(StrokeRow(layout=layout, positions=positions, ngram=ngram,
                                  frequency=len(keep), samples=keep))
    return rows


def agg_fn(arm):
    if arm == "T-MED":
        return lambda durs: float(np.median(durs))
    return iqr_average


def build_cells_agg(rows, agg):
    cells = []
    for r in rows:
        by_bucket = defaultdict(list)
        for wpm, dur, pid, hold in r.samples:
            if not CELL_KW["wpm_lo"] <= wpm < CELL_KW["wpm_hi"]:
                continue
            b = CELL_KW["wpm_lo"] + ((wpm - CELL_KW["wpm_lo"]) // CELL_KW["bucket_width"]) * CELL_KW["bucket_width"]
            by_bucket[b].append((wpm, dur, pid, hold))
        for b, items in by_bucket.items():
            if len(items) < CELL_KW["min_cell_samples"]:
                continue
            cells.append(Cell(
                layout=r.layout, ngram=r.ngram, positions=r.positions, frequency=len(r.samples),
                bucket=b, wpm=b + CELL_KW["bucket_width"] / 2,
                obs=agg([d for _, d, _, _ in items]), n=len(items), samples=items,
            ))
    return cells


def fit_model(Xm, ym, seed, weight, objective):
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION, feature_names=list(BIGRAM_FEATURE_NAMES),
        wpm_range=(60, 120), ngram="bigram",
    )
    kw = {"objective": objective} if objective else {}
    m = XGBoostTypingModel(meta, random_state=seed, n_jobs=0, **kw)
    m._regressor.fit(Xm, ym, sample_weight=weight)
    m._fitted = True
    return m


results = {}
for arm in ARMS:
    rows = arm_rows(arm)
    agg = agg_fn(arm)
    objective = "reg:absoluteerror" if arm == "T-MAE" else None
    cells = build_cells_agg(rows, agg)
    obs_table = aggregate_layout_table(cells)
    LAYOUTS = sorted({c.layout for c in cells})
    ceilings = {}
    for lay in LAYOUTS:
        ceilings[lay] = split_half_ceiling(
            [r for r in rows if r.layout == lay], n_boot=20, seed=0, **CELL_KW
        )
    # training examples: group by (row, wpm-label); target via arm's agg
    feats, targets, ngrams_, layouts_, counts = [], [], [], [], []
    for r in rows:
        by_wpm = defaultdict(list)
        for wpm, dur, pid, hold in r.samples:
            by_wpm[wpm].append(dur)
        for wpm, durs in by_wpm.items():
            feats.append(bigram_features_from_positions(geom, r.positions, wpm=wpm))
            targets.append(agg(durs))
            ngrams_.append(r.ngram)
            layouts_.append(r.layout)
            counts.append(float(len(durs)))
    X = np.vstack(feats); y = np.array(targets)
    ngrams_ = np.array(ngrams_, dtype=object); layouts_ = np.array(layouts_, dtype=object)
    counts = np.array(counts)
    Xc = np.vstack([bigram_features_from_positions(geom, c.positions, wpm=c.wpm) for c in cells])
    pooled = {s: {} for s in SEEDS}
    fracs, wmaes, umaes, dec3 = [], [], [], []
    for holdout in LAYOUTS:
        mask = layouts_ != holdout
        idx = [i for i, c in enumerate(cells) if c.layout == holdout]
        test_cells = [cells[i] for i in idx]
        obs = np.array([c.obs for c in test_cells])
        for seed in SEEDS:
            w = layout_balance_weights(layouts_[mask])
            model = fit_model(X[mask], y[mask], seed, w, objective)
            bmap = {}
            for _ in range(2):
                bmap = fit_practice_term(ngrams_[mask], y[mask] - model.predict(X[mask]), counts[mask], k=K)
                bvec = np.array([bmap.get(g, 0.0) for g in ngrams_[mask]])
                model = fit_model(X[mask], y[mask] - bvec, seed, w, objective)
            pred = model.predict(Xc[idx]) + np.array([bmap.get(c.ngram, 0.0) for c in test_cells])
            fracs.append(_centered_spearman(test_cells, pred, obs) / ceilings[holdout])
            wmaes.append(weighted_mae(test_cells, pred, obs))
            umaes.append(uniform_mae(pred, obs))
            dm = freq_decile_mae(test_cells, pred, obs)
            dec3.append(np.mean([dm[d] for d in (1, 2, 3) if d in dm]))
            pooled[seed][holdout] = aggregate_layout_table(test_cells, pred)[holdout]
    taus = [layout_ranking_tau(obs_table, pooled[s]) for s in SEEDS]
    results[arm] = {
        "taus": taus, "rho_frac": float(np.mean(fracs)), "wmae": float(np.mean(wmaes)),
        "umae": float(np.mean(umaes)), "dec3": float(np.mean(dec3)), "n_cells": len(cells),
    }
    r = results[arm]
    log(f"{arm}: tau {[f'{t:+.2f}' for t in taus]} rho/ceil {r['rho_frac']:+.4f} "
        f"wmae {r['wmae']:.2f} umae {r['umae']:.2f} dec3 {r['dec3']:.2f} cells {r['n_cells']}")

json.dump(results, open("runs/robustness_arms.json", "w"), indent=2, default=float)
print("\n=== ROBUSTNESS SCOREBOARD ===")
for arm, r in results.items():
    print(f"{arm:<7} tau {[f'{t:+.2f}' for t in r['taus']]} rho/ceil {r['rho_frac']:+.4f} "
          f"wmae {r['wmae']:.2f} umae {r['umae']:.2f} dec3(rare) {r['dec3']:.2f}")
print("ALL-DONE")
