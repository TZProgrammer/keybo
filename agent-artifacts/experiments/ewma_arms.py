"""Session-seeded EWMA local speed (user proposal, monkeytype-style): the interpolation.

local_speed := alpha * local_speed + (1-alpha) * instantaneous_rate, SEEDED at session
wpm. alpha->1 recovers the incumbent exactly, so this family CONTAINS the champion — the
question is whether any alpha < 1 beats it. This is genuinely untested: prior local arms
were REPLACEMENTS (trailing-median L8) or raw signals (PREV), not a session-anchored
smoother.

Typos/modifiers/deletions (the user's question): the contiguity machinery answers it —
the EWMA updates ONLY on clean intervals (consecutive original stream indices, parseable
times, interval < 2000ms hesitation cap) and FREEZES across mistypes, control keys,
deletions, and pauses, resuming at the next clean interval. The seed is the session wpm
(uses no information the incumbent doesn't already use).

Arms (bigram target, shipped R1W recipe, shared extraction/folds; S anchors internally):
  S      session wpm (anchor)
  ER90   EWMA(alpha=0.90) REPLACING session
  ER98   EWMA(alpha=0.98) replacing
  ES90   session + EWMA(0.90) as a second feature

Rule (preregistered): adopt iff tau >= anchor AND mean rho/ceiling > S + 0.005.
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
from keybo.data.strokes import iqr_average
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
    layout_ranking_tau,
    split_half_ceiling,
)

SEEDS = [0, 1, 2]
FILES = "dataset/Keystrokes/files"
META = "dataset/Keystrokes/files/metadata_participants.txt"
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
K = 100.0
CAP_MS = 2000.0
geom = ROW_STAGGERED_30
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def extract(records, char_map, layout, pid):
    """Bigram occurrences with (session_wpm, ewma90, ewma98) per occurrence."""
    if not records:
        return []
    expected = records[0].get("SENTENCE") or ""
    single = [(i, r) for i, r in enumerate(records) if len(_letter(r)) == 1]
    if not single:
        return []
    typed = "".join(_letter(r) for _, r in single)
    flags = mark_correct_flags(typed, expected)
    correct = [(i, r) for (i, r), ok in zip(single, flags, strict=False) if ok]
    if not correct:
        return []
    times = []
    for _i, r in correct:
        try:
            times.append(float(r["PRESS_TIME"]))
        except (TypeError, ValueError, KeyError):
            times.append(None)
    try:
        swpm = compute_session_wpm(times[0], times[-1], len(correct))
    except TypeError:
        return []
    allowed = set(char_map)
    out = []
    e90 = e98 = float(swpm)  # session-seeded
    for i in range(len(correct) - 1):
        contiguous = correct[i + 1][0] - correct[i][0] == 1
        la, lb = _letter(correct[i][1]), _letter(correct[i + 1][1])
        t1, t2 = times[i], times[i + 1]
        clean = (
            contiguous
            and t1 is not None
            and t2 is not None
            and 0 < (t2 - t1) < CAP_MS
            and len(la) == 1
            and len(lb) == 1
        )
        if clean and la.upper() not in BANNED_KEYS and lb.upper() not in BANNED_KEYS \
                and la.lower() in allowed and lb.lower() in allowed:
            out.append(
                (
                    layout,
                    (char_map[la.lower()], char_map[lb.lower()]),
                    la.lower() + lb.lower(),
                    swpm,
                    e90,
                    e98,
                    int(t2 - t1),
                    pid,
                )
            )
        if clean:  # update ONLY on clean intervals; freeze otherwise
            rate = 12000.0 / (t2 - t1)
            e90 = 0.90 * e90 + 0.10 * rate
            e98 = 0.98 * e98 + 0.02 * rate
    return out


metadata = load_participant_metadata(META)
char_maps = {n: build_char_map(n) for n in ("qwerty", "azerty", "dvorak", "qwertz")}
occ = []
n_files = 0
for fname in os.listdir(FILES):
    if not fname.endswith("_keystrokes.txt"):
        continue
    pid_s = fname.split("_")[0]
    md = metadata.get(pid_s)
    if not md:
        continue
    n_files += 1
    with open(os.path.join(FILES, fname), newline="", encoding="utf-8", errors="replace") as f:
        rows_raw = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    for sess in group_sessions(rows_raw).values():
        occ.extend(extract(sess, char_maps[md["LAYOUT"]], md["LAYOUT"], int(pid_s)))
    if n_files % 10000 == 0:
        log(f"{n_files} files, {len(occ)} occurrences")
log(f"extraction done: {len(occ)} occurrences from {n_files} files")

by_key = defaultdict(list)
for layout, positions, ngram, swpm, e90, e98, dur, pid in occ:
    by_key[(layout, positions, ngram)].append((swpm, e90, e98, dur, pid))

# Shared eval cells (session-wpm buckets — identical for all arms) + per-cell ewma means.
cells, cell_e90, cell_e98 = [], [], []
for (layout, positions, ngram), samples in by_key.items():
    by_bucket = defaultdict(list)
    for swpm, e90, e98, dur, pid in samples:
        if not CELL_KW["wpm_lo"] <= swpm < CELL_KW["wpm_hi"]:
            continue
        b = CELL_KW["wpm_lo"] + ((int(swpm) - CELL_KW["wpm_lo"]) // CELL_KW["bucket_width"]) * CELL_KW["bucket_width"]
        by_bucket[b].append((swpm, e90, e98, dur, pid))
    for b, items in by_bucket.items():
        if len(items) < CELL_KW["min_cell_samples"]:
            continue
        cells.append(
            Cell(
                layout=layout, ngram=ngram, positions=positions, frequency=len(samples),
                bucket=b, wpm=b + CELL_KW["bucket_width"] / 2,
                obs=iqr_average([d for _, _, _, d, _ in items]), n=len(items),
                samples=[(int(s), d, p, 0) for s, _, _, d, p in items],
            )
        )
        cell_e90.append(float(np.mean([e for _, e, _, _, _ in items])))
        cell_e98.append(float(np.mean([e for _, _, e, _, _ in items])))
cell_e90 = np.array(cell_e90)
cell_e98 = np.array(cell_e98)
log(f"{len(cells)} eval cells")
obs_table = aggregate_layout_table(cells)
LAYOUTS = sorted({c.layout for c in cells})

from keybo.data.strokes import StrokeRow

CEILINGS = {}
for lay in LAYOUTS:
    rows_l = [
        StrokeRow(layout=k[0], positions=k[1], ngram=k[2], frequency=len(v),
                  samples=[(int(s), d, p, 0) for s, _, _, d, p in v])
        for k, v in by_key.items() if k[0] == lay
    ]
    CEILINGS[lay] = split_half_ceiling(rows_l, n_boot=30, seed=0, **CELL_KW)
    log(f"ceiling[{lay}] = {CEILINGS[lay]:.3f}")


def build_examples(arm):
    feats, targets, ngrams, layouts_, counts = [], [], [], [], []
    for (layout, positions, ngram), samples in by_key.items():
        groups = defaultdict(list)
        for swpm, e90, e98, dur, pid in samples:
            if arm == "S":
                key = int(swpm)
            elif arm == "ER90":
                key = int(e90 // 4 * 4)
            elif arm == "ER98":
                key = int(e98 // 4 * 4)
            else:  # ES90
                key = (int(swpm), int(e90 // 8 * 8))
            groups[key].append((swpm, e90, e98, dur))
        for _key, items in groups.items():
            base = bigram_features_from_positions(geom, positions, wpm=0.0)
            s_m = float(np.mean([s for s, _, _, _ in items]))
            e90_m = float(np.mean([e for _, e, _, _ in items]))
            e98_m = float(np.mean([e for _, _, e, _ in items]))
            v = base.copy()
            if arm == "S":
                v[-1] = s_m
            elif arm == "ER90":
                v[-1] = e90_m
            elif arm == "ER98":
                v[-1] = e98_m
            else:
                v[-1] = s_m
                v = np.append(v, e90_m)
            feats.append(v)
            targets.append(iqr_average([d for _, _, _, d in items]))
            ngrams.append(ngram)
            layouts_.append(layout)
            counts.append(float(len(items)))
    return (
        np.vstack(feats), np.array(targets), np.array(ngrams, dtype=object),
        np.array(layouts_, dtype=object), np.array(counts),
    )


Xc_base = np.vstack([bigram_features_from_positions(geom, c.positions, wpm=c.wpm) for c in cells])


def cell_features(arm):
    if arm == "S":
        return Xc_base
    Xc = Xc_base.copy()
    if arm == "ER90":
        Xc[:, -1] = cell_e90
        return Xc
    if arm == "ER98":
        Xc[:, -1] = cell_e98
        return Xc
    return np.column_stack([Xc_base, cell_e90])  # ES90


def fit_model(Xm, ym, seed, weight, n_extra):
    names = list(BIGRAM_FEATURE_NAMES) + (["ewma90"] if n_extra else [])
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION, feature_names=names,
        wpm_range=(60, 120), ngram="bigram",
    )
    m = XGBoostTypingModel(meta, random_state=seed, n_jobs=0)
    m._regressor.fit(Xm, ym, sample_weight=weight)
    m._fitted = True
    return m


results = {}
for arm in ("S", "ER90", "ER98", "ES90"):
    X, y, ngrams, layouts_, counts = build_examples(arm)
    Xc = cell_features(arm)
    n_extra = 1 if arm == "ES90" else 0
    log(f"ARM {arm}: {len(y)} examples")
    pooled = {s: {} for s in SEEDS}
    fracs = []
    for holdout in LAYOUTS:
        mask = layouts_ != holdout
        idx = [i for i, c in enumerate(cells) if c.layout == holdout]
        test_cells = [cells[i] for i in idx]
        obs = np.array([c.obs for c in test_cells])
        for seed in SEEDS:
            w = layout_balance_weights(layouts_[mask])
            model = fit_model(X[mask], y[mask], seed, w, n_extra)
            bmap = {}
            for _ in range(2):
                bmap = fit_practice_term(ngrams[mask], y[mask] - model.predict(X[mask]), counts[mask], k=K)
                bvec = np.array([bmap.get(g, 0.0) for g in ngrams[mask]])
                model = fit_model(X[mask], y[mask] - bvec, seed, w, n_extra)
            pred = model.predict(Xc[idx]) + np.array([bmap.get(c.ngram, 0.0) for c in test_cells])
            rho = _centered_spearman(test_cells, pred, obs)
            fracs.append(rho / CEILINGS[holdout])
            pooled[seed][holdout] = aggregate_layout_table(test_cells, pred)[holdout]
    taus = [layout_ranking_tau(obs_table, pooled[s]) for s in SEEDS]
    results[arm] = {"taus": taus, "mean_frac": float(np.mean(fracs))}
    log(f"  {arm}: tau {[f'{t:+.2f}' for t in taus]} mean rho/ceiling {results[arm]['mean_frac']:+.4f}")

json.dump(results, open("runs/ewma_arms.json", "w"), indent=2, default=float)
print("\n=== EWMA SCOREBOARD ===")
for arm, r in results.items():
    print(f"{arm:<6} tau {[f'{t:+.2f}' for t in r['taus']]} rho/ceiling {r['mean_frac']:+.4f}")
print("ALL-DONE")
