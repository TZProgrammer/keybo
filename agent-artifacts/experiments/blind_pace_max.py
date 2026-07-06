"""Blind-pace maximization (user directive #9): the best content-blind interval predictor.

Leakage contract: features are timing scalars + session/participant indices ONLY. Model
class capped at linear over robust aggregates (expressive models could fingerprint
content from neighbor-value patterns). Every winner passes a leakage audit: its held-out
residuals must not encode more ngram-identity information than the LOO-mean's residuals.

Candidates (nested, so each lever's marginal value is visible):
  M0  LOO session mean (incumbent)
  M1  LOO session median (robust location)
  M2  participant prior alone: shrunk mean of the typist's OTHER sessions
  M3  M1 + M2 blended (ridge over [loo_med_resid... no — features below])
  M4  full linear blend: [LOO-median, participant-prior, windowed neighbor medians
      (h=3, h=10, residual vs LOO-median), sentence-position frac, log session length]
  M5  M4 in log space (fit on log intervals, predictions exponentiated)
  L5  per-offset ridge 10+10 (the +0.60% reference, recomputed on this split)

Eval: 70/30 session split (split at PARTICIPANT level so M2 never sees test typists'
sessions in training its blend weights — though the prior itself uses only each test
typist's own OTHER sessions, which is legitimate persistence, not leakage).
"""

import csv
import os
import time
from collections import defaultdict

import numpy as np

from keybo.data.keystrokes import (
    BANNED_KEYS,
    _letter,
    build_char_map,
    group_sessions,
    load_participant_metadata,
    mark_correct_flags,
)

t0 = time.time()
FILES = "dataset/Keystrokes/files"
CAP_MS = 2000.0
MAX_FILES = 12000
SHRINK_SESS = 3.0  # participant prior shrinkage (in sessions)


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


metadata = load_participant_metadata("dataset/Keystrokes/files/metadata_participants.txt")
cmap = build_char_map("qwerty")
allowed = set(cmap)

# sessions: (pid, [intervals w/ NaN gaps], [ngram labels for the leakage audit])
sessions = []
n_files = 0
for fname in sorted(os.listdir(FILES)):
    if not fname.endswith("_keystrokes.txt"):
        continue
    pid_s = fname.split("_")[0]
    md = metadata.get(pid_s)
    if not md or md["LAYOUT"] != "qwerty":
        continue
    n_files += 1
    if n_files > MAX_FILES:
        break
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
        if len(correct) < 13:
            continue
        ivals, labels = [], []
        for a, b in zip(correct, correct[1:]):
            la, lb = _letter(a[1]), _letter(b[1])
            try:
                t1, t2 = float(a[1]["PRESS_TIME"]), float(b[1]["PRESS_TIME"])
            except (TypeError, ValueError, KeyError):
                ivals.append(np.nan); labels.append(None)
                continue
            if (
                b[0] - a[0] == 1 and 0 < t2 - t1 < CAP_MS
                and la.upper() not in BANNED_KEYS and lb.upper() not in BANNED_KEYS
                and la.lower() in allowed and lb.lower() in allowed
            ):
                ivals.append(t2 - t1); labels.append(la.lower() + lb.lower())
            else:
                ivals.append(np.nan); labels.append(None)
        arr = np.array(ivals, dtype=float)
        if np.isfinite(arr).sum() >= 12:
            sessions.append((int(pid_s), arr, labels))
log(f"{len(sessions)} sessions from {n_files} files")

# participant-level split
pids = sorted({p for p, _, _ in sessions})
rng = np.random.default_rng(11)
rng.shuffle(pids)
cut = int(0.7 * len(pids))
train_pids = set(pids[:cut])

# participant prior: per pid, per session -> mean of OTHER sessions' medians (shrunk to
# global median). Computed for ALL pids (test typists' priors use their own other
# sessions — persistence, not leakage).
sess_medians = defaultdict(list)
for pid, arr, _ in sessions:
    sess_medians[pid].append(float(np.nanmedian(arr)))
global_med = float(np.median([m for v in sess_medians.values() for m in v]))


def participant_prior(pid, own_median):
    others = [m for m in sess_medians[pid] if m != own_median]
    if not others:
        return global_med
    n = len(others)
    return (np.sum(others) + SHRINK_SESS * global_med) / (n + SHRINK_SESS)


# --- assemble example matrices ------------------------------------------------------------
def windowed_median_resid(vals, ci, h, center):
    lo, hi = max(0, ci - h), min(len(vals), ci + h + 1)
    neigh = np.concatenate([vals[lo:ci], vals[ci + 1 : hi]])
    if len(neigh) < 2:
        return 0.0
    return float(np.median(neigh) - center)


DATA = {"train": defaultdict(list), "test": defaultdict(list)}
for pid, arr, labels in sessions:
    finite = np.isfinite(arr)
    vals = arr[finite]
    labs = [l for l, f in zip(labels, finite) if f]
    n = len(vals)
    total, ssum = vals.sum(), float(np.nansum(arr))
    s_med = float(np.median(vals))
    prior = participant_prior(pid, s_med)
    part = "train" if pid in train_pids else "test"
    for ci in range(n):
        loo_mean = (total - vals[ci]) / (n - 1)
        # LOO median (cheap approximation: median of all-but-one)
        loo_med = float(np.median(np.delete(vals, ci))) if n <= 400 else s_med
        d = DATA[part]
        d["y"].append(vals[ci])
        d["loo_mean"].append(loo_mean)
        d["loo_med"].append(loo_med)
        d["prior"].append(prior)
        d["w3"].append(windowed_median_resid(vals, ci, 3, loo_med))
        d["w10"].append(windowed_median_resid(vals, ci, 10, loo_med))
        d["posfrac"].append(ci / max(n - 1, 1))
        d["loglen"].append(np.log(n))
        d["label"].append(labs[ci])
        # raw offsets for L5
        offs = np.zeros(20)
        for k, o in enumerate([*range(-10, 0), *range(1, 11)]):
            j = ci + o
            if 0 <= j < n:
                offs[k] = vals[j] - loo_mean
        d["offsets"].append(offs)

for part in DATA:
    for k in DATA[part]:
        DATA[part][k] = np.array(DATA[part][k]) if k != "label" else DATA[part][k]
ytr, yte = DATA["train"]["y"], DATA["test"]["y"]
log(f"{len(ytr)} train / {len(yte)} test intervals ({len(train_pids)}/{len(pids) - len(train_pids)} pids)")


def mae(pred):
    return float(np.mean(np.abs(yte - pred)))


def fit_ridge(feats_tr, target_tr, feats_te, lam=10.0):
    A = feats_tr.T @ feats_tr + lam * np.eye(feats_tr.shape[1])
    w = np.linalg.solve(A, feats_tr.T @ target_tr)
    return feats_te @ w, w


results = {}
results["M0 loo-mean"] = mae(DATA["test"]["loo_mean"])
results["M1 loo-median"] = mae(DATA["test"]["loo_med"])
results["M2 participant-prior"] = mae(DATA["test"]["prior"])

# M3/M4/M5: ridge over feature blends, predicting the interval directly.
def blend(names, log_space=False):
    ftr = np.column_stack([DATA["train"][n] for n in names] + [np.ones(len(ytr))])
    fte = np.column_stack([DATA["test"][n] for n in names] + [np.ones(len(yte))])
    ttr = np.log(ytr) if log_space else ytr
    pred, w = fit_ridge(ftr, ttr, fte)
    if log_space:
        pred = np.exp(pred)
    return pred, w


pred_m3, _ = blend(["loo_med", "prior"])
results["M3 med+prior blend"] = mae(pred_m3)
M4_FEATS = ["loo_med", "prior", "w3", "w10", "posfrac", "loglen"]
pred_m4, w4 = blend(M4_FEATS)
results["M4 full blend"] = mae(pred_m4)
pred_m5, _ = blend(M4_FEATS, log_space=True)
results["M5 full blend (log)"] = mae(pred_m5)
# L5 reference: offsets ridge predicting residual vs loo_mean
otr = np.array(DATA["train"]["offsets"]); ote = np.array(DATA["test"]["offsets"])
rtr = ytr - DATA["train"]["loo_mean"]
pred_r, _ = fit_ridge(otr, rtr, ote)
results["L5 offsets ridge"] = mae(DATA["test"]["loo_mean"] + pred_r)

base = results["M0 loo-mean"]
print("\n=== BLIND-PACE FRONTIER (test MAE, ms; improvement vs LOO mean) ===")
for name, m in results.items():
    print(f"{name:<24} {m:8.2f}   ({(base - m) / base * 100:+.2f}%)")
print("\nM4 weights:", dict(zip([*M4_FEATS, 'const'], np.round(w4, 4))))

# --- leakage audit on the best candidate ---------------------------------------------------
best_name = min(results, key=results.get)
pred_best = {  # regenerate the winning prediction
    "M0 loo-mean": DATA["test"]["loo_mean"],
    "M1 loo-median": DATA["test"]["loo_med"],
    "M2 participant-prior": DATA["test"]["prior"],
    "M3 med+prior blend": pred_m3,
    "M4 full blend": pred_m4,
    "M5 full blend (log)": pred_m5,
    "L5 offsets ridge": DATA["test"]["loo_mean"] + pred_r,
}[best_name]


def ngram_r2(pred):
    """R² of held-out residuals on ngram one-hots (top-200 ngrams) — leakage detector."""
    resid = yte - pred
    labs = DATA["test"]["label"]
    counts = defaultdict(int)
    for l in labs:
        counts[l] += 1
    top = {l for l, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:200]}
    groups = defaultdict(list)
    for r, l in zip(resid, labs):
        if l in top:
            groups[l].append(r)
    grand = np.mean([r for v in groups.values() for r in v])
    sst = sum((r - grand) ** 2 for v in groups.values() for r in v)
    sse = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in groups.values())
    return 1 - sse / sst if sst else 0.0


r2_base = ngram_r2(DATA["test"]["loo_mean"])
r2_best = ngram_r2(pred_best)
print(f"\nleakage audit — residual ngram-R²: loo-mean {r2_base:.4f} vs {best_name} {r2_best:.4f}")
print("PASS (no added content info)" if r2_best <= r2_base + 0.002 else "FAIL — DISQUALIFIED (timing fingerprint leakage)")
print("ALL-DONE")
