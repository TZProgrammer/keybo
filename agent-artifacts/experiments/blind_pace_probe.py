"""Blind pace model probe (user proposal #7): can ANY content-blind estimator beat the
leave-one-out session mean at predicting the current interval?

Residual formulation makes the test exact: predict r_i = x_i - LOO_mean(i) from centered
neighbor residuals (missing -> 0). Estimators:
  (a) LOO session mean alone (predict r=0) — the incumbent
  (b) centered window means of neighbor residuals, h in {2, 5, 10}
  (c) the user's exact model: ridge-linear on residuals at offsets -10..-1,+1..+10,
      fit on 70% of sessions, evaluated on held-out 30%
Falsifiable analytical prediction (GLS with measured ~zero autocorrelation): the ridge
weights come out ~0 and no estimator improves on (a).
Rule (preregistered): best improvement < 2% relative MAE => stage-1 unattainable, close.
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
H_MAX = 10


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


metadata = load_participant_metadata("dataset/Keystrokes/files/metadata_participants.txt")
cmap = build_char_map("qwerty")
allowed = set(cmap)

sessions = []  # list of np.array of clean intervals (contiguous runs preserved via NaN)
n_files = 0
for fname in os.listdir(FILES):
    if not fname.endswith("_keystrokes.txt"):
        continue
    md = metadata.get(fname.split("_")[0])
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
        ivals = []
        for a, b in zip(correct, correct[1:]):
            la, lb = _letter(a[1]), _letter(b[1])
            try:
                t1, t2 = float(a[1]["PRESS_TIME"]), float(b[1]["PRESS_TIME"])
            except (TypeError, ValueError, KeyError):
                ivals.append(np.nan)
                continue
            if (
                b[0] - a[0] == 1
                and 0 < t2 - t1 < CAP_MS
                and la.upper() not in BANNED_KEYS
                and lb.upper() not in BANNED_KEYS
                and la.lower() in allowed
                and lb.lower() in allowed
            ):
                ivals.append(t2 - t1)
            else:
                ivals.append(np.nan)
        arr = np.array(ivals, dtype=float)
        if np.isfinite(arr).sum() >= 12:
            sessions.append(arr)
log(f"{len(sessions)} usable sessions from {n_files} files")
lengths = [int(np.isfinite(s).sum()) for s in sessions]
print(f"clean-interval count per session: p10 {np.percentile(lengths, 10):.0f} "
      f"median {np.median(lengths):.0f} p90 {np.percentile(lengths, 90):.0f} "
      f"(NOTE: sessions are single sentences — a 10+10 window ~ the whole session)")

# Build the residual design: for each finite interval, LOO mean + neighbor residuals.
rng = np.random.default_rng(7)
order = rng.permutation(len(sessions))
split = int(0.7 * len(sessions))
train_idx, test_idx = set(order[:split].tolist()), set(order[split:].tolist())

offsets = [o for o in range(-H_MAX, H_MAX + 1) if o != 0]


def rows_for(session):
    finite = np.isfinite(session)
    vals = session[finite]
    n = len(vals)
    total = vals.sum()
    # map from original index to compact clean index (windows in CLEAN sequence space,
    # matching the user's "10 bigrams before/after")
    out = []
    for ci in range(n):
        loo = (total - vals[ci]) / (n - 1)
        r_target = vals[ci] - loo
        feats = np.zeros(len(offsets))
        for k, o in enumerate(offsets):
            j = ci + o
            if 0 <= j < n:
                feats[k] = vals[j] - loo
        out.append((r_target, feats))
    return out


Xtr, ytr, Xte, yte = [], [], [], []
for si, sess in enumerate(sessions):
    rows = rows_for(sess)
    if si in train_idx:
        for r, f in rows:
            ytr.append(r)
            Xtr.append(f)
    else:
        for r, f in rows:
            yte.append(r)
            Xte.append(f)
Xtr = np.array(Xtr); ytr = np.array(ytr); Xte = np.array(Xte); yte = np.array(yte)
log(f"{len(ytr)} train / {len(yte)} test intervals")

# (a) incumbent: predict residual 0
mae_loo = float(np.mean(np.abs(yte)))

# (b) centered window means over available neighbor residuals
def window_mae(h):
    cols = [k for k, o in enumerate(offsets) if abs(o) <= h]
    sub = Xte[:, cols]
    counts = (sub != 0).sum(axis=1)  # zeros = missing (approx: exact-zero residual rare)
    means = np.where(counts > 0, sub.sum(axis=1) / np.maximum(counts, 1), 0.0)
    return float(np.mean(np.abs(yte - means)))


# (c) ridge on all 20 offsets
lam = 10.0
A = Xtr.T @ Xtr + lam * np.eye(len(offsets))
w = np.linalg.solve(A, Xtr.T @ ytr)
mae_ridge = float(np.mean(np.abs(yte - Xte @ w)))

print(f"\n=== BLIND PACE PROBE (test MAE on interval residuals, ms) ===")
print(f"(a) LOO session mean          {mae_loo:8.2f}   (improvement — )")
for h in (2, 5, 10):
    m = window_mae(h)
    print(f"(b) centered window h={h:<2}       {m:8.2f}   ({(mae_loo - m) / mae_loo * 100:+.2f}%)")
print(f"(c) ridge 10+10 (user model)  {mae_ridge:8.2f}   ({(mae_loo - mae_ridge) / mae_loo * 100:+.2f}%)")
print(f"\nridge weights (|w| max {np.max(np.abs(w)):.4f}, mean {np.mean(np.abs(w)):.4f}) — "
      f"prediction was ~0:")
print("  " + " ".join(f"{o:+d}:{wt:+.3f}" for o, wt in zip(offsets, w) if abs(o) <= 3))
print("ALL-DONE")
