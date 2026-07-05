"""Phase B physics probes, one dump pass: (1) roll sub-additivity, (2) error geometry.

PROBE R (roll effects, the trigram objective's physical basis): does the SECOND bigram of
a trigram get faster/slower depending on the FIRST bigram's motion class? Bucket qwerty
trigram rows by (class of bg1, class of bg2, direction-continuity) and compare the
bg2-interval (last-press minus middle-press... not stored per-row — so instead compare
FULL trigram time minus the standalone-bigram expectation). Implementation: for each
trigram row, time = full span. Standalone expectation = median bigram time of bg1 +
median bigram time of bg2, from the bistroke table at the matched wpm bucket. The
DIFFERENCE, bucketed by context class, measures super/sub-additivity: rolls (same-hand
continuation) sub-additive? redirects super-additive?

PROBE E (error geometry): per-position and per-bigram-class mistype rates on qwerty from
the RAW dump: for each expected character (difflib-aligned), was it typed correctly?
Rate by intended key position (row/finger) and by preceding-transition class. If error
rate is geometry-structured, an error-cost term belongs in fitness (weight: corrections
cost ~5.4x an interval, OQ-12).
"""

import ast
import csv
import os
import time
from collections import defaultdict

import numpy as np

from keybo.data.keystrokes import (
    build_char_map,
    group_sessions,
    load_participant_metadata,
    mark_correct_flags,
    _letter,
)
from keybo.features.classify import classify_positions
from keybo.geometry import ROW_STAGGERED_30

t0 = time.time()
geom = ROW_STAGGERED_30
FILES = "dataset/Keystrokes/files"
META = "dataset/Keystrokes/files/metadata_participants.txt"


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# ==== PROBE R: roll super/sub-additivity ================================================
# Reference bigram medians by (positions, wpm decade) from the bistroke table.
log("building bigram reference from bistrokes_v3.tsv (qwerty rows)")
bg_ref = {}
with open("bistrokes_v3.tsv") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if parts[0] != "qwerty":
            continue
        try:
            positions = ast.literal_eval(parts[1])
        except (ValueError, SyntaxError):
            continue
        by_dec = defaultdict(list)
        for tok in parts[4:]:
            try:
                w, d, p, h = ast.literal_eval(tok)
            except (ValueError, SyntaxError):
                continue
            if 40 <= w < 140 and 0 < d < 1500:
                by_dec[w // 20 * 20].append(d)
        for dec, ds in by_dec.items():
            if len(ds) >= 10:
                bg_ref[(positions, dec)] = float(np.median(ds))
log(f"{len(bg_ref)} (bigram-position, wpm-bucket) reference medians")

log("scanning tristrokes_v1.tsv for additivity")
CTX = defaultdict(list)  # (bg1 class, bg2 class, continuity) -> [observed - expected]
n_used = 0
with open("tristrokes_v1.tsv") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if parts[0] != "qwerty":
            continue
        try:
            positions = ast.literal_eval(parts[1])
        except (ValueError, SyntaxError):
            continue
        if len(positions) != 3:
            continue
        a, b, c = positions
        if 0 in (a[0], b[0], c[0]):
            continue
        cls1 = classify_positions(geom, a, b).value
        cls2 = classify_positions(geom, b, c).value
        # direction continuity for same-hand runs: |col| strictly monotone = a "roll-through"
        same_hand_run = (
            geom.hand(a[0]) == geom.hand(b[0]) == geom.hand(c[0]) and geom.hand(a[0]) != 0
        )
        if same_hand_run:
            going1 = abs(b[0]) - abs(a[0])
            going2 = abs(c[0]) - abs(b[0])
            if going1 == 0 or going2 == 0:
                cont = "run-flat"
            elif (going1 > 0) == (going2 > 0):
                cont = "run-continue"  # the community's "roll through"
            else:
                cont = "run-redirect"
        else:
            cont = "mixed" if (cls1 != "alt" or cls2 != "alt") else "alt-alt"
        by_dec = defaultdict(list)
        for tok in parts[4:]:
            try:
                w, d, p, h = ast.literal_eval(tok)
            except (ValueError, SyntaxError):
                continue
            if 40 <= w < 140 and 0 < d < 2500:
                by_dec[w // 20 * 20].append(d)
        for dec, ds in by_dec.items():
            if len(ds) < 8:
                continue
            r1 = bg_ref.get(((a, b), dec))
            r2 = bg_ref.get(((b, c), dec))
            if r1 is None or r2 is None:
                continue
            CTX[cont].append(float(np.median(ds)) - (r1 + r2))
            n_used += 1
log(f"{n_used} (trigram, bucket) observations matched to bigram references")

print("\n=== PROBE R: trigram time minus sum-of-constituent-bigram medians ===")
print(f"{'context':<14} {'n':>7} {'median delta':>14} {'mean delta':>12}")
for ctx in ("alt-alt", "mixed", "run-continue", "run-flat", "run-redirect"):
    v = CTX.get(ctx, [])
    if len(v) >= 20:
        print(f"{ctx:<14} {len(v):>7} {np.median(v):>+12.0f}ms {np.mean(v):>+10.0f}ms")

# ==== PROBE E: error geometry ============================================================
log("PROBE E: per-position error rates (raw dump scan, qwerty participants)")
metadata = load_participant_metadata(META)
cmap = build_char_map("qwerty")
attempts = defaultdict(int)
errors = defaultdict(int)
by_class_att = defaultdict(int)
by_class_err = defaultdict(int)
n_files = 0
for fname in os.listdir(FILES):
    if not fname.endswith("_keystrokes.txt"):
        continue
    pid = fname.split("_")[0]
    md = metadata.get(pid)
    if not md or md["LAYOUT"] != "qwerty":
        continue
    n_files += 1
    if n_files > 8000:  # 8k files ~ plenty of power for rates
        break
    try:
        with open(os.path.join(FILES, fname), newline="", encoding="utf-8", errors="replace") as f:
            rows_raw = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    except OSError:
        continue
    for sess in group_sessions(rows_raw).values():
        if not sess:
            continue
        expected = sess[0].get("SENTENCE") or ""
        single = [(i, r) for i, r in enumerate(sess) if len(_letter(r)) == 1]
        if not single:
            continue
        typed = "".join(_letter(r) for _, r in single)
        flags = mark_correct_flags(typed, expected)
        prev_pos = None
        for (idx, r), ok in zip(single, flags, strict=False):
            ch = _letter(r).lower()
            pos = cmap.get(ch)
            if pos is None:
                prev_pos = None
                continue
            attempts[pos] += 1
            if not ok:
                errors[pos] += 1
            if prev_pos is not None and prev_pos != pos:
                cls = classify_positions(geom, prev_pos, pos).value
                by_class_att[cls] += 1
                if not ok:
                    by_class_err[cls] += 1
            prev_pos = pos
log(f"{n_files} qwerty files scanned")

print("\n=== PROBE E: error rate by intended-key ROW (qwerty) ===")
row_att = defaultdict(int)
row_err = defaultdict(int)
for pos, n in attempts.items():
    row_att[pos[1]] += n
    row_err[pos[1]] += errors[pos]
for y, label in ((3, "top"), (2, "home"), (1, "bottom"), (0, "space")):
    if row_att[y]:
        print(f"  {label:<8} {row_err[y] / row_att[y]:.2%}  (n={row_att[y]})")
print("=== error rate by FINGER (landing key) ===")
fin_att = defaultdict(int)
fin_err = defaultdict(int)
for pos, n in attempts.items():
    f = geom.finger(pos[0]).value
    fin_att[f] += n
    fin_err[f] += errors[pos]
for f in sorted(fin_att):
    print(f"  {f:<14} {fin_err[f] / fin_att[f]:.2%}  (n={fin_att[f]})")
print("=== error rate by PRECEDING transition class ===")
for cls in ("alt", "shb", "sfb"):
    if by_class_att[cls]:
        print(f"  {cls:<6} {by_class_err[cls] / by_class_att[cls]:.2%}  (n={by_class_att[cls]})")
print("ALL-DONE")
