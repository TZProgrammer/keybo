"""How much less blind is a trigram model to finger reuse? — the lag-2 penalty, measured.

The bigram objective prices same-finger reuse at LAG 1 (consecutive keys). A trigram
objective additionally sees LAG 2 (keys 1 and 3 on the same finger — `sg_same_finger`).
This measures the lag-2 penalty from the real tristroke data, CONTROLLED for what the
bigram model already prices: compare trigram times where the two constituent bigrams have
identical motion classes, but the 1→3 skipgram is same-finger vs not.

Also measures per-finger service rates (mean interval by finger, matched to same-row
adjacent-column bigrams) — the data-derived utilization weights for the finger-load
objective term (gaps-and-roadmap 1.1).
"""

import ast
import time
from collections import defaultdict

import numpy as np

from keybo.data.keystrokes import build_char_map
from keybo.features.classify import classify_positions
from keybo.geometry import ROW_STAGGERED_30, Finger

t0 = time.time()
geom = ROW_STAGGERED_30


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# --- lag-2 penalty from tristrokes -------------------------------------------------------
# Bucket trigram rows by (bg1 class, bg2 class, lag2 same-finger?); compare durations.
acc = defaultdict(list)
n_rows = 0
with open("tristrokes_v1.tsv") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if parts[0] != "qwerty":  # qwerty only: within-layout control, dominant data
            continue
        try:
            positions = ast.literal_eval(parts[1])
        except (ValueError, SyntaxError):
            continue
        if len(positions) != 3:
            continue
        a, b, c = positions
        # exclude trigrams touching space (thumb) — different mechanism
        if 0 in (a[0], b[0], c[0]):
            continue
        cls1 = classify_positions(geom, a, b).value
        cls2 = classify_positions(geom, b, c).value
        lag2_sf = geom.same_finger(a[0], c[0]) and a != c
        durs = []
        for tok in parts[4:]:
            try:
                w, d, p, h = ast.literal_eval(tok)
            except (ValueError, SyntaxError):
                continue
            if 40 <= w < 140 and 0 < d < 2000:
                durs.append(d)
        if len(durs) >= 5:
            acc[(cls1, cls2, lag2_sf)].append(float(np.median(durs)))
        n_rows += 1
log(f"{n_rows} qwerty trigram rows bucketed")

print("\n=== LAG-2 SAME-FINGER PENALTY (qwerty tristrokes, wpm 40-140) ===")
print(f"{'bg1':<6} {'bg2':<6} {'lag2=no':>14} {'lag2=YES':>14} {'penalty':>10}")
total_w = 0.0
weighted_pen = 0.0
for cls1 in ("alt", "shb"):
    for cls2 in ("alt", "shb"):
        no = acc.get((cls1, cls2, False), [])
        yes = acc.get((cls1, cls2, True), [])
        if len(no) >= 10 and len(yes) >= 10:
            m_no, m_yes = float(np.median(no)), float(np.median(yes))
            pen = m_yes - m_no
            w = len(yes)
            total_w += w
            weighted_pen += pen * w
            print(
                f"{cls1:<6} {cls2:<6} {m_no:10.0f} ({len(no):4d}) {m_yes:9.0f} ({len(yes):4d}) {pen:+9.0f}ms"
            )
if total_w:
    print(f"\nweighted mean lag-2 penalty: {weighted_pen / total_w:+.0f} ms "
          f"(reference: lag-1 SFB penalty ~= 196-165 = +31 ms per the model's class means)")

# --- per-finger service rates ------------------------------------------------------------
# Matched-geometry comparison: same-row adjacent-column bigrams from the BIGRAM table,
# grouped by the finger of the SECOND (landing) key.
log("per-finger service rates (matched same-row adjacent bigrams, bistrokes)")
finger_acc = defaultdict(list)
with open("bistrokes_v3.tsv") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if parts[0] != "qwerty":
            continue
        try:
            (x1, y1), (x2, y2) = ast.literal_eval(parts[1])
        except (ValueError, SyntaxError):
            continue
        if y1 != y2 or y1 not in (2, 3) or abs(abs(x1) - abs(x2)) != 1 or x1 * x2 <= 0:
            continue  # same row, adjacent columns, same hand
        if geom.same_finger(x1, x2):
            continue
        durs = []
        for tok in parts[4:]:
            try:
                w, d, p, h = ast.literal_eval(tok)
            except (ValueError, SyntaxError):
                continue
            if 40 <= w < 140 and 0 < d < 1000:
                durs.append(d)
        if len(durs) >= 10:
            finger_acc[geom.finger(x2)].append(float(np.median(durs)))

print("\n=== PER-FINGER SERVICE RATE (landing-key finger, matched geometry) ===")
for f in (Finger.LP, Finger.LR, Finger.LM, Finger.LI, Finger.RI, Finger.RM, Finger.RR, Finger.RP):
    v = finger_acc.get(f, [])
    if v:
        print(f"  {f.value:<14} {np.median(v):6.0f} ms  (n={len(v)} bigram cells)")
print("ALL-DONE")
