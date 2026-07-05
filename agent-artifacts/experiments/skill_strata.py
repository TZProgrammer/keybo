"""Skill stratification of every objective-driving effect (user challenge #3).

Are the pooled effect estimates — roll sub-additivity (−22 ms), redirect null, lag-2 null
(−13 ms), SFB penalty, alternation gap — skill-mixtures that apply to nobody? For wpm
bands 40–70 / 70–100 / 100–130 (qwerty rows, matched construction to the original probes):

  1. roll sub-additivity contrast (run-continue delta − alt-alt delta) per band
  2. redirect contrast (run-redirect delta − alt-alt delta) per band
  3. lag-2 same-finger penalty per band
  4. SFB (lag-1) penalty per band — same-finger vs matched same-row adjacent bigrams
  5. alternation-vs-same-hand gap per band

Plus the MODEL-side check: the shipped trigram/bigram models take wpm as a feature — do
their table contrasts AT wpm=55/85/115 track the raw per-band physics (sign+magnitude)?
If yes, per-wpm tables (already one batch predict away) capture the skill dependence and
the machinery needs no redesign — only per-wpm optimization runs.

Pre-registered rule (memory.md): an effect is SKILL-DEPENDENT if |band3 − band1| >
max(10 ms, 50% of the pooled effect) AND the trend is monotone.
"""

import ast
import time
from collections import defaultdict

import numpy as np

from keybo.features.classify import classify_positions
from keybo.geometry import ROW_STAGGERED_30

t0 = time.time()
geom = ROW_STAGGERED_30
BANDS = [(40, 70), (70, 100), (100, 130)]


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def band_of(w):
    for lo, hi in BANDS:
        if lo <= w < hi:
            return f"{lo}-{hi}"
    return None


# ==== bigram-level effects (bistrokes pass) ==============================================
log("bistrokes pass: SFB penalty + alternation gap + bigram references per band")
bg_ref = {}  # (positions, band) -> median duration  [for the trigram additivity probe]
cls_acc = defaultdict(list)  # (band, class) -> [medians]  [alternation gap]
sfb_acc = defaultdict(list)  # (band, kind) -> [medians]   [sfb vs matched adjacent]
with open("bistrokes_v3.tsv") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if parts[0] != "qwerty":
            continue
        try:
            positions = ast.literal_eval(parts[1])
        except (ValueError, SyntaxError):
            continue
        (x1, y1), (x2, y2) = positions
        by_band = defaultdict(list)
        for tok in parts[4:]:
            try:
                w, d, p, h = ast.literal_eval(tok)
            except (ValueError, SyntaxError):
                continue
            b = band_of(w)
            if b and 0 < d < 1500:
                by_band[b].append(d)
        for b, ds in by_band.items():
            if len(ds) < 8:
                continue
            med = float(np.median(ds))
            bg_ref[(positions, b)] = med
            if 0 in (x1, x2):
                continue
            cls = classify_positions(geom, (x1, y1), (x2, y2)).value
            cls_acc[(b, cls)].append(med)
            # SFB vs matched control: same-row, adjacent |col|, same hand, diff finger
            if cls == "sfb" and (x1, y1) != (x2, y2):
                sfb_acc[(b, "sfb")].append(med)
            elif (
                y1 == y2
                and x1 * x2 > 0
                and abs(abs(x1) - abs(x2)) == 1
                and not geom.same_finger(x1, x2)
            ):
                sfb_acc[(b, "adj")].append(med)

print("\n=== 5. ALTERNATION vs SAME-HAND gap by band (median ms) ===")
for lo, hi in BANDS:
    b = f"{lo}-{hi}"
    alt = cls_acc.get((b, "alt"), [])
    shb = cls_acc.get((b, "shb"), [])
    if len(alt) >= 20 and len(shb) >= 20:
        print(
            f"  {b}: alt {np.median(alt):6.0f} ({len(alt):4d})  shb {np.median(shb):6.0f} "
            f"({len(shb):4d})  gap {np.median(shb) - np.median(alt):+5.0f}ms"
        )

print("\n=== 4. SFB penalty by band (sfb vs matched same-row adjacent) ===")
for lo, hi in BANDS:
    b = f"{lo}-{hi}"
    sfb = sfb_acc.get((b, "sfb"), [])
    adj = sfb_acc.get((b, "adj"), [])
    if len(sfb) >= 10 and len(adj) >= 10:
        print(
            f"  {b}: sfb {np.median(sfb):6.0f} ({len(sfb):3d})  adj {np.median(adj):6.0f} "
            f"({len(adj):3d})  penalty {np.median(sfb) - np.median(adj):+5.0f}ms"
        )

# ==== trigram-level effects (tristrokes pass) ============================================
log("tristrokes pass: roll/redirect additivity + lag-2 per band")
ctx_acc = defaultdict(list)  # (band, context) -> [obs - expected]
lag2_acc = defaultdict(list)  # (band, lag2sf?) -> [median]
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
        a, b3, c = positions
        if 0 in (a[0], b3[0], c[0]):
            continue
        same_hand_run = geom.hand(a[0]) == geom.hand(b3[0]) == geom.hand(c[0]) != 0
        cls1 = classify_positions(geom, a, b3).value
        cls2 = classify_positions(geom, b3, c).value
        if same_hand_run:
            g1 = abs(b3[0]) - abs(a[0])
            g2 = abs(c[0]) - abs(b3[0])
            if g1 == 0 or g2 == 0:
                ctx = "run-flat"
            elif (g1 > 0) == (g2 > 0):
                ctx = "run-continue"
            else:
                ctx = "run-redirect"
        elif cls1 == "alt" and cls2 == "alt":
            ctx = "alt-alt"
        else:
            ctx = "mixed"
        lag2_sf = geom.same_finger(a[0], c[0]) and a != c
        by_band = defaultdict(list)
        for tok in parts[4:]:
            try:
                w, d, p, h = ast.literal_eval(tok)
            except (ValueError, SyntaxError):
                continue
            bnd = band_of(w)
            if bnd and 0 < d < 2500:
                by_band[bnd].append(d)
        for bnd, ds in by_band.items():
            if len(ds) < 6:
                continue
            med = float(np.median(ds))
            r1 = bg_ref.get(((a, b3), bnd))
            r2 = bg_ref.get(((b3, c), bnd))
            if r1 is not None and r2 is not None:
                ctx_acc[(bnd, ctx)].append(med - (r1 + r2))
            if cls1 in ("alt", "shb") and cls2 in ("alt", "shb"):
                lag2_acc[(bnd, lag2_sf)].append(med)

print("\n=== 1+2. ROLL / REDIRECT contrasts by band (delta vs alt-alt, ms) ===")
print(f"{'band':<9} {'alt-alt':>9} {'continue':>10} {'redirect':>10} {'roll-contrast':>14} {'redir-contrast':>15}")
for lo, hi in BANDS:
    bnd = f"{lo}-{hi}"
    aa = ctx_acc.get((bnd, "alt-alt"), [])
    rc = ctx_acc.get((bnd, "run-continue"), [])
    rr = ctx_acc.get((bnd, "run-redirect"), [])
    if len(aa) >= 20 and len(rc) >= 10 and len(rr) >= 10:
        m_aa, m_rc, m_rr = np.median(aa), np.median(rc), np.median(rr)
        print(
            f"{bnd:<9} {m_aa:>+7.0f}ms {m_rc:>+8.0f}ms {m_rr:>+8.0f}ms "
            f"{m_rc - m_aa:>+12.0f}ms {m_rr - m_aa:>+13.0f}ms   "
            f"(n={len(aa)}/{len(rc)}/{len(rr)})"
        )

print("\n=== 3. LAG-2 same-finger penalty by band ===")
for lo, hi in BANDS:
    bnd = f"{lo}-{hi}"
    no = lag2_acc.get((bnd, False), [])
    yes = lag2_acc.get((bnd, True), [])
    if len(no) >= 20 and len(yes) >= 10:
        print(
            f"  {bnd}: no {np.median(no):6.0f} ({len(no):4d})  yes {np.median(yes):6.0f} "
            f"({len(yes):3d})  penalty {np.median(yes) - np.median(no):+5.0f}ms"
        )

# ==== MODEL-side: do the learned wpm-interactions track the physics? =====================
log("model-side: table contrasts at wpm 55/85/115")
import sys

sys.path.insert(0, "/tmp/keybo_prod5/src")
from keybo.layout import Layout  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402
from keybo.features import bigram_features_from_positions  # noqa: E402

bi_models = [XGBoostTypingModel.load(f"models/bigram_d3_seed{s}.json") for s in (0, 1, 2)]
pairs = {
    "alt (f->j)": ((-2, 2), (2, 2)),
    "shb (a->d)": ((-5, 2), (-3, 2)),
    "sfb (ju)": ((2, 2), (2, 3)),
}
print("\n=== MODEL bigram-class predictions by wpm (mean of 3 seeds, ms) ===")
print(f"{'pair':<12} {'wpm55':>7} {'wpm85':>7} {'wpm115':>7}")
for name, (pa, pb) in pairs.items():
    row = []
    for wpm in (55.0, 85.0, 115.0):
        v = bigram_features_from_positions(geom, (pa, pb), wpm=wpm)
        row.append(float(np.mean([m.predict(v[None, :])[0] for m in bi_models])))
    print(f"{name:<12} {row[0]:>7.0f} {row[1]:>7.0f} {row[2]:>7.0f}")
alt55 = None
print("ALL-DONE")
