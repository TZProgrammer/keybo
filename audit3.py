"""REFLECT audit item 3: can the bl contradiction be settled cheaply?

The contradiction: for the SAME layout change,
  raw-cell path : bl moves from class (lower=middle,upper=pinky,dy2)=+0.2643 to (…,dy1)=+0.1494 => CHEAPER
  fitted _T2    : bl moves from b(3,1)->l(5,3) mean 142.22ms to b(3,1)->l(5,2) mean 146.50ms => COSTLIER

Three candidate discriminators, all cheap:

(D1) UNIT MISMATCH. The raw-cell numbers are CLASS means over many position pairs; the _T2 numbers
     are for TWO SPECIFIC position pairs. A class mean need not agree with a member. Test: compute
     the _T2 CLASS mean over exactly the position pairs each raw cell covers, and see whether _T2
     agrees with the raw path AT CLASS LEVEL. If yes, the contradiction is a level-of-aggregation
     artifact, not a disagreement between data sources.

(D2) BASELINE MISMATCH. raw `rel` is relative to a same-row two-finger baseline; _T2 is absolute ms
     including the landing-key/frequency-free geometry terms. Test: convert _T2 to the SAME estimand
     (excess over the same-row same-hand two-finger _T2 mean) and re-compare.

(D3) DIRECT OBSERVATION. Does the Aalto sample contain the specific position pair b(3,1)->l(5,2) or
     (3,1)->(5,3) at all? If a source layout puts SOME letter pair on those exact positions, the raw
     data can price the pair directly with no class-mean step. That would be decisive.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import _DEX  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402

corpus_dir = production_corpus_dir(None)
tri = load_frequencies(str(corpus_dir / "trigrams.txt"))
surf = TimeSurface(tri, target_wpm=90.0, geometry=G)
T2 = surf._T2
positions = (*G.slots, G.space_position)
idx = {p: i for i, p in enumerate(positions)}


def kind(x):
    return G.finger(x).value.split("-")[1]


def ms(a, b):
    return float(T2[idx[a], idx[b]])


# ---------- (D2) the same-row two-finger baseline IN _T2 UNITS -------------------------
same_row = []
for a in G.slots:
    for b in G.slots:
        if a == b or not C.same_hand(G, a, b) or C.same_finger(G, a, b):
            continue
        if a[1] == b[1]:
            same_row.append(ms(a, b))
base_T2 = float(np.mean(same_row))
print(f"_T2 same-row same-hand two-finger baseline: {base_T2:.3f} ms  (over {len(same_row)} pairs)")


def rel_T2(a, b):
    return ms(a, b) / base_T2 - 1.0


# ---------- (D1) class means in _T2, over the SAME position pairs the raw cells cover ----
def class_pairs(lk, uk, dy):
    out = []
    for a in G.slots:
        for b in G.slots:
            if a == b or not C.same_hand(G, a, b) or C.same_finger(G, a, b):
                continue
            if abs(a[1] - b[1]) != dy:
                continue
            lower = a if a[1] < b[1] else b
            upper = b if a[1] < b[1] else a
            if kind(lower[0]) == lk and kind(upper[0]) == uk:
                out.append((a, b))
    return out


RAW = {
    ("middle", "pinky", 2): +0.2643,
    ("middle", "pinky", 1): +0.1494,
    ("pinky", "middle", 1): +0.0181,
    ("pinky", "middle", 2): +0.4122,
    ("middle", "index", 2): +0.7136,
    ("index", "middle", 2): +0.0066,
}
print("\n" + "=" * 104)
print("(D1)+(D2) CLASS-LEVEL comparison: raw-cell rel  vs  _T2 rel over the SAME position pairs")
print("=" * 104)
print(f"{'class (lower|upper|dy)':<34}{'raw rel':>10}{'_T2 class rel':>15}{'_T2 mean ms':>13}{'n pairs':>9}  agree?")
rows = []
for (lk, uk, dy), raw in RAW.items():
    ps = class_pairs(lk, uk, dy)
    vals = [rel_T2(a, b) for a, b in ps]
    m = float(np.mean(vals))
    agree = "YES" if (m > 0) == (raw > 0) else "sign differs"
    print(f"{f'{lk}|{uk}|dy{dy}':<34}{raw:>+10.4f}{m:>+15.4f}{np.mean([ms(a,b) for a,b in ps]):>13.2f}"
          f"{len(ps):>9}  {agree}")
    rows.append({"class": f"{lk}|{uk}|dy{dy}", "raw": raw, "t2_class_rel": m, "n_pairs": len(ps)})

# rank correlation between the two paths at class level
raw_v = [r["raw"] for r in rows]
t2_v = [r["t2_class_rel"] for r in rows]
order_raw = sorted(range(len(rows)), key=lambda i: raw_v[i])
order_t2 = sorted(range(len(rows)), key=lambda i: t2_v[i])
print(f"\n  raw ordering : {[rows[i]['class'] for i in order_raw]}")
print(f"  _T2 ordering : {[rows[i]['class'] for i in order_t2]}")
rk_raw = {i: r for r, i in enumerate(order_raw)}
rk_t2 = {i: r for r, i in enumerate(order_t2)}
n = len(rows)
d2 = sum((rk_raw[i] - rk_t2[i]) ** 2 for i in range(n))
rho = 1 - 6 * d2 / (n * (n * n - 1))
print(f"  Spearman rho (raw vs _T2, class level, n={n}) = {rho:+.4f}")

# ---------- the specific bl question, at BOTH levels ----------------------------------
print("\n" + "=" * 104)
print("THE bl QUESTION AT BOTH LEVELS OF AGGREGATION")
print("=" * 104)
bl_lsb = ((3, 1), (5, 3))
bl_lm = ((3, 1), (5, 2))
for name, (a, b) in (("keybo-lsb  b(3,1)->l(5,3) dy2", bl_lsb), ("keybo-lsb+lm b(3,1)->l(5,2) dy1", bl_lm)):
    both = (ms(a, b) + ms(b, a)) / 2
    print(f"  PAIR-LEVEL  {name}: mean {both:.2f} ms   rel_T2 {both/base_T2-1.0:+.4f}")
print(f"  => pair-level _T2 says +lm is "
      f"{'COSTLIER' if (ms(*bl_lm)+ms(bl_lm[1],bl_lm[0])) > (ms(*bl_lsb)+ms(bl_lsb[1],bl_lsb[0])) else 'cheaper'}")
c2 = float(np.mean([rel_T2(a, b) for a, b in class_pairs("middle", "pinky", 2)]))
c1 = float(np.mean([rel_T2(a, b) for a, b in class_pairs("middle", "pinky", 1)]))
print(f"  CLASS-LEVEL _T2: middle|pinky|dy2 {c2:+.4f}  ->  dy1 {c1:+.4f}   "
      f"=> class-level _T2 says +lm is {'CHEAPER' if c1 < c2 else 'costlier'}")
print(f"  CLASS-LEVEL raw: dy2 +0.2643 -> dy1 +0.1494  => raw says CHEAPER")

# ---------- (D3) is the exact position pair OBSERVED in the Aalto sample? -------------
print("\n" + "=" * 104)
print("(D3) DIRECT OBSERVATION: does the Aalto sample contain these exact position pairs?")
print("=" * 104)
TSV = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
PUNCT = frozenset(".,'-;/[]\\=")
BUCKETS = (40, 60, 80, 100, 120)
TARGETS = {
    "(3,1)<->(5,3)  [bl on keybo-lsb]": {frozenset({(3, 1), (5, 3)})},
    "(3,1)<->(5,2)  [bl on keybo-lsb+lm]": {frozenset({(3, 1), (5, 2)})},
    "(-3,1)<->(-5,3) [mirror of lsb]": {frozenset({(-3, 1), (-5, 3)})},
    "(-3,1)<->(-5,2) [mirror of +lm]": {frozenset({(-3, 1), (-5, 2)})},
}
found = {k: defaultdict(lambda: [0.0, 0]) for k in TARGETS}
bg_seen = {k: set() for k in TARGETS}
base = defaultdict(lambda: [0.0, 0])


def bucket_of(w):
    for b in BUCKETS:
        if b - 10 <= w < b + 10:
            return b
    return None


with open(TSV, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) < 5:
            continue
        lay, ngram = p[0], p[2]
        if PUNCT & set(ngram) or " " in ngram:
            continue
        nums = p[1].replace("(", " ").replace(")", " ").replace(",", " ").split()
        if len(nums) != 4:
            continue
        try:
            ax, ay, bx, by = (int(v) for v in nums)
        except ValueError:
            continue
        if ax == 0 or bx == 0 or (ax > 0) != (bx > 0):
            continue
        fa, fb = kind(ax), kind(bx)
        if fa == fb:
            continue
        samples = []
        for tok in p[4:]:
            tok = tok.strip()
            if not tok.startswith("("):
                continue
            q = tok.strip("()").split(",")
            if len(q) < 3:
                continue
            try:
                samples.append((int(q[0]), float(q[1])))
            except ValueError:
                continue
        if not samples:
            continue
        if ay == by:
            for w, d in samples:
                bk = bucket_of(w)
                if bk:
                    base[bk][0] += d
                    base[bk][1] += 1
            continue
        key = frozenset({(ax, ay), (bx, by)})
        for tname, want in TARGETS.items():
            if key in want:
                bg_seen[tname].add((lay, ngram))
                for w, d in samples:
                    bk = bucket_of(w)
                    if bk:
                        found[tname][bk][0] += d
                        found[tname][bk][1] += 1

base_ms = {bk: s / n for bk, (s, n) in base.items() if n >= 200}
for tname in TARGETS:
    tot = sum(n for (_s, n) in found[tname].values())
    print(f"\n  {tname}: n_raw {tot}, bigram-identities {sorted(bg_seen[tname])}")
    if tot == 0:
        print("     NOT OBSERVED — no source layout places any letter pair on these exact positions")
        continue
    rels = []
    for bk, (s, n) in sorted(found[tname].items()):
        if bk in base_ms:
            r = s / n / base_ms[bk] - 1.0
            mark = "USED" if n >= 200 else "dropped(n<200)"
            print(f"     bucket {bk:>3}: n {n:>7}  rel {r:+.4f}  {mark}")
            if n >= 200:
                rels.append(r)
    if rels:
        print(f"     => DIRECT rel = {sum(rels)/len(rels):+.4f}")
    else:
        print("     => observed but NO bucket clears the n>=200 floor: not directly estimable")

json.dump({"class_level": rows, "spearman_rho": rho, "t2_baseline_ms": base_T2},
          open("/tmp/lmscissor_audit3.json", "w"), indent=2)
print("\nwrote /tmp/lmscissor_audit3.json")
