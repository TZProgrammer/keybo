"""lmscissor: the crux. R6 (explicit cells) flips the order, R7 (coarse cells) does not.
The whole disagreement is ONE question:

  is the class `bl` sits in (lower=middle, upper=pinky, dy2) cheap (its coarse aggregate
  weakTOP|dy2|nonadj = -0.0500) or costly (its explicit cell = +0.2643)?

Print every bigram identity in both crux classes with its own relative excess, source layout,
and n — so the reader can see what the estimate rests on (trap 16: check whether the key sets
are disjoint, which would mean the contrast is not identified).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TSV = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
PUNCT = frozenset(".,'-;/[]\\=")
BUCKETS = (40, 60, 80, 100, 120)
_ABS = {6: "pinky", 5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}
_ROW = {3: "top", 2: "home", 1: "bottom"}


def bucket_of(w):
    for b in BUCKETS:
        if b - 10 <= w < b + 10:
            return b
    return None


CLASSES = {
    "ld-class  lower=pinky upper=middle dy1": ("pinky", "middle", 1),
    "bl-class  lower=middle upper=pinky dy2": ("middle", "pinky", 2),
    "bl-class-dy1 lower=middle upper=pinky dy1": ("middle", "pinky", 1),
    "motivating lower=pinky upper=middle dy2": ("pinky", "middle", 2),
}

# (class, bigram, source_layout) -> bucket -> [durations]
detail = defaultdict(lambda: defaultdict(list))
baseline = defaultdict(list)
# coarse: weakTOP|dy2|nonadj  -- every bigram in it, to see what drives the -0.0500
coarse_detail = defaultdict(lambda: defaultdict(list))

with open(TSV, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        src, ngram = parts[0], parts[2]
        if PUNCT & set(ngram) or " " in ngram:
            continue
        nums = parts[1].replace("(", " ").replace(")", " ").replace(",", " ").split()
        if len(nums) != 4:
            continue
        try:
            ax, ay, bx, by = (int(n) for n in nums)
        except ValueError:
            continue
        if ax == 0 or bx == 0 or (ax > 0) != (bx > 0):
            continue
        fa, fb = _ABS[abs(ax)], _ABS[abs(bx)]
        if fa == fb:
            continue
        samples = []
        for tok in parts[4:]:
            tok = tok.strip()
            if not tok.startswith("("):
                continue
            p = tok.strip("()").split(",")
            if len(p) < 3:
                continue
            try:
                samples.append((int(p[0]), float(p[1])))
            except ValueError:
                continue
        if not samples:
            continue
        if ay == by:
            for w, d in samples:
                bk = bucket_of(w)
                if bk:
                    baseline[bk].append(d)
            continue
        dy = abs(ay - by)
        lower_kind = fa if ay < by else fb
        upper_kind = fb if ay < by else fa
        _DEX = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}
        weak_lower = _DEX[lower_kind] < _DEX[upper_kind]
        adjacent = abs(abs(ax) - abs(bx)) == 1 or {abs(ax), abs(bx)} == {6, 4}

        for cname, (lk, uk, tdy) in CLASSES.items():
            if lower_kind == lk and upper_kind == uk and dy == tdy:
                lowpos = (ax, ay) if ay < by else (bx, by)
                upppos = (bx, by) if ay < by else (ax, ay)
                key = (cname, ngram, src, lowpos, upppos)
                for w, d in samples:
                    bk = bucket_of(w)
                    if bk:
                        detail[key][bk].append(d)

        if (not weak_lower) and dy == 2 and not adjacent:
            for w, d in samples:
                bk = bucket_of(w)
                if bk:
                    coarse_detail[(ngram, src, lower_kind, upper_kind)][bk].append(d)

base_ms = {bk: sum(v) / len(v) for bk, v in baseline.items() if len(v) >= 200}
print(f"baseline: { {k: round(v,2) for k,v in sorted(base_ms.items())} }")


def rel_of(by_bucket, min_raw=200):
    rels, n = [], 0
    for bk, ds in by_bucket.items():
        n += len(ds)
        if bk in base_ms and len(ds) >= min_raw:
            rels.append(sum(ds) / len(ds) / base_ms[bk] - 1.0)
    return (sum(rels) / len(rels) if rels else None), n


for cname in CLASSES:
    print(f"\n{'='*100}\n{cname}\n{'='*100}")
    print(f"{'bigram':<8}{'src':<9}{'lower pos':<14}{'upper pos':<14}{'rel':>10}{'n_raw':>10}")
    rows = []
    for (cn, bg, src, lp, up), byb in detail.items():
        if cn != cname:
            continue
        r, n = rel_of(byb)
        rows.append((bg, src, lp, up, r, n))
    rows.sort(key=lambda t: -(t[4] if t[4] is not None else -9))
    for bg, src, lp, up, r, n in rows:
        rs = f"{r:+.4f}" if r is not None else "   n/a  "
        print(f"{bg:<8}{src:<9}{str(lp):<14}{str(up):<14}{rs:>10}{n:>10}")
    letters = {bg[0] for _, _, _, _, _, _ in [] } or set()
    keyset = set()
    for bg, src, lp, up, r, n in rows:
        keyset |= set(bg)
    print(f"  letter set: {sorted(keyset)}")

print(f"\n{'='*100}")
print("WHAT DRIVES THE COARSE weakTOP|dy2|nonadj = -0.0500 CELL (the R7 weight for `bl`)")
print(f"{'='*100}")
print(f"{'bigram':<8}{'src':<9}{'lower':<9}{'upper':<9}{'rel':>10}{'n_raw':>10}")
crows = []
for (bg, src, lk, uk), byb in coarse_detail.items():
    r, n = rel_of(byb)
    crows.append((bg, src, lk, uk, r, n))
crows.sort(key=lambda t: -t[5])
tot_by_lower = defaultdict(int)
for bg, src, lk, uk, r, n in crows:
    tot_by_lower[lk] += n
for bg, src, lk, uk, r, n in crows[:25]:
    rs = f"{r:+.4f}" if r is not None else "   n/a  "
    print(f"{bg:<8}{src:<9}{lk:<9}{uk:<9}{rs:>10}{n:>10}")
print(f"\n  n_raw by LOWER finger inside this coarse cell: {dict(tot_by_lower)}")
tot = sum(tot_by_lower.values())
for k, v in sorted(tot_by_lower.items(), key=lambda kv: -kv[1]):
    print(f"    lower={k:<8} {100.0*v/tot:6.2f}%  n={v}")

json.dump(
    {
        "note": "per-bigram detail for the crux classes; see stdout",
        "baseline_ms": base_ms,
    },
    open("/tmp/lmscissor_crux.json", "w"),
    indent=2,
)
