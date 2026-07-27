"""REFLECT audit item 1+2, stage 1: lean TSV pass -> compact per-unit aggregate.

Applying my own T43 lesson (a backgrounded long job dies with no rc sentinel): do the 609MB read
ONCE, write a small file, then do all arithmetic and bootstrapping offline in seconds.

Scope: the weak-on-TOP dy==2 class ONLY (the class whose n I must reconcile and whose sub-split the
defect claim rests on), keyed by (source_layout, bigram, wpm_bucket) so BOTH accounting rules can be
reproduced from the same pass:
  * bs01/mine : no per-unit floor, POOLED across-layout baseline
  * bs06/spec : MIN_UNIT=50 per (layout,bigram,bucket), PER-LAYOUT baseline
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TSV = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
OUT = Path("/tmp/lmscissor_audit1_units.json")
PUNCT = frozenset(".,'-;/[]\\=")
BUCKETS = (40, 60, 80, 100, 120)
_ABS = {6: "pinky", 5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}
_DEX = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}


def bucket_of(w):
    for b in BUCKETS:
        if b - 10 <= w < b + 10:
            return b
    return None


pooled = defaultdict(lambda: [0.0, 0])       # bucket -> [sum, n]
perlayout = defaultdict(lambda: [0.0, 0])    # "layout|bucket" -> [sum, n]
units = defaultdict(lambda: [0.0, 0])        # "layout|bigram|bucket" -> [sum, n]
unit_meta = {}                               # same key -> lower_kind
unit_pids = defaultdict(set)

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
            ax, ay, bx, by = (int(n) for n in nums)
        except ValueError:
            continue
        if ax == 0 or bx == 0 or (ax > 0) != (bx > 0):
            continue
        fa, fb = _ABS[abs(ax)], _ABS[abs(bx)]
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
                samples.append((int(q[0]), float(q[1]), int(q[2])))
            except ValueError:
                continue
        if not samples:
            continue
        if ay == by:
            for w, d, _pid in samples:
                bk = bucket_of(w)
                if bk:
                    pooled[bk][0] += d
                    pooled[bk][1] += 1
                    pl = perlayout[f"{lay}|{bk}"]
                    pl[0] += d
                    pl[1] += 1
            continue
        dy = abs(ay - by)
        weak = min((fa, fb), key=lambda f: _DEX[f])
        wy = ay if fa == weak else by
        sy = by if fa == weak else ay
        if not (dy == 2 and wy > sy):   # weak-on-TOP, dy == 2 only
            continue
        lower_kind = fa if ay < by else fb
        for w, d, pid in samples:
            bk = bucket_of(w)
            if bk:
                key = f"{lay}|{ngram}|{bk}"
                u = units[key]
                u[0] += d
                u[1] += 1
                unit_meta[key] = lower_kind
                unit_pids[key].add(pid)

json.dump(
    {
        "pooled_baseline": {str(k): v for k, v in sorted(pooled.items())},
        "perlayout_baseline": dict(perlayout),
        "units": dict(units),
        "unit_lower": unit_meta,
        "unit_npids": {k: len(v) for k, v in unit_pids.items()},
    },
    open(OUT, "w"),
)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes); units={len(units)}")
