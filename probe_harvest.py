"""lmscissor: ONE pass over the Aalto TSV -> compact per-(cell, bigram, bucket) aggregates.

The TSV is 609MB but only ~2202 rows; the bulk is inline per-keystroke samples. Reducing to
(sum_duration, n, n_pids) per (cell, bigram-identity, wpm-bucket) makes every downstream
estimate and bootstrap draw cheap, and means the 609MB read happens exactly once.

Cell keys emitted (all from the same row, so all mutually consistent):
  E:<lower>|<upper>|dy<n>                     explicit  (lower finger x upper finger x dy)
  C:<weakLOWER|weakTOP>|dy<n>|<adj|nonadj>    coarse    (the shipped predicate's orientation x dy x adjacency)
  L:lower=<kind>|dy<n>                        lower-finger-matched fallback
  A:<weakLOWER|weakTOP>|dy<n>                 the shipped predicate's own view
  X:<weakLOWER|weakTOP>|dy<n>|lower=<index|nonindex>   the decisive split
  P:<pair>|<weakLOWER|weakTOP>|dy<n>          per finger pair
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TSV = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
OUT = Path("/tmp/lmscissor_harvest.json")

PUNCT = frozenset(".,'-;/[]\\=")
BUCKETS = (40, 60, 80, 100, 120)
_ABS = {6: "pinky", 5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}
_DEX = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}


def bucket_of(w: int) -> int | None:
    for b in BUCKETS:
        if b - 10 <= w < b + 10:
            return b
    return None


# cell -> bigram -> bucket -> [sum_dur, n]
agg: dict[str, dict[str, dict[int, list]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
)
# cell -> set of pids (support check)
pids: dict[str, set] = defaultdict(set)
# baseline: bucket -> [sum, n]
baseline: dict[int, list] = defaultdict(lambda: [0.0, 0])
# provenance: cell -> bigram -> set of source layouts + positions
prov: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

n_rows = n_kept = 0
with open(TSV, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        n_rows += 1
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
                samples.append((int(p[0]), float(p[1]), int(p[2])))
            except ValueError:
                continue
        if not samples:
            continue
        n_kept += 1

        if ay == by:  # same-row same-hand two-finger -> the baseline
            for w, d, _p in samples:
                bk = bucket_of(w)
                if bk:
                    baseline[bk][0] += d
                    baseline[bk][1] += 1
            continue

        dy = abs(ay - by)
        lower_kind = fa if ay < by else fb
        upper_kind = fb if ay < by else fa
        weak_lower = _DEX[lower_kind] < _DEX[upper_kind]
        adjacent = abs(abs(ax) - abs(bx)) == 1 or {abs(ax), abs(bx)} == {6, 4}
        orient = "weakLOWER" if weak_lower else "weakTOP"
        pair = "-".join(sorted((fa, fb), key=lambda k: -_DEX[k]))

        keys = (
            f"E:{lower_kind}|{upper_kind}|dy{dy}",
            f"C:{orient}|dy{dy}|{'adj' if adjacent else 'nonadj'}",
            f"L:lower={lower_kind}|dy{dy}",
            f"A:{orient}|dy{dy}",
            f"X:{orient}|dy{dy}|lower={'index' if lower_kind == 'index' else 'nonindex'}",
            f"P:{pair}|{orient}|dy{dy}",
        )
        for key in keys:
            cell = agg[key][ngram]
            prov[key][ngram].add(src)
            pid_set = pids[key]
            for w, d, p in samples:
                bk = bucket_of(w)
                if bk:
                    slot = cell[bk]
                    slot[0] += d
                    slot[1] += 1
                    pid_set.add(p)

print(f"rows read {n_rows}, rows kept {n_kept}")
out = {
    "baseline": {str(k): v for k, v in sorted(baseline.items())},
    "cells": {
        key: {
            "n_pids": len(pids[key]),
            "bigrams": {
                bg: {str(bk): v for bk, v in sorted(buckets.items())}
                for bg, buckets in by_bg.items()
            },
            "sources": {bg: sorted(s) for bg, s in prov[key].items()},
        }
        for key, by_bg in agg.items()
    },
}
json.dump(out, open(OUT, "w"))
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes, {len(out['cells'])} cells)")
