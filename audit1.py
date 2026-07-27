"""lmscissor REFLECT audit items 1 + 2: explain the n-discrepancy, and CI the decisive sub-cell.

Item 1. My weakTOP dy2 n=1,644,724 vs the spec's 1,643,289 (diff 1,435). Candidate cause found by
reading the producing script: the spec's -0.0179 comes from `bs06_orientation.json`
`row_grid["weak=top,strong=bottom"]`, whose load() keeps a (layout, bigram, bucket) unit ONLY if it
holds >= MIN_UNIT=50 samples, AND requires a PER-LAYOUT baseline (>=200). My probe mirrored
`bs01_surface.py` instead: no per-unit floor, and a POOLED (across-layout) baseline. Test the
attribution by reproducing BOTH accounting rules on the same pass.

Item 2. CI the lower=NON-index sub-cell (+0.2777, n=515) that the defect claim rests on, with its
per-bucket support laid bare, and say whether the claim degrades to "untested" rather than "wrong".
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

TSV = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
PUNCT = frozenset(".,'-;/[]\\=")
BUCKETS = (40, 60, 80, 100, 120)
_ABS = {6: "pinky", 5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}
_DEX = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}
MIN_UNIT = 50           # bs06's per-(layout,bigram,bucket) floor
MIN_RAW, MIN_PIDS = 200, 20   # bs01's cell-level floors (what I used)


def bucket_of(w):
    for b in BUCKETS:
        if b - 10 <= w < b + 10:
            return b
    return None


# weakTOP dy2 accounting under both rules
pooled_base = defaultdict(list)                 # bucket -> durations           (bs01 / mine)
perlayout_base = defaultdict(list)              # (layout,bucket) -> durations  (bs06)
# (layout, bigram, bucket) -> [durations] for the weakTOP-dy2 class, split by lower-finger identity
units: dict[tuple, list] = defaultdict(list)
unit_lower: dict[tuple, str] = {}
unit_pids: dict[tuple, set] = defaultdict(set)

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
                    pooled_base[bk].append(d)
                    perlayout_base[(lay, bk)].append(d)
            continue
        dy = abs(ay - by)
        weak = min((fa, fb), key=lambda f: _DEX[f])
        wy = ay if fa == weak else by
        sy = by if fa == weak else ay
        if not (dy == 2 and wy > sy):        # weak-on-TOP, dy == 2
            continue
        lower_kind = fa if ay < by else fb
        for w, d, pid in samples:
            bk = bucket_of(w)
            if bk:
                key = (lay, ngram, bk)
                units[key].append(d)
                unit_lower[key] = lower_kind
                unit_pids[key].add(pid)

pooled_ms = {bk: sum(v) / len(v) for bk, v in pooled_base.items() if len(v) >= MIN_RAW}
perlayout_ms = {k: sum(v) / len(v) for k, v in perlayout_base.items() if len(v) >= MIN_RAW}

total_all = sum(len(v) for v in units.values())
kept_bs06 = {
    k: v
    for k, v in units.items()
    if len(v) >= MIN_UNIT and (k[0], k[2]) in perlayout_ms
}
total_bs06 = sum(len(v) for v in kept_bs06.values())

print("=" * 96)
print("ITEM 1 — the n-discrepancy, attributed")
print("=" * 96)
print(f"  MY accounting (bs01 rule: no per-unit floor, pooled baseline)   n = {total_all}")
print(f"     … reported in my report/table                                    1644724")
print(f"  bs06 accounting (MIN_UNIT=50 per (layout,bigram,bucket) + per-layout baseline)")
print(f"                                                                 n = {total_bs06}")
print(f"     … the spec's published figure                                    1643289")
print(f"  difference under bs06 rule vs mine: {total_all - total_bs06}")
dropped_small = sum(len(v) for k, v in units.items() if len(v) < MIN_UNIT)
dropped_nobase = sum(
    len(v) for k, v in units.items() if len(v) >= MIN_UNIT and (k[0], k[2]) not in perlayout_ms
)
print(f"     of which dropped by MIN_UNIT<50            : {dropped_small}")
print(f"     of which dropped by missing per-layout base: {dropped_nobase}")
print(f"  distinct bigrams, my accounting  : {len({k[1] for k in units})}")
print(f"  distinct bigrams, bs06 accounting: {len({k[1] for k in kept_bs06})}  (spec says 18)")

print("\n" + "=" * 96)
print("ITEM 2 — the decisive sub-cell: weakTOP dy2 by lower-finger identity, with CIs")
print("=" * 96)


def cell_estimate(keys, base, per_layout: bool):
    """(rel, n_raw, n_pids, n_bigrams, per_bucket) under the given baseline convention."""
    agg = defaultdict(list)
    pids = set()
    bgs = set()
    for k in keys:
        agg[k[2]].extend(units[k])
        pids |= unit_pids[k]
        bgs.add(k[1])
    rels, n_raw, per_bucket = [], 0, {}
    for bk, ds in sorted(agg.items()):
        n_raw += len(ds)
        b = base.get(bk) if not per_layout else None
        if b is None and not per_layout:
            continue
        r = sum(ds) / len(ds) / b - 1.0 if b else None
        per_bucket[bk] = {"n": len(ds), "rel": r}
        if r is not None and len(ds) >= MIN_RAW:
            rels.append(r)
    return (
        (sum(rels) / len(rels)) if rels else None,
        n_raw,
        len(pids),
        len(bgs),
        per_bucket,
        sorted(bgs),
    )


for label, pred in (
    ("lower = index", lambda k: unit_lower[k] == "index"),
    ("lower = NON-index", lambda k: unit_lower[k] != "index"),
):
    keys = [k for k in units if pred(k)]
    rel, n_raw, n_pids, n_bg, pb, bgs = cell_estimate(keys, pooled_ms, False)
    print(f"\n  {label}:  rel {rel:+.4f}   n_raw {n_raw}   pids {n_pids}   bigrams {n_bg}")
    print(f"     bigrams: {bgs}")
    print(f"     per-bucket support (rel averaged ONLY over buckets with n>={MIN_RAW}):")
    for bk, v in pb.items():
        used = "USED" if (v["rel"] is not None and v["n"] >= MIN_RAW) else "dropped(n<200)"
        rr = f"{v['rel']:+.4f}" if v["rel"] is not None else "  n/a  "
        print(f"       bucket {bk:>3}: n {v['n']:>8}  rel {rr}  {used}")
    # sources
    src = defaultdict(int)
    for k in keys:
        src[k[0]] += len(units[k])
    print(f"     by source layout: {dict(sorted(src.items(), key=lambda kv: -kv[1]))}")

    # bootstrap over bigram identities
    rng = random.Random(20260727)
    ids = sorted({k[1] for k in keys})
    by_id = defaultdict(list)
    for k in keys:
        by_id[k[1]].append(k)
    draws = []
    for _ in range(4000):
        samp = [rng.choice(ids) for _ in ids]
        kk = [k for bid in samp for k in by_id[bid]]
        r, *_ = cell_estimate(kk, pooled_ms, False)
        if r is not None:
            draws.append(r)
    draws.sort()
    if draws:
        lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
        p_pos = sum(1 for x in draws if x > 0) / len(draws)
        print(
            f"     bigram-clustered 95% CI [{lo:+.4f}, {hi:+.4f}]   "
            f"P(rel > 0) = {p_pos:.3f}   draws {len(draws)}/4000"
        )
    # also: participant-level bootstrap is not available (pids are per-sample, durations pooled)

    # how much of this sub-cell would bs06's MIN_UNIT=50 have KEPT?
    kept = [k for k in keys if len(units[k]) >= MIN_UNIT and (k[0], k[2]) in perlayout_ms]
    print(
        f"     under bs06's MIN_UNIT=50 + per-layout baseline: "
        f"{sum(len(units[k]) for k in kept)} of {n_raw} samples survive, "
        f"{len({k[1] for k in kept})} of {n_bg} bigrams"
    )

json.dump(
    {"n_mine": total_all, "n_bs06_rule": total_bs06, "dropped_small": dropped_small,
     "dropped_nobase": dropped_nobase},
    open("/tmp/lmscissor_audit1.json", "w"),
    indent=2,
)
print("\nwrote /tmp/lmscissor_audit1.json")
