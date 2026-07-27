"""REFLECT audit item 1+2, stage 2: offline arithmetic from the 21KB aggregate.

Item 1: reconcile my n=1,644,724 against the spec's n=1,643,289 by reproducing BOTH accounting rules.
Item 2: CI the lower=NON-index sub-cell the defect claim rests on.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

D = json.load(open("/tmp/lmscissor_audit1_units.json"))
pooled = {int(k): v for k, v in D["pooled_baseline"].items()}
perlayout = D["perlayout_baseline"]
units = D["units"]
lower_of = D["unit_lower"]
npids = D["unit_npids"]

MIN_RAW, MIN_PIDS, MIN_BIGRAMS, MIN_UNIT = 200, 20, 3, 50
pooled_ms = {bk: s / n for bk, (s, n) in pooled.items() if n >= MIN_RAW}
perlayout_ms = {k: s / n for k, (s, n) in perlayout.items() if n >= MIN_RAW}


def parts(key):
    lay, bg, bk = key.split("|")
    return lay, bg, int(bk)


total_mine = sum(n for (_s, n) in units.values())
kept_bs06 = {
    k: v for k, v in units.items()
    if v[1] >= MIN_UNIT and f"{parts(k)[0]}|{parts(k)[2]}" in perlayout_ms
}
total_bs06 = sum(n for (_s, n) in kept_bs06.values())
dropped_small = sum(n for k, (_s, n) in units.items() if n < MIN_UNIT)
dropped_nobase = sum(
    n for k, (_s, n) in units.items()
    if n >= MIN_UNIT and f"{parts(k)[0]}|{parts(k)[2]}" not in perlayout_ms
)

print("=" * 100)
print("ITEM 1 — reconciling n = 1,644,724 (mine) vs n = 1,643,289 (spec / bs06_orientation.json)")
print("=" * 100)
print(f"  my accounting  (bs01 rule: no per-unit floor, pooled baseline) : {total_mine}")
print(f"  bs06 accounting (MIN_UNIT=50 per unit + PER-LAYOUT baseline)   : {total_bs06}")
print(f"  the spec's published n                                          : 1643289")
print(f"  MATCH? {'YES — EXACT' if total_bs06 == 1643289 else 'NO'}")
print(f"  my n minus bs06 n = {total_mine - total_bs06}   (the report's unexplained 1,435)")
print(f"     dropped by MIN_UNIT<50               : {dropped_small}")
print(f"     dropped by missing per-layout baseline: {dropped_nobase}")
print(f"  distinct bigrams mine : {len({parts(k)[1] for k in units})}")
print(f"  distinct bigrams bs06 : {len({parts(k)[1] for k in kept_bs06})}   (spec says 18)")

# which units bs06 drops, and are any of them non-index?
dropped_keys = [k for k in units if k not in kept_bs06]
dn = defaultdict(int)
for k in dropped_keys:
    dn[lower_of[k]] += units[k][1]
print(f"  lower-finger census of the {len(dropped_keys)} dropped units: {dict(dn)}")


def estimate(keys, per_layout: bool):
    """(rel, n_raw, n_pids, bigrams, per_bucket) under the chosen baseline convention."""
    if per_layout:
        # bs06: per (layout,bucket) rel, n-weighted mean over units (its nwmean)
        num = den = 0.0
        n_raw = 0
        bgs, pids = set(), 0
        for k in keys:
            s, n = units[k]
            lay, bg, bk = parts(k)
            b = perlayout_ms.get(f"{lay}|{bk}")
            if b is None or n < MIN_UNIT:
                continue
            rel = (s / n) / b - 1.0
            num += rel * n
            den += n
            n_raw += n
            bgs.add(bg)
            pids += npids[k]
        return (num / den if den else None), n_raw, pids, len(bgs), {}
    agg = defaultdict(lambda: [0.0, 0])
    bgs, pids = set(), 0
    for k in keys:
        s, n = units[k]
        _lay, bg, bk = parts(k)
        a = agg[bk]
        a[0] += s
        a[1] += n
        bgs.add(bg)
        pids += npids[k]
    rels, n_raw, pb = [], 0, {}
    for bk, (s, n) in sorted(agg.items()):
        n_raw += n
        if bk in pooled_ms:
            r = (s / n) / pooled_ms[bk] - 1.0
            pb[bk] = (n, r, n >= MIN_RAW)
            if n >= MIN_RAW:
                rels.append(r)
        else:
            pb[bk] = (n, None, False)
    return (sum(rels) / len(rels) if rels else None), n_raw, pids, len(bgs), pb


print(f"\n  rel under MY rule  : {estimate(list(units), False)[0]:+.4f}   (report said -0.0139)")
print(f"  rel under bs06 rule: {estimate(list(units), True)[0]:+.4f}   (spec said -0.017863)")

print("\n" + "=" * 100)
print("ITEM 2 — the decisive sub-cell CI: weakTOP dy2 split by lower-finger identity")
print("=" * 100)

for label, pred in (
    ("lower = index", lambda k: lower_of[k] == "index"),
    ("lower = NON-index", lambda k: lower_of[k] != "index"),
):
    keys = [k for k in units if pred(k)]
    rel, n_raw, pids, nbg, pb = estimate(keys, False)
    print(f"\n  {label}: rel {rel:+.4f}  n_raw {n_raw}  bigrams {nbg}")
    print(f"    bigrams: {sorted({parts(k)[1] for k in keys})}")
    print(f"    sources: {dict(sorted(((l, sum(units[k][1] for k in keys if parts(k)[0]==l)) for l in {parts(k)[0] for k in keys}), key=lambda kv:-kv[1]))}")
    print(f"    per-bucket (rel averaged ONLY over buckets with n>={MIN_RAW}):")
    for bk, (n, r, used) in pb.items():
        rs = f"{r:+.4f}" if r is not None else "  n/a "
        print(f"      bucket {bk:>3}: n {n:>8}  rel {rs}  {'USED' if used else 'DROPPED (n<200)'}")

    rng = random.Random(20260727)
    ids = sorted({parts(k)[1] for k in keys})
    by_id = defaultdict(list)
    for k in keys:
        by_id[parts(k)[1]].append(k)
    draws = []
    for _ in range(4000):
        samp = [rng.choice(ids) for _ in ids]
        kk = [k for b in samp for k in by_id[b]]
        r, *_ = estimate(kk, False)
        if r is not None:
            draws.append(r)
    draws.sort()
    lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
    p_pos = sum(1 for x in draws if x > 0) / len(draws)
    print(f"    bigram-clustered 95% CI [{lo:+.4f}, {hi:+.4f}]  P(rel>0)={p_pos:.3f}  "
          f"draws {len(draws)}/4000")
    # how much survives bs06's stricter rule?
    kept = [k for k in keys if k in kept_bs06]
    print(f"    under bs06's MIN_UNIT=50 + per-layout base: "
          f"{sum(units[k][1] for k in kept)} of {n_raw} samples, "
          f"{len({parts(k)[1] for k in kept})} of {nbg} bigrams survive")
    if kept:
        r2, *_ = estimate(kept, True)
        print(f"    rel under bs06 rule on the survivors: {r2:+.4f}" if r2 is not None else
              "    rel under bs06 rule: UNMEASURABLE")

json.dump(
    {"n_mine": total_mine, "n_bs06": total_bs06, "spec_n": 1643289,
     "dropped_small": dropped_small, "dropped_nobase": dropped_nobase},
    open("/tmp/lmscissor_audit1.json", "w"), indent=2)
print("\nwrote /tmp/lmscissor_audit1.json")
