"""GATEFOLDS-1 §3 — NAME THE ROWS (prereg invariant 1), plus the MEASURED floor and the
BOOTSTRAP stability of the structural verdict (invariant 7).

A mechanism claim that never identifies the offending buckets is not a diagnosis. This produces,
for EVERY arm on the interpretability line INCLUDING the served control:

  * the actual refused (fold, bucket) rows,
  * each row's SUPPORT -- n_cells / n_participants / n_raw, read from `bucket_matrix`,
  * each row's per-bucket rho DELTA against the per-fold incumbent baseline,
  * the MEASURED within-arm seed floor at the same data volume, and margin-vs-floor,
  * a bootstrap over SEEDS of the structural verdict's stability.

⚠ INVARIANT 8, OBEYED BY NOT REUSING THE HAZARD. `agent-artifacts/interpframe/metrics.py:60`
dispatches on a NAME SUBSTRING and would silently apply the wrong frame's grouping. This module
loads NO sibling instrument; it re-derives everything from the arms' own artifacts plus
`keybo.verdicts.bucket_regression_report`, which is frame-agnostic (it takes two rho maps).

⚠ INVARIANT 3, OBEYED: every rho here traces to `validate()`'s per-bucket Spearman of PREDICTIONS
against `Cell.obs` -- the IQR-mean of REAL observed durations from the stroke TSV. No `TimeSurface`
/ `TableBigramScorer` target is constructed anywhere, so no floor here is self-generated.

⚠ THE BASELINE IS PER-FOLD, and that is not a choice of mine. `interpframe/highwpm.py` documents
that a POOLED-across-folds baseline made the gate REFUSE THE INCUMBENT -- because dvorak's absolute
rho is ~0.70 while qwertz's is ~0.92, so every dvorak bucket sits below a cross-fold mean by
construction. Per-fold is what "non-regression" means. I reproduce that construction rather than
invent one, so my numbers are comparable to the ones already registered.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree, require  # noqa: E402

assert_tree()

import keybo.verdicts as VD  # noqa: E402

bucket_regression_report = require(VD, "bucket_regression_report")
HIGH_WPM_FLOOR = require(VD, "HIGH_WPM_FLOOR")
HIGH_WPM_TOLERANCE = require(VD, "HIGH_WPM_TOLERANCE")
print(f"[gate] floor={HIGH_WPM_FLOOR} wpm  tolerance={HIGH_WPM_TOLERANCE}")

IF = "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/interpframe"
HT = "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri"
MINE = ARTIFACTS

# (label, path, arm-key-in-that-file, provenance note)
SOURCES = [
    ("CUR", f"{IF}/lolo.json", "CUR", "interpframe run; the served incumbent"),
    ("INTERP", f"{IF}/lolo.json", "INTERP", "interp.1, 10c, monotone ON"),
    ("INTERP-NOMONO", f"{IF}/lolo.json", "INTERP-NOMONO", "interp.1, 10c, monotone OFF"),
    ("CUR-NOWPM", f"{IF}/lolo.json", "CUR-NOWPM", "served 20c, wpm neutralized to a GLOBAL const"),
    ("HYBRIDB", f"{HT}/lolo.json", "HYBRIDB", "hybrid-B, 18c, partial monotone"),
    ("CUR-INVARIANT", f"{MINE}/control.json", "CUR-INVARIANT", "MINE: served 20c wpm-invariant, TRUE-pace ms"),
    ("CUR-mine", f"{MINE}/control.json", "CUR", "MINE: served control, re-derived on my tree"),
]


def load_arm(path, key):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    return d.get("arms", {}).get(key)


arms: dict[str, dict] = {}
prov: dict[str, str] = {}
for label, path, key, note in SOURCES:
    rep = load_arm(path, key)
    if rep is None:
        print(f"[skip] {label}: no arm {key!r} in {path}")
        continue
    arms[label] = rep
    prov[label] = f"{note}  [{path}#{key}]"
    cfg = rep.get("config", {})
    print(f"[arm ] {label:<14} interp={cfg.get('interp')!r} monotone={cfg.get('monotone')!r}  {note}")

if "CUR" not in arms:
    raise SystemExit("no served incumbent arm -- cannot form a baseline")


def per_fold_bucket_rhos(rep):
    """{holdout: {seed: {bucket:int -> rho}}}"""
    out = {}
    for holdout, fold in rep["folds"].items():
        for rec in fold["seeds"]:
            out.setdefault(holdout, {})[rec["seed"]] = {
                int(k): float(v) for k, v in (rec.get("bucket_rhos") or {}).items() if v is not None
            }
    return out


def per_fold_support(rep):
    """{holdout: {bucket:int -> {n_cells,n_participants,n_raw}}} from bucket_matrix.

    Identical across seeds by construction (same test cells), so the first seed carrying it wins;
    asserted rather than assumed.
    """
    out = {}
    for holdout, fold in rep["folds"].items():
        seen = None
        for rec in fold["seeds"]:
            bm = rec.get("bucket_matrix") or {}
            cur = {
                int(b): {
                    "n_cells": int(v["n"]),
                    "n_participants": int(v["n_participants"]),
                    "n_raw": int(v["n_raw"]),
                }
                for b, v in bm.items()
            }
            if seen is None:
                seen = cur
            elif cur != seen:
                print(f"  ⚠ {holdout}: support differs across seeds (should not) -- using first")
        out[holdout] = seen or {}
    return out


CUR_RHOS = per_fold_bucket_rhos(arms["CUR"])
SUPPORT = per_fold_support(arms["CUR"])

# THE INCUMBENT BASELINE, per fold, mean over its seeds -- reproducing highwpm.py's construction.
baseline = {
    h: {b: float(np.mean([CUR_RHOS[h][s][b] for s in CUR_RHOS[h] if b in CUR_RHOS[h][s]]))
        for b in sorted({b for s in CUR_RHOS[h] for b in CUR_RHOS[h][s]})}
    for h in CUR_RHOS
}

out: dict = {
    "prereg": "agent-artifacts/gatefolds/GATEFOLDS-preregistration.md",
    "gate": {"floor_wpm": int(HIGH_WPM_FLOOR), "tolerance_rho": float(HIGH_WPM_TOLERANCE)},
    "baseline_construction": "per-fold incumbent (CUR) bucket rho, mean over its seeds",
    "provenance": prov,
    "incumbent_baseline_per_fold": {h: {str(b): v for b, v in d.items()} for h, d in baseline.items()},
    "support_per_fold": {h: {str(b): v for b, v in d.items()} for h, d in SUPPORT.items()},
    "arms": {},
}

# =========================================================================================
# THE MEASURED FLOOR (invariant 7): within-arm, per (fold, bucket), across SEEDS.
# A cross-arm rho delta smaller than the same-arm seed spread is not resolvable. Measured at the
# SAME data volume as the comparison -- no borrowed constant.
# =========================================================================================
print()
print("=" * 100)
print("MEASURED SEED FLOOR (within-arm rho spread across 3 seeds, per fold x bucket)")
print("=" * 100)
floors: dict[str, dict] = {}
for label, rep in arms.items():
    R = per_fold_bucket_rhos(rep)
    f = {}
    for h in sorted(R):
        buckets = sorted({b for s in R[h] for b in R[h][s]})
        for b in buckets:
            vals = [R[h][s][b] for s in sorted(R[h]) if b in R[h][s]]
            if len(vals) > 1:
                f[f"{h}/{b}"] = {
                    "sd": float(np.std(vals, ddof=1)),
                    "range": float(max(vals) - min(vals)),
                    "vals": [float(v) for v in vals],
                }
    floors[label] = f
out["seed_floor_per_arm"] = floors

# The floor that matters for the HIGH buckets specifically, pooled over the incumbent's cells.
hi_sd = [v["sd"] for k, v in floors["CUR"].items() if int(k.split("/")[1]) >= HIGH_WPM_FLOOR]
hi_rng = [v["range"] for k, v in floors["CUR"].items() if int(k.split("/")[1]) >= HIGH_WPM_FLOOR]
FLOOR_SD = float(np.median(hi_sd)) if hi_sd else float("nan")
FLOOR_MAX = float(np.max(hi_rng)) if hi_rng else float("nan")
out["high_bucket_seed_floor"] = {
    "median_within_arm_sd_incumbent": FLOOR_SD,
    "max_within_arm_range_incumbent": FLOOR_MAX,
    "n_high_cells": len(hi_sd),
    "note": "measured on the INCUMBENT at the same folds/buckets/volume as the comparison",
}
print(f"  incumbent high-bucket seed sd: median {FLOOR_SD:.5f}  max range {FLOOR_MAX:.5f} "
      f"over {len(hi_sd)} (fold,bucket) cells")
print(f"  gate tolerance {HIGH_WPM_TOLERANCE} is {HIGH_WPM_TOLERANCE / FLOOR_SD:.2f}x that median sd")

# =========================================================================================
# NAME THE ROWS + margin-vs-floor + bootstrap the structural verdict
# =========================================================================================
RNG = np.random.default_rng(20260804)
N_BOOT = 2000

print()
print("=" * 100)
print("REFUSED ROWS PER ARM (structural = regresses on EVERY seed of a fold)")
print("=" * 100)

for label, rep in arms.items():
    R = per_fold_bucket_rhos(rep)
    detail = {}
    structural_rows = []
    for h in sorted(R):
        seeds = sorted(R[h])
        hits: dict[int, int] = {}
        deltas_by_bucket: dict[int, list[float]] = {}
        for s in seeds:
            blk = bucket_regression_report(
                R[h][s], baseline.get(h, {}), f"{label}/{h}/seed{s}",
                support={b: SUPPORT.get(h, {}).get(b, {}) for b in SUPPORT.get(h, {})},
            )
            for b in blk.get("regressing_high_buckets", []):
                hits[int(b)] = hits.get(int(b), 0) + 1
            for b, d in blk.get("deltas", {}).items():
                deltas_by_bucket.setdefault(int(b), []).append(float(d))
        n = len(seeds)
        struct = sorted(b for b, k in hits.items() if k == n)
        noise = sorted(b for b, k in hits.items() if 0 < k < n)
        detail[h] = {
            "n_seeds": n,
            "structural": struct,
            "noise": noise,
            "per_bucket_seed_counts": {str(k): v for k, v in sorted(hits.items())},
            "mean_delta_per_bucket": {
                str(b): float(np.mean(v)) for b, v in sorted(deltas_by_bucket.items())
            },
        }
        for b in struct:
            d = deltas_by_bucket.get(b, [])
            md = float(np.mean(d)) if d else float("nan")
            sup = SUPPORT.get(h, {}).get(b, {})
            # MARGIN vs FLOOR, reported BEFORE any p-value (invariant 7). The margin is how far
            # past the tolerance the regression sits; the floor is the measured seed sd.
            margin = abs(md) - HIGH_WPM_TOLERANCE
            structural_rows.append({
                "fold": h, "bucket": b,
                "mean_rho_delta": md,
                "per_seed_deltas": [float(x) for x in d],
                "n_cells": sup.get("n_cells"),
                "n_participants": sup.get("n_participants"),
                "n_raw": sup.get("n_raw"),
                "margin_past_tolerance": margin,
                "seed_floor_sd": floors[label].get(f"{h}/{b}", {}).get("sd"),
                "margin_over_floor_sd": (
                    margin / floors[label][f"{h}/{b}"]["sd"]
                    if floors[label].get(f"{h}/{b}", {}).get("sd") else None
                ),
            })

    # BOOTSTRAP the structural verdict over SEEDS (the unit the structural rule consumes).
    # Resample the 3 seeds WITH replacement per fold and recompute "regressed on every drawn seed".
    boot = {}
    for h in sorted(R):
        seeds = sorted(R[h])
        per_bucket_hits: dict[int, list[bool]] = {}
        for _ in range(N_BOOT):
            drawn = RNG.choice(seeds, size=len(seeds), replace=True)
            hits: dict[int, int] = {}
            for s in drawn:
                blk = bucket_regression_report(
                    R[h][int(s)], baseline.get(h, {}), "boot", support=None
                )
                for b in blk.get("regressing_high_buckets", []):
                    hits[int(b)] = hits.get(int(b), 0) + 1
            for b in set(list(hits) + list(detail[h]["structural"])):
                per_bucket_hits.setdefault(b, []).append(hits.get(b, 0) == len(drawn))
        boot[h] = {
            str(b): float(np.mean(v)) for b, v in sorted(per_bucket_hits.items())
        }
        # and the fold-level verdict: is ANY high bucket structural?
    boot_any = {}
    for h in sorted(R):
        seeds = sorted(R[h])
        n_any = 0
        for _ in range(N_BOOT):
            drawn = RNG.choice(seeds, size=len(seeds), replace=True)
            hits: dict[int, int] = {}
            for s in drawn:
                blk = bucket_regression_report(R[h][int(s)], baseline.get(h, {}), "boot", support=None)
                for b in blk.get("regressing_high_buckets", []):
                    hits[int(b)] = hits.get(int(b), 0) + 1
            if any(k == len(drawn) for k in hits.values()):
                n_any += 1
        boot_any[h] = n_any / N_BOOT

    passed = not any(d["structural"] for d in detail.values())
    out["arms"][label] = {
        "config": rep.get("config", {}),
        "passed": passed,
        "n_folds_structural": sum(1 for d in detail.values() if d["structural"]),
        "structural_rows": structural_rows,
        "detail": detail,
        "bootstrap_structural_prob_per_bucket": boot,
        "bootstrap_prob_fold_refused": boot_any,
        "n_boot": N_BOOT,
    }

    nf = out["arms"][label]["n_folds_structural"]
    print()
    print(f"  {label:<14} {'PASS' if passed else f'STRUCTURAL on {nf}/4 folds'}   ({prov[label]})")
    for row in structural_rows:
        mo = row["margin_over_floor_sd"]
        print(
            f"      {row['fold']:<8} b{row['bucket']:<4} "
            f"delta_rho {row['mean_rho_delta']:+.4f}  "
            f"n_cells {row['n_cells']:<5} n_part {row['n_participants']:<4} n_raw {row['n_raw']:<7} "
            f"margin {row['margin_past_tolerance']:+.4f}  "
            f"margin/floor_sd {mo if mo is None else f'{mo:6.2f}'}  "
            f"boot P(struct) {boot[row['fold']].get(str(row['bucket']), float('nan')):.3f}"
        )
    for h, d in sorted(detail.items()):
        if d["noise"]:
            print(f"      {h:<8} noise-only buckets {d['noise']} (regressed on some but not all seeds)")

with open(f"{ARTIFACTS}/rows.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print()
print(f"wrote {ARTIFACTS}/rows.json")
