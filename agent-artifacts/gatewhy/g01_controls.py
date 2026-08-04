"""GATEWHY-1 §2 — THE CONTROLS, which run BEFORE any frame-side story (INVARIANT 2).

GC1  reproduce the PUBLISHED gate verdicts from the on-disk bucket_rhos. If this fails, my reader is
     wrong and nothing after it is valid.
GC2  the SYMMETRIC leave-one-seed-out control the published runs did NOT do: the published baseline
     is CUR's per-fold MEAN OVER ITS 3 SEEDS, and CUR's own seeds are compared against a mean that
     INCLUDES THEM, while a candidate seed faces a mean built from data it did not contribute to.
     GC2 makes both sides face the same estimator.
GC3  this gate's OWN measured floor: CUR's seed-to-seed spread per (fold, bucket), and the
     distribution of |seed_i - mean(other seeds)| -- the exact quantity GC2 thresholds.
GC4  support recovery. The published gate blocks carry `support: None` (prereg C0.2), so the
     n_cells / n_participants behind every refusal are recovered from `bucket_matrix`.

No model is trained here: every number comes from bucket_rhos / bucket_matrix already on disk, which
makes the controls cheap and exactly reproducible. `obs` in those rhos is the MEASURED duration from
bistrokes31_v1.tsv, so the gate statistic is NOT self-generated (INVARIANT 3).
"""

from __future__ import annotations

import itertools
import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-gatewhy/agent-artifacts/gatewhy")
from _boot import ARTIFACTS, assert_tree, require, require_key  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo import verdicts as VD  # noqa: E402
from keybo.verdicts import (  # noqa: E402
    HIGH_WPM_FLOOR,
    HIGH_WPM_TOLERANCE,
    bucket_regression_report,
)

require(VD, "bucket_regression_report", "HIGH_WPM_FLOOR", "HIGH_WPM_TOLERANCE")
print(f"[gate] HIGH_WPM_FLOOR={HIGH_WPM_FLOOR}  HIGH_WPM_TOLERANCE={HIGH_WPM_TOLERANCE}")

HYBRIDTRI = "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri/lolo.json"
INTERPFRAME = "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe/lolo.json"

out: dict = {
    "prereg": "agent-artifacts/gatewhy/GATEWHY-preregistration.md @ c5e1559",
    "gate_constants": {"floor": HIGH_WPM_FLOOR, "tolerance": HIGH_WPM_TOLERANCE},
    "sources": {"hybridtri_lolo": HYBRIDTRI, "interpframe_lolo": INTERPFRAME},
}

# ---------------------------------------------------------------------------------------------
# Load, asserting every key I read exists (rc=0 with all-None output is a key-not-present bug).
# ---------------------------------------------------------------------------------------------
HT = json.load(open(HYBRIDTRI))
IF = json.load(open(INTERPFRAME))
require_key(HT, "arms", "deltas", where="hybridtri/lolo.json")
require_key(HT["arms"], "CUR", "HYBRIDB", "INTERP", where="hybridtri arms")
require_key(IF["arms"], "CUR", "INTERP", "INTERP-NOMONO", "CUR-NOWPM", where="interpframe arms")


def rhos(rep: dict) -> dict[str, dict[int, dict[int, float]]]:
    """{fold: {seed: {bucket: rho}}} from a validate() report."""
    out_: dict[str, dict[int, dict[int, float]]] = {}
    for holdout, fold in rep["folds"].items():
        for rec in fold["seeds"]:
            br = rec.get("bucket_rhos") or {}
            if not br:
                raise SystemExit(f"ABORT: {holdout} seed {rec.get('seed')} has no bucket_rhos")
            out_.setdefault(holdout, {})[int(rec["seed"])] = {
                int(k): float(v) for k, v in br.items() if v is not None
            }
    return out_


def support(rep: dict) -> dict[str, dict[int, dict[str, int]]]:
    """{fold: {bucket: {n_cells, n_participants, n_raw}}} recovered from bucket_matrix (GC4).

    The published gate blocks carry `support: None` (prereg C0.2) -- the drivers passed
    `rec.get("bucket_support")` and validate() never writes that key. bucket_matrix is where the
    counts actually live. Verified identical across seeds (same test cells), asserted below.
    """
    out_: dict[str, dict[int, dict[str, int]]] = {}
    for holdout, fold in rep["folds"].items():
        per_seed = []
        for rec in fold["seeds"]:
            bm = rec.get("bucket_matrix") or {}
            per_seed.append(
                {
                    int(b): {
                        "n_cells": int(v["n"]),
                        "n_participants": int(v["n_participants"]),
                        "n_raw": int(v["n_raw"]),
                    }
                    for b, v in bm.items()
                }
            )
        if not per_seed:
            raise SystemExit(f"ABORT: {holdout} has no bucket_matrix")
        first = per_seed[0]
        for other in per_seed[1:]:
            if other != first:
                raise SystemExit(f"ABORT: {holdout} support differs across seeds -- not a constant")
        out_[holdout] = first
    return out_


# GC4 — support recovery + the explicit statement that the published blocks had none.
published_support_present = []
for arm, rep in HT["arms"].items():
    for holdout, fold in rep["folds"].items():
        for rec in fold["seeds"]:
            blk = rec.get("high_wpm_gate") or {}
            published_support_present.append(blk.get("support") is not None)
out["GC4_support"] = {
    "published_high_wpm_gate_blocks_carrying_support": int(sum(published_support_present)),
    "published_high_wpm_gate_blocks_total": len(published_support_present),
    "recovered_from": "bucket_matrix (n -> n_cells, n_participants, n_raw)",
    "support_per_fold_bucket": {
        f: {str(b): v for b, v in sorted(s.items())} for f, s in support(HT["arms"]["CUR"]).items()
    },
}
print(
    f"[GC4] published gate blocks carrying support: "
    f"{out['GC4_support']['published_high_wpm_gate_blocks_carrying_support']}"
    f"/{out['GC4_support']['published_high_wpm_gate_blocks_total']}"
)

SUP = support(HT["arms"]["CUR"])


# ---------------------------------------------------------------------------------------------
# GC1 — reproduce the PUBLISHED verdicts. Baseline = CUR per-fold mean over ALL its seeds.
# ---------------------------------------------------------------------------------------------
def verdict(cand: dict[str, dict[int, dict[int, float]]], base: dict[str, dict[int, float]]) -> dict:
    """The published gate rule: per fold, a bucket regressing on EVERY seed is STRUCTURAL."""
    detail = {}
    for holdout, seeds in cand.items():
        hits: dict[int, int] = {}
        per_seed_deltas: dict[int, dict[int, float]] = {}
        for seed, br in sorted(seeds.items()):
            blk = bucket_regression_report(br, base.get(holdout, {}), f"{holdout}/s{seed}")
            for b in blk["regressing_high_buckets"]:
                hits[int(b)] = hits.get(int(b), 0) + 1
            per_seed_deltas[seed] = {int(k): float(v) for k, v in blk["deltas"].items()}
        n = len(seeds)
        detail[holdout] = {
            "n_seeds": n,
            "structural": sorted(b for b, h in hits.items() if h == n),
            "noise": sorted(b for b, h in hits.items() if 0 < h < n),
            "seed_counts": {str(k): v for k, v in sorted(hits.items())},
            "per_seed_deltas": {str(s): d for s, d in per_seed_deltas.items()},
        }
    structural = {h: d["structural"] for h, d in detail.items() if d["structural"]}
    return {
        "passed": not structural,
        "structural": structural,
        "noise_only": {h: d["noise"] for h, d in detail.items() if d["noise"]},
        "detail": detail,
    }


CUR_R = rhos(HT["arms"]["CUR"])
BASE_PUB = {
    h: {b: float(np.mean([CUR_R[h][s][b] for s in CUR_R[h] if b in CUR_R[h][s]])) for b in buckets}
    for h, buckets in (
        (h, sorted({b for s in seeds.values() for b in s})) for h, seeds in CUR_R.items()
    )
}

print("\n[GC1] PUBLISHED baseline = CUR per-fold mean over its 3 seeds")
for h, b in sorted(BASE_PUB.items()):
    print(f"      {h:<8} " + "  ".join(f"b{k}:{v:.4f}" for k, v in sorted(b.items())))

gc1 = {}
for arm in ("CUR", "HYBRIDB", "INTERP"):
    gc1[arm] = verdict(rhos(HT["arms"][arm]), BASE_PUB)
    v = "PASS" if gc1[arm]["passed"] else f"STRUCTURAL {gc1[arm]['structural']}"
    print(f"  {arm:<10} {v}   noise-only: {gc1[arm]['noise_only']}")

EXPECT = {
    "CUR": {},
    "HYBRIDB": {"azerty": [120], "dvorak": [80, 100], "qwerty": [120], "qwertz": [100]},
    "INTERP": {"azerty": [120], "dvorak": [100, 120], "qwerty": [120], "qwertz": [100]},
}
gc1_ok = all(gc1[a]["structural"] == EXPECT[a] for a in EXPECT)
out["GC1_reproduce_published"] = {
    "baseline_per_fold": {h: {str(b): v for b, v in sorted(d.items())} for h, d in BASE_PUB.items()},
    "arms": gc1,
    "expected_structural": EXPECT,
    "reproduced": bool(gc1_ok),
}
print(f"[GC1] reproduced published verdicts EXACTLY: {gc1_ok}")
if not gc1_ok:
    print("!! GC1 FAILED -- my reader disagrees with the published artifact. Everything after is void.")

# ---------------------------------------------------------------------------------------------
# GC3 — this gate's OWN measured floor, from CUR alone (no training).
#   (a) seed-to-seed spread per (fold, bucket)
#   (b) |seed_i - mean(other seeds)| -- the LOSO deviation GC2 thresholds
#   (c) the candidate-side analogue: |seed_i - mean(2 OTHER seeds)| for a SAME-frame comparison,
#       which is the null distribution of the statistic the published gate applies to candidates.
# ---------------------------------------------------------------------------------------------
print("\n[GC3] the gate's own floor, measured on CUR (nothing but reseeding)")
gc3: dict = {"per_fold_bucket": {}, "pooled_high": {}}
loso_dev_high: list[float] = []
loso_dev_all: list[float] = []
xseed_high: list[float] = []
for h, seeds in sorted(CUR_R.items()):
    gc3["per_fold_bucket"][h] = {}
    buckets = sorted({b for s in seeds.values() for b in s})
    for b in buckets:
        vals = [seeds[s][b] for s in sorted(seeds) if b in seeds[s]]
        if len(vals) < 2:
            continue
        arr = np.array(vals)
        # LOSO deviation: each seed against the mean of the OTHERS (symmetric estimator)
        loso = [abs(arr[i] - np.mean(np.delete(arr, i))) for i in range(len(arr))]
        # cross-seed pairwise |difference|: same frame, different seed -- pure reseed noise
        pair = [abs(x - y) for x, y in itertools.combinations(vals, 2)]
        rec = {
            "n_seeds": len(vals),
            "rhos": [float(x) for x in vals],
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)),
            "range": float(arr.max() - arr.min()),
            "loso_dev_max": float(max(loso)),
            "loso_dev_mean": float(np.mean(loso)),
            "pairwise_absdiff_max": float(max(pair)),
            "pairwise_absdiff_mean": float(np.mean(pair)),
            "support": SUP.get(h, {}).get(b),
            "exceeds_tolerance_pairwise": bool(max(pair) > HIGH_WPM_TOLERANCE),
        }
        gc3["per_fold_bucket"][h][str(b)] = rec
        loso_dev_all.extend(loso)
        if b >= HIGH_WPM_FLOOR:
            loso_dev_high.extend(loso)
            xseed_high.extend(pair)
        flag = "  <-- HIGH" if b >= HIGH_WPM_FLOOR else ""
        print(
            f"      {h:<8} b{b:<4} sd {rec['sd']:.4f}  range {rec['range']:.4f}  "
            f"loso_max {rec['loso_dev_max']:.4f}  pair_max {rec['pairwise_absdiff_max']:.4f}  "
            f"n_cells {(rec['support'] or {}).get('n_cells')}{flag}"
        )

gc3["pooled_high"] = {
    "n_loso_dev": len(loso_dev_high),
    "loso_dev_mean": float(np.mean(loso_dev_high)),
    "loso_dev_p50": float(np.percentile(loso_dev_high, 50)),
    "loso_dev_p95": float(np.percentile(loso_dev_high, 95)),
    "loso_dev_max": float(np.max(loso_dev_high)),
    "crossseed_absdiff_mean": float(np.mean(xseed_high)),
    "crossseed_absdiff_p95": float(np.percentile(xseed_high, 95)),
    "crossseed_absdiff_max": float(np.max(xseed_high)),
    "tolerance": HIGH_WPM_TOLERANCE,
    "frac_crossseed_exceeding_tolerance": float(
        np.mean([x > HIGH_WPM_TOLERANCE for x in xseed_high])
    ),
}
gc3["pooled_all_buckets_loso_dev_p95"] = float(np.percentile(loso_dev_all, 95))
out["GC3_measured_floor"] = gc3
p = gc3["pooled_high"]
print(
    f"[GC3] HIGH buckets (>= {HIGH_WPM_FLOOR}): reseed |Delta rho| mean {p['crossseed_absdiff_mean']:.4f}"
    f"  p95 {p['crossseed_absdiff_p95']:.4f}  max {p['crossseed_absdiff_max']:.4f}"
)
print(
    f"[GC3] fraction of SAME-FRAME reseed pairs exceeding the {HIGH_WPM_TOLERANCE} tolerance: "
    f"{p['frac_crossseed_exceeding_tolerance']:.3f}"
)

# ---------------------------------------------------------------------------------------------
# GC2 — THE SYMMETRIC CONTROL. Both sides face the same estimator: a 2-seed CUR mean built from
# seeds the candidate seed did not contribute to.
#   * CUR seed i        vs mean(CUR's other 2 seeds)                  -- honest self-comparison
#   * candidate seed i  vs mean(CUR's other 2 seeds), for each held-out seed, and a bucket counts
#                          as regressing for that seed only if it regresses under EVERY choice of
#                          which CUR seed is held out (the conservative reading: any doubt = pass)
#     ALSO reported: the LENIENT reading (regresses under ANY choice), so the rule's sensitivity is
#     visible rather than hidden in a choice I made.
# ---------------------------------------------------------------------------------------------
print("\n[GC2] SYMMETRIC leave-one-seed-out control (both sides face a 2-seed CUR mean)")


def loso_baselines(fold: str) -> dict[int, dict[int, float]]:
    """{held_out_seed: {bucket: mean of CUR's OTHER seeds}}."""
    seeds = CUR_R[fold]
    out_: dict[int, dict[int, float]] = {}
    for held in sorted(seeds):
        others = [s for s in sorted(seeds) if s != held]
        buckets = sorted({b for s in others for b in seeds[s]})
        out_[held] = {
            b: float(np.mean([seeds[s][b] for s in others if b in seeds[s]])) for b in buckets
        }
    return out_


def symmetric_verdict(cand_r: dict[str, dict[int, dict[int, float]]], *, is_cur: bool) -> dict:
    detail = {}
    for holdout, seeds in sorted(cand_r.items()):
        bases = loso_baselines(holdout)
        n = len(seeds)
        hits_all: dict[int, int] = {}  # conservative: regresses under EVERY baseline choice
        hits_any: dict[int, int] = {}  # lenient: regresses under ANY baseline choice
        deltas_rec: dict[str, dict] = {}
        for seed, br in sorted(seeds.items()):
            if is_cur:
                # An incumbent seed is compared against the mean of the OTHER incumbent seeds --
                # the honest self-comparison the published baseline never made.
                choices = {seed: bases[seed]}
            else:
                # A candidate seed faces every same-cardinality baseline, so the verdict does not
                # depend on which incumbent seed happens to be excluded.
                choices = bases
            regress_sets = []
            per_choice = {}
            for held, base in sorted(choices.items()):
                blk = bucket_regression_report(br, base, f"{holdout}/s{seed}/held{held}")
                regress_sets.append({int(b) for b in blk["regressing_high_buckets"]})
                per_choice[str(held)] = {
                    "regressing": sorted(int(b) for b in blk["regressing_high_buckets"]),
                    "deltas": {str(k): float(v) for k, v in blk["deltas"].items()},
                }
            inter = set.intersection(*regress_sets) if regress_sets else set()
            union = set.union(*regress_sets) if regress_sets else set()
            for b in inter:
                hits_all[b] = hits_all.get(b, 0) + 1
            for b in union:
                hits_any[b] = hits_any.get(b, 0) + 1
            deltas_rec[str(seed)] = per_choice
        detail[holdout] = {
            "n_seeds": n,
            "structural_conservative": sorted(b for b, h in hits_all.items() if h == n),
            "structural_lenient": sorted(b for b, h in hits_any.items() if h == n),
            "noise_conservative": sorted(b for b, h in hits_all.items() if 0 < h < n),
            "per_seed": deltas_rec,
        }
    return {
        "passed_conservative": not any(d["structural_conservative"] for d in detail.values()),
        "passed_lenient": not any(d["structural_lenient"] for d in detail.values()),
        "structural_conservative": {
            h: d["structural_conservative"] for h, d in detail.items() if d["structural_conservative"]
        },
        "structural_lenient": {
            h: d["structural_lenient"] for h, d in detail.items() if d["structural_lenient"]
        },
        "detail": detail,
    }


gc2 = {}
for arm in ("CUR", "HYBRIDB", "INTERP"):
    gc2[arm] = symmetric_verdict(rhos(HT["arms"][arm]), is_cur=(arm == "CUR"))
    print(
        f"  {arm:<10} conservative: "
        f"{'PASS' if gc2[arm]['passed_conservative'] else gc2[arm]['structural_conservative']}"
    )
    print(
        f"  {' ':<10} lenient:      "
        f"{'PASS' if gc2[arm]['passed_lenient'] else gc2[arm]['structural_lenient']}"
    )
out["GC2_symmetric_loso"] = gc2
out["GC2_headline"] = {
    "cur_fails_structurally_conservative": not gc2["CUR"]["passed_conservative"],
    "cur_fails_structurally_lenient": not gc2["CUR"]["passed_lenient"],
    "interp_still_fails_conservative": not gc2["INTERP"]["passed_conservative"],
    "hybridb_still_fails_conservative": not gc2["HYBRIDB"]["passed_conservative"],
}

with open(f"{ARTIFACTS}/g01_controls.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(f"\n[done] wrote {ARTIFACTS}/g01_controls.json")
