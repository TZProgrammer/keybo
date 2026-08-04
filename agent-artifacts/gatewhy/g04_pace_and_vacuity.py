"""GATEWHY-1 §7.1/§7.2 + §6.1 — H-PACE by construction, and the VACUITY of the gate control.

Registered at GATEWHY-preregistration.md §7 @ 569b68d BEFORE these numbers existed.

Three things, each a demonstration rather than an assertion:

1. H-PACE. `Cell.wpm` is the BUCKET MIDPOINT, so within a bucket the LOGRAT->ms conversion multiplies
   every prediction by one positive constant -- a monotone transform, invisible to a within-bucket
   Spearman. Demonstrated two ways: (a) numerically, on the shipped arithmetic, that an arbitrary
   pace rescaling leaves a within-bucket rho EXACTLY unchanged; (b) empirically, that INTERPFRAME-1's
   CONFOUNDED and FIXED CUR-NOWPM arms have byte-identical bucket_rhos while their wmae/tau differ.

2. VACUITY. That the published gate control cannot fail: the incumbent's per-seed deltas are
   deviations from a mean computed FROM THOSE SAME SEEDS, so they sum to zero and cannot all be
   below -tolerance. Shown as an identity on the real data AND as an exhaustive search over adversarial
   inputs (INVARIANT 5: an assertion whose subject cannot vary is vacuous -- so I show the subject
   cannot vary, rather than observing one pass).

3. H-BASIS re-scored from the FIXED CUR-NOWPM file (§7.2), asserting it agrees with the confounded
   one's verdict. If they disagree the fixed one governs.
"""

from __future__ import annotations

import itertools
import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-gatewhy/agent-artifacts/gatewhy")
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from keybo.training.validate import Cell, build_cells  # noqa: E402
from keybo.verdicts import HIGH_WPM_TOLERANCE, bucket_regression_report  # noqa: E402

IFDIR = "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe"
out: dict = {"prereg": "GATEWHY-preregistration.md @ 569b68d §6.1/§7.1/§7.2"}

# =============================================================================================
# 1a. H-PACE, numerically: within a bucket the conversion is a POSITIVE CONSTANT multiplier.
# =============================================================================================
print("[H-PACE 1a] Cell.wpm is the bucket midpoint -> the ms conversion is rank-preserving IN-bucket")
# Cell.wpm's construction, asserted on real cells rather than read off the docstring.
from keybo.data.strokes import load_strokes  # noqa: E402

rows = load_strokes(
    "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv", ngram_len=2, wpm_threshold=0, min_samples=1
)
cells = build_cells(rows, wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
by_bucket: dict[int, set[float]] = {}
for c in cells:
    by_bucket.setdefault(c.bucket, set()).add(c.wpm)
distinct = {str(b): sorted(v) for b, v in sorted(by_bucket.items())}
one_wpm_per_bucket = all(len(v) == 1 for v in by_bucket.values())
print(f"           distinct Cell.wpm values per bucket: {distinct}")
print(f"           EXACTLY ONE wpm per bucket: {one_wpm_per_bucket}")

rng = np.random.default_rng(11)
pace_checks = []
for trial in range(200):
    n = int(rng.integers(6, 60))
    pred_log = rng.normal(0.0, 0.7, n)  # a LOGRAT prediction
    obs = rng.normal(140.0, 25.0, n)  # measured durations
    wpm_bucket = float(rng.choice([50.0, 70.0, 90.0, 110.0, 130.0]))
    ms_true = np.exp(pred_log) * 12000.0 / wpm_bucket
    # any OTHER pace choice -- e.g. the global constant the confounded arm used
    ms_wrong = np.exp(pred_log) * 12000.0 / 72.69299654601069
    r_true = spearmanr(ms_true, obs).statistic
    r_wrong = spearmanr(ms_wrong, obs).statistic
    pace_checks.append(abs(r_true - r_wrong))
out["H_PACE_numeric"] = {
    "distinct_cell_wpm_per_bucket": distinct,
    "exactly_one_wpm_per_bucket": bool(one_wpm_per_bucket),
    "n_trials": len(pace_checks),
    "max_abs_rho_difference_under_any_pace": float(np.max(pace_checks)),
}
print(
    f"           max |rho difference| over {len(pace_checks)} trials, true pace vs a WRONG constant "
    f"pace: {np.max(pace_checks):.3e}"
)

# =============================================================================================
# 1b. H-PACE, empirically: the confounded and fixed CUR-NOWPM arms have IDENTICAL bucket rhos.
# =============================================================================================
LJ = json.load(open(f"{IFDIR}/lolo.json"))
FX = json.load(open(f"{IFDIR}/lolo_nowpm_fixed.json"))
conf, fixed = LJ["arms"]["CUR-NOWPM"], FX["arm"]
maxd_rho, maxd_wmae, n = 0.0, 0.0, 0
for f in sorted(conf["folds"]):
    for a, b in zip(conf["folds"][f]["seeds"], fixed["folds"][f]["seeds"], strict=True):
        if a["seed"] != b["seed"]:
            raise SystemExit("ABORT: seed order differs between the two CUR-NOWPM artifacts")
        ra, rb = a.get("bucket_rhos") or {}, b.get("bucket_rhos") or {}
        if sorted(ra) != sorted(rb):
            raise SystemExit(f"ABORT: bucket sets differ at {f}/s{a['seed']}")
        for k in ra:
            maxd_rho = max(maxd_rho, abs(float(ra[k]) - float(rb[k])))
        maxd_wmae = max(maxd_wmae, abs(a["wmae"] - b["wmae"]))
        n += 1


def pooled(rep, key):
    return float(np.mean([m[key] for f in rep["folds"].values() for m in f["seeds"]]))


out["H_PACE_empirical"] = {
    "n_fold_seed_cells": n,
    "max_abs_bucket_rho_difference": maxd_rho,
    "max_abs_wmae_difference": maxd_wmae,
    "confounded_pooled_wmae": pooled(conf, "wmae"),
    "fixed_pooled_wmae": pooled(fixed, "wmae"),
    "confounded_tau": [p["tau_heldout"] for p in conf["pooled"]],
    "fixed_tau": [p["tau_heldout"] for p in fixed["pooled"]],
    "wpm_const_used": FX["wpm_const_used_for_the_column"],
}
e = out["H_PACE_empirical"]
print(
    f"[H-PACE 1b] confounded vs fixed CUR-NOWPM over {n} fold x seed cells: "
    f"max |d bucket_rho| = {maxd_rho:.3e}  BUT max |d wmae| = {maxd_wmae:.4f}"
)
print(
    f"           pooled wmae {e['confounded_pooled_wmae']:.4f} -> {e['fixed_pooled_wmae']:.4f}; "
    f"tau {e['confounded_tau'][0]:.4f} -> {e['fixed_tau'][0]:.4f}"
)

# =============================================================================================
# 2. VACUITY of the gate control -- an identity, then an exhaustive adversarial search.
# =============================================================================================
print("\n[VACUITY] the published control compares the incumbent against a mean OF ITS OWN SEEDS")
HT = json.load(open("/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri/lolo.json"))
cur = HT["arms"]["CUR"]
sum_abs, worst = [], 0.0
for holdout, fold in cur["folds"].items():
    acc: dict[int, list[float]] = {}
    for rec in fold["seeds"]:
        for b, r in (rec.get("bucket_rhos") or {}).items():
            acc.setdefault(int(b), []).append(float(r))
    base = {b: float(np.mean(v)) for b, v in acc.items()}
    for b, vals in acc.items():
        s = float(sum(v - base[b] for v in vals))
        sum_abs.append(abs(s))
        worst = max(worst, abs(s))
out["VACUITY_identity"] = {
    "n_fold_bucket_cells": len(sum_abs),
    "max_abs_sum_of_incumbent_deltas": worst,
    "note": "each incumbent bucket's 3 seed-deltas are deviations from their own mean => sum == 0",
}
print(f"          max |sum of the 3 incumbent deltas| over {len(sum_abs)} (fold,bucket) cells: {worst:.3e}")

# The impossibility, proved by exhaustive adversarial search rather than by one observation.
rng2 = np.random.default_rng(7)
found_failing = None
for trial in range(200_000):
    # Adversarial: any 3 rho values at all, including pathological spreads.
    x = rng2.uniform(-1.0, 1.0, 3)
    d = x - x.mean()
    if all(v < -HIGH_WPM_TOLERANCE for v in d):
        found_failing = [float(v) for v in x]
        break
# and via the gate's OWN function, so this tests the shipped code path, not my algebra
gate_failed = 0
N_SHIPPED_TRIALS = 3_000
for trial in range(N_SHIPPED_TRIALS):
    x = rng2.uniform(-1.0, 1.0, 3)
    base = {120: float(x.mean())}
    hits = sum(
        1
        for v in x
        if bucket_regression_report({120: float(v)}, base, "adv")["regressing_high_buckets"]
    )
    if hits == 3:
        gate_failed += 1
out["VACUITY_impossibility"] = {
    "adversarial_trials_algebra": 200_000,
    "found_3_of_3_negative_deviations": found_failing,
    "adversarial_trials_through_shipped_gate": N_SHIPPED_TRIALS,
    "n_trials_where_incumbent_failed_structurally": gate_failed,
    "loso_scaling_factor_dev_over_delta": 1.5,
}
print(
    f"          adversarial search, 200k random incumbents: a 3/3 structural self-refusal was "
    f"{'FOUND: ' + str(found_failing) if found_failing else 'NEVER FOUND (impossible)'}"
)
print(
    f"          through the SHIPPED bucket_regression_report, {N_SHIPPED_TRIALS} adversarial incumbents: "
    f"{gate_failed} failed structurally"
)

# LOSO scaling, verified exactly (this is why my own H-ASYM's fix could not work).
ratios = []
for trial in range(2000):
    x = rng2.normal(0.8, 0.05, 3)
    d = x - x.mean()
    dev = np.array([x[i] - np.delete(x, i).mean() for i in range(3)])
    ratios.extend((dev / d).tolist())
out["VACUITY_loso_ratio"] = {
    "min": float(np.min(ratios)),
    "max": float(np.max(ratios)),
    "max_abs_deviation_from_1p5": float(np.max(np.abs(np.array(ratios) - 1.5))),
}
print(
    f"          LOSO dev_i / published d_i over 2000 random incumbents: "
    f"[{np.min(ratios):.10f}, {np.max(ratios):.10f}] (exactly 1.5 => same sign, same verdict)"
)

# =============================================================================================
# 3. H-BASIS re-scored from the FIXED file (§7.2).
# =============================================================================================
print("\n[H-BASIS §7.2] re-scored from lolo_nowpm_fixed.json, and compared to the confounded verdict")


def rhos(rep):
    o = {}
    for holdout, fold in rep["folds"].items():
        for rec in fold["seeds"]:
            o.setdefault(holdout, {})[int(rec["seed"])] = {
                int(k): float(v) for k, v in (rec.get("bucket_rhos") or {}).items() if v is not None
            }
    return o


CUR_R = rhos(LJ["arms"]["CUR"])
BASE = {
    h: {
        b: float(np.mean([s[b] for s in seeds.values() if b in s]))
        for b in sorted({b for s in seeds.values() for b in s})
    }
    for h, seeds in CUR_R.items()
}


def structural_of(rep, name):
    R = rhos(rep)
    st = {}
    for holdout, seeds in sorted(R.items()):
        hits: dict[int, int] = {}
        for seed, br in sorted(seeds.items()):
            blk = bucket_regression_report(br, BASE.get(holdout, {}), f"{name}/{holdout}/s{seed}")
            for b in blk["regressing_high_buckets"]:
                hits[int(b)] = hits.get(int(b), 0) + 1
        s = sorted(b for b, h in hits.items() if h == len(seeds))
        if s:
            st[holdout] = s
    return st


st_conf = structural_of(conf, "CUR-NOWPM-CONFOUNDED")
st_fix = structural_of(fixed, "CUR-NOWPM-FIXED")
print(f"          CONFOUNDED structural: {st_conf}")
print(f"          FIXED      structural: {st_fix}")
print(f"          VERDICTS AGREE: {st_conf == st_fix}")
out["H_BASIS_provenance"] = {
    "confounded_structural": st_conf,
    "fixed_structural": st_fix,
    "verdicts_agree": bool(st_conf == st_fix),
    "fixed_folds_failing": len(st_fix),
    "fixed_arm_magnitudes": {
        "pooled_wmae": pooled(fixed, "wmae"),
        "delta_wmae_vs_CUR": FX["deltas_vs_CUR"]["wmae"]["mean_paired_delta"],
        "delta_rho_vs_CUR": FX["deltas_vs_CUR"]["rho"]["mean_paired_delta"],
        "tau_heldout": [p["tau_heldout"] for p in fixed["pooled"]],
    },
    "arm_config": fixed.get("config"),
}

with open(f"{ARTIFACTS}/g04_pace_and_vacuity.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(f"\n[done] wrote {ARTIFACTS}/g04_pace_and_vacuity.json")
