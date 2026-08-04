"""GATEWHY-1 §6.3/§6.4 — H-BASIS, H-MONO, H-EXTRAP, H-PACE from ALREADY-TRAINED arms.

Registered at GATEWHY-preregistration.md §6.3/§6.4 @ 821ce01 BEFORE this ran.

H-BASIS is the confound nobody controlled. Every refused arm is BOTH an interpretability frame AND a
model whose high-wpm rho level differs from CUR's by more than the tolerance. `CUR-NOWPM` separates
them: the SERVED 20-column basis with `wpm` neutralized -- no interp construction, no ordinal
recoding, no monotone constraints, and a MATCHED accuracy cost (published dwmae +6.07 vs interp's
+5.77). If it ALSO fails structurally, the refusal is not about interpretability at all.

H-MONO: INTERP-NOMONO, the same 10 columns with `monotone=False`. If it PASSES where INTERP fails,
the constraints are the cause.

⚠ EVERY ARM HERE IS BASELINED AGAINST INTERPFRAME'S OWN CUR (same-run pairing). Cross-run baselining
would compare against a different training run; only same-instrument deltas compare (INVARIANT 7's
logic). The two runs' CURs are compared explicitly so the reader can see the run-to-run spread.

Also computed: the full delta-rho-vs-bucket PROFILE (H-EXTRAP -- is the deficit monotone in wpm?) and
the margin-vs-floor table against g01's MEASURED reseed floor (INVARIANT 8), reported before any
p-value.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-gatewhy/agent-artifacts/gatewhy")
from _boot import ARTIFACTS, assert_tree, require_key  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.features import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    BIGRAM_HYBRIDB_FEATURE_NAMES,
    BIGRAM_INTERP_FEATURE_NAMES,
)
from keybo.verdicts import HIGH_WPM_FLOOR, HIGH_WPM_TOLERANCE, bucket_regression_report  # noqa: E402

IF_PATH = "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe/lolo.json"
HT_PATH = "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri/lolo.json"
IF = json.load(open(IF_PATH))
HT = json.load(open(HT_PATH))
G01 = json.load(open(f"{ARTIFACTS}/g01_controls.json"))
require_key(IF["arms"], "CUR", "INTERP", "INTERP-NOMONO", "CUR-NOWPM", where="interpframe arms")

out: dict = {
    "prereg": "agent-artifacts/gatewhy/GATEWHY-preregistration.md @ 821ce01 §6.3/§6.4",
    "sources": {"interpframe": IF_PATH, "hybridtri": HT_PATH},
    "frame_widths": {
        "served": len(BIGRAM_FEATURE_NAMES),
        "interp": len(BIGRAM_INTERP_FEATURE_NAMES),
        "hybridb": len(BIGRAM_HYBRIDB_FEATURE_NAMES),
    },
}


def rhos(rep: dict) -> dict[str, dict[int, dict[int, float]]]:
    o: dict[str, dict[int, dict[int, float]]] = {}
    for holdout, fold in rep["folds"].items():
        for rec in fold["seeds"]:
            br = rec.get("bucket_rhos") or {}
            if not br:
                raise SystemExit(f"ABORT: {holdout} s{rec.get('seed')} has no bucket_rhos")
            o.setdefault(holdout, {})[int(rec["seed"])] = {
                int(k): float(v) for k, v in br.items() if v is not None
            }
    return o


def mean_baseline(r: dict[str, dict[int, dict[int, float]]]) -> dict[str, dict[int, float]]:
    return {
        h: {
            b: float(np.mean([s[b] for s in seeds.values() if b in s]))
            for b in sorted({b for s in seeds.values() for b in s})
        }
        for h, seeds in r.items()
    }


def verdict(cand: dict, base: dict, name: str) -> dict:
    detail = {}
    for holdout, seeds in sorted(cand.items()):
        hits: dict[int, int] = {}
        deltas: dict[str, dict[str, float]] = {}
        for seed, br in sorted(seeds.items()):
            blk = bucket_regression_report(br, base.get(holdout, {}), f"{name}/{holdout}/s{seed}")
            for b in blk["regressing_high_buckets"]:
                hits[int(b)] = hits.get(int(b), 0) + 1
            deltas[str(seed)] = {str(k): float(v) for k, v in blk["deltas"].items()}
        n = len(seeds)
        detail[holdout] = {
            "n_seeds": n,
            "structural": sorted(b for b, h in hits.items() if h == n),
            "noise": sorted(b for b, h in hits.items() if 0 < h < n),
            "per_seed_deltas": deltas,
        }
    structural = {h: d["structural"] for h, d in detail.items() if d["structural"]}
    return {
        "passed": not structural,
        "structural": structural,
        "noise_only": {h: d["noise"] for h, d in detail.items() if d["noise"]},
        "detail": detail,
    }


# ---------------------------------------------------------------------------------------------
# ARM IDENTITY: verify each arm is what its LABEL claims, from its own config block.
# ---------------------------------------------------------------------------------------------
print("[identity] arm configs from interpframe/lolo.json (a label is a claim, not the referent)")
ident = {}
for arm in ("CUR", "INTERP", "INTERP-NOMONO", "CUR-NOWPM"):
    cfg = IF["arms"][arm].get("config", {})
    ident[arm] = {
        "interp": cfg.get("interp"),
        "monotone": cfg.get("monotone"),
        "seeds": cfg.get("seeds"),
        "direction": cfg.get("direction"),
        "kitchensink": cfg.get("kitchensink"),
    }
    print(
        f"  {arm:<15} interp={cfg.get('interp')!r} monotone={cfg.get('monotone')!r} seeds={cfg.get('seeds')}"
    )
out["arm_identity"] = ident

# H-MONO's arm must differ from INTERP in EXACTLY the monotone flag, or the test is not the test.
mono_ok = (
    ident["INTERP"]["interp"] == ident["INTERP-NOMONO"]["interp"]
    and ident["INTERP"]["monotone"] is True
    and ident["INTERP-NOMONO"]["monotone"] is False
)
print(f"[identity] H-MONO contrast valid (same interp flag, monotone True vs False): {mono_ok}")
out["h_mono_contrast_valid"] = bool(mono_ok)
# H-BASIS's arm must be the SERVED basis (interp False/absent).
basis_ok = not ident["CUR-NOWPM"]["interp"]
print(f"[identity] H-BASIS arm CUR-NOWPM is on the SERVED basis (interp falsy): {basis_ok}")
out["h_basis_arm_is_served_basis"] = bool(basis_ok)

# ---------------------------------------------------------------------------------------------
# Cross-run CUR agreement -- so no cross-run delta is quoted without the run-to-run spread visible.
# ---------------------------------------------------------------------------------------------
IF_CUR, HT_CUR = rhos(IF["arms"]["CUR"]), rhos(HT["arms"]["CUR"])
IF_BASE, HT_BASE = mean_baseline(IF_CUR), mean_baseline(HT_CUR)
cross = {}
print("\n[cross-run] interpframe CUR vs hybridtri CUR, per (fold,bucket) baseline mean")
for h in sorted(set(IF_BASE) & set(HT_BASE)):
    row = {}
    for b in sorted(set(IF_BASE[h]) & set(HT_BASE[h])):
        row[str(b)] = float(IF_BASE[h][b] - HT_BASE[h][b])
    cross[h] = row
    print(f"  {h:<8} " + "  ".join(f"b{k}:{v:+.4f}" for k, v in row.items()))
hi = [v for h in cross for k, v in cross[h].items() if int(k) >= HIGH_WPM_FLOOR]
out["cross_run_cur_baseline_delta"] = {
    "per_fold_bucket": cross,
    "high_bucket_absmax": float(np.max(np.abs(hi))),
    "high_bucket_absmean": float(np.mean(np.abs(hi))),
    "n_high_exceeding_tolerance": int(sum(1 for v in hi if abs(v) > HIGH_WPM_TOLERANCE)),
    "n_high": len(hi),
}
print(
    f"[cross-run] |same-frame, DIFFERENT-RUN baseline delta| at high buckets: "
    f"mean {out['cross_run_cur_baseline_delta']['high_bucket_absmean']:.4f} "
    f"max {out['cross_run_cur_baseline_delta']['high_bucket_absmax']:.4f}; "
    f"{out['cross_run_cur_baseline_delta']['n_high_exceeding_tolerance']}/{len(hi)} exceed the "
    f"{HIGH_WPM_TOLERANCE} tolerance"
)

# ---------------------------------------------------------------------------------------------
# THE ARMS, each vs interpframe's OWN CUR baseline (same-run pairing).
# ---------------------------------------------------------------------------------------------
print("\n[arms] verdicts vs interpframe's OWN CUR baseline (same run)")
arms = {}
for arm in ("CUR", "INTERP", "INTERP-NOMONO", "CUR-NOWPM"):
    arms[arm] = verdict(rhos(IF["arms"][arm]), IF_BASE, arm)
    v = "PASS" if arms[arm]["passed"] else f"STRUCTURAL {arms[arm]['structural']}"
    nf = len(arms[arm]["structural"])
    print(f"  {arm:<15} {v}   ({nf}/4 folds)   noise-only: {arms[arm]['noise_only']}")
out["arms_vs_own_run_baseline"] = arms
out["H_BASIS"] = {
    "cur_nowpm_structural_folds": len(arms["CUR-NOWPM"]["structural"]),
    "cur_nowpm_structural": arms["CUR-NOWPM"]["structural"],
    "interp_structural_folds": len(arms["INTERP"]["structural"]),
    "interp_structural": arms["INTERP"]["structural"],
    "supported_refusal_is_not_about_interpretability": bool(
        len(arms["CUR-NOWPM"]["structural"]) >= 3
    ),
}
out["H_MONO"] = {
    "nomono_structural_folds": len(arms["INTERP-NOMONO"]["structural"]),
    "nomono_structural": arms["INTERP-NOMONO"]["structural"],
    "supported_constraints_are_the_cause": bool(arms["INTERP-NOMONO"]["passed"]),
}

# ---------------------------------------------------------------------------------------------
# H-EXTRAP: the delta-rho PROFILE across buckets. Is the deficit monotone in wpm?
# ---------------------------------------------------------------------------------------------
print("\n[H-EXTRAP] mean paired delta rho by bucket (all folds x seeds), vs each run's own CUR")
prof = {}
for src, (arms_r, base) in (
    ("interpframe", (IF["arms"], IF_BASE)),
    ("hybridtri", (HT["arms"], HT_BASE)),
):
    for arm in arms_r:
        if arm == "CUR":
            continue
        R = rhos(arms_r[arm])
        by_bucket: dict[int, list[float]] = {}
        for h, seeds in R.items():
            for _s, br in seeds.items():
                for b, r in br.items():
                    if b in base.get(h, {}):
                        by_bucket.setdefault(b, []).append(r - base[h][b])
        prof[f"{src}/{arm}"] = {
            str(b): {
                "mean_delta": float(np.mean(v)),
                "n": len(v),
                "frac_negative": float(np.mean([x < 0 for x in v])),
            }
            for b, v in sorted(by_bucket.items())
        }
        line = "  ".join(
            f"b{b}:{prof[f'{src}/{arm}'][str(b)]['mean_delta']:+.4f}" for b in sorted(by_bucket)
        )
        print(f"  {src}/{arm:<14} {line}")
out["H_EXTRAP_profile"] = prof


def monotone_decreasing(rec: dict) -> bool:
    ks = sorted(int(k) for k in rec)
    vals = [rec[str(k)]["mean_delta"] for k in ks]
    return all(b <= a for a, b in zip(vals, vals[1:], strict=False))


out["H_EXTRAP_monotone_decreasing"] = {k: bool(monotone_decreasing(v)) for k, v in prof.items()}
print(f"[H-EXTRAP] monotone-decreasing in wpm? {out['H_EXTRAP_monotone_decreasing']}")

# ---------------------------------------------------------------------------------------------
# MARGIN vs the MEASURED FLOOR (INVARIANT 8), reported BEFORE any p-value.
# The floor is g01's same-frame reseed spread AT THAT BUCKET -- matched to the comparison's data
# volume, not a borrowed constant, and not gateaudit's calibration-gate 49% (a different instrument).
# ---------------------------------------------------------------------------------------------
FLOOR = G01["GC3_measured_floor"]["per_fold_bucket"]
SUP = G01["GC4_support"]["support_per_fold_bucket"]
print(
    "\n[margin] every REFUSED (fold,bucket): mean delta rho vs the MEASURED reseed floor at that bucket"
)
print(
    f"{'arm':<22}{'fold':<9}{'bkt':<6}{'mean drho':>11}{'floor(pairmax)':>16}"
    f"{'margin':>10}{'x floor':>9}{'n_cells':>9}{'n_ppts':>8}"
)
rowsout = []
SRC = {
    "INTERP (10c)": (IF["arms"]["INTERP"], IF_BASE, arms["INTERP"]),
    "INTERP-NOMONO (10c)": (IF["arms"]["INTERP-NOMONO"], IF_BASE, arms["INTERP-NOMONO"]),
    "CUR-NOWPM (20c served)": (IF["arms"]["CUR-NOWPM"], IF_BASE, arms["CUR-NOWPM"]),
}
# hybridtri arms against hybridtri's own baseline
for arm in ("HYBRIDB", "INTERP"):
    SRC[f"hybridtri {arm}"] = (
        HT["arms"][arm],
        HT_BASE,
        verdict(rhos(HT["arms"][arm]), HT_BASE, arm),
    )
for label, (rep, base, verd) in SRC.items():
    R = rhos(rep)
    for fold, bl in sorted(verd["structural"].items()):
        for b in bl:
            dl = [R[fold][s][b] - base[fold][b] for s in sorted(R[fold]) if b in R[fold][s]]
            fl = FLOOR.get(fold, {}).get(str(b), {})
            floor = fl.get("pairwise_absdiff_max")
            sup = SUP.get(fold, {}).get(str(b), {})
            md = float(np.mean(dl))
            margin = abs(md) - floor if floor else None
            rowsout.append(
                {
                    "arm": label,
                    "fold": fold,
                    "bucket": b,
                    "mean_delta_rho": md,
                    "per_seed_delta_rho": [float(x) for x in dl],
                    "measured_reseed_floor_pairmax": floor,
                    "measured_reseed_floor_sd": fl.get("sd"),
                    "margin_vs_floor": margin,
                    "multiples_of_floor": (abs(md) / floor) if floor else None,
                    "n_cells": sup.get("n_cells"),
                    "n_participants": sup.get("n_participants"),
                    "tolerance": HIGH_WPM_TOLERANCE,
                }
            )
            print(
                f"{label:<22}{fold:<9}b{b:<5}{md:>+11.4f}{floor:>16.4f}"
                f"{margin:>+10.4f}{abs(md) / floor:>9.2f}{sup.get('n_cells', 0):>9}"
                f"{sup.get('n_participants', 0):>8}"
            )
out["margin_vs_measured_floor"] = rowsout
n_below = sum(1 for r in rowsout if r["margin_vs_floor"] is not None and r["margin_vs_floor"] <= 0)
out["margin_summary"] = {
    "n_refused_rows": len(rowsout),
    "n_within_measured_reseed_floor": n_below,
    "frac_within_floor": (n_below / len(rowsout)) if rowsout else None,
}
print(
    f"\n[margin] {n_below}/{len(rowsout)} refused rows sit WITHIN the same-frame reseed floor "
    f"measured at their own bucket"
)

with open(f"{ARTIFACTS}/g03_basis.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(f"\n[done] wrote {ARTIFACTS}/g03_basis.json")
