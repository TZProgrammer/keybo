"""GATEWHY-1 §8 — THE PARTITION and the baseline-composition BOOTSTRAP.

Registered at GATEWHY-preregistration.md §8 @ 79cd8df BEFORE these numbers existed.

Every refused (fold, bucket) row is classified against the SEEDNOISE control (the served frame merely
reseeded, which the gate REFUSES on azerty b120):
  NOISE-PROVEN    SEEDNOISE is ALSO structurally refused at that same (fold, bucket)
  NOISE-SUSPECT   SEEDNOISE shows some-but-not-all-seed regression there
  SURVIVES        SEEDNOISE is clean there

Then the bootstrap the prereg registered: resample WHICH SEEDS fill the incumbent baseline (all three
2-subsets and all three singletons) and report how often each arm's per-fold structural verdict
survives. That prices the verdict's dependence on the baseline's seed composition -- the fragility
SEEDNOISE demonstrates.

⚠ A 3-seed SEEDNOISE establishes EXISTENCE (the gate CAN refuse the incumbent), not a RATE. No
false-refusal probability is quoted anywhere in this driver (prereg §8.2).
"""

from __future__ import annotations

import itertools
import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-gatewhy/agent-artifacts/gatewhy")
from _boot import ARTIFACTS, assert_tree, require_key  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.verdicts import HIGH_WPM_FLOOR, HIGH_WPM_TOLERANCE, bucket_regression_report  # noqa: E402

HT = json.load(open("/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri/lolo.json"))
IF = json.load(open("/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe/lolo.json"))
FX = json.load(open("/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe/lolo_nowpm_fixed.json"))
SN = json.load(open(f"{ARTIFACTS}/g02_seednoise.json"))
G01 = json.load(open(f"{ARTIFACTS}/g01_controls.json"))
require_key(SN, "report", "inline_verdict", where="g02_seednoise.json")

# The SEEDNOISE arm must be what it claims: served basis, NEW seeds. Verified, not trusted.
sncfg = SN["report"].get("config", {})
print(f"[identity] SEEDNOISE config: seeds={sncfg.get('seeds')} interp={sncfg.get('interp')!r} "
      f"monotone={sncfg.get('monotone')} columns={len(SN['served_columns'])}")
if list(sncfg.get("seeds") or []) != [3, 4, 5] or sncfg.get("interp") is not False:
    raise SystemExit(f"ABORT: SEEDNOISE is not the served frame at new seeds: {sncfg}")
if len(SN["served_columns"]) != 20 or SN["served_columns"][-1] != "wpm":
    raise SystemExit("ABORT: SEEDNOISE did not use the shipped 20c served frame")

out: dict = {
    "prereg": "GATEWHY-preregistration.md @ 79cd8df §8",
    "seednoise_identity": {
        "seeds": sncfg.get("seeds"),
        "interp": sncfg.get("interp"),
        "monotone": sncfg.get("monotone"),
        "n_columns": len(SN["served_columns"]),
    },
}


def rhos(rep):
    o = {}
    for holdout, fold in rep["folds"].items():
        for rec in fold["seeds"]:
            o.setdefault(holdout, {})[int(rec["seed"])] = {
                int(k): float(v) for k, v in (rec.get("bucket_rhos") or {}).items() if v is not None
            }
    return o


CUR_R = rhos(HT["arms"]["CUR"])
BASE = {
    h: {
        b: float(np.mean([s[b] for s in seeds.values() if b in s]))
        for b in sorted({b for s in seeds.values() for b in s})
    }
    for h, seeds in CUR_R.items()
}


def classify(rep, name, base=None):
    """{fold: {"structural": [...], "noise": [...]}} under the PUBLISHED rule."""
    base = base if base is not None else BASE
    R = rhos(rep)
    o = {}
    for holdout, seeds in sorted(R.items()):
        hits: dict[int, int] = {}
        for seed, br in sorted(seeds.items()):
            blk = bucket_regression_report(br, base.get(holdout, {}), f"{name}/{holdout}/s{seed}")
            for b in blk["regressing_high_buckets"]:
                hits[int(b)] = hits.get(int(b), 0) + 1
        n = len(seeds)
        o[holdout] = {
            "structural": sorted(b for b, h in hits.items() if h == n),
            "noise": sorted(b for b, h in hits.items() if 0 < h < n),
        }
    return o


SN_CLASS = classify(SN["report"], "SEEDNOISE")
print("\n[SEEDNOISE] the served frame merely RESEEDED, scored against the published CUR baseline:")
for h, d in sorted(SN_CLASS.items()):
    print(f"   {h:<8} structural {d['structural']}   noise {d['noise']}")
NOISE_PROVEN = {(h, b) for h, d in SN_CLASS.items() for b in d["structural"]}
NOISE_SUSPECT = {(h, b) for h, d in SN_CLASS.items() for b in d["noise"]}
out["seednoise_verdict"] = {
    "per_fold": SN_CLASS,
    "noise_proven_cells": sorted(f"{h}/b{b}" for h, b in NOISE_PROVEN),
    "noise_suspect_cells": sorted(f"{h}/b{b}" for h, b in NOISE_SUSPECT),
    "gate_refuses_the_reseeded_incumbent": bool(NOISE_PROVEN),
}

# ---------------------------------------------------------------------------------------------
# THE PARTITION
# ---------------------------------------------------------------------------------------------
ARMS = {
    "INTERP (10c interp.1)": IF["arms"]["INTERP"],
    "HYBRIDB (18c hybrid-B)": HT["arms"]["HYBRIDB"],
    "INTERP-NOMONO (10c, unconstrained)": IF["arms"]["INTERP-NOMONO"],
    "CUR-NOWPM (20c SERVED basis)": FX["arm"],
    "SEEDNOISE (20c SERVED, reseeded)": SN["report"],
}
FLOOR = G01["GC3_measured_floor"]["per_fold_bucket"]
SUP = G01["GC4_support"]["support_per_fold_bucket"]

print("\n[PARTITION] every refused (fold,bucket), classified against the SEEDNOISE control")
print(
    f"{'arm':<36}{'fold':<9}{'bkt':<6}{'drho':>9}{'xfloor':>8}{'n_cells':>8}{'n_ppts':>8}  class"
)
rowsout = []
per_arm = {}
for label, rep in ARMS.items():
    cls = classify(rep, label)
    R = rhos(rep)
    survives_folds, all_folds = set(), set()
    for fold, d in sorted(cls.items()):
        for b in d["structural"]:
            all_folds.add(fold)
            key = (fold, b)
            klass = (
                "NOISE-PROVEN"
                if key in NOISE_PROVEN
                else ("NOISE-SUSPECT" if key in NOISE_SUSPECT else "SURVIVES")
            )
            if klass == "SURVIVES":
                survives_folds.add(fold)
            dl = [R[fold][s][b] - BASE[fold][b] for s in sorted(R[fold]) if b in R[fold][s]]
            fl = FLOOR.get(fold, {}).get(str(b), {})
            floor = fl.get("pairwise_absdiff_max")
            sup = SUP.get(fold, {}).get(str(b), {})
            md = float(np.mean(dl))
            rowsout.append(
                {
                    "arm": label,
                    "fold": fold,
                    "bucket": b,
                    "mean_delta_rho": md,
                    "per_seed_delta_rho": [float(x) for x in dl],
                    "measured_reseed_floor_pairmax": floor,
                    "multiples_of_floor": (abs(md) / floor) if floor else None,
                    "n_cells": sup.get("n_cells"),
                    "n_participants": sup.get("n_participants"),
                    "classification": klass,
                }
            )
            print(
                f"{label:<36}{fold:<9}b{b:<5}{md:>+9.4f}"
                f"{(abs(md) / floor if floor else 0):>8.2f}{sup.get('n_cells', 0):>8}"
                f"{sup.get('n_participants', 0):>8}  {klass}"
            )
    per_arm[label] = {
        "folds_refused_published": sorted(all_folds),
        "n_folds_published": len(all_folds),
        "folds_refused_after_removing_noise_proven": sorted(survives_folds),
        "n_folds_after": len(survives_folds),
        "per_fold": cls,
    }
out["partition_rows"] = rowsout
out["per_arm"] = per_arm

print("\n[PARTITION] fold counts BEFORE vs AFTER removing the NOISE-PROVEN cells")
print(f"{'arm':<36}{'published':>11}{'after':>8}  surviving folds")
for label, d in per_arm.items():
    print(
        f"{label:<36}{d['n_folds_published']:>8}/4{d['n_folds_after']:>6}/4  "
        f"{d['folds_refused_after_removing_noise_proven']}"
    )
n_np = sum(1 for r in rowsout if r["classification"] == "NOISE-PROVEN")
n_ns = sum(1 for r in rowsout if r["classification"] == "NOISE-SUSPECT")
n_sv = sum(1 for r in rowsout if r["classification"] == "SURVIVES")
out["partition_summary"] = {
    "n_rows": len(rowsout),
    "n_noise_proven": n_np,
    "n_noise_suspect": n_ns,
    "n_survives": n_sv,
}
print(f"\n[PARTITION] rows: {n_np} NOISE-PROVEN, {n_ns} NOISE-SUSPECT, {n_sv} SURVIVES (of {len(rowsout)})")

# ---------------------------------------------------------------------------------------------
# BOOTSTRAP over the baseline's SEED COMPOSITION (prereg §8.3)
# ---------------------------------------------------------------------------------------------
print("\n[BOOTSTRAP] resampling WHICH SEEDS fill the incumbent baseline (3 pairs + 3 singletons)")
seeds_all = sorted(next(iter(CUR_R.values())).keys())
compositions = [tuple(c) for k in (2, 1) for c in itertools.combinations(seeds_all, k)]
boot: dict[str, dict[str, dict]] = {}
for label, rep in ARMS.items():
    R = rhos(rep)
    tally: dict[str, int] = {}
    for comp in compositions:
        base_c = {
            h: {
                b: float(np.mean([CUR_R[h][s][b] for s in comp if b in CUR_R[h][s]]))
                for b in sorted({b for s in CUR_R[h].values() for b in s})
            }
            for h in CUR_R
        }
        cls = classify(rep, label, base=base_c)
        for fold, d in cls.items():
            if d["structural"]:
                tally[fold] = tally.get(fold, 0) + 1
    boot[label] = {
        "n_compositions": len(compositions),
        "folds_refused_count": {f: tally.get(f, 0) for f in sorted(CUR_R)},
        "n_folds_refused_in_all_compositions": sum(
            1 for f in CUR_R if tally.get(f, 0) == len(compositions)
        ),
    }
    print(
        f"  {label:<36} refused in all {len(compositions)} compositions on "
        f"{boot[label]['n_folds_refused_in_all_compositions']}/4 folds   "
        f"{boot[label]['folds_refused_count']}"
    )
out["bootstrap_baseline_composition"] = boot

with open(f"{ARTIFACTS}/g05_partition.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(f"\n[done] wrote {ARTIFACTS}/g05_partition.json")
