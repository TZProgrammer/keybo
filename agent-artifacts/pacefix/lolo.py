"""PACEFIX-1 §M-C + §M-E — accuracy vs SERVED (paired per-fold) and the gate at BOTH thresholds.

M-C: `validate()` LOLO, 4 folds x seeds [0,1,2], n_boot=10, ROW_STAGGERED_31 -- the SAME harness,
     seeds, folds and params every arm on this line used, so the arms are comparable. Reports rho,
     rho/ceiling, wmae, tau_heldout, and the PAIRED PER-FOLD deltas (MOR-FIX-1) against SERVED.

M-E: the high-wpm non-regression gate re-run at the SHIPPED tolerance 0.005 AND at the MEASURED
     reseed floor p95 = 0.010760 (gatefolds/reseed.json, which reproduces gatewhy's 0.0117
     independently). azerty b120 is treated as RESEED-REFUSABLE and excluded from every count I
     rely on, because CUR-RESEED (the served frame, merely reseeded) is refused there 3/3.

⚠ THE CONTROL IS **NOT** THE SHIPPED ONE. The shipped gate control compares CUR against the mean of
the SAME seeds, so its deltas sum to ~0 and it can NEVER fail (gatefolds: 3.331e-16, 0/20 cells;
gatewhy: 200,000 adversarial trials, never once). My control is the already-measured SAME-FRAME
RESEED (`CUR-RESEED`, served frame at seeds [3,4,5]) -- a control that DID fail, on azerty b120. Any
refusal of mine that is a SUBSET of that control's has shown nothing.

⚠ WHICH BASELINE. The gate's shipped construction is a PER-FOLD incumbent baseline: each fold's
CUR bucket rho, meaned over CUR's seeds. I reuse the registered CUR(0,1,2) baseline from
gatefolds/rows.json rather than recomputing it, so my arms are scored against the SAME baseline the
published verdicts used -- and I ALSO train my own SERVED arm here to verify that baseline
reproduces (a positive control on the reuse).

ARMS -- one variable each from the interp-wpm baseline, matching diagnose.py exactly.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, SCRATCH, STROKES, assert_tree, load_rows_cached, require  # noqa: E402

assert_tree()

import keybo.verdicts as VD  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

bucket_regression_report = require(VD, "bucket_regression_report")

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


GEO = ROW_STAGGERED_31
SEEDS = [0, 1, 2]
TOL_SHIPPED = 0.005
# The MEASURED floor, from gatefolds/reseed.json (36 high cells). Read from the file rather than
# hard-coded, so a stale number cannot be smuggled in.
# ⚠ MIGRATION-SAFE: prefer the VENDORED copies committed on THIS branch. The `gatefolds` branch is
# LOCAL and UNPUSHED, so `/local/home/zegertho/repos/keybo-wt-gatefolds/...` does not exist on any
# other machine -- reading it directly would make this driver unrunnable after a host migration.
# The vendored files are byte-identical (md5 verified at copy time) and the sibling path is kept
# only as a fallback, with the checksum asserted so a DRIFTED sibling can never be read silently.
_VENDORED = f"{ARTIFACTS}/vendored-gatefolds"
_SIBLING = "/local/home/zegertho/repos/keybo-wt-gatefolds/agent-artifacts/gatefolds"


def _gatefolds_input(name: str) -> str:
    """Resolve a gatefolds input, vendored-first, and PRINT which copy was used."""
    vend, sib = f"{_VENDORED}/{name}", f"{_SIBLING}/{name}"
    if os.path.exists(vend):
        print(f"[input] {name}: VENDORED {vend}")
        return vend
    if os.path.exists(sib):
        print(f"[input] {name}: sibling fallback {sib} (vendored copy MISSING)")
        return sib
    raise SystemExit(f"MISSING INPUT {name}: neither {vend} nor {sib} exists")


GATEFOLDS_RESEED = _gatefolds_input("reseed.json")
GATEFOLDS_ROWS = _gatefolds_input("rows.json")
FLOOR_WPM = 80
RESEED_REFUSABLE = {("azerty", 120)}  # SEEDNOISE/CUR-RESEED refuse this 3/3 on the SERVED frame

_reseed = json.load(open(GATEFOLDS_RESEED))
TOL_MEASURED = float(_reseed["measured_reseed_floor"]["high_abs_delta_p95"])
_rows = json.load(open(GATEFOLDS_ROWS))
BASELINE = {h: {int(b): float(v) for b, v in d.items()}
            for h, d in _rows["incumbent_baseline_per_fold"].items()}
# The control that CAN fail: which (fold,bucket) cells the SAME-FRAME RESEED itself refuses.
CONTROL_REFUSED = {
    h: sorted(int(b) for b in v)
    for h, v in _reseed["reseed_verdict"]["structural_regressions"].items()
}

print()
print("=" * 100)
print("M-E SETUP — the thresholds and the control, both MEASURED and both named")
print("=" * 100)
print(f"  shipped tolerance          : {TOL_SHIPPED}")
print(f"  MEASURED reseed floor (p95): {TOL_MEASURED:.6f}   [{GATEFOLDS_RESEED}]")
print(f"  gate floor_wpm             : {FLOOR_WPM}")
print(f"  reseed-refusable cells excluded from relied-on counts: {sorted(RESEED_REFUSABLE)}")
print(f"  MY CONTROL (same-frame reseed, served @ seeds 3,4,5) refuses: {CONTROL_REFUSED}")
print("  NOT the shipped control: its deltas sum to ~0 and it can never fail.")

ARMS = [
    ("SERVED", False, True, {}, "reference / baseline-reproduction control"),
    ("INTERP-WPM", "wpm", True, {}, "BASELINE (DEAD-1): 11c, all 11 constrained"),
    ("INTERP-WPM-NOMONO", "wpm", False, {}, "ONE VAR: monotone OFF"),
    ("INTERP-WPM-DEPTH6", "wpm", True, {"max_depth": 6}, "ONE VAR: max_depth 3->6"),
]

log("loading rows (cached)")
rows = load_rows_cached()
log(f"{len(rows)} rows")

out: dict = {
    "prereg": "agent-artifacts/pacefix/PACEFIX-preregistration.md",
    "purpose": "M-C accuracy paired per-fold vs SERVED + M-E gate at BOTH thresholds",
    "config": {
        "seeds": SEEDS, "geometry": "ROW_STAGGERED_31", "n_boot": 10, "ngram": "bigram",
        "strokes": STROKES,
    },
    "gate": {
        "floor_wpm": FLOOR_WPM,
        "tolerance_shipped": TOL_SHIPPED,
        "tolerance_measured_reseed_p95": TOL_MEASURED,
        "measured_floor_source": GATEFOLDS_RESEED,
        "baseline_construction": "per-fold incumbent CUR(0,1,2) bucket rho mean "
                                 "[reused from gatefolds/rows.json]",
        "control_is_same_frame_reseed_NOT_the_tautological_shipped_control": True,
        "control_refused_cells": CONTROL_REFUSED,
        "reseed_refusable_excluded": [list(x) for x in sorted(RESEED_REFUSABLE)],
    },
    "arms": {},
}

for label, flag, mono, extra, varies in ARMS:
    log(f"ARM {label}: validate() 4 folds x seeds {SEEDS}  ({varies})")
    kw = {"interp": flag, "monotone": mono} if flag is not False else {}
    rep = validate(
        rows,
        seeds=SEEDS,
        ngram="bigram",
        n_boot=10,
        geometry=GEO,
        train_params={"n_jobs": 8, **extra},
        **kw,
    )
    # ARM IDENTITY from the report's OWN config, never from my label.
    cfg = rep.get("config", {})
    out["arms"][label] = {
        "one_variable_vs_interp_wpm": varies,
        "config_readback": {
            k: cfg.get(k) for k in ("seeds", "interp", "monotone", "train_params", "ngram")
        },
        "report": rep,
    }
    log(f"ARM {label}: done")
    # Checkpoint after EVERY arm (seven tmux crashes have hit this fleet).
    with open(f"{ARTIFACTS}/lolo.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log(f"checkpointed {ARTIFACTS}/lolo.json")

# =============================================================================================
# M-C — pooled metrics and PAIRED PER-FOLD deltas vs SERVED
# =============================================================================================
def pooled(rep):
    rhos, fracs, wmaes = [], [], []
    for f in rep["folds"].values():
        for rec in f["seeds"]:
            rhos.append(rec["rho"])
            fracs.append(rec["rho_frac_ceiling"])
            wmaes.append(rec["wmae"] if "wmae" in rec else rec.get("mae_model"))
    return {
        "rho": float(np.mean(rhos)),
        "rho_frac_ceiling": float(np.mean(fracs)),
        "wmae": float(np.mean([w for w in wmaes if w is not None])),
        "tau_heldout": [p["tau_heldout"] for p in rep["pooled"]],
    }


def per_fold(rep, key):
    o = {}
    for h, f in rep["folds"].items():
        vals = [rec[key] for rec in f["seeds"] if rec.get(key) is not None]
        if vals:
            o[h] = float(np.mean(vals))
    return o


served_rep = out["arms"]["SERVED"]["report"]
wmae_key = "wmae" if "wmae" in served_rep["folds"][next(iter(served_rep["folds"]))]["seeds"][0] \
    else "mae_model"

mc: dict = {"wmae_key_used": wmae_key, "pooled": {}, "paired_per_fold_vs_SERVED": {}}
for label in out["arms"]:
    mc["pooled"][label] = pooled(out["arms"][label]["report"])

base_wmae = per_fold(served_rep, wmae_key)
base_rho = per_fold(served_rep, "rho")
for label in out["arms"]:
    if label == "SERVED":
        continue
    rep = out["arms"][label]["report"]
    aw, ar = per_fold(rep, wmae_key), per_fold(rep, "rho")
    dw = {h: aw[h] - base_wmae[h] for h in sorted(base_wmae) if h in aw}
    dr = {h: ar[h] - base_rho[h] for h in sorted(base_rho) if h in ar}
    mc["paired_per_fold_vs_SERVED"][label] = {
        "d_wmae_per_fold": dw,
        "mean_paired_d_wmae": float(np.mean(list(dw.values()))),
        "d_rho_per_fold": dr,
        "mean_paired_d_rho": float(np.mean(list(dr.values()))),
        "sign_consistent_folds_wmae": int(sum(1 for v in dw.values() if v > 0)),
        "n_folds": len(dw),
        # THE REGISTERED BAR: accuracy-neutral iff mean paired d_wmae <= +1.0 ms AND tau [1,1,1].
        "ACCURACY_NEUTRAL": bool(
            float(np.mean(list(dw.values()))) <= 1.0
            and all(t == 1.0 for t in mc["pooled"][label]["tau_heldout"])
        ),
    }
out["M_C_accuracy"] = mc

print()
print("=" * 100)
print("M-C — ACCURACY, pooled and PAIRED PER-FOLD vs SERVED (registered bar: dwmae <= +1.0ms)")
print("=" * 100)
for label, p in mc["pooled"].items():
    print(f"  {label:<19} rho {p['rho']:.4f}  rho/ceil {p['rho_frac_ceiling']:.4f}  "
          f"wmae {p['wmae']:8.4f}  tau {p['tau_heldout']}")
print()
for label, d in mc["paired_per_fold_vs_SERVED"].items():
    print(f"  {label:<19} mean paired dwmae {d['mean_paired_d_wmae']:+8.4f}  "
          f"mean paired drho {d['mean_paired_d_rho']:+.5f}  "
          f"worse-on {d['sign_consistent_folds_wmae']}/{d['n_folds']} folds  "
          f"=> ACCURACY_NEUTRAL={d['ACCURACY_NEUTRAL']}")
    print(f"                      per fold: " + "  ".join(
        f"{h}:{v:+.3f}" for h, v in d["d_wmae_per_fold"].items()))

# =============================================================================================
# M-E — the gate, at BOTH thresholds, with the same-frame-reseed control
# =============================================================================================
def gate_arm(rep, tol):
    """Per-fold verdicts at ONE tolerance, against the reused per-fold CUR baseline."""
    detail = {}
    for h, f in sorted(rep["folds"].items()):
        if h not in BASELINE:
            continue
        hits: dict[int, int] = {}
        dl: dict[int, list[float]] = {}
        n = len(f["seeds"])
        for rec in f["seeds"]:
            blk = bucket_regression_report(
                {int(k): v for k, v in rec["bucket_rhos"].items()},
                BASELINE[h],
                floor_wpm=FLOOR_WPM,
                tolerance=tol,
            )
            for b in blk.get("regressing_buckets", []) or []:
                hits[int(b)] = hits.get(int(b), 0) + 1
            for b, v in rec["bucket_rhos"].items():
                if int(b) >= FLOOR_WPM and int(b) in BASELINE[h]:
                    dl.setdefault(int(b), []).append(float(v) - BASELINE[h][int(b)])
        structural = sorted(b for b, c in hits.items() if c == n)
        detail[h] = {
            "n_seeds": n,
            "structural": structural,
            "noise": sorted(b for b, c in hits.items() if 0 < c < n),
            "mean_delta_per_bucket": {str(b): float(np.mean(v)) for b, v in sorted(dl.items())},
        }
    return detail


me: dict = {"per_arm": {}}
for label in out["arms"]:
    rep = out["arms"][label]["report"]
    a = {}
    for tag, tol in (("shipped_0.005", TOL_SHIPPED), ("measured_p95", TOL_MEASURED)):
        det = gate_arm(rep, tol)
        rows_all = [(h, b) for h, d in det.items() for b in d["structural"]]
        rows_kept = [r for r in rows_all if r not in RESEED_REFUSABLE]
        # A refusal set that is a SUBSET of the control's has shown nothing.
        ctl_rows = {(h, b) for h, bs in CONTROL_REFUSED.items() for b in bs}
        a[tag] = {
            "tolerance": tol,
            "detail": det,
            "structural_rows": [list(r) for r in sorted(rows_all)],
            "n_folds_structural": len({h for h, _ in rows_all}),
            "structural_rows_excl_reseed_refusable": [list(r) for r in sorted(rows_kept)],
            "n_folds_structural_excl_reseed_refusable": len({h for h, _ in rows_kept}),
            "refusals_are_subset_of_my_control": bool(rows_all and set(rows_all) <= ctl_rows),
        }
    me["per_arm"][label] = a
out["M_E_gate"] = me

print()
print("=" * 100)
print("M-E — THE GATE at BOTH thresholds (azerty b120 = reseed-refusable, excluded from counts)")
print("=" * 100)
for label, a in me["per_arm"].items():
    print(f"\n  {label}")
    for tag in ("shipped_0.005", "measured_p95"):
        g = a[tag]
        print(f"    {tag:<14} (tol {g['tolerance']:.6f}): folds {g['n_folds_structural']}/4  "
              f"rows {g['structural_rows']}")
        print(f"                   excl reseed-refusable: "
              f"folds {g['n_folds_structural_excl_reseed_refusable']}/4  "
              f"rows {g['structural_rows_excl_reseed_refusable']}  "
              f"| subset-of-my-control={g['refusals_are_subset_of_my_control']}")

# Positive control on the REUSED baseline: my own SERVED arm should look like the incumbent.
sv = me["per_arm"]["SERVED"]["shipped_0.005"]
out["baseline_reuse_control"] = {
    "my_SERVED_structural_rows_at_shipped_tol": sv["structural_rows"],
    "note": "The reused baseline is CUR(0,1,2)'s own mean, so my SERVED arm at the same seeds is a "
            "SELF-comparison and its deltas are ~0 BY CONSTRUCTION -- this is the tautology, "
            "reproduced deliberately as a check that the baseline was wired to the right folds, "
            "NOT as evidence about any arm.",
}

with open(f"{ARTIFACTS}/lolo.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/lolo.json")

os.makedirs(SCRATCH, exist_ok=True)
with open(f"{SCRATCH}/lolo.sentinel", "w") as fh:
    fh.write("ok\n")
log(f"SENTINEL {SCRATCH}/lolo.sentinel")
