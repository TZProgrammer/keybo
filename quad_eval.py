"""QUADGRAM-1: head-to-head quadgram vs trigram on held-out cross-layout transfer.

Three arms, all through the SHIPPED validate() LOLO harness (4 folds azerty/dvorak/qwerty/
qwertz x seeds 0/1/2, bucket-centered rho vs split-half ceiling, CAND4 params, LOGRAT,
ROW_STAGGERED_31 — the production trigram recipe):

  A. QUAD-FULL   — the full 4-key quadgram frame (72 cols) on the quadgram cells.
  B. QUAD-TRICTX — the trigram-context sub-frame (46 cols; drops the leading key's tg1_/bg1_
                   blocks) on the SAME quadgram cells/target. Bit-identical to a trigram model
                   on the last 3 keys, verified. This is the MATCHED control.
  C. TRI-INCUMBENT — the shipped trigram frame on the trigram cells (tristrokes31_cond_v1).
                   The standing baseline, on a DIFFERENT frame (caveat stated).

The decisive comparison is A vs B: identical rows, cells, target, harness, seeds — the ONLY
difference is whether the model may see the fourth (leading) context key. Paired per-fold
deltas (MOR-FIX-1). Mandatory high-wpm gate on A (require_no_high_wpm_regression_in_report).

Usage: python quad_eval.py            # full run, writes /tmp/quad_eval_result.json
"""

import json
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/quadgram-wt/src")

from keybo.data.strokes import load_strokes
from keybo.geometry import ROW_STAGGERED_31
from keybo.training.validate import require_no_high_wpm_regression_in_report, validate

QUAD_TSV = "/tmp/quadstrokes31_cond_v1.tsv"
TRI_TSV = "/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv"
CKPT_DIR = "/tmp/quad_eval_ckpt"  # per-arm checkpoints so an OOM-kill only loses the live arm
SEEDS = [0, 1, 2]
# n_jobs=8: the shared host is memory-pressured (a big KaenaCompiler build runs alongside); a
# leaner thread count is a better citizen and the run is not the bottleneck.
N_JOBS = 8
CEILING_N_BOOT = 30  # split-half ceiling replicate count (cheap; not the CI)
CAND4 = dict(
    n_estimators=427,
    max_depth=5,
    learning_rate=0.10903767015375725,
    min_child_weight=6,
    subsample=0.6086566147198375,
    colsample_bytree=0.9893815206317236,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    n_jobs=N_JOBS,
)
FOLDS = ["azerty", "dvorak", "qwerty", "qwertz"]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


def per_fold_seed_rho(report):
    """{holdout: {seed: rho}} and {holdout: {seed: bucket_rhos}} from a validate() report."""
    rho = {}
    buckets = {}
    for holdout, fold in report["folds"].items():
        rho[holdout] = {}
        buckets[holdout] = {}
        for rec in fold["seeds"]:
            rho[holdout][rec["seed"]] = rec["rho"]
            buckets[holdout][rec["seed"]] = {int(k): v for k, v in rec["bucket_rhos"].items()}
    return rho, buckets


def paired_delta_summary(rho_a, rho_b, label_a, label_b):
    """Paired per-(fold,seed) deltas A - B (MOR-FIX-1: never a mean of ratios).

    Reports: per-fold mean delta, per-fold sign consistency across seeds, overall W/L,
    and how many of the 4 folds are sign-consistent winners for A.
    """
    per_fold = {}
    all_deltas = []
    folds_sign_consistent_win = 0
    for holdout in sorted(rho_a):
        seed_deltas = []
        for seed in sorted(rho_a[holdout]):
            da = rho_a[holdout][seed]
            db = rho_b[holdout].get(seed)
            if da is None or db is None or not np.isfinite(da) or not np.isfinite(db):
                continue
            seed_deltas.append(da - db)
        if not seed_deltas:
            continue
        all_deltas.extend(seed_deltas)
        signs = [np.sign(d) for d in seed_deltas]
        consistent = all(s > 0 for s in signs) or all(s < 0 for s in signs)
        if consistent and seed_deltas[0] > 0:
            folds_sign_consistent_win += 1
        per_fold[holdout] = {
            "seed_deltas": [round(float(d), 6) for d in seed_deltas],
            "mean_delta": round(float(np.mean(seed_deltas)), 6),
            "sign_consistent": bool(consistent),
            "direction": "A>B" if np.mean(seed_deltas) > 0 else "A<B",
        }
    wins = sum(1 for d in all_deltas if d > 0)
    losses = sum(1 for d in all_deltas if d < 0)
    return {
        "label": f"{label_a} - {label_b}",
        "per_fold": per_fold,
        "overall_W": wins,
        "overall_L": losses,
        "mean_paired_delta": round(float(np.mean(all_deltas)), 6) if all_deltas else None,
        "n_folds_sign_consistent_win_for_A": folds_sign_consistent_win,
        "n_folds": len(per_fold),
    }


def mean_frac_ceiling(report):
    fr = [
        m["rho_frac_ceiling"]
        for fold in report["folds"].values()
        for m in fold["seeds"]
        if m["rho_frac_ceiling"] is not None
    ]
    return round(float(np.mean(fr)), 4) if fr else None


def mean_metric(report, key):
    vals = [m[key] for fold in report["folds"].values() for m in fold["seeds"] if key in m]
    return round(float(np.mean(vals)), 4) if vals else None


def pooled_tau(report):
    return [round(float(p["tau_heldout"]), 4) for p in report["pooled"]]


def _run_arm(name, tsv, ngram_len, ngram, quad_context, rows_cache):
    """Run one validate() arm with per-arm disk checkpoint (resume-safe against OOM kills).

    Returns the validate() report dict. If a checkpoint exists it is loaded and the arm is
    skipped — so a killed run resumes at the first unfinished arm rather than from scratch.
    """
    import os

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = f"{CKPT_DIR}/{name}.json"
    if os.path.exists(ckpt):
        log(f"{name}: checkpoint found, loading (skip compute)")
        return json.load(open(ckpt))
    if tsv not in rows_cache:
        log(f"loading rows from {tsv}")
        rows_cache[tsv] = load_strokes(tsv, ngram_len=ngram_len, wpm_threshold=0, min_samples=1)
        log(f"  {len(rows_cache[tsv])} rows; layouts {sorted({r.layout for r in rows_cache[tsv]})}")
    rows = rows_cache[tsv]
    log(f"{name}: running validate (ngram={ngram}, quad_context={quad_context}, bootstrap_ci=False)")
    kw = dict(
        seeds=SEEDS,
        ngram=ngram,
        holdouts=FOLDS,
        n_boot=CEILING_N_BOOT,
        geometry=ROW_STAGGERED_31,
        train_params=CAND4,
        bootstrap_ci=False,
    )
    if ngram == "quadgram":
        kw["quad_context"] = quad_context
    rep = validate(rows, **kw)
    json.dump(rep, open(ckpt, "w"))
    log(f"  {name} mean rho/ceiling = {mean_frac_ceiling(rep)}, pooled tau {pooled_tau(rep)}")
    return rep


def main():
    result = {"config": {"seeds": SEEDS, "cand4": CAND4, "folds": FOLDS, "bootstrap_ci": False}}
    rows_cache = {}

    # The gate reference for "does the 4th key regress high-wpm accuracy" is the MATCHED
    # trigram-context arm (B) on the SAME cells; we gate A against B's own (fold,seed) bucket
    # rhos below. Arms are checkpointed so an OOM-kill resumes at the first unfinished arm.

    # --- Arm B: trigram-context on quad cells (matched control) ---------------------------
    rep_b = _run_arm("arm_B_quad_trictx", QUAD_TSV, 4, "quadgram", False, rows_cache)
    rho_b, buckets_b = per_fold_seed_rho(rep_b)

    # --- Arm A: full quadgram frame on quad cells -----------------------------------------
    rep_a = _run_arm("arm_A_quad_full", QUAD_TSV, 4, "quadgram", True, rows_cache)
    rho_a, buckets_a = per_fold_seed_rho(rep_a)

    # --- high-wpm gate: arm A vs arm B, per (fold,seed), bucket >= 80 ---------------------
    from keybo.verdicts import HIGH_WPM_FLOOR, HIGH_WPM_TOLERANCE

    hw = {"floor": HIGH_WPM_FLOOR, "tolerance": HIGH_WPM_TOLERANCE, "per_fold": {}}
    structural = []
    for holdout in FOLDS:
        counts = {}
        n = 0
        for seed in SEEDS:
            ba = buckets_a.get(holdout, {}).get(seed, {})
            bb = buckets_b.get(holdout, {}).get(seed, {})
            if not ba or not bb:
                continue
            n += 1
            for bucket in sorted(bb):
                if bucket < HIGH_WPM_FLOOR or bucket not in ba:
                    continue
                if ba[bucket] - bb[bucket] < -HIGH_WPM_TOLERANCE:
                    counts[bucket] = counts.get(bucket, 0) + 1
        struct = sorted(b for b, h in counts.items() if h == n and n > 0)
        hw["per_fold"][holdout] = {
            "n_seeds": n,
            "regressing_bucket_seed_counts": {str(k): v for k, v in sorted(counts.items())},
            "structural_buckets": struct,
            "noise_buckets": sorted(b for b, h in counts.items() if 0 < h < n),
        }
        if struct:
            structural.append(f"{holdout} buckets {struct} on {n}/{n} seeds")
    hw["structural_regressions"] = structural
    hw["high_wpm_pass"] = not structural
    log(f"  high-wpm gate (A vs B, floor 80): {'PASS' if not structural else 'FAIL ' + str(structural)}")

    # --- paired deltas A vs B (the decisive comparison) -----------------------------------
    ab = paired_delta_summary(rho_a, rho_b, "QUAD-FULL", "QUAD-TRICTX")
    log(
        f"  A vs B paired: mean delta {ab['mean_paired_delta']}, W/L {ab['overall_W']}/{ab['overall_L']}, "
        f"sign-consistent winning folds {ab['n_folds_sign_consistent_win_for_A']}/4"
    )

    # --- Arm C: incumbent trigram on trigram cells (standing baseline, different frame) ---
    # Free the quad rows before loading the trigram table (memory-pressured host).
    rows_cache.pop(QUAD_TSV, None)
    rep_c = _run_arm("arm_C_tri_incumbent", TRI_TSV, 3, "trigram", None, rows_cache)

    # --- assemble ------------------------------------------------------------------------
    def arm_summary(rep):
        return {
            "mean_rho_frac_ceiling": mean_frac_ceiling(rep),
            "mean_wmae": mean_metric(rep, "wmae"),
            "mean_umae": mean_metric(rep, "umae"),
            "mean_rho": mean_metric(rep, "rho"),
            "pooled_tau_heldout": pooled_tau(rep),
            "ceilings": {k: round(float(v), 4) for k, v in rep["ceilings"].items()},
            "per_fold_seed_rho": {
                h: {str(s): round(float(r), 4) if r is not None and np.isfinite(r) else None
                    for s, r in sd.items()}
                for h, sd in per_fold_seed_rho(rep)[0].items()
            },
        }

    result["arm_A_quad_full"] = arm_summary(rep_a)
    result["arm_B_quad_trictx"] = arm_summary(rep_b)
    result["arm_C_tri_incumbent"] = arm_summary(rep_c)
    result["A_vs_B_paired"] = ab
    result["high_wpm_gate"] = hw
    # A vs C is cross-frame; report the raw fit levels but flag the caveat.
    result["A_vs_C_note"] = (
        "cross-frame (different cells/target): A predicts the c->d interval of 4-key windows; "
        "C predicts the b->c interval of 3-key windows. Compare rho/ceiling and tau LEVELS only, "
        "not paired deltas. The decisive matched comparison is A vs B."
    )

    with open("/tmp/quad_eval_result.json", "w") as f:
        json.dump(result, f, indent=2)
    log("wrote /tmp/quad_eval_result.json")
    log("ALL-DONE")


if __name__ == "__main__":
    main()
