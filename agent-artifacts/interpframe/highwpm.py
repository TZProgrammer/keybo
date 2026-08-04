"""INTERPFRAME-1 §6 — the HIGH-WPM non-regression gate, run properly.

The first pass called ``require_no_high_wpm_regression_in_report`` on reports built WITHOUT
``baseline_buckets``, so it refused all four arms with "carries no high-wpm verdict" — which is the
function working as designed ("'Not measured' is not 'did not regress'"), not four failures. The
gate needs an INCUMBENT's per-bucket rhos to compare against.

Here the incumbent is CUR's own per-bucket rhos from the completed run, pooled across its folds and
seeds. That makes the comparison self-consistent: every arm is judged against the served frame
measured on the SAME data volume, the SAME folds and the SAME cell construction — not against a
borrowed constant.

⚠ STRUCTURAL vs NOISE is the distinction that decides an arm (the function's own rule): a bucket
that regresses on EVERY seed of a fold is structural and RAISES; one that regresses on some seeds
is seed wobble and is reported without vetoing. SRROLL-1's precedent is that a STRUCTURAL high-wpm
regression is "worse than a plain null", so this is the one place §6 registered a refusal.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

from keybo.verdicts import bucket_regression_report  # noqa: E402

lolo = json.load(open(f"{ARTIFACTS}/lolo.json"))
ARMS = list(lolo["arms"])
print(f"[hw] arms: {ARMS}")

# --- the INCUMBENT baseline: CUR's per-bucket rhos, PER FOLD --------------------------------
# ⚠ MY FIRST BASELINE WAS WRONG AND THE GATE CAUGHT IT. Pooling CUR's bucket rhos across folds
# into ONE baseline made the gate refuse CUR ITSELF -- structural regressions on dvorak
# [80,100,120] and qwerty [80]. A gate that refuses the incumbent is measuring per-fold
# HETEROGENEITY, not candidate quality: dvorak's absolute rho is ~0.70 while qwertz's is ~0.92, so
# every dvorak bucket sits below a cross-fold average by construction, on the incumbent and on any
# candidate alike. This is exactly the failure mode `bucket_regression_report`'s own docstring
# names ("A gate that refuses the incumbent is measuring instability, not the candidate").
# THE FIX: compare each fold against ITS OWN incumbent rhos, which is what "non-regression" means.
cur = lolo["arms"]["CUR"]
baseline_per_fold: dict[str, dict[int, float]] = {}
for holdout, fold in cur["folds"].items():
    acc: dict[int, list[float]] = {}
    for rec in fold["seeds"]:
        for bucket, rho in (rec.get("bucket_rhos") or {}).items():
            if rho is not None:
                acc.setdefault(int(bucket), []).append(float(rho))
    baseline_per_fold[holdout] = {b: float(np.mean(v)) for b, v in sorted(acc.items())}
print("[hw] incumbent (CUR) per-bucket rho, PER FOLD (mean over its 3 seeds):")
for holdout, b in sorted(baseline_per_fold.items()):
    print(f"      {holdout:<8} " + "  ".join(f"b{k}:{v:.4f}" for k, v in b.items()))
if not baseline_per_fold:
    raise SystemExit("no bucket_rhos in the CUR report -- cannot form an incumbent baseline")

out: dict = {"incumbent_baseline_bucket_rhos_per_fold": baseline_per_fold, "arms": {}}

print()
print("[hw] per-arm verdict (STRUCTURAL = regresses on every seed of a fold => refusal)")
for name in ARMS:
    rep = lolo["arms"][name]
    counts: dict[str, dict[int, int]] = {}
    detail: dict[str, list] = {}
    for holdout, fold in rep["folds"].items():
        n_seeds = len(fold["seeds"])
        hits: dict[int, int] = {}
        for rec in fold["seeds"]:
            block = bucket_regression_report(
                {int(k): v for k, v in (rec.get("bucket_rhos") or {}).items()},
                baseline_per_fold.get(holdout, {}),
                f"{name}/{holdout}/seed{rec['seed']}",
                support=rec.get("bucket_support"),
            )
            for bucket in block.get("regressing_high_buckets", []):
                hits[int(bucket)] = hits.get(int(bucket), 0) + 1
        counts[holdout] = hits
        detail[holdout] = {
            "n_seeds": n_seeds,
            "structural": sorted(b for b, h in hits.items() if h == n_seeds),
            "noise": sorted(b for b, h in hits.items() if 0 < h < n_seeds),
            "per_bucket_seed_counts": {str(k): v for k, v in sorted(hits.items())},
        }
    structural = {h: d["structural"] for h, d in detail.items() if d["structural"]}
    out["arms"][name] = {
        "passed": not structural,
        "structural_regressions": structural,
        "detail": detail,
    }
    verdict = "PASS" if not structural else f"STRUCTURAL REGRESSION {structural}"
    noise = {h: d["noise"] for h, d in detail.items() if d["noise"]}
    print(f"  {name:<16} {verdict}" + (f"   (noise-only: {noise})" if noise else ""))

# --- THE GATE'S OWN CONTROL: it must PASS the incumbent it is built from --------------------
# A gate that refuses CUR against CUR's own per-fold rhos would be measuring seed noise, and every
# candidate verdict from it would be uninterpretable. Checked explicitly rather than assumed.
if not out["arms"]["CUR"]["passed"]:
    print()
    print("!! GATE CONTROL FAILED: the gate refuses the INCUMBENT against its own per-fold rhos.")
    print("!! Any candidate verdict from it is measuring seed noise, not the candidate.")
out["gate_control_incumbent_passes"] = bool(out["arms"]["CUR"]["passed"])

with open(f"{ARTIFACTS}/highwpm.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print()
print(f"[hw] wrote {ARTIFACTS}/highwpm.json")
