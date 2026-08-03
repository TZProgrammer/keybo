"""INVARIANT 5 end-to-end: does the PROPOSAL actually behave through `validate()`?

Three runs on the same LOLO data, one seed (the shape is what is under test, not the numbers, which
g01 already established on 3 seeds):

  1. band=None (the new DEFAULT)   -> every slope reported, gated False, passed None, band None
  2. band=(0.90,1.10) + deciding=RECOMMENDED -> qwerty fails, other three pass
  3. band=(0.90,1.10) + deciding=None (every slice, the branch's behaviour) -> all four fail

If (1) does not report the slopes, the change has thrown away the measurement. If (2) does not fail
qwerty, the change has defanged the gate. If (3) does not reproduce the branch, the comparison in the
report is not apples-to-apples.
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

WT = "/local/home/zegertho/repos/keybo-wt-gateaudit"
OUT = "/tmp/gateaudit/run/g07_e2e_proposal.json"
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"

import keybo  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.training.validate import validate  # noqa: E402
from keybo.verdicts import CALIBRATION_DECIDING_SLICES_RECOMMENDED  # noqa: E402

assert keybo.__file__.startswith(WT), f"WRONG TREE: {keybo.__file__}"
print("PROVENANCE keybo.__file__ =", keybo.__file__, flush=True)
print("DECIDING RECOMMENDED =", CALIBRATION_DECIDING_SLICES_RECOMMENDED, flush=True)

rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
print(f"loaded {len(rows)} rows", flush=True)

ARMS = {
    "1_no_band_default": {"calibration_band": None,
                          "calibration_deciding": CALIBRATION_DECIDING_SLICES_RECOMMENDED},
    "2_band_scoped_recommended": {"calibration_band": (0.90, 1.10),
                                  "calibration_deciding": CALIBRATION_DECIDING_SLICES_RECOMMENDED},
    "3_band_every_slice_as_branch": {"calibration_band": (0.90, 1.10),
                                     "calibration_deciding": None},
}

out = {"provenance": {"keybo__file__": keybo.__file__}, "n_rows": len(rows), "arms": {}}
for name, kw in ARMS.items():
    rep = validate(rows, seeds=[0], ngram="bigram", progress=False, **kw)
    arm = {}
    for layout, fold in rep["folds"].items():
        g = fold["seeds"][0]["calibration_gate"]
        arm[layout] = {
            "gated": g["gated"],
            "passed": g["passed"],
            "band": g["band"],
            "deciding_slices": g["deciding_slices"],
            "n_slices_reported": g["n_slices"],
            "slopes": g["slopes"],
            "out_of_band": g["out_of_band"],
            "out_of_band_deciding": g["out_of_band_deciding"],
            "out_of_band_advisory": g["out_of_band_advisory"],
            "support_keys": sorted((g.get("support") or {}).keys()),
        }
    out["arms"][name] = arm
    passed = {L: v["passed"] for L, v in arm.items()}
    print(f"{name:32s} -> {passed}", flush=True)

# ---- the assertions this run exists to make ---------------------------------------------------
a1 = out["arms"]["1_no_band_default"]
a2 = out["arms"]["2_band_scoped_recommended"]
a3 = out["arms"]["3_band_every_slice_as_branch"]
checks = {
    "1_reports_every_slope_without_a_band":
        all(v["n_slices_reported"] == 7 for v in a1.values()),
    "1_is_ungated_and_passed_is_None":
        all(v["gated"] is False and v["passed"] is None and v["band"] is None
            for v in a1.values()),
    "2_fails_qwerty": a2["qwerty"]["passed"] is False,
    "2_passes_the_other_three":
        all(a2[L]["passed"] is True for L in ("azerty", "dvorak", "qwertz")),
    "2_still_REPORTS_the_thin_buckets_it_did_not_gate_on":
        all(a2[L]["out_of_band"] for L in ("azerty", "dvorak", "qwertz")),
    "2_relegates_them_to_advisory":
        all(a2[L]["out_of_band_advisory"] == a2[L]["out_of_band"]
            for L in ("azerty", "dvorak", "qwertz")),
    "3_reproduces_the_branch_all_four_fail":
        all(v["passed"] is False for v in a3.values()),
    "bucket_centered_support_now_recorded":
        all("bucket_centered" in v["support_keys"] for v in a1.values()),
}
out["checks"] = checks
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print()
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("\nALL CHECKS PASS:", all(checks.values()))
print("WROTE", OUT)
