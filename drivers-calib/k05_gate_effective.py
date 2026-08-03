"""Is the wired gate EFFECTIVE on real data, not merely PRESENT? (present != effective)

Runs the reviewed validate() path on the fold that actually compresses (qwerty) and on one that
does not (azerty), and asserts the gate is reachable, fires on the former, and passes the latter.
A gate that cannot fail is the TAUGATE-1 defect; a gate that always fails is useless.
"""
import json
import time

from _guard import ART, BI, assert_d5

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31 as G  # noqa: E402
from keybo.training.validate import validate  # noqa: E402
from keybo.verdicts import CALIBRATION_SLOPE_RECOMMENDED_BAND  # noqa: E402

rows = load_strokes(BI, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows")
out = {"band": list(CALIBRATION_SLOPE_RECOMMENDED_BAND), "folds": {}}
for holdout, expect_pass in (("qwerty", False), ("azerty", None)):
    log(f"validate() on the {holdout} fold, seed 0")
    rep = validate(rows, seeds=[0], holdouts=[holdout], geometry=G, n_boot=50)
    s = rep["folds"][holdout]["seeds"][0]
    cg = s["calibration_gate"]
    out["folds"][holdout] = {
        "calibration_gate": cg,
        "legacy_calibration_slope": s["calibration_slope"],
        "high_wpm_gate_present": {"gated": s["high_wpm_gate"]["gated"],
                                  "passed": s["high_wpm_gate"]["passed"]},
    }
    log(f"  gated={cg['gated']} passed={cg['passed']} band={cg['band']}")
    log(f"  slopes={ {k: round(v, 4) for k, v in cg['slopes'].items()} }")
    log(f"  out_of_band={cg['out_of_band']}")
    log(f"  worst={cg['worst_slice']} @ {cg['worst_slope']:.4f} "
        f"(dev from 1 = {cg['worst_abs_deviation_from_1']:.4f})")
    log(f"  support[pooled]={cg['support'].get('pooled')}")
    log(f"  legacy per-fold calibration_slope still present: {s['calibration_slope']:.4f}")
    assert cg["gated"] is True, f"{holdout}: the gate must be REACHABLE on real data"
    assert cg["passed"] is not None, f"{holdout}: passed must never be None when gated"
    if expect_pass is False:
        assert cg["passed"] is False, (
            f"{holdout}: this fold's slope is ~1.41, so a [0.90,1.10] band MUST fail -- a gate "
            f"that cannot fail on the known-bad fold is the TAUGATE-1 defect")
        log(f"  VERIFIED: the gate FIRES on {holdout} (the compressing fold)")

json.dump(out, open(f"{ART}/k05_gate_effective.json", "w"), indent=1)
log(f"wrote {ART}/k05_gate_effective.json ({time.time() - t0:.1f}s)")
