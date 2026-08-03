"""INVARIANT 2/3/4 driver: the FULL calibration gate blocks, per fold AND per seed, WITH support.

Runs the reconstructed-tree `validate()` (3-file cherry-pick onto current origin/main) and dumps
EVERY calibration_gate block verbatim, including the `support` map the parent's artifact reduced to
a bare `support_present: true`. Support is the crux of INVARIANT 2: a slope from a 12-cell bucket
and one from a 900-cell bucket are the same number and very different claims.

Nothing is thresholded here. Scope rules are applied POST-HOC to the recorded slopes by g02, so a
single expensive LOLO serves every candidate configuration and no configuration can be accused of
having been measured on its own private run.
"""
import json
import os
import sys

# All four thread vars BEFORE any import that pulls in xgboost/numpy BLAS (after is inert).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

WT = "/local/home/zegertho/repos/keybo-wt-gateaudit"
OUT = "/tmp/gateaudit/run/g01_scope.json"
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"

import keybo  # noqa: E402
import keybo.verdicts as V  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

# PROVENANCE FIRST — the venv resolves `keybo` to the shared checkout silently.
prov = {
    "keybo__file__": keybo.__file__,
    "expected_prefix": WT,
    "band_constant": list(V.CALIBRATION_SLOPE_RECOMMENDED_BAND),
    "has_require_calibration": hasattr(V, "require_calibration"),
    "argv": sys.argv,
}
print("PROVENANCE:", json.dumps(prov, indent=2), flush=True)
if not keybo.__file__.startswith(WT):
    raise SystemExit(f"WRONG TREE: keybo resolved to {keybo.__file__}, expected under {WT}")

rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
print(f"loaded {len(rows)} bigram rows", flush=True)

report = validate(rows, seeds=[0, 1, 2], ngram="bigram", progress=False)

# Dump every calibration_gate block VERBATIM, per fold per seed, support included.
folds = {}
for layout, fold in report["folds"].items():
    folds[layout] = {
        "n_cells": fold["n_cells"],
        "ceiling": report["ceilings"][layout],
        "seeds": [
            {
                "seed": m["seed"],
                "rho": m.get("rho"),
                "tau_all4": m.get("tau_all4"),
                "wmae_model": m.get("wmae_model"),
                "calibration_gate": m["calibration_gate"],
                "bucket_matrix": {
                    str(b): {
                        "slope": bm.get("slope"),
                        "n": bm.get("n"),
                        "n_participants": bm.get("n_participants"),
                        "rho": bm.get("rho"),
                        "wmae": bm.get("wmae"),
                    }
                    for b, bm in (m.get("bucket_matrix") or {}).items()
                },
            }
            for m in fold["seeds"]
        ],
    }

out = {
    "provenance": prov,
    "config": report["config"],
    "n_rows": len(rows),
    "folds": folds,
    "pooled": report.get("pooled"),
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print("WROTE", OUT, flush=True)
