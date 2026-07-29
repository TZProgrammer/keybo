"""One blend-search cell: `blend_cell.py <weighting-label> <seed>` -> runs/blend-<label>-<seed>.json.

Split out so the cells run as independent single-threaded processes. Each re-runs the gates
itself: a cell whose evaluator or anchors are wrong must not contribute a number because a
sibling process happened to check.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_blend import ANCHORS, RUNS, BlendSearch, log, weightings  # noqa: E402

from keybo.analysis import surfaces as S  # noqa: E402
from keybo.scoring import model_norm as MN  # noqa: E402


def main(label: str, seed: int) -> int:
    fits = MN.SurfaceFits()
    fits.assert_batch_invariant(S.C30M)
    anchors = MN.Anchors.read(ANCHORS)
    anchors.assert_direction()
    anchors.assert_matches_surfaces(fits, anchors.provenance["probe_layout"])
    spec = weightings()[label]
    log(f"cell {label} seed={seed}: {spec.describe()}")
    payload = BlendSearch(fits, anchors, spec).run(seed)
    payload["weighting"] = label
    payload["weights"] = dict(spec.weights)
    payload["rule"] = spec.rule
    RUNS.mkdir(exist_ok=True)
    target = RUNS / f"blend-{label}-{seed}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.rename(target)
    log(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
