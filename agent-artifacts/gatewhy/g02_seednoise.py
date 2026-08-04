"""GATEWHY-1 §6.2 — THE NON-VACUOUS GATE CONTROL: SEEDNOISE.

The published gate control cannot fail (g01: the incumbent's deltas are deviations from its own
3-seed mean, so they sum to zero and all three can never be negative). So it licenses nothing.

This is the control that CAN fail: retrain the SERVED 20-column frame -- identical in every respect,
same data, same folds, same hyperparameters -- at NEW seeds [3,4,5], and score it against the
PUBLISHED CUR baseline (the seeds [0,1,2] per-fold mean). Nothing but the seed differs, so a
structural refusal here is INSTABILITY BY CONSTRUCTION, not a property of any candidate.

ROWOFFSETS-1 ran the same idea for a geometry arm and found the shipped geometry merely reseeded
FAILED the gate. Whether that holds for the FRAME comparison at these folds/seeds is unmeasured, and
it is the hinge of the whole question.

Registered at GATEWHY-preregistration.md §6.2 @ 821ce01, BEFORE this ran.

Detached-friendly: writes a SENTINEL when finished (never `wait $PID`), checkpoints after every fold.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-gatewhy/agent-artifacts/gatewhy")
from _boot import ARTIFACTS, assert_tree, require  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training import validate as V  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

require(V, "validate")

# The SERVED frame, asserted -- this arm's whole meaning is "nothing changed but the seed".
print(f"[frame] served bigram columns: {len(BIGRAM_FEATURE_NAMES)} ending {BIGRAM_FEATURE_NAMES[-1]!r}")
if len(BIGRAM_FEATURE_NAMES) != 20 or BIGRAM_FEATURE_NAMES[-1] != "wpm":
    raise SystemExit(f"ABORT: served frame is not the shipped 20c/wpm frame: {BIGRAM_FEATURE_NAMES}")

NEW_SEEDS = [3, 4, 5]  # registered: seeds NOT used by the published run
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
SENTINEL = "/tmp/gatewhy_wk/g02.sentinel"
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows; layouts {sorted({r.layout for r in rows})}")

# EXACTLY hybridtri's CUR arm configuration, seeds swapped. Any other difference would make this a
# different comparison rather than a reseed.
log(f"SEEDNOISE: served frame, seeds {NEW_SEEDS} (published used [0,1,2])")
rep = validate(
    rows,
    seeds=NEW_SEEDS,
    ngram="bigram",
    n_boot=10,
    geometry=ROW_STAGGERED_31,
    train_params={"n_jobs": 8},
)
cfg = rep.get("config", {})
log(f"done. config seeds={cfg.get('seeds')} interp={cfg.get('interp')!r} monotone={cfg.get('monotone')}")
# VERIFY THE ARM IS WHAT IT CLAIMS (a config label is a claim, not the referent).
if cfg.get("interp") is not False or list(cfg.get("seeds") or []) != NEW_SEEDS:
    raise SystemExit(f"ABORT: SEEDNOISE arm config is not served/new-seeds: {cfg}")

out = {
    "prereg": "agent-artifacts/gatewhy/GATEWHY-preregistration.md @ 821ce01 §6.2",
    "arm": "SEEDNOISE (served 20c, seeds [3,4,5])",
    "n_rows": len(rows),
    "seeds": NEW_SEEDS,
    "served_columns": list(BIGRAM_FEATURE_NAMES),
    "report": rep,
}
with open(f"{ARTIFACTS}/g02_seednoise.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/g02_seednoise.json")

# Quick inline verdict so the log alone carries the answer even if the JSON is never read.
HT = json.load(open("/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri/lolo.json"))
cur = HT["arms"]["CUR"]
base = {}
for holdout, fold in cur["folds"].items():
    acc: dict[int, list[float]] = {}
    for rec in fold["seeds"]:
        for b, r in (rec.get("bucket_rhos") or {}).items():
            if r is not None:
                acc.setdefault(int(b), []).append(float(r))
    base[holdout] = {b: float(np.mean(v)) for b, v in sorted(acc.items())}

from keybo.verdicts import bucket_regression_report  # noqa: E402

print()
print("SEEDNOISE vs the PUBLISHED CUR baseline (seeds [0,1,2] per-fold mean):")
structural = {}
for holdout, fold in rep["folds"].items():
    hits: dict[int, int] = {}
    n = len(fold["seeds"])
    for r in fold["seeds"]:
        blk = bucket_regression_report(
            {int(k): v for k, v in (r.get("bucket_rhos") or {}).items()},
            base.get(holdout, {}),
            f"SEEDNOISE/{holdout}/s{r['seed']}",
        )
        for b in blk["regressing_high_buckets"]:
            hits[int(b)] = hits.get(int(b), 0) + 1
    st = sorted(b for b, h in hits.items() if h == n)
    if st:
        structural[holdout] = st
    print(f"  {holdout:<8} structural {st}  noise {sorted(b for b, h in hits.items() if 0 < h < n)}")
print()
print(f"SEEDNOISE PASSES: {not structural}" + ("" if not structural else f"  -> {structural}"))
out["inline_verdict"] = {"structural": structural, "passed": not structural}
with open(f"{ARTIFACTS}/g02_seednoise.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)

with open(SENTINEL, "w") as fh:
    fh.write("done\n")
log(f"SENTINEL {SENTINEL}")
