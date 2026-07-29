"""Is the LOLO objective REALLY unscoreable? The brief says every ceiling is nan (1 pid/layout).
Measured: the aalto frame has 64-54690 pids per layout, so split_half_ceiling should WORK."""
import json, time
from keybo.testkit import assert_module_under
assert_module_under("keybo", "/tmp/kaggle")
from keybo.data.strokes import load_strokes
from keybo.training.validate import split_half_ceiling, build_cells

t0 = time.time()
rows = load_strokes("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv", ngram_len=2,
                    wpm_threshold=0, min_samples=1)
print(f"loaded {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)
out = {}
for holdout in sorted({r.layout for r in rows}):
    test = [r for r in rows if r.layout == holdout]
    npid = len({s[2] for r in test for s in r.samples})
    t1 = time.time()
    c = split_half_ceiling(test, n_boot=10, seed=0)
    cells = build_cells(test)
    out[holdout] = {"ceiling": c, "n_participants": npid, "n_cells": len(cells)}
    print(f"  {holdout:8s} pids={npid:6d} cells={len(cells):5d} ceiling={c!r}  ({time.time()-t1:.0f}s)", flush=True)
print()
finite = [k for k, v in out.items() if v["ceiling"] == v["ceiling"]]
print(f"FOLDS WITH A FINITE CEILING: {len(finite)}/{len(out)} -> {finite}", flush=True)
print("BRIEF CLAIM was: every ceiling nan => tune --objective lolo cannot score at all.", flush=True)
print("VERDICT:", "REFUTED on this frame" if finite else "CONFIRMED on this frame", flush=True)
json.dump(out, open("/tmp/kaggle-work/ceiling_probe.json", "w"), indent=2)
