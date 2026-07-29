# normgauge — artifacts index

All artifacts live on branch `normgauge` in `/tmp/normgauge` (a git worktree; **committed**, not
pushed) plus this state dir. Frame for every number: shipped `.standardized` surfaces, geometry-only
`g`, **BAKED 90 WPM**, corpus `blend-v1`. **MODELLED ONLY.**

## LOAD-BEARING (a conclusion rests on these)

| artifact | what it is | why load-bearing |
|---|---|---|
| `drivers-normgauge/anchors.json` | the 0/1 anchors per model + full provenance (frame, corpus sha256, per-surface sha256, pool seed/n/statistic, achieved unique_evals, champion per model, pinned tile, numpy/python) | **the gauge is undefined without it**; `--model-anchors` reads it and refuses on drift |
| `drivers-normgauge/weight-evidence.json` | the four candidate rules measured, the decision-tree branch, the shipped weights, the CI self-diagnostics, and the IQR-variant weights | **the weights and their justification** |
| `drivers-normgauge/PREREGISTRATION.md` | prereg + AMENDMENT 1 (pre-result) + AMENDMENT 2 (post-result, with blast radius) | proves thresholds were not tuned; records 3 defects I killed in my own design |
| `drivers-normgauge/blend-report.json` | 18 cells, search-noise quadruple, 15-gauge + ms/char table, contested-axis counts, ms/char agreement | **deliverable 4** |
| `drivers-normgauge/SELF-KILL.md` | the hostile pass: 5 kills incl. one in a commit I had already made | the corrected verdicts (drop-pool is a TIE, not a 2.4x win) |
| `state/normgauge/artifacts/BRIEF-CORRECTION-AUDIT.md` | audit of the parent's two mid-turn brief corrections | **found a wrong constant inside a correction** (COMMUNITY has 7 participants, not 9) |
| `src/keybo/scoring/model_norm.py` | the shipped gauges + blend scorer | the deliverable itself |
| `src/keybo/cli/optimize.py` | `--model-weight` / `--model-anchors` | **deliverable 2**: usable by the shipped optimizer |
| `tests/scoring/test_model_norm.py` (27) · `tests/cli/test_optimize_model_weight.py` (9) | mutation-controlled tests | **deliverable 3** |

## BULK (reproducible from the above; kept for audit, not for reading)

| artifact | note |
|---|---|
| `drivers-normgauge/runs/anchor-<POOL>-<seed>.json` (9) | per-cell anchor searches, 5M unique evals each |
| `drivers-normgauge/runs/blend-<label>-<seed>.json` (18) | per-cell blend searches |
| `drivers-normgauge/anchors-evidence.json`, `blend-runs.json` | per-island traces |
| `drivers-normgauge/support-cells.json`, `support-*.npy`, `rows-*.npy` | per-cell training support maps |
| `drivers-normgauge/logs/*.log` | run logs; rc read from `runs/.sentinel-*`, never from a callback |
| `drivers-normgauge/cache/rows-*.pkl` | **~370 MB**, a parse cache keyed by (path, mtime, size, labels). Regenerable; delete freely |
| `drivers-normgauge/probe_*.py`, `reorder_check.py` | FIND-phase probes + the reordering analysis |

## VERIFIED-DURABLE

Everything above is **committed on branch `normgauge`** (`git log --oneline`: prereg `64c9ddf` →
code `156bd47` → anchors `8387f1c` → amendments `d517811`/`11755aa` → weights `8105bec` → results
`c6f9932` → self-kill `afb83e8` → CLI `aba7c69`). The two state-dir markdowns are outside the repo
and live here. **No push, no CR** — `/tmp/normgauge` is a worktree of the shared clone, so the branch
survives this workspace's destruction.

⚠ `drivers-normgauge/cache/` is the only bulky item and is **excluded from git** (untracked).
