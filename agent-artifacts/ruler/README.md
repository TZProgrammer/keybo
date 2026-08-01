# `ruler` drivers — the reported gauge as an OPT-IN `optimize` objective

Measurement drivers behind `optimize --gauge-objective` (branch `trigram-objective`). Product code
is under `src/`; these are the instruments, kept so every number in the report is reproducible.

Run each with the repo's four thread vars pinned and `PYTHONPATH` on THIS worktree — the shared
`.venv` resolves `keybo` to whatever branch the shared checkout is on, and it moved twice in one
session. Every driver asserts its own provenance and refuses to run from the wrong tree.

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  PYTHONPATH=<this worktree>/src <venv>/bin/python agent-artifacts/ruler/<driver>.py
```

| Driver | Question | Verdict |
|---|---|---|
| `p0_feasibility.py` | Can the gauge be expressed as one `(31,31,31)` ms table and searched? | YES — parity vs `analyze` worst rel **1.213e-14** over 6 boards; eval 0.185 ms vs `card()` 49.9 ms (**270x**) |
| `p1_decompose.py` | Does `TableBigramScorer + TableTrigramScorer` reconstitute the gauge? | **NO — rel 1.5013e-2 off.** Bigram-term weighting (trigram first-2 marginal vs `bigrams.txt`, kept mass 887,147,352 vs 913,956,722) is ~1.41 of the 1.50 pp; the 3-seed table MEAN is the rest. This is why the objective is built from `TimeSurface.triple_ms_table()` |
| `rerun_matched.py` | On the CORRECT ruler, what is the restart curve, and does α=0.98 still beat α=0.999? | α=0.98 wins: **+1.39 ms/char** mean paired gain, **88/96**, bootstrap CI95 [1.227, 1.547]; **6/6** equal-time budgets. Saturates at N=2 vs α=0.999's N=32. Mechanism: `T_end/T0` 0.0046 vs **0.894** — at α=0.999 SA returns the unmodified start layout in **44/96** runs |

`rerun_matched.py` takes the seed count as `argv[1]` (default 48; the report uses 48 and 96) and
writes its result JSON to `state/ruler/artifacts/`. It re-asserts the parity gate AFTER the search
and refuses to report if it fails.
