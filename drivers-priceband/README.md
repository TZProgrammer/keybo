# `priceband` drivers — the constrained-frontier (shadow-price) estimator

Closes **OPENQ-1 A1**: is a gauge's speed price identifiable *among near-optimal boards*?

Prior arms priced `sfb` by **perturbing a fixed board** and reading `d(ms)` off `d(sfb)`. All four
failed (collider / disruption-confounded / floor-artifact / under-powered). This arm changes the
estimand instead of the estimator: it prices the **constraint**, not a perturbation.

    F(c) = min { ms_per_char(L) : sfb(L) <= c }        price(c) = -dF/dc

Every point of `F` is near-optimal *by construction*, so there is no disruption term to difference
out — at each cap **every other gauge is re-optimized**. See
`state/priceband/PREREGISTRATION.md` for the identification argument and the six pre-registered
criteria that gate reporting a price.

| driver | what it does |
|---|---|
| `_env.py` | env guard: pins the 4 thread vars, asserts `keybo` resolves to THIS worktree (shared-venv footgun) |
| `boards.py` | the 14-board field, verbatim from `state/pair-perturb/artifacts/v01_table.json` |
| `search.py` | `swap_perms`/`cycle_perms` (verified exact), 2-opt, cap-constrained descent, 3-opt |
| `c00_env.py` | evaluator verification gate (fast evaluators vs shipped `card()`/`KmStats`) |
| `c01_scope.py` | `swap_perms` correctness, field 2-opt convergence, random-start reachability |
| `c02_feas.py` | point-target constrained search — **measured to stall ~3 ms/char above the field** |
| `c03_cap.py` | inequality-cap form + best-of-N behaviour (restart-count input) |
| `c04_geom.py` | sfb floor, where the UNCONSTRAINED speed optimum sits, 3-opt cost, `cycle_perms` check |
| `c05_premise.py` | **tests the register's central premise** — separates LOCAL vs GLOBAL "at the floor" |
| `c06_control.py` | **P1 positive control** (+0.39 on qwerty) + the literal sign-blind falsifier |
| `c07_frontier.py` | **PRIMARY**: the frontier, R replicates x caps, incl. inert placebo caps |
| `c08_analyze.py` | evaluates the six criteria, emits per-interval prices + CIs |
| `c09_warm.py` | **F5** warm-start cross-seeding (the conservative-direction falsifier) |

Run order: `c00` → `c01`–`c05` (scoping) → `c06` (gate) → `c07` (long; detached) → `c08` → `c09` → `c08` again.
All artifacts land in `/local/home/zegertho/agent/state/priceband/artifacts/`. Nothing is
hand-transcribed. Evaluators are re-verified on every run (measured 3.1e-12 / 7.1e-15).
