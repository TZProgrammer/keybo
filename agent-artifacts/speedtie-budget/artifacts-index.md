# Artifacts index — `speedtie` (SPEEDTIE-BUDGET)

Subagent of `keybo-optimization`. Question: **does SPEEDTIE-1's free gauge headroom survive at
the campaign's full ~10M-eval budget, or is it an under-convergence artifact?**

**VERDICT (by the rule pre-registered before any result existed): INDETERMINATE.** Post-hoc, the
evidence leans H-REAL and against H-UNDER; that reading is labelled post-hoc in `docs/`.

**MODELLED ONLY** — g-frame, baked 90 WPM, blend-v1, `skipgrams=1-skip31`. Nothing here is a
claim about realized human typing speed. **No layout is adopted or recommended for adoption.**

> ## ⚠ READ THE SELF-AUDIT BEFORE REUSING ANY OF THIS — `report.md` §5b (also committed as
> `agent-artifacts/speedtie-budget/SELF-AUDIT-and-report.md` on branch `speedtie-budget` @ `d2f612e`)
> 1. **Do NOT merge/cherry-pick branch `speedtie-budget` into `main` expecting artifacts only.**
>    `git diff 45ea276 b36b8d8 -- src/ tests/` = **+912 lines resurrecting
>    `src/keybo/analysis/evidence_scorer.py`**, a module the ledger deliberately deleted, with no
>    test covering it. Take the `agent-artifacts/` commits; leave the `src/` change.
> 2. **`drivers/run_budget.py:35` and `analyze_budget.py:26` hardcode `WORKTREE = /tmp/speedtie`**
>    (plus `OUTDIR`/`STATE`/`RUNS`/`FROZEN_1M`). That is trap 35 — the same defect I flagged in
>    `search_placebo.py`. Copying these into a fresh worktree launches the search into a dead path.
>    Derive it from `git rev-parse --show-toplevel` or pass it as an argument first.
> 3. **The size-matched spread test is partly circular** — all 3 of the 10M distinct champions are
>    members of the 1M pool the placebo subsets are drawn from, so "12 of 14 at/above median" is
>    *consistent with* H-REAL, not discriminating evidence. The two non-circular legs
>    (`M_gauge = 1.0000`, Hamming-over-distinct 26.20 → 26.00) are unaffected.
> 4. **Every ms/char figure rests on ONE timing implementation** — `evobj.py:306-308` imports the
>    same `keybo.analysis.timecard.TimeSurface` the CLI uses, so driver-vs-CLI agreement
>    (2.98e-12) is a consistency check that shares the component under test.
> 5. `--resume` on these exact runs is impossible (`.keys.npy` deleted); a fresh same-seed run is
>    deterministic and equivalent.

The `/tmp/speedtie` worktree (branch `speedtie-budget`) is EPHEMERAL. Everything needed to read
or reproduce this result is copied here. Commits on that branch: `40ff53c` (pre-registration), `23584ea` (drivers),
`b36b8d8` (result + addendum), **`d2f612e` (self-audit) = the tip**. The commits SURVIVE workspace
destruction in the shared `.git` (verified empirically by the parent on two earlier children); the
`/tmp/speedtie` checkout does not, so **this directory is the durable copy.**

---

## Headline numbers

| | 1M (frozen placebo) | ~8.4M achieved ("10M") |
|---|---|---|
| unique evals | 0.99M–1.08M | **7,787,578 – 9,216,894** (mean 8,434,001) |
| n seeds | 6 | 6 completed (n=5 primary; 1 below the 80% floor) |
| distinct champions | 6 of 6 | **3 of 5** primary / 4 of 6 all |
| speed range | 0.1760 ms/char | **0.1236** (= 2.00x arm B's own sd 0.0617) |
| median per-gauge range ratio | — | **M_gauge = 1.0000** |
| mean Hamming over runs | 26.20 | 19.20 |
| mean Hamming over **distinct** champions | 26.20 | **26.00** ← unchanged |
| dominating pairs | 0 | **0** (zero ties in 84 cells) |

**The mechanism:** the extra ~7.4M evaluations found **zero new territory**. 2 seeds kept their
own 1M champion (Hamming 0); 3 moved **onto another seed's existing 1M champion**. The n=5 10M
champion set is a strict **subset** of the 1M set.

---

## Files

### `docs/` — read these first
| file | what it is |
|---|---|
| `PREREGISTRATION.md` | The decision rule, **committed at `40ff53c` before any 10M result existed.** Numeric thresholds for H-REAL / H-UNDER / indeterminate, the asymmetric-case clause, the one-factor design, the 6 pre-run positive controls, and the 4 defects in the inherited `search_placebo.py` that are NOT inherited. |
| `ADDENDUM-post-hoc.md` | ⚠ **Computed AFTER reading the result, and labelled so.** Why the rule returned indeterminate; the set-size-artifact analysis of the one failing leg; the size-matched placebo; the mechanism; what budget would settle it. |

### Results
| file | what it is |
|---|---|
| `speedtie-budget-10000000.json` | **The primary result.** Both budgets scored through the shipped CLI, all pre-registered statistics, per-gauge spreads, Hamming (over runs AND over distinct champions), the full dominance matrix with `n_ge`/`n_strict`/`n_ties`, the `sfr` invariance check, and `VERDICT`. Includes `sensitivity_including_subfloor_runs` (n=6) — same verdict, so the exclusion is not what produced the result. |
| `convergence-10000000.json` | Per-seed convergence by ARME-1's registered criterion (*has best-fitness stopped improving?*, not budget fraction): last-improvement epoch, unique evals at that point, epochs flat after, improvement over the final half. |

### `runs/` — raw per-seed search output (one set per seed, r=0..5; seed = `900000 + 7919*r`)
| file | what it is |
|---|---|
| `b10000000-r<N>.json` | Champion, `unique_evals` **achieved**, `budget_requested`, epochs run, elapsed, top-50 archive, per-island bests. |
| `b10000000-r<N>.log` | Full per-epoch trace (unique count, calls, best fitness, best layout). The convergence diagnostic is derived from these. |
| `b10000000-r<N>.ckpt.json` | Per-epoch checkpoint (trap 7). Independently records `n_unique`. |
| `budget-10000000-summary.json` | All six runs, rc per seed, achieved evals, wall clock. |
| `keys-npy-census.json` | **Absent by design** — see the note below. |
| `timing-r0.out`, `run5.out`, `timing-rc.txt`, `run5-rc.txt` | Runner stdout + rc sentinels (rc=0 both). |

> **Deleted on purpose:** each run also wrote a `b10000000-r<N>.keys.npy` (~70 MB; 388 MB total)
> holding the blake2b-8 dedup key set. These are pure search-internal state with no analytical
> content, and the `unique_evals` figure they back is **triply recorded and verified to agree**
> across (a) the run JSON, (b) the `.ckpt.json` `n_unique` field, and (c) the independent
> per-epoch log trace — checked for all six seeds before deletion. Removing them costs no
> verifiability. **Caveat for a successor: `--resume` needs them, so a resumed continuation of
> these exact runs is no longer possible; a fresh run from the same seed is (the search is
> deterministic given seed+islands).**

### `drivers/` — everything needed to re-run
| file | what it is |
|---|---|
| `run_budget.py` | **Mine.** Replaces `search_placebo.py`: no subprocess timeout, writes only into my own state dir, islands/epochs as args. ⚠ its `cwd` is correct **for me only** — `WORKTREE` is hardcoded to `/tmp/speedtie` (line 35). See hazard 2 above before reuse. |
| `analyze_budget.py` | **Mine.** Scores champions via the shipped CLI and evaluates the pre-registered rule mechanically. |
| `convergence.py` | **Mine.** Convergence diagnostic from the epoch traces. |
| `search.py`, `evobj.py` | Copied **UNMODIFIED** from `optevidence-1/drivers/` (md5-verified: `2e499152489dbdc7e7f6c1a69a8c71a8`, `dc45ef503792576157a872a996d9e9d7`). |
| `search_placebo.py` | The inherited 1M driver, kept **unmodified** for provenance. **Not used** — see `PREREGISTRATION.md` §6. |
| `evidence_scorer.RESTORED.py` | `src/keybo/analysis/evidence_scorer.py` (md5 `01f3a95ab7a0f53f8f9d5be057fc437e`), restored byte-identically from commit `1b4a4d8` because `evobj.py:42` imports `LIVE_GAUGES` from it and it is **deleted** at ledger HEAD `45ea276`. Without this the driver cannot import. |

### Inputs (elsewhere — not copied, they are another workspace's durable artifacts)
- 1M frozen champions + gauges: `state/keybo-optimization/artifacts/speedtie-1/speedtie-summary.json`
- 1M source pool (seeds, achieved evals, bands): `state/keybo-optimization/artifacts/optevidence-1/search-noise-placebo.json`
- Objective weights: `state/evidence-scorer/artifacts/arm-random400-native.json` (loaded by `FastEval`; the `baseline` arm never reads the evidence score)
- Corpus: `data/corpus/blend-v1` in the repo. `trigrams.txt` md5 **re-derived** = `c5066fa7bcc46dea1ecbc987fb465b4a`.

---

## Reproduce

```bash
git worktree add /tmp/<name> <ledger-sha>          # 45ea276 or later
cd /tmp/<name> && uv sync --frozen                 # `--no-sync` needs a synced venv first
cp <this>/drivers/{search,evobj,run_budget,analyze_budget,convergence}.py agent-artifacts/…/drivers/
cp <this>/drivers/evidence_scorer.RESTORED.py src/keybo/analysis/evidence_scorer.py
uv run --no-sync python …/run_budget.py 10000000 6 20 120     # ~1000s/seed alone, 2322s for 5 parallel
uv run --no-sync python …/analyze_budget.py 10000000
```

`islands=20` is **load-bearing**: `search.py:318-323` derives the initial population from
`(seed, islands)` alone, so 20 makes the 10M population bit-identical to the 1M one. `epochs=120`
makes calls/island/epoch = 8,125, exactly the placebo's per-epoch spend.

## Verified controls (all passed; the first four ran BEFORE any result was read)
1. **Cold start read, not assumed** — `search.py:318-323`: `islands x 64` uniformly random C30M perms from `default_rng(seed)`, no incumbent, no warm start.
2. **Worktree isolation POSITIVE, not just absent hardcodes** — no `repos/keybo` literals, all `sys.path.append` (trap 35), **and** `FastEval.corpus_dir` resolved to `/tmp/speedtie/data/corpus/blend-v1`.
3. **All six frozen 1M champions reproduce** in my worktree: worst |diff| **2.814e-12**, arm B **exactly 0.0**.
4. **My analysis code reproduces every frozen SPEEDTIE-1 number** — 13 published gauge spreads to worst **4.5e-5** (frozen table is 4dp), 0 dominating pairs, and the five better/worse counts vs arm B **exactly** (7/7, 4/10, 9/5, 9/5, 8/6, zero ties). Re-verified after a mid-run refactor.
5. **The 10M run reproduces the 1M run's trajectory** — seed 900000's own log at epoch 9: `unique=1,008,758`, champion `flmpg-yuo,sntdcireahkxbwv'.jzq` — the 1M placebo's exact achieved count and champion. Unplanned, and the strongest confirmation that this is the same search continued.
6. **ms/char code path bit-identical** to the commit the 1M run used: 0 diff lines across `timecard.py`, `kmstats.py`, `corpus.py`, `geometry.py`, `classify.py`, `comfort.py`, `oxey.py` and all of `data/`.
7. **`sfr` invariance tested directly** (trap 23, not via a variance threshold): **1 distinct value** (`2.659577102696`) across all champions at both budgets. Excluded from every spread/ratio/dominance/win-count.
8. **`lsb`/`lsb-dist` duplication measured, not cited** (trap 25): spearman **1.0000** on the six 1M champions (pearson 0.9987) and **1.0000** on the three 10M survivors.
