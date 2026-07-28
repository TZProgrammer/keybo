# SPEEDTIE-BUDGET — does the free gauge headroom survive at the full budget?

Subagent `speedtie` of `keybo-optimization`. Worktree `/tmp/speedtie`, branch `speedtie-budget`
(commits `40ff53c` prereg → `23584ea` drivers → `b36b8d8` result; **nothing pushed**,
`PREREGISTRATIONS.md` untouched). Durable record: `state/speedtie/artifacts/` +
`profiles-and-artifacts-index.md`.

**MODELLED ONLY** — g-frame, baked 90 WPM, blend-v1, `skipgrams=1-skip31`. Nothing below is a
claim about realized human typing speed. **No layout adopted or recommended for adoption.**

---

## VERDICT: **INDETERMINATE** by the rule I pre-registered before any result existed

Post-hoc, the evidence **leans H-REAL and against H-UNDER** — every H-UNDER-specific prediction
failed — but I do not upgrade the verdict, and §4 says exactly why not.

## 1. What was run

Same engine, same objective (`--arm baseline` = minimize served ms/char), same corpus, **same
seed formula the 1M placebo used** (`900000 + 7919*r`), so the two budgets are paired run-for-run.
One factor varies: **epochs 12 → 120**. `islands` held at the placebo's **20**, which makes the
initial population **bit-identical** (`search.py:318-323` derives it from `(seed, islands)` alone),
and 120 epochs makes calls/island/epoch **8,125** — exactly the placebo's per-epoch spend. So this
is literally *the same search continued ~8.4x longer*.

**Achieved `unique_evals` — NOT the 10M requested.** The run stops on the epoch schedule, not the
unique target:

| seed | achieved | % of request | champion |
|---|---|---|---|
| 900000 | 8,791,523 | 87.9% | `flmpg-yuo,sntdcireahkxbwv'.jzq` (= arm B) |
| 907919 | 8,546,624 | 85.5% | `pyu.,gdfnlhieaocstrmkj'-qbwzvx` |
| 915838 | 8,252,292 | 82.5% | `pyou,vdflrghaeictsnmk'j.-wbzxq` |
| 923757 | 9,216,894 | 92.2% | `flmpg-yuo,sntdcireahkxbwv'.jzq` (= arm B) |
| 931676 | **7,787,578** | **77.9%** | `lnfdg-.yehcrstmaoiupxqbwvk,'jz` — **below my 80% floor, EXCLUDED from primary** |
| 939595 | 8,009,098 | 80.1% | `flmpg-yuo,sntdcireahkxbwv'.jzq` (= arm B) |

n=6 completed, **n=5 primary**. A sensitivity analysis including the sub-floor seed returns the
**same verdict**, so the exclusion is not what produced the result. Mean achieved 8,434,001 — an
**~8.4x** budget increase, which is what I claim, not 10x. (For scale, the campaign's own "10M"
arm B also fell short, at 9,252,349.) Cost: 1014s for one seed alone, 2322s for five in parallel —
so the brief's 2.5h/seed ceiling was never in danger and I ran the full n=6 rather than reducing
the budget.

## 2. The pre-registered rule, and exactly why it did not fire

| statistic | threshold | observed | fires? |
|---|---|---|---|
| `R_speed` (range ratio) | H-UNDER ≤ 0.50 | **0.7023** (0.1760 → 0.1236) | ✗ |
| `M_gauge` (median per-gauge range ratio) | H-UNDER ≤ 0.50 / H-REAL ≥ 0.80 | **1.0000** | ✗ / ✓ |
| mean Hamming ratio | H-UNDER ≤ 0.75 | **0.7328** (26.20 → 19.20) | ✓ |
| live gauges with `ratio_g` ≥ 5.0 at 10M | H-REAL ≥ 2 | **1** (oxey-style 5.92x) | ✗ |
| dominating pairs | H-REAL = 0 | **0** of 6, zero ties in 84 cells | ✓ |

- **H-UNDER requires all three legs.** One fired. **H-UNDER is not supported.**
- **H-REAL** got two of three and failed exactly one — the absolute-magnitude leg — **solely**
  because `imbalance`'s ratio fell 17.70x → 3.29x.
- `M_gauge = 1.0000`: **8 of 14 live gauges have a range ratio of exactly 1.0000** (a 9th at
  0.9968). The additive gauge spread did not move.

## 3. The mechanism — the real finding

**The extra ~7.4M evaluations bought no new territory.** Run-for-run: 2 seeds kept their own 1M
champion (Hamming 0), and **3 moved onto another seed's existing 1M champion** (2 of them onto arm
B's). The n=5 10M champion set is a **strict subset** of the 1M set — **zero new layouts**. Across
all six seeds exactly one layout appears that wasn't already in the 1M pool, and it is 7/30 from
that seed's own champion. **The 1M pool already contained every optimum an ~8.4M-eval search could
find.**

**The decisive dissociation.** Mean Hamming *over runs* falls 26.20 → 19.20 — but **entirely**
because 3 run-pairs became identical (`n_zero_pairs` 0 → 3). Mean Hamming *over distinct
champions* is **26.20 → 26.00, unchanged.** The surviving optima are as far apart as ever; the runs
merely stopped disagreeing about **which** to return. Quoting only the over-runs figure would have
read as convergence of the optima. It isn't.

**The failing leg is a set-size artifact, and the defect is mine.** The 10M set has 3 distinct
champions vs the 1M set's 6, and `max/min` over 3 draws is mechanically smaller. Drawing every
3-of-6 subset of the 1M pool: **10 of 20 give an `imbalance` ratio ≤ the 10M-observed 3.29x
(p=0.50)** — the 10M value is the *median* outcome of a 3-draw. My prereg put every *threshold* on
`range_g` for exactly this reason, then put the one magnitude leg on `ratio_g` anyway.
**Size-matched on `range_g` instead: 12 of 14 live gauges have 10M spread at or above the median
same-size 1M draw, 6 at the 100th percentile.** H-UNDER predicts the opposite. (The 2 below-median
gauges are the duplicated pair — spearman(`lsb`,`lsb-dist`) = **1.0000** measured here, not cited.)

**Convergence, by ARME-1's criterion (*has best-fitness stopped improving?*) — mixed, and it cuts
both ways:** 4 of 6 seeds stopped improving by ~1.8M evals (seed 900000 by **518,313**, half the
1M budget), but **2 of 6 were still improving past 5M**. So the 1M runs were *partly*
under-converged — a real point for H-UNDER. What defeats H-UNDER is **where that improvement
went**: onto the other seeds' already-known champions.

**The three survivors** are within **0.1236 ms/char = 2.00x** arm B's own noise sd (0.0617;
SPEEDTIE-1 registered 2.85x at 1M) while still spanning **5.92x on oxey-style**, **3.29x on
imbalance**, **2.88x on scissor**, with **0 dominance and zero ties in 84 cells**. They remain
inside SPEEDTIE-1's registered "within 2x sd → choose on the gauge frame" band at ~8.4M evals.

## 4. Why I still call it INDETERMINATE

1. `R_speed = 0.7023` never reached my registered ≤0.50 asymmetric-case threshold. Reading that
   clause as fired would be moving the line after seeing the data.
2. **n_distinct = 3 is too thin to bank a spread claim on.** With 3 champions `ratio_g` is
   demonstrably at the mercy of which 3 survive — that is my own §3 finding, and it cuts against
   my preferred reading as much as against the rule.

**What would settle it, as a number:** the blocker is *distinct champions*, not evals per run.
**n = 16 seeds** (same formula, r = 0..15) at **≥ 9.5M achieved** unique evals each → ~9–10
distinct champions at the observed ~60% survival rate, enough that `ratio_g` is no longer
size-limited. That needs **epochs ≈ 135**, not 120 (my schedule tops out at 7.8–9.2M). Cost ~4.5h
serial / ~1h at 5-way parallelism. **Pre-register the magnitude leg on `range_g` with a
size-matched subset placebo** — the specific fix for the defect above.

## 5. Controls (first four ran before any result was read)

Cold start **read** at `search.py:318-323`, not assumed · worktree isolation **positive**
(`FastEval.corpus_dir` → `/tmp/speedtie/...`), not merely no hardcodes · all six frozen 1M
champions reproduce, worst **2.814e-12**, arm B **exactly 0.0** · my analysis code reproduces
**every** frozen SPEEDTIE-1 number (13 spreads to 4.5e-5; better/worse 7/7, 4/10, 9/5, 9/5, 8/6
exactly), re-verified after a refactor · **the 10M run reproduces the 1M trajectory** — seed
900000's log at epoch 9 shows `unique=1,008,758` and the 1M placebo's exact champion (unplanned,
and the strongest confirmation the design worked) · ms/char code path **0 diff lines** vs the
commit the 1M run used · `sfr` invariance tested **directly** (1 distinct value), excluded
everywhere.

## 5b. HOSTILE SELF-AUDIT (reflection pass, before reaping) — branch `speedtie-budget` @ `b36b8d8`

Four findings, all of which cut against my own writeup. Worktree verified clean (`git status
--short` empty) at tip `b36b8d8`; the 3 commits are `40ff53c` → `23584ea` → `b36b8d8`.

**(a) WEAKEST CLAIM — the size-matched test in §3 is PARTLY CIRCULAR, and it is the one I leaned
on hardest.** I wrote "12 of 14 live gauges have 10M spread at or above the median same-size 1M
draw" as the statistic that "discriminates" against H-UNDER. But **all 3 of the 10M distinct
champions are members of the 1M pool** I drew the placebo subsets from — verified: the 10M triple
is *literally one of* the `C(5,3)` subsets it is being compared against. So the statistic answers
*"is the selected triple a typical triple of this pool?"* — **not** *"did the extra budget shrink
the spread?"* It cannot fall far below its own median by construction. The count is numerically
robust to pool choice (12/14 with the 6-champion pool, 12/14 with the paired-5 pool), so the
*number* stands; the *inference* is weaker than I stated. **What overturns it:** a pool the 10M
champions are NOT drawn from — e.g. 16 fresh seeds at ≥9.5M (my §4 proposal) scored against the 1M
pool, or the ~5,120-layout final populations rather than champions only. A successor should treat
§3 as *consistent with* H-REAL, not as discriminating evidence. **Note this does NOT rescue
H-UNDER:** the two non-circular legs — `M_gauge = 1.0000` (8 of 14 gauges *exactly* unchanged) and
Hamming-over-distinct 26.20 → 26.00 — are computed on the 10M champions directly and still fail
every H-UNDER prediction.

**(b) A NUMBER VERIFIED ONLY VIA A SHARED COMPONENT.** In this pass I cross-checked the driver's
search fitness against the shipped CLI's ms/char for all six champions: worst |diff| **2.98e-12**.
That looks like independent corroboration and **is not** — `evobj.py:306-308` does
`from keybo.analysis.timecard import TimeSurface` and builds `S = surface._T2 + surface._Tc`, i.e.
**the same class the CLI's time card uses, on the same corpus.** So it is a *consistency* check
that shares the component under test (traps 45/27). Consequence: every ms/char figure I publish —
253.9006, the 0.1236 range, the "2.00x sd" — rests on **one** timing implementation. The
2.814e-12 frozen-champion reproduction has the same property (same code, different commit). What
is genuinely multi-path: `unique_evals` (run JSON + ckpt `n_unique` + independent log trace, all
six seeds agree) and the 1M gauge spreads (my code vs the frozen SPEEDTIE-1 table, 4.5e-5).

**(c) CONTROLS THAT RAN AFTER I USED THE RESULT — three, and I should say so plainly.** The
size-matched subset placebo, the `spearman(lsb, lsb-dist) = 1.0000` measurement used to discount
the two below-median gauges, and the driver-vs-CLI cross-check all ran **after** the verdict was
computed. Only the frozen-number reproduction, the cold-start read, the corpus md5 and the
isolation check were pre-result. **The test I should have written and did not:** a size-matched
subset placebo *inside the pre-registration* — my own §4b guard identified `ratio_g` as the
unstable statistic and I put the one absolute-magnitude leg on it anyway. Also absent: any unit
test on `analyze_budget.py` itself; I caught the dict-collapse defect by noticing duplicate
champion strings by eye, not by a test, and a test would have caught it before it could bias the
verdict toward H-UNDER.

**(d) HAZARDS I AM LEAVING — two, and the first is the exact defect I criticised.**
1. **My own drivers hardcode absolute paths**, including `WORKTREE = Path("/tmp/speedtie")` in
   `run_budget.py:35` and `analyze_budget.py:26`, plus `OUTDIR`/`STATE`/`RUNS`/`FROZEN_1M` under
   `state/speedtie/`. This is trap 35 — the very thing I flagged in `search_placebo.py`'s
   `cwd="/tmp/optev"`. **A successor who copies `run_budget.py` into a fresh worktree will launch
   the search in `/tmp/speedtie`, which will not exist.** Fix before reuse: make `WORKTREE` an
   argument or derive it from `git rev-parse --show-toplevel`.
2. **My branch resurrects a deliberately-deleted module.** `git diff 45ea276 b36b8d8 -- src/ tests/`
   is **+912 lines, one file: `src/keybo/analysis/evidence_scorer.py`**, committed in `40ff53c`
   because `evobj.py:42` imports `LIVE_GAUGES` from it. There is **no test file** covering it at
   this HEAD. So **cherry-picking or merging `speedtie-budget` silently un-deletes that module** —
   do not merge this branch to `main` expecting artifacts only. Take the `agent-artifacts/`
   commits and leave the `src/` change, or re-restore it locally per the index's recipe.
3. Already documented above but restated as a hazard: the `.keys.npy` sidecars are gone, so
   `--resume` on these exact runs is impossible (a fresh run from the same seed is deterministic
   and equivalent). Provenance for every artifact is recorded in
   `artifacts/profiles-and-artifacts-index.md` next to the files, with md5s.

## 6. Things a successor should know

- **`search_placebo.py` is the wrong runner for any new budget** — it hardcodes `cwd="/tmp/optev"`
  (another agent's worktree, a *different* commit), a 3600s subprocess timeout, a write path into
  another workspace's state dir, and islands/epochs. Trap 35 wearing a subprocess's clothes.
- **`evobj.py:42` imports `keybo.analysis.evidence_scorer`, which is DELETED at ledger HEAD
  `45ea276`.** The driver cannot import without restoring it (byte-identical copy kept in
  `artifacts/drivers/evidence_scorer.RESTORED.py`, md5 `01f3a95a`).
- **Three CLI shapes that silently corrupt numbers:** ms/char is `row["time"]["ms_per_char"]` (the
  top-level key doesn't exist and yields **None** for every layout); `blob["gauge_frame"]` is a
  **description string**, not a list of names; and the CLI emits an extra `--ref` row, so
  `len(rows) == len(specs)` is a self-inflicted trap-38 false positive.
- **A defect in my own analysis code, worth generalizing:** I keyed profiles on the layout string.
  At 10M several seeds return the **same** champion, so that silently collapsed 6 runs into 4
  entries and would have computed every spread over the wrong n — **biasing toward H-UNDER**, i.e.
  toward a false convergence verdict. Trap 38's shape (a collection keyed on a lossy form) in a
  new dress. Index by **run**, and report `n_runs` and `n_distinct` separately.
- **A spread statistic needs a size-matched placebo when the number of items can change.** This is
  trap 17/32's logic (a same-size placebo) applied to a `max/min` ratio, and it is the one thing I
  would add to the pre-registration if I ran this again.
- **The `.keys.npy` dedup sidecars were deleted** (388 MB, no analytical content; `unique_evals` is
  triply recorded in the run JSON, the ckpt's `n_unique`, and the log trace — verified to agree for
  all six seeds first). Consequence: `--resume` on these exact runs is no longer possible; a fresh
  run from the same seed is (deterministic given seed+islands).
