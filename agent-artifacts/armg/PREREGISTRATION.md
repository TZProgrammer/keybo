# ARM G — PRE-REGISTRATION

**Written and committed BEFORE any search has been launched.** Nothing in this file is
informed by an ARM G search result, because none exists at commit time. Everything numeric
below is either (a) a frozen figure from a prior arm's artifact that I re-derived myself in
this worktree, or (b) a design constant I am choosing now.

**MODELLED ONLY.** Every number here is on the g-frame (geometry-only), a **baked 90 WPM**
fitted timing surface, corpus **blend-v1**, skipgrams `1-skip31.txt`. Nothing ARM G produces
is a claim about realized human typing speed. No layout will be adopted or recommended.

---

## 0. Pre-flight, run BEFORE this file was written (the discipline that gates everything)

`keybo/testkit.py` is **NOT on this branch** (`optimize-arm-g` @ 66d0715) — it is on
`qap-audit`. I applied the discipline manually instead.

| Check | Result | Verdict |
|---|---|---|
| `keybo.__file__` | `/tmp/armg/src/keybo/__init__.py` | 🟢 inside my worktree |
| `sys.prefix` | `/tmp/armg/.venv` | 🟢 own venv, not the shared clone's |
| `production_corpus_dir(None)` | `/tmp/armg/data/corpus/blend-v1` | 🟢 inside my worktree |
| `FastEval.corpus_dir` | `/tmp/armg/data/corpus/blend-v1` | 🟢 **positive** control (trap 35), not "no hardcodes found" |
| blend-v1 `trigrams.txt` md5 | `c5066fa7bcc46dea1ecbc987fb465b4a` | 🟢 == trap-8 reference |
| arm B through **shipped** `keybo analyze --json` | `253.90057910352797` vs frozen `253.90057910352604` | 🟢 diff **1.93e-12** |

**Driver dependency, declared:** `evobj.py:42` imports `keybo.analysis.evidence_scorer`,
which is **deleted at my base commit**. I restored it byte-identically from
`state/speedtie/artifacts/drivers/evidence_scorer.RESTORED.py` (md5 `01f3a95ab7a0f53f8f9d5be057fc437e`,
re-verified after copy). **Zero shipped files import it** (`grep -rn evidence_scorer src/ tests/`
= 0 hits), so restoring it changes no shipped scoring path.

**Repointed hardcoded paths (traps 35/65), stated explicitly.** I reuse `search.py` and
`evobj.py` from `state/speedtie/artifacts/drivers/` **unmodified** (md5 `2e499152489dbdc7e7f6c1a69a8c71a8`
and `dc45ef503792576157a872a996d9e9d7`). I do **not** reuse `search_placebo.py`
(hardcodes `cwd="/tmp/optev"`, a 3600 s timeout, and a write path into another workspace) or
`run_budget.py`/`analyze_budget.py` (hardcode `WORKTREE=/tmp/speedtie`). My own runner
hardcodes `/tmp/armg` and writes only into my own tree and my own state dir.

**Cross-path positive control — the gate on this whole design.** ARM G searches a
`FastEval` objective and judges on **shipped** `analyze` gauges. If those two paths disagree,
every number I report is a cross-path artifact (trap 13). Measured over 7 layouts × 15 axes:
worst relative disagreement **1.233e-14**; **10 of 15 gauges bit-exact (0.0)**; the five
non-exact are `sfb-dist` 7.0e-16, `sfs-dist` 6.7e-16, `oxey-style` 8.3e-16, `_ms_per_char`
1.2e-14. **Mutation-controlled:** planting a `*1.000000001` factor on the FastEval side makes
the harness exit **rc=1**; removing it returns **rc=0**. A check that cannot fail tests
nothing, so I verified this one bites.

**Frozen-champion reproduction (my own run, not inherited):** all six SPEEDTIE-1 1M champions
re-score through my shipped CLI to worst |Δ| **2.814e-12** (arm B 1.93e-12). This independently
matches SPEEDTIE-BUDGET-1's reported worst diff of 2.814e-12.

**Convention declared:** my base does **not** carry the OXEYFIX-1 repair (`oxey.py` last
touched by `c37a080`; no `_v1_pattern` delegation, `bad_redirect` still nested inside
`redirect`, so a bad redirect is charged +6.0 not +4.0 per the PENALTYAUDIT-1 CORRECTION).
So my `oxey-style` is the **as-shipped nested convention** — which is *the same convention*
SPEEDTIE-1's frozen 14.05× spread was measured on. Comparability with the figure I am
targeting is therefore preserved; comparability with a post-OXEYFIX board is **not**, and I
will not quote one.

---

## 1. Which question is primary — and why I am choosing (2) over (1)

The brief offers two: **(1)** beat arm B on predicted time, or **(2)** tie arm B on speed and
win on the gauge frame. **I pre-register (2) as PRIMARY and (1) as a co-recorded secondary.**

Reasons, as numbers:

1. **(1) is the question four arms have already answered NO.** Arms A, D, E and modelnorm all
   came in slower than arm B (256.85, worse-than-qwerty, 258.18/267.61, 256.63). The
   *baseline* objective *is* the thing that produced 253.9006.
2. **Re-running arm B's own objective is now known to buy nothing.** SPEEDTIE-BUDGET-1 showed
   the ~8.4M champion set is a **strict subset** of the 1M set (`s10 - s1` is empty — zero new
   layouts) and 3 of 5 seeds independently rediscovered arm B. Arm B is the champion of **four**
   independent cold starts. A fifth cold start of the identical objective is the one experiment
   the campaign has most thoroughly established as saturated.
3. **(2) is unexplored and the headroom is measured.** I re-derived it myself: the six frozen
   champions span **14.05×** on `oxey-style` (1.0078 → 14.1613) inside a **0.1760 ms/char**
   speed range, and **arm B is rank 5 of 6** on `oxey-style` (8.6110). SPEEDTIE-1 registered
   the selection rule but explicitly noted nobody has ever run a search that *deliberately
   collects* this. That is the lever.
4. **The brief itself says so**, and I agree with its reasoning rather than deferring to it:
   a selection rule applied to whichever seeds happened to run is strictly weaker than an
   objective that optimizes the quantity the rule selects on.

**Objective, stated as an equation.** ARM G minimizes a **lexicographic-by-penalty**
objective over the 30 movable C30M characters (space pinned to slot 30):

```
F(L) = D(L)  +  LAMBDA * max(0, ms(L) - (ARMB_MS + EPS))^2 / EPS^2
```

where

- `ms(L)` = predicted ms/char, served K31 surface, baked 90 WPM, blend-v1 — **the identical
  quantity arm B's own objective minimizes**, computed by the identical code path
  (`FastEval.gauges(...)["_ms_per_char"]`).
- `ARMB_MS = 253.90057910352604` (arm B, frozen; re-verified in my tree to 1.93e-12).
- **`EPS = 0.1234`** ms/char `= 2 x 0.0617`, i.e. **exactly SPEEDTIE-1's registered 2×-sd
  band**, using the *borrowed* baseline-objective sd. ⚠ This is a **design constant for the
  SEARCH**, not a ruler for the VERDICT. Per the standing (POOL × REPLICATE-STRUCTURE × SCALE
  × STATISTIC) rule I may not judge against a borrowed floor, and I will not: **§4 judges
  against a sd I measure from my own seeds.** I need *some* number to define the feasible
  region before any of my seeds exist, and the pre-registered band of the very finding I am
  testing is the only defensible choice. If my own measured sd turns out larger than 0.0617,
  my search will have been *stricter* than my verdict rule — the conservative direction.
- `LAMBDA = 1000.0`, with the quadratic normalized by `EPS^2` so one full `EPS` of speed
  violation costs 1000 D-units against a D scale whose entire 14-gauge range is ~14. A
  violation therefore dominates any achievable gauge gain — this is trap 51's lesson applied
  in advance: **a maximizer does not read flags, so the speed band must be a hard-in-effect
  constraint, not an advisory one.**
- **`D(L)` = the dominance deficit against arm B**, the quantity that is zero exactly when a
  layout is no worse than arm B on all 14 live gauges:

```
D(L) = SUM over the 14 live gauges of  max(0,  dir_g * (g(L) - g(ARMB)) / s_g )
```

**Gauge frame (pre-declared, 14 axes).** `LIVE_GAUGES` = sfb, sfs, sfb-dist, sfs-dist, lsb,
lsb-dist, alt, roll, sr-roll, redir, scissor, imbalance, oxey-style, comfort. **`sfr` is
excluded** — it counts doubled letters and is a **permutation invariant** (trap 23), so it is
a tie by construction on every pair and cannot be earned.

**Directions `dir_g` — DERIVED, not assumed (trap 5).** I verified the shipped `EXPECTED_SIGN`
table two independent ways rather than trusting it: (a) rank-correlation of each gauge with
predicted ms/char over 4000 random permutations agrees on **13/14** (sole disagreement `sfs`
at rho −0.0157 ≈ 0 — *exactly* the disagreement ARME-1 independently reported); (b) the
qwerty-is-worst reference point agrees on **14/14**. I adopt: `+1` (lower better) for sfb,
sfs, sfb-dist, sfs-dist, lsb, lsb-dist, redir, scissor, imbalance, oxey-style, comfort; `−1`
(higher better) for alt, roll, sr-roll.

**Scale `s_g` — pool-matched, and this choice is load-bearing.** `s_g` = the range of gauge
`g` **across the six frozen 1M champions**. That pool is (i) near-optimal, (ii) the same
KIND of object my champions will be, and (iii) the exact pool whose spread SPEEDTIE-1
measured. Using a *random*-permutation range instead would be a Simpson artifact (trap 26)
and would make every near-optimal difference look negligible. Frozen values, computed now:

| gauge | arm B | s_g | | gauge | arm B | s_g |
|---|---|---|---|---|---|---|
| sfb | 2.5391 | 0.7619 | | sr-roll | 17.8131 | 7.4726 |
| sfs | 6.7995 | 3.7068 | | redir | 4.4206 | 1.0305 |
| sfb-dist | 3.0423 | 0.8995 | | scissor | 0.2567 | 0.1885 |
| sfs-dist | 8.0056 | 4.4089 | | imbalance | 4.8754 | 4.5999 |
| lsb | 1.1411 | 0.9720 | | oxey-style | 8.6110 | 13.1534 |
| lsb-dist | 2.3227 | 2.2700 | | comfort | 3.4140 | 0.6334 |
| alt | 37.1373 | 8.2825 | | | | |

**`D` is NOT a multi-gauge aggregate for ranking.** GEOMEAN-1 registered DO-NOT-SHIP on
aggregating gauges into a score, and I am not doing that: `D` is a **penalty that is zero iff
Pareto-non-inferiority holds**, used as a search gradient toward the feasible set. Every
verdict in §4 is adjudicated **per-axis**, never on `D`. `D` orders only my own candidates
inside a pre-declared speed band; it never compares two layouts' quality.

**Positive control on D, already run:** `D(arm B) == 0.0` exactly, by construction. Existing
layouts, computed now (this is the bar ARM G must beat):

| layout | ms/char | D | worse | better | tie |
|---|---|---|---|---|---|
| arm B | 253.9006 | **0.0000** | 0 | 0 | 14 |
| s939595 | 253.9827 | 3.4619 | 6 | 8 | 0 |
| s923757 | 254.0056 | 2.8723 | 5 | 9 | 0 |
| s915838 | 254.0242 | 6.0966 | 10 | 4 | 0 |
| s931676 | 254.0517 | 3.9787 | 5 | 9 | 0 |
| s907919 | 254.0766 | 2.4945 | 7 | 7 | 0 |
| keybo-lsb | 254.6307 | 2.1317 | 5 | 9 | 0 |
| keybo-lsb+lm | 254.6847 | 1.9092 | 5 | 9 | 0 |
| flagship-c3 | 254.9761 | 1.4878 | 4 | 10 | 0 |
| arm-A | 256.8466 | 0.4533 | 1 | 13 | 0 |
| graphite | 258.1696 | 3.2226 | 6 | 8 | 0 |

**No existing layout has `D = 0` except arm B itself.** (arm A's D = 0.4533 is low but it is
2.95 ms/char slower — far outside any speed band.) So `D = 0` inside the band is a genuinely
unoccupied cell, and a **feasibility question with a real chance of NO**.

⚠ **Known relaxation bound, stated in advance so I cannot claim surprise.** Arm B is
**best-of-six** on 4 of the 14 gauges (`sfs`, `sfs-dist`, `roll`, `sr-roll`). A `D = 0`
layout must therefore *match or beat* arm B on axes where no sibling managed to — the six
frozen champions give a per-axis lower bound of `D >= 0` only if those four can be held. This
is why I register `D = 0` as **plausible but not expected**, and why the primary verdict
below is graded, not binary.

---

## 2. Seeds, budget, and what I will report

- **n = 5 seeds** (exceeds the registered `n >= 3` minimum; arms A–D were all n=1 and their
  gap sizes are retracted). Seed formula, fixed now: **`seed(r) = 20260728 + 7919*r` for
  r = 0..4** → `20260728, 20268647, 20276566, 20284485, 20292404`. Deliberately **disjoint**
  from the `900000 + 7919*r` placebo family so my draws are independent of the six frozen
  champions rather than a re-run of them.
- **Budget: 1,000,000 unique evals/seed, islands=20, epochs=12, overshoot=1.95,
  ga-share=0.6, polish-sweeps=40** — the **identical engine configuration** as the 1M placebo
  whose sd I borrow for `EPS`, so my ARM G runs are configuration-matched to the reference
  band. Justified by SPEEDTIE-BUDGET-1's central mechanism result: ~7.4M extra evals bought
  **zero new territory** (strict-subset champion set), so 1M is the right budget and a
  larger one is the experiment already shown to be saturated.
- **A BASELINE CONTROL ARM RUNS TOO — same 5 seeds, same budget, unmodified `--arm baseline`.**
  This is essential and non-optional: it is (a) how I **measure my own search-noise sd** on
  *my own* seeds rather than borrowing one, and (b) the same-size, same-seed placebo that
  makes any ARM G gauge gain attributable to **the objective** rather than to the draw
  (traps 17 / 32 / 34: a nested-frame or count change needs a same-size placebo). Without it
  I would be comparing 5 new seeds against 6 old ones and calling the difference an effect.
- **`unique_evals` will be reported ACHIEVED, never requested.** The engine stops on the
  epoch schedule, so a run can fall short and still exit 0. **Pre-registered floor: a seed
  achieving < 80% of 1,000,000 unique evals is excluded from the primary n and reported as
  excluded**, with a sensitivity analysis that includes it. (SPEEDTIE-BUDGET-1's seed 931676
  hit 77.9% and was excluded; its sensitivity analysis returned the same verdict.)
- **Distinct-champion count reported alongside n_runs** (trap: keying a per-run collection on
  the RESULT silently collapses n — SPEEDTIE-BUDGET-1's trap 2). I will report `n_runs` and
  `n_distinct` side by side, and Hamming **both** over runs and over distinct champions,
  because those two diverge and either alone supports the opposite reading.

---

## 3. Predictions, registered now (so they can be scored against, and can fail)

| # | Prediction | Refuted by |
|---|---|---|
| P1 | ARM G finds ≥1 champion with `ms <= 253.9006 + EPS` **and** `D < 1.9092` (beats flagship-c3's and keybo-lsb+lm's D) in ≥3 of 5 seeds | fewer than 3 seeds do |
| P2 | ARM G does **NOT** achieve `D = 0` in any seed | any seed returns D = 0 exactly |
| P3 | ARM G's `oxey-style` beats arm B's 8.6110 in ≥4 of 5 seeds | fewer than 4 do |
| P4 | ARM G is **NOT faster** than arm B: its best `ms` > 253.9006 in 5 of 5 seeds | any seed lands below 253.9006 |
| P5 | My own measured baseline-control sd differs from the borrowed 0.0617 by ≥1.5× in either direction (a borrowed floor is a *different quadruple*: my seed family and n differ) | it lands within 1.5× |
| P6 | The baseline control reproduces a known champion (arm B or one of the six) in ≥1 of 5 seeds | none of the 5 do |

P2 and P4 are deliberately **self-adverse** — I am predicting my own arm fails to reach the
cleanest possible result. If P2 or P4 is refuted, that is a *better* outcome than predicted
and I will say so.

---

## 4. Decision rule — SPEEDTIE-1's, applied verbatim, and the FAILURE conditions

**Ruler I will judge against (measured, not borrowed).** `sd_G` = the sample sd (ddof=1) of
champion `ms/char` across my **own baseline-control** seeds. Its quadruple, printed beside
it as the standing rule requires: **POOL** = my 5 ARM-G-family seeds' baseline champions
(near-optimal, cold start); **REPLICATE-STRUCTURE** = independent cold-start search runs,
one champion each; **SCALE** = raw ms/char; **STATISTIC** = sd, ddof=1, n=5.

**Selection among my own champions** (SPEEDTIE-1's registered rule, verbatim): among ARM G
champions whose predicted time is within **2 × sd_G** of the best, the winner is chosen on
the **pre-declared gauge frame** (lowest `D`, ties broken by count of strictly-better axes),
**never on the objective**.

**Primary verdict, graded, decided in this order:**

- **FASTER** — an ARM G or control champion with `ms < 253.9006 - 2*sd_G`. (Answers the
  brief's Q1 YES.)
- **TIED-AND-STRICTLY-BETTER** — a champion with `|ms - 253.9006| <= 2*sd_G` **and** `D = 0`
  **and** ≥1 strictly-better axis. (Answers Q2 YES: a genuine Pareto dominator inside the
  speed tie. Requires the strict-win term — trap 33: a `>=`-only predicate credits ties as
  wins, a defect now found at four independent sites.)
- **TIED-AND-PARTIALLY-BETTER** — in-band, `D > 0`, but `D` strictly below **every** existing
  layout's D and the per-axis better/worse count favours ARM G. Reported as a *partial* win
  with the un-collected axes named.
- **NEITHER / FAILURE** — see below.

**I will report FAILURE if any of these holds** (an arm that cannot fail is not an experiment):

1. **No ARM G champion lands inside the band at all** (`min ms > 253.9006 + 2*sd_G`): the
   penalty formulation failed to hold the speed constraint, so the objective is mis-specified
   — the same failure mode arms D/E hit from the other direction.
2. **In-band but `D >= 1.4878`** (flagship-c3's D, the lowest of any existing layout other
   than arm A): ARM G would have bought no more non-inferiority than a layout the campaign
   already has, so the deliberate collection achieved nothing.
3. **`sd_G` is so large that arm B and every ARM G champion are mutually indistinguishable
   AND `D` shows no ordering** — i.e. the free lunch is inside the noise, so SPEEDTIE-1's
   headroom is not collectable by search. This would be a **direct refutation of my own
   premise** and I will register it as one.
4. **Fewer than 3 seeds clear the 80% `unique_evals` floor**, leaving me below the registered
   `n >= 3` minimum.

**What I will NOT claim, pre-committed:**
- No adoption, no recommendation to adopt, no push, no CR, no edit to `PREREGISTRATIONS.md`.
- No **bare `n/15`** win counts. Per ULTRAAUDIT-INTERIM, `alt` and `imbalance` are
  **hand-partition invariants** and `sfr` is a permutation invariant, so some pairs tie **by
  construction**. I will report **per-pair CONTESTED axis counts** with the tie cells named.
- No multiplier for `scissor`'s mispricing — **direction only** (under-priced relative to
  `sfb`), per SCISSORPRICE-1's unsettled level.
- No claim that any gauge difference is **perceptible**. SPEEDTIE-1's caveat binds: the
  differences are **free**, not necessarily felt.
- No re-use of another arm's resolution floor for my own verdicts.

---

## 5. Self-separation plan (committed before results exist)

After the FIND pass I will re-read my own output as a hostile stranger and, for each claim,
name: (i) what would refute it; (ii) whether my check **shares a component** with the target
(trap 45 — a difference statistic cannot test a shared component; and the SELF-AUDIT SWEEP
found two "independent" controls that shared the component under test); (iii) whether any
control ran only **after** I had used its result. I will report what I **killed** as well as
what survived.

Two shared-component risks I can already name, and how each is handled:
- **`oxey-style` is R² = 0.9937 on {sfb, lsb, scissor, imbalance, redir, alt}** (trap 27). So
  `oxey-style` is **not** independent evidence alongside those six — it restates them. My `D`
  therefore **over-weights that cluster by construction**, and I will report a
  cluster-corrected reading (count wins per correlation cluster, not per gauge — trap 39,
  which reversed a "broad competence" verdict when the ~4× over-count was corrected)
  alongside the raw one.
- **`lsb` and `lsb-dist` are near-duplicates** (spearman 1.0000 measured by a sibling), so
  they are one axis wearing two names in any leave-one-out (trap 25).

---

## 6. Provenance of every borrowed number in this file

| Number | Value | Source | Re-derived by me? |
|---|---|---|---|
| arm B ms/char | 253.90057910352604 | ARM-B / SPEEDTIE-1 | 🟢 yes, 1.93e-12 |
| baseline search-noise sd | 0.0617 | `optevidence/artifacts/search-noise-placebo.json` `bands.baseline` | 🟢 read from artifact; **used for EPS only, not for any verdict** |
| six frozen 1M champions | 6 layouts | same artifact, `runs.baseline` | 🟢 all six re-scored, worst 2.814e-12 |
| oxey-style spread across six | 14.05× | SPEEDTIE-1 | 🟢 yes, independently 14.05× |
| keybo-lsb / +lm / flagship-c3 / arm-A / graphite | 5 strings | campaign artifacts | 🟢 all re-scored in my tree |
| `alt`/`imbalance` invariance | hand-partition | ULTRAAUDIT-INTERIM | ⚠ accepted from the ledger; not re-derived (I only *exclude* on it, which is the conservative direction) |

**Engine config source:** islands=20 / epochs=12 / overshoot=1.95 / ga-share=0.6 /
polish-sweeps=40 read from `search_placebo.py` (the driver that produced the reference band),
not from a docstring.
