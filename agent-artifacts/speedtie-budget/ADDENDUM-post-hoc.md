# SPEEDTIE-BUDGET — POST-HOC ADDENDUM

⚠ **Everything in this file was computed AFTER reading the result.** The pre-registered verdict
is **INDETERMINATE** and it stands — see `PREREGISTRATION.md` §4 and
`state/speedtie/artifacts/speedtie-budget-10000000.json` → `VERDICT`. Nothing here overturns it.
This file exists because the pre-registered rule failed on **one leg** for a reason the rule did
not anticipate, and that reason is itself measurable.

---

## 1. WHAT THE PRE-REGISTERED RULE SAID, AND EXACTLY WHY IT DID NOT FIRE

Primary analysis, n=5 paired seeds (seed 931676 excluded: 7,787,578 evals = 77.9%, below the
pre-registered 80% floor; a sensitivity analysis including it gives the same verdict):

| statistic | pre-registered threshold | observed | fires? |
|---|---|---|---|
| `R_speed` (range) | H-UNDER needs ≤ 0.50 | **0.7023** (0.1760 → 0.1236) | ✗ |
| `M_gauge` (median per-gauge range ratio) | H-UNDER ≤ 0.50 / H-REAL ≥ 0.80 | **1.0000** | ✗ H-UNDER / ✓ H-REAL |
| mean Hamming ratio | H-UNDER needs ≤ 0.75 | **0.7328** (26.20 → 19.20) | ✓ |
| gauges with `ratio_g` ≥ 5.0 at 10M | H-REAL needs ≥ 2 | **1** (oxey-style 5.92x) | ✗ |
| dominating pairs | H-REAL needs 0 | **0** of 6 ordered pairs | ✓ |

- **H-UNDER requires all three** of its legs. Only one fired. H-UNDER is **not** supported.
- **H-REAL** fired on two of three legs and failed on exactly one: the "still large in absolute
  terms" leg, which needed ≥2 gauges at `ratio_g` ≥ 5.0 and got 1, **solely because
  `imbalance`'s ratio fell 17.70x → 3.29x**.

`M_gauge = 1.0000` is worth stating plainly: **8 of 14 live gauges have a range ratio of exactly
1.0000** (sfs, sfb-dist, sfs-dist, alt, roll, sr-roll, redir, comfort) — their additive spread did
not move at all — and a 9th, `sfb`, is 0.9968.

## 2. THE FAILING LEG IS A SET-SIZE ARTIFACT, NOT A BUDGET EFFECT

The 10M primary set has **3 distinct champions**; the 1M set has 5 (6 over all seeds). A `max/min`
ratio over 3 draws is mechanically smaller than over 6. So I tested whether the `imbalance`
collapse is a *budget* effect or a *set-size* effect, by drawing every same-size subset of the
1M champion pool — the placebo the pre-registered rule should have had:

| gauge | 1M all-6 ratio | 10M observed | 1M 3-subsets (n=20): min / median / max | P(1M 3-subset ≤ 10M observed) |
|---|---|---|---|---|
| imbalance | 17.70x | **3.29x** | 1.00 / **5.35** / 17.70 | **0.50** |
| oxey-style | 14.05x | **5.92x** | 2.11 / **5.92** / 14.05 | **0.55** |
| scissor | 3.76x | **2.88x** | 1.37 / 2.08 / 3.76 | 0.80 |

The 10M `imbalance` ratio is **the median outcome of drawing 3 champions from the 1M pool**
(p=0.50). The observed "collapse" is what a 3-draw *is*. The leg was measuring **how many
champions survived**, not **how far apart they are** — and my own §4b guard anticipated exactly
this hazard for `ratio_g` ("unstable when min approaches 0"), which is why every *threshold* in
the rule is on `range_g`. I put the one absolute-magnitude leg on `ratio_g` anyway. That is the
defect in my pre-registration, and it is mine, not the data's.

## 3. THE SIZE-MATCHED TEST, WHICH IS THE ONE THAT DISCRIMINATES

Compare the 10M per-gauge **range** (additive, stable) against the distribution of ranges over
all 20 same-size (3-of-6) subsets of the 1M pool. **H-UNDER predicts the 10M spread sits BELOW
that distribution** (extra search collapsing the profiles). It does not:

| | count |
|---|---|
| live gauges where 10M range ≥ the size-matched 1M **median** | **12 of 14** |
| live gauges at the **100th percentile** (10M range = the max attainable from any 1M 3-subset) | **6** (sfs, sfb-dist, sfs-dist, alt, roll, redir) |
| live gauges below the median | 2 (lsb 40th, lsb-dist 40th) — and these two are the correlation-duplicated pair, which I **measured on this data rather than citing**: spearman(lsb, lsb-dist) = **1.0000** on the six 1M champions (pearson 0.9987) and **1.0000** on the three 10M survivors. So this is **one** fact, not two (trap 25). |

## 4. THE MECHANISM: THE EXTRA ~7.4M EVALUATIONS BOUGHT NO NEW TERRITORY

Run-for-run, which is what the seed-matching was for:

| seed | 1M → 10M | Hamming 1M→10M |
|---|---|---|
| 900000 | **kept its own** champion | 0 |
| 915838 | **kept its own** champion | 0 |
| 907919 | moved **onto seed 939595's 1M champion** | 19 |
| 923757 | moved **onto seed 900000's 1M champion** (= arm B) | 24 |
| 939595 | moved **onto seed 900000's 1M champion** (= arm B) | 29 |
| 931676 *(sub-floor, excluded)* | moved to a **7/30 perturbation of its own** 1M champion | 7 |

**The n=5 primary 10M champion set is a strict SUBSET of the 1M champion set — zero new
layouts.** Over all six seeds exactly one layout appears that was not already in the 1M pool,
and it is 7 of 30 positions from that seed's own 1M champion, not a new region. The 1M pool
already contained every optimum an ~8.4M-eval search could find.

**The decisive dissociation.** Mean Hamming *over runs* falls 26.20 → 19.20 (0.73x) — but that
is **entirely** because 3 run-pairs became identical (`n_zero_pairs`: 0 → 3). Mean Hamming *over
distinct champions* is **26.20 → 26.00, essentially unchanged.** The surviving optima are just as
far apart as before; the runs merely stopped disagreeing about **which** of them to return.
Quoting only the over-runs number would have read as convergence of the optima. It is not.

## 5. CONVERGENCE, BY ARME-1'S REGISTERED CRITERION (has best-fitness STOPPED IMPROVING?)

| seed | last improvement | unique evals at that point | epochs flat after |
|---|---|---|---|
| 900000 | epoch 4 | **518,313** | 116 |
| 915838 | epoch 12 | 1,032,473 | 108 |
| 931676 | epoch 15 | 1,297,858 | 105 |
| 939595 | epoch 24 | 1,804,747 | 96 |
| 923757 | epoch 62 | 5,008,037 | 58 |
| 907919 | epoch 100 | 7,161,858 | 20 |

So the honest reading is **mixed, and it cuts both ways**: 4 of 6 seeds had stopped improving by
~1.8M evals (seed 900000 by ~518k, i.e. *half* the 1M budget), but **2 of 6 were still improving
past 5M** — so the 1M runs were *partly* under-converged, which is a real point for H-UNDER.
What defeats H-UNDER's prediction is **where that improvement went**: onto the other seeds'
already-known champions, not toward a common new optimum.

## 6. THE THREE SURVIVORS — the objective still cannot choose between them

| layout | ms/char |
|---|---|
| `flmpg-yuo,sntdcireahkxbwv'.jzq` (= arm B) | 253.9006 |
| `pyu.,gdfnlhieaocstrmkj'-qbwzvx` | 253.9827 |
| `pyou,vdflrghaeictsnmk'j.-wbzxq` | 254.0242 |

Speed range **0.1236 ms/char = 2.00x** arm B's own registered 6-seed search-noise sd of 0.0617
(SPEEDTIE-1 registered 2.85x at 1M). Still spanning **5.92x on oxey-style**, **3.29x on
imbalance**, **2.88x on scissor** — with **0 of 6 ordered pairs dominating** (all mixed;
better/worse 6/8, 10/4, 8/6, 10/4, 4/10, 4/10; **zero ties in 84 cells**).

**And note this survives SPEEDTIE-1's own registered selection rule verbatim:** "among champions
whose predicted time is within 2x that objective's OWN search-noise sd, the winner is chosen on
the pre-declared gauge frame." At 2.00x sd these three are inside that band at ~8.4M evals, and
the gauge frame still separates them. The free lunch is still on the table at the full budget.

## 7. WHAT I ACTUALLY CONCLUDE, AND WHAT WOULD SETTLE IT

**Registered verdict: INDETERMINATE** (the rule's own output; I do not restate it as a win).

**Post-hoc reading, stated as such:** the evidence **leans toward H-REAL and against H-UNDER**,
because every H-UNDER-specific prediction failed while the one failing H-REAL leg is explained
by a set-size artifact my rule did not control:
- H-UNDER predicts gauge spread shrinks → `M_gauge = 1.0000`, 8 of 14 gauges *exactly* unchanged (a 9th at 0.9968), 12 of 14 at-or-above a size-matched 1M draw. **Failed.**
- H-UNDER predicts champions converge toward each other → Hamming over *distinct* champions 26.20 → 26.00. **Failed.**
- H-UNDER predicts search slack resolves to better common optima → zero new layouts found; the 10M set is a subset of the 1M set. **Failed.**
- H-REAL predicts the objective still cannot break the tie → 0 dominating pairs, zero ties in 84 cells, three survivors within 2.00x sd. **Held.**

**Why I still call it INDETERMINATE rather than H-REAL (STRONG FORM).** Two honest reasons:
1. `R_speed = 0.7023` did not reach my ≤0.50 asymmetric-case threshold, so the *registered*
   strong-form clause genuinely did not fire. Reading it as fired would be moving the line after
   seeing the data.
2. **n_distinct = 3 is a thin base for a spread claim.** With 3 champions the `ratio_g` statistic
   is demonstrably at the mercy of which 3 survive (§2). I will not bank a spread verdict on it,
   even though the size-matched test points my way.

**What would settle it — stated as a number, per the brief.** The blocker is *distinct champions
at the full budget*, not evaluations per run. At ~8.4M evals only 3 of 5 seeds stay distinct, and
20 same-size subsets are not enough to place a spread. So:

> **Run n = 16 seeds (same formula, `900000 + 7919*r`, r = 0..15) at ≥ 9.5M ACHIEVED unique
> evals each.** At the observed coalescence rate (~60% of seeds distinct) that yields ~9-10
> distinct champions — enough that `ratio_g` is no longer size-limited, and enough to compare
> the 10M spread against a properly-sized 1M draw. Cost: 16 x ~1000s ≈ 4.5h serial, or ~1h at
> 5-way parallelism (measured: 1014s for 1 seed alone, 2322s for 5 in parallel). Reaching 9.5M
> *achieved* needs **epochs ≈ 135**, not 120 — my 120-epoch schedule tops out at 7.8-9.2M
> because the run stops on the epoch schedule, not on the unique-eval target.
> **Pre-register the absolute-magnitude leg on `range_g` with a size-matched subset placebo**,
> which is the specific fix for the defect in §2.

## 8. SCOPE

**MODELLED ONLY:** g-frame, baked 90 WPM, blend-v1, `skipgrams=1-skip31`. Nothing here is a
claim about realized human typing speed, and **no layout is adopted or recommended for
adoption.** One objective (baseline served), one corpus, n=5 primary / 6 total, and the "10M"
runs achieved **7.79M-9.22M** unique evals (mean 8.43M), not 10M — so this is a **~8.4x** budget
increase over the 1M placebo, not 10x. `sfr` excluded from every spread, ratio, dominance and
win-count as a permutation invariant, verified directly: **1 distinct value (2.659577102696)
across all champions at both budgets**, not via a variance threshold (trap 23).
