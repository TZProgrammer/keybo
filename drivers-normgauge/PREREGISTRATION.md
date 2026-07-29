# NORMGAUGE-1 — PRE-REGISTRATION

**Registered BEFORE any anchor, weight, or blend result exists.** Commit this file, then
compute. Every threshold, weighting rule, and success criterion below is fixed here so it
cannot be tuned to whichever layout wins.

Branch `normgauge` in worktree `/tmp/normgauge`. No push, no CR, no `PREREGISTRATIONS.md`
edit, no layout adopted or recommended.

---

## 0. FRAME AND UNITS — stated once, and every number below is on it

| Property | Value |
|---|---|
| Surfaces | the **SHIPPED** `data/surfaces/<POOL>_TRI_PS_FREQ_PRIOR.standardized.npy.gz`, loaded through `keybo.analysis.surfaces.load_surface` |
| Frame | geometry-only **`g`**; the layout-independent `b(ngram)` term is excluded |
| WPM | **BAKED at 90.0** and NOT re-evaluable (7 of 8 per-seed models are gone) |
| Corpus | production **`blend-v1`** (`data/corpus/blend-v1/trigrams.txt`) |
| Objective | `fit(L) = Σ_t F[t] · S[p(t₀), p(t₁), p(t₂)]` in predicted ms, **lower = faster** |
| Charset | C30M `qwertyuiopasdfghjkl'zxcvbnm,.-` + space at slot 30 |

**MODELLED ONLY.** Nothing here is a claim about realized typing speed.

### 0.1 The frame fact that constrains every interpretation below (measured in FIND)

`standardized − native` is **exactly independent of the third slot** (max variation over `c`:
AALTO 0.0, COMMUNITY 1.14e-13, POOL 1.14e-13), and is **identically 0 for AALTO**. So the
shipped standardization **substitutes AALTO's bigram tensor `T2` into all three sources**,
leaving each source only its own *conditional trigram increment*.

⚠ **Consequence, registered up front:** on the shipped frame the three "models" are LESS
independent than on `.native` — they already share a bigram tensor. MODELNORM-1 chose
`.native` for exactly this reason. I ship on `.standardized` anyway, because the user asked
for a gauge the shipped optimizer can use and the shipped resolver reads only
`.standardized`. **This is a real cost of shipping and I will report it, not bury it.**

---

## 1. THE GAUGES (deliverable 1)

Three normalized gauges `aalto-n`, `comm-n`, `pool-n`:

```
norm_m(L) = (zero_m − fit_m(L)) / (zero_m − one_m)
```

* `zero_m` = **mean** fit over **n = 100** uniformly random C30M permutations, pool seed
  **20260728**. (MODELNORM-1 verified n=100 is sufficient: n=1000 moves it <1 SE and the
  ranking is unchanged at n=100/1000/10000. **I do not silently inflate n.** I will report
  the n=1000 value as a *stability check only*, not as the shipped anchor.)
* `one_m` = the best fit found by a per-model search at a fixed, identical budget.
* `fit == zero_m → 0`; `fit == one_m → 1`; **higher normalized = better** (the numerator is
  inverted on purpose because `fit` is time).

### 1.1 DIRECTION GUARD — registered, and it is NOT qwerty

**The guard is: each model's own searched optimum normalizes to EXACTLY 1.0** (to <1e-12),
and the random pool's mean normalizes to EXACTLY 0.0 (to <1e-12).

⚠ **`qwerty30m ≈ 0` is FALSE and must NOT be used as a guard.** MODELNORM-1 measured it at
`[0.5649, 0.4243, 0.5239]` because it sits at the 0.00–0.20 percentile of a random pool while
the scale's zero is the pool **MEAN**, not its floor. A correctly-signed implementation FAILS
a "qwerty ≈ 0" check, so that check would invert the sign. **I pre-register the prediction
that qwerty30m lands in 0.35–0.65 on all three gauges**, and treat a value near 0 as evidence
of a BUG.

### 1.2 Search budget, and the free positive control that sizes it

**AALTO's `.native` and `.standardized` arrays are byte-identical** (verified, max|d| = 0.0),
so MODELNORM-1's AALTO champion `lnfdg-,yehcrstmaoiupxqbwv.k'jz` is a **10M-unique-eval
optimum on exactly my frame**. I verified it rescores to `223236317224.4177` vs its recorded
`223236317224.4182` — **reldiff −2.3e-15**. That is a free end-to-end control on corpus +
loader + fit arithmetic + charset, obtained before computing anything.

**Registered budget:** per-model multi-restart steepest-descent + perturbation (memetic
island), **identical budget and structure across all three models**, ≥3 independent seeds
per model. Budget fixed at **40 islands × 5,000,000 unique evals requested per model**.

**Registered acceptance criterion for `one_AALTO`:** it must come within **+0.05%** of
223236317224.4177. A 32-sweep plain greedy from one random start already reaches +0.2263%,
so +0.05% is a real bar and not a formality. **If my AALTO search misses that bar, my search
is too weak and I report the anchors as a LOWER BOUND with the shortfall stated, rather than
quietly shipping a weak anchor.**

For COMMUNITY and POOL there is no external target (their frames differ), so their quality
evidence is **seed agreement**: I report the max across-seed spread as a % of each model's
own span.

**An optimizer output bounds the true optimum from ONE SIDE ONLY**, so every normalized
score is an **upper bound** on the true normalized score. Stated in the artifact.

### 1.3 Anchors persisted with provenance

A versioned JSON artifact carrying: frame (`standardized`), family, corpus name + trigram
file SHA256, per-surface SHA256, pool seed, n, statistic (`mean`), search budget
**requested AND achieved `unique_evals`**, islands, seeds, per-model champion layout, per-seed
fits, numpy version, and the pinned evaluator tile. **Anchors that cannot be reproduced are
not anchors.**

---

## 2. THE WEIGHTS (the heart of the task) — RULE REGISTERED BEFORE COMPUTATION

### 2.1 First, a constant in my own brief that I checked and must correct

My brief says **"AALTO has 7,669,316 in-frame samples vs COMMUNITY's 11,930 — a 643x
difference"** and calls it "the strongest single fact available about relative reliability."

**The conclusion is true. The constant is mis-scoped.** Traced to
`state/scissorsupport/artifacts/`:

| Scope | AALTO | COMMUNITY | ratio |
|---|---|---|---|
| `ss2d` scissor-neighbourhood, **covered-pair-filtered** | 7,669,316 | 11,930 | **642.9x** ← the brief's number |
| `ss2` same 6 groups, **unfiltered** | 7,669,316 | 151,365 | 50.7x |
| `ss2` **whole stroke table** totals | 18,535,823 | 401,543 | 46.2x |

⚠ **The filter is ASYMMETRIC: AALTO's count is IDENTICAL in both artifacts while COMMUNITY
loses 92.1% of its samples** (151,365 → 11,930). So 643x is *the ratio inside the scissor
neighbourhood after a per-pair coverage filter that only bit COMMUNITY* — it is not "AALTO
has 643x the data."

**Registered replacement, measured on the frame the gauge actually uses** (my own scan of
the training tables, filtered exactly as the recipe filters them — `wpm_lo=40`,
`min_cell_samples=10`, the same 4-label AALTO subset and 4-label rowStagger COMMUNITY subset
scissorsupport identified by exact practice-term key-set match):

| | AALTO | COMMUNITY | ratio |
|---|---|---|---|
| training samples landing on 31³ surface cells | 26,368,247 | 29,047 | **907.8x** |
| surface cells covered (of 29,791) | 5,219 | 1,044 | 5.0x |
| stroke rows kept | 8,620 | 1,327 | |

**I will use MY OWN measured numbers, on MY OWN stated scope, and I will not quote 643x as
the reliability ratio.** The qualitative fact the brief relies on — AALTO is far better
supported, COMMUNITY prices some cells off a handful of samples — is **confirmed and
strengthened** (907.8x on the surface-cell frame). Only the constant changes.

### 2.2 The four candidate rules, and what the evidence can actually identify

| | Rule | Identifiable here? |
|---|---|---|
| (a) | precision / sample-size weighting | **YES** — per-cell support measured (§2.1), and COMMUNITY's per-seed fit reliability is measurable (per-seed arrays exist for the BASE family) |
| (b) | independence correction (POOL is a union) | **YES, and quantifiably** — measured, not asserted |
| (c) | held-out predictive weighting | **YES and it is the strongest** — see §2.4 |
| (d) | equal weights | the fallback if (a)–(c) are not separable |

### 2.3 (b) The double-count is MEASURED, not asserted

At **fit** level (what the optimizer sees), over a 400-layout random pool on the shipped
frame:

```
POOL = 0.498757·AALTO + 0.508017·COMMUNITY + const     R² = 0.93881   coef sum 1.00677
resid sd = 24.74% of POOL's own fit sd
```

POOL is a **near-exactly symmetric 0.5/0.5 blend** of the other two, with 93.9% of its
fit variance explained by them. **So POOL is not an independent third vote — it is ~a
sample mean of the two.** (At cell level: `0.454530·AALTO + 0.449591·COMMUNITY`, R² 0.87400,
also symmetric. The two frames agree, which is why I trust the structure.)

**Registered consequence, decided before seeing any blend result:** including POOL as an
equal third vote gives the AALTO+COMMUNITY *consensus* **1.5× the weight** it would get from
the two sources alone and gives POOL's own 6% of unique variance a full vote. Equal weights
`(1/3, 1/3, 1/3)` therefore effectively implement about
`(0.417·AALTO, 0.417·COMMUNITY, 0.167·unique-POOL)`. **Equal weights are NOT neutral — this
is now a measured statement with a number attached.**

### 2.4 (c) HELD-OUT PREDICTIVE WEIGHTING — feasible WITHOUT refitting, and registered as primary

The key realization, verified in FIND: **the two sources are disjoint**, so each source's
data is **already out-of-sample** for the other's surface. No refit is needed.

* AALTO surface trained on `{azerty, dvorak, qwerty, qwertz}` (Aalto participants, pids
  <200000).
* COMMUNITY surface trained on 4 `@rowStagger` labels from 7 community submitters (pids
  200001–200007). Verified: `tristrokes_last_community.tsv` loads 11,084 rows with exactly
  7 distinct pids.

**Registered design:** for each ordered pair (surface `m`, held-out source `s ≠ m`), score
surface `m`'s predicted per-cell time against source `s`'s **observed** per-cell mean time,
using the **campaign's own** cell machinery (`keybo.training.validate.build_cells`,
`wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10`) and the campaign's own
bucket-centered Spearman (`_centered_spearman`). Normalize each rho by the held-out source's
**split-half reliability ceiling** (`split_half_ceiling`, participant-bisected,
Spearman-Brown length-corrected — the post-2026-07-28 convention) so a source with noisy
targets is not penalized for its own noise.

**Registered weighting rule (primary), fixed now:**

```
w_m ∝ max(0, rho_ceil_m)        where rho_ceil_m = cross-source rho / ceiling
```

restricted to the **two data-bearing sources AALTO and COMMUNITY**, with POOL's weight set
by §2.5. Rationale: predictive skill on *another population's held-out data* is the only one
of (a)–(d) that measures the thing the gauge is for — generalizing to typists we did not
train on.

⚠ **REGISTERED FALSIFIER, so this cannot be a rubber stamp:** if **either** cross-prediction
rho/ceiling is not statistically distinguishable from 0 (bootstrap 95% CI over participants
crosses 0), or if the two are **within 1 bootstrap SE of each other**, then **(c) is
UNDERPOWERED and I fall back to (a)+(b)** by the rule in §2.5 and say so. **A single
cross-prediction number is a weak reed and I am registering in advance that I expect it may
fail this bar** — n=7 community participants is very thin.

### 2.5 THE DECISION TREE, fully specified before results

```
STEP 1 — POOL's weight, from (b). REGISTERED UNCONDITIONALLY:
   POOL is a measured 0.5/0.5 blend with R²=0.939 at fit level, so it is NOT an
   independent vote. Its weight is set to its UNIQUE variance share:
       w_POOL_raw = 1 − R²(POOL ~ AALTO + COMMUNITY)   [random-pool, fit level]
   and AALTO/COMMUNITY split the remaining (1 − w_POOL_raw) by STEP 2.
   => This is derived from a measured quantity, not chosen.

STEP 2 — the AALTO : COMMUNITY split.
   IF (c) passes its §2.4 falsifier  -> split ∝ rho_ceil_AALTO : rho_ceil_COMMUNITY
   ELSE IF (a) is usable             -> split ∝ effective sample size, using the
                                        REGISTERED shrinkage form below
   ELSE                              -> 0.5 : 0.5 and report (d) as the honest answer

REGISTERED SHRINKAGE FORM for (a), fixed now so it cannot be tuned:
   A raw 907.8:1 sample ratio would give COMMUNITY weight 0.0011 — i.e. it would DELETE
   the source the user explicitly asked to include, which is a design answer masquerading
   as an evidence answer. Inverse-variance weighting on a MEAN scales as n, but these are
   FITTED SURFACES whose error is dominated by cell COVERAGE and model variance, not by
   raw sample count. So the registered precision proxy is
       ESS_m = (cells covered by m) x sqrt(median samples per covered cell of m)
   i.e. linear in coverage (the thing that makes a surface able to answer at all) and
   sqrt in depth (the standard error scaling of a per-cell mean). REGISTERED BEFORE the
   numbers are combined.

STEP 3 — SENSITIVITY BAND, mandatory regardless of which branch fires.
   I report the combined gauge under ALL of: the STEP-2 winner, equal (1/3,1/3,1/3),
   AALTO-only, COMMUNITY-only, POOL-only, and the AALTO+COMMUNITY-only 0.5/0.5 drop-POOL
   variant. If the champion is the same across the band, THAT is the headline result and
   the weight is revealed as non-load-bearing — which is a finding, not a failure.
```

### 2.6 What the three-model gauge must earn against `ms/char`

My brief established `spearman(ms/char, AALTO) = +1.0000` over 9 layouts: **weighting AALTO
heavily ≈ optimizing `ms/char`, which the campaign already does.** So the combined gauge
only earns its keep if it does something `ms/char` does not.

**Registered test:** over a pool of ≥200 layouts (not 9), measure `spearman(combined,
ms/char)` and `spearman(aalto-n, ms/char)`. The gauge earns its keep iff the combined
objective's champion **differs** from the `ms/char`-optimal direction by more than search
noise. **If it does not, I say plainly that the gauge is a re-parameterization of `ms/char`
plus a preference knob.**

---

## 3. PREDICTIONS — scored honestly, failures reported

| # | Prediction | Falsifier |
|---|---|---|
| P1 | AALTO anchor lands within +0.05% of 223236317224.4177 | it does not |
| P2 | Each model's own optimum normalizes to 1.0 ± 1e-12 and pool mean to 0.0 ± 1e-12 | it does not |
| P3 | qwerty30m normalizes into **0.35–0.65** on all three gauges (NOT ≈0) | outside |
| P4 | The 7 real candidates occupy **<25%** of each per-model 0–1 range (MODELNORM saw 0.09–0.17) | ≥25% |
| P5 | POOL's unique variance share (1−R²) is **<0.15** at fit level over the random pool | ≥0.15 |
| P6 | (c) **FAILS** its §2.4 falsifier (n=7 participants is too thin), so (a)+(b) decides | it passes |
| P7 | The combined-objective champion is **identical or statistically tied** across the entire §2.5 sensitivity band — i.e. the weight is not load-bearing | some weighting gives a distinct, margin-clearing champion |
| P8 | The combined champion does **not** dominate the incumbent field on the 15-gauge frame | it dominates |
| P9 | `spearman(combined, ms/char)` over ≥200 layouts is **>0.85** — the gauge is largely a re-parameterization | ≤0.85 |
| P10 | Search noise sd, measured by me on MY quadruple, is **within 2× of 0.0492–0.0995** (the two prior arms' values on this engine) — a check on borrowing, not a borrowed floor | outside 2x |

**I measure my OWN search-noise sd** and state the
**(pool × replicate-structure × scale × statistic)** quadruple. A floor is a property of
that quadruple, not of a metric or a corpus, and **I do not borrow one.**

---

## 4. GATES (a result that fails a gate is not reported as a result)

1. `assert_module_under("keybo", "/tmp/normgauge")` before any number.
2. **Harness positive control first:** `assert_harness_detects_a_fatal_mutant` — the suite
   must go rc≠0 with a planted fatal mutant and back to rc=0 on restore, BEFORE any PASS is
   trusted.
3. **Evaluator bit-exactness under batch shape.** The `bincount`-then-matmul idiom is the
   campaign's BLAS shape-dependence class (three known instances). The evaluator pins its
   matmul tile and asserts **bit-exact** equality across batch lengths, with a mutation
   control that fails if the unpadded path ever becomes batch-invariant.
4. **Planted-drift refusal:** a gauge whose persisted anchors do not match the surfaces it
   is asked to score must **refuse**, not silently rescale.
5. `oxey-style` values must be **freshly computed** (the nested `bad_redirect` fix landed
   2026-07-28; every earlier ledger value is ~0.65–1.45 higher than current code produces).
6. Contested-axis counts, never a bare n/15: `sfr` is a permutation invariant, and
   `alt`/`imbalance` tie by construction for layouts sharing a hand partition.
7. `unique_evals` **ACHIEVED**, not requested. rc read from a **SENTINEL**, not a callback.
8. **GENERATE OR ASSERT, NEVER RETYPE** — every number in the report is emitted by code or
   asserted against code.

---

## 5. WHAT I AM NOT CLAIMING

* No claim about realized typing speed (modelled, `g`-frame, baked 90 WPM, blend-v1).
* No layout adopted or recommended.
* No claim that the scheme re-orders anything — MODELNORM-1 found 0 discordant pairs, and
  **if I reproduce that null I report it as the answer.** The deliverable is then the
  interpretable weight, which is still worth shipping.
* No claim that the three sources are independent — §0.1 and §2.3 say the opposite.

---

# AMENDMENT 1 — registered BEFORE any cross-prediction result exists (2026-07-28)

Two defects in §2.4 above, both found by me while diagnosing a slow run, both corrected here
**before a single cross-prediction number was produced.** No result has been seen; the
amendment is therefore a pre-registration, not a post-hoc adjustment. The original §2.4 text is
left standing above so the change is auditable.

## A1.1 🔴 MY OWN PARTICIPANT COUNT WAS WRONG — and the true number is WORSE for my design

§2.4 says *"n=7 community participants is very thin."* **The 7 pids are in the whole community
file; the 4-label rowStagger TRAINING subset the COMMUNITY surface was fitted on has only
FOUR** — 200001, 200003, 200006, 200007 (generated, not retyped). So the held-out design is
thinner than I registered, and my registered falsifier is *more* likely to fire, not less.

This is a **wrong constant attached to a true conclusion** — the conclusion ("thin, may be
underpowered") holds and in fact strengthens. Logged as one of this arm's own kills.

## A1.2 🔴 MY REGISTERED BOOTSTRAP CANNOT PROPAGATE PARTICIPANT UNCERTAINTY ON THE AALTO SIDE

§2.4 registered a participant bootstrap that **keeps a cell if it contains ANY drawn
participant**. Measured, before use:

| side | cells | pids | median pids/cell | fraction of cells surviving a resample |
|---|---|---|---|---|
| held-out COMMUNITY | 866 | 4 | 1.0 | mean **0.6827**, min 0.1547 |
| held-out AALTO | 24,079 | 55,404 | 139.0 | mean **0.999992**, min 0.999917 |

⚠ **On the AALTO side the resample is a NO-OP: essentially every cell survives every draw, so
the cell VALUES never move and the interval collapses toward zero width.** An inclusion-only
bootstrap over pid-rich cells does not resample the estimator's inputs — it resamples which
cells exist, and when that set is invariant the CI is an artifact. **It would have manufactured
significance on exactly the side with the most data**, which is the worst possible direction for
an error in a weighting rule.

Note the two sides fail in *opposite* ways, which is why one number could not have revealed it:
COMMUNITY's cells are pid-POOR (median 1 pid/cell) so its inclusion-bootstrap does move, while
AALTO's are pid-RICH so its does not. A single-sided check would have looked fine.

**CORRECTED ESTIMATOR, registered now:** a **cluster bootstrap over participants that
RE-AGGREGATES each cell's observed value from the drawn participants' own samples** (drawn with
multiplicity), rather than including or excluding whole cells. Participant sampling then
propagates into the cell values, which is the quantity the correlation is computed over.

**Two consequences I state rather than hide:**
1. The replicate's cell value is a **plain sample mean** over drawn participants, whereas the
   point estimate uses the shipped **IQR-mean** (`keybo.data.strokes.iqr_average`). Re-running
   an IQR-mean per cell per resample is not affordable. **So I also report the point estimate
   under BOTH aggregations**; if they disagree materially, the CI is reported as
   indicative-only rather than load-bearing.
2. A cell whose drawn participants contribute no samples is dropped from that replicate. This is
   unavoidable and is reported as a per-replicate surviving-cell count.

**The falsifier of §2.4 is UNCHANGED** (CI crossing 0, or the two rho/ceiling values within one
pooled bootstrap SE → (c) is refuted and the tree falls through to (a)+(b)). Only the estimator
that computes the interval changes, and it changes in the CONSERVATIVE direction: a wider,
honest interval makes the falsifier *easier* to trigger, so this amendment cannot be a way of
rescuing my preferred branch.

**Prediction P6 is unchanged and still predicts (c) FAILS.**
