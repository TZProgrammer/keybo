# FRAMEDIAG-1 PREREGISTRATION — the model-free frame collapse diagnostic as committed code

Registered BEFORE any number produced by this arm exists. Base: `interpframe` @ b973f39.
Branch: `framediag`. Task: promote INTERPFRAME-1's H3 diagnostic (currently prose in the ledger
plus `agent-artifacts/interpframe/resolution.py`, a throwaway driver) into a committed,
frame-agnostic library surface + CLI with mutation-tested tests.

## §0 — WHAT IS BEING BUILT, AND WHAT IT IS NOT

The diagnostic answers ONE question with NO model, NO SHAP and NO training:

> Given a featurizer and a geometry, how many of the geometry's position cells does the
> featurizer make INDISTINGUISHABLE from some other cell, and what is the best-case error any
> model on that frame could achieve on a known per-cell target?

It is a NECESSARY-condition instrument. Registered NON-CLAIMS (§6) below.

## §1 — THE ESTIMANDS, DEFINED BEFORE MEASUREMENT

Let the geometry supply `P` positions (`slots` + `space_position`), so the cell space for an
order-`k` frame is `P**k` cells. Let `X` be the `(P**k, C)` feature matrix, one row per cell.

* **GROUPING.** Cells are grouped by their feature row under an explicit equality rule (§3).
  A group is a maximal set of cells sharing one row. `distinct_feature_rows = #groups`.
* **`collapsed_cells`** = number of cells in a group of size > 1. `collapsed_share` = that / `P**k`.
* **`resolution`** = `distinct_feature_rows / P**k` — the headline single number (1.0 = no collapse).
* **`mass_share_collapsed`** = the corpus-weight share of cells in groups of size > 1. With no
  weights supplied, weights are uniform and this equals `collapsed_share`.
* **`largest_group`** = max group size.

## §2 — THE WITHIN-GROUP FLOOR: THE EXACT ESTIMATOR (INVARIANT 4)

Given a target vector `t` (the TRUE shipped per-cell value; ms) and weights `w >= 0`, every cell in
a group must receive ONE prediction `p_g`. The floor is the error of the BEST such assignment:

    floor(L) = min over {p_g} of  sum_i w_i * L(t_i, p_g(i))  /  sum_i w_i

**REGISTERED, AND IT IS A CORRECTION TO INTERPFRAME-1's ESTIMATOR (see §5, hypothesis T2):**
the minimizer depends on the loss, and the two are NOT interchangeable:

* **`floor_wmae`** (L = absolute error) is minimized by the **weighted MEDIAN** of `t` within the
  group, NOT the weighted mean. `resolution.py` used the weighted MEAN for a wmae floor. A mean-based
  wmae "floor" is an ACHIEVABLE ERROR of one specific predictor, hence an UPPER bound on the true
  floor — it is valid as "a model can do at least this well" but it is NOT the greatest lower bound.
* **`floor_wmse`** / **`floor_wrmse`** (L = squared error) IS minimized by the weighted MEAN.

DECISION RULE, registered now: the implementation reports **both** `floor_wmae` (median-based, the
mathematically correct L1 floor) and `floor_wmae_at_group_mean` (mean-based, exactly reproducing
INTERPFRAME-1's published quantity) so the published number stays checkable and the corrected number
is available. Ledger-comparison uses `floor_wmae_at_group_mean`.

**Weighted median convention (registered to remove the tie ambiguity):** the lower weighted median —
sort group members by `t`, take the smallest `t` whose cumulative weight reaches `0.5 * W_g`. On an
even 2-cell equal-weight group this selects the LOWER value; L1 cost is identical for any point in
the interval, so the choice cannot change `floor_wmae`, only which minimizer is reported.

## §3 — THE FLOAT-TOLERANCE DECISION (INVARIANT 3)

Grouping needs an equality rule on float rows. Registered design:

* **`tol=0` is the DEFAULT and means EXACT bitwise equality** (`np.unique(X, axis=0)`), which is what
  `resolution.py` did. Exact is the default because it is the only rule that is transitive, reproducible
  across BLAS versions, and free of a tuned parameter.
* **`tol>0` quantizes**: each column is mapped to `round(x / tol)`. Quantization is an explicit
  COARSENING: it can only MERGE groups, never split them, so `distinct_rows(tol)` is
  **NON-INCREASING in tol**. This is registered as a testable monotonicity property (§4, T-MONO).
* Quantization is per-column on RAW units. Registered non-claim: it is not scale-invariant across
  columns (a `dx` in key-widths and a `wpm` in words/min are quantized by the same absolute step).
  The default `tol=0` avoids the question entirely; a caller passing `tol>0` accepts it.

**THE 765-vs-775 PREDICTION, REGISTERED BEFORE MEASURING (this is the arm's sharpest test).**
The parent attributes its own 775 (vs the ledger's 765) to "rounding features at 12 decimals vs an
exact comparison". **I register the OPPOSITE prediction: rounding CANNOT be the cause, on arithmetic
alone.** Rounding is a function; `x == y` implies `round(x) == round(y)`; therefore the rounded
partition is a COARSENING of the exact partition and `distinct_rounded <= distinct_exact` ALWAYS.
765 -> 775 is an INCREASE, which no coarsening can produce. So at most one of these holds:
  (a) the 775 run differed in the FRAME (e.g. included `wpm`, or the direction channel, or a
      different featurizer entry point);
  (b) the 775 run differed in the GEOMETRY / cell space (e.g. ROW_STAGGERED_30+space = 31 positions
      is the same 961, but a different `space_position` or slot order changes rows);
  (c) the 775 run differed in the TARGET WPM or another featurizer argument;
  (d) 765 or 775 is a transcription error;
  (e) the 775 run rounded and ALSO did something that splits — impossible from rounding alone.
DECISION RULE: I will measure `distinct_rows` for the served frame at `tol=0` from committed code,
and sweep `tol` over `{0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3}`. If the exact count is 765 and the sweep is
non-increasing, the ledger's 765 is CORRECT and the parent's 775 is a run-configuration difference,
NOT a rounding artifact — and I must say so as a correction to the parent. If the exact count is 775,
the LEDGER is wrong. Either way the monotonicity result decides the mechanism claim.

## §4 — THE TESTS, AND THEIR PASS/FAIL BARS, REGISTERED BEFORE THEY ARE WRITTEN

Every test below is MUTATION-TESTED: for each, a named mutation of the implementation must turn it
RED. A test that stays green under its own mutation is reported as VACUOUS (INVARIANT 5; the campaign
found three such). Mutation verdicts are collected with rc captured from the TEST process, never from
a pipe tail (`cmd | tail; rc=$?` captures tail's rc — a known campaign harness bug).

* **T-KNOWN-FLOOR (INVARIANT 4's known-answer test).** A synthetic 2-cell frame whose featurizer maps
  BOTH cells to the same row, with targets `t = [t0, t0+d]` and equal weights. Registered exact
  expected values, derivable by hand: group mean = `t0 + d/2`, so `floor_wmae_at_group_mean = d/2`;
  the L1 minimizer is any point in `[t0, t0+d]` so `floor_wmae = d/2` too (equal on a 2-point set);
  `floor_wrmse = d/2`. With weights `(w0, w1)` unequal, `floor_wmae` (median) = `d * min(w0,w1)/(w0+w1)`
  while `floor_wmae_at_group_mean` = `2*d*w0*w1/(w0+w1)**2` — **these differ**, which is the test that
  distinguishes the two estimators. Chosen numbers: `d = 4.0`, `w = (3.0, 1.0)` giving
  `floor_wmae = 1.0` and `floor_wmae_at_group_mean = 1.5`. Both asserted exactly.
* **T-ZERO-COLLAPSE.** An injective featurizer (row = the cell index) must give
  `distinct_feature_rows == n_cells`, `collapsed_cells == 0`, `resolution == 1.0` and BOTH floors
  EXACTLY 0.0.
* **T-TOTAL-COLLAPSE.** A constant featurizer must give `distinct_feature_rows == 1`,
  `collapsed_cells == n_cells`, and `floor_wmae_at_group_mean` == the weighted mean absolute deviation
  of the whole target about its weighted mean.
* **T-MONO.** `distinct_feature_rows` is non-increasing over the registered tol sweep on a fixed frame.
* **T-WEIGHTS.** Supplying weights changes `mass_share_collapsed` away from `collapsed_share` on a
  frame where the two provably differ; zero weight on a cell removes its contribution to the floor.
* **T-SERVED-961 / T-INTERP-378.** The committed code, on ROW_STAGGERED_31 + space, reproduces
  `n_cells == 961` and the ledger's distinct-row counts for both frames, and the floors to 4 dp.
  ⚠ Registered as a REPRODUCTION CHECK, not a bar on my code: if it disagrees, INTERPFRAME-1's number
  is wrong and I report that loudly rather than tuning to match (INVARIANT 1).
* **T-TRIGRAM.** The diagnostic runs at order 3 on the 46-column trigram frame over its own
  `31**3 = 29791` cell space and returns finite statistics. Registered up front: the FLOOR at order 3
  needs an order-3 target. `TimeSurface` supplies `triple_ms_table()` = `T2[a,b] + Tcond[a,b,c]`, so
  the floor IS computable at order 3 with that as `t`. Whether it is cheap is a measurement.

## §5 — HYPOTHESES ABOUT INTERPFRAME-1's PUBLISHED NUMBERS, REGISTERED BEFORE CHECKING

* **T1 (the 765/775 gap): rounding cannot explain it.** §3. Falsifiable: if a 12-dp rounding of the
  served frame yields MORE than the exact count, my arithmetic is wrong and I say so.
* **T2 (the floor estimator): the published `floor_wmae` is a MEAN-based quantity and is therefore an
  UPPER bound on the true L1 floor, so the true L1 floor for interp.1 is <= 2.2399 ms**, and the
  "38.9% of the gap" share is correspondingly an over-statement of the *lower bound* (though a valid
  statement about the group-mean predictor). Falsifiable: if median- and mean-based floors agree to
  4 dp on the real interp.1 frame, T2 is confirmed-but-immaterial and I report that.
* **T3 (the served floor): `0.0000 ms` is a ROUNDED 3.1496e-15, not an exact zero** — it is float
  noise around a genuine zero, since the served frame's 380 collapsed cells sit in groups whose
  members share the same shipped time to float precision. Falsifiable by inspecting whether any served
  group has a non-degenerate target spread.
* **T4 (the mass share): 93.2% is the collapsed-mass share for interp.1, NOT for served.** The ledger
  sentence "817 cells = 93.2% of corpus MASS" pairs interp.1's 817 collapsed cells with interp.1's
  93.2% mass; served's own numbers are 380 cells / 53.5%. My brief compressed this into a form that
  could be read as a property of the served frame. Falsifiable against `resolution.json`.

## §6 — WHAT THE DIAGNOSTIC CANNOT PREDICT (INVARIANT 6), REGISTERED AS NON-CLAIMS

1. **Zero collapse does NOT imply a good frame.** `resolution == 1.0` says only that no two cells are
   forced to share a prediction. A frame of 20 columns of pure noise plus one cell-id column has
   perfect resolution and no predictive value.
2. **The floor is a LOWER bound on error, never a prediction OF the error.** Measured LOLO wmae was
   15.70 vs a 2.24 floor: the floor explained 38.9% of the *gap*, not the level.
3. **It says nothing about GENERALIZATION.** It is computed on the cell space, not on held-out data,
   so it cannot see overfitting, extrapolation, or train/serve skew.
4. **It says nothing about the OPTIMIZER.** A frame with low collapse can still contain a null space a
   search exploits (this repo's own GOODHART/row-blindness precedent). High resolution is not safety.
5. **It is TARGET-RELATIVE.** The floor depends on the target vector supplied; a different target (a
   different surface, corpus or WPM) gives a different floor on the same frame.
6. **It cannot rank two frames' accuracy.** It bounds one frame's best case. A frame with a lower
   floor may still train worse.
7. **The mass share is CORPUS-relative** and inherits the corpus's own biases (98.7% qwerty data).

## §7 — SCOPE / SAFETY BARS

`FEATURE_VERSION` untouched · `data/models/k31/` unmodified · `layouts.py` untouched · no CODE push ·
no branch merge/delete · additive files only where possible; any edit to an existing file is reported.
Full suite must be >= the base branch's 1393 passed / 3 skipped / 0 failed.

---

## §8 — ADDENDUM 1: CORRECTING MY OWN T-MONO BAR (registered BEFORE the corrected result is written up)

**A BAR I REGISTERED IN §3/§4 IS FALSE AS STATED, AND THE CODE IS RIGHT — SO THE BAR MOVES, NOT THE
CODE** (the campaign's standing rule from INTERPFRAME-1's `wpm`-equality correction).

§3 registered: *"Quantization is an explicit COARSENING: it can only MERGE groups, never split them,
so `distinct_rows(tol)` is NON-INCREASING in tol."* §4 registered T-MONO to pin that.

🟢 **MEASURED COUNTEREXAMPLE on the real served frame:** the sweep reads
`0.25 -> 765, 0.5 -> 701, 0.75 -> 709, 1.0 -> 649`. **0.5 -> 0.75 RISES by 8 rows.** Minimal scalar
instance, verified: with `x=0.3, y=0.4`, `round(x/0.5) == round(y/0.5) == 1` (MERGED) but
`round(x/0.75) = 0 != 1 = round(y/0.75)` (SPLIT). **Mechanism: the grid's bin BOUNDARIES move with
`tol`, so the family of quantized partitions is NOT a nested refinement chain — a coarser grid can
split a pair a finer grid merged.**

**WHAT IS ACTUALLY TRUE, AND IT IS THE CLAIM THE 765-vs-775 ARGUMENT NEEDS (so that result stands
unweakened):** for ANY `tol >= 0`, exact-equal rows have equal quantizations, so the quantized
partition IS a coarsening **of the EXACT partition** specifically:

    distinct_rows(tol)  <=  distinct_rows(exact)      for every tol >= 0.       [TRUE, kept]
    distinct_rows(t2)   <=  distinct_rows(t1) for t2 > t1 > 0.                  [FALSE, retracted]

The retracted clause was never needed: the 765-vs-775 prediction (§3) rests only on the surviving
clause — no tolerance can take 765 UP to 775 — and that is exactly what the measurement shows (765 at
every tolerance from exact to 1e-3, and never above 765 anywhere in the widened sweep to tol=10).

**REGISTERED REPLACEMENTS, decided before the corrected numbers are written up:**
* **T-MONO is REPLACED by T-COARSENING:** for a registered tol list spanning
  `{0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}`, assert
  `distinct_rows(tol) <= distinct_rows(exact)` for every tol — the true guarantee — and assert it is
  NOT vacuous by requiring at least one tol in the list to be strictly less.
* **T-NONMONOTONE (new, positive test):** pin the non-monotonicity as REAL with the `0.3/0.4` scalar
  counterexample, so a future contributor cannot "fix" the docstring into the false monotone claim.
  This is the same defensive shape INTERPFRAME-1 used for the sign-disagreement it could not license.
* **Non-monotonicity must be REPORTED, not hidden:** `tolerance_sweep`'s human output must FLAG a rise
  rather than call it a bug, and the module docstring must carry the counterexample.
* **Registered non-claim (new, non-claim 9):** a `tol>0` grouping is a QUANTIZATION, not an
  equivalence up to `tol` — two rows within `tol` of each other may land in different bins (and two
  rows up to `2*tol` apart may share one). Single-linkage-within-`tol` IS monotone in `tol` (verified
  on a 60x3 random matrix: 60,60,60,59,53,41,25,8,2,1,1) but is a CHAINING relation, not equality,
  and costs `O(n^2 C)` — infeasible at the 29791-cell trigram space. Quantization is kept as the
  shipped rule with `tol=0` (exact) the default; the monotone alternative is documented as
  deliberately NOT shipped, with its reason.

⇒ **The practical guidance is unchanged and is now the DOCUMENTED reason `tol=0` is the default:** a
nonzero tolerance is a parameter whose value must be reported beside any number it produced, and on
these frames it buys nothing (flat to 1e-3, and only bites at `tol >= 0.5`, i.e. half a key width —
far outside "float noise" territory).
