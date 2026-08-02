# ROWOFFSETS-1 — PRE-REGISTRATION (written BEFORE any INVARIANT B/C fit or LOLO result exists)

Agent `stagger`, child of `keybo-optimization`. Branch `stagger-rowoffsets`, worktree /tmp/stagger-wt.
Question (human's words): "Why is row offset hardcoded like this? Is there a way to properly assign
the best row_offsets for the learning of the model?"

Target: `src/keybo/geometry.py` `Geometry.row_offsets = {1: 0.5, 2: 0.0, 3: -0.25}` (y=3 top, y=2 home,
y=1 bottom), consumed by `stagger_adjusted_dx` -> features `dx`, `sg_dx`, `lsb`, `angle`, and
`analysis.lateral_span`.

## What is ALREADY MEASURED at registration time (INVARIANT A, complete; does not depend on any fit)

A1. 🟢 VERIFIED. Uniform shift c added to all three rows: over the 900 ordered letter-letter position
    pairs of ROW_STAGGERED_30 the 20-column bigram feature matrix is BIT-IDENTICAL for every DYADIC c
    (0.5, 1.0, 7.0, +/-0.25, 0.125). Ledger:9313's cancellation claim is CONFIRMED for letter pairs.
A2. 🟢 VERIFIED, AND IT CORRECTS THE LEDGER AND MY PARENT'S BRIEF: the cancellation FAILS on
    SPACE-touching pairs. Space is (0,0); `row_offsets` has no key 0, so `.get(ay, 0.0)` PINS space at
    0.0 and it does not move with the shift. 60 of 61 space-involving ordered pairs change (max |d| =
    c exactly, column `dx`). Same on the trigram frame (`sg_dx`). So the correct statement is:
    *offsets enter only inside differences, but one arm of the difference (space) is a hardcoded 0
    that no shift moves* -- a uniform shift is a gauge freedom ONLY on the letter-letter block.
A3. 🟢 VERIFIED. Same-row pairs: max |feature delta| = 0.0000000000 under arbitrary random
    perturbation of all three offsets (300 same-row pairs, 3 trials). Cross-row: all 600/600 move.
    Only cross-row (and space-touching) pairs carry information about the offsets.
A4. 🟢 VERIFIED. Numerical Jacobian rank of (3 offsets) -> features:
      letter-letter only  : rank 2, null space exactly (1,1,1)  => 2 free params, home pinnable at 0
      space-involving only: rank 3
      all pairs           : rank 3, third singular value 4.47 vs 2.0e5 / 3.0e3 -- i.e. the third
                            direction is present but ~450x weaker than the first two.
    => PARAMETER COUNT IS 2 on the letter block (parent's expectation CONFIRMED), but 3 on the full
    frame the model is actually trained on, because space breaks the gauge. The 3rd param is best
    read as "space's own offset" (an orthogonal 4th knob: setting row_offsets[0] moves space pairs
    and touches NO letter-letter pair, max|d| = 0.0e+00).
A5. 🟢 VERIFIED, and it is a real defect not a curiosity: at NON-DYADIC shift (c = -0.30, c = 1/3) the
    `lsb` INDICATOR FLIPS on 4 pairs (max |d| = 1.0, not float noise). `is_lsb` tests
    `stagger_adjusted_dx > 1.5` and 1.5 is exactly attainable, so the gauge freedom is exact in real
    arithmetic but NOT in floating point at the threshold. Any re-parameterization must therefore be
    checked for indicator flips, not just for dx equality.

## PRE-REGISTERED DECISION RULES (nothing below has been measured yet)

### B — the fit
B1. Estimator: profile the held-out-prediction objective over a GRID of (off_top, off_bottom) with
    home PINNED at 0.0 (justified by A4: home is the gauge fixing). Space is pinned at its shipped
    implicit 0.0 for the main arm; a separate arm B4 frees it.
B2. Grid: off_top in [-1.00, +0.50], off_bottom in [-0.50, +1.00], step 0.125 (DYADIC by A5, so the
    lsb indicator cannot flip on float noise). Shipped point (-0.25, +0.50) is ON the grid.
B3. Leave-one-layout-out over all 4 layouts (qwerty, qwertz, dvorak, azerty), seeds 0,1,2. The
    per-fold argmin is the per-fold estimate; the pooled estimate minimizes the summed per-fold
    held-out error. REPORT ALL FOUR per-fold estimates, never the pooled alone.
B4. Uncertainty: the CI is the set of grid points whose pooled held-out error is within the
    SEED-SPREAD of the pooled optimum (a profile-likelihood-style interval using the instrument's own
    resolution as the tolerance). RULE, REGISTERED: if that set spans >= 1.0 key width in either
    coordinate, the registered conclusion is "NOT IDENTIFIED AT THIS SAMPLE SIZE" and NO point
    estimate is quoted as the answer.
B5. Distance from shipped is reported in key widths as (|off_top - (-0.25)|, |off_bottom - 0.50|).
    If the shipped point is INSIDE the B4 interval, the registered reading is "THE HARDCODE IS
    VINDICATED -- the data cannot distinguish it from the best fit."

### C — the held-out effect, and the BAR (registered before running)
C1. Metric: the repo's OWN path -- `keybo.training.validate.validate(..., geometry=<candidate>)`,
    4 folds x 3 seeds, wpm [40,140) x 20, min_cell_samples 10. `geometry` is already a validate()
    parameter, so NO source edit is needed and the arm is a pure single-variable change.
2C. PRIMARY = paired per-fold delta of held-out wmae (MOR-FIX-1: mean of per-fold differences,
    NOT mean of ratios), candidate minus shipped, same fold, same seed.
C3. THE BAR, REGISTERED: an ADOPT requires ALL THREE of
      (a) mean paired per-fold wmae delta <= -0.135 ms/char (the MODEL-SEED floor for fixed boards --
          NOT the 0.883 SEARCH-seed spread; mixing those two is a known recurring error here),
      (b) the sign holds on >= 3 of 4 folds,
      (c) `require_no_high_wpm_regression_in_report` PASSES (no structural high-wpm regression on
          any fold, all seeds) -- and the arm is reported as UNGATED, not as passing, if
          baseline_buckets cannot be supplied.
    Anything less is registered NOW as "measurable but below the instrument's floor => DO NOT ADOPT".
C4. A same-width PLACEBO is mandatory (the DIRECTION-1 lesson: a zero-information change moved wmae
    by more than the real effect). Placebo = a geometry whose offsets are a DYADIC UNIFORM SHIFT of
    the shipped ones (+0.5 to all three rows). By A1 that is bit-identical on letter pairs, so its
    ONLY effect is via the 9.8% space-touching rows -- it measures exactly the nuisance channel A2
    discovered. Any candidate effect not exceeding the placebo's is NOT attributable to the fit.

### D — the mirror interaction
D1. Re-run the sibling `mirror`'s T2 asymmetry measurement (mean/median/p90/max over 870 pairs) with
    the fitted offsets substituted, and report before vs after. Read-only across `mirror`'s trees.
D2. REGISTERED PREDICTION, made before running (so it can be wrong): asymmetry will NOT vanish and
    will not shrink materially, because `mirror` has already VERIFIED that 100% of the asymmetry mass
    sits in the 540 cross-row pairs whose feature row CHANGES under x-negation, and that only
    row_offsets == all-zero takes that count to 0. Any NONZERO stagger keeps the board physically
    non-mirror-symmetric. So a re-fit relocates the asymmetry, it does not remove it -- UNLESS the fit
    lands near zero stagger, which would be a genuine surprise and a real finding.
D3. Registered as the DECISIVE quantity for D: the ratio (asymmetry with fitted offsets) /
    (asymmetry with shipped offsets), plus the all-zero-offsets floor as the reference point.
    "Materially shrinks" is registered as <= 0.5x on the mean.

## Hard constraints acknowledged
data/models/k31/ read-only; no merge to main; no push of non-ledger code; no layout adopted; nothing
in ~/agent-workflow-tool/; `mirror`/`latspan` trees read-only.

---

# ADDENDUM 1 — THE SPACE AXIS (`row_offsets[0]`). Registered BEFORE any space number exists.
Added 2026-08-02 on parent instruction, arising from my own INVARIANT A finding. Nothing in this
addendum has been measured at the time of writing. The 7x7 letter grid was already in flight and is
UNAFFECTED, because A4 established the axes are exactly orthogonal (setting `row_offsets[0]` moves
space-touching pairs and touches NO letter-letter pair, max|d| = 0.0e+00).

## Why it is in scope
`row_offsets` has keys {1,2,3}; space sits at `space_position=(0,0)`, so `.get(ay, 0.0)` silently
supplies **0.0 for space**. That value was never chosen -- it is an artifact of DICT OMISSION. It is
the single most consequential unexamined constant in the geometry, because space is the highest-
frequency key in any corpus and **876 of 5854 bigram eval cells (15.0%)** touch it. A2/A4 proved it is
a genuine free parameter of the model's feature map, not a gauge artifact.

## What is being estimated
`off_space` = the offset added to space's column when measuring `stagger_adjusted_dx` to/from space.
Letter offsets are PINNED at shipped (top -0.25, home 0.0, bottom +0.50) so this is a clean 1-D scan
and cannot be confounded with the letter block.

## S1 — grid
`off_space` in [-1.00, +1.00], step 0.125 (DYADIC, per A5: `is_lsb` tests `dx > 1.5` and 1.5 is
exactly attainable, so non-dyadic values can flip an indicator on float noise). 17 points. The shipped
implicit value **0.0 is ON the grid**, so the incumbent is evaluated by the identical code path.
Physical prior, recorded now so it cannot be retrofitted: on a real ANSI board the spacebar is wide and
centred, so a typist's thumb contact point has no single "column"; plausible values span roughly
[-0.5, +0.5] and anything outside that would be a surprise needing explanation, not a result.

## S2 — estimator and folds
Leave-one-layout-out over all 4 layouts, seeds 0,1,2, trained AND evaluated under the same geometry
(never `validate(geometry=)` -- VALIDATE-GEOM-1). Per-fold argmin = per-fold estimate. Report ALL FOUR
per-fold estimates, never the pooled alone (parent's explicit instruction, and justified by the 42.4%
dvorak / 59.9% qwerty per-fold cross-row spread).

## S3 — THE BAR for the space axis, registered now
Identical in form to C3 so the two axes are commensurable. An ADOPT of a non-zero `off_space` requires
ALL THREE of:
  (a) mean paired per-fold wmae delta vs shipped-implicit-0.0 <= **-0.135 ms/char** (the MODEL-SEED
      floor for fixed boards -- NOT the 0.883 SEARCH-seed spread),
  (b) the sign holds on **>= 3 of 4 folds**,
  (c) no STRUCTURAL high-wpm regression (`require_no_high_wpm_regression_in_report`), or the result is
      reported as UNGATED rather than as passing.
Below that bar the registered conclusion is: **"the space offset is measurable but below the
instrument's floor => the accidental 0.0 is not refuted, and it stays."**

## S4 — CI and the honest-null rule
CI = the set of grid points whose pooled held-out wmae is within the SEED-SPREAD of the pooled optimum.
REGISTERED: if that set spans **>= 1.0 key width**, the conclusion is **"NOT IDENTIFIED AT THIS SAMPLE
SIZE"** and no point estimate is quoted as the answer. A wide CI here is a perfectly good result and
will be reported as such.

## S5 — priority, registered so a budget squeeze cannot silently reorder it
C (held-out effect vs the pre-registered bar) > B (letter grid) > S (space axis). If budget does not
stretch to S, the report must say **"space axis: NOT MEASURED"** explicitly. Inferring it from the
letter result is forbidden -- absence of measurement is not a null result.
