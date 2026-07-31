# RETRAIN-DIRECTION — pre-registration

**Written and committed BEFORE any model was trained and before any score was read.**
Base: `16b1a06` (`redirect-samefinger-gate`) + the cherry-picked high-wpm gate (`2b0254e`, `e868657`).

## The question

`DIRECTION-1` and `REDIRGATE-1` added order-aware direction columns that **no model has ever
trained on**. Do those columns carry **out-of-sample** signal — i.e. does a model trained on the
widened frame transfer better to a layout it has never seen than the matched narrow model?

This measurement informs a scoring-policy decision that is the human's. It does not make it.

## What "matched pair" means here, exactly

For each `(ngram, holdout layout, seed)` cell, two models are fit that differ in **exactly one**
respect: the feature matrix has 20 columns (narrow) or 22 (widened) for bigram, and 46 or 50 for
trigram. Identical between arms:

- the same `StrokeRow` list, the same leave-one-layout-out partition, the same held-out cells;
- the same seed set `{0, 1, 2}` fed to XGBoost `random_state`;
- the same hyperparameters (whatever `train_params` carries — default XGBoost otherwise);
- the same `target_space="LOGRAT"` (matches all six shipped `data/models/k31/` models);
- the same practice term, layout balance weights, cell floor, wpm band, bucket width, `n_boot`;
- the same ceiling (the split-half ceiling depends only on the held-out data, not the model).

The widened arm stamps `FEATURE_VERSION_DIRECTION`; the narrow arm stamps `FEATURE_VERSION`.
A widened model is therefore load-time-distinguishable from a served one and cannot be mistaken
for it.

## Pre-registered prediction

**I predict `null`** — specifically: no reliable held-out rho improvement, the new columns getting
near-zero feature importance, and no change in layout ordering.

Reasoning, in descending weight:

1. **`d66e1dc` already measured direction against time directly and found nothing usable**: only
   1 of 4 roll classes agreed in sign, effect ~0 to −1.75 ms/char, below the resolution floor.
   A feature that does not move the target in a direct measurement is unlikely to be picked up
   by a tree ensemble as a transferable signal.
2. **The training data is 98.7% qwerty** (the frequency-as-feature trap, `OQ-1`). The direction
   columns are a *geometric* recoding, and the LOLO folds hold out whole layouts, so any
   qwerty-specific use of the new columns is precisely what LOLO is built to refuse credit for.
3. **The information is largely already present.** `dx`, `dy`, `angle` are signed and computed
   from the ordered pair; a tree can in principle already separate an inward from an outward
   stroke using them. The ordered roll columns are a *thresholded* recoding of information the
   frame carries — so the marginal information is small even though the columns are new.
4. Against the null: `YUO-1` showed the *models* do price stroke order (`yuo` 105.813 vs `oyu`
   133.622 ms). But that is evidence the **target** carries order, which points at `dx/dy/angle`
   already doing the work — consistent with (3), not with a widened-frame win.

## Falsifier

The prediction is falsified if **both** of these hold:

- the widened arm wins the paired per-fold rho delta on **≥ 3 of 4 folds** with the **same sign
  on all three seeds** (i.e. sign-consistent, not a seed-wobble artifact), **and**
- the new columns take **> 2% of total gain importance** in the widened models (they are actually
  being used, not ignored).

A rho win with the new columns at ~0% importance is not a direction result — it is seed noise, and
I will say so.

## Decision rule (fixed now, applied without amendment)

The verdict is one of the four the brief names, decided in this order:

1. **HIGH-WPM GATE FIRST, and it is a veto.** Using `verdicts.require_no_high_wpm_regression`
   (`HIGH_WPM_FLOOR = 80`, `HIGH_WPM_TOLERANCE = 0.005`), with the **narrow arm as baseline** and
   the **widened arm as candidate**, per fold and per seed. If the widened arm regresses **any**
   bucket ≥ 80 wpm beyond tolerance in **any** (fold, seed) cell, the widened frame **FAILS**
   regardless of its mean rho. I report the per-bucket deltas either way.
   *If the gate cannot run in a cell (`gated=False`), that cell is reported UNSCOREABLE — not
   passing.* "Not measured" is not "did not regress".
2. **Transfer (fit).** Paired per-fold delta `rho_widened − rho_narrow` **per (fold, seed) cell**
   — never a mean of ratios across folds (`MOR-FIX-1`: folds have different ceilings and a
   mean-of-ratios can reorder). I report every cell, the count of wins/losses, and whether the
   sign is consistent across seeds within a fold. A "win" requires sign-consistency across all
   three seeds in a fold; a fold whose three seeds disagree in sign is a **tie**, not a win.
3. **Ranking.** `tau_heldout` (every layout scored only by the fold that held it out — fully
   out-of-sample) and `tau_all4`, per seed. Plus the 13-layout field ordering under each arm.
   **A fit win with a ranking loss is a FAILURE**, per `NGRAM-FE` (improved fit, destroyed served
   ranking 0.852 → 0.164). Ranking is checked, not assumed.
4. **Importance.** Total gain per column in the widened models. If the new columns are at ~0,
   the honest statement is "the columns are ignored", whatever the rho did.

### Verdict mapping

| condition | verdict |
|---|---|
| high-wpm gate fails in any cell | **FAIL** (veto), regardless of rho |
| rho tie/loss AND ranking unchanged AND importance ~0 | **null** |
| rho win (per rule 2) AND ranking degrades | **fit-win-ranking-loss** |
| rho win AND ranking holds or improves | **both-win** |
| rho loss AND ranking degrades | **both-lose** |

## Scope constraints held throughout

- Nothing pushed, nothing merged, no layout adopted, no shipped weight flipped.
- `PREREGISTRATIONS.md` untouched.
- `data/models/k31/` never written to. Any model artifact goes to a NEW directory.
- Every table in the report is **generated** by a driver, never hand-transcribed.
- Every `n` travels with its scope.
