# OQ-2 — The `freq` feature saturates and is dual-purpose: acceptable, or a smell?

**Status: 🟡 mostly subsumed by OQ-1.** If OQ-1 resolves weight-only, OQ-2 dissolves — there
is no freq feature left to saturate, and "dual-purpose" reduces to the uncontroversial weight
role. This artifact records the evidence and the residual question if OQ-1 goes the other way.

## Best current answer

The saturation itself is 🟢 VERIFIED (audit #6 + reproduced): XGBoost trained on
keystroke-count freqs (~1–500) maps ALL corpus-scale freqs (10³–10⁷) to one leaf value —
the feature contributes a constant offset at scoring time. Two consequences:

1. As-is, the feature is *harmless but useless* at serve time (constant ≈ no ranking signal)
   — which is exactly why the OQ-1 probe found no ranking difference with/without it.
2. The *dual role* (same number is both a model input and the summation weight) is
   conceptually muddled but currently benign: the weight does all the work; the feature does
   ~none. The muddle becomes dangerous only under OQ-1="keep" + a distribution fix (OQ-3),
   because then the feature stops being constant and starts steering predictions — at which
   point feature-freq and weight-freq MUST be deliberately decoupled (they answer different
   questions: "how practiced is this pattern" vs "how much does this pattern matter").

## If OQ-1 resolves "keep the feature" — residual work

- Log-transform the freq feature (or train on corpus-scale counts) so the training range
  covers the serve range; re-check saturation with a partial-dependence sweep.
- Rename to `practice_freq` vs the weight to keep the two roles from re-merging.
- Add a schema test asserting the feature's train/serve ranges overlap (guard against
  silent re-saturation after a data change).

## Definitive close

Closed automatically by the OQ-1 experiment: weight-only → **closed (moot)**; keep-feature →
run the partial-dependence sweep after the OQ-3 distribution fix and require the feature to be
non-constant over the serve distribution (else drop it anyway).
