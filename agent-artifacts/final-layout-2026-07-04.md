# Final layout — R1W model + QAP deep search (2026-07-04)

```
w a e , y   l r s t f
g o i u p   m n c d b
q ; . k z   h v x j /
```

String form: `wae,ylrstfgoiupmncdbq;.kzhvxj/` (slot order; space fixed at thumb).

## Numbers (mean of 3 R1W seeds, 840-bigram common typable subset = 93.8% of corpus weight)

Two percentage conventions, both correct, measuring different things:

| layout | vs qwerty, g-only denominator | vs qwerty, **total-time denominator (honest)** |
|---|---|---|
| **this layout** | +2.55% | **+3.49%** |
| dvorak | +1.03% | +1.41% |
| semimak | +0.70% | +0.96% |
| graphite | +0.39% | +0.53% |
| colemak | −0.04% | −0.06% |

- The model is `time = g(geometry, wpm) + b(bigram)`; the practice term `b` is
  layout-independent, so **differences** between layouts come from `g` alone — but a
  *percentage* needs a denominator, and the meaningful one is predicted **total** typing
  time `Σ freq·(g+b)`, not `Σ freq·g`. On the common subset `Σ freq·b = −2.08×10¹⁰`
  (freq-weighted mean b ≈ −40 ms: practiced bigrams are genuinely fast), so the honest
  percentages are *larger*: qwerty total-time baseline 5.63×10¹⁰ vs g-only 7.70×10¹⁰.
- Per-seed consistency (g-only convention): +2.45 / +2.58 / +2.61%.
- Search: QAP position-pair table (TableBigramScorer, 8.1 µs/eval, exact-parity-tested
  against BigramModelScorer), 28 SA restarts × 60k iters + exhaustive 2-opt + 30
  perturbation kicks + 3-cycle polish, optimizing the mean of the 3 seed tables.

## Structure (emergent, not imposed)

All five vowels on the left hand (`a e` top, `o i u` home); the `r s t` consonant core on
right home with `n c d`; rare letters (`q z x j`) in bottom-corner slots. A dvorak-like
hand-alternation split emerged from the data alone — consistent with the de-confounded
model now ranking dvorak as the best *named* layout (+1.41%), where the practice-inflated
model had ranked colemak/semimak-style rolls higher.

## Caveats (standing, from the OQ-5 verdict)

1. **Within-model numbers.** R1W ~passes the pre-registered LOLO bar (pooled τ +1.0 all
   seeds, beats-baseline 12/12, ρ ≥ 0.8× ceiling on 3/4 folds; qwerty .796 borderline) —
   but validation spans only the 4-layout QWERTY-adjacent family. "Transfers to a
   from-scratch alien layout" remains extrapolation.
2. **The optimum is a plateau.** Prior rounds showed layouts sharing 6/30 positions can
   score within ~0.5%; the top of the search landscape is wide and near-flat, so this
   exact permutation is "a best layout," not "the unique best."
3. **Speed-only objective.** Comfort/effort terms (OQ-4) are not yet in the objective.

## Post-hoc adversarial audit (2026-07-05; parent-run after the fable auditor died to API churn)

Attack surfaces checked, with verdicts:

1. **Does the practice term b really cancel in the ranking τ? SOUND — exact, structural.**
   `layout_ranking_tau` restricts to the common ngram set and takes an unweighted mean of
   per-ngram means, so adding b(ng) shifts every layout's score by the same constant
   `mean_common(b)`. Verified numerically on three adversarial constructions (equal cell
   sets, unequal per-layout cell sets, a layout missing an ngram entirely): τ identical to
   machine precision. The +1.0 τ and the QAP search objective are not b-inflated.
2. **Is the saved model really freq-inert (table reduction assumption)? VERIFIED.** All
   three seed boosters have **zero** splits on the freq column (`get_score` scan) — the
   constructor's 2-point probe is confirmed at the artifact level.
3. **Is W a no-op given 98.7% qwerty? NO.** At the training-example level (row × wpm
   group) the raw shares are qwerty 58.8% / qwertz 19.4% / azerty 12.7% / dvorak 9.1%;
   the capped inverse-share weights equalize to 25% each with an effective sample size of
   62.3% — a real reweighting. (R1W's edge over R1 is nonetheless small: .931 vs .928.)
4. **Is the k=100 shrinkage load-bearing? NO — conclusion robust.** Dvorak fold, seed 0:
   ρ = .545 / .529 / .503 at k = 10 / 100 / 1000, all far above B's .310. Less shrinkage
   is marginally better on this fold, but k stays at the pre-registered 100 — switching
   post-hoc based on test-fold performance would be exactly the overfitting the
   pre-registration guards against.
5. **NEW HONEST FINDING — decomposition of the per-cell ρ on dvorak.** The practice term
   *alone* (no geometry model at all) scores ρ +0.59–0.61 on dvorak's held-out cells —
   as high as the full g+b model (+0.56–0.59). So on the hardest fold, the ρ-vs-ceiling
   bar is passed mostly by *practice prediction*, not geometry transfer. This is
   legitimate prediction (practice is real and layout-independent — predicting it is
   correct), but the geometry-specific transfer signal on dvorak is better read from the
   b-invariant τ (+1.0) and from arm B's geometry-only ρ (.45–.47 of ceiling). The
   "beats the distance baseline 12/12" claim does not hinge on this (arm B, with no b,
   also beats it 12/12).

Probes: `keybo-e2e/takeover_probes.{py,log}`, `probe_b_alone.{py,log}`, plus inline
τ-invariance checks logged in the workspace state.

## Provenance

- Models: `keybo-e2e/models/bigram_r1w_seed{0,1,2}.json` + `.practice.json` sidecars.
- Result: `keybo-e2e/runs/final_layout.json`; log `final_search.log`.
- Driver: `agent-artifacts/experiments/final_search.py` (archived copy).
- Arm-matrix evidence for R1W: `agent-artifacts/OQ1-frequency-feature.md` (tables),
  `keybo-e2e/runs/arms_matrix.json`.
