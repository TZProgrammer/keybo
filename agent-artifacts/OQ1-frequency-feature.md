# OQ-1 — Should n-gram frequency be a model FEATURE, or only an objective WEIGHT?

**Status: 🟡 leaning weight-only.** The lean rests on the **saturation evidence**, not on the
synthetic probe: a fable-tier audit (2026-07-04) correctly showed the probe's original
framing was near-tautological and seed-fragile. This artifact was rewritten to reflect that
honestly. Definitive closure needs the real-data LOLO harness (OQ-5) with the **tightened
decision rule below**.

## Best current answer

**Keep frequency as the objective weight; drop it as a model feature — pending real-data
confirmation.** Evidence, in order of actual strength:

1. 🟢 VERIFIED, and the load-bearing argument (audit #6): at corpus scale the freq feature
   **saturates** — the model trains on keystroke-occurrence counts (~1–500), so every real
   corpus bigram (10³–10⁷) lands in the same top bin; predictions are flat in freq across the
   entire scoring range. A serve-time-constant feature cannot help rank layouts, while its
   presence still imports a train/serve distribution mismatch. This argument is independent
   of any synthetic probe.
2. 🟡 HIGH (structural risk): frequency is a *language* property, invariant to key placement.
   Trained only on QWERTY-family layouts — where frequency and geometry are CONFOUNDED
   (frequent bigrams sit in practiced/fast positions) — a freq feature lets the model absorb
   a disguised geometry bonus and hand it to frequent bigrams wherever the optimizer puts
   them. XGBoost + saturation makes this inert *today*; any magnitude-sensitive model behind
   the plug-and-play seam would be exposed.
3. 🟢 VERIFIED-DIRECTIONAL (probe v2, seed-swept): in the *confounded* synthetic world
   (time = geometry + a real log-freq practice bonus — the world the risk describes), the
   WITHOUT-freq model tracks the geometry-only ranking signal better on a novel layout in
   **12/12 seeds** (ρ 0.9828±0.0004 vs 0.9603±0.0012). In the freq-neutral world the two
   arms are statistically indistinguishable (0.9982 vs 0.9981) — i.e. dropping the feature
   costs nothing where freq is irrelevant and helps where it is confounded. Synthetic, so
   still not decisive — but directionally consistent and no longer single-seed.
4. 🟡 Practical: weight-only makes the objective corpus swappable per user (OQ-3) without
   retraining, and resolves audit #5 by deleting the constituent-freq features.

**Counter-position (unchanged, still open):** for a fully-proficient user, "frequent → fast"
(muscle memory) is real and layout-independent, so a properly-trained freq feature could
legitimately transfer. Empirical question; the harness decides.

## Post-mortem of the original probe (kept for honesty; do not cite it as evidence)

The first version of `experiments/oq1_freq_feature_probe.py` constructed a ground truth
where frequency had **zero causal effect** and reported that with-freq vs without-freq
models produced identical rankings. Fable-audit finding 1 (2026-07-04, CONFIRMED by the
author with an independent seed sweep):

- **Circularity:** in a world with no freq effect there is no freq-correlated variance for
  the tree to absorb, so "the model didn't use freq" is true by construction. The feared
  failure mode (confounded freq⇄geometry variance) was excluded from the world being tested.
- **Seed fragility:** across 20 seeds, 4 produced ranking disagreements (always the nearly
  tied graphite/semimak pair); the shipped probe's seed 0 happened to agree. Author's
  independent re-check: 1/4 additional seeds disagreed (seed 15). A single-seed "rankings
  agree" was overclaimed.

The probe file now runs BOTH worlds (freq-neutral and freq-confounded) across a seed sweep
and reports distributions, not a single lucky draw. Its output is illustrative; the
saturation measurement (evidence 1) remains the citable result.

## Definitive close (the experiment — decision rule TIGHTENED per fable-audit finding 2)

Prereqs: layout+participant labels in stroke rows (TODO P1) + the OQ-5 harness.

1. Process the real 136M dump with layout labels.
2. Train **A** (freq feature from the training data's own distribution) and **B** (freq
   pinned; weight-only) on {QWERTY, AZERTY, QWERTZ}; hold out Dvorak. Repeat with each
   layout held out (4-fold). **Each arm trained with ≥3 seeds** — conclusions must hold
   across seeds, not on one draw.
3. Metrics — two levels, with the layout level DECISIVE:
   - **Decisive:** layout-level ranking agreement — Kendall's τ (or pairwise accuracy) of
     predicted vs observed *layout ordering* across the 4 layouts, using each fold's held-out
     data. Rationale: an additive "frequent bigrams are faster everywhere" practice effect
     inflates per-bigram ρ for arm A while being **ranking-irrelevant** (it adds a
     layout-constant to Σw·t̂ over the common bigram set, cancelling in fitness differences).
     Per-bigram ρ alone would therefore confirm "keep" for a reason that cannot help the
     optimizer. (Fable-audit finding 2.)
   - **Supplementary:** per-bigram Spearman ρ on the held-out layout, computed on
     mean-centered predictions (removing the additive component), plus MAE.
4. Pre-registered decision rule:
   - If B ≥ A on the decisive layout-level metric (within bootstrap 95% CI, participant-level
     resampling) → **drop the feature**; execute the delete-freq fork, bump FEATURE_VERSION.
   - If A > B on the decisive metric clearly and consistently across folds and seeds → keep
     the feature AND immediately fix OQ-3 (single distribution for feature + weight).
   - If A > B **only** on per-bigram ρ but not on the layout-level metric → drop the feature
     (the gain is real fit, not real ranking power) and record the practice effect as a
     known, deliberately-unmodeled phenomenon.

Cost: one processing run + (2 arms × 4 folds × 3 seeds) = 24 trainings; still hours.
