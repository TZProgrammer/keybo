# OQ-1 — Should n-gram frequency be a model FEATURE, or only an objective WEIGHT?

**Status: 🟡 leaning weight-only.** New evidence below (a controlled synthetic probe) plus the
saturation measurement move this from "argument" to "supported lean". Definitive closure needs
the real-data LOLO harness (OQ-5).

## Best current answer

**Keep frequency as the objective weight; drop it as a model feature — pending one real-data
confirmation.** Reasoning, strongest-first:

1. 🟢 VERIFIED (audit finding #6): at corpus scale the freq feature **saturates** — the model
   was trained on keystroke-occurrence counts (~1–500), so every real corpus bigram (10³–10⁷)
   lands in the same top bin; predictions are flat in freq across the entire scoring range
   (164.8→188.2 ms over freq 1→500, then *constant* from 500→9.7M). A feature that is
   effectively constant at serve time cannot help rank layouts — it only adds a train/serve
   distribution mismatch.
2. 🟢 VERIFIED (new probe, `experiments/oq1_freq_feature_probe.py`): in a controlled synthetic
   world where typing time depends ONLY on geometry, training with vs. without the freq
   feature produced **identical layout rankings** (and r=0.9994 per-bigram prediction
   agreement on a held-out novel layout). I.e. in the regime our data actually occupies
   (freq-saturated trees), the feature adds ~nothing to ranking — so the *cost* of dropping it
   is empirically small, while the *risk* of keeping it (below) is structural.
3. 🟡 HIGH (structural risk): frequency is a *language* property, invariant to key placement.
   Trained only on QWERTY-family layouts, a freq feature lets the model attribute
   QWERTY-geometry speed to "frequent → fast", then hand that bonus to frequent bigrams
   *wherever the optimizer puts them* — a disguised geometry bonus for unpracticed layouts.
   The probe could not exhibit this (XGBoost + saturated freq ≈ inert), but a
   magnitude-sensitive model behind the same seam (linear/kNN/NN) would be exposed. The seam
   is advertised as plug-and-play, so the hazard is real even if today's model dodges it.
4. 🟡 Practical: weight-only makes the objective corpus swappable per user (OQ-3) without
   retraining, and resolves audit #5 by *deleting* the constituent-freq features instead of
   building a train-time corpus join.

**Counter-position (why not closed yet):** for a *fully-proficient* user, "frequent → fast"
(muscle memory) is real and layout-independent, so a freq feature could legitimately transfer
— IF trained on frequencies drawn from the same distribution used at serve (OQ-3 fix) and IF
it improves novel-layout ranking. That is an empirical claim; test it, don't debate it.

## What was run (evidence)

- `experiments/oq1_freq_feature_probe.py` — synthetic geometry-only world; with-freq vs
  without-freq models agree on ranking (5 named layouts) and correlate r=0.9994 on a novel
  layout. Limitation: synthetic target, single seed, XGBoost-specific.
- Saturation sweep (delta-audit finding #6, reproduced): predictions flat in freq above ~500.

## Definitive close (the experiment)

Prereqs: layout label retained in stroke rows (TODO P1 schema change) + LOLO harness (OQ-5).

1. Process the real 136M dump with layout labels.
2. Train **A** (with freq feature, freqs drawn from the *training data's own* distribution)
   and **B** (freq pinned; weight-only) on {QWERTY, AZERTY, QWERTZ}.
3. Evaluate both on held-out **Dvorak** (the most geometrically distinct layout):
   - per-bigram time R²/MAE, and
   - Spearman rank correlation of per-bigram predicted vs observed times.
4. Decision rule (pre-registered):
   - If B ≥ A on Dvorak ranking (within noise, bootstrap 95% CI) → **drop the feature**
     (OQ-1 closed: weight-only). Execute TODO P1 fork: delete `freq`-family features, bump
     `FEATURE_VERSION`.
   - If A > B clearly → keep the feature, and immediately fix OQ-3 (single distribution for
     feature and weight) since keeping it makes the mismatch load-bearing.
5. Repeat with each layout held out (4-fold) to check the conclusion isn't Dvorak-specific.

Cost: one processing run + 8 model trainings; hours, not days, once the harness exists.
