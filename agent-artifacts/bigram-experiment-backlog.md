# Bigram-world experiment backlog (2026-07-05)

Now that the LOLO harness gives a real generalization signal, every idea below is cheap to
adjudicate: run the arm, read pooled τ + mean ρ/ceiling + beats-baseline, adopt or kill.
**Standing decision rule:** adopt over the shipped recipe only if pooled held-out τ ≥ the
incumbent's AND mean ρ/ceiling beats it; ties break toward the simpler model. ≥3 seeds.
Scope: exhaust bigrams first; trigrams after (their objective questions are OQ-10).

## Context: how identifying are today's features? (measured)

All 961 ordered position pairs → **765 distinct feature vectors** (184 collision classes,
380 pairs sharing). So yes — the geometry features are a *near*-ID for the position pair,
and within qwerty (one position per bigram) they are a near-ID for the bigram. Two
consequences, one benign, one not:

- **Position-ID is not a bug.** Position is the optimization variable; the model's job is
  literally a position-pair → time table (the QAP reduction). Memorizing *positions* is
  the objective. The danger was never "ID-ness" per se — it was an ID that **travels with
  the bigram** (freq) rather than **with the position** when the optimizer moves keys.
- **But within qwerty, position ≡ bigram** — so geometry features absorb bigram-identity
  (practice) effects there, which is (a) why the practice term helps so much, (b) why the
  cross-layout rows are precious (same bigram, different position = the only signal that
  separates the two), and (c) why the qwerty fold is the weakest (.80 of ceiling): models
  trained on the other three layouts can't lean on the memorized confound.
- The 184 collisions are the features' deliberate abstractions: hand-mirroring (|x|),
  unsigned dx/dy, and same-row rolls where inwards/outwards both gate to 0. One genuine
  gap found: **pinky→ring and middle→ring same-row rolls produce the identical vector**
  (no first-key finger feature) — pinky-initiated motion is plausibly slower.

## A. Feature engineering arms

- **A1. First-key one-hots** (row + finger of key 1). Fixes the measured collision above.
- **A2. Hand indicator** (left/right). Mirrored pairs currently share parameters — good
  for data efficiency, but assumes symmetric hands; most typists are right-handed.
- **A3. Signed dx + directional dy** (up vs down motion differ biomechanically; both are
  absolute today).
- **A4. Finger-pair identity** (finger(a)×finger(b), 5×5 incl. thumb, or hand-abstracted).
  The classic typing-model parameterization; subsumes A1/A2 at higher dimension.
- **A5. LESS positional info** (ablation): drop the second-key row/finger one-hots, keep
  only relational+geometric features. Tests whether abstraction *helps* transfer (the
  distance+wpm baseline's strength suggests the smooth core carries most signal).
- **A6. MORE positional info** (other endpoint): full per-key one-hots for both keys
  (62 cols). Maximally identifying — likely wins ρ in-family and loses τ; worth one run
  to map the memorization⇄abstraction frontier.
- **A7. Explicit interactions**: same_finger×distance, scissor×dy, lsb×dx. Trees can form
  these, but shallow trees (C2) can't — pairs with C2.
- **A8. Hold-time / rollover position aggregates**: per-position mean hold and per-pair
  rollover rate, aggregated from TRAINING data, keyed by position (serve-computable).
  OQ-11's carry-forward. Caveat: position-keyed data aggregates reintroduce a mild
  memorization channel — LOLO decides if it's net-positive.

## B. Target / context engineering

- **B1. Participant normalization**: target = duration / participant-median (or z-score);
  serve-time multiplies back a reference pace. Removes between-typist scale variance that
  session-WPM only coarsely captures. (OQ-9 killed *local* pace; per-PARTICIPANT scale is
  untested and the pid column now exists.)
- **B2. Log-target**: train on log(duration) — multiplicative effect composition, tames
  the heavy right tail (hesitations) that IQR-mean only clips.
- **B3. WPM-relative target**: duration × (wpm/60) ≈ "beats at own pace" — makes the
  target dimensionless across typists; serve rescales by target wpm.
- **B4. Practice-term variants**: hierarchical toward the freq curve (R3W — in flight);
  wpm-bucketed b(ngram, wpm-band) (still layout-independent ⇒ still cancels);
  letter-level + bigram-level decomposition (b = u(a)+u(b)+pair residual).
- **B5. Multiplicative practice** time = g(pos)·m(bigram): algebraically equivalent to
  reweighting the corpus (Σ f·m·g = Σ f'·g), so it changes *which bigrams the objective
  cares about*, not per-position times. A modeling question about whether practice scales
  or shifts time — worth one arm.

## C. Model class / hyperparameters

- **C1. Retarget `tune` at the harness.** The current tune CLI optimizes pooled CV MAE —
  which *rewards* memorization (the exact failure LOLO exists to catch). Make the tuner's
  objective mean LOLO ρ/ceiling (or τ-gated ρ). Highest-leverage item in this file: it
  turns every other knob into something optimizable for the right thing.
- **C2. Shallow trees** (max_depth 2–3): fewer interaction slots = less memorization.
- **C3. Monotonicity constraints** (xgboost native): time non-decreasing in distance,
  same_finger, scissor. Physics-shaped regularization; should specifically help transfer.
- **C4. GAM / linear-basis model as the MODEL**: the distance+wpm baseline was hard to
  beat pre-practice-term — a smooth additive model with the full feature set may transfer
  better than trees. Plug-and-play seam makes this cheap.
- **C5. Seed-ensemble as a first-class model** (average ≥3 seeds; the search already does
  this informally via the mean table).

## D. Data-side

- **D1. Threshold sensitivity**: harness loads wpm_threshold=0/min_samples=1 while the
  train CLI defaults 60/25 — reconcile, then sweep (does dropping slow/thin rows help?).
- **D2. Aggregation variants**: per-participant-first pooling; winsorize vs IQR-mean.
- **D3. Case-folding** (OQ-13): recover the 6.1% capital weight for the objective.
- **D4. Example weighting by sample count** (beyond layout balance).

## E. Evaluation hardening (do alongside)

- **E1. Bootstrap CIs** (participant-level) on τ and ρ — needed as arms get closer.
- **E2. OQ-8 worst-cell matrix** — a great mean with one terrible {layout×wpm} cell is a trap.
- **E3. Qwerty-fold deep-dive** — the borderline fold; which bigram classes miss?
- **E4. Calibration slope per fold** (user question, 2026-07-05): regress observed on
  predicted held-out cell times; want slope ≈ 1. Rank metrics (τ, ρ) are blind to
  dynamic-range COMPRESSION — perfect ranks with squashed gaps mis-weight the optimizer's
  trade-offs (fitness is a weighted sum; relative gaps are load-bearing). MAE-vs-baseline
  only partially guards this; the slope isolates it.

## Sequencing

1. (in flight) red-flag tests + F20W/R3W arms.
2. **C1 tune-retarget** + **feature arm matrix** (A1–A5, C2, C3 — one driver, shared folds).
3. Target engineering (B1–B3) on the winner.
4. E1/E2 hardening; then the trigram world (OQ-10) with everything learned.
