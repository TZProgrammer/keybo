# Open Questions

Unresolved *conceptual/modeling* questions — things where the right answer isn't obvious and
a wrong choice quietly produces worse layouts. These are distinct from `TODO.md` (concrete
work items). Each question records why it matters, the positions, a current lean, and how we
would actually resolve it (usually an experiment, not an argument).

Status legend: 🔴 open · 🟡 leaning · 🟢 decided (move the decision to a design doc + TODO).

---

## OQ-1 🟢 CLOSED, twice-refined (2026-07-04): freq = objective weight + explicit additive practice term (never a raw feature).

**Closed by the pre-registered decisive experiment** (real-data LOLO A/B through the OQ-5
harness): pinning freq **doubled** the pooled out-of-sample layout-ranking τ (+0.333 →
+0.667, all seeds), made every fold beat the distance baseline (4/4 vs ~1/4), and won the
per-fold τ everywhere — while the freq-live arm won only per-bigram ρ, the exact
"practice-effect fit, ranking-irrelevant" branch pre-registered as *drop the feature*.
REFINED same day after a correct user challenge (practice is real — model it, don't just
drop it): the winning arm R1W models `time = g(geometry) + b(bigram)` with b backfit
(shrinkage k=100) and residualized out of g's target, plus inverse-layout-share weights —
pooled τ +1.0 all seeds, ρ/ceiling .931, beats-baseline 12/12, passes the 0.8×-ceiling bar
on 3/4 layouts (qwerty .796 borderline). b is layout-independent so it cancels in ranking;
its job is cleaning g's training target. Details + tables:
`agent-artifacts/OQ1-frequency-feature.md`. Follow-up due: productionize R1W training
(currently in keybo-e2e/final_search.py), delete freq features from the schema (incl.
trigram constituents), bump FEATURE_VERSION.

**The distinction.** Frequency plays two roles that are easy to conflate:
1. **Weight** in the objective: `fitness = Σ time(ngram) × freq(ngram)`. Uncontroversial — we
   optimize for text people actually type. Keep.
2. **Feature** of the time model: `time = model(geometry, freq, wpm)`. **This is the question.**

**Why it matters for optimization (not just fit).** A bigram's frequency (e.g. `th` ≈ 9.7M)
is a property of the *language* — fixed no matter where T and H sit. The model is trained on
QWERTY-family layouts, so it can't cleanly separate "fast because frequent (muscle memory)"
from "fast because of where it sits on QWERTY (geometry)." When the optimizer moves keys and
still feeds the fixed frequency, the learned "frequent → fast" bonus is partly a disguised
*QWERTY-geometry* bonus applied to a layout nobody has practiced — biasing rankings toward
layouts that keep frequent bigrams in QWERTY-favorable spots. Fitting the training data well
(high R²) is **not** the same as ranking novel layouts well, which is the actual goal.

**Positions.**
- *Keep it (it transfers):* if we optimize for a hypothetical fully-proficient user, they'd
  have muscle memory for their language's frequent bigrams wherever those bigrams sit — so
  "frequent → fast" is a property of (language + proficiency) and transfers to any mastered
  layout. Frequency is then a legitimate, even necessary, feature.
- *Drop it (weight only):* model `time = f(geometry, wpm)` as pure biomechanics; let frequency
  live only in the weight. Removes the train/serve *semantic* mismatch, avoids overfitting to
  trained layouts, and makes the objective corpus swappable per user without retraining.

**Current lean: 🟡 weight, not feature.** Two supporting points: (a) the freq *feature*
saturates at corpus scale (see OQ-2 / audit finding #6) — it barely differentiates layouts as
a feature while staying load-bearing as a weight; (b) internal inconsistency (OQ-3). Cost of
dropping: lower training R² and reliance on a "proficient user" idealization.

**How to resolve:** leave-one-layout-out validation (see OQ-5). Train with vs. without the
frequency feature; whichever ranks the held-out layout's known relative speed better wins.
This is an experiment, and it depends on the eval harness (TODO).

**Blocks:** the "proper" fix for audit finding #5 (constituent bg/sg frequency features).
Keep = join real corpus freqs into training; Drop = delete those features. Until decided, the
scorer feeds the training-time default (1.0) on both sides — consistent, inert, not skewed.

---

## OQ-2 🔴 The `freq` feature saturates and is dual-purpose — is that acceptable, or a smell?

`freq` is used **both** as a model input feature *and* as the summation weight. As a feature it
saturates: XGBoost was trained on keystroke-data occurrence counts (~1–500 in realistic rows),
so every real corpus bigram (thousands→millions) lands in the same top bin and the feature is
effectively constant across the corpus — it can't help rank layouts. Largely a facet of OQ-1;
if OQ-1 resolves to "weight only," this dissolves. If we keep frequency as a feature, we must
at least (a) put the *feature* and the *weight* on the same distribution (OQ-3) and (b) handle
the scale mismatch (log-transform? train on corpus-scale frequencies?).

---

## OQ-3 🔴 Which frequency distribution — and should the feature-freq match the weight-freq?

Today the model is **trained** with frequencies from the 136M keystroke *dataset* (what
test-takers typed) but **scored/weighted** with *iWeb* corpus frequencies — two different
distributions. And (your point) neither necessarily matches a given user's real typing (code,
chat, non-English). Sub-questions:
- If frequency is a feature (OQ-1), the training feature-freq and the scoring weight-freq
  **must** come from the same distribution or the model is applied off-distribution.
- Should the objective corpus be **user-configurable** (weight by *your* text: prose vs. code
  vs. another language)? This is arguably the highest-leverage "best layout for ME" feature.
- iWeb is licensed/not-freely-redistributable; the derived frequency files are committed, but a
  swappable-corpus feature needs a documented way to generate new ones.

---

## OQ-4 🔴 Is the objective (predicted typing time) the right thing to optimize at all?

The whole pipeline optimizes *predicted speed*. But the original paper itself notes comfort,
effort, and injury-avoidance may matter more to users, and that speed gains are marginal (~6%).
Should the objective be multi-term (speed + effort + comfort penalties like SFBs/scissors/
redirects as first-class costs), and/or Pareto rather than single-scalar? Related: the geometry
features (SFB, scissor, LSB, redirect) are computed but only enter via their learned effect on
*time* — some may deserve to be explicit *comfort* costs regardless of predicted time.

---

## OQ-5 🟢 harness built + run (2026-07-04); freq-live model FAILED the bar, R1W ~passes.

**`keybo validate` / `just validate`** — leave-one-layout-out with a split-half noise
ceiling (participant-level), decisive layout-level Kendall's τ (incl. a pooled
fully-out-of-sample τ), bucket-centered per-cell ρ supplementary, a distance+wpm linear
baseline floor, ≥3 seeds; discrimination-tested against synthetic lawful/lawless worlds.

**Real-dump verdicts (same day, in order):** (1) the then-current freq-live model reached
only .59–.80 of each layout's ceiling, beat the dumb distance baseline on ~1/4 folds,
pooled τ +0.333 → the pre-registered "QWERTY-family model" caveat fired. (2) The
remediation arm matrix (see OQ-1) produced **R1W** — explicit additive practice term +
inverse-layout-share weights — which reaches pooled τ **+1.0 on all seeds**, beats the
baseline **12/12**, and clears the 0.8×-ceiling bar on 3/4 layouts (qwerty .796,
borderline). The de-confounded model also **reorders the named layouts** (dvorak becomes
the top named layout; colemak drops below qwerty; gains compress ~4×) — the harness caught
exactly the practice-inflation failure it was built to catch. Standing caveat: validation
spans only the 4-layout QWERTY-adjacent family; "transfers to an alien layout" remains
extrapolation. Numbers + honest readings:
`agent-artifacts/OQ5-generalization-validation.md`.

---

## OQ-7 🔴 How do we leverage the non-QWERTY data given the heavy class imbalance?

The dataset is *mostly* QWERTY, with a minority of AZERTY / Dvorak / QWERTZ. (Exact split:
**measure it** via `load_participant_metadata` once the dump is fetched — don't guess.) The
non-QWERTY rows are the most valuable data we have for OQ-1/OQ-5 (they're the only evidence of
how typing time behaves *off* QWERTY), so we must not let them be drowned out.

**Options (not mutually exclusive):**
- **Resampling with replacement (oversample minority layouts).** Simple, but adds no
  information — it just reweights, and it *inflates apparent confidence* on the minority
  layouts (a handful of Dvorak typists' idiosyncrasies get counted many times), risking
  overfitting to those specific people. Prefer for quick experiments, distrust for final numbers.
- **Class/sample weighting (inverse layout frequency in the loss).** Same balancing intent
  without duplicating rows; XGBoost supports per-sample weights. Usually preferable to
  resampling.
- **Stratified splits.** Ensure each layout appears in both train and test, so metrics are
  reportable per layout regardless of balance (feeds OQ-8).
- **Reframe: treat non-QWERTY purely as a held-out generalization signal (OQ-5)** rather than
  as training mass to balance. If the goal is "does it generalize," the minority layouts may be
  worth *more* as test than as (upsampled) train.

**Caution:** balancing interacts with OQ-1. If frequency is a feature, upsampling changes the
effective frequency distribution the model sees — another reason the frequency-as-feature
decision comes first. **How to resolve:** try weighting vs. resampling vs. none, and read the
*per-layout* held-out metrics (OQ-8) — the winner is whatever most improves minority-layout
generalization without wrecking QWERTY.

---

## OQ-8 🔴 What should the evaluation be sliced by — layout, and proficiency bucket?

A single aggregate R²/MAE **hides the exact failure modes we care about.** Two slicings, both
strongly worth having:

- **By layout (one score per layout).** Detects the central risk: a model that predicts QWERTY
  validation data beautifully but fails on Dvorak/AZERTY/QWERTZ has learned *QWERTY geometry*,
  not *typing* — and would produce confident, wrong layout rankings. This is the single most
  diagnostic slice. **Blocked on the schema change** (retain layout label; see OQ-5/OQ-8 in
  TODO).
- **By proficiency bucket (WPM bands).** Detects the other failure axis: maybe the model
  predicts slow typists well but fast typists poorly (or vice versa). This matters because the
  project deliberately targets an *above-average* WPM (the paper used ≥80 as a proficiency
  litmus), so accuracy *in the fast band* is what actually matters for the final layout. WPM is
  already retained per stroke sample, so this slice is **feasible today** — no schema change.

**Design implications:** the eval harness should report a matrix of {layout × wpm-bucket} →
{R², MAE, ranking error}, not one number. Report the *worst* cell prominently, not just the
mean — a great average with one terrible cell is a trap. This subsumes OQ-5's ranking check.

---

## OQ-6 🔵 Do we ever need non-row-stagger geometry (ortho / column-stagger / thumb keys)?

Decided *for now* during the rewrite design: **no** — the data is 30-key row-staggered, so any
other geometry would produce unvalidated extrapolation, and thumb/key-count changes have zero
supporting data. Geometry is isolated behind one object so this is extensible later. Re-open
only if we collect data on other physical layouts. (Kept here as a recorded decision, not an
active question.)
