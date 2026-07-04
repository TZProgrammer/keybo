# OQ-5 — How do we validate that the model ranks NOVEL layouts well?

**Status: 🟢 harness BUILT and RUN on the real dump (2026-07-04). Verdict: the current
model FAILS the pre-registered transfer bar — the "QWERTY-family model" caveat fires —
and the freq-feature A/B run through the same harness closes OQ-1 as weight-only.**

The schema change (layout/pid/hold; commits `10c1fa9..4a936e7`) and the harness
(`keybo validate`, commit `3eb7d0f`) are discrimination-tested against synthetic worlds
with known answers (a geometry-lawful world must show transfer; a lawless holdout must not
— a harness that can't tell those apart would pass any model).

## Real-data verdict (bistrokes_v3.tsv; seeds 0/1/2; wpm [40,140)×20; cell floor 10)

**Data context:** qwerty 31.2M samples / 54,690 participants; qwertz 277k / 485; azerty
92k / 166; dvorak 37k / 64 — the non-QWERTY folds are 100–1000× smaller, which is why every
ρ is judged against that layout's own split-half ceiling, never as a bare number.

**Arm A — current model (freq feature live), `runs/lolo_v3.json`:**

| holdout | cells | ceiling | ρ (frac of ceiling) | τ_all | beats dist+wpm baseline |
|---|---|---|---|---|---|
| azerty | 1077 | .780 | .61–.62 (.78–.80) | +.667 | 1/3 seeds |
| dvorak | 819 | .669 | .41–.43 (.62–.64) | +.333 | 0/3 |
| qwerty | 3833 | .964 | .63–.64 (.65–.67) | +.333 | 2/3 |
| qwertz | 1627 | .837 | .50–.52 (.59–.62) | +.333 | 0/3 |

Pooled fully-out-of-sample layout-ranking τ: **+0.333** (all seeds). Seed spread < 0.03 ρ.

**Arm B — freq pinned to 1 (feature constant-folded ⇒ no split possible),
`runs/lolo_v3_nofreq.json`, same config:**

| holdout | ρ (frac of ceiling) | τ_all | beats baseline |
|---|---|---|---|
| azerty | .55–.56 (.70–.72) | **+1.000** | 3/3 |
| dvorak | .30–.31 (.45–.47) | **+1.000** | 3/3 |
| qwerty | .59 (.61) | **+1.000** | 3/3 |
| qwertz | .65 (.78) | +.667 | 3/3 |

Pooled fully-out-of-sample τ: **+0.667** (all seeds).

## Against the pre-registered acceptance criteria

1. **ρ ≥ 0.8 × ceiling on every layout: FAIL** in both arms (best: azerty .80 in A,
   qwertz .78 in B; dvorak is worst everywhere).
2. **τ > 0 in every fold: PASS** (both arms; B far stronger: +1.0 on 3/4 folds).
3. **Beats the distance-only baseline on ≥3 of 4 layouts: FAIL for A (~1/4), PASS for B
   (4/4, all seeds).**
4. {layout × wpm} worst-cell check: deferred to the OQ-8 slicing report (cells exist).
5. **Stable across ≥3 seeds: PASS** (both arms; spreads ≲0.03).

**Consequence (pre-registered, fires as written):** the current-schema model is a
QWERTY-family model — optimization output must carry a loud caveat, and OQ-1/OQ-7
remediation is the priority. The B-arm shows the single biggest remediation is already
identified: **the freq feature was actively hurting cross-layout ranking** (it feeds
memorization of the practiced-position confound; dropping it doubles pooled τ and flips
beats-baseline to 4/4). OQ-1 closes accordingly (see OQ-1 artifact).

## Honest readings, not spin

- "Loses to the baseline" does NOT mean nothing transfers: the baseline is itself a
  geometry model (distance+wpm), and centered ρ of .3–.65 is real signal. It means
  XGBoost's 19 extra features currently add ≈nothing *transferable* beyond distance.
- Dvorak — the only non-QWERTY-family board — is the worst fold in both arms (B's ρ is
  *lower* there while its τ is perfect: the model gets the layout's overall standing right
  but not its within-layout bigram ordering). Consistent with family memorization.
- The τ metric is coarse with 4 layouts (one inversion = −0.333) and confounded by
  population (dvorak's 64 typists are enthusiasts, qwerty's 54k are everyone), so
  ρ-vs-ceiling is the cleaner per-layout signal; τ is decisive only because it is the
  quantity the optimizer actually consumes.
- Arm B was freq *pinned*, not schema-deleted — functionally equivalent for XGBoost (a
  constant column can never split) but the real deletion + FEATURE_VERSION bump is the
  follow-up item (TODO P1), which also kills the constituent-freq landmine (audit #5).

## What this changes

1. **Every cross-layout % in this repo now carries the caveat explicitly** (score/optimize
   claims are within-model comparisons, not validated human-time predictions).
2. OQ-1: **closed, weight-only** — delete freq features, bump FEATURE_VERSION.
3. OQ-7 reweighting experiments are now cheap to judge: run `keybo validate` on each arm.
4. The remediation target is concrete: close the ρ gap to ceiling on dvorak (.45 → .8×.67)
   — feature work (hold-time, rollover-rate from OQ-11; comfort terms OQ-4) and OQ-7
   weighting are the levers, judged by this same harness.

## Why it's THE gate

The model's job during optimization is to judge layouts *no human has ever typed on*. Fitting
the four training layouts well is compatible with being systematically wrong off-QWERTY (the
QWERTY-geometry-memorization failure mode). The only evidence the data can give us is
cross-layout transfer: hide one layout entirely, predict it from the others.

## Harness design (build spec)

**Data prep (prereq — TODO P1 schema change):** stroke rows must carry their source layout.
Aggregation key becomes `(layout, positions, ngram)`; TSV gains a `layout` column; `StrokeRow`
gains a `layout` field. (Also unlocks OQ-8's per-layout slicing with the same change.)

**Split module (`keybo/training/validate.py`):**
- `leave_one_layout_out(rows, holdout)` → (train_rows, test_rows)
- `stratified_split(rows, by=("layout", wpm_bucket), test_frac)` for within-layout eval.

**Metrics (per evaluation, reported as a matrix — see OQ-8). Tightened per the fable audit
(2026-07-04, findings 2+3):**
- **Noise ceiling FIRST (harness pre-step):** split-half ρ — randomly bisect the held-out
  layout's *participants*, correlate the two halves' mean per-bigram times. This is the
  maximum ρ any model could achieve on this data; every ρ threshold below is expressed as a
  fraction of it, not as an absolute number (an absolute 0.6 is unjustifiable: a genuine
  pass if the ceiling is 0.7, a very low bar if it's 0.95).
- **Decisive metric — layout-level ranking:** Kendall's τ / pairwise accuracy of predicted
  vs observed *layout ordering*. Rationale: an additive practice effect ("frequent bigrams
  faster everywhere") inflates per-bigram ρ while being ranking-irrelevant — it cancels in
  fitness differences over a common bigram set. Per-bigram metrics alone can therefore
  reward fit that gives the optimizer nothing.
- **Supplementary:** per-bigram Spearman ρ (on mean-centered predictions) + R²/MAE on
  held-out (positions, wpm) → duration.
- **Baseline floor:** a dumb distance-only linear model. The learned model must beat it ON
  THE HELD-OUT LAYOUT, else the learning added nothing transferable.
- **Seeds:** every trained arm runs ≥3 seeds; a conclusion must hold across seeds (the
  original OQ-1 probe's single lucky seed is the cautionary tale).

**CLI:** `keybo validate --strokes ... --holdout dvorak` → prints the matrix; `just validate`.

**Statistical care:** bootstrap CIs over participants (not rows — rows within a participant
are correlated); report n per cell; refuse to print a cell with n below a floor.
**Reproducibility pin:** harness trainings run `device=cpu`, `n_jobs=1`, pinned xgboost
version recorded in model metadata — else LOLO results won't reproduce across machines.

## Acceptance criteria (what "the model generalizes" means, pre-registered)

- Held-out per-bigram ρ ≥ **0.8 × the split-half noise ceiling** on every layout, AND
- layout-level ranking (decisive): predicted ordering of the 4 layouts matches observed
  with Kendall's τ > 0 in every fold (no inverted rankings), AND
- learned model beats the distance-only baseline on held-out MAE for ≥3 of 4 layouts, AND
- no {layout × wpm-bucket} cell (OQ-8) catastrophically worse than the aggregate (< half the
  aggregate ρ), AND
- all of the above stable across ≥3 training seeds.
Failing these = the model is a QWERTY model; optimization output should carry a loud caveat,
and OQ-1/OQ-7 remediation (drop freq feature, reweight training) becomes the priority.

## Definitive close

The question closes when the harness exists AND has been run on the real dump: the numbers
either meet the pre-registered bar (→ model trusted for novel layouts, claims licensed) or
they don't (→ documented limitation + remediation plan). Either outcome closes the question;
what's open today is that we cannot even ask it.

Estimated work: schema change ~0.5 day (touches keystrokes.py, strokes.py, TSV round-trip,
tests); validate.py + CLI ~1 day; real-data run: fetch (1.5 GB) + process + 8 trainings.
