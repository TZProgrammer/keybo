# OQ-5 — How do we validate that the model ranks NOVEL layouts well?

**Status: 🔴 open — this is the keystone. Nothing cross-layout is trustworthy until it
exists.** This artifact is the build spec; once the harness runs on real data, OQ-1 and OQ-7
close as byproducts and every "X% faster than QWERTY" claim gains (or loses) its license.

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
