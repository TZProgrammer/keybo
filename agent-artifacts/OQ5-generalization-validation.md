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

**Metrics (per evaluation, reported as a matrix — see OQ-8):**
- Time error: R², MAE on held-out (positions, wpm) → duration.
- **Ranking**: Spearman ρ between predicted and observed per-ngram times on the held-out
  layout — this is the metric that matters for optimization (the SA only consumes order).
- Baseline floor: a dumb distance-only linear model. The learned model must beat it
  ON THE HELD-OUT LAYOUT, else the learning added nothing transferable.

**CLI:** `keybo validate --strokes ... --holdout dvorak` → prints the matrix; `just validate`.

**Statistical care:** bootstrap CIs over participants (not rows — rows within a participant
are correlated); report n per cell; refuse to print a cell with n below a floor.

## Acceptance criteria (what "the model generalizes" means, pre-registered)

- Held-out Spearman ρ ≥ 0.6 on every layout (order mostly right), AND
- learned model beats the distance-only baseline on held-out MAE for ≥3 of 4 layouts, AND
- no {layout × wpm-bucket} cell (OQ-8) catastrophically worse than the aggregate (< half the
  aggregate ρ).
Failing these = the model is a QWERTY model; optimization output should carry a loud caveat,
and OQ-1/OQ-7 remediation (drop freq feature, reweight training) becomes the priority.

## Definitive close

The question closes when the harness exists AND has been run on the real dump: the numbers
either meet the pre-registered bar (→ model trusted for novel layouts, claims licensed) or
they don't (→ documented limitation + remediation plan). Either outcome closes the question;
what's open today is that we cannot even ask it.

Estimated work: schema change ~0.5 day (touches keystrokes.py, strokes.py, TSV round-trip,
tests); validate.py + CLI ~1 day; real-data run: fetch (1.5 GB) + process + 8 trainings.
