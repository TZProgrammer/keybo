# OQ-8 — What should evaluation be sliced by (layout / proficiency)?

**Status: 🟡 design settled; implementation blocked on the layout-label schema change.** The
question is less "should we" (yes, clearly) than "what exactly does the report look like and
what counts as failure" — answered below.

## Best current answer

**Report a {layout × WPM-bucket} matrix of {R², MAE, Spearman ρ, n}; surface the WORST cell
as prominently as the mean.** Both slicings catch a distinct failure mode a single aggregate
number provably hides:

- **Layout slice** — catches "learned QWERTY, not typing": excellent QWERTY validation, junk
  Dvorak. This is the central risk of the whole project (OQ-1/OQ-5) and is invisible in
  aggregate because QWERTY dominates the row count. **Blocked** on retaining the layout label
  (today `Occurrence`/`StrokeRow` keep only positions/ngram/wpm/duration — verified).
- **Proficiency slice** — catches "fits slow typists, fails fast ones" (or the reverse).
  Matters doubly here because (a) the project intentionally optimizes for an above-average
  target WPM, so the fast band is the band that counts; and (b) the paper's own analysis says
  feature effects CHANGE with WPM (SFBs get much worse at high speed) — so uniform accuracy
  across bands is not expected by default. **Feasible today** (WPM is retained per sample).

Bucket proposal (align with the data, refine after measuring the WPM histogram — OQ-7's
unblocked step): `<40` (excluded by the pipeline's metadata filter anyway), `40–60`, `60–80`,
`80–100`, `100+`. Five-ish buckets keeps every cell populated; report `n` per cell and refuse
to print cells under a floor (say n<200 samples) rather than print noise.

**Statistical care:** resample by PARTICIPANT, not by row, for CIs (rows within a typist are
correlated); the fast buckets will be thin — that is a finding (how much high-WPM evidence do
we actually have?), not an inconvenience to hide.

## Definitive close

This closes by BUILDING it (it's part of the OQ-5 harness, same module):

1. Schema change (layout label) — shared prereq.
2. `keybo validate` prints the matrix; `--holdout <layout>` adds the LOLO column.
3. Acceptance: the matrix renders on real data with per-cell n; the worst cell is printed in
   the summary line; a regression-style test pins the report format.

Once it produces numbers, OQ-8 is closed as a question and becomes a monitoring practice:
every model change is judged on the matrix (worst cell first), never on a single aggregate.
