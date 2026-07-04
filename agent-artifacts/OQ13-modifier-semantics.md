# OQ-13 — Modifier keys: SHIFT, CAPS_LOCK, and friends — what do they do to our data and objective?

**Status: 🟡 measured (2026-07-04) — mechanics quantified; two decisions resolved, one
(case-folding) now has its blocking number and needs a final call.**

## Measured (muscle-C: 2000 files / 1.46M rows)

- **SHIFT costs 2.52× the median inter-key interval** (424ms vs 168ms; 1.9–3.1× per key) —
  capital-adjacent transitions are HEAVILY contaminated, validating the current exclusion.
- **Chord mechanics:** 88.7% of capitals via SHIFT-hold, 9.8% CAPS_LOCK, 1.5% neither
  (autocorrect/paste?); in 98.5% of shift-paired capitals SHIFT is DOWN at the letter press
  (a true chord — parsing capitals from presses alone is impossible, as predicted).
  CAPS_LOCK appears in 9.3% of sessions.
- **Data quirk resolved:** ZERO literal-"LETTER" header rows in 2000 files — the parent's
  earlier 299-row observation was itself an artifact of the csv-quote corruption (below).
  Under correct parsing only 14/1.46M rows (0.001%) are malformed. The planned header-row
  ingest filter is therefore unnecessary; the rejection-counter still is.
- **🔴→🟢 INGEST BUG found by this measurement (fixed, tested):** the dump contains literal
  double-quote keystrokes (LETTER = `"`, keycode 222). csv's DEFAULT dialect treats that as
  an opening quote and swallows the rest of the line PLUS following rows into one field —
  silent corruption of every quote-containing session. Fixed with `quoting=csv.QUOTE_NONE`
  on both DictReader sites + a regression test reproducing the swallow.

## Resolved / updated decisions

1. Chord modeling stays out of scope (unchanged; data-ceiling logic).
2. **Case-folding for the objective weight:** the 2.52× slowdown is about the *training*
   side (we keep excluding capital-adjacent transitions there — now with a measured basis).
   For the *scoring* weight, folding ' I'→' i' etc. recovers ~6.1% of corpus weight at the
   cost of ignoring SHIFT-hand asymmetry. Leaning ADOPT for the weight; final call bundled
   into the OQ-5 harness round (it changes absolute fitness, not obviously ranking).
3. Header-row filter: dropped (measured nonexistent). Rejection-counter: still recommended.
4. Capital-adjacent training exclusion: KEEP — now evidence-based (2.52×).

## The three distinct problems modifiers create

### 1. Parsing: a capital is TWO-KEY CHORD, not a keystroke
"W" arrives as press(SHIFT) → press(w) → release(w) → release(SHIFT) (or messier: SHIFT
released early/late; CAPS_LOCK toggling instead). Today:
- SHIFT/CAPS_LOCK rows are multi-char LETTER fields → they break n-gram windows
  (contiguity), and the capital letter row itself ("W") is a single char that participates.
- Consequence: any bigram TOUCHING a capital is currently **excluded** (the SHIFT press
  between the previous key and the capital creates an index gap). That is *safe* but
  *lossy*: sentence-initial capitals discard the first transition of nearly every sentence.
- The alternative — modeling shifted keystrokes as chords (finger occupancy: right-SHIFT
  while left hand types) — is a real extension of the geometry (a 31st..32nd key + hand
  constraint). It's ALSO what a 30-key objective can't score today (capitals are 6.1% of
  corpus weight, currently skipped at scoring — audit finding, documented).

### 2. Timing: does SHIFT slow the surrounding transitions?
Muscle-C measures shifted-vs-unshifted inter-key intervals. 🟠 expectation: yes,
noticeably; the chord costs both mechanical time and coordination. If large, then even for
a lowercase-only objective, transitions ADJACENT to capitals are systematically slowed —
another argument for the current exclusion (we're excluding contaminated timings, not just
unparseable ones).

### 3. Objective: should layouts be scored on shifted characters at all?
The corpus has ' I', 'Th' etc. (6.1% of weight). Options:
a) **Fold case**: map capitals to their lowercase key for scoring (the fingers travel to
   the same key; SHIFT cost becomes a constant offset ≈ layout-independent for a fixed
   SHIFT position). Recovers most of the 6.1% at the cost of ignoring chord asymmetry
   (left-SHIFT vs right-SHIFT choice — which IS layout-dependent for which hand types the
   letter). 🟡 lean: adopt for the weight, with the SHIFT-hand asymmetry noted as a
   comfort-term candidate (OQ-4).
b) Model SHIFT as a key: honest but expands the geometry and needs chord data (see 1).
c) Status quo: skip capitals, print the coverage line (what we do; transparent, lossy).

## Data quirks confirmed on the real dump (2026-07-04)

- Multi-char vocabulary (300-file sample): BKSP 9781, SHIFT 7414, CAPS_LOCK 692,
  ARW_LEFT/RIGHT/UP, CTRL, ALT, NUM_LK, DELETE, INSERT, WIN, HOME — all handled by the
  contiguity rule (window-breaking).
- **299 rows whose LETTER is literally "LETTER"** — embedded header rows (files
  concatenated with headers repeated). Harmless today (multi-char → window-break; PRESS_TIME
  "PRESS_TIME" → float() fails → row unusable anyway) but worth an explicit ingest filter +
  a rejection counter so real-data hygiene is visible. (Fold into the rejection-counter
  item.)
- Short rows (None fields) — already fixed and tested (commit 6e3fa08).

## The rejection-counter principle (meta-fix, recommended regardless)

Every filter that drops data should COUNT what it drops, and `process-data` should print
the breakdown (N windows dropped: banned-key / non-contiguous / off-layout / unparseable-
time / post-correction). One dict, near-zero cost, and it turns every future data quirk
from a silent skew into a visible number. This is the single cheapest change with the
highest debugging value in the whole data path. 🟢 do it with the schema change.

## Pre-registered decisions

1. Keep chord modeling OUT of scope until non-QWERTY/thumb-style data exists (same
   data-ceiling logic as OQ-6); record left/right-SHIFT asymmetry as an OQ-4 comfort
   candidate.
2. Adopt case-folding for the objective weight (option a) IF muscle-C shows the
   SHIFT-adjacent slowdown is roughly uniform across letter keys (i.e. folding doesn't
   distort per-key geometry signal); else keep skipping capitals and say so.
3. Add the "LETTER"-header ingest filter + the rejection-counter breakdown with the OQ-5
   schema change.
4. Training rows: keep excluding capital-adjacent transitions (measured contamination
   basis pending muscle-C) — revisit only if the slowdown proves negligible.
