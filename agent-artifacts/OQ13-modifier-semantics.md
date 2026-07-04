# OQ-13 — Modifier keys: SHIFT, CAPS_LOCK, and friends — what do they do to our data and objective?

**Status: 🔴 open — vocabulary confirmed on the real dump (SHIFT 7.4k, CAPS_LOCK 692,
ARW_*/CTRL/ALT present in a 300-file sample, plus 299 embedded-header "LETTER" rows);
mechanics measurements pending (muscle-C).**

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
