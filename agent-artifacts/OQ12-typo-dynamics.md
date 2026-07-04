# OQ-12 — How should typos shape what we keep, exclude, and model?

**Status: 🟢 measured (2026-07-04) — decisions resolved below. Headline: speed recovery
after a correction is IMMEDIATE (radius ~1 key), but ERROR probability stays elevated ~5
keys; and duration caps would clip substantial legitimate hesitation.**

## Measured (muscle-D: 2000 qualifying files; 29.4k correction events)

- **Contamination radius:** the interval SPANNING the correction gap is 5.4× session median
  (scales with deletion size: 4.5× for 1-char, 8.6× for 3+), but the median interval is back
  within 10% of baseline at offset **+1** — and the interval BEFORE the typo is itself
  elevated ~1.23× (the stumble precedes the error). The legacy's hardcoded "2 keys after"
  was roughly right by accident; the measured radius is ~1 (the correction-spanning window
  itself, which the contiguity rule ALREADY drops).
- **Typo-begets-typo: REAL but about errors, not speed.** Second-correction hazard 1.7× base
  at k=1, decaying to ~1.25× at k=5–10. Since a second error breaks windows structurally, no
  timing action needed — but error-adjacent stretches are less trustworthy for ~5 keys.
- **Hesitation tails (clean stretches):** 2.9% of intervals >3× median, 0.62% >5×, 0.04%
  >10× — and these hold **11.4% of clean typing TIME above 3×** (3.7% above 5×). A duration
  cap at 3–5× would clip a large share of legitimate hesitation time.

## Resolved decisions (pre-registered criteria applied)

1. **Exclusion radius: no new exclusion needed.** Measured recovery at +1 means the
   contiguity rule (which already drops the correction-spanning window) suffices. The
   legacy-style "N keys after" filter is NOT reinstated.
2. **Clustering: no radius extension** — hazard 1.7× at k=1 is below the pre-registered 2×
   threshold, and structure already handles repeat errors.
3. **No duration caps** — confirmed: gate typo-vs-hesitation by BKSP-proximity (already the
   mechanism), never by a duration threshold; IQR-mean aggregation keeps handling the tail.

Report: `state/keybo-muscle-d/artifacts/report.md` (probe_2_typo.py). Original analysis kept
below for the reasoning record.

## What the pipeline already does (post the splice-fix series)

- A mistyped character, control key (BKSP/SHIFT/arrows), or malformed row occupies a stream
  index; any n-gram window whose keys aren't **originally consecutive** is rejected. So no
  window ever spans a correction — the `('ab', 4100ms)` class is dead.
- Only correct-aligned characters (difflib vs the expected sentence) form n-grams at all.

That's the structural containment. Three open temporal/behavioral questions remain:

## 1. Post-correction contamination: how many keys after a typo are "dirty"?

After BKSP+retype, the typist re-orients: the next few intervals are slower and more
variable than their clean baseline. Windows that START right after a correction are
currently kept (their keys are consecutive) — but their *timing* may not represent the
transition's true cost. 🟠 expectation: elevated intervals for ~2-5 keys post-correction,
decaying fast. Muscle-D measures the profile at offsets +1..+10 (normalized to session
median): the offset where it returns within ~10% of baseline is the principled exclusion
radius. The legacy code hardcoded "2 keys after a backspace" with no evidence; we'll have
the measured number. Cost of exclusion is small (typos are localized) — IF the measured
radius is small.

## 2. Does "typo begets typo" hold — and does it matter for us?

The user's intuition: post-typo stretches have elevated typo probability, so a window
there is doubly suspect. Muscle-D measures the hazard ratio (P(second BKSP within k keys)
vs base rate). Note the structural fix already handles the *worst* consequence (a second
error breaks windows again); what clustering would add is justification for a slightly
larger exclusion radius after corrections. If the hazard ratio is ≈1 beyond k≈3, no change.

## 3. What about local-window contamination (ties to OQ-9)?

If OQ-9 adopts local-pace normalization, a pace window overlapping a correction inherits
its slowdown. Cheap robust default: median (not mean) within the window + skip windows
containing any non-character index gap. Muscle-D's contamination radius bounds how much
this matters.

## Hesitations are not typos

Even in typo-free stretches, intervals have a heavy right tail (attention lapses, reading
ahead). Muscle-D measures the tail (frac > 3×/5×/10× session median in clean stretches).
This bounds the effect of a duration CAP — the blunt alternative to exclusion windows.
🟡 lean: prefer *robust aggregation* (we already IQR-mean per (bigram, wpm) group) over
hard caps; the IQR-mean already clips the tail, which is why the splice fix (which produced
mid-tail values like 4100ms rather than extreme ones) mattered more than a cap would.

## Modifier-adjacent corrections (cross-ref OQ-13)

CAPS_LOCK typos ("HELLO" → correct → retype) produce long BKSP runs; SHIFT slips produce
single-char corrections. Both are handled structurally (windows break); their timing
contamination is covered by the same measured radius.

## Pre-registered decisions

1. Adopt a post-correction exclusion radius = the measured return-to-baseline offset
   (round up), replacing the legacy's unevidenced "2". Implement as: reject windows whose
   FIRST key is within R indices after a non-character/incorrect index. Test with a fixture
   reproducing the measured profile shape.
2. If typo hazard ratio > 2 at k ≤ 3: extend R by the clustering window; else ignore.
3. No duration caps while IQR-mean aggregation stands; revisit only if muscle-D's tail
   numbers show the IQR window itself is being dragged (>10% of clean intervals beyond 3×).
