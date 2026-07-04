# OQ-12 — How should typos shape what we keep, exclude, and model?

**Status: 🔴 open — the pipeline already has a principled *structural* answer (contiguity);
the open part is the *temporal* one: how long after a correction is timing contaminated?
Muscle-D is measuring the contamination radius and typo clustering on the real dump.**

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
