# OQ-11 — Which inter-key interval: press→press, press→release, release→press, release→release?

**Status: 🔴 open — analysis below; real-data overlap/hold measurements pending (muscle-C),
which mostly settle it.**

## The four candidates, for consecutive keys A then B

| name | definition | meaning |
|---|---|---|
| PP | press(B) − press(A) | initiation-to-initiation: the typing *rate* quantum |
| PR | press(B) − release(A) | "air time": gap between finishing A and starting B |
| RP | release(B) − press(A) | full envelope incl. B's hold |
| RR | release(B) − release(A) | completion-to-completion rate |

## Analysis

### The rollover problem kills PR as a target
Proficient typists overlap keystrokes: B is pressed *before* A is released (rollover). Then
PR is **negative** — not noise, but a systematic signature of skill and of comfortable
transitions (rolls). A model target that goes negative for exactly the best-executed
bigrams is hostile to squared-error learning and to the "time to type the corpus" summation
(negative times would *reward* stacking rollovers linearly, which doesn't compose —
you can't finish a sentence in negative time). PR is however an excellent *feature idea*
(rollover rate as a comfort/fluency signal — OQ-4 adjacent) — just not the target.
🟠 expectation: muscle-C will find PR<0 rates rising steeply with WPM; >30% negative in the
fast band would confirm this decisively.

### PP is the composable one
Sentence time = Σ PP intervals (telescoping) + first press offset + last hold. Neither
PR, RP, nor RR telescopes cleanly (holds and overlaps double-count or cancel obscurely).
Since the OBJECTIVE is corpus typing time = a sum over transitions, **the target must be
the quantity whose sum is the total: PP**. This is also what the pipeline already uses
('full' mode = last press − first press = Σ of interior PPs for the window).

### RR ≈ PP + (hold(B) − hold(A))
If hold times are roughly i.i.d. per key/person, RR is PP plus mean-zero noise → strictly
noisier same-signal. Muscle-C's PP↔RR correlation quantifies; expect r>0.9 with RR wider. 🟠

### Where release times DO matter
1. **Hold time itself** (release−press of one key) is a possible effort/comfort feature.
2. **Rollover rate** (fraction PR<0) per transition class — a fluency signature that could
   be a training feature or an OQ-4 comfort term.
3. **Modifier mechanics** (OQ-13): SHIFT's press/release bracket the shifted letter; you
   cannot parse capitals correctly from presses alone.
So: keep recording both timestamps in any schema change (we already have them in the raw
data; the TSV currently discards releases — **schema note**: carry hold time forward when
the OQ-5 schema change happens, it's one extra column and unblocks 1-2 above).

## Recommendation (pre-registered, pending muscle-C confirmation)

**Target = PP** (status quo, now with an argued basis rather than inheritance): it is the
only definition that (a) telescopes into corpus time, (b) stays positive under rollover,
(c) matches the objective's semantics. Close the question when muscle-C confirms: PP ≥ 0
everywhere (modulo data errors), PR negativity is common and WPM-correlated (disqualifying
it as target), RR adds hold-noise (r high but wider spread). Adopt hold-time and
rollover-rate as candidate FEATURES in the feature-pipeline backlog, not targets.
