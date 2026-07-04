# OQ-10 — What should the trigram model actually predict?

**Status: 🔴 open — three candidate targets analyzed; decision needs the OQ-5 harness (the
candidates are cheap to A/B once it exists).**

## The question

For trigram ABC, at least three defensible prediction targets exist:
1. **Full span**: press(C) − press(A) — what the current pipeline records (`time_mode=full`).
2. **Sum of parts**: predict bigram AB and bigram BC separately and add (what the legacy
   Freya cost function did: C(t) = C(b1) + C(b2) + skip-penalty).
3. **Conditioned second step**: predict press(C) − press(B) *given* A — i.e. the second
   bigram's time, conditioned on the preceding key (context-conditioned bigram).

## Analysis

### The identity that frames everything
press(C)−press(A) = [press(B)−press(A)] + [press(C)−press(B)] — the full span IS the sum of
the two bigram intervals, *per occurrence, exactly*. So target 1 vs target 2 is not about
different quantities; it's about **where the model error lands**:
- Predicting the full span (1) lets the model learn interactions between the halves
  (a redirect makes BOTH transitions slower than their independent estimates) but doubles
  the target's variance (sum of two noisy intervals) and halves the effective sample count
  vs bigram rows.
- Sum-of-parts (2) reuses the (much larger) bigram evidence and makes trigram-level features
  (redirect, SFS) into *corrections* on top — structurally cleaner, and it's how the paper
  conceived it. Its weakness is exactly the interactions (2) can't see unless we add an
  explicit interaction correction term.
- **Conditioning (3) is the statistically sharpest formulation**: the marginal cost of
  transition B→C given the hand/finger state A left behind. Note (3) composes into a
  full-word model: time(word) = Σ time(kᵢ→kᵢ₊₁ | kᵢ₋₁), a proper chain decomposition —
  whereas (1) over overlapping trigrams DOUBLE-COUNTS every interior bigram when summed
  over a corpus (trigrams 'the'+'hea' both contain 'he'). 🟡 This double-count is baked
  into today's trigram objective: Σ_tg freq·T_full(tg) counts each interior transition
  twice. It's a consistent-over-layouts distortion (so ranking damage is muted), but it's
  the kind of unprincipled weighting OQ-1 taught us to distrust.

### Serve-time coherence
The optimizer needs Σ over the corpus. Under (3): fitness = Σ_{abc} freq(abc)·t(bc|a) —
every transition counted once, context-weighted. Under (1): interior transitions counted
twice (above). Under (2): equals bigram model + corrections, i.e. the trigram model
*degenerates gracefully* toward the bigram model when trigram data is thin — a desirable
property given trigram rows have ~1/10th the samples of bigram rows.

### Current lean 🟡
(3) **conditioned-second-bigram** as the target, with (2)'s framing as the architecture:
model = bigram model + a learned correction f(context A, transition BC). Rationale: single
counting at serve, reuses bigram statistical strength, isolates exactly the quantity a
trigram adds (context effects: SFS, redirects, rolls). The current recorded `time_mode`
already ALSO stores per-window data sufficient to build target 3 (press times of all three
keys are in the raw stream — but NOT in the current TSV, which stores only one duration:
**schema note** — the stroke TSV must record (t_B−t_A, t_C−t_B) separately, or just all
three press times, to enable this A/B. Fold into the OQ-5 schema change.)

## Pre-registered decision criteria

Run all three targets through the OQ-5 harness (same features, same folds, ≥3 seeds):
- Decisive: held-out **layout-level ranking** (Kendall's τ), as everywhere.
- Secondary: held-out per-transition MAE, and calibration of corpus-level sums (predicted
  vs observed total time on held-out sessions — target 1's double-count should show here).
Adopt the winner; if (3) ≈ (2) ≈ (1) on ranking, prefer (3) for the principled counting and
graceful degeneration. Prereq: the TSV schema change recording per-interval times.
