# OQ-9 — Should a bigram's predictor condition on LOCAL typing speed (surrounding n keys)?

**Status: 🟢 CLOSED (2026-07-04) — measured on the real dump: NO. Session WPM already
captures the entire exploitable speed component; local windows add ≤0.011 R² even combined.**

## Measured answer (2000 qualifying files; 26.5k sessions; 1.02M bigram PP intervals)

- **Variance decomposition of log(PP interval):** session identity R²=0.208, bigram identity
  R²=0.181, additive both=0.382 — leaving 61.8% residual. A local-speed signal could only
  live in that residual…
- **…and it doesn't:** within-session autocorrelation of PP intervals is ≈ ZERO once session
  speed is removed (lag-1 log-centered r=0.004; with BOTH session speed and bigram identity
  removed, r=−0.043 — no positive momentum at all). The only autocorrelation in the raw
  series (~0.13, flat across lags) is pure between-session speed — exactly what session WPM
  already encodes.
- **Predictor experiment:** trailing / EWMA (halflife 3, 8) / surrounding windows over
  n∈{2,3,5,8,12,20} add ≤0.011 R² over the session mean (all windows combined, fair common
  subset), ~0.001–0.003 on the full set. De-biased ≈ biased (the shared-key contamination
  concern turned out immaterial); the non-causal surrounding window is NOT better than
  trailing — there is simply no local signal to harvest.

**Decision (per the pre-registered criteria below): keep session WPM; do not add local-pace
features or normalization.** Criterion 1 required ≥20% relative MAE gain; measured gain is
~1-3% R² at best. The typing-speed process is (session-level pace) + (bigram identity) +
(unpredictable noise) — "momentum" between keystrokes is a myth at this data's resolution.

Report: `state/keybo-muscle-d/artifacts/report.md` (probe_1_autocorr.py, probe_3_variance.py).
Original analysis kept below for the reasoning record.

## The question

Today each training row's context is the **session WPM** — one number for a whole test
sentence. The proposal: derive the "how fast was this person going *right here*" signal from
the surrounding keystrokes instead — the trailing n intervals, or the surrounding n — and
sub-questions: what n? weight closer keys more? trailing (causal) or centered (uses the
future)?

## Why this matters more than it looks

The model learns `time = f(geometry, context)`. Everything the context variable absorbs is
variance the geometry features DON'T have to explain. Session WPM absorbs between-session
skill/fatigue; a local window would additionally absorb *within*-session drift (warmup,
mid-sentence hesitations, bursts). Cleaner residual variance → sharper geometry estimates →
better layout rankings. It is the same statistical move as OQ-A4's participant effects, one
level down the hierarchy: participant → session → local window.

## Analysis

### Trailing vs centered vs session-level
- **Trailing (previous n)** is causal and matches the mental model "current pace". Safe.
- **Centered (± n/2)** leaks the *future* — including, for the bigram being predicted, keys
  typed after it. For TRAINING context that's not fatal (we're estimating a latent pace, not
  forecasting), but there's a subtle trap: the window must EXCLUDE the target interval
  itself, else the "context" contains the label (target leakage: R² inflates spuriously and
  the geometry features get starved of signal they legitimately own).
- **Adjacent-interval contamination**: the interval immediately before bigram (A,B) ends at
  key A — it shares a keystroke with the target. A hesitation on A contaminates both. The
  window should exclude the immediately adjacent interval(s); muscle-D measures both ways.

### What n / what weighting?
This is empirical (autocorrelation structure decides): if PP intervals decorrelate by lag
~5, an n=20 window adds noise, not signal; if correlation decays smoothly, exponential
weighting (recency) beats a hard cutoff. Prior expectation 🟠: typing pace has both a slow
component (fatigue/skill: session-level) and a fast component (bursts/hesitations: lag 1-3),
so an EW-mean with a short halflife PLUS the session mean (two features, not one) likely
dominates any single window. The predictive experiment (muscle-D: session-mean vs trailing-n
vs EW vs centered, R²/MAE) answers directly.

### The serve-time problem — the real design constraint
**At scoring time there is no keystroke stream.** The optimizer asks "how fast is bigram X
on candidate layout Y for a target typist" — there's no "previous 8 keys" to condition on.
So a local-speed feature can enter the model ONLY in ways that remain well-defined at serve:
1. **As a training-time cleanup, not a serve feature** (recommended path): use the local
   window to NORMALIZE the target (e.g. target = interval / local-pace, or regress out
   local pace), train geometry features on the cleaned target, and at serve multiply back a
   target pace. The model never sees a window feature; train/serve parity is preserved.
2. As a feature frozen to a constant at serve (like wpm today) — legal but adds another
   train-varying/serve-constant column (the same smell the audits flagged for freq/wpm).
3. NOT as a raw feature the optimizer varies — meaningless at serve.

### Interaction with typos (OQ-12)
A window spanning a correction inherits the typo's slowdown. Options: exclude windows
containing corrections (shrinks data), cap window intervals, or use median instead of mean
within the window (robust, cheap — likely the right default 🟡). Muscle-D's contamination
radius tells us how far after a typo the window must be considered dirty.

## Pre-registered decision criteria (so this closes, not drifts)

Adopt local-pace normalization iff, on the real data:
1. trailing-window predictor beats session-mean by ≥ 20% relative MAE on held-out intervals
   (else the added machinery isn't paying), AND
2. training on pace-normalized targets improves held-out **per-bigram Spearman ρ within
   layout** (the OQ-5 harness metric) vs session-wpm-only, AND
3. the chosen window (n, weighting) is stable across WPM buckets (else per-bucket windows —
   complexity we'd rather avoid).
Fill in: muscle-D table → verified numbers → decision. Until then the pipeline keeps
session WPM.
