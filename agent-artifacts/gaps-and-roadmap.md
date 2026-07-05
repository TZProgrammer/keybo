# Gaps audit + roadmap to "decisively the best layout" (2026-07-05)

The user's charge: identify gaps on every axis — the finger-utilization blindness is one
instance of one axis — and lay out everything between here and a layout we can call best
*decisively*. This document is the master audit. Per-item experiments graduate into
`bigram-experiment-backlog.md` (mechanics) or their own OQ artifacts (concepts).

**The one-sentence gap:** we currently optimize *predicted bigram press-press time under a
validated-but-family-limited model* — the true goal is *a layout a real human, after
mastering it, types faster and more comfortably on than any alternative, provably*. Every
axis below is a segment of that delta.

---

## AXIS 1 — Objective gaps (what fitness doesn't see)

**1.1 Finger utilization / load balancing (user's example — anatomy).** What the bigram
objective actually prices: consecutive same-finger use (lag 1), exactly and per-finger
(same_finger × finger one-hots). The trigram schema already carries `sg_same_finger`, so a
trigram objective prices lag 2. What NOTHING prices: lag ≥ 3 reuse and overall duty-cycle
(fatigue). Semimak's "use fingers proportionally to their speed" is best understood as a
closed-form proxy for the whole lag spectrum: reuse probability scales ~ load², so
balancing loads minimizes expected same-finger collisions at all lags plus fatigue.
- Blindness quantification (MEASURED, same day): the lag-2 same-finger penalty on qwerty
  tristrokes is ≈ ZERO TO NEGATIVE (−6 ms in alt/alt contexts, −25 ms in shb/shb; weighted
  −13 ms; vs the lag-1 SFB penalty of ~+31 ms). Controlled by constituent-bigram classes,
  wpm 40–140, ≥5 samples/row. READING: the "finger needs recovery time at lag 2" mechanism
  is NOT supported — by lag 2 the finger has had an intervening keystroke to reset, and
  same-finger-at-distance-2 patterns may even be practiced motifs. CONSEQUENCES: (a) a
  trigram model would be only marginally "less blind" to utilization via reuse — the
  speed-mediated utilization argument mostly collapses to lag 1 (SFBs), which we already
  price exactly; (b) utilization balancing's real justification shifts to FATIGUE/COMFORT
  — unmeasurable in 15-minute-session speed data — i.e. it belongs in the OQ-4 comfort
  axis with user-owned weights, not in the speed model; (c) the load-term calibration plan
  below survives but its h() should be justified by comfort, not collision arithmetic.
  Caveats: single-layout (qwerty), distance-within-class not fully controlled, and the
  per-finger service-rate probe came back too thin to use (1–6 matched cells per finger —
  needs relaxed matching before calibrating anything from it).
- Lag-3: NGRAM_SPECS addition deprioritized (lag-2 already ~zero; no fat tail to chase).
- The right way (candidate): an explicit `Σ_f h(load_f)` objective term with h convex,
  load_f = corpus share on finger f. Linear in assignment variables ⇒ QAP-compatible,
  delta-evaluable in O(1) per swap. Calibrate h from the measured lag-k penalty curve +
  measured per-finger service rates (per-finger mean intervals at matched geometry) —
  data-derived, not community dogma.
- Visibility first (this round): `keybo inspect <layout>` prints per-finger corpus load,
  per-finger SFB share, lag-2 same-finger share, row/hand/motion-class shares, vs named
  layouts. Every future objective gap conversation starts from this table.

**1.2 Trigram effects — rolls, redirects. MEASURED (2026-07-05, roll_error_probe):**
rolls ARE sub-additive — a same-hand roll-through beats the alternation-context baseline
by −22 ms per trigram (run-continue −46 vs alt-alt −24, deltas vs sum-of-constituent
medians) → the trigram objective materially matters, confirming the dvorak red-flag
analysis. SURPRISE: redirects are NOT super-additive (−46, same as roll-through) — a
same-hand direction reversal costs nothing beyond its constituent bigrams, contradicting
community doctrine that redirects deserve extra penalty. Consequence for the combined
objective: reward same-hand continuation ≈22 ms/occurrence; do NOT add a redirect
penalty. Trigram LOLO harness SHIPPED (`keybo validate --ngram trigram`, commit 0874c57);
first real-data run in flight — its pre-registered rule is in PREREGISTRATIONS.md.

**1.5-outcome (error rate): measured FLAT — stays out of the objective.** Rows spread
1.09× (5.4–5.9%), sfb-preceded 1.22× (< the 1.5× bar); right-pinky's 10.4% is
letter-difficulty-confounded and low-load. Recorded in PREREGISTRATIONS.md.

**1.3 Comfort/effort (OQ-4) — now with a concrete first job.** OQ-14 proved top-vs-home is
a speed tie ⇒ a comfort term breaks it toward home at ~zero cost. Beyond that: explicit
penalties (SFB, scissor, LSB, redirect, bottom-row, lateral stretch) as a SECOND objective
axis. Design decision: scalarized weight vs Pareto front (`optimize --comfort-weight w`
sweeping w gives the front; users pick). Honesty constraint: comfort weights are
preferences, not data — separate them visibly from the measured speed term.

**1.4 Objective is per-bigram-mean time, not throughput/variance.** Σ f·E[t] ignores
variance (a layout with occasional very-slow transitions may be worse than its mean
suggests) and WPM-mixture effects. Cheap probe: optimize Σ f·quantile_0.8(t) as an arm;
if the layout barely moves (plateau logic), close it.

**1.5 Error rate is invisible.** We optimize time-between-correct-keys; a layout that
induces typos (adjacent confusions, awkward chords) pays nowhere. The dump HAS errors
(mistype rate per bigram/position is extractable). Probe: per-position/per-bigram error
rates on qwerty; if geometry-structured, an error-cost term (weight = correction time
~5.4× interval, already measured in OQ-12) belongs in fitness.

**1.6 Corpus representativeness (OQ-3).** iWeb English prose. Code, chat, other languages
shift bigram mass materially. Support user corpora (`--corpus mytext.txt` + a documented
freq-file generator) and publish per-corpus optima; check how much the optimum moves.

**1.7 Modifier/capital weight (OQ-13 leftover).** 6.1% of corpus weight skipped. Case-fold
recovery is cheap and measured-safe for the weight.

## AXIS 2 — Model gaps

**2.1 Position-practice channel (PU) parked.** Validated (ρ/ceiling 1.032) but unadopted.
Revisit when any objective change lands — it's the cleanest de-confounder we have.
**2.2 Hold/rollover features (OQ-11 leftovers).** hold column exists in the schema; per-
position hold time + per-class rollover rate as features — one arm each.
**2.3 Heteroscedasticity.** Slow bigrams have larger variance; squared-error over-weights
them. One arm: log-target (B2) or weighted loss.
**2.4 Model class.** GAM/spline arm (C4) still unrun; monotone-constraint arm was ~neutral;
seed-ensemble as first-class model (C5).
**2.5 Tune retarget (C1) still open** — the tuner optimizes CV-MAE (rewards memorization).
Retarget at LOLO ρ/ceiling; then actually sweep hyperparams for transfer.

## AXIS 3 — Data gaps

**3.1 The 4-layout family ceiling (THE hard wall).** Everything validates within
QWERTY-family row-staggered boards. "Transfers to an alien layout" is unproven and cannot
be proven from this dump. Paths, in increasing cost: (a) mine public per-user typing data
from layout-community typing tests (monkeytype/keybr publish some per-layout aggregates);
(b) a small targeted collection: even 10–20 colemak/workman typists doing 15 minutes each
gives a NOVEL holdout family (the community would plausibly volunteer — they love this
topic); (c) longitudinal self-experiment: the user learns the candidate layout with typing
logs on (n=1 but decisive for the actual end-user). This is the single highest-leverage
data move; everything else refines within-family precision.
**3.2 Session-order signal unused.** The dump has per-participant session sequence —
practice curves (within-participant improvement) could give a *causal* practice estimate
instead of our associational shrinkage term.
**3.3 Hold column carried but unused** (see 2.2).
**3.4 Other geometries out of scope by data ceiling** (OQ-6) — correct call, revisit only
if 3.1(b) ever includes ortho/split typists.

## AXIS 4 — Validation/methodology gaps

**4.1 No uncertainty on the headline metrics (E1).** τ is 6 pairwise comparisons; ρ gaps
of 0.01 get treated as real. Participant-level bootstrap CIs on τ, ρ, AND on scoreboard
percentages (the +2.49% needs a ±). Without this, arm adoption at <0.01 margins is vibes.
**4.2 Calibration slope (E4) designed, not implemented.** Rank metrics are compression-
blind; the optimizer consumes magnitudes.
**4.3 Worst-cell matrix (OQ-8/E2)** designed, not implemented.
**4.4 Plateau ambiguity → rank-stability report.** Optima are ~0.5%-wide; single "best
layout" strings are unstable across seeds/restarts. Ship "the top-K set + their pairwise
CI overlaps" rather than one string; report position-level consensus (which letters are
stably placed?) — turns the plateau from an embarrassment into a deliverable.
**4.5 E5 gate is manual.** Make `inspect`-based structural checks an automatic postflight
of every `optimize` run (home share, finger loads, SFB count vs named layouts).
**4.6 Preregistration hygiene is informal** (memory.md). Move decision rules into the repo
(a PREREGISTRATIONS.md log) so the discipline survives the session.

## AXIS 5 — Search gaps

**5.1 No optimality certificate.** SA+2opt on a plateau. The QAP table admits Gilmore-
Lawler / spectral lower bounds → "within X% of optimal" statements; small neighborhoods
admit exact branch-and-bound. This is what "decisively" needs on the search side.
**5.2 Trigram-objective search scaling.** Per-bigram tables (840×31×31) worked for PU;
trigram terms need either sparse trigram tables (top-K trigrams cover most mass) or
delta-evaluation tricks. Design before the trigram objective lands.
**5.3 Multi-start diversity is ad hoc.** Use the position-consensus report (4.4) to seed
diverse restarts deliberately.

## AXIS 6 — Correctness/engineering gaps

**6.1 Experiment drivers duplicate the training recipe** (the freq= crash proved the
hazard). Extract a `keybo.research` module (matrix building, fold loops, table search) so
drivers compose library calls instead of copying them.
**6.2 `keybo optimize` CLI still uses the slow scorer** — table path not wired in (known).
**6.3 Threshold mismatch (D1)**: harness loads wpm_threshold=0/min_samples=1; train CLI
defaults 60/25. Reconcile.
**6.4 tune CLI broken invariants**: uses CV-MAE + full-width features; predates every
adoption. Fix with 2.5.
**6.5 No CI.** Tests run ad hoc; a GitHub Action (pytest + ruff on push) is an hour of
work and guards the laptop↔dev-box flow.
**6.6 e2e workspace (keybo-e2e/) is untracked scripts + artifacts.** The drivers now live
in agent-artifacts/experiments/, but runs/*.json provenance (which commit, which data
hash) is by-convention only. Add a tiny run-manifest writer.

## AXIS 7 — Deliverable gaps ("decisively" is a claim to an audience)

**7.1 No reproducible claim bundle.** "Best" needs: pinned data hash + commit + one
command (`just reproduce-best`) that regenerates the layout and its evidence table.
**7.2 No comparison under COMMUNITY metrics.** To convince layout people, also score our
layout under their analyzers (oxeylyzer/genkey-style metrics); where we disagree, explain
from data. Outside-the-box: their metrics are free heuristic judges — cheap sanity axes.
**7.3 No human validation protocol.** The end of the road (D below) needs a written
protocol before anyone types a key, or it proves nothing.

---

# The plan to "decisively the best" (phased, each phase harness-judged)

**Phase A — see clearly (days).** `keybo inspect` (this round) · lag-2/3 blindness
quantification (running) · bootstrap CIs (4.1) + calibration slope (4.2) + worst-cell
(4.3) · rank-stability/top-K report (4.4) · auto-postflight E5 (4.5) · PREREGISTRATIONS.md.

**Phase B — complete the SPEED objective (1–2 weeks).** Trigram LOLO harness → measure
roll sub-additivity / redirect super-additivity → combined bigram+trigram objective →
finger-load term calibrated from lag-curve + service rates (1.1) → error-rate probe (1.5)
→ search scaling for the combined objective (5.2) + GL bound (5.1). Exit criterion: the
objective can *see* everything semimak/graphite were designed around, priced from data.

**Phase C — comfort axis + user dimension (parallel to B).** OQ-4 comfort term with
explicit user-owned weights (first job: break the OQ-14 tie toward home) · Pareto
front deliverable · per-corpus optima (1.6) · case-folding (1.7).

**Phase D — break the family ceiling (the decisive step).** 3.1: community data collection
(a hosted 15-minute typing test for alt-layout users is the realistic path — the analysis
pipeline is DONE, it's a data-acquisition problem) → LOLO with a truly novel holdout family
→ if the model transfers there, "best" stops being extrapolation. In parallel: n=1
longitudinal self-test protocol (7.3) — the user learns the layout, logs, and the practice
curve is measured against prediction.

**Phase E — the decisive claim (assemble).** Reproducible bundle (7.1) + optimality
certificate (5.1) + community-metric crosswalk (7.2) + human-validation results (D) +
uncertainty on every number (4.1). "Decisively best" = *best predicted speed with
certificate, on a model validated beyond its training family, Pareto-presented with
comfort, reproducible by anyone, and consistent with (or better than, with explained
disagreements) the community's own judges — with at least one human data point confirming
the prediction direction.*

Sequencing note: A is prerequisite tooling (cheap, do now); B and C run in parallel; D is
the long pole and should START (community outreach design) while B runs; E assembles.
