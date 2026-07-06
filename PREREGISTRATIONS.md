# Preregistrations

Decision rules recorded BEFORE seeing results. The discipline that kept this project
honest lived in session notes until now (gaps-audit 4.6); this file makes it durable.
Append-only: each entry states the experiment, the decision rule, and — filled in later —
the outcome. An analysis whose rule isn't written down here first doesn't get to call
itself confirmatory.

---

## 2026-07-04 — OQ-5 acceptance criteria (written before the harness ran)

Rule: model "generalizes" iff (1) held-out ρ ≥ 0.8× split-half ceiling on every layout,
(2) layout-ranking τ > 0 every fold, (3) beats distance+wpm baseline on ≥3/4 layouts,
(4) no catastrophic {layout×wpm} cell, (5) stable across ≥3 seeds. Failing ⇒ label the
model QWERTY-family, remediate via OQ-1/OQ-7.
**Outcome:** freq-live model FAILED (1) and (3) → caveat fired. R1W remediation later
passed (3) 12/12, (1) on 3/4 (qwerty .796–.800 borderline). Recorded in OQ-5 artifact.

## 2026-07-04 — OQ-1 decisive A/B (freq feature vs pinned)

Rule: decisive = layout-level τ; if B ≥ A on τ → drop the feature; A > B only on
per-bigram ρ → STILL drop (practice-fit is ranking-irrelevant).
**Outcome:** B won τ (+0.667 vs +0.333) and beats-baseline (4/4 vs ~1/4); A won only ρ —
the exact pre-registered drop branch. Freq deleted from schema (2026-07-05.1).

## 2026-07-04 — practice-confound arm matrix (B/R1/R2/W/R1W)

Rule: decisive = pooled held-out τ (mean over 3 seeds); tie-break 1 = mean ρ/ceiling;
tie-break 2 = beats-baseline count; winner must ≥ B on decisive.
**Outcome:** R1W won (τ +1.0 all seeds, ρ/ceiling .931); adopted, productionized,
prod-path verified.

## 2026-07-05 — dvorak red flag (3 hypotheses)

Rule: (1) zero-dvorak retrain — rank unchanged ⇒ structural; (2) no-weights scoreboard —
dvorak drops ⇒ weighting confound; (3) alternation arithmetic must reproduce the gap for
the structural reading to stand.
**Outcome:** weighting refuted (ordering unchanged), structural confirmed (arithmetic
matched: predicted 0.33% vs observed 0.29pp), zero-dvorak caveat noted (self-supported
evidence). Recorded in redflag artifact.

## 2026-07-05 — F20W / R3W (bucketed-frequency arms)

Rule: adopt over shipped R1W only if pooled τ ≥ +1.0 AND mean ρ/ceiling > 0.931.
**Outcome:** F20W 0.763, R3W 0.921 — both rejected; R1W held.

## 2026-07-05 — feature-arm matrix (A1/A2/A3/A5/C2/C3 + combos)

Rule: winner = highest mean ρ/ceiling among arms holding τ +1.0 all seeds; adopt only if
> BASE + 0.005.
**Outcome:** C2A5 won (1.0002) and was adopted — then REVERTED same day when the E5
search exposed row-blindness (Goodhart). Depth-3 kept. Lesson institutionalized as E5:
feature DELETIONS additionally require an optimizer-side structural gate.

## 2026-07-05 — local-WPM arms (S/L8/LS)

Rule: adopt local iff τ ≥ anchor AND mean ρ/ceiling > S + 0.005; anchor S must reproduce
the shipped pipeline for external validity.
**Outcome:** L8 rejected decisively (0.841 vs 0.918). LS nominally +0.014 but anchor
under-reproduced prod (extraction mismatch) and grouping confound (3.65M vs 145k
examples) → no adoption; documented.

## 2026-07-05 — OQ-14 (position practice) three tests

Rule: probe-S gap must GROW with wpm for the practice reading; arm PU adopted only if
LOLO holds τ AND E5 home-share rises materially; arm DW home-share rise ⇒ dvorak-limited
signal.
**Outcome:** gap +1/+6/+3 ms (no growth) → near-tie; PU improved LOLO (1.032) but
home-share flat (34.1% vs 33.7%) → validated-available, unadopted; DW 42.8% → home signal
lives in dvorak's data. OQ-14 resolved: comfort question.

## 2026-07-05 — lag-2 finger-reuse penalty

Rule (implicit, stated in the driver): a recovery-time mechanism must show a positive
lag-2 penalty after controlling constituent classes; ~zero ⇒ utilization is a comfort
axis, not a speed term.
**Outcome:** −13 ms weighted (−6 alt/alt, −25 shb/shb) → no recovery mechanism at lag 2;
utilization reclassified to OQ-4.

## 2026-07-05 — trigram LOLO (Phase B keystone; rule written before the run)

Experiment: `keybo validate --ngram trigram` on tristrokes_v1.tsv, seeds 0/1/2, same cell
construction (layout, trigram, session-wpm bucket), split-half ceilings per layout.
Rule: the trigram model earns objective-status iff (1) pooled held-out τ = +1.0 all
seeds, (2) mean ρ/ceiling ≥ 0.80 (trigram cells are thinner; the bigram 1.0 bar is not
expected), (3) beats a distance-sum baseline (dist(a,b)+dist(b,c)+wpm linear) on ≥ 8/12
fold-seeds. Secondary (roll question): the model's predicted run-continue vs run-redirect
contrast must agree in SIGN with the raw-data probe (roll_error_probe.py) for the
trigram objective to claim it prices rolls from data.
**Outcome: PASSED every criterion, decisively** (`runs/lolo_trigram_v1.json`):
(1) pooled τ **+1.0 all seeds**; per-fold τ +1.0 everywhere. (2) mean ρ/ceiling ≈ **1.16**
— azerty 1.32, dvorak 1.28, qwertz 1.30 (all far ABOVE their thin split-half ceilings:
cross-layout pooling shines exactly where per-layout data is thin), qwerty 0.72 (the
familiar hardest-fold pattern; ceiling .938). (3) beats the distance-sum baseline
**12/12**. Hardened metrics: calibration slope 1.04 on qwerty (no compression), worst
wpm-bucket ρ .653 vs mean .677 (no catastrophic cell). The trigram model EARNS
OBJECTIVE-STATUS. Secondary sign-check deferred to the combined-objective build (the
model's class contrasts to be read from its table during that work). NEXT: combined
bigram+trigram objective + trigram-scale search (roadmap 1.2 / 5.2).

## 2026-07-05 — roll additivity + error geometry probes

Rule: (R) run-continue median delta < alt-alt delta − 10 ms ⇒ rolls are sub-additive and
the trigram objective materially matters; run-redirect > alt-alt + 10 ms ⇒ redirects
super-additive. Between ±10 ms ⇒ bigram sums approximately suffice and the trigram
objective's value is small. (E) any row/finger error-rate spread > 1.5× between best and
worst cell, or sfb-preceded error rate > 1.5× alt-preceded ⇒ error term enters the
objective backlog with measured weights; else error stays out of scope.
**Outcome (R):** MIXED, surprising in one direction. Deltas vs sum-of-bigram-medians:
alt-alt −24 ms, run-continue −46, run-flat −21, run-redirect −46, mixed −37 (all
sub-additive — trigram spans overlap constituent windows, so the LEVEL is expected
negative; the CONTRASTS carry the signal). run-continue − alt-alt = −22 ms → ROLLS ARE
SUB-ADDITIVE per the rule → the trigram objective materially matters. run-redirect −
alt-alt = −22 ms too → REDIRECTS ARE **NOT** SUPER-ADDITIVE — a same-hand direction
reversal costs no more than its bigrams say, contradicting community doctrine that
redirects deserve extra penalty. Roll-through and redirect deltas are equal; what
distinguishes them must already live in the constituent bigrams. Consequence: a trigram
objective should reward same-hand continuation (~−22 ms per occurrence vs alternation)
and NOT add a redirect penalty beyond bigram costs.
**Outcome (E):** error rate is essentially geometry-FLAT where it matters: rows 5.4–5.9%
(spread 1.09×), fingers 4.9–6.2% except right-pinky 10.4% (qwerty right-pinky keys are
p and ;/-row edges — plausibly letter-difficulty, not geometry; and right-pinky load in
any sane layout is small), sfb-preceded 6.29% vs alt 5.15% (1.22× < the 1.5× bar).
Per the rule: ERROR STAYS OUT OF THE OBJECTIVE. The SFB-error correlation is another
reason SFBs are bad, but their time penalty already prices them.


## 2026-07-05 — skill stratification of objective-driving effects (user challenge)

Experiment: `skill_strata.py` — roll/redirect contrasts, lag-2 penalty, SFB penalty,
alternation gap, each re-measured within wpm bands 40–70 / 70–100 / 100–130 (qwerty,
matched construction to the pooled probes); plus the model-side check (do the shipped
models' wpm-conditioned table contrasts track the raw per-band physics?).
Rule: an effect is SKILL-DEPENDENT if |band3 − band1| > max(10 ms, 50% of the pooled
effect) AND monotone across bands. Consequences: skill-dependent roll/SFB effects ⇒
per-wpm objective tables become REQUIRED (machinery already supports them — one batch
predict per wpm) and layouts should be optimized at the user's target wpm; ~flat ⇒ pooled
tables stand, documented. Model-side disagreement in sign ⇒ the wpm feature isn't
capturing the interaction and needs explicit wpm×class features.
**Outcome (mixed — the user's intuition confirmed on the decisive effect):**
- **ROLL BONUS: SKILL-DEPENDENT, fires the rule.** Roll contrast −8 → −22 → −28 ms across
  40-70/70-100/100-130 (|Δ|=20 ms > 10 ms and ~90% of pooled −22; monotone). Motor
  chunking IS a fluency skill: beginners get almost no roll benefit; experts get 3.5× the
  beginner bonus. **Per-wpm objective tables are now REQUIRED for the trigram objective**,
  and "optimize at your WPM" is a real product dimension. Redirect contrast tracks the
  roll contrast at every band (−6/−23/−28) — the redirect-null is skill-robust: at NO
  skill level do redirects cost extra beyond their bigrams.
- **SFB penalty: mildly skill-dependent, fires the rule.** +27 → +32 → +38 ms (Δ=11 ms >
  10 ms, monotone) — absolute penalty GROWS with speed while everything else shrinks;
  relative to interval it roughly doubles (13%→29%). Fast typists should avoid SFBs even
  harder — direction favors per-wpm tables too.
- **Alternation gap: skill-INVARIANT.** +32/+31/+32 ms — remarkably constant; the
  alternation advantage is not practice, it is structure. (Relative share grows with
  speed since intervals shrink — consistent with the colemak-vs-qwerty flip at 120.)
- **Lag-2: null at every band** (−8/−7/+2 ms) — the reclassification to comfort is
  skill-robust.
- **Model-side: the shipped bigram model's wpm feature captures the direction** — its
  sfb-vs-alt contrast grows 29→36→40 ms across wpm 55/85/115, matching the raw physics
  (27→32→38). No explicit wpm×class features needed for bigrams; the trigram model's
  roll interaction should be verified the same way when the combined objective lands.


## 2026-07-06 — trigram target decomposition A/B (user question; OQ-10's untested fork)

Experiment: reprocess tristrokes with `--time-mode last` (target = press2→press3, the
CONDITIONED second bigram; features unchanged = all three positions) and run the trigram
LOLO; compare against the existing full-span run (`runs/lolo_trigram_v1.json`).
Why `last` should win on theory: Σ f·t(bg2|bg1) telescopes EXACTLY into corpus time (full
span double-counts, ~2x, ranking-safe but blunt), and the conditioned target isolates the
context effect (the measured roll physics) instead of smearing it with bg1's geometry
variance.
Rule: adopt `last` as the trigram target iff pooled τ stays +1.0 all seeds AND mean
ρ/ceiling exceeds the full-span run's on the SAME folds (ceilings recomputed per target —
they are target-definition-dependent — so the comparison is frac-of-own-ceiling). If
`last` wins: retrain trigram models on last-mode, re-run the per-wpm combined search with
the corrected objective (fitness then = corpus time exactly, no 2x note), update docs.
If it loses or ties: keep full-span, document that the double-count is empirically
harmless.
**Outcome: `last` REJECTED by the rule — full-span keeps.** τ +1.0 all seeds in BOTH
runs, but mean ρ/own-ceiling: full 1.157 vs last 1.043 (full wins 3/4 folds — azerty
1.32 vs 1.19, dvorak 1.28 vs 1.04, qwertz 1.30 vs 1.17; last wins only qwerty 0.78 vs
0.72). Reading: the conditioned target's THEORETICAL telescoping cleanliness loses to a
practical effect — the full span averages over two intervals, roughly halving target
noise, and on the thin folds noise dominates. The theory-vs-measurement scoreboard grows
again (cf. redirects). Full-span stays the trigram target; its ~2× double-count remains a
documented ranking-safe convention. The conditioned run remains valuable as evidence the
context effect is learnable either way (qwerty fold actually improved — worth revisiting
if qwerty-fold remediation ever becomes the binding constraint).


Local-WPM note (user asked): not re-run for trigrams. The bigram end-to-end arms rejected
local-as-replacement decisively (0.841 vs 0.918) and the mechanism is interval-level
(OQ-9: within-session autocorrelation ≈ 0 after session-speed removal) — it applies to
any interval target, trigram included. A trigram-specific arm would be confirmatory with
a strong null prior; deprioritized rather than run, noted here for honesty.

## 2026-07-06 — finger-load frontier (the utilization term's E5-style verification)

Experiment: sweep --finger-load-weight-equivalent w over {0, 20, 50, 100, 200}
(w normalized so w=100 ⇒ the load term ≈1% of qwerty's speed fitness); search each;
report speed loss vs w=0, load spread, pinky share.
Rule (sanity, not adoption — the term is a user-owned preference knob): (a) load spread
must shrink monotonically with w (else the term is mis-wired); (b) record the speed cost
of balance — if spread halves for <0.2% speed loss, balance is a near-free lunch (plateau
logic) and a nonzero DEFAULT becomes defensible to propose to the user; if it costs >1%,
balance genuinely fights speed and the default stays 0.
**Outcome (both sanity checks pass; cost sits between the pre-registered thresholds):**
| w | speed loss | max/min load | spread | pinkies |
|---|---|---|---|---|
| 0 | — | 20.9%/3.4% | 17.5% | 7.2% |
| 20 | +0.27% | 13.2%/5.5% | 7.7% | 12.4% |
| 50 | +0.37% | 13.2%/6.7% | 6.5% | 13.7% |
| 100 | +0.43% | 12.6%/7.0% | 5.6% | 14.5% |
| 200 | +0.55% | 12.7%/7.3% | 5.4% | 14.6% |
(a) PASS: spread shrinks monotonically 17.5%→5.4% — the term is wired right. (b) The big
move is the FIRST step: w=20 buys 56% of the total spread reduction for +0.27% — more
than the 0.2% "free lunch" bar but far under the 1% "fights speed" bar. Per the rule:
neither branch fires cleanly ⇒ DEFAULT STAYS 0 (strict reading), with the honest note
that w≈20 is an attractive elbow the user may want (semimak-like balance for a quarter
percent of predicted speed). One surprise worth flagging: balancing RAISES pinky load
(7.2%→12–15%) — the unconstrained optimizer had been sparing the pinkies more than the
capacity-weighted balance target does; a user who mainly wants LOW PINKY LOAD should
lower the pinky capacities rather than raise w.


Note on the watchdog's LOLO suggestion: LOLO validates predictive MODELS; the finger-load
term is an OBJECTIVE preference (the lag-2 probe measured that no speed mechanism exists
to validate it against). Its correct verification is this frontier study. Recorded here
so the discipline distinction is durable.

## 2026-07-06 — trigram-only vs combined + oxey joint-optimization (user questions)

A (trigram-only sufficiency): the trigram full-span target embeds bigram physics, so the
combined sum's implicit ~3:1 bigram-physics weight is an unprincipled (if lower-variance)
ensemble. Rule: search both objectives; if mutual cross-objective regret ≤ 0.15% (plateau
noise), SIMPLIFY to trigram-only as the canonical objective; else keep combined with the
ensemble justification documented.
B (oxey frontier): sweep community-judgment weight w ∈ {0, 0.5, 1, 2, 4} (w=1 ≈ 1% of
speed fitness); record the speed price of community-approved pattern profiles (sfb%,
dsfb%, rolls%, redirects%). Sanity: oxey score must improve monotonically with w. No
adoption rule — the weight is user-owned; the deliverable is the priced frontier.
**Outcome (A): borderline — combined KEPT, by the letter of the rule.** Regret of the
tri-only winner under combined +0.152% (a hair over the 0.15% bar); combined winner under
tri-only +0.076%. The objectives are near-interchangeable; combined stays as canonical
with the ensemble justification. NOTE: this verdict is about FULL-SPAN tables and is
superseded in spirit by the conditioned-target program (below) — re-run scheduled for the
rebuilt objective.
**Outcome (B): the community's pattern profile is CHEAP.** w=1 buys SFB 1.65%→0.99%
(−40%), DSFB 6.5%→4.7%, inrolls 6.4%→7.9% for **+0.20% predicted speed**; w=2 halves SFBs
(0.64%) for +0.42%. Monotone sanity passes through w=2; w=4 shows saturation/search noise
(oxey score plateaus −29). The deliverable stands: a speed-vs-doctrine PRICE LIST —
community-approved layouts cost a fifth of a percent, which users can decide with.

## 2026-07-06 — conditioned-target program (user challenge #4: the full-span win is an artifact)

User's argument, ACCEPTED with a sharpened mechanism: the full-span target's ρ advantage
is earned by re-predicting bigram-sum variance the bigram model already captures —
t(1→3) = t(bg1) + t(bg2) + context, and frac-of-own-ceiling measures share-of-predictable
variance, NOT novelty. The trigram model's only job is the CONTEXT increment, so the
conditioned target (press2→press3, features = all three positions) is canonical BY
DESIGN-ARGUMENT; the earlier A/B's frame ("which target is easier to predict") was the
wrong question. Model selection now happens ON the conditioned target.

Experiment (cond_target_arms.py): sample-aligned join of full-span and last-mode tables
gives per-occurrence prev = t(bg1) (the sharpest possible local-context signal — one
actual interval back, same trigram occurrence). Arms on the conditioned target, shipped
recipe, shared folds: C-BASE (anchor, must reproduce 1.043) / C-D2 / C-D4 (architecture)
/ C-PREV (+ actual-prev feature, teacher-forced at eval; serve-side story required if it
wins) / C-PREV-D2. Local-window wpm arm not run: prev IS the local signal here, strictly
sharper than any window (bigram-window null carries; reasoning recorded).
Rule: winner = highest mean ρ/own-ceiling holding τ +1.0. If C-PREV wins by >0.02,
the local-context channel is REAL for trigram targets (contra the bigram null) →
productionize prev into the schema + a serve-side story. Secondary novelty check on the
winner: its predicted run-continue-vs-alt-alt contrast must agree in sign with the
measured −22 ms. Either way, the trigram objective REBUILDS on the winning conditioned
model and the trigram-only-vs-combined A/B re-runs on the new tables.
**Outcome: architecture flat; PREV HURTS — the local-context null now holds for trigram
targets too.** Scoreboard (τ +1.0 everywhere): C-D4 1.0254 ≈ C-D2 1.0250 ≈ C-BASE 1.0223
≫ C-PREV 1.0075 > C-PREV-D2 1.0044. The actual previous interval — the sharpest local
signal constructible (same occurrence, one interval back, teacher-forced at eval) —
REDUCES transfer by ~0.015-0.018: it injects participant/session noise the geometry
features then partially fit around, and even with the answer sheet at eval it loses.
This is the strongest evidence yet for the local-context null: not windows (bigram L8),
not the actual adjacent interval (here) — the speed process really is (session pace) +
(ngram identity) + (geometry) + noise. Depth stays 3 (D4's +0.003 is inside seed noise;
ties break simpler per standing rule). The conditioned target with the shipped recipe is
the trigram model going forward; objective rebuild + tri-vs-combined re-run queued.

## 2026-07-06 — session-seeded EWMA local speed (user proposal, monkeytype-style)

What is genuinely new: local = α·prev + (1−α)·rate SEEDED AT SESSION WPM is an
INTERPOLATION between the incumbent (α→1) and pure-local (which failed three ways:
OQ-9 window probe, L8 end-to-end, PREV teacher-forced). The family contains the champion;
the question is whether any α < 1 beats it. Typo/modifier handling (user asked): the
EWMA updates only on CLEAN intervals (contiguous original indices, parseable times,
< 2000 ms) and freezes across mistypes/control keys/deletions/pauses — the contiguity
machinery already provides this.
Arms: S (anchor) / ER90, ER98 (EWMA replaces session) / ES90 (both features).
Rule: adopt iff τ ≥ anchor AND mean ρ/ceiling > S + 0.005. Prior: three-deep null stack —
expect null; the high-α arms are the ones that could evade it (gentle regularization of
session rather than noisy replacement).
**Outcome:** (pending)
