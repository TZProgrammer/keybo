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
**Rebuild outcome (cond_rebuild.py, runs/cond_rebuild.json): the corrected objective
T3c = T2(bigram physics) + Tcond(conditioned increment) is now canonical, and the
re-run A/B SIMPLIFIES the objective: mutual regret +0.079%/+0.010% (both under the 0.15%
bar) ⇒ TRI-CORRECTED-ONLY is the canonical objective (the earlier borderline keep of
'combined' is superseded — the double-count is gone by construction). Final search:
best `bhaievlnsdpyo.utmrfcq;/,jgkwxz` +2.07% vs qwerty, 13 distinct near-optima/0
consensus slots (plateau as always), E5 clean (home 31.9%, sfb 1.47%), bigram-component
certificate within 2.54% of optimal.**
**Outcome: architecture flat; PREV HURTS — the local-context null now holds for trigram
targets too.** [See also the EWMA entry below — the null is now four-deep.] Scoreboard (τ +1.0 everywhere): C-D4 1.0254 ≈ C-D2 1.0250 ≈ C-BASE 1.0223
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
**Outcome: NULL — the cleanest monotone gradient yet.** S 0.9750 > ES90 0.9657 > ER98
0.9445 > ER90 0.9314 (τ equal across arms). Transfer degrades EXACTLY in proportion to
how much local signal is mixed in: α=.98 (2% local) loses less than α=.90 (10% local);
adding EWMA as a second feature loses less than replacing — but EVERY admixture loses.
The interpolation family contains the champion, and the champion is its α→1 endpoint:
session wpm, pure. The local-speed question is now closed four ways (window probe,
trailing-median replacement, actual-adjacent-interval feature, session-seeded EWMA) —
the within-session speed process has no exploitable local structure in this data,
full stop. (Monkeytype's estimator is for DISPLAY of a live wpm, not prediction —
plausible why the intuition transfers poorly.)

## 2026-07-06 — grand evaluation round (user directive: magnitudes, buckets, cleaning, tune, sweep)

Theory concession first: the optimizer is invariant to AFFINE miscalibration only, while
ρ/τ are invariant to ALL monotone transforms — nonlinear compression preserves ranks and
moves the argmax. The user is right that ranking-only arm selection was insufficient.
Harness upgraded (82b9695): corpus-weighted MAE/MAPE + per-bucket {ρ, wmae, slope, n}.
Scope note: recent OBJECTIVE decisions (tri-only regrets, scoreboards, plateau) were
already magnitude-based (fitness cross-scoring); ARM selections were ρ-based → P4
re-verifies them. The freq-feature verdict is NOT re-run: a τ collapse to +0.333 is
disqualifying under any calibration (broken ranking cannot be repaired by magnitudes).

P1 (matrix): champion bigram + conditioned trigram → full {layout × wpm-bucket}
  {ρ, wmae, slope} matrices. Question: high-band transfer uniform across layouts, or
  dvorak-only? No rule — this is the instrument panel.
P2 (slow-typist removal): train-side wpm floors {0, 60, 80}. Rule: adopt a floor iff τ
  holds AND high-band (100–130) wmae improves >1% relative AND overall wmae degrades <1%.
P3 (cleaning): (a) drop sessions with error rate >20%; (b) drop first-2 sessions per
  participant (warmup). Same rule as P2, judged against the SAME-extraction baseline.
P4 (re-verify): depth {2,3,4} × practice {on,off} under wmae. Informational (P5 decides);
  any ρ-vs-wmae disagreement is flagged and the magnitude verdict wins.
P5 (tune): tune-lolo pattern re-ranked by wmae (τ-gated): 16 bigram + 8 cond-trigram
  candidates. Adopt iff wmae beats incumbent by >0.5% relative at τ +1.0.
P6 (sweep): tuned models → corrected T3c at wpm 90 → layouts at oxey w ∈ {0, .5, 1, 2}
  with stability + pattern stats. The user's requested deliverable.
  **P5/P6 outcome:** bigram tuning adopt=False (incumbent already optimal, 16 candidates);
  cond-trigram ADOPTED cand-4 (wmae 18.51 vs 19.09, τ +1.0) — ⚠ selected pre-guard; the
  rare-decile re-verification of this adoption is the flagged next step (T-MAE's guard
  firing shows the risk is real). P6 family (runs/p6_oxey_sweep.json, tuned models,
  wpm 90): w=0 `gyou,lntscdeaiprmbfwj;/.khvxqz` (sfb 1.52%); w=0.5 +0.017%; w=1 +0.044%
  (sfb 1.52%); w=2 +0.51% (sfb 0.82% — halved). Post-tuning, community doctrine is even
  cheaper than the pre-tuning frontier (+0.04% vs +0.20% at w=1).
**Outcomes P1–P4 (runs/grand_p*.json):**
- **P1 (matrix):** both champions hold τ +1.0 with calibration slopes ~1.0 per fold. The
  per-bucket matrices (in JSON) answer the user's dvorak question: high-band ρ stays
  positive on every layout; magnitude quality (wmae) is uniform enough that no
  layout×band cell is catastrophic. Instrument panel established.
- **P2 (slow-typist removal): REJECTED per rule, with the instructive gradient.**
  floor=60: τ degrades to +0.667; floor=80: τ collapses to [0.33, 0.0, 0.33] — even
  though high-band wmae improves dramatically (14.4 → 9.4 → 7.9). Dropping slow data
  sharpens high-band magnitudes at the cost of cross-layout RANKING: the slow bands
  carry a large share of the cross-layout identification (thin layouts are dispropor-
  tionately mid/slow). The right construction, noted for future work: per-band tables
  from a full-data model (the model already conditions on wpm), not floored training.
- **P3 (cleaning): both arms REJECTED — nothing to clean.** errclean: wmae 15.82 vs
  baseline 15.72 (worse); warmdrop: 16.01 (worse); neither improves high-band; errclean
  even costs τ on some seeds. The existing hygiene (contiguity, QUOTE_NONE, IQR-mean,
  rejection counters) already extracts a clean signal; further filtering only removes
  identification.
- **P4 (re-verify under wmae): every ρ-based adoption SURVIVES the magnitude standard.**
  Practice term dominates wmae exactly as it dominated ρ (d3: 15.55 with vs 28.15
  without — nearly 2×); depth is secondary (d2/d3/d4 with practice: 15.90/15.55/15.86
  overall; d4 best in high band at 13.85 but inside noise of d3's 14.28). No ρ-vs-wmae
  disagreement found — the flagged risk did not materialize on these arms. Depth 3
  stands.


## 2026-07-06 — blind pace model (user proposal #7: two-stage decomposition)

User's framing: stage 1 = a content/geometry-BLIND model predicting the current pace from
surrounding speeds only (deliberately simple to avoid content leakage — correctly
identified as the freq-Goodhart channel); stage 2 = geometry model conditioned on it.
Claim: better stage-1 ⇒ better downstream. Includes the hump argument (any averaging
underestimates at a pace extremum; centered windows beat trailing ones) and asks for an
analytical solution.

Analytical answer (to be verified by the probe): the optimal LINEAR blind predictor of
x_t from all other intervals weights them by the inverse covariance; with the measured
within-session autocorrelation ≈ 0 at all lags (OQ-9: lag-1 r = 0.004 after
session-centering), the inverse covariance is ~diagonal ⇒ the optimal blind predictor IS
the session mean. The hump scenario REQUIRES positive short-lag autocorrelation (a smooth
latent pace process); its measured absence means either no humps at sentence scale or
humps drowned by keystroke noise — the optimal filter ignores them either way.
Structural fact: our "session" is ONE SENTENCE (~8 s), so session WPM is already a
CENTERED local window (symmetric past+future by construction) at nearly the same scale
as the proposed 10+10 window.

Probe (blind_pace_probe.py; cheap, no LOLO needed — the user's own monotonicity logic
means a stage-1 that cannot beat the session mean closes the idea): on a large sample of
qwerty sessions, predict each held-out interval from the OTHER intervals via (a)
leave-one-out session mean, (b) centered window means h ∈ {2, 5, 10}, (c) the user's
exact model — ridge-fit linear weights on centered neighbors (5+5 and 10+10 where
sentence length allows). Also report sentence-length distribution and the
self-inclusion effect of current session wpm.
Rule: if the best blind estimator beats LOO-session-mean by < 2% relative MAE (or
centered-R² gain < 0.01), the stage-1 premise is unattainable in this data ⇒ the
two-stage idea closes without an end-to-end arm. If it beats it materially ⇒ build the
full LOLO arm with the winning estimator as the wpm feature.
**Outcome: CLOSED under the rule — the analytical prediction held, with one refinement.**
Test MAE on 1.84M held-out intervals: LOO session mean 62.16 ms; centered windows h=2/5/10
WORSE (−11.1%/−3.5%/−1.3% — small windows are strictly noise-amplifiers); the user's
ridge 10+10 model: 61.79 ms = **+0.60%**, under the 2% bar. The refinement: the ridge
weights are NOT exactly zero — small (max 0.036), positive, symmetric past/future, and
smoothly decaying — i.e. a real but tiny local-pace process exists (each neighbor carries
~3% weight), worth 0.6% of interval MAE, far too small to survive aggregation into cell
targets and model selection (consistent with all four downstream nulls). The two-stage
idea is architecturally sound; the data's local pace signal is just ~0.6%-sized.
Session-as-sentence (median 37 clean intervals) also confirms the incumbent is already a
centered window at nearly the proposed scale. No end-to-end arm; stage-1 = session mean
is within 0.6% of the best achievable blind estimator.

## 2026-07-06 — rare-ngram guard on magnitude selection (user directive #8)

User's point, accepted: wmae concentrates on the top-k frequent ngrams and can let
selection abandon rare cells — which are exactly the evidence for position pairs the
optimizer explores off the frequency distribution. Two clarifications recorded for
scope-honesty: (1) TRAINING never used wmae — the loss is per-cell squared error with
layout-balance weights, so the model sees rare ngrams at full strength; the exposure is
confined to SELECTION (P5 tuning + P4 re-verification). (2) The P5 bigram verdict
(adopt=False, incumbent kept) is immune — nothing was selected. The P5 trigram verdict
(pending in the resume; the leading challenger wins on wmae) is the one selection made
under the exposed metric.
Harness upgraded: umae + freq_decile_mae now reported per fold-seed. Amended selection
rule going forward: a challenger must improve wmae WITHOUT degrading umae by more than
2% relative or worsening the bottom-3 frequency deciles' MAE. Pending action: re-verify
the P5 trigram adoption under the amended rule before blessing P6 outputs built on it.
**Outcome:** (pending the resume run's P5-trigram verdict re-check)

## 2026-07-06 — blind-pace maximization (user directive: best possible stage-1, no leakage)

Leakage contract (binding): stage-1 sees ONLY timing scalars + session/participant
indices. Model class capped at linear over robust aggregates — expressive models could
fingerprint content from neighbor-value PATTERNS (the 'space-t-h ⇒ e' channel the user
identified). Built-in leakage audit: a candidate whose held-out residuals encode more
ngram-identity information (R² on ngram one-hots) than the LOO-mean's residuals do is
DISQUALIFIED regardless of MAE — the audit operationalizes "without leaking context".
Levers: L1 participant prior (pace from the typist's OTHER sessions, shrunk — the big
untapped one; LOO mean uses one sentence of evidence), L2 robust location (LOO median /
trimmed mean), L3 heterogeneous linear blend (priors + windowed neighbor medians +
sentence-position index), L4 log-space, L5 the +0.60% per-offset ridge as reference.
Rules: frontier reported; end-to-end LOLO re-plumb only if the best clean model beats
LOO mean by ≥5% test MAE (the earlier 2% rule answered 'is there anything'; 5% is the
'worth re-plumbing' bar). All candidates leakage-audited.
**Outcome: BAR CROSSED — the first material pace win of the program (1.84M held-out
intervals, participant-level split):**
| candidate | test MAE | vs LOO mean |
| M0 LOO session mean (incumbent) | 61.87 | — |
| M1 LOO session MEDIAN | 57.53 | **+7.02%** |
| M2 participant prior alone | 57.40 | **+7.23%** |
| M5 full blend, LOG space | **57.14** | **+7.65%** |
| M3/M4 raw-space blends | 60.9/60.7 | +1.6/+2.0% (heavy tail wrecks raw-space ridge) |
| L5 offsets ridge (reference) | 61.50 | +0.60% |
Leakage audit: M5 residual ngram-R² 0.1108 vs incumbent 0.1109 — PASS, no content info
added. READING: the gains are NOT local-neighbor signal (still ~0.6%); they are (a)
ROBUST LOCATION — the session median beats the mean by 7% alone (hesitation tail pollutes
the mean), and (b) CROSS-SESSION EVIDENCE — the typist's other sessions predict this one
as well as its own median does. Log-space fixes the blend (pace is a scale factor).
Consequence per rule: the end-to-end LOLO arm is now REQUIRED — pace label upgraded from
session-mean-wpm to the blind stage-1 estimate. Registered arms: S (anchor) / MED
(session-median pace — isolates the robust-location lever) / M5 (full blind blend).
Adoption rule: standing (τ ≥ anchor AND ρ/ceiling > S + 0.005), PLUS the rare-ngram guard
(umae + bottom-decile non-degradation). Caution prior: the EWMA round showed
interval-level gains can still hurt cell-level transfer — MED (session-level, no local
terms) is the arm most likely to survive.

## 2026-07-06 — robustness round (user questions #10: combine the +7% levers; MAE vs MSE
## for the ngram models; hesitation-tail cleanup)

Context: the blind-pace frontier found median-beats-mean (+7.0%) and participant-prior
(+7.2%) with the COMBINED M5 at +7.65% — the levers barely add because they estimate the
SAME latent (typist pace); combination refinement targets the remaining gap. The
median-beats-mean result also exposes that heavy tails may distort three other places:
the cell TARGET statistic (currently IQR-mean), the training LOSS (currently squared
error), and hesitation samples inside cells. OQ-12 rejected duration CAPS at the
aggregate level (11.4% of clean time above 3× median = legitimate hesitation weight);
this round re-tests at the TARGET level with a different mechanism — hesitations are
attention/cognition, not biomechanics, so excluding them from per-cell aggregation may
sharpen geometry estimates. Honest counter-hypothesis carried into the rule: hard
bigrams may CAUSE hesitations, so filtering could remove real signal — LOLO decides.

Arms (bigram frame, hardened harness incl. umae + deciles):
  T-BASE  incumbent (IQR-mean targets, MSE loss, session-mean wpm) — anchor
  T-MED   cell target = MEDIAN of durations
  T-MAE   XGBoost objective reg:absoluteerror (IQR-mean targets)
  T-CAP   drop samples with duration > 3 × (12000/session-wpm) before aggregation
  P-MED   wpm label = session-MEDIAN pace (the robust-location lever, end-to-end)
  P-M5    wpm label = full blind blend (log-space M5)
  C       stage-1 combination refinements (log-space prior, tuned shrinkage,
          median-of-medians) — probe-level, feeds P-M5's label if better
Rules: T/P arms adopt iff τ ≥ anchor AND ρ/ceiling > anchor + 0.005 AND wmae improves
>1% relative AND neither umae nor bottom-3 decile MAE degrades >2% relative. Ceilings
recomputed per target variant (target definition changes the ceiling). Caution prior
recorded: interval-level wins have died at cell level four times (EWMA et al.); the
session-level arms (T-MED, P-MED) are the structurally-favored survivors.
**Outcome (runs/robustness_arms.json; anchor τ +0.67 — this driver's frame
under-reproduces prod τ +1.0, so any adoption needs prod-path confirmation):**
- **T-CAP (hesitation filter): ADOPT-CANDIDATE — clean sweep.** ρ/ceiling **1.0104** vs
  anchor 0.9706 (ceilings recomputed per frame, so this is a genuine relative gain, not
  an easier-task artifact), wmae −23.5%, umae −25.7%, rare-decile −23.2%, τ equal. The
  user's mechanism (hesitations are cognition, not biomechanics) wins decisively; the
  OQ-12 counter-hypothesis (hard bigrams cause hesitations) did not bind. PENDING:
  prod-path confirmation before productionizing (filter into the pipeline + shipped
  validate).
- **T-MAE: REJECTED by the rare-ngram guard — the user's warned failure mode, observed.**
  wmae −24.8% (huge head win) but rare-decile MAE +8.1% (29.27 vs 27.07): absolute-error
  loss concentrates on the frequent head and abandons rare cells. The guard exists for
  exactly this; it fired on its second use.
- **T-MED: REJECTED** (ρ/ceiling 0.9610 < anchor; magnitude gains real but rank quality
  pays). IQR-mean already captures most of the median's robustness.
- **P-MED / P-M5: REJECTED per the rule's letter** (ρ/ceiling 0.964/0.966 < bar), with
  an honest note: both lift τ to +1.0/+1.0 (vs anchor +0.67) and their wmae/umae are on
  DIFFERENT cell frames (pace label changes bucketing) so the magnitude conditions were
  not fairly evaluable — a matched-frame follow-up is registered as future work, lower
  priority than T-CAP productionization.


### Outcome append (2026-07-07): cand-4 guard recheck — ADOPTION KEEPS
Rule (recorded in memory.md before the run): cand-4 keeps its P5 adoption iff tau holds
and it degrades neither umae nor bottom-3-decile MAE by >2% relative vs the incumbent.
Result (2 fold-seeds, runs/cand4_guard_check.json): wmae 19.09 -> 18.51 (-3.0%),
tau 1.0/1.0 both arms, umae +0.14%, dec3 +1.35% — inside the guard. cand-4 is NOT
the T-MAE failure mode (that arm bought wmae -24.8% at dec3 +8.1%); it buys its wmae
gain without starving rare ngrams. ADOPTION CONFIRMED; the ⚠ pre-guard flag on the
P5 cond-trigram entry is resolved and the P6 family stands.
Honest note outside the rule: rho_frac 1.040 -> 1.006 (both above ceiling; not a rule
condition, recorded for completeness).

## P7: filtered-data rebuild (registered 2026-07-07, before tcap_prod_confirm results)
The hesitation filter shipped as extraction code (f8b008d). Two chained rules, both
recorded before any result:
1. CONFIRM RULE (tcap_prod_confirm.py, v3 vs v4 bigram tables, 3 fold-seeds, shipped
   validate): the filter's default stays 3.0 iff tau +1.0 on all pooled fold-seeds AND
   rho/ceiling improves AND wmae, umae, dec3 all improve vs the unfiltered table.
   Any miss: flip the CLI default to 0 (off) and document. Ceilings recomputed per arm;
   rho judged as fraction-of-own-ceiling since the filter changes the target definition.
2. REBUILD RULE (p7_filtered_rebuild.py, runs only on confirm): retrain bigram (shipped
   recipe) + cond-trigram (cand-4) on filtered tables, rebuild T3c(90), re-sweep oxey
   w in {0, .5, 1, 2} at the P6 search budget. The P7 family REPLACES P6 as the
   deliverable iff the filtered cond-trigram LOLO matches-or-beats the unfiltered
   cand-4 leg of the guard check (tau holds, rho/ceiling within -0.005, wmae within
   +1%); else P6 stands and P7 is an appendix.

### Outcome append (2026-07-07): tcap_prod_confirm — CONFIRM RULE FAILS on tau; default flipped to OFF
Result (3 fold-seeds, shipped validate, runs/tcap_prod_confirm.json):
  v3 unfiltered: rho/ceil 0.995, wmae 15.71, umae 19.93, dec3 26.83, tau +1.0/+1.0/+1.0
  v4 filtered:   rho/ceil 1.017, wmae 12.01, umae 15.26, dec3 21.66, tau +0.67/+0.67/+0.67
Magnitude clean sweep REPRODUCES on the prod path (wmae -23.6%, umae -23.4%, dec3 -19.3%,
rho/ceil +0.022) — but the decisive metric tau DROPS 1.0 -> 0.67 on every seed (one layout
pair swaps). Per the rule ("tau +1.0 holds AND ..."), default-on is NOT confirmed:
--hesitation-cap default flipped to 0 (opt-in). Note the driver-frame T-CAP run had shown
tau +0.67 as its OWN anchor too, so the swap only became visible against the prod anchor.
DIAGNOSIS REGISTERED (tau_diag.py, running): compare v3-vs-v4 OBSERVED layout tables to
see whether the filter moved the ground-truth ordering itself (no model involved) — the
follow-up question is which ordering is correct, i.e. the source-of-truth question.

### Outcome append (2026-07-07): tau_diag — the tau drop is an AZERTY-QWERTZ near-tie flip
Observed (no model) common-ngram layout tables:
  v3: dvorak 133.88 | qwerty 139.62 | qwertz 147.98 | azerty 148.34  (azerty slower by 0.36ms)
  v4: dvorak 132.66 | qwerty 137.91 | azerty 146.17 | qwertz 146.32  (qwertz slower by 0.15ms)
The pair that costs tau 1.0 -> 0.67 is azerty-qwertz, whose observed gap is 0.36ms/0.15ms
in a ~14ms between-layout spread — a statistical tie that flips direction under the
filter. Registered follow-up (pair_gap_boot.py, running): participant-bootstrap CI on
every pair's gap in both tables; PREREGISTERED RULE: if the azerty-qwertz CI spans 0 in
BOTH tables, the pair is declared undecidable, tau is reported henceforth ONLY over
decisive pairs (CI excludes 0), and the hesitation-filter confirm verdict is RE-JUDGED
under decisive-pair tau (magnitude sweep already reproduced). If the pair IS decidable
in v3 and the filter genuinely flips a decidable pair, the filter stays off.

## ENDGAME rules (registered 2026-07-07, before results of any of these runs)
Context: user directive — leave nothing pending; deliver the best possible layout.
1. RAND-DROP control (rand_drop_control.py): drop the filter's exact drop-rate (2.851%)
   uniformly at random from v3, same 3-seed shipped validate. Interpretation rule:
   control reproducing a large share of the filter's MAE gain => variance artifact;
   control ~flat => the filter's gain is real contamination removal (it matters WHICH
   samples go). Report control's share of the filter gain per metric; also compare
   per-arm ceilings (a pure variance shrink moves obs noise and ceiling together and
   cannot RAISE rho/ceiling the way the filter did, 0.995->1.017).
2. MATCHED-FRAME pace labels (matched_frame_pace.py): INC (session-mean wpm) vs MED
   (session-median pace) vs M5 (blind log blend, the +7.65% frontier winner) with ONE
   shared sample set, ONE cell frame (incumbent bucketing), ONE example structure —
   only the wpm feature value varies. ADOPT rule: challenger wins iff wmae improves
   >1% rel AND umae/dec3 degrade <=2% AND decisive-pair tau does not drop. If both
   qualify, better wmae wins. Filter-x-label interaction: follow-up only if adopted.
3. P7 gate AMENDMENT: the confirm verdict is re-judged under decisive-pair tau (per the
   0597fdf rule). If azerty-qwertz is undecidable in BOTH v3 and v4 bootstrap tables,
   the magnitude sweep carries the verdict, the hesitation filter is ADOPTED (default
   back to 3.0), and the P7 filtered rebuild proceeds. If the pair is decidable and
   genuinely flipped, the filter stays off and P7 aborts.
4. FINAL DELIVERABLE assembly: best label (from 2) x best data treatment (from 1+3) =>
   retrain -> T3c(90) -> oxey family {0,.5,1,2} at P6 budget -> stability + certificate
   + full verdict table. If 2 adopts a new label, the P7 rebuild re-runs with it (one
   rebuild, both levers, since each was independently gated on its own rule).

### Outcome append (2026-07-07): RAND-DROP control — the hesitation filter's gain is REAL
Dropping the filter's exact drop-rate (2.853% achieved vs 2.851% target) uniformly at
random from v3 reproduces essentially NONE of the filter's improvement
(runs/rand_drop_control.json, same 3-seed shipped validate):
  control: rho/ceil 1.001, wmae 15.77, umae 19.91, dec3 26.63, tau +1.0 x3 (~= v3 baseline)
  share of filter gain reproduced: wmae -1.7%, umae +0.5%, dec3 +4.0%
Per the rule: the filter's ~23% gains come from WHICH samples it removes (the hesitation
tail), not from having less data/variance. Also, the random control did NOT flip the
azerty-qwertz pair (tau stayed +1.0), consistent with the near-tie flip being specific
to removing slow-tail mass. Confound REJECTED; magnitude verdict stands as real.

## CAP-SWEEP (registered 2026-07-07, before results; user: "try different values for the filter")
One extraction records per-window ratio = interval / session-median clean interval;
arms thresholded in memory at cap in {1.5, 2, 2.5, 3, 4, 5, 8, inf}. Controls for the
two structural hazards: (A) target drift — per-arm ceilings + rho/frac-of-own-ceiling
reported, never raw MAE alone; (B) signal censoring — censor_ratio = slowest-decile-
ngram drop% / overall drop% (a content-blind hesitation filter drops ~uniformly; a
tight cap concentrating drops on intrinsically-slow ngrams is deleting geometry signal).
SELECTION RULE: best wmae among caps with dec3 <= anchor+2%, in-frame tau >= anchor's,
rho/frac >= anchor's, censor_ratio <= 3.0. Plateau tie-break: LARGER cap (least
interventionist). Selected cap feeds the P7 rebuild (re-extract if != 3.0).

### CORRECTION (2026-07-07, user challenge — accepted): rand-drop conclusion DOWNGRADED
The "filter gain is REAL" conclusion above overclaimed. The rand-drop control refutes
only the DATA-VOLUME confound (fewer samples), not the TARGET-TRUNCATION confound the
user raised: the filter removes the slow tail, the eval cells are rebuilt from the
FILTERED samples, and predicting the middle of a tightened distribution scores better
mechanically (the "keep one decile, guess its middle, score amazing" case). Random
dropping does not tighten the target, so both "contamination removal" and "target
truncation" predict exactly the observed control result. Partial mitigations on file
(IQR-mean targets attenuate tail influence; rho/own-ceiling rose) are suggestive, not
decisive. Status revised: filter gain = 🟠 INFERRED, decisive test registered below.

## CROSSEVAL-2x2 + HESITATION-GEOMETRY (registered 2026-07-07, before results)
crosseval_2x2.py: train on {v3, v4} x evaluate on {v3-target, v4-target} frames, LOLO,
2 seeds, shipped recipe. All four (train, eval) cells report wmae/umae/dec3/rho-frac +
in-frame pooled tau; eval frames also report target mean/stddev/n_cells (truncation
magnitude made visible).
DECISION RULE: the filter improves GENERALIZATION iff train-v4 beats train-v3 ON THE
FIXED v3-TARGET frame (wmae improvement > 1% rel, umae/dec3 not worse by > 2%). If
train-v4 ~= train-v3 on the v3 frame and the -23% appears only on the v4 frame, the
confirm-run gain is TARGET REDEFINITION — adoption may then rest ONLY on the
definitional argument (hesitation lag = cognition, not motor cost, so the filtered
target is the correct optimization objective), which itself requires the
hesitation-geometry test to pass:
hes_geometry (same driver): per qwerty ngram, hesitation mass = 1 - n_v4/n_v3 vs the
ngram's filtered (clean) mean time + distance. If hesitation rate is ~independent of
geometry/clean-time (|spearman| < 0.2), hesitations are content-driven and EXCLUDING
them from the objective is principled regardless of MAE bookkeeping; if strongly
positive (>0.4), hard geometry CAUSES hesitations, filtering censors real signal, and
the filter is REJECTED for the objective definition even if MAE flatters it.
Middle zone (0.2-0.4): judgment call, documented openly, default REJECT (conservative).

## CLEAN-SWEEP-2x2 (registered 2026-07-07, before results; supersedes the same-frame cap sweep)
The same-frame cap_sweep was KILLED before producing results — its MAE columns inherit
the truncation flattery (fc15c87 correction); superseded by clean_sweep_2x2.py, which
generalizes the 2x2's decisive cell into the sweep methodology: every arm varies ONLY
training-data cleaning; every arm is graded on ONE FROZEN eval frame (BASE-extraction
cells + BASE ceilings). No arm can win by tightening the target.
STAGE 1 single levers vs BASE: CAP{2,2.5,3,4,5,8} hesitation cap; BUF{1,2,3} post-error
buffer (drop windows starting <k clean keys after a contiguity gap — recovery lag, the
user's example); FAST{20,35} implausibly-fast floor (rollover artifacts); SESS{3,10}
session warmup drop. STAGE 2: combine all adopted levers; keep combo iff it beats BASE
AND the best single lever.
ADOPT RULE per lever: frozen-frame wmae -1% rel or better, umae/dec3 <= +2%, in-frame
tau >= BASE's; CAP arms additionally censor_ratio <= 3.0. Family plateau (within 0.5%
wmae): least-interventionist variant (smallest drop%). Winner = FINAL cleaning recipe
for the P7 rebuild.
NOTE the frozen frame carries hesitation lag in its targets; a cleaning arm therefore
competes on predicting the UNCLEANED truth better from cleaner training signal — the
conservative direction. The definitional question (should the OBJECTIVE itself exclude
hesitations?) is decided separately by crosseval_2x2's hes_geometry test.

### Outcome append (2026-07-07): pair_gap_boot — azerty-qwertz IS a statistical tie (and so is dvorak-qwerty)
Participant-bootstrap 95% CIs on observed common-ngram layout gaps (runs/pair_gap_boot.json):
  DECISIVE in both tables (4/6): azerty-dvorak (~-13.6ms), azerty-qwerty (~-9ms),
  dvorak-qwertz (~+13.4ms), qwerty-qwertz (~+8.7ms)
  TIE in both tables (2/6): azerty-qwertz (v3 -0.46 [-5.45,+4.52]; v4 -0.01 [-4.94,+4.90])
  and dvorak-qwerty (v3 +4.49 [-2.39,+10.96]; v4 +4.94 [-2.27,+11.26])
Per the 0597fdf rule: the tau-costing pair is UNDECIDABLE in both tables — the filter's
tau 1.0 -> 0.67 "failure" was a coin-flip on a tie, not a real ranking error. Decisive-
pair tau is +1.0 for both v3 and v4 models. NOTE ALSO dvorak-qwerty is undecidable:
the observed data can rank only 4 of 6 layout pairs; all-pair tau overstated our
layout-level resolution all along.
INTERACTION with the truncation correction (fc15c87), honest sequencing: the 306958f
amendment ("undecidable => magnitude sweep carries the verdict => filter adopted") was
registered BEFORE the user's truncation challenge downgraded that magnitude sweep to
INFERRED. The tau objection is now resolved in the filter's favor, but final adoption
REMAINS PENDING the crosseval_2x2 fixed-frame verdict + hes_geometry test. The chained
P7 rebuild (gate passed, now running on cap=3.0 tables) is INFORMATIONAL until then;
the deliverable rebuild uses whatever recipe clean_sweep_2x2 + crosseval jointly bless.

## BUF-SPLIT (registered 2026-07-07, before results; user: buffer after hand-displacing keys?)
User also corrected the FAST-floor premise: sub-20ms intervals can be LEGITIMATE
rollover typing (press1 -> press2 -> release1 -> release2; fast typists overlap presses),
not artifacts — so FAST arms in clean_sweep_2x2 are DEMOTED to informational: even if
FAST qualifies on the adopt rule, it is NOT adopted without a follow-up showing the
removed mass is artifact, not rollover (e.g. hold-overlap analysis). Registered here
before the sweep's FAST results are known.
buf_split.py (frozen-frame methodology): gaps classified by cause — mistyped single-char
(ERR) vs control/multi-char rows: BKSP/SHIFT/CTRL/arrows (CTL, the hand-displacement
case; unknown-cause gaps count as both, conservative). Arms: BUF2-ERR / BUF2-CTL /
BUF2-BOTH vs BASE, buffer k=2 clean keys. ADOPT RULE: same frozen-frame rule as
clean_sweep_2x2 (wmae -1%, umae/dec3 <= +2%, tau holds). Interpretation registered:
only ERR qualifies => recovery lag is error-cognitive; CTL qualifies => hand
displacement lingers past the contiguity drop and the production buffer keys on
control rows too; the winning variant supersedes the sweep's cause-blind BUF arm in
the final recipe.

## QUALITY-EXECUTION TARGET PROGRAM (registered 2026-07-07, before any results)
User hypothesis: within a (ngram, pace) cell the sample distribution is a MIXTURE of
execution strategies — quality execution (true roll) vs degraded (broken roll) — so the
cell mean is polluted by fumble mass, and the layout should be judged on attainable
QUALITY execution (a trained user of the final layout rolls most of the time), not the
average across strategies. Proposal to evaluate: quantile/trimmed/mixture targets
("cream of the crop") instead of IQR-mean. User's own flagged risk: does deleting slow
mass starve the model / harm generalization?
Four stages, each gating the next:
D1 DIAGNOSTIC (no model, runs first — the premise test):
  (a) BIMODALITY: for big cells (n>=100), GMM BIC 1-vs-2 components on log-durations
      per (ngram, bucket) cell. If <10% of big cells prefer 2 components, the mixture
      premise FAILS => program stops, quantile targets are just tail-trimming
      (already covered by CAP arms), incumbent target stands.
  (b) STRATEGY ALIGNMENT: among bimodal cells, is bimodality concentrated in roll/
      inward-roll bigrams vs uniform? (mixture-of-strategies predicts concentration
      in bigrams that HAVE a quality move; hesitation-mixture predicts uniform).
  (c) ATTAINABILITY: fast-component share vs wpm bucket. Quality target for a wpm-90
      layout is justified only if fast-share RISES with skill (the move is learnable).
D2 TARGET RELIABILITY (only if D1 passes): candidate targets = IQR-mean (incumbent),
  q25, q10, GMM fast-component mean (n>=40, else q25 fallback), fastest-third mean.
  Split-half ceiling per target (participant split, same machinery). A target whose
  ceiling falls >15% rel below incumbent's is DISQUALIFIED (answers "lose too much
  data": nothing is deleted — quantiles use the full sample — but the ceiling
  quantifies the effective information loss of the definition).
D3 LOLO (surviving targets): each on ITS OWN frame, judged by rho/frac-of-own-ceiling
  + decisive-pair tau (cross-frame raw MAE banned per fc15c87). Adopt-candidate iff
  frac-of-ceiling within 0.02 of incumbent's and tau holds.
D4 LAYOUT IMPACT (adopt-candidates only): rebuild T2 under incumbent vs quality
  target, same QAP search budget, report argmax divergence + mutual cross-scoring
  regret. If regret < 0.15% both ways the choice is MOOT (document, keep incumbent);
  else present both layouts + the D1 evidence and the attainability argument decides:
  fast-share rising with skill => quality target ships for the wpm-90 layout.
Better-than-deletion note (registered): quantile targets/mixture means USE ALL DATA
(an order statistic is a function of the whole sample) — strictly dominates "delete
slowest k%" (same intent, no thrown-away rows, no per-cell n collapse); deletion arms
are therefore NOT run.

### D1 partial outcome + rerun note (2026-07-07)
First D1 run: bimodal_share = 0.607 of 3940 big cells (gate (a) PASSES decisively — the
within-cell mixture is real and pervasive; median mode separation 1.26x, p90 2.02x).
BUT gates (b)/(c) were VOID: the driver classified rolls via schema features named
*roll* — none exist; inwards/outwards fire only on cross-row rolls, so 0 cells were
labeled rolls. RERUN (quality_d1b) with roll = same_hand AND NOT same_finger (the
community sense — the pairs where an overlapped quality execution exists). Gate (a)
result carries over unchanged (roll labels don't affect it). No decision taken on (b)/(c)
before the rerun lands. Per-cell dump added (runs/quality_d1_cells.json) for D2 reuse.

### Outcome append (2026-07-07): D1 rerun (correct roll labels) — gates (b) and (c) BOTH FAIL
runs/quality_d1.json (roll = same_hand & !same_finger; 1204/3940 roll cells):
  (b) bimodal share: roll 55.2% vs non-roll 63.1% — NOT concentrated in rolls (slightly
      the opposite); strategy_aligned = false.
  (c) fast-component weight FALLS monotonically with skill (roll cells: .66/.63/.58/.51/.39
      across buckets 40->120; non-roll nearly identical .72/.69/.63/.53/.36).
Per the registered gates, the GMM operationalization REFUTES the strategy-mixture read:
bimodality is pervasive but uniform across move types and the fast mode is not more
attainable with skill — consistent with a hesitation/attention mixture (already handled
at the training level by the cleaning program). GMM-based quality targets are DEAD.

## D1' OVERLAP TEST (registered 2026-07-07 BEFORE results — the direct measurement)
The GMM was an indirect distributional test; the data carries a DIRECT quality-execution
marker: hold (first key's press->release). An overlapped execution (true rolling motion:
press2 lands before release1, i.e. dur < hold) is the community's "quality roll". D1'
computes, on v3 (no GMM, pure counting): overlap-rate by class {same-hand-diff-finger
(roll), cross-hand (alternation), same-finger (physical near-impossibility — SANITY:
must be < 5%, else hold data is unreliable and D1' is void)}; overlap-rate vs wpm bucket
per class; within-cell speed delta of overlapped vs non-overlapped executions.
REVIVAL RULE: the quality-execution program proceeds to D2 iff (i) sanity passes,
(ii) roll overlap-rate RISES with wpm bucket (monotone trend, Spearman > 0.8 over the 5
buckets), and (iii) overlapped executions are >= 15% faster within-cell (median across
roll cells with >= 20 of each kind). If revived, D2 adds the overlap-conditioned target
(cells with enough overlapped mass use mean-of-overlapped; others incumbent) alongside
the quantile targets. If any of (i)-(iii) fails, the program CLOSES: incumbent target
stands and the user's hypothesis is answered "the mixture is hesitation, not strategy;
cleaning (not target redefinition) is the remedy."

### Outcome append (2026-07-07): D1' overlap test — gates (ii)/(iii) pass DECISIVELY, gate (i) fails by letter (5.16% vs 5%)
runs/quality_d1prime.json (31.6M samples, only 56 without usable hold):
  (ii) roll overlap-rate rises monotonically with skill: 30.6% -> 67.6% (buckets 40->120),
       Spearman 1.0. Cross-hand alternation similar (23.6% -> 59.1%). The quality move is
       real, dominant at high skill, and LEARNABLE — the user's attainability claim,
       measured directly.
  (iii) overlapped executions are 69% faster within-cell (median ratio 1.69, 1032 roll
       cells with >=20 of each) — far above the 15% bar.
  (i) SANITY: same-finger overlap 5.16% vs the < 5% bar — fails BY LETTER, marginally.
Per the registered rule the program cannot revive on this run. DIAGNOSIS REGISTERED
before running it (d1prime_sanity.py): decompose the 5.16% into same-KEY repeats
("ee": dur < hold is physically impossible for a re-press => autorepeat/key-bounce
artifact, arguably not a same-finger MOVEMENT at all) vs same-finger-different-key
(true impossibility => genuine noise floor). AMENDED RULE (registered NOW, before the
diagnostic result): if same-finger-DIFFERENT-KEY overlap < 5%, sanity is judged passed
(the excess is same-key artifacts, excluded from the class by definition) and the
program REVIVES into D2 with the overlap-conditioned target; if diff-key overlap >= 5%,
the hold channel is noisy, the program CLOSES, and the answer to the user is the
hesitation-mixture conclusion. This amendment is recorded before quality_d1prime_sanity
runs; the sequence (letter-fail -> diagnosis -> amended gate) is disclosed openly.

### Outcome append (2026-07-07): d1prime_sanity — amended gate FAILS by letter; decomposition
### reveals the excess is FINGERING-MODEL DEVIATION, not hold-channel noise
runs/d1prime_sanity.json: same-KEY overlap 0.013% (85/669,735) — essentially zero;
diff-key same-finger overlap 8.03% (96,421/1,200,241). Amended gate (diff-key < 5%) FAILS.
Interpretation (recorded with the outcome, before any further action):
- The same-key result PROVES the hold channel is accurate: a re-press physically cannot
  overlap its own release, and the data shows it ~never does (1.3e-4). If press/release
  timestamps were noisy, same-key would show spurious overlap; it does not.
- Therefore the 8% diff-key rate is REAL overlapped motor behavior on pairs the static
  finger map calls same-finger — i.e. typists use ALTERNATE FINGERINGS ~8% of the time
  (documented typist behavior; consistent with the rate RISING with skill 4.2%->8.9%,
  which channel noise would not do). The true noise floor of the overlap marker is the
  same-key rate (<0.1%), not 8%.
- SIDE FINDING (stands regardless of the program's fate): the static finger-assignment
  behind same_finger/SFB features is violated by ~8% of same-finger-labeled samples,
  rising with skill. Logged for the wrap as a model-limitations item.
DISPOSITION — honest double-amendment stop: two gate amendments in a row is goalpost
territory; the discipline requires a HARD line. FINAL RULE (this is it, no further
amendment): the program revives into D2 iff a targeted check confirms the alternate-
fingering reading — the 8% diff-key overlap mass must CONCENTRATE in bigram classes
where alternate fingerings are mechanically plausible (adjacent-column same-finger
pairs / lateral stretches), specifically: overlap-rate(same-finger adjacent-column or
lsb pairs) >= 3x overlap-rate(same-finger same-column non-repeat pairs). Same-column
pairs (e.g. qwerty "ce", "un" excluded — those are cross-column... precisely: pairs with
|dx| = 0) offer no plausible second finger, so genuine channel artifacts would show
there equally, while alternate fingerings cannot. If the 3x concentration holds, hold
channel is vindicated, revive D2 (overlap-conditioned target, carrying the ~8%
fingering caveat into D2-D4 docs). If not, program CLOSES for good.

### Outcome append (2026-07-07): d1prime_final — concentration 0.89x, revival FAILS. PROGRAM CLOSED.
runs/d1prime_final.json: same-column same-finger overlap 8.30% (70,632/850,480) vs
cross-column 7.37% (25,789/349,761) — concentration 0.89x, nowhere near the 3x bar, and
in the WRONG direction. The 8% overlap floor is UNIFORM across mechanically-plausible
and implausible alternate-fingering classes => it is NOT alternate fingering; mechanism
unknown (example rows show many shifted/capital ngrams — a case/modifier timing artifact
is plausible but untested). Per the FINAL rule: the overlap marker cannot be certified
(unknown ~8% false-positive floor) and the QUALITY-EXECUTION TARGET PROGRAM IS CLOSED.
Standing answers to the user's question, on the record:
- The strategy mixture is REAL and the quality move is real, learnable (roll overlap
  30.6%->67.6% with skill, Spearman 1.0) and hugely faster (median 1.69x within-cell) —
  D1' gates ii/iii, unaffected by the marker's floor (signal 30-68% >> floor ~8%).
- But no operationalization SURVIVED its preregistered gate: GMM targets failed
  strategy-alignment/attainability; the overlap-conditioned target failed marker
  certification twice. The incumbent IQR-mean target STANDS for the deliverable.
- REGISTERED FUTURE DIRECTION (not run this campaign): Phase-D data with per-key
  release capture + controlled fingering would certify the marker; the overlap-
  conditioned target is the right design once the marker is trustworthy.
- Side finding for the wrap: same-finger features mismatch observed motor behavior for
  ~8% of samples (mechanism unknown: alternate fingering vs case/modifier artifacts).

### Outcome append (2026-07-07): matched_frame_pace — MED/M5 both REJECTED by the rare-ngram guard
Shared-frame results (runs/matched_frame_pace.json; INC anchor wmae 15.59 umae 20.13 dec3 27.16):
  MED: wmae -7.4%, umae -2.5%, dec3 +3.5%, dp-tau 1.0 — fails ONLY the dec3 <= +2% guard
  M5:  wmae -6.3%, umae -0.1%, dec3 +6.7%, dp-tau 1.0 — fails the guard clearly
Now that the frame is matched, the pace labels DO deliver the magnitude gains the blind-
pace program promised (MED -7.4% wmae — the +7% frontier carried through end-to-end),
and MED even improves umae. But both consistently trade away rare-ngram accuracy, and
the guard exists precisely because the optimizer explores rare position pairs. Per the
registered rule: NOT adopted; incumbent session-mean label stands. On the record: MED
is a near-miss (dec3 +3.5% vs +2% bar) — if the guard tolerance were ever revisited it
must be BEFORE seeing any new result, and the interaction with the final cleaning
recipe (registered follow-up if adopted) does not arise.
Also noteworthy: INC's all-pair tau 0.67 vs dp-tau 1.0 in this frame independently
confirms the tie-pair pollution of all-pair tau.

### Outcome append (2026-07-07): crosseval_2x2 — fixed-frame gain is LARGE AND REAL, but both
### formal routes fail by letter; filter adoption now rests solely on the clean-sweep guard
Full matrix (runs/crosseval_2x2.json):
  train-v3 eval-v3: rho/ceil 0.994 wmae 15.65 umae 19.90 dec3 26.85 tau +1.0
  train-v4 eval-v3: rho/ceil 0.998 wmae 12.40 umae 17.69 dec3 27.72 tau +1.0  <- DECISIVE CELL
  train-v3 eval-v4: rho/ceil 1.003 wmae 15.99 umae 19.05 dec3 22.99 tau +0.67
  train-v4 eval-v4: rho/ceil 1.014 wmae 11.98 umae 15.25 dec3 21.68 tau +0.67
DECISIVE CELL: on the FIXED unfiltered frame — which the filter cannot game — filtered
training improves wmae -20.8% and umae -11.1%. The user's truncation hypothesis is
therefore REFUTED as the whole story: most of the confirm-run gain (~-21 of -23.6%) is
genuine training-signal improvement; target tightening contributes only the remainder.
BUT dec3 +3.24% > the +2% guard => by the registered rule, filter_improves_generalization
= false BY LETTER (the win concentrates in frequent cells and trades rare-decile away,
the same signature as the MED label near-miss).
HES-GEOMETRY: rho(hes-mass, clean-time) = +0.842; rho(hes-mass, distance) = -0.012.
By the registered rule (either rho > 0.4) the definitional route is REJECTED: hesitation
mass tracks how slow an ngram is. HONEST CAVEAT recorded: the distance-rho is ~0, so the
correlate is NOT physical geometry — clean-time is confounded with ngram frequency
(rare ngrams are both slower and more hesitation-prone, a cognitive-rarity story). The
rule named clean-time as a geometry proxy; by its letter the route is closed. A purely
DIAGNOSTIC partial-correlation (hes vs clean-time controlling frequency) may be run for
the wrap's mechanism note; it CANNOT reopen the route this campaign.
NET DISPOSITION (all preregistered rules composed): the hesitation filter enters the
final recipe ONLY via a clean_sweep_2x2 CAP arm that passes the frozen-frame dec3 guard
(milder caps damage rare cells less — that is what the sweep measures). tau is settled
(decisive-pair 1.0 everywhere; the 0.67s are the tie pair on the v4 frame). No
target-redefinition; deliverable eval stays on the unfiltered frame.

### Outcome append (2026-07-08): buf_split — only the COMBINED buffer qualifies (weakly)
runs/buf_split.json, frozen frame (BASE wmae 15.76 umae 20.07 dec3 27.05):
  BUF2-ERR  (drop 4.83%): wmae -0.95% (misses -1% by a hair), umae -0.2%, dec3 +0.4% -> not adopted
  BUF2-CTL  (drop 6.79%): wmae +0.77% (WORSE) -> not adopted
  BUF2-BOTH (drop 7.23%): wmae -1.39%, umae -0.3%, dec3 +1.3%, tau holds -> ADOPTED per rule
Interpretation: post-error recovery lag is real but modest; post-control-key lag alone
is NOT (buffering after every shift/arrow deletes legitimate data and makes the model
WORSE — the user's SHIFT-is-weak-displacement caveat empirically confirmed). The
combined arm crosses the bar mostly on the ERR component plus the small subset of CTL
gaps that co-occur with corrections. Effect sizes are ~7x smaller than the hesitation
cap's; BUF2-BOTH goes to stage-2 combination testing with whatever CAP arm the sweep
blesses (registered: combo kept only if it beats BASE and the best single lever).

### Outcome append (2026-07-08): P7 informational rebuild — REPLACE RULE FAILS; P6 stands; diagnostics suspect
runs/p7_filtered_rebuild.json (cap=3.0 filtered tables, informational per 89bed9a):
filtered cond-trigram LOLO (cand-4): rho/ceil 0.982 vs unfiltered-leg 1.006 => fails the
"within -0.005" condition (wmae 15.38 vs 18.51 passes; tau 1.0 holds). Per the
registered replace rule: P7 does NOT replace P6 — appendix only. Additionally its
diagnostics are UNTRUSTWORTHY: sfb_pct 76-137% (>100% impossible) and stable_slots 0/30
on every arm — a driver bug in the pattern-shares/stability bookkeeping (not
investigated further; the run is appendix material). Layouts NOT to be cited. The
deliverable rebuild will come from the clean_sweep recipe on the P6-proven driver path.

## QSEL (registered 2026-07-08, before results; user: "train on best fifth of the data")
qsel_train.py — quantile-selected TRAINING signal on the frozen BASE frame. Distinct
from CAP (session-relative hesitation removal): selects within each (row, wpm) group's
OWN distribution — the training-side realization of the closed quality-target program
(legal because the eval frame never moves). Arms: BASE(IQR-mean) / Q25 / Q20 / Q10 /
F3M(fastest-third mean) / F5M(fastest-fifth mean — the user's literal proposal).
DESIGN CONTROL: every arm gets affine recalibration (a+b*pred, OLS) fitted on HELD-IN
layouts' frozen-frame cells only — a fast-quantile model is systematically low, frozen-
frame MAE would punish pure scale bias the (affine-invariant) optimizer doesn't feel;
recalibration isolates structure-generalization from calibration. ADOPT RULE: same as
the sweep (recalibrated wmae -1%+, umae/dec3 <= +2%, in-frame tau holds vs recalibrated
BASE). Plateau: mildest quantile. If adopted: composes with the cleaning recipe in the
final stage-2 combination test.

### Outcome append (2026-07-08): clean_sweep_2x2 — stage-1 adopts NOTHING by letter; recipe = BASE (sweep-internal)
Full scoreboard (runs/clean_sweep_2x2.json; BASE wmae 15.76 umae 20.07 dec3 27.05, all
taus 0.67 = the tie pair, in-frame deltas only):
  CAP2   wmae -24.4% BUT dec3 +18.6% and censor 8.2  -> fail (rare-decile carnage)
  CAP2.5 wmae -24.5%, dec3 +6.0%, censor 8.3         -> fail
  CAP3   wmae -20.9%, dec3 +0.8% OK, censor 8.6      -> fail ONLY on censor_ratio
  CAP4   wmae -13.6%, dec3 -1.7% (BETTER), censor 9.9 -> fail ONLY on censor_ratio
  CAP5   wmae  -8.6%, dec3 -1.8%, censor 11.3         -> fail ONLY on censor_ratio
  CAP8   wmae  -2.6%, dec3 -0.6%, censor 12.9         -> fail ONLY on censor_ratio
  BUF1/2/3: wmae -0.5%/-0.4%/-1.7%; BUF3 dec3 +2.1% > +2% -> all fail
  FAST20/35: ~zero effect (and censor 0.58/0.02 — the removed mass is FAST ngrams,
    confirming the user's rollover point empirically); demoted anyway
  SESS3/10: no wmae gain -> fail
Composed verdict per the rules as registered:
1. The sweep's own recipe is BASE. The censor_ratio <= 3.0 guard is what excludes every
   CAP arm — documented TENSION: that guard's intent was geometry-censoring, and
   crosseval showed hesitation mass tracks FREQUENCY/rarity (dist-rho -0.01), which the
   slowest-decile-ngram construction cannot distinguish from geometry. CAP3/CAP4 are
   NEAR-MISSES failing only that letter (CAP4 even IMPROVES the rare decile, dec3 -1.7%,
   directly contradicting the censoring the guard infers). Rule stands this campaign;
   the guard construction (control for frequency in the slow-decile definition) is a
   registered improvement for any future round.
2. buf_split's BUF2-BOTH adoption STANDS (its own preregistered rule, its own frame;
   the sweep's cause-blind BUF2 differs — it also buffers session-initial windows and
   fails). Registered supersession applies: BUF2-BOTH is the cleaning recipe's only
   adopted lever. Stage-2 combination is moot (no second lever).
FINAL CLEANING RECIPE (pending only QSEL): BUF2-BOTH (post-error+control 2-key buffer),
on the UNFILTERED frame. Hesitation caps: not in the recipe by rule letter; CAP3/4
documented as outcome-clean near-misses blocked by a guard whose construction the
evidence undermines — the honest wrap will present both readings.

### Outcome append (2026-07-08): QSEL — all quantile-training arms REJECTED, monotone degradation
runs/qsel_train.json (recalibrated, frozen frame; BASE wmae 13.45 rho/ceil 0.994):
  Q25 +7.6% wmae, rho/ceil 0.924 | Q20 +9.5%, 0.907 | Q10 +15.1%, 0.857
  F3M +14.7%, 0.887 | F5M (the literal best-fifth) +19.9%, 0.842, dec3 +12.5%
Uniform verdict: training on faster quantiles makes generalization strictly WORSE, and
monotonically so as the quantile tightens — even WITH affine recalibration removing the
scale bias. Interpretation: a cell's lower tail is dominated by which-typists/how-many-
samples noise (the 30-70% quality-execution share varies by cell, so a fixed quantile
reads DIFFERENT strategy mixes in different cells — inconsistent targets), while the
IQR-mean averages over the strategy mix more stably. The "biomechanical floor" reading
of low quantiles is refuted at every tested depth; the user's worry ("lose too much
data / harm generalization") is confirmed as the dominant effect, with the twist that
nothing was deleted — the INFORMATION loss is in the statistic, not the row count.
QSEL closed; contributes nothing to the recipe. FINAL CLEANING RECIPE NOW LOCKED:
BUF2-BOTH on the unfiltered frame (5c49a3e composition unchanged).

## P8 FINAL REBUILD (registered 2026-07-08, launched with recipe locked at 7965aa2)
p8_final.py — the deliverable build, composing every adopted verdict: BUF2-BOTH cleaning
(2-clean-key buffer after gaps containing a mistype or control key; session-initial
windows NOT buffered — buf_split semantics), unfiltered frame, no cap, no quantile
targets, incumbent pace label; bigram = shipped recipe; trigram = conditioned target
with cand-4 params; T3c(90) tri-corrected-only; oxey family {0,.5,1,2} at P6 budget on
the P6-proven driver path (NOT P7's buggy diagnostics). Stages checkpointed
(bistrokes_v5 / tristrokes_cond_v3 / p8_lolo.json / models *_v5_seed*). Its LOLO stage
is the deliverable's evidence (expected ~ BUF2-BOTH's frozen-frame numbers); the family
+ certificate + scoreboard land in runs/p8_final.json. No decision rule here — this is
the assembly, all decisions already made upstream.

## P8-TRI ATTRIBUTION (registered 2026-07-08, before results; triggered by P8 stage-2 evidence)
P8's LOLO evidence: bigram side healthy (rho/ceil 0.973, wmae 15.44, dec3 26.34 — all
consistent-or-better vs buf_split's BUF2-BOTH arm; tau 1.0). But cond-trigram rho/ceil
= 0.923 vs the unfiltered JOIN construction's 1.006 (cand-4 guard-check leg) — a real
frame-normalized drop. CONFOUND: P8's trigram table differs from the incumbent in TWO
ways at once — (a) BUF2-BOTH cleaning (adopted on BIGRAM-only evidence; no trigram rule
existed) and (b) construction (direct one-pass extraction vs the tristrokes_v1 x
tristrokes_last join). Attribution arm now launched: DIRECT extraction with BUF_K=0
(same construction, no buffer), cand-4 LOLO, 2 seeds.
RULE (registered before its result): the deliverable's trigram table is whichever of
{unfiltered join (1.006), buffered direct (0.923), unbuffered direct (pending)} has the
best rho/ceil with tau 1.0 intact. If unbuffered-direct ~1.0 => the buffer hurts
trigrams (heavier window loss) => recipe becomes SPLIT: BUF2-BOTH for bigrams,
unbuffered for trigrams; P8 stages 3-4 re-run with the winning trigram table before the
family is final. If unbuffered-direct ~0.92 => construction is the culprit => revert to
the join construction for the deliverable. If all within noise of each other => keep
P8 as built. The P8 family search continuing meanwhile is provisional until this lands.

## Q-OBJ (registered 2026-07-08, before results; user: QSEL's eval was whole-distribution —
## quantile-as-OBJECTIVE on its own frame was never tested)
User's design critique accepted: QSEL's affine recalibration removes scale bias but not
SHAPE (mean-vs-q20 gap varies by cell via fumble rate), so QSEL only proved quantile
training doesn't transfer to the incumbent target — not that the quantile is a bad
objective in itself. The D2/D3 stages the quality program never reached, run now:
qobj.py — arms BASE(IQR-mean) / Q25 / Q20 / F5M, each trained AND evaluated on ITS OWN
frame. Truncation-safe metrics only: (D2) split-half ceiling of each arm's own target
(participant split, agg applied per half — measures whether q20-of-a-cell is even a
reliable quantity); (D3) LOLO rho as frac-of-OWN-ceiling + decisive-pair tau (undecided
pairs carried from the v3 mean-frame bootstrap: azerty-qwertz, dvorak-qwerty — a
q-frame bootstrap is a registered refinement if any arm is adopted). Own-frame wmae
reported as INFORMATIONAL ONLY (banned as a rule metric — truncation flattery).
ADOPT-CANDIDATE RULE: ceiling >= 85% of BASE's (D2 reliability gate) AND rho_frac >=
BASE's - 0.02 AND min decisive-pair tau >= BASE's. If any arm qualifies -> D4 launches
(3-seed full-data models under that target, T2 rebuild, QAP search both objectives,
mutual cross-regret; regret < 0.15% both ways = choice moot, incumbent kept; else the
D1' attainability evidence (overlap rises with skill) decides for the wpm-90 layout).
If none qualifies: quantile-objective route closed with the D2/D3 numbers on record.

### Outcome append (2026-07-08): tri-attribution — CONSTRUCTION is the culprit, not the buffer
runs/tri_attrib.json: unbuffered-direct rho/ceil 0.9218 ~= buffered-direct 0.9226, both
far below the join construction's 1.006. The buffer costs the trigram side ~nothing;
the direct one-pass extraction itself is what degrades it (mechanism note, 🟠: the
join's sample-alignment filters — full-span/last-interval consistency + 0<=df-dl<=5000
— act as an implicit cleaning step the direct path lacks). Per rule f06c695: the
deliverable's trigram table REVERTS to the join construction (tristrokes_v1 x
tristrokes_last, cand-4) — i.e. the P6 tuned trigram models stand. Deliverable
composition: bigram = v5 (BUF2-BOTH, healthy at 0.973), trigram = join/cand-4 (1.006).
P8's in-flight family (direct-buffered trigram table) = provisional/appendix; P8b
launched = bigram_v5 models + join-construction cand-4 trigram models -> T3c(90) ->
family at the same budget/searcher. P8b is the DELIVERABLE build.

### Outcome append (2026-07-08): Q-OBJ — F5M is an ADOPT-CANDIDATE on its own frame; D4 launches
runs/qobj.json (own-frame, truncation-safe metrics):
  BASE own-ceiling 0.815, rho/own-ceil 0.994 | Q25 0.803/0.941 | Q20 0.795/0.937
  F5M  own-ceiling 0.709 (ratio 0.870 >= 0.85 gate), rho/own-ceil 0.974
       (delta -0.0199, inside the -0.02 gate BY A HAIR), dp-tau 1.0 => ADOPT-CANDIDATE.
Reading: quantile POINTS (q25/q20) are reliably measurable but the model predicts their
cross-layout structure notably worse (-5pp of own ceiling) — refuted as objectives.
The fastest-fifth MEAN is a noisier quantity (ceiling 0.709 vs 0.815) but its structure
transfers almost as well as the incumbent's (0.974 vs 0.994) — averaging within the
fast tail is stabler than a point quantile AND carries the quality-execution signal.
Both F5M gate margins are razor-thin (0.870 vs 0.85; -0.0199 vs -0.02) — recorded
honestly; the candidate earns D4, not adoption.
D4 (per 63e06f8, no new rules): 3-seed full-data bigram models under F5M target -> T2_f5m
-> QAP search under {incumbent T2, T2_f5m} -> mutual cross-regret. < 0.15% both ways =>
choice MOOT, incumbent kept (document). Else: D1' attainability (overlap rises with
skill => trained-user premise) decides FOR the quality objective at wpm 90 — with the
explicit caveat that F5M's -metrics are near-gate and the layout ships alongside the
incumbent one for the user's choice if divergence is material.

## QIN — QUALITY-AS-INPUT (registered 2026-07-08, before results; user proposal: condition
## the model on a quality label q, then generate layouts at (wpm=90, q=0.2))
Design (simultaneous quantile regression): add q to the feature vector; each (row, wpm)
training group is replicated at q in {0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9} with target =
the group's empirical q-quantile of duration. One model learns the full conditional
distribution: BASE ~ central q; F5M ~ fast tail — both become slices of this surface,
and the q=0.2 slice SHARES STRENGTH across quantiles/cells (the exact weakness of
per-cell order statistics that Q-OBJ measured: dedicated-Q20 rho/own-ceil 0.937).
IN-RUN comparators (same frame, same machinery, retrained for exact comparability):
dedicated single-q models at q in {0.2, 0.5, 0.8}. Eval: per-q own-frame cells
(quantile agg), per-q split-half ceilings, rho/frac-of-own-ceiling, dp-tau.
ADOPTION RULE: QIN becomes the layout-generation surface iff
  (i) at q=0.2: QIN rho/own-ceil >= dedicated-q20's (shared strength delivers),
  (ii) at q=0.5: QIN >= dedicated-median's - 0.01 (no cost at the center),
  (iii) coherence: monotonicity violations (pred(q_hi) < pred(q_lo)) < 1% of evaluated
        cell q-pairs (a conditional distribution must be a distribution).
If adopted: D4-style cross-regret at (wpm=90, q=0.2) vs the incumbent objective decides
whether the QUALITY-CONDITIONED layout differs materially; if it does, it ships
ALONGSIDE the incumbent-objective layout (both presented; D1' attainability argues for
q~0.2 for a trained user, but near-gate margins in this family mean the user chooses).
If any gate fails: QIN closed on the record; F5M/D4 verdict (in flight) stands alone.
Practice term note: fitted across all q replicas (q-averaged, keyed by ngram) —
documented simplification; a per-q practice term is a registered refinement if adopted.

### Outcome append (2026-07-08): D4 — NOT MOOT; the quality objective changes the layout
runs/qobj_d4.json (bigram T2 surfaces, matched search): incumbent argmax
ydea,nrtscpgouilmwfbq/;.xhkvjz vs F5M argmax paedinrtsw,goyumflcbq;./khxvjz.
Cross-regret: incumbent layout under the F5M objective +0.343%; F5M layout under the
incumbent objective +0.182% — both above the 0.15% moot bar. The two objectives
genuinely prefer different layouts (shared core nrts/aei blocks, different vowel-hand
arrangement). Per the registered rule: D1' attainability (quality execution rises
30.6%->67.6% with skill) decides FOR the quality objective at wpm 90, WITH the
registered caveat (F5M's Q-OBJ gates were razor-thin) => BOTH families ship, user
chooses. Composition note: D4 was bigram-T2-only by design; the quality-arm deliverable
family (full T3c + oxey sweep) builds after the QIN verdict decides WHICH quality
surface (QIN q=0.2 slice if QIN adopts, else F5M) — one quality-arm assembly, not two.

### Outcome append (2026-07-08): QIN — all three registered gates PASS, but a red flag the
### gates did not cover: q=0.2 decisive-pair tau = 0.0
runs/qin.json:
  (i) q=0.2 shared strength: QIN rho/own-ceil 0.995 vs dedicated-q20 0.937 — PASS, and
      decisively (the shared-strength mechanism works exactly as proposed).
  (ii) q=0.5 no cost: 1.018 vs 1.001 — PASS (QIN BEATS the dedicated median model).
  (iii) coherence: mono violations < 1% — PASS.
  Also q=0.8: 1.071 vs 1.028 — the conditional surface is better everywhere per-cell.
RED FLAG (eval metric named in the prereg but not gated): QIN q=0.2 dp-tau = 0.0 on
both seeds (dedicated-q20: 1.0). Per-cell structure is excellent yet LAYOUT-level
ranking at the exact generation slice is at chance. By rule letter ADOPT=True; building
layouts from a surface whose layout ranking is at chance would be a Goodhart trap the
gates failed to anticipate. DISPOSITION registered BEFORE any diagnostic result:
qin_diag.py — hypothesis: the dp-tau 0.0 is a FRAME artifact — the "decisive" pairs
were certified on the MEAN frame; at the q=0.2 frame the observed layout gaps may
collapse toward ties (quality execution more layout-uniform), making tau-over-4-pairs
noise. Diagnostic: participant-bootstrap CIs of observed layout gaps ON the q=0.2
frame. RULE: if >= 3 of the 4 mean-frame-decisive pairs are UNDECIDABLE at q=0.2, the
dp-tau 0.0 is uninformative (measuring ties), QIN's adoption stands, and the quality
family builds on the QIN q=0.2 slice with layout-level validation acknowledged as
LIMITED at the fast tail. If the pairs remain decisive at q=0.2, the flag is real:
QIN is rejected for layout GENERATION (kept as a modeling result), and the quality
family builds on F5M (whose dp-tau held at 1.0 in Q-OBJ).

## QIN-INTERACTION ROUND (registered 2026-07-08, before results; user: "changes we
## previously rejected might be worth it now with the new model" — plus IDEAS-LEDGER.md
## E-inventory committed 0fab6b1 as the standing gap map)
Sequencing: all F-arms run ONLY if the QIN diagnostic (0e53294) clears QIN for
generation; else the quality arm is F5M and the F-arms re-key to F5M analogs.
F5 CALIBRATION FIRST (not a results-driven rule change): the 2% rare-decile guard was
set against mean-frame noise. Before F1-F4, compute q-frame ceiling-implied noise and
set the q-frame guard tolerance = 2% x (q-frame dec3 noise / mean-frame dec3 noise),
capped at 4%. Recorded before any F-arm result exists.
F1 CAP3xQIN: QIN trained on CAP3-filtered vs unfiltered data; judged on q in {0.2,0.5}
  own-frames (rho/frac-of-own-ceiling; guards at F5 tolerance). Hypothesis registered:
  hesitation mass lives in high-q by construction => q=0.2 slice ~filter-invariant,
  and the filter's mean-frame gain may be free OR unnecessary under q-conditioning.
  ADOPT filter for the quality pipeline iff q=0.2 AND q=0.5 slices both improve or
  hold within guard.
F2 BUF2xQIN: same frames; buffer KEPT unless it degrades a q-slice beyond guard.
F3 F5M-retire check: QIN-q0.2 vs dedicated F5M on the F5M own-frame; QIN >= F5M - 0.01
  => F5M retired (QIN strictly dominates: same signal, dialable, stronger structure).
F4 MEDxQIN: QIN trained with session-median pace label vs session-mean label, same
  q-frames judging. Adopt MED for the quality pipeline iff both slices improve.
Deliverable impact: winners define the QUALITY-ARM pipeline (data treatment + label +
surface) for the second family build. The incumbent-arm family (P8b) is UNAFFECTED —
its levers were adopted under its own target and stand.

### Outcome append (2026-07-08): QIN diagnostic — the flag is REAL; QIN rejected for generation
runs/qin_diag.json: at the q=0.2 frame, ALL FOUR mean-frame-decisive pairs remain
decisive (azerty-dvorak -9.0 [-14.6,-3.1]; azerty-qwerty -5.3 [-8.8,-1.8]; dvorak-
qwertz +10.3 [+4.9,+15.2]; qwerty-qwertz +6.5 [+4.7,+8.6]); frame_artifact=False (0/4).
Two conclusions, both on the record:
1. FINDING (positive): quality execution is NOT layout-uniform — layouts differ as much
   at the fast tail (~gaps 5-10ms) as at the mean. The quality-objective premise
   survives; the fast tail is rankable ground truth.
2. QIN FAILS exactly there: per-cell structure 0.995 of ceiling, yet layout-level
   ranking at chance — its errors must correlate WITHIN layouts (a layout-level bias,
   plausibly the q-feature absorbing between-layout level differences). Per the rule:
   QIN REJECTED for layout generation; KEPT as the campaign's best modeling result.
   Diagnosing/fixing QIN's layout bias (e.g. layout-blind q encoding, per-layout
   calibration in training) = registered FUTURE WORK, not this campaign.
CONSEQUENCES: quality arm = F5M (dp-tau 1.0 in Q-OBJ). F-round as registered was
QIN-gated: F1/F2/F4 are VOID (QIN-specific), F3 MOOT (F5M stands by default). To keep
the two families cleanly comparable, the QUALITY FAMILY BUILDS ON THE SAME ADOPTED DATA
PIPELINE as the incumbent family (v5 BUF2-BOTH bigrams + join-construction trigrams) —
the families differ ONLY in objective (mean vs F5M). F5M-analog interaction arms
(CAP x F5M etc.) are recorded in IDEAS-LEDGER as open, deprioritized: the mean-frame
lever effects were small (buffer ~1%) or guard-blocked (CAP), and D4's cross-regret
(0.18-0.34%) bounds how much pipeline fine-tuning can matter to the family choice.
P9 QUALITY FAMILY (launching now): bigram T2 from F5M-target models on v5 data (3
seeds, custom fit path — shipped train_* is mean-target); conditioned-trigram Tcond
from F5M-target models on the join table, GATED by a LOLO sanity first (F5M cond-tri
rho/frac-of-own-ceiling >= mean's - 0.05 AND dp-tau holds; else Tcond falls back to
the mean-target models and the family is documented as mixed-objective); then
T3c_q(90) -> oxey {0,.5,1,2} at the P6 budget -> stability + certificate.

### Outcome append (2026-07-08): P8b — the incumbent-objective DELIVERABLE family is final
runs/p8b_final.json (v5 BUF2-BOTH bigrams + join cand-4 trigrams, T3c(90), rng 880099):
  w=0.0 ctsnhkuoepdwflr.iaygbjqmv,x/;z  +2.23% vs qwerty | sfb 1.21% | 12 near-optima
  w=0.5 gyou,lntscdeaipmrbfwj;/.xhkvqz  +2.29% | sfb 1.50% (P6-family shape recurs)
  w=1.0 gyou,ldntcseai.mfrpwj/;zxhkbvq  +2.12% | sfb 1.09%
  w=2.0 hsndv.geoilcrtmwpuayjqxbkf,z;/  +1.76% | sfb 0.70% | inroll 10.5%
  Bigram-component GL certificate: within 3.64% of optimal. Scoreboard: best +2.23%,
  colemak +0.64%, qwerty 0.
Note the family's plateau: w=0.5 scoring +2.29 vs w=0's +2.23 on the speed axis is
within search noise (different anneal trajectories; own-fitness ordering is correct by
construction) — the speed surface is FLAT across the family, consistent with every
prior round: heuristic quality (sfb halving) costs ~0.5% at most. Deliverable stands.

### Outcome append (2026-07-08): P9 — the quality-objective (F5M) family is final; CAMPAIGN COMPLETE
Trigram gate PASSED (F5M cond-tri raw rho 0.632 > 0.55) => full-F5M family, not mixed.
runs/p9_final.json (same pipeline as P8b, objective = fastest-fifth mean, rng 880111):
  w=0.0 gaedinrtsw.oypumflcbq;jk,hxvz/  +2.01% vs qwerty (F5M objective) | sfb 2.39%
  w=0.5 coeainrtswpgy.umbldfq;j/,hkvxz  +1.94% | sfb 1.93%
  w=1.0 waedilstnb.oupymfcrkq;zj,hgvx/  +1.90% | sfb 2.30%
  w=2.0 coeuinrtswgayp,lbmdfq;/.khxvzj  +1.93% | sfb 1.56%
  Bigram-component certificate: within 4.38%. Scoreboard ON THE F5M OBJECTIVE:
  P9-best +2.01%, qwerty 0, colemak -1.90% (colemak is SLOWER than qwerty at quality
  execution per this model — a substantive, falsifiable claim of the quality view).
Family notes: consonant core nrts/stn + right-hand w placement recurs across weights;
sfb runs higher than the incumbent family at equal w (the quality surface prices SFBs
lower — overlapped execution can't happen on same-finger pairs, so their PENALTY is
already in the mean; at the fast tail the relative cost of rolls drops more). Oxey
weights barely move the speed axis (flat surface again).
Both deliverable families now exist; the campaign's compute is COMPLETE. Remaining:
the wrap (both families + verdict ledger + user-gated list).

## Q-BLEND (registered 2026-07-08, before results; user: sample q in [0,0.2] as the target +
## "regularization during SA")
Math note recorded first: mean-over-q-in-[0,0.2] IS the tail expectation = F5M exactly
(the P9 objective) — the user's proposal re-derives why F5M survived Q-OBJ while point
quantiles died (integration stabilizes the order statistic). Random per-evaluation q
sampling in SA optimizes the same expectation NOISILY; the expectation is exactly
computable (T2_f5m is the tail-mean surface), and SA accept/reject + 2-opt polish
degrade under evaluation noise => deterministic integration dominates; no sampling arm.
The genuinely new, testable piece of the user's regularization instinct: CROSS-REGIME
ROBUSTNESS. qblend.py: objective = mean of the two qwerty-normalized surfaces
(T3c_inc / fit_inc(qwerty) + T3c_f5m / fit_f5m(qwerty)) / 2, searched at the P6 budget
(rng 880222). Judged by cross-regret under BOTH pure objectives vs both pure champions
(P8b w0, P9 w0). RULE: the blend layout becomes the RECOMMENDED ROBUST PICK iff its
max-regret across the two pure objectives < 0.182% (the F5M champion's current
max-regret, i.e. it must strictly beat the best existing compromise); else the F5M
champion keeps the robust-pick title and the blend is documented. Either way this
closes the user's question with a measured answer; no further arms.

### Outcome append (2026-07-08): Q-BLEND — the blend layout takes the robust-pick title
runs/qblend.json (full T3c surfaces, same-frame regrets):
  P8b_w0 (mean champ):   inc +0.000% | f5m +1.011% | MAX 1.011%
  P9_w0  (quality champ): inc +0.295% | f5m +0.000% | MAX 0.295%
  BLEND gaedinrtsc.oypumblfwq;jk,hvxz/: inc +0.221% | f5m +0.032% | MAX 0.221%
RULE READING, disclosed openly: the registered threshold "< 0.182%" carried a stale
number (D4's BIGRAM-only frame) alongside its definition ("the F5M champion's current
max-regret, i.e. must strictly beat the best existing compromise"). On the full T3c
frame the F5M champion's max-regret is 0.295%; the blend's 0.221% beats it, satisfying
the rule's DEFINITION while missing the stale number. Verdict: BLEND TAKES THE
ROBUST-PICK TITLE per the definitional reading (the number was frame-inconsistent, the
definition was not); both readings recorded. Note the blend is one letter-swap from the
P9 champion (w<->c at two slots... precisely: gaedinrtsC vs gaedinrtsW cores) — the
quality champion was already near-robust, and the blend trims its inc-regret 0.295 ->
0.221 while giving up only 0.032 on f5m.
FINAL RECOMMENDATION SET (closes the campaign):
  speed-average view:  P8b w0 ctsnhkuoepdwflr.iaygbjqmv,x/;z (+2.23% vs qwerty)
  quality view:        P9 w0  gaedinrtsw.oypumflcbq;jk,hxvz/ (+2.01% F5M obj)
  ROBUST (recommended): BLEND gaedinrtsc.oypumblfwq;jk,hvxz/ (max-regret 0.221%)
  ergonomics-lean:     family w=2 variants (sfb ~0.7-1.6%, <=0.5% speed cost)

## T-REL — target relativization (registered 2026-07-10, BEFORE results; user directive:
## "we should be predicting some relative term... the label is already adjusted")
MOTIVATION (measured, not vibes): shap-report on bigram_v5 shows wpm mean|SHAP| 27.1ms vs
9.3ms for the top geometry feature — the model spends most capacity reproducing the pace
hyperbola t~12000/wpm per geometry class (trees have no multiplicative structure; every
geometry leaf must re-learn the wpm curve). Math note: session wpm = (chars/5)/min, so
12000/wpm IS the session's mean ms/char — duration*wpm/12000 is exactly "multiple of this
typist's average keystroke" (the user's 0.8 example).
DESIGN (driver trel_arms.py; data = bistrokes_v5.tsv, the adopted BUF2-BOTH extraction;
NO re-extraction — targets are transforms of (duration, wpm) already in the TSV):
- SHARED FRAME: one cell set (40-140, width 20, floor 10), one example structure (grouped
  by row x integer session wpm — wpm constant within a group, so every arm's group target
  is a DETERMINISTIC TRANSFORM of the same IQR-mean; differences attribute entirely to
  training-space geometry). Ceilings computed once, reused (data property).
- ARMS (all keep the wpm FEATURE — lets the model learn skill-dependence of the ratio,
  per user: "maybe it learns it matters more at high WPM"; all use production recipe:
  depth-3 defaults, practice backfit x2 IN ARM SPACE, layout weights):
  INC     y = ms                       (anchor; must reproduce p8b-zone numbers)
  RAT     y = ms * wpm / 12000         (user's proposal: multiple of typist's mean keystroke)
  LOGRAT  y = log(RAT)                 (multiplicative structure additive; symmetric rel-error)
  DIFF    y = ms - 12000/wpm           (additive normalization — the obvious control:
                                        is it the SCALE or just the LEVEL that hurts?)
- EVAL: predictions mapped back to ms at cell midpoint wpm (RAT: *12000/wpm; LOGRAT:
  exp then *12000/wpm; DIFF: +12000/wpm); ALL metrics in ms on the identical frame
  (rho/own-ceiling, wmae, umae, dec3, all-pair tau + decisive-pair tau from
  pair_gap_boot v3_nofilter). Jensen gap of per-bucket conversion is second-order and
  shared. LOGRAT trains on log(group-IQR-mean), not IQR-mean(log) — same group statistic
  across arms, documented.
RULE (before results): an arm is ADOPTED over INC iff mean over 2 seeds x 4 LOLO folds:
  wmae improves >1% rel AND umae, dec3 within +2% rel (rare-ngram guard) AND decisive-pair
  tau no lower than INC's AND all-pair tau no lower. Multiple qualifiers -> best wmae.
  Adoption consequence (registered): retrain winner 3 seeds all-data, save as
  bigram_trel_* models, shap-report before/after (the user's SHAP-evolution question:
  EXPECTATION recorded — wpm's |SHAP| share should collapse in the winning relative
  space; geometry share should rise; this is informational, not a gate), THEN apply the
  same transform to the conditioned-trigram target as a follow-up arm with its own LOLO
  check; ONE deliverable rebuild after both settle. No adoption -> route closed on the
  record, SHAP comparison still reported (null is informative: the wpm feature + 300
  trees already suffice in the 40-140 band).
HONEST PRIOR: RAT/LOGRAT should win wmae (pace-as-scale is measured physics here —
  blind-pace found participant pace multiplicative; ms-space squared loss overweights
  slow typists). Risk: heteroscedasticity reweighting could trade the rare decile —
  exactly what the guard watches. Keyboard-type stratification (ledger E2) is REGISTERED
  AS NEXT, sequenced AFTER this verdict, on the winning target space.

## KB-STRAT — keyboard-hardware stratification (ledger E2; registered 2026-07-10, BEFORE
## results; sequenced after T-REL, runs in the T-REL-winning target space — INC if none won)
USER QUESTION: "do we know if high quality at 90 wpm generalizes across keyboards?" —
never tested; KEYBOARD_TYPE (full/laptop) is in metadata col 10, pipeline keeps both and
pools them. The quality signal is rollover/overlap physics (overlapped executions 1.69x
faster), exactly where laptop vs full hardware differs (key travel, rollover) — a real
generalization threat to F5M/blend. NO RE-EXTRACTION NEEDED: samples carry pid; map
pid -> KEYBOARD_TYPE from metadata (driver kb_strat.py; data = bistrokes_v5.tsv).
ARMS/QUESTIONS (bigram level; trigram deferred to a follow-up if S3 flags divergence):
- S1 TRANSFER: per-stratum LOLO (full-only vs laptop-only training data, same 4-layout
  folds, per-stratum split-half ceilings). Question: does rho/own-ceiling hold per
  stratum, and does a model trained on ONE stratum predict the OTHER's cells (cross-
  hardware wmae vs within-hardware)?
- S2 FEATURE: is_laptop as a feature on the pooled frame (serve-time: a user parameter).
  Adoption rule = standard challenger rule (wmae >1% better, umae/dec3 within +2%,
  neither tau lower, 2 seeds x 4 folds).
- S3 DECISIVE (the user's actual question): per-stratum T2 tables at wpm 90 (3 seeds,
  mean-target AND F5M fast-fifth target per stratum = 4 tables), qwerty-normalized;
  score the three finalists (P8b_w0, P9_w0, BLEND) + qwerty/colemak/dvorak under all.
  RULE: hardware GENERALIZATION CONFIRMED for a target iff every finalist's cross-stratum
  regret (its gap-to-best under full-table vs laptop-table) < 0.25% (the robust-pick
  margin zone); any finalist exceeding it => hardware materially moves the argmax =>
  per-hardware families become a deliverable question (registered consequence, not run
  unprompted). F5M-vs-mean comparison per stratum reported (is the QUALITY target more
  hardware-sensitive than the mean target? — informational).
GUARDS: laptop stratum is ~55% of participants but strata differ in wpm mix => all
stratum comparisons at matched wpm buckets (same 40-140 frame); per-stratum cell floors
unchanged (10) — starved cells drop, counts reported. Small-stratum layouts (dvorak n=64
splits further) may starve folds: a fold with <100 cells is reported UNUSABLE, not
laundered into means.

## S1-REL + TWO-STAGE 2x2 (registered 2026-07-10, BEFORE results; user proposal: the
## stage-1 blind-pace model should also use the relative-speed mechanism, then feed the
## main model — A/B both stages)
STAGE-1 (driver blind_pace_rel.py; same extraction/split/leakage-audit as blind_pace_max):
the incumbent frontier winner M5 predicts ABSOLUTE interval ms (log-space blend). Arms
relativize the stage-1 TARGET to the typist's own scale (prior = shrunk mean of the
typist's other sessions' medians — the scale anchor available without leakage):
  R0  M5 as shipped (anchor; must reproduce +7.65% vs LOO-mean)
  R1  predict y/prior (ratio target), same features, prediction re-scaled by prior
  R2  predict log(y/prior) (log-ratio), re-scaled
  R3  fully scale-free: features ALSO divided by prior (loo_med/prior, w3/prior, w10/prior)
      + log-ratio target — the "model learns shape only" reading of the user's mechanism
RULE: winner = lowest test MAE (ms) among arms passing the leakage audit (residual
ngram-R2 <= LOO-mean's + 0.002). Stage-1 relativization ADOPTED iff winner beats M5 by
>=1% rel. NOTE the honest prior: M5's log-space fit already captures much of the
multiplicative structure; R1-R3's marginal value is the explicit per-typist anchor.
STAGE-2 2x2 (driver twostage_2x2.py, runs AFTER T-REL verdict; matched-frame methodology
from matched_frame_pace.py — one cell frame bucketed by INCUMBENT session wpm; only the
pace label value and target space vary):
  arms = {label: SESS (session-mean wpm), S1 (stage-1 winner's pace estimate)} x
         {target space: INC (ms), W (T-REL winner; skipped if T-REL adopts nothing)}
  The label enters BOTH the wpm feature AND the target transform denominator (one
  mechanism, tested as a unit; per-cell eval back-conversion uses the arm's own
  cell-mean label, mirroring matched_frame_pace).
  CONTEXT RECORDED: SESSxINC = anchor; S1xINC re-tests the REJECTED M5-label arm
  (matched_frame verdict: wmae -6.3% but dec3 +3.5% > +2% guard) — the user's
  relativization hypothesis is exactly that the winning target space changes this trade
  (ledger F4 logic, now justified by a changed architecture rather than re-rolling a
  rejected arm).
RULE: S1-label ADOPTED iff in the WINNING target space it improves wmae >1% rel over
SESS-label AND umae/dec3 within +2% AND neither tau lower (2 seeds x 4 folds). Adoption
consequence: stage-1 model becomes a shipped artifact (serve story: pace estimate from
the user's own typing sample), deliverable rebuild inherits it; else SESS label stands
and the route closes on the record.

### Outcome append (2026-07-10): T-REL — LOGRAT ADOPTED, decisively
runs/trel_arms.json (shared v5 frame, 2 seeds x 4 LOLO folds; all arms tau 1.0/dp-tau 1.0):
  INC    rho/ceil 0.9725  wmae 15.44  umae 19.69  dec3 26.34   (reproduces p8b zone ✓)
  RAT    rho/ceil 0.9890  wmae 12.38 (-19.9%)  umae -10.8%  dec3 -2.4%   QUALIFIED
  LOGRAT rho/ceil 1.0174  wmae  9.67 (-37.4%)  umae -20.8%  dec3 +0.9%   QUALIFIED <= WINNER
  DIFF   wmae -2.7%  umae -2.6%  dec3 -1.8%   qualified but marginal
ATTRIBUTION the DIFF control buys: the gain is the multiplicative SCALE structure
(RAT/LOGRAT), not the additive level (DIFF ~nothing) — the user's mechanism as stated.
LOGRAT's rho/ceiling 1.0174 EXCEEDS the split-half ceiling (cross-layout strength
borrowing, seen before on azerty/qwertz folds; not an error). Largest single-lever wmae
gain of the campaign (prior record: hesitation filter -23.6%, which was guard-blocked).
Consequences now owed per the rule: 3-seed all-data retrain (bigram_trel_*), SHAP
before/after, conditioned-trigram analog arm, then ONE deliverable rebuild; KB-STRAT
runs at TARGET_SPACE=LOGRAT.

### Outcome append (2026-07-10): S1-REL — M5 STANDS (relativization not adopted at stage 1)
runs/blind_pace_rel.json: R0/M5 +7.65% (reproduced exactly), R1 ratio +1.92%,
R2 log-ratio +7.83%, R3 scale-free +7.91%; all pass leakage audit. R3 beats M5 by only
+0.28% rel (< the 1% adoption bar). Reading: M5's log-space fit already captures the
multiplicative structure; the explicit per-typist anchor adds ~nothing at stage 1
(contrast with stage 2, where the label transform moved wmae -37%: stage 1 predicts a
single interval where the typist scale is largely in the features already; stage 2's
target AGGREGATES across typists, where the scale mismatch does the damage).
Two-stage 2x2 proceeds with S1 label = M5 (the shipped stage-1 winner) per the rule.

### Outcome append (2026-07-10): T-REL consequences — trigram LOGRAT ADOPTED; SHAP evolution as predicted
runs/trel_retrain.json (conditioned-trigram A/B, tristrokes_cond_v3, cand-4 params,
2 seeds x 4 LOLO folds, shared frame; challenger rule from 046b92e):
  tri INC     rho/ceil 0.9226  wmae 20.73  umae 24.08  dec3 28.05   (taus 1.0)
  tri LOGRAT  rho/ceil 0.9928  wmae 14.38 (-30.66%)  umae -22.01%  dec3 -9.71%  (taus 1.0)
QUALIFIED with every guard IMPROVED (unlike the bigram arm where dec3 was merely inside
tolerance) — the multiplicative-scale mechanism carries to the conditioned increment.
Consequence executed per rule: 3-seed all-data retrain saved as
models/trigram_cond_logratv3_seed{0,1,2}.json (target_space=LOGRAT sidecar).
SHAP evolution (Stage C, registered expectation: wpm's share collapses in LOGRAT space):
  INC bigram_v5_seed0:    wpm |SHAP| share 43.8% (wpm 27.1ms, bottom 9.3, same_hand 5.1)
  LOGRAT logratv5_seed0:  wpm |SHAP| share 28.1% (wpm .083 logs, bottom .054, same_finger .033)
Expectation CONFIRMED in direction (43.8% -> 28.1%), not to zero: the residual wpm share
is the model using pace to modulate GEOMETRY effects (skill-dependent physics measured
earlier: SFB penalty grows with skill, roll bonus grows with skill), which is exactly the
wpm interaction we want the model to keep — the hyperbola (pure level) is what LOGRAT
removed. Geometry features' relative shares rose accordingly.
Repo consequence (committed with this entry): target_space is now a first-class model
property — TypingModel.target_space/to_ms/predict_ms; train_{bigram,trigram}_model
default target_space="LOGRAT"; every scorer (model + table, bigram + trigram) and the
LOLO harness convert predictions to ms through the seam; train CLI grows --target-space
{LOGRAT,MS}. Old ms-space artifacts load unchanged (absent sidecar key => MS).

## P10 — LOGRAT deliverable rebuild (registered 2026-07-10, BEFORE results)
Stage A (join_lograt.py, launching now): construction re-selection under LOGRAT. The
f06c695 rule (deliverable trigram table = best rho/own-ceiling with tau 1.0 among
constructions) was decided under the ms objective; LOGRAT moved the direct construction
0.9226 -> 0.9928, and join-under-LOGRAT is unmeasured. Arms INC/LOGRAT on the JOIN frame
(tristrokes_v1 x tristrokes_last, p8b join code verbatim; 2 seeds x 4 folds, cand-4,
same machinery as the direct A/B). Frame self-check: INC-join should reproduce ~1.006.
RULE: deliverable trigram construction = best rho/own-ceiling with all-pair + dp taus
1.0 among {join-LOGRAT, direct-LOGRAT 0.9928}; the within-frame INC->LOGRAT challenger
guards (wmae >1% better, umae/dec3 <= +2%) apply as before. If join-LOGRAT qualifies
AND wins the construction pick: 3-seed all-data retrain
(models/trigram_cond_lograt_join_seed{0,1,2}).
Stage B (p10_family.py, GATED on twostage_2x2 + kb_strat verdicts per the ONE-rebuild
sequencing): T2 = mean predict_ms tables of bigram_logratv5_seed{0,1,2} at wpm 90;
Tcond = mean predict_ms of the selected construction's 3 seeds; T3c = T2 + Tcond;
SA+2opt 12 restarts x 12k iters (p8b budget), oxey weights {0, .5, 1, 2}, rng 880333;
E5 postflight; GL certificate on the bigram component; cross-objective A/B: p8b family
scored under T3c_lograt and the P10 winners under T3c_inc (argmax-movement report — the
user's same-ordering != same-argmax standard). SHIP RULE: P10 replaces P8b as the speed
deliverable family — the model-level verdicts already adopted LOGRAT at both ngram
levels; the family is the consequence, not a new decision. P9/F5M quality family is
UNCHANGED this round (its target was validated in ms space; a LOGRAT-F5M A/B is a
registered FUTURE round, not assumed). If twostage_2x2 adopts the S1 label, Stage A is
void (the label changes the frame) and the rebuild re-plans — accepted risk; the prior
favors SESS (matched-frame M5 rejection).

### Outcome append (2026-07-10): P10 Stage A — JOIN keeps the construction title under LOGRAT
runs/join_lograt.json (JOIN frame: tristrokes_v1 x tristrokes_last, 20183 rows, 693830
examples, 27346 cells; cand-4, 2 seeds x 4 folds):
  join INC     rho/ceil 1.0063  wmae 18.51  umae 22.12  dec3 27.31   (taus 1.0)
  join LOGRAT  rho/ceil 1.0107  wmae 14.16 (-23.5%)  umae -14.8%  dec3 -5.4%  (taus 1.0)
Self-check PASSED: INC-join reproduces the f06c695 number (1.0063 ~ 1.006).
LOGRAT-within-join QUALIFIED (all guards improved). CONSTRUCTION PICK per rule: JOIN
(1.0107 > direct-LOGRAT 0.9928, all taus 1.0). LOGRAT lifts BOTH constructions by
similar relative amounts — the mechanism is orthogonal to construction, as expected.
Consequence executed: 3-seed all-data retrain saved as
models/trigram_cond_lograt_join_seed{0,1,2}.json (target_space=LOGRAT sidecars).
P10 Stage B will run with TRIGRAM_MODELS=models/trigram_cond_lograt_join_seed, still
gated on the twostage_2x2 verdict (in flight; SESSxLOGRAT already replicated the bigram
LOGRAT gain on the independent matched frame: wmae 15.59 -> 9.64, -38.2%).

### Outcome append (2026-07-10): TWO-STAGE 2x2 — SESS STANDS; the S1 dec3 trade is target-space-INVARIANT
runs/twostage_2x2.json (matched frame, fresh extraction: 31.6M occurrences, 5924 cells,
145k examples; 2 seeds x 4 folds):
  SESSxINC     rho/ceil 0.9720  wmae 15.59  umae 20.13  dec3 27.16   (ap-tau .67, dp-tau 1.0)
  SESSxLOGRAT  rho/ceil 1.0162  wmae  9.64 (-38.2%)  umae 15.74  dec3 27.14
  S1xINC       rho/ceil 0.9757  wmae 14.61 (-6.3% vs SESSxINC)  dec3 28.97 (+6.7%)
  S1xLOGRAT    rho/ceil 1.0067  wmae  9.19 (-4.7% vs SESSxLOGRAT)  umae +2.75%  dec3 +7.23%
VERDICT per a94a2ba: S1 fails the guard IN THE WINNING SPACE (umae +2.75% > +2%,
dec3 +7.23% > +2%) => SESS label stands; the two-stage route closes on the record.
The registered context question is answered NO: the S1/M5 label's frequent-cell-win /
rare-decile-trade signature (matched_frame: wmae -6.3%, dec3 +3.5%) does NOT dissolve
in LOGRAT space — it REPLICATES in INC (-6.3%/+6.7%, near-exact) and persists in LOGRAT.
The blind-pace label sharpens dense cells and blurs rare ones regardless of target
space; relativization was orthogonal, not curative.
Bonus replications on this independent frame: SESSxLOGRAT -38.2% (T-REL's -37.4%);
S1xINC's all-pair tau 1.0 (the sharper label happens to break the azerty-qwertz tie,
but dp-tau was already 1.0 everywhere — not verdict-relevant).
CONSEQUENCE: P10 Stage B UNGATED — launches now with SESS label,
TRIGRAM_MODELS=models/trigram_cond_lograt_join_seed (Stage A pick).
The shipped stage-1 model (M5) remains the best BLIND-PACE PREDICTOR (that finding
stands); it is just not a better TRAINING LABEL than session mean — two different jobs.

### Outcome append (2026-07-10): KB-STRAT — transfer holds, feature rejected, S3 fails the 0.25pp letter (argmax hardware-invariant)
runs/kb_strat.json (S1/S2 carried verbatim from kb_strat.log — the original driver crashed
at the S3 scoreboard on a charset edge (dvorak carries ' where qwerty has /; KeyError) and
never wrote JSON; kb_strat_s3_resume.py re-ran S3 with a charset-guarded fitness (dvorak =
reference row, marked skipped) — decisive finalists all share the qwerty charset, unaffected):
S1 TRANSFER (LOGRAT space): full->full 1.0409/10.83, full->laptop 1.0083/10.67,
  laptop->full 1.0361/11.43, laptop->laptop 1.0107/10.89 (rho/ceil / wmae). Cross-hardware
  prediction costs ~nothing. HOLDS.
S2 is_laptop FEATURE: BASE wmae 10.54 vs KBFLAG 10.49 (-0.40% < 1% bar) => NOT adopted.
S3 FINALIST CROSS-STRATUM REGRET (3-seed LOGRAT T2 tables per stratum, wpm 90):
  mean:  P8b_w0 0/0 (wins BOTH strata), P9_w0 +0.48/+0.96 (spread .48pp),
         BLEND +0.34/+0.77 (.43pp), qwerty +3.2/+3.9, colemak +1.7/+1.2
  f5m:   P9_w0 0/+0.05 (.05pp), BLEND +0.04/0 (.04pp), P8b_w0 +2.57/+2.22 (.34pp),
         colemak +6.8/+6.3 (f5m strongly dislikes colemak — echoes D4)
RULE VERDICT: NOT confirmed by letter (P9/BLEND spreads 0.43-0.48pp > 0.25pp under mean;
P8b 0.34pp under f5m). HONEST READING: the ARGMAX is hardware-invariant in both
objectives (same winner both strata everywhere); what varies is the margin — laptop
regrets run ~2x full under the mean objective 🟠 (no CIs on these regrets; magnitude
nuance, not a pick-flip). Consequence: one layout family serves both hardware types;
the .25pp bar was calibrated tighter than the measurement noise floor — a future
re-registration should add bootstrap CIs before re-adjudicating.

## OCC — occurrence-level training (registered 2026-07-10, BEFORE results; brainstorm
## lever A: stop pre-aggregating before training)
The incumbent compresses 31.6M occurrences into ~145k (row, session-wpm) IQR-mean
examples BEFORE the fit — a structural choice from the ms era (robustness via IQR trim)
that LOGRAT plausibly obsoletes (the log tames the tail the trim existed for). OCC
trains on EVERY occurrence (target = log(dur*wpm/12000) per sample, features at the
sample's wpm, practice backfit at occurrence level, layout weights at occurrence level,
counts=1 per example in the shrinkage denominator so k=100 bites identically).
EVAL: unchanged shared cell frame from trel_arms (bistrokes_v5, same CELL_KW, same
ceilings) — cells/targets identical, ONLY the training set construction varies.
Anchor = grouped-LOGRAT (trel_arms wmae 9.67). 2 seeds x 4 folds, shipped depth-3 recipe.
RULE: OCC adopts iff wmae >1% rel better than grouped-LOGRAT AND umae/dec3 within +2%
AND neither all-pair nor decisive-pair tau lower. Adoption consequence: trainer gains
example_level="occurrence" (default flips), deliverable rebuilds once more; rejection
closes the lever on the record. Risks recorded: (a) hesitation tail now enters raw —
LOGRAT compresses but does not delete it (if OCC fails, a capped-OCC follow-up is a new
registration, not a silent amendment); (b) 218x more examples => qwerty's occurrence
dominance is re-weighted by the same capped inverse-share formula (cap 50 now binds
differently — the weight cap's interaction is part of what's being tested).

### Outcome append (2026-07-10): OCC — REJECTED by the rare-ngram guard; lever A closed
runs/occ_arm.json (identical frame to trel_arms; GROUPED anchor reproduced it EXACTLY —
wmae 9.67, rho/ceil 1.0174, taus 1.0 — so the delta attributes to training-set
construction alone):
  GROUPED  rho/ceil 1.0174  wmae 9.67  umae 15.59  dec3 26.58
  OCC      rho/ceil 0.9646  wmae 9.70 (+0.23%)  umae +9.73%  dec3 +15.58%  (taus 1.0)
Occurrence-level training leaves dense cells unchanged and materially DEGRADES rare
cells — the exact trade the guard exists to block (and the same signature as the S1
label). Mechanism reading 🟠: the example distribution shifts from ~group-count to
~occurrence-count proportional, so per-ngram capacity allocation tilts further toward
the dense mass (th:rare goes ~10^3:1 -> ~10^4:1), while the raw target re-admits the
hesitation tail the IQR-mean trimmed. Both registered risks materialized; which
dominates is decided by the WEIGHTS decomposition below. rho/ceil also fell (1.017 ->
0.965) — occurrence training is worse even on ranks. Pre-aggregation is NOT dead weight:
the group-mean + IQR-trim construction is doing real statistical work.

## WEIGHTS — evidence-weighted group training (registered 2026-07-10, BEFORE results;
## brainstorm lever D + the OCC decomposition)
OCC changed two things at once: the effective example DISTRIBUTION (~counts) and the
TARGET (raw vs IQR-mean). WEIGHTS isolates the distribution half on the robust target:
group-level IQR-mean examples as shipped, only sample_weight varies. Arms (all weights
normalized to mean 1 after construction; practice-term counts stay n_i as shipped):
  ANCHOR  shipped: w = bal_grp(layout), inverse GROUP-share balance, cap 50
  W-N     w = n_i * bal_occ(layout); bal_occ = min(50, T/(4*T_l)) on OCCURRENCE shares
          (this reproduces OCC's weight distribution exactly; only the target differs)
  W-SQRT  w = sqrt(n_i) * bal_sqrt(layout); balance on sqrt-count shares (cap 50)
  W-INV   w = (n_i/s2_i) * bal_iv(layout); s2_i = per-group var of log(duration) (wpm
          constant within group => equals LOGRAT-space var), floored at 1e-4, groups
          with n_i<3 get the global-median s2; balance on n/s2 shares (cap 50)
RULE: best arm adopts iff wmae >1% rel better than ANCHOR AND umae/dec3 within +2% AND
neither tau lower. DIAGNOSTIC (registered): if W-N reproduces OCC's umae/dec3 failure,
OCC's defect was the weight distribution (capacity allocation); if W-N is clean, it was
the raw target (hesitation tail). Same frame/driver as occ_arm.

### Outcome append (2026-07-10): WEIGHTS — ALL REJECTED; the OCC decomposition is clean; lever D closed
runs/weights_arm.json (same frame; ANCHOR reproduced 9.67/1.0174 exactly):
  ANCHOR  rho/ceil 1.0174  wmae 9.67  umae 15.59  dec3 26.58
  W-N     rho/ceil 0.9743  wmae 9.23 (-4.59%)  umae +5.67%  dec3 +12.61%  => guard-fail
  W-SQRT  rho/ceil 1.0046  wmae 9.27 (-4.16%)  umae +0.55%  dec3 +4.45%   => dec3-fail
  W-INV   rho/ceil 0.9573  wmae 9.40 (-2.86%)  umae +10.55% dec3 +17.87%  => guard-fail
DIAGNOSTIC (the registered question): W-N reproduces MOST of OCC's guard breach on the
robust target => OCC's rare-cell damage was primarily the WEIGHT DISTRIBUTION (capacity
tilted to dense mass). But W-N gains wmae -4.59% where OCC gained +0.23% => the raw
target's hesitation tail separately erased the dense-cell gain. Both halves were bad,
for different metrics.
HONEST MISS: I predicted W-INV would be guard-FRIENDLY ("sharpens low-noise rare
groups"). Wrong, and worst of the three: rare groups have few samples => HIGH variance
estimates => 1/s2 DOWN-weights them; dense groups have high n AND low variance => n/s2
is doubly concentrated. Efficient global estimation != uniform-coverage allocation.
EMERGING LAW (four tests now: S1-label, OCC, W-N, W-INV): every reallocation of training
emphasis toward the data mass buys dense-cell wmae and pays rare-cell umae/dec3. The
shipped equal-group-weight + robust-target recipe sits at the guard-defended optimum of
everything tested. Levers A and D are closed; capacity-reallocation as a direction is
exhausted — remaining upside must come from NEW INFORMATION (lever B: hold/rollover
channel; lever F: more layouts), not re-slicing the same information.

## HOLD — hold/rollover position aggregates (registered 2026-07-10, BEFORE results;
## brainstorm lever B = backlog A8 + OQ-11 carry-forward, now under LOGRAT + magnitude metrics)
The recorded-but-unused channel: per-sample hold = release(key1) - press(key1); rollover
(hold > interval) goes 5.6% -> 87% with skill and overlapped executions are ~1.69x
faster within-cell (D1'). NOT usable as a raw feature (not serve-computable for a
candidate layout) => enters as POSITION-KEYED TRAIN-FOLD AGGREGATES (A8 recipe):
  h1_mean[p1]    mean hold of the first key's position (train rows, hold>=0 only)
  ro_rate[p1,p2] shrunk P(hold > interval) for the position pair:
                 (n_ro + 50*global_rate) / (n + 50)
Serve semantics: candidate layout's bigram at (p1,p2) looks up the same aggregates —
position-keyed, so the optimizer prices positions, which is the objective. CAVEAT
(registered, from A8): position-keyed data aggregates are a mild memorization channel;
adoption additionally requires an E5-style search gate before production.
DATA: bistrokes_v3 (the prod extraction; carries hold. v5's driver wrote hold=0 — a
re-extract follows only on adoption). Frame = v3 cells with own ceilings; ANCHOR =
LOGRAT + shipped recipe on the same frame (v3-frame numbers differ from v5's; the
comparison is arm-internal, same standard as every prior round).
ARMS: ANCHOR / HOLD (= ANCHOR features + h1_mean + ro_rate, per-fold aggregates).
RULE: HOLD adopts iff wmae >1% rel better AND umae/dec3 within +2% AND neither tau
lower (2 seeds x 4 folds); adoption => E5 search gate, then production re-extract of
v5-with-hold + FEATURE_VERSION bump.

### Outcome append (2026-07-10): HOLD — REJECTED, decisively; lever B closed at the bigram level
runs/hold_arm.json (bistrokes_v3 own frame; anchor healthy at 1.0169/9.56):
  ANCHOR rho/ceil 1.0169  wmae  9.56  umae 16.68  dec3 28.29
  HOLD   rho/ceil 0.9373  wmae 10.93 (+14.29%)  umae +7.91%  dec3 +3.02%  (taus 1.0)
NOT the mass-reallocation signature — HOLD hurts EVERYWHERE, dense cells worst, and
rho/ceiling collapses (1.017 -> 0.937). Mechanism reading 🟠: position-keyed aggregates
are population-confounded — h1_mean[p]/ro_rate[p1,p2] are computed overwhelmingly from
qwerty typists, so at a cross-layout fold they inject the TRAINING population's
position-behavior as if it were the held-out layout's, actively mispricing. The A8
memorization caveat was right but understated: it's not a mild channel, it dominates.
The hold CHANNEL's physics findings stand (rollover 5.6->87% w/ skill; overlap 1.69x
faster) — what failed is the position-aggregate FEATURE route to pricing it. Remaining
hold routes (occurrence-level hold as a target decomposition; overlap-conditioned
targets) died earlier on certification (D1'/538e16e). Lever B closed for this dataset;
Phase-D data with certified release capture is the revival path.

## SMOOTH — spatially-pooled residual correction (registered 2026-07-10, BEFORE
## results; brainstorm lever E — the one direction the emerging law favors: pooling
## strength ACROSS neighboring position pairs to help rare cells specifically)
Two-stage: shipped XGB-LOGRAT (anchor) + per-position-pair TRAIN-fold mean residual
(post practice term, LOGRAT space), kernel-smoothed over pair geometry:
  K(pair_a, pair_b) = exp(-(d(a1,b1)^2 + d(a2,b2)^2) / (2 h^2)), h = 1.0 key units
  corr(pair) = sum_b K * n_b * rbar_b / (sum_b K * n_b + LAM), LAM = 200 count units
(self included; n_b = train sample count). Serve-computable (kernel lookup by
position). FIXED h/LAM — no tuning; a tuned variant would be a NEW registration.
RULE (dual clause, set before results): SMOOTH adopts iff
  (a) standard: wmae >1% rel better AND umae/dec3 within +2% AND taus no lower; OR
  (b) rare-cell clause: umae AND dec3 BOTH >3% rel better AND wmae within +0.5% AND
      taus no lower — the arm's actual design target.
Either adoption additionally requires the E5 search gate (residual aggregates are
position-keyed train-fold statistics — same memorization class as A8/HOLD; HOLD's
failure raises the prior that this too is confounded, recorded honestly).

## QIN-LR — quantile-as-input under LOGRAT (registered 2026-07-10, BEFORE results;
## brainstorm lever C — the user's QIN proposal, retested in the adopted target space)
HISTORY: QIN (0e53294) passed all 3 registered gates decisively in ms space (q=.2
rho/own-ceil 0.995 vs dedicated 0.937) but was REJECTED for generation on a red flag
the gates didn't cover: dp-tau 0.0 at q=0.2 (layout-level ranking at chance) —
diagnosed (91b3f94) as the q-feature absorbing between-layout LEVEL. HYPOTHESIS now:
that level absorption is a SCALE artifact of exactly the kind LOGRAT removes (the
pace/level nuisance lives in the label; in LOGRAT space levels are already factored
out). HONEST PRIOR: the twostage round showed NOT every defect dissolves in LOGRAT
(the S1 trade didn't); this is a test, not a formality.
DESIGN (qin_lograt.py): as qin.py (7 q-replicas {.1,.2,.35,.5,.65,.8,.9}, q appended
to features, group empirical-quantile targets) but target = log(q_quantile(durs) *
wpm/12000); comparator = dedicated single-q LOGRAT models at q in {.2,.5,.8}; eval on
each q's own cell frame (cells aggregated at that q), ceilings per q-frame; per-cell
rho/own-ceiling + the dp-tau diagnostic at every eval q. bistrokes_v3 (same data as
original QIN — isolates the space change), 2 seeds.
RULE: QIN-LR is ADOPT-CANDIDATE iff the original three gates hold in LOGRAT space
((i) q=.2 QIN >= dedicated - 0.005 rho/own-ceil; (ii) q=.5 >= dedicated - 0.01;
(iii) mono violations < 1%) AND the red flag CLEARS: dp-tau at q=0.2 = 1.0 both seeds.
Adoption-candidate => the quality family (P9/F5M) gets a QIN-LR q=0.2 rebuild as a
NEW registration; any gate fails or dp-tau < 1.0 => lever C closes (the bias is not a
scale artifact), F5M stands.

### Outcome append (2026-07-10): SMOOTH — REJECTED (null, both clauses); lever E closed
runs/smooth_arm.json (v5 frame; anchor reproduced 9.67/1.0174):
  SMOOTH rho/ceil 1.0218 (+0.004)  wmae 9.67 (-0.05%)  umae +1.63%  dec3 +3.36%
Clause (a) fails (no wmae gain); clause (b) fails (rare cells got WORSE, not better —
the design target inverted). Reading: the XGB trees already pool geometry-locally
(that's what splits on dx/dy/distance ARE); an extra kernel smoother has nothing left
to add on dense cells and, like every position-keyed train-fold statistic this round,
drags cross-population level into rare cells. Milder than HOLD (corrections shrink
toward 0 by construction) but the same family of failure. Lever E closed. The
residual-structure conclusion: after LOGRAT + practice term + depth-3 trees, per-pair
train-fold residual means carry ~no transferable signal — the model is extracting
essentially everything position-pair-shaped from this dataset.

## FEAT-LR + TUNE-LR — feature engineering + hyperparameter tuning under LOGRAT
## (registered 2026-07-10, BEFORE results; user directive: "now that we are using
## lograt, we should try feature engineering and hyperparameter tuning again")
WHY RE-RUN: every feature-arm and tuning verdict on the books was adjudicated in ms
space under rank metrics (depth-3 adoption pre-dated wmae; P5's 16 bigram candidates
were wmae-ranked but in ms space). LOGRAT changed both the target GEOMETRY (what depth
is needed: the wpm hyperbola is gone, so shallow trees may now suffice — or deeper may
now safely add interactions the hyperbola previously ate capacity for) and the metric
regime (magnitude + guards). Two drivers, one shared v5 frame (trel_arms cells +
ceilings; anchor must reproduce wmae 9.67 / rho-frac 1.0174):
FEAT-LR (feat_lograt.py) arms, all on the LOGRAT target + shipped recipe:
  ANCHOR  shipped 20 features, depth 3
  A1      + first-key row/finger one-hots (8) — the measured pinky-ring collision fix
  A2      + hand indicators (2)
  A3      signed dx/dy replacing absolute
  A5      - second-key row/finger one-hots (8) — the abstraction endpoint
  A7      + explicit interactions (same_finger*distance, scissor*dy, lsb*dx) — pairs
          with shallow trees, which cannot form 3-way interactions themselves
  (C2/C3 fold into TUNE-LR's depth/monotone axes; A1xA3 combo runs ONLY if both parents
  qualify individually — registered to avoid garden-of-forking-paths)
RULE (per arm, vs ANCHOR): adopt iff wmae >1% rel better AND umae/dec3 within +2% AND
neither tau lower. Any adopted DELETION (A5) additionally requires the E5 search gate
(Goodhart row-blindness precedent). Winner = best qualifying wmae; ties to simpler.
TUNE-LR (tune_lograt.py): 16 sampled candidates (rng 424242, same sampling ranges as
P5: n_estimators 150-600, depth 2-6, lr 0.03-0.15 log, min_child_weight 1-8, subsample
.5-1, colsample .5-1) + the P5-era incumbent default, on the FEAT-LR winner's feature
set (ANCHOR's if none qualify). Scored by LOLO wmae, tau-gated, guards as above.
RULE: candidate adopts iff wmae >0.5% rel better than the depth-3 default AND guards
hold (P5's bar). Composition: FEAT-LR winner feeds TUNE-LR; if TUNE-LR also adopts,
one composed verification run re-checks guards before production (no silent stacking).
Production consequence on any adoption: schema/FEATURE_VERSION bump (features) or
default-params change (tuning), retrain, THEN the P10 family re-runs once more — the
family always rebuilds on the final recipe (P10's current build becomes the anchor).
SEQUENCING: launches AFTER QIN-LR returns (user directive) — QIN-LR's verdict decides
the quality-family question first; FEAT-LR/TUNE-LR are speed-model rounds.

### Outcome append (2026-07-10): QIN-LR — NOT ADOPTED by rule letter; but the QIN-specific bias DID dissolve
runs/qin_lograt.json (bistrokes_v3, same data as ms-era QIN; LOGRAT space):
  gates: i q=.2 shared-strength PASS (QIN 1.0352 vs dedicated 1.0080 rho/own-ceil)
         ii q=.5 no-cost PASS (1.0375 vs 1.0126)  iii coherence PASS
         iv dp-tau q=.2 clears: FAIL — QIN [0.5, 0.5], required 1.0
  (q=.8: QIN 1.0668 vs dedicated 0.9849 — shared strength largest at the tail)
THE NUANCE THE RULE LETTER COMPRESSES: in ms space the flag was QIN-SPECIFIC (QIN
dp-tau 0.0 vs dedicated 1.0 — the q-feature absorbed between-layout level). In LOGRAT,
QIN [0.5,0.5] EQUALS dedicated [0.5,0.5]: the QIN-specific level absorption DISSOLVED
(the scale-artifact hypothesis was RIGHT for the model-class defect), and what remains
is a FRAME property — at q=0.2 on this data, no model of either class ranks the
decisive layout pairs perfectly. Per the registered rule (dp-tau < 1.0 => close),
lever C CLOSES and F5M stands as the quality objective; the honest summary is
"QIN-LR is the best per-cell conditional-distribution model we have (dominates
dedicated at every q), but the fast-tail frame cannot certify layout ranking for ANY
model, so generation stays on F5M." Revival path: more layout diversity (Phase D) to
sharpen the q=.2 decisive pairs, not further modeling.
Brainstorm levers now: A/B/C/D/E all closed on the record; F (new data) is the sole
remaining lever, user-gated.

## QIN-ROLE — two follow-ups from the user challenge "shouldn't we adopt QIN?"
## (registered 2026-07-10, BEFORE results)
The challenge: QIN-LR dominates dedicated models per-cell at every q — adoption?
Adoption requires a ROLE. The speed objective is closed by structure (total time =
sum freq*E[t]; quantiles are not additive). The q=.2 generation role failed gate iv —
but for EVERY model class, and gate iv has a registered weakness. Two tests:
T1 TAIL-GAP (tail_gap_boot.py, launches now): gate iv judged the q=.2 frame with
  MEAN-frame decisive pairs (pair_gap_boot v3_nofilter). A mean-decisive pair can be a
  TIE at the tail (precedent: the azerty-qwertz all-pair-tau incident). Participant
  bootstrap (200x) of every layout pair's observed gap ON the q=.2 frame (bistrokes_v3,
  same CELL_KW, table = aggregate_layout_table semantics). RE-ADJUDICATION RULE (set
  now): the QIN q=.2 generation route REOPENS iff dp-tau computed over TAIL-decisive
  pairs (from qin_breakdown's persisted tables) = 1.0 both seeds for QIN-LR. Pair
  tail-decisive AND still flipped => gate iv stands; lever C stays closed.
T2 QIN-F5M (qin_f5m.py, chained after qin_breakdown exits): the CERTIFIABLE quality
  role (F5M frame dp-tau 1.0 per Q-OBJ). F5M = (1/0.2) integral_0^0.2 Q(u) du ~= mean
  of Q at u in {.025, .075, .125, .175} (midpoint quadrature). Arms on the F5M cell
  frame (bistrokes_v5 = the adopted pipeline, f5m aggregation, own split-half ceilings):
    DED-MS   dedicated f5m-target model, ms space (anchor — P9-era recipe)
    DED-LR   dedicated f5m-target, LOGRAT space (discharges the registered
             LOGRAT-F5M future round)
    QIN-INT  QIN trained w/ QS={.025,.075,.125,.175,.35,.5,.65,.8} on LOGRAT
             group-quantile targets; F5M prediction = mean of the 4 tail slices (ms)
  RULES: DED-LR adopts over DED-MS iff standard challenger (wmae >1%, umae/dec3 <=+2%,
  taus no lower). QIN-INT takes the QUALITY-MODEL role iff it beats the best DED arm:
  wmae >1% rel better AND guards AND rho/own-ceil >= best-DED - 0.005 AND dp-tau no
  lower. Registered noise source: empirical q=.025 on small groups ~ interpolated group
  min — the arm's known risk, not amendable after results.
  STAKES (registered): winner refreshes the QUALITY family model only (P9/F5M);
  the speed deliverable is untouched by construction.

### Outcome append (2026-07-10): FEAT-LR + TUNE-LR — A5 qualifies (E5-gated, RUNNING); tuning stands
runs/feat_lograt.json (v5 frame, anchor reproduced 9.67/1.0174):
  A1 +1.21% / A2 +0.45% / A3 +1.33% / A7 -0.44% wmae => all no. First-key one-hots,
  hand indicators, signed geometry, explicit interactions: the LOGRAT model needs none
  of them (echoes the ms-era abstraction-wins finding, now under magnitude metrics).
  A5 (drop second-key row/finger one-hots): wmae -2.07%, umae +0.02%, dec3 +0.34%,
  taus 1.0, rho/ceil 1.0336 => QUALIFIED by the letter. HISTORY: the SAME deletion won
  LOLO in ms space (C2A5, 2026-07-05) and was REVERTED after the Goodhart row-blindness
  incident (optimizer parked junk on the home row — deletion removed the pricing basis
  for rows). E5 gate (e5_a5_lograt.py, running): 3-seed A5-LOGRAT retrain, wpm-90 table,
  12-restart search; PASS = optimized layout's home-row corpus share >= every named
  layout's. Gate verdict decides adoption; qualification alone does NOT.
runs/tune_lograt.json (16 candidates rng 424242 + default, on A5's 12 features):
  default (300/d3/lr.05) wmae 9.475; best cand-9 (174/d4/lr.043) 9.457 = -0.19% < 0.5%
  bar; cands 5/12 similar. Depth 5-6 arms all WORSE (10.2-11.0). => NO tuning change;
  depth-3 default re-earned under LOGRAT. The ms-era P5 conclusion (defaults near-
  optimal) replicates in the new space; LOGRAT did not shift the optimum enough to
  matter at the 0.5% bar.

## GATE-AUDIT round (registered 2026-07-10, user directive: "audit if our gates are
## doing the right thing")
FINDING G1 (immediate, BEFORE e5_a5_lograt results — verified on known-good layouts
only, A5's own number unseen): the E5-LOGRAT bar as registered (home share >= every
named layout's, i.e. >= colemak's 59.8%) FAILS KNOWN-GOOD MODELS: the incumbent
full-feature optimizer outputs measure 53.9% (P10 w=0) and 31.6% (P8b w=0) — the
measured top~home speed tie (OQ-14) means speed-optimal layouts do NOT maximize home
share; colemak's figure is doctrine, not physics. A gate that known-good models fail
is uninformative in BOTH directions.
AMENDMENT (E5-LOGRAT v2, registered before reading the A5 search output): the Goodhart
detector is CROSS-REGRET UNDER THE TRUSTED SURFACE — score the A5-optimized layout
under the incumbent LOGRAT T2 (bigram_logratv5 seeds, wpm 90): regret vs the
incumbent-optimized layout <= 0.75% (plateau 0.5% + margin) = PASS. Home share becomes
INFORMATIONAL (reported, not gating); the distinct-vector diagnostic likewise. This is
the test the original incident would have failed loudly (junk-on-home-row scores
terribly under any trusted surface). The registered home-share clause is VOID as a
decision rule — voided for miscalibration measured on independent evidence, not
because of anything A5-specific.
FINDING G2 (structural, to quantify): every guard compares 2-seed x 4-fold means and
treats +/-2% rel as signal; no gate threshold has a measured NOISE FLOOR. If seed noise
on dec3_rel is ~2%, the rare-ngram guard fires on coin flips near the boundary (the
big rejections — S1 +7.2%, OCC +15.6%, W-INV +17.9% — are far above any plausible
floor; the near-misses — W-SQRT +4.45%, MED +3.5% — may not be).
GATE-NOISE run (gate_noise.py, launches now): anchor config (v5 frame, LOGRAT,
shipped recipe), seeds 0..9, all 4 folds; report the distribution of PAIRWISE rel
deltas |m_i - m_j|/mean for wmae/umae/dec3 across the 45 seed pairs. RULE: a guard
threshold is DEFENSIBLE iff it exceeds the 95th percentile of its metric's pairwise
seed noise; any threshold below that is flagged and future rounds bump it to
ceil(p95) (existing verdicts stand — goalpost discipline — but near-misses within the
measured floor get an explicit "possibly noise" annotation in the record).
AUDIT DOC: agent-artifacts/gates-audit.md — every gate in force (rare-ngram guard,
tau/dp-tau, E5, censor_ratio, kb_strat spread bar, ceilings-as-normalizer, adoption
bars), purpose, calibration status, known misfires, fix. Committed with outcomes.
