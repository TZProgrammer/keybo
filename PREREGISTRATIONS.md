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
