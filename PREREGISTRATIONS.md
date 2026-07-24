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

### Outcome append (2026-07-10): E5-LOGRAT on A5 — BLOCKED by the corrected gate; row-blindness is space-invariant
Sequence (all timestamps in events.log): the v1 home-share gate was VOIDED for
miscalibration BEFORE the A5 search output was read (5d4228e — known-good incumbent
outputs measure 53.9%/31.6% vs the 59.8% bar; OQ-14's top~home tie means speed-optimal
does not maximize home share). The v2 cross-regret gate then ran per the amendment:
  A5-optimized layout nzcdgxaihelwstb,oukyrmfpvq./;j scored under the INCUMBENT
  LOGRAT T2: regret +0.815% vs the incumbent optimum (bar 0.75%; qwerty +4.21% for
  scale) => FAIL. runs/e5_a5_v2.json; runs/e5_a5_lograt.json (v1 informational:
  home share 31.0%, distinct vectors 961->155 vs full set's 765).
VERDICT: A5 adoption BLOCKED. The LOLO gain (-2.07% wmae) is real ON REAL LAYOUTS but
the deletion still hands the optimizer a null space (155 distinct vectors = 5x more
collisions); its optimum drifts +0.815% off the trusted surface — beyond plateau
noise. Row-blindness is confirmed SPACE-INVARIANT (ms-era incident reproduced under
LOGRAT, now with a calibrated detector). ANCHOR features stand; FEAT-LR round closes
with zero adoptions; the shipped 20-feature depth-3 LOGRAT recipe survives its full
re-adjudication (features x tuning) in the new space.
NOTE ON THE NEAR-MISS: +0.815% vs 0.75% is within plausible search noise of the bar;
per goalpost discipline the letter stands (FAIL), and the margin note is recorded so a
future round with more restarts can re-test — as a NEW registration, not a re-read.

### Outcome append (2026-07-10): T1 TAIL-GAP — the user's red flag CONFIRMED at the frame level
runs/tail_gap_boot.json (200x participant bootstrap ON the q=0.2 frame) vs the mean
frame's pair verdicts (pair_gap_boot v3_nofilter):
  pair            mean-frame verdict        tail-frame verdict
  azerty-dvorak   DECISIVE (az -13.7ms)     TIE   (+1.7, CI [-8.3,+7.4])
  azerty-qwerty   DECISIVE (az  -9.2ms)     DECISIVE (az +6.9 — SIGN FLIPPED)
  azerty-qwertz   TIE                       TIE
  dvorak-qwerty   TIE                       DECISIVE (dv +5.2 — qwerty faster)
  dvorak-qwertz   DECISIVE (qz +13.2ms)     TIE   (-1.0, CI [-5.6,+11.0])
  qwerty-qwertz   DECISIVE (qz  +8.7ms)     DECISIVE (qw -6.1 — SIGN FLIPPED)
THE TAIL IS A DIFFERENT WORLD: qwerty is FASTEST at q=0.2 (92.9ms vs dvorak 98.1,
qwertz 99.1, azerty 99.8) while mid-pack on means; two mean-decisive pairs are tail
TIES; two pairs flip sign outright. Gate iv therefore judged tail predictions on a
pair set where HALF the pairs are tail coin-flips — dp-tau 0.5 was uninterpretable
by construction (exactly audit finding G-C: decisive sets are frame-specific).
RE-ADJUDICATION (rule 14f929a, unchanged): QIN's q=.2 generation route reopens iff
dp-tau over the TAIL-decisive pairs {azerty-qwerty, dvorak-qwerty, qwerty-qwertz}
= 1.0 both seeds, computed from qin_breakdown's persisted tables (run in flight).
CAVEAT registered with the tail frame itself: per-cell empirical q=0.2 with n as low
as 10 carries small-sample quantile bias, and layer n differs ~1000x across layouts
(qwerty deepest); the bootstrap CIs capture variance, not this bias. The qwerty-fastest
-at-tail finding is 🟠 pending a bias probe (matched-n subsample) — registered as a
follow-up, NOT blocking the pair re-adjudication (bias affects LEVELS; the decisive
pairs' SIGNS at these magnitudes would need implausible bias to flip).

### Outcome append (2026-07-10): GATE-NOISE — three thresholds defensible; tune bar inside noise
runs/gate_noise.json (10 seeds x 4 folds, anchor config; pairwise rel deltas):
  wmae  p95 0.91% (max 1.29%) | umae p95 0.38% | dec3 p95 0.73% | rho_frac p95 0.26%
  wmae 1% adopt bar: DEFENSIBLE (barely — 1.0% vs 0.91%; margin thin, keep 2-seed
    verdicts honest by preferring bigger effects)
  umae/dec3 2% guards: DEFENSIBLE (5x / 2.7x above floor). Near-miss annotations:
    W-SQRT dec3 +4.45% and MED +3.5% are 6x/4.8x the p95 floor => NOT noise; those
    rejections stand on the merits. (The annotation-on-near-miss clause turns out to
    be unneeded for past verdicts — nothing rejected was within the floor.)
  tune 0.5% bar: INSIDE NOISE (0.5% < 0.91%) => FLAGGED. Consequence per the rule:
    future tuning rounds use ceil(p95) = 1% adopt bar. Retro-check: TUNE-LR's winner
    was -0.19% — inside noise under EITHER bar, so the no-change verdict is UNAFFECTED
    (and P5's ms-era cand-4 trigram adoption was -3.4%, well above any floor).
Gate-audit doc updated with the measured floors. Bottom line of the audit: one gate
voided+fixed (E5), one pair-set bug found+fixed at the tail frame (gate iv / G-C),
one threshold flagged (tune bar), everything else measured DEFENSIBLE — and no past
verdict flips from calibration alone.

### Outcome append (2026-07-10): P10 Stage B — THE LOGRAT DELIVERABLE FAMILY; the argmax MOVED
runs/p10_family.json + runs/p10_reverse_ab.json (T2 = bigram_logratv5 x3, Tcond =
trigram_cond_lograt_join x3, wpm 90, rng 880333, 12 restarts x 12k + 2opt):
  w=0    cgldk.yuo,srthmpnieaxqwbvfj/;z  +3.87% vs qwerty  sfb 1.14%  home 53.9%
  w=0.5  clgmk.,ouysrthdpnaeiqxwbvfz/;j  +3.91%(*)         sfb 0.74%  home 55.0%
  w=1    bnldk.yuo,srthmgcieaxqjfvpwz;/  +3.79%            sfb 0.78%
  w=2    hrfkv.y,oulnstdgciaezxbmqwpj;/  +3.68%            sfb 0.59%  inroll 8.5%
  (*) w=0.5 scoring 0.038% better on SPEED than w=0 = search noise; the family
  plateau is ~0.2% wide, consistent with every prior round.
  Bigram-component certificate: within 3.35% of optimal (GL bound).
  Scoreboard under the LOGRAT objective: P10 +3.87% > p8b +2.74% > colemak +2.09%.
CROSS-OBJECTIVE A/B (the user's same-ordering != same-argmax standard, both directions):
  p8b-w0 regret under the LOGRAT objective: +1.18%
  P10-w0 regret under the ms-era objective:  +0.62%
  Shared positions: 5/30.
The argmax GENUINELY MOVED — both regrets exceed the ~0.2% plateau width. P10's
asymmetric advantage (its layout loses only 0.62% on the old surface; the old layout
loses 1.18% on the new one) is what adopting the better-validated objective buys.
STRUCTURE of the new family: consonant home core srth|nie-a (left-right split), vowels
upper-right, c/l/g/d top-left — a rolls-forward arrangement the wmae-blind era priced
differently. P10 REPLACES p8b as the speed deliverable family per the ship rule.
NOTE vs qwerty margin (+3.87% vs p8b-era +2.23%): numbers are NOT comparable across
objectives (different tables/scales); the cross-objective regrets above are the
apples-to-apples statement.

## PINKY-GAP probe (registered 2026-07-10, BEFORE results; user challenge: "pinky->ring
## and middle->ring produce the same vector — a large gap we should fix")
WHY THE COLLISION EXISTS (mechanism, for the record): the schema encodes the LANDING
key fully (row+finger one-hots) but the origin key only through relational/geometric
features. For same-row neighbors those are symmetric: dy=0 gates angle/inwards/
outwards to 0, dx is unsigned (and stagger cancels within a row), distance/adjacent
equal => pinky->ring and middle->ring into the same key are byte-identical. Deliberate
abstraction (hand-mirroring, unsigned geometry pools data for transfer) reaching one
config too far. A1 (first-key one-hots) failed the corpus-weighted LOLO rule twice
(ms era; FEAT-LR +1.21% wmae) — but LOLO is blind to the fix's value BY CONSTRUCTION:
collision configs are rare on real layouts, while the OPTIMIZER prices them freely
(same null-space logic as E5). So the decisive question is the DATA, not LOLO:
IS there a first-finger timing signal the schema cannot see?
PROBE (pinky_gap_probe.py): qwerty matched pairs differing ONLY in first-key finger
(vector identity asserted programmatically per pair): into-ring dx=1 pinky-vs-middle
(as/ds, qw/ew, zx/cx, po/io, ;l/kl, /.-,.); into-middle dx=2 pinky-vs-index (qe/te,
ad/gd, zc/bc, pi/yi, ;k/hk); into-middle dx=1 ring-vs-index (sd/fd, we/re, xc/vc,
oi/ui, lk/jk). Per (pair, wpm-bucket 60-80/80-100) with BOTH members n>=50: IQR-mean
each; practice control = per-bucket OLS of cell time on log10(total qwerty samples)
over ALL qwerty same-hand same-row non-same-finger bigrams (n>=50);
  d_adj = [t_outerfirst - t_innerfirst] - c1*[log10 n_outer - log10 n_inner].
RULE: the gap is REAL iff count-weighted mean d_adj >= +5ms AND >65% of qualifying
cells have d_adj > 0 (weight = min pair count). REAL => preregister the FIX arm
(first-key finger one-hots + signed same-row column-order term) under a DUAL rule:
LOLO wmae within +1% (non-degradation; p95 noise 0.91%) + guards + E5-v2 cross-regret
<= 0.75% + the fix model must price the probe pairs with the measured sign. NULL =>
the collision is documented HARMLESS (the abstraction is correct: origin finger does
not matter for into-key time) and the feature stays out.

### Trigram collision census (2026-07-10, user question: "make sure trigrams don't
### conflict either") — CLEAN
All 31^3 = 29,791 ordered position triples featurized (46-col trigram row, wpm 90):
28,006 distinct vectors; 1,785 collision classes, every one of size EXACTLY 2, and
every one a pure LEFT-RIGHT HAND MIRROR (x -> -x on all three keys); zero classes
survive mirror-quotienting. Contrast the bigram world: 961 -> 765 with 184 classes,
some NOT mirror-explained (the pinky->ring/middle->ring same-row family — under
active probe). Why trigrams are cleaner: the row carries BOTH constituent bigrams'
placement features + trigram-level sg_* features, so an origin ambiguity in bg1's
relational encoding is usually broken by bg2's landing one-hots and the skipgram
geometry; the same-row degeneracy needs MORE symmetry to survive one level up, and
only the exact mirror provides it. Hand-mirroring is the schema's DELIBERATE pooling
assumption (symmetric hands), shared with the bigram model and load-bearing for data
efficiency; the mirror-asymmetry question (most typists are right-handed) was tested
as A2 (hand indicators) and rejected twice — ms era and FEAT-LR (+0.45% wmae). So:
no trigram analogue of the pinky gap exists; the trigram feature map is injective up
to the intended symmetry.

### Outcome append (2026-07-10): PINKY-GAP — REAL, decisively; the user found a genuine schema hole
runs/pinky_gap_probe.json (qwerty matched pairs, vector identity asserted per pair —
11/16 byte-identical, 5 excluded honestly; practice controlled at -45/-33 ms per
log10(count) by bucket):
  as/ds  +31.2 / +26.9 ms   po/io  +48.0 / +43.6 ms   (pinky-first vs middle-first)
  we/re  +22.3 / +22.3 ms   oi/ui   +8.6 /  +2.0 ms   (ring-first vs index-first)
  count-weighted mean d_adj +27.4ms, 8/8 cells positive => RULE FIRES (>=5ms & >65%).
The origin-finger effect is LARGE — same order as the SFB penalty (+27-38ms) — and
graded by finger (pinky-first worst, ring-vs-index smaller), exactly the biomechanic
ordering. The model prices these pairs IDENTICALLY by construction; the optimizer has
been placing bigrams into pinky-first patterns it cannot price. NOTE the raw gaps are
sometimes NEGATIVE (as/ds raw -14ms) — practice masks the physics; only the matched-
pair + practice-control design exposes it. Caveat: 8 cells from 4 pair families,
one layout (qwerty), one hand each mostly; graded consistency + magnitude make 🟡 HIGH.
FIX ARM (rule 0f77e65, now active): add first-key finger one-hots + signed same-row
column-order term; DUAL rule — LOLO wmae within +1% (non-degradation) + umae/dec3
guards + E5-v2 cross-regret <= 0.75% + fixed model prices the probe pairs with the
measured sign (pinky-first slower). FEATURE_VERSION bump + retrain + family re-run
on adoption.

## TRI-FEAT — triple_roll + back_forth trigram features (registered 2026-07-10, BEFORE
## results; user proposals: "roll with 3 consecutive fingers, same hand, no redirect"
## and "third key == first key, middle on a different finger")
REPRESENTABILITY AUDIT (measured, runs in the record): neither is a collision — the
schema DISTINGUISHES both patterns (in-roll vs out-roll triple differ via landing
one-hots; a-b-a is exactly sg_distance==0 & !bg1_same_finger). The question is
UNDERFITTING, not blindness: back_forth is a 2-split conjunction (trivially formable);
triple_roll is a 4-way conjunction (same_hand_tri & !redirect & bg1_adjacent &
bg2_adjacent) — formable on one depth-5 path (trigram prod uses cand-4 depth 5) but
capacity-expensive; an explicit column is a shortcut. Precedent AGAINST: A7 explicit
interactions failed for bigrams. Untested at trigram level.
DRIVER (trifeat_arm.py), JOIN cond frame (canonical), cand-4, 2 seeds x 4 folds:
Stage 0 DIAGNOSTIC (free, from ANCHOR's held-out predictions): mean signed residual
  (obs - pred, ms) by pattern class {triple_roll_in, triple_roll_out, back_forth,
  redirect-nonroll, other-same-hand, alternating} — does the INCUMBENT already price
  these classes? |mean resid| <~ 3ms => already priced.
Arms: ANCHOR / +TRIPLE (triple_roll_in, triple_roll_out — signed, 2 cols) / +BF
  (back_forth, 1 col) / +BOTH.
RULE (per arm vs ANCHOR, trigram challenger standard): adopt iff wmae >1% rel better
AND umae/dec3 <= +2% AND taus no lower. Stage-0 large-residual on a class + its arm
qualifying => productionize (TRIGRAM_FEATURE_NAMES + FEATURE_VERSION bump + family
re-run, composed with whatever pinkyfix decides). All-null + Stage-0 small => both
features documented as already-priced; census note extended.

### Outcome append (2026-07-10): PINKY-FIX — Stages A+B PASS, Stage C FAILS 0/8; the
### free-fit route is UNIDENTIFIED, not the physics wrong
runs/pinkyfix_arm.json: LOLO non-degradation PASS (wmae +0.44% <= 1%); E5-v2 PASS
(+0.179% cross-regret — the new columns are harmless to the optimizer); sign agreement
FAIL 0/8 — the fitted g prices outer-first FASTER by 5-11ms, the OPPOSITE of the
measured +27ms.
MECHANISM (diagnosed from the practice terms, runs/pinkyfix_arm.json + sidecars):
time = g(geometry) + b(ngram) is NEARLY UNIDENTIFIED for identity-correlated geometry
within one layout: on qwerty (98.7% of data) "first finger of this position pair" is a
function of BIGRAM IDENTITY, so the per-ngram b and the new finger columns compete for
the same variance. The backfit let b keep the physics (b_fix(as)-b_fix(ds) = -0.061,
nearly the incumbent's -0.093 = the raw practice+physics blend), leaving g's new
columns to fit noise — sign inverted. The probe could see the physics only because its
design (matched pairs + GLOBAL practice curve) constrains the decomposition; the free
per-ngram b does not. The cross-layout rows that would identify it are 1.3% of data.
CONSEQUENCE: the feature-column route via free fitting is CLOSED (Stage C is exactly
the check that caught it — the dual rule worked). The physics finding STANDS (probe
🟡 HIGH); what failed is attribution, and it needs a CONSTRAINED estimator:
## PINKY-CAL (registered now, BEFORE results): calibrated-offset route
Instead of learnable columns, inject the finger effect as a FIXED offset measured by
the probe's constrained design, at the pipeline level:
  t_adj = t_raw setting: target' = y_arm - delta(first_finger_class) applied at
  TRAINING; serve adds delta back per candidate-layout position pair. deltas (LOGRAT
  units, from the probe's per-class means at the band midpoints): pinky-first-into-
  adjacent +0.19, ring-first-into-adjacent(vs index) +0.15, else 0 — computed from
  d_adj/typical-ms; exact values recomputed in-driver from runs/pinky_gap_probe.json
  and recorded.
RULE: PINKY-CAL adopts iff LOLO wmae within +1% AND guards AND E5-v2 <= 0.75% AND the
served surface prices the probe pairs with the measured sign (>=6/8 — by construction
it should be 8/8; the check verifies the plumbing). Else the pinky physics is
documented as measured-but-not-installable pending more cross-layout data (Phase D),
and the collision note stands with the sign-inversion caveat.

### Outcome append (2026-07-10): QIN-BREAKDOWN — the user's wmae table; QIN's real profile is SHARPER than "just better"
runs/qin_breakdown.json (diagnostic re-run, per-cell predictions persisted):
wmae (ms) by quality x layout, QIN vs dedicated (rel delta):
  q=0.2: azerty +10.6%, dvorak -12.9%, qwerty +21.1%, qwertz +22.3% (overall +21.1%)
  q=0.5: azerty -1.2%, dvorak -5.9%, qwerty -8.4%, qwertz +3.3% (overall -8.3%)
  q=0.8: azerty +14.9%, dvorak -2.1%, qwerty +18.3%, qwertz +15.3% (overall +18.2%)
READING: QIN dominates on RANKS everywhere (rho/own-ceil 1.035/1.038/1.067) and on
MAGNITUDES at the median (-8.3%), but its TAIL MAGNITUDES are 18-21% WORSE — the
shared-q model compresses extreme-quantile LEVELS toward the body (classic pooling
bias). Exception: dvorak (n=64, scarcest data) where shared strength wins even at the
tails — pooling helps where data is scarce, biases where it is plentiful. wpm profile:
QIN's q=.2 deficit concentrates at low wpm (40-bucket +26%) and vanishes at 120.
CONSEQUENCE for the pending decisions: (a) T2 (QIN-F5M) is now sharper — QIN-INT
integrates the exact tail slices where QIN's levels are biased; if it still beats the
dedicated F5M models, shared-strength ranks outweigh level bias, else the bias story
holds. (b) The tail re-adjudication (T1 rule) proceeds as registered, but a reopened
generation route would additionally face this magnitude deficit at any adoption test
(noted now, before the re-adjudication result).
RE-ADJUDICATION MECHANICS: qin_breakdown persisted per-cell predictions but NOT
per-ngram tables (needed for the common-ngram layout scores); a targeted rerun
(qin_tail_readj.py, q=0.2 eval only, both model kinds, per-pair agreement vs the
TAIL-frame observed gaps over tail-decisive pairs) launches now.

### Outcome append (2026-07-10): PINKY-CAL — ADOPTED, all three stages
runs/pinky_cal.json (deltas from the probe: pinky_first +42.1ms, ring_first +21.3ms;
1923 calibrated examples of 140537):
  Stage A LOLO: wmae -0.05%, umae +0.03%, dec3 -0.22%, taus 1.0 => PASS (non-degrading;
    the offset is nearly invisible to corpus-weighted metrics, as expected — the
    calibrated classes are rare on real layouts)
  Stage B E5-v2: +0.130% cross-regret => PASS (the CAL optimum stays near-optimal on
    the trusted surface; the offset steers placement without distorting the rest)
  Stage C served-sign: 8/8 — as/ds, po/io, we/re, oi/ui all priced outer-first slower
    at both wpms (e.g. as 158.9 vs ds 124.6 @90)
CONSEQUENCE (executing now): productionize as a first-class calibration seam
(keybo/training/calibration.py: finger_class + delta in LOGRAT units; train subtracts
for bigram LOGRAT models, sidecar records it, position-aware consumers add it back),
FEATURE-level version unchanged (features untouched) but CALIBRATION_VERSION recorded;
retrain production bigram models; P10 family re-runs (P11) with the calibrated T2.
SCOPE NOTE (registered): the calibration applies to the BIGRAM surface only. The
conditioned-trigram increment shares the origin-finger blindness for key2->key3, but
the probe measured isolated bigram intervals; extrapolating the deltas to the
conditioned increment is a NEW measurement question (registered as a follow-up, not
assumed). T3c inherits the fix through T2.

### Outcome append (2026-07-10): T2 QIN-F5M — DED-LR takes the quality role (LOGRAT-F5M
### round discharged, big win); QIN-INT REJECTED for this role (level bias confirmed)
runs/qin_f5m.json (F5M frame, own ceilings; 2 seeds x 4 folds):
  DED-MS   rho/ceil 1.0442  wmae 27.57  umae 30.10  dec3 33.09  (P9-era anchor)
  DED-LR   rho/ceil 1.0555  wmae 17.17 (-37.7%)  umae 23.70  dec3 28.81  taus 1.0
    => ADOPTED as the quality model: the LOGRAT lever transfers to the F5M target
    fully intact (-37.7% ~= the mean-target -37.4%). The P9/F5M quality family's
    model is now DED-LR; a quality-family rebuild (P9-LR) is licensed.
  QIN-INT  rho/ceil 1.1455(!) but wmae +27.0% vs DED-LR, umae +13.4%, dec3 +8.4%,
    all-pair tau 0.33, dp-tau 0.0 => REJECTED decisively for the quality-MODEL role.
READING: exactly the breakdown's prediction — QIN's tail LEVELS are pooling-biased
(the quadrature integrates q=.025-.175, the most-biased slices), and at the F5M frame
that bias is layout-correlated enough to break even the ranking (dp-tau 0.0). The
per-cell rho/ceil 1.146 being the HIGHEST ever recorded while wmae/tau fail is the
sharpest demonstration in the campaign that rank-metrics-only selection would have
shipped a broken model (the user's magnitude standard, vindicated again).
QIN's ledger closes: best-in-class per-cell conditional-distribution model (kept as a
modeling result), NOT usable for speed (structure), tail generation (pending
qin_tail_readj), or quality-model (this).

### Outcome append (2026-07-10): qin_tail_readj — route stays CLOSED; the miss is
### MODEL-CLASS-INDEPENDENT (and shared with dedicated), plus TRI-FEAT null
runs/qin_tail_readj.json: over the TAIL-decisive pairs, QIN agrees on dvorak-qwerty
and qwerty-qwertz but MISSES azerty-qwerty — and the dedicated model misses the SAME
pair identically (both seeds, both kinds: predicted azerty < qwerty; tail-observed
azerty > qwerty by +6.9ms). dp-tau 2/3, rule requires 3/3 => the q=0.2 generation
route stays closed. READING: with the corrected pair set the failure is no longer
uninterpretable — it is a REAL, SHARED miss: no trained model reproduces qwerty's
observed tail advantage over azerty (plausibly practice-at-the-tail, which g cannot
carry cross-layout by design). Fair gate, honest fail, both model classes equal =>
adopting QIN would still buy nothing for tail generation. F5M remains the sole
certified quality frame.
### TRI-FEAT outcome: all arms NULL (+TRIPLE -0.17%, +BF +0.19%, +BOTH -0.06% wmae —
all inside the 0.91% noise floor); Stage-0 diagnostic: back_forth residual +4.2ms ~=
the global +4.4ms mean (already priced); triple_roll_in +14.3ms / out +8.3ms residuals
EXIST (n=182/174 cells — the incumbent does underprice fast triple rolls by ~10ms) but
the classes are too rare (0.6% of cells) to move any corpus-weighted metric, and the
explicit columns did not transfer that residual into held-out gains. Both features
documented: representable, marginally mispriced, adoption unjustified at this data
size. Census note extended: user's triple_roll instinct was HALF right — there IS an
unpriced ~10-14ms effect, but it is corpus-weight-invisible; Phase-D data with more
roll-heavy layouts is the revival path.

## PINKY-FIT (registered 2026-07-10, BEFORE results; user challenge: "hardcoded offset
## smells like a hack — the model should learn this properly")
CONCEDED: the literal DELTA_MS in calibration.py is bad engineering (frozen, non-
updating, doctrine-shaped). NOT conceded: that a free fit can learn the effect — that
was PINKYFIX, sign-inverted 0/8, and the failure is IDENTIFIABILITY, not capacity:
within one layout first-finger-class is a function of ngram identity, so class effect
and per-ngram practice are collinear; any estimator needs an identifying restriction
(the probe's: practice = global smooth curve in log count). Under that restriction a
curve-anchored joint fit converges to the probe estimator — "learn it" and "calibrate
it" coincide; the actionable defect is WHERE the number lives.
CHANGE (PINKY-FIT): calibration deltas are FITTED IN-PIPELINE from the training rows —
fit_first_finger_deltas(rows): generic matched-cell estimator (per layout: same-row
adjacent-finger position pairs, outer-first vs inner-first-control cells per wpm
bucket, practice-adjusted via the layout+bucket log-count slope, min-count floor 50,
count-weighted pooling). train_bigram_model(calibration=True) fits on ITS OWN rows
(leakage-clean per LOLO fold), stores fitted deltas_ms in the sidecar; ALL serve paths
(predict_ms_at, TableBigramScorer) read deltas from the SIDECAR, never module
constants. Insufficient data for a class => that class uncalibrated, recorded.
RULE: PINKY-FIT replaces PINKY-CAL iff (a) full-data fitted deltas are positive for
both classes and within a factor of 2 of the probe's (+42.1/+21.3 — estimator-port
sanity, not a tuning knob); (b) LOLO non-degradation + guards; (c) E5-v2 <= 0.75%;
(d) served-sign >= 6/8. Then P11 builds on the fitted seam. Any failure => report
honestly and hold P11 for a decision (the hardcoded seam is NOT silently kept).

## PACE-2 — the pace-label investigation, reopened with mechanism hypotheses
## (registered 2026-07-10, BEFORE results; user: "investigate more carefully the pace
## label model idea — I believe it can be better than session average")
WHY REOPEN (goalpost discipline — new DESIGN, not a re-roll): three prior rejections
(P-MED, matched-frame M5, twostage S1) all showed the same signature — dense-cell wmae
wins (-4.7 to -6.3%), rare-cell guard breach (dec3 +3.5 to +7.2%). Two NEWLY-IDENTIFIED
mechanisms could produce that breach WITHOUT the label being worse, plus one untested
decomposition:
H1 CONVERSION NOISE (eval plumbing): back-conversion to ms divides by the CELL-MEAN of
  the arm's label. SESS is the bucketing variable (within-cell spread <= 20wpm, mean
  well-constrained); M5 is not (unbounded within-cell spread) => small cells get extra
  multiplicative conversion noise that dense cells don't. Predicts: breach concentrates
  in the smallest-n cells; a SHRUNK conversion label removes it.
H2 POPULATION-CONFOUNDED PRIOR: M5's participant prior shrinks toward the GLOBAL
  median (130ms, qwerty-dominated) => rare-layout typists' labels are biased toward
  qwerty pace => their LOGRAT targets systematically mis-normalized. Predicts: breach
  concentrates on non-qwerty cells; a LAYOUT-aware prior reduces it.
H3 ROLE DECOMPOSITION (never tested): the label enters as FEATURE and as DENOMINATOR;
  prior rounds tested the roles as a unit. Either alone may carry the win without the
  breach.
Also fixed in-construction: prior arms transformed the GROUP-MEAN ms with a GROUP-MEAN
label (approximate — M5 varies within a (row, sess-wpm) group); the new arms use
PER-SAMPLE targets: IQR-mean over samples of log(ms_i * L_i / 12000) (exact).
ARMS (driver pace2_arms.py; shared incumbent-bucketed frame, LOGRAT, 2 seeds x 4 folds):
  ANCHOR      SESS/SESS, shipped construction (must reproduce twostage 9.64)
  ANCHOR-PS   SESS/SESS, per-sample targets (isolates the construction change)
  F-M5        feature=M5, denom=SESS (role H3a)
  S1-PS-RAW   feature+denom=M5, per-sample, raw cell-mean conversion (old plumbing)
  S1-PS-SHR   same + SHRUNK conversion label: (n*mean_cell + 25*mean(layout,bucket))
              / (n+25) — input-side info only, no timing leakage (H1 fix)
  M5L-PS-SHR  M5 with LAYOUT-median prior (H2 fix) + shrunk conversion
RULE: best arm adopts iff wmae >1% rel better than ANCHOR AND umae/dec3 <= +2% AND
neither tau lower. DIAGNOSTIC (registered, per-cell detail persisted): dec3/umae
deltas by cell-size tercile and by layout — H1 predicts smallest-tercile concentration
shrinking RAW->SHR; H2 predicts non-qwerty concentration shrinking M5->M5L. If no arm
qualifies, the route closes WITH mechanism attribution (the investigation deliverable);
adoption => stage-1 model becomes a shipped artifact + deliverable rebuild, as always.
HONEST PRIOR: the stage-1 model is a proven better PACE PREDICTOR (+7.65%); what three
rounds failed to show is that this transfers to a better TRAINING LABEL. These arms
are the first that could show the failures were plumbing, not physics.

### Outcome append (2026-07-10): PACE-2 — a REAL adoption (per-sample targets), H1+H2
### refuted, H3 answered, and the M5 route closes WITH mechanism localization
runs/pace2_arms.json (fresh extraction; ANCHOR reproduced twostage SESSxLOGRAT 9.64 ✓):
  ANCHOR      wmae 9.64  umae 15.74  dec3 27.14
  ANCHOR-PS   wmae 9.49 (-1.61%)  umae +0.09%  dec3 +1.43%  taus equal => QUALIFIED
  F-M5        wmae +25%, tau 0.0(!) — catastrophic
  S1-PS-RAW   wmae 8.72 (-9.58%)  umae +0.73%(clean!)  dec3 +6.73% => dec3-fail
  S1-PS-SHR   wmae -9.44%  dec3 +6.49% => dec3-fail   (H1 fix moved dec3 by 0.24pp)
  M5L-PS-SHR  wmae -8.24%  dec3 +7.43% => dec3-fail   (H2 fix moved it NEGATIVELY)
VERDICTS BY HYPOTHESIS:
  H1 (conversion noise) REFUTED — SHR ~= RAW on every metric; the breach is not
    eval-plumbing. H2 (population prior) REFUTED — layout-median prior does not help,
    and the layout diagnostic shows the damage is ON QWERTY (umae 21.7->22.9), with
    azerty/qwertz IMPROVING (12.6->12.2) — opposite of H2's prediction.
  H3 (role split) ANSWERED — feature and denominator must stay COUPLED: M5-feature
    with SESS-denominator destroys even tau (0.0). The wpm feature's job under LOGRAT
    is indexing the target's own normalization; decoupling them is incoherent.
  CONSTRUCTION FIX ADOPTED — ANCHOR-PS qualifies: per-sample log targets (= robust
    LOG-space aggregation; equals IQR-mean of log(d_i) + log(w/12000), i.e. a trimmed
    geometric mean) beat log-of-IQR-mean by -1.61% wmae with guards held. Consistent
    with the whole LOGRAT story: multiplicative noise => aggregate in log space.
  M5 ROUTE CLOSED, mechanism LOCALIZED: with per-sample construction the label's win
    is bigger than ever measured (-9.6% wmae) and umae is now CLEAN (+0.73%) — the
    residual failure is specifically the rare-FREQUENCY deciles, concentrated on
    qwerty (which owns most rare-ngram cells), NOT small-n cells (tercile-0 flat),
    NOT the prior, NOT conversion. Remaining candidate mechanism (🟠, would need a NEW
    registration): practice-term interaction — rare ngrams get b~0 by shrinkage, so
    their predictions ride on g alone, and g trained on sharper-label targets shifts
    the g/b decomposition against ngrams the backfit cannot reach.
CONSEQUENCE (executing): productionize per-sample LOGRAT aggregation in train.py
(TDD); pinkyfit_gates (in flight, old code in memory) stays valid as the calibration-
isolated verdict on the incumbent construction; then ONE COMPOSED verification
(per-sample + calibration vs the group-mean anchor, v5 frame, standard guards +
E5-v2 + served-sign — the no-silent-stacking rule) gates P11, which rebuilds on the
composed recipe. The old direct pinkyfit->P11 chain is retired in favor of
pinkyfit -> composed_gates -> P11.

## QIN-FIX — repairing QIN-INT's tail-level bias (registered 2026-07-10, BEFORE
## results; user: "is there an experiment we could do to fix QIN-INT tail bias?")
MECHANISM HYPOTHESIS (new, testable): the bias may live in the TARGETS, not the model.
QIN trains on EMPIRICAL group quantiles; for a 10-50-sample group, the empirical
q=0.025 quantile is essentially the interpolated sample minimum, whose expectation sits
far CENTER-ward of the true 2.5th percentile (order statistics: E[min of n=10] ~ the
9th percentile). So the extreme-q training targets are themselves compressed toward
the body — the model faithfully learns a biased target. Predicts: bias worst at
extreme q and small groups; dvorak exception explained (its cells are small for BOTH
kinds, so the dedicated F5M targets share the bias there).
ARMS (qin_fix.py, F5M frame, same machinery/rule as qin_f5m 14f929a):
  DED-LR    anchor (the adopted quality model; must reproduce wmae 17.17)
  QIN-PIN   fixes the CAUSE: proper quantile regression — per-sample pinball loss
            (xgboost reg:quantileerror, quantile_alpha=q per replica) on raw per-sample
            LOGRAT values, no empirical-quantile targets at all; F5M by the same
            4-slice quadrature. Asymptotically unbiased for the true quantile.
  QIN-RECAL fixes the SYMPTOM: QIN-INT as-was + per-q affine recalibration in log
            space, fitted on TRAIN-fold cells only (observed-vs-predicted log cell
            quantiles per tail slice), applied at serve before quadrature.
RULE (unchanged from 14f929a): a QIN arm takes the QUALITY-MODEL role iff it beats
DED-LR: wmae >1% rel better AND umae/dec3 <= +2% AND rho/own-ceil >= DED-LR - 0.005
AND taus no lower. Neither qualifies => QIN's ledger stays closed, now with the
target-bias mechanism confirmed or refuted as the deliverable.

## COMPOSED — the no-silent-stacking verification for P11 (registered 2026-07-10)
Two adoptions landed independently: PINKY-FIT (calibration; LOLO +0.11%) and ANCHOR-PS
(per-sample targets; LOLO -1.61%). P11 ships their COMPOSITION, which must be verified
as a unit (composed_gates.py, v5 frame):
  ANCHOR    group-mean LOGRAT, no calibration (frozen reference construction, manual
            fit — the pre-2026-07-10 recipe; must reproduce wmae 9.67)
  COMPOSED  the production train_bigram_model defaults (per-sample + fitted
            calibration), position-aware serve
RULE: COMPOSED passes iff wmae <= ANCHOR's (improvement expected ~-1.5%) AND umae/dec3
<= +2% AND taus no lower AND E5-v2 cross-regret <= 0.75% AND served-sign >= 6/8.
PASS => 3-seed all-data retrain through the production seam => P11 FINAL builds on
those models (rng 881333 family protocol; the in-flight calibration-only P11 becomes
the ablation reference). FAIL => report the interaction honestly; no build.

## TRI-PS — per-sample trigram targets (registered 2026-07-10, BEFORE results; the
## composition-completeness arm before the final builds)
ANCHOR-PS was adopted on BIGRAM evidence only; the production _group_target now applies
per-sample log aggregation to trigrams too, UNMEASURED at the trigram level. Before any
trigram retrain flows into P11-final, the analog A/B: JOIN cond frame, cand-4, 2 seeds
x 4 folds; ANCHOR = log-of-IQR-mean construction (join-LOGRAT baseline, must reproduce
wmae 14.16); TRI-PS = per-sample construction (the new production default).
RULE: TRI-PS confirms iff wmae <= ANCHOR AND umae/dec3 <= +2% AND taus no lower
(non-degradation + expected small win, mirroring the bigram -1.6%). CONFIRMS =>
trigram retrain through the production path joins P11-final. FAILS => trigram models
pin target_space construction to group-mean (explicit code split) and the failure is
reported — the composition does NOT silently ship.

## ENDGAME SEQUENCE (registered 2026-07-10, the 10-hour plan)
1. In flight: composed_gates (bigram composition), qin_fix, P11-ablation family.
2. TRI-PS (launches now). 3. Brainstorm subagent (spawned) -> any idea implementable
in <=40min with a clean preregisterable rule gets ONE arm each, standard guards; ideas
needing >40min or new data are recorded for the wrap, not run.
4. P11-FINAL (speed family): composed bigram models + best-verified trigram
construction, wpm 90, oxey {0,.5,1,2}, rng 882333, certificate, cross-family A/B vs
P10+P11-ablation, outer-first census. Gated on composed_gates PASS + TRI-PS verdict.
5. P9-LR (quality family): DED-LR (+ pinball model IF qin_fix adopts it) on the F5M
frame at wpm 90, same oxey sweep, rng 882444; cross-checked vs the P9 (ms-era) family.
6. Wrap: layout artifacts for both families' recommended picks, full verdict table,
PREREG outcomes, index, report.

### Outcome append (2026-07-11 ~00:20): COMPOSED gates FAIL by letter; TRI-PS run INVALID (driver bug); brainstorm processed
composed_gates: ANCHOR (frozen group-mean) reproduced 9.67 exactly; COMPOSED (production
per-sample + calibration) wmae +0.26%, umae +0.97%, dec3 +1.46% => FAILS the wmae<=0
letter. NOTE +0.26% is INSIDE the measured wmae noise floor (p95 0.91%) and the guards
are clean — but the expected ~-1.5% per-sample win did NOT appear on the v5 frame.
FRAME-DEPENDENCE HYPOTHESIS (registered): PACE-2's frame was a PLAIN extraction (no
BUF2-BOTH cleaning); per-sample log aggregation is a robustness lever, so it wins big on
dirtier data and ~nothing on v5 where BUF2-BOTH already removed the contamination.
tri_ps: INVALID — driver bug (per-sample target array built with sorted(by_wpm) while
the feature matrix uses insertion order => target-feature misalignment; the +30%/tau-0.33
"result" is the misalignment, not physics). Fixed + relaunched.
BRAINSTORM (subagent report, state/brainstorm-keybo/report.md, 11 ideas): idea #1
variance-corrected objective — INDEPENDENTLY VERIFIED the factual premise myself:
LOGRAT predictions are trimmed geometric means; eval obs is arithmetic; and the decisive
unknown MEASURED: within-cell log-variance DOES correlate with geometry class
(same_finger s2=0.012 vs rolls/alternation 0.034-0.042 => exp(s2/2) spread 1.5%
multiplicative — same order as family decision margins). Idea #1 and #2 get arms; #3
(multi-wpm argmax) folds into the P11-final driver; #4-#11 recorded for the wrap.

## FINAL-NIGHT ARMS (registered 2026-07-11 ~00:25, BEFORE results; all on the v5 frame,
## standard challenger guards, 2 seeds x 4 folds unless noted)
PS-V5 (ps_v5.py): the composition decomposed — ANCHOR (group-mean) vs PS-ONLY
  (per-sample, calibration OFF both arms). DECIDES the production bigram construction:
  PS-ONLY must beat ANCHOR (wmae < 0, guards) to keep per-sample in the recipe; else
  _group_target's LOGRAT branch REVERTS to group-mean (code change + PREREG amendment:
  adopted-on-frame-A, failed-replication-on-frame-B, reverted — the honest record) and
  the ANCHOR-PS adoption is marked frame-specific.
VAR (var_arm.py, brainstorm #1): shipped g + sigma2(geometry,wpm) head (2nd GBM,
  depth 3, fit on per-cell trimmed log-variance from TRAIN rows, shrunk toward its own
  smooth prediction for thin cells); serve T *= exp(sigma2/2). Judged vs ANCHOR on the
  standard frame (obs is arithmetic IQR-mean => the correction should REDUCE systematic
  under-prediction): adopt iff wmae >0.5% better AND umae/dec3 <= +2% AND taus no lower
  AND E5-v2 <= 0.75%. Null (flat sigma2-head => rank-invariant global factor) certifies
  the mean-only objective variance-unbiased — closes the question either way.
B-LETTER (bletter_arm.py, brainstorm #2): practice term shrunk toward letter-additive
  baseline u(a)+u(b) (fit by ridge on ngram residuals) instead of toward 0; arms ANCHOR
  / B-LETTER; adopt iff dec3 OR umae >2% better AND wmae <= +0.5% AND taus no lower
  (the rare-cell design target) OR standard wmae rule. Distinct from rejected R3W:
  LOGRAT space, magnitude-judged, letter-additive form (untested B4).
TRI-PS-FIXED: the repaired tri_ps rerun; rule unchanged (faa5565).
P11-FINAL amendment: fold brainstorm #3 in — build T3c at wpm {70, 90, 110}, search
  each (6 restarts each for the side wpms), report cross-wpm argmax regret matrix; the
  DELIVERABLE family stays wpm-90 (skill-invariance was measured on layout choice, this
  quantifies it on the FINAL objective); plus the standard oxey sweep at 90.
SHIP RULE for the night: P11-final bigram models = best-verified construction +
  calibration + any of VAR/B-LETTER that adopt (composed verification per pair; if a
  composition check fails, ship the largest verified-clean subset, favoring simpler).

### Outcome append (2026-07-11): PS-V5 + TRI-PS — per-sample aggregation is FRAME-SPECIFIC; reverted; composition resolved
ps_v5 (calibration OFF both arms, v5 frame): PS-ONLY wmae +0.38%, umae +0.91%, dec3
+1.52% vs group-mean => REVERT rule fires. The ANCHOR-PS adoption is now marked
FRAME-SPECIFIC on the record: -1.6% on PACE-2's plain extraction, +0.4% (noise) on the
BUF2-BOTH-cleaned v5 frame — the per-sample robustness pays only where the tails are
dirty. Production _group_target reverted to group-mean (ad6d651); the composed-gate
failure is thereby EXPLAINED (its +0.26% was the per-sample half; calibration's own
LOLO was +0.11%, clean). Composition question RESOLVED: P11-final bigram models =
group-mean LOGRAT + fitted calibration = exactly the bigram_cal_seed{0,1,2} already
trained and gate-verified (pinkyfit_gates: LOLO +0.11%, E5-v2 -0.003%, sign 8/8).
tri_ps (fixed driver): TRI-PS wmae -3.88%, umae -2.31%, dec3 -1.09% (all better) BUT
all-pair tau dropped 1.0 -> 0.67 => FAILED by the taus-no-lower letter. NOTE the
dropped pair is the azerty-qwertz DECISIVE-set member? dp-tau stayed 1.0 — the all-pair
drop is on a tie-pair by construction (dp-tau is the calibrated metric); HOWEVER the
registered rule listed "taus no lower" over BOTH metrics, so the letter stands:
trigram construction pins to GROUP-MEAN (matching the reverted bigram default —
consistent recipe, no code change needed post-revert). The -3.9% wmae improvement is
recorded as a near-miss for a future registration w/ a tie-aware tau clause.

## DATA-BRAINSTORM outcomes + RO-MIX arm (registered 2026-07-11 ~00:50, BEFORE results)
The data-exploitation audit (state/data-brainstorm-keybo/report.md, premises measured on
raw files; owner's rollover correction independently verified: rollover 26/39/54% by
band, overlap ~37ms skill-stable, overlap-FRACTION rising, same-key marker exact,
SFdiffkey 7.6% contaminated floor) killed several owner angles cleanly on MEASUREMENT:
KEYCODE cannot distinguish shift sides (all SHIFT=16); scalar hold covariate is
information-free for speed (hold perp interval within-cell, r=+0.039); motor-span /
inter-release targets duplicate press-press at cell level. The genuinely un-mined
modalities: the ROLLOVER REGIME STRUCTURE (below), the discarded error stream (54.6% of
substitutions physically adjacent — 5x random), non-9-10-finger population (+29% data),
language/experience covariates. Errors + fingers + language are recorded as REGISTERED
FUTURE ROUNDS (each needs a re-extraction; > tonight's budget).
RO-MIX (ro_mix.py, tonight's one new-modality arm — the regime-aware release target,
the record's own registered revival path for the overlap channel):
  MECHANISM: press-press time is a 2-regime mixture (sequential vs rollover); the
  incumbent prices the MIXTURE MEAN. A typist-facing objective at wpm 90+ should price
  the mixture at the TARGET skill's regime shares, and P(rollover) is geometry-
  dependent (same-hand rolls 40.9% vs cross-hand 35.4% at matched skill) => the
  incumbent misprices patterns by their regime composition.
  ARM (bigram, v5 frame, standard guards): ANCHOR = shipped. RO = shipped features ->
  TWO extra serve-computable model heads trained on TRAIN rows: p_hat(rollover |
  geometry, wpm) and the per-regime LOGRAT means m_seq/m_ro; serve prediction =
  log-mix: exp-weighted combination at the cell's wpm. SFdiffkey pairs EXCLUDED from
  regime-head training (the 7.6% floor); they fall back to the shipped single-head
  prediction. Adopt iff wmae >0.5% better AND umae/dec3 <= +2% AND taus no lower AND
  E5-v2 <= 0.75% before any P11 use.
  NULL teaches: the mixture mean is already sufficient (regime composition either
  geometry-flat or self-averaging) — closes the release channel entirely, with the
  error stream as the dataset's remaining un-mined vein.

### Outcome append (2026-07-11): VAR — REJECTED; the mean-only objective is certified,
### with a sharper reading than the null we registered
runs/var_arm.json: VAR wmae +5.76%, umae +2.02% (breach), dec3 -0.70%, taus unchanged.
NOT the registered null (flat sigma2-head => rank-invariant): the head found REAL
geometry-dependent variance (s2_hat spread 0.123 across the grid — consistent with the
probe's class spread) and applying exp(s2/2) made ms predictions WORSE against the
arithmetic-mean obs. READING: the trimmed geometric mean plus the practice/calibration
stack is apparently already calibrated to the IQR-trimmed arithmetic obs (IQR trimming
itself removes most of the tail mass that separates the two means); adding the full
log-normal correction OVER-corrects — the objective's "geometric mean" is closer to a
trimmed arithmetic mean than the raw math suggested. The brainstorm's factual premise
stands (the gap exists in raw moments); its practical consequence does not survive
trimming. Objective-form question CLOSED: mean-only, trimmed, LOGRAT — certified.

### Outcome append (2026-07-11): RO-MIX — REJECTED decisively; the release channel closes
runs/ro_mix.json (v3 frame): RO wmae +34%, umae +25%, dec3 +17%, rho/ceil 1.017->0.994,
all-pair tau 0.67. The regime-mixture serve (p_hat-weighted per-regime heads) is far
worse than pricing the mixture mean directly. READING: the two per-regime heads are
each trained on a REGIME-SELECTED subsample whose selection is skill-correlated within
cells (who rolls over at a given wpm is a typist-quality signal), so m_seq/m_ro inherit
a selection bias that the mixture reconstruction amplifies — while the incumbent's
mixture-mean target integrates the selection out by construction. Combined with the
earlier verdicts (hold-scalar dead, motor-span duplicate, travel-target regime-broken,
overlap-target certification-failed, hold aggregates population-confounded), this
completes the release-clock audit: SIX routes into the release channel measured, zero
survive. The press-clock mixture mean is the right object on this dataset. The
dataset's remaining un-mined veins are the ERROR STREAM, FINGERS population, and
language covariates (registered future rounds, each needs re-extraction).

### Outcome append (2026-07-11): B-LETTER — REJECTED (null); the bigram stack for P11-FINAL is settled
runs/bletter_arm.json: B-LETTER wmae +0.34%, umae +0.40%, dec3 +1.20% — neither the
rare-cell clause nor the standard rule fires; all deltas inside/near the noise floor.
Letter-additive practice inheritance does not help rare cells on this frame; the
practice-term x rare-ngram interaction (open mystery #1) is NOT resolved by a smarter
shrinkage TARGET — the mystery narrows to the LABEL-side interaction (only sharper
labels trigger it), consistent with PACE-2's localization. Final tally of tonight's
improvement arms: VAR no, RO-MIX no, B-LETTER no, TRI-PS no (tau letter), per-sample
REVERTED, brainstorm #3 folded into P11-final. ZERO adoptions => P11-FINAL builds on
exactly the gate-verified stack: bigram = bigram_cal_{0,1,2} (group-mean LOGRAT +
fitted first-finger calibration; pinkyfit_gates LOLO +0.11%/E5 -0.003%/sign 8/8),
trigram = trigram_cond_lograt_join_{0,1,2} (group-mean construction). LAUNCHING NOW
per the registered protocol (multi-wpm argmax + oxey sweep + cert + cross-family A/B).

### Outcome append (2026-07-11): P11-ablation family (calibration-only) — the pinky physics steers placement; speed surface stays flat
runs/p11_final.json (bigram_cal group-mean-era models + join trigram, rng 881333):
  w=0   uoy,.vldfgaeinprhtcs;/jkbmwxzq  +4.02% vs qwerty  sfb 1.24%  outer-first 0.49%
  w=0.5 hcgkm.,ouylrstdpnaiezxwbvfjq;/  +3.95%            sfb 0.98%  outer-first 0.57%
  w=1   gnldk.,yousrthmpcieaqxzbvfwj;/  +3.90%            sfb 0.76%  outer-first 0.48%
  w=2   uoy,.kdlnvaeicpmhtrs;/jwbgfzxq  +3.86%            sfb 0.83%  outer-first 0.44%
  GL certificate 3.40%. Scoreboard: P11 +4.02% > P10 +3.95% > colemak +2.07%.
THE CALIBRATION'S SIGNATURE, measured: the family's outer-first (calibrated-class)
corpus share collapses to 0.44-0.57% vs P10's 1.22% and qwerty's 1.08% — the optimizer
now actively avoids the pinky/ring-initiated same-row rolls it can finally price.
Speed cost of that avoidance ~nil: P10-w0's regret under the calibrated objective is
only +0.072% (inside plateau noise) — the calibration reprices a NARROW pattern class,
so the argmax moves within the plateau (0/30 shared positions is plateau-degeneracy,
many near-equivalent optima — consistent with every rank-stability finding).
NOTE the w=1 member gnldk.,yousrthmpcieaqxzbvfwj;/ pairs sfb 0.76% with outer-first
0.48% at -0.12% speed vs w0 — an unusually strong balanced pick for this family.
This build is the ABLATION REFERENCE; P11-FINAL (rng 882333 + multi-wpm stage) is the
shipping family and is now running.

### Outcome append (2026-07-11): QIN-FIX — both arms fail by rule; the target-bias mechanism is CONFIRMED; QIN's ledger closes with a full causal story
runs/qin_fix.json (F5M frame; DED-LR anchor reproduced 17.17):
  QIN-PIN   wmae 48.68 (+183%) — catastrophic. The pinball route as built is broken
    beyond the hypothesis it tested (likely compounding: practice backfit on pinball
    residuals is ill-defined — quantile residuals aren't mean-centered — plus pooled
    per-sample quantiles across typist mixtures != per-cell tail quantiles). Recorded
    as an implementation-confounded null: it does NOT adjudicate the pinball idea
    cleanly, and tonight's budget does not permit a rebuild.
  QIN-RECAL wmae 8.98 (-47.7%!), umae -37.0%, dec3 -29.2% — the affine per-q
    recalibration RECOVERS the tail levels almost completely => the ORDER-STATISTIC
    TARGET-BIAS MECHANISM IS CONFIRMED (a 2-parameter log-space map per q fixes the
    levels; the bias was systematic compression, exactly as hypothesized). BUT dp-tau
    0.0 / all-pair 0.33: the recalibration is fit on large-n train cells (qwerty-
    dominated), so the map absorbs between-layout level — the SAME layout-confounding
    signature as the original QIN rejection, now moved into the recalibration layer.
  VERDICT: neither takes the quality role; DED-LR stands. QIN's final ledger: best
  per-cell conditional-distribution architecture; tail LEVELS fixable by
  recalibration; layout-level ranking not certifiable on 4 layouts because every
  flexible level-map (q-feature or recalibration) absorbs between-layout differences.
  Phase-D layout diversity is the structural unlock. P9-LR proceeds on DED-LR.

### Outcome append (2026-07-11): P11-FINAL — THE speed deliverable family; skill-invariance confirmed on the final objective
runs/p11F_final.json (bigram_cal + join-LOGRAT trigram, rng 882333):
  w=0   uoy,.vlmdgaeinprhtcs;/jkbfwxzq  +3.99%  sfb 1.36%  outer-first 0.50%
  w=0.5 cgldk.,yousrthmpnieaqxwbvfzj;/  +4.00%  sfb 1.09%  outer-first 0.42%  <= the pick
  w=1   uoy,.kdlnbaeicpmhtrs;/jwqgfvxz  +3.89%  sfb 0.87%
  w=2   uoy,.kdlnvaeicpmhtrs;/jwbgfzxq  +3.86%  sfb 0.83%
  GL certificate 3.41%. Cross-family: P10-w0 regret +0.042% (plateau), outer-first
  1.22% -> 0.50%. The w=0.5 member is speed-TIED with w=0 (+4.00 vs +3.99 = noise)
  at sfb 1.09% and the family's lowest outer-first share — the recommended pick.
MULTI-WPM ARGMAX (brainstorm #3, the registered stage): the wpm-90 champion carries
  +0.057% regret at wpm 70 and -0.010%(!) at wpm 110 — i.e. it is inside the plateau
  at BOTH ends; the per-wpm optima differ by <=0.32% cross-regret. SKILL-INVARIANCE OF
  THE ARGMAX is now confirmed on the FINAL calibrated objective (previously only on
  earlier surfaces): ONE layout family serves 70-110 wpm. The wpm-90 build is not a
  compromise — it is optimal (within noise) across the band.
Two family shapes tie on speed: uoy/aei-left (w=0/1/2) vs cgld/srth-left (w=0.5, the
P10 lineage) — plateau degeneracy at +4%, richness for the report, robustness for the
pick (structurally different layouts, same predicted speed).

### Outcome append (2026-07-11): P9-LR — the QUALITY family final (pure F5M-LR, gate passed)
runs/p9lr_final.json (F5M-LOGRAT bigram + trigram, trigate 0.69 > 0.55 => PURE quality
family, no mixed fallback; rng 882444):
  w=0   wothsineac.blkvyjufqzgmxdrp/,;  +2.88%  sfb 4.58%
  w=0.5 cithsnleak.pgvfwzuojqydbmrx/,;  +2.72%  sfb 1.19%  <= the pick
  w=1   kaedsrntiwjoyfblxpu.;qzgchmv,/  +2.61%  sfb 1.21%
  w=2   cithslneakwygbfzquo.,pdvmrx/;j  +2.49%  sfb 1.06%
  ms-era P9 w0 regret under the LR quality objective: +0.431% — the argmax moved
  modestly with the -37.7% model upgrade (beyond plateau, consistent with the pattern
  that better-calibrated surfaces relocate optima).
FAMILY SIGNATURE preserved from ms-era P9: the quality objective puts the letter core
on the TOP row (wothsineac / cithsnleak) — the fast-tail surface exploits the OQ-14
top~home tie differently than the mean surface (which centers on home). The w=0.5
member is the recommended quality pick: sfb collapses 4.58 -> 1.19% for -0.16% speed.
BOTH DELIVERABLE FAMILIES ARE NOW FINAL: speed = P11-FINAL (fd06e42), quality = P9-LR
(this entry). Campaign wrap follows.

## QIN-LODO (registered 2026-07-11, BEFORE results; user challenge: "recal recovered
## -48% of tail error — should we not try to get that to work? maybe modifications
## pass the 4-layout blocker")
THE BLOCKER, precisely: QIN-RECAL's per-q affine maps were fit on POOLED train cells
(qwerty-dominated), so the maps absorb between-layout level; at serve on a held-out
layout the map imports the training layouts' level => layout ranking breaks (dp-tau 0)
even though per-cell levels are nearly fixed (-48%). Modifications that could evade it:
  RECAL-LODO: fit the per-q recalibration LEAVE-ONE-DECILE-OUT within each layout —
    no, the confound is BETWEEN layouts. Correct form: fit the maps on the train
    layouts but constrain them to be LEVEL-FREE — slope-only in log space (a0 := 0,
    a1 fit): a pure SHAPE correction cannot import a level. If the order-statistic
    bias is mostly a compression (slope) rather than a shift, slope-only recovers most
    of the -48% without the confound.
  RECAL-WPM: additionally let a1 vary smoothly with wpm (2-knot linear), still no
    intercept.
ARM (qin_lodo.py, F5M frame, DED-LR anchor must reproduce 17.17): QIN + slope-only
recal (Q-SLOPE) and QIN + slope(wpm) recal (Q-SLOPEW). RULE unchanged from 14f929a:
takes the quality role iff beats DED-LR on wmae >1% with guards AND rho/own-ceil >=
DED-LR - 0.005 AND taus no lower. Registered risk: if the bias has a large layout-
independent SHIFT component, slope-only under-corrects and the arm nulls — which
would localize the bias decomposition (shift vs compression) as the deliverable.

### Outcome append (2026-07-11): QIN-LODO — slope-only recal FAILS; the bias decomposition lands SHIFT-dominated; QIN's ledger re-closes
runs/qin_lodo.json: Q-SLOPE wmae 23.23 (+35% vs DED-LR 17.17; cf. affine QIN-RECAL's
8.98), Q-SLOPEW worse; dp-tau still 0.0 for both. TWO findings:
(1) The registered decomposition question is ANSWERED: removing the intercept destroys
the -48% recovery => the order-statistic bias is predominantly a LEVEL SHIFT (per-q
intercept), not a compression (slope). (2) The shift is exactly the component that
cannot be fit without importing between-layout level on 4 layouts — AND dp-tau stayed
0.0 even slope-only, meaning the layout confound also lives in the shared q-surface
itself, not only in the recal layer. CONCLUSION (final, with the decomposition as the
deliverable): the -48% is real but irreducibly entangled with layout level at this
diversity; no recalibration form available on 4 layouts can keep it and certify
ranking. DED-LR remains the quality model. Phase-D layout diversity remains the
unlock — now with a precise statement of WHAT it unlocks (per-layout-identifiable
per-q intercepts).

### Informational (2026-07-11): community scoreboard under both final models, wpm 70/90/110
(runs recorded in events.log; per-trigram mean ms, per-layout charset, % vs qwerty)
SPEED model ranking (stable, tau 0.91-1.00 across wpm): P11-w0.5 +4.00 > dvorak +3.39
> semimak +2.54 > graphite +2.49 > colemak +2.07 > colemak-dh +1.71 > workman +1.55
> norman +1.05 > qwerty. Consistent with every prior scoreboard (alternation-heavy
dvorak outranking modern roll-centric layouts is the model's measured signature).
QUALITY model ranking carries a KNOWN CONFOUND for cross-layout use and is NOT a
valid community ordering: qwerty ranks 2nd (+0 baseline) with ALL community layouts
NEGATIVE — the signature of TAIL PRACTICE (tail_gap_boot measured qwerty FASTEST at
q=0.2, 92.9ms vs 98-100; qwerty typists' lifetime practice shows at their best
executions; the practice term removes bigram-identity practice, not layout-level tail
practice). The quality model is licensed for its validated uses — per-cell prediction
and WITHIN-search optimization (a fixed position table cannot leak layout-level
practice into permutation comparisons) — but cross-layout quality RANKING of named
layouts is practice-contaminated at this diversity. Recorded as the honest boundary
of the quality family's claims; Phase-D diversity is (again) the unlock.

### Informational (2026-07-11): dvorak-vs-moderns decomposition (user's recurring red
### flag, now fully attributed on the final model)
Bigram level (calibrated T2, corpus-weighted): dvorak's ENTIRE edge is alternation
share — 80.2% vs semimak 77.4 / graphite 78.9 / colemak 72.1. Per-class PRICES are
nearly layout-independent (alt ~127ms everywhere; rolls 133.6-136.5). Gap attribution:
vs colemak, dvorak saves -10.1ms/bigram on alternation and gives back +12.5 on rolls
+ -1.6 SFB => net ~1.1ms. vs semimak/graphite the bigram level is a TIE (+-0.2ms) —
the moderns' lower SFB fully pays for their roll share at the bigram level.
Trigram level (Tcond) is where dvorak actually wins: -2.1 to -2.4ms/trigram vs all
three moderns, attributed to (a) redirect exposure (colemak +5.8ms contrib, semimak
+2.5) and (b) mixed-flow cost (+1.3 to +3.4) — the moderns' roll-chains put more mass
in one-hand continuations and redirects, which the conditioned increment prices at
+25-40ms over hand-switches. Verdict: NOT an artifact — a coherent two-level story
(alternation share at bigram level ties-or-wins, redirect/continuation exposure at
trigram level decides), robust to the calibration, and the artifact channels were
independently controlled earlier (no-dvorak retrain kept dvorak's rank; population
weighting controlled; tail-practice confound documented as QUALITY-side only).
P11-w0.5 beats dvorak by winning BOTH levels: comparable alternation flow-cost with
better within-class placement (roll price 130.3 vs 133.6 — the calibration + finger
assignment) and a better trigram increment (-0.48ms vs dvorak).
COMMUNITY-DOCTRINE note for the report: our data prices redirects-after-rolls and
one-hand continuations HIGHER than community scorers assume, and inroll/outroll
direction at ~0; if the community ordering (semimak > dvorak) is right about
something, it is a quantity this dataset cannot see (comfort, error rates at speed,
long-run fatigue) — which is a Phase-D question, not a model bug we can fix here.

## FU round — finger utilization / dislocation / multi-analyzer robustness
## (registered 2026-07-11, BEFORE results; user directives x3)
VERIFIED PROBLEM: P11-w0.5 loads R-ring 20.0% (> both indexes), R-pinky 11% > R-middle
9.1%; P9LR R-pinky 0.9%. MECHANISM (named in the gaps audit, never fixed): the
objective prices lag-1 (SFB) + lag-2 (sg_*) reuse and landing costs; lag>=3 reuse and
duty-cycle are INVISIBLE, so the optimizer freely concentrates load on one finger.
FU-1 LAG3-PROBE (lag3_probe.py, launches now): does lag-3 same-finger reuse cost time?
  4-gram extraction (extract_occurrences n=4, time_mode=last => press3->press4
  increment), qwerty-only, matched design: cells where finger(k1)==finger(k4) (k1!=k4,
  and NO closer same-finger collision within the window) vs matched controls (same
  k3->k4 bigram class + row + wpm band, finger(k1) different). Practice-controlled via
  the k3k4-bigram identity match (same landing bigram in both arms). RULE: lag-3
  penalty REAL iff count-weighted mean gap >= +3ms AND >65% of matched cells positive.
  REAL => the utilization term is calibrated PHYSICS; NULL => it ships as an explicit
  documented PREFERENCE (like oxey), never silently.
FU-2 DISLOC scorer (user's heuristic, exact form): per-position cost c(pos) =
  dist(pos, home(finger(pos))) * slowness(finger(pos)); layout penalty =
  sum_letters freq(l)*c(pos_l) (LINEAR in assignment => composes into the QAP
  objective exactly) + optional superlinear spread term sum_f D_f^2. Slowness weights
  MEASURED from our own data: the fitted calibration deltas (pinky +43ms, ring +21ms)
  + T2 landing prices per finger — recorded in the scorer's docstring with provenance.
FU-3 FSPEED scorer (genkey-style): semi's fingerspeed philosophy implemented natively
  (per-finger distance-weighted usage / finger strength weights), as a composable
  IScorer. HONESTY: our implementation is an approximation from documented behavior,
  like OxeyStyleScorer; exact-tool parity (running genkey/keymeow binaries) +
  newer-community-optimizer survey = REGISTERED FOLLOW-UP (needs tool vetting/install).
FU-4 P12 ROBUST FAMILY: search T3c + w_d*DISLOC (w_d in {0, small, med}) at oxey 0.5;
  deliverable = cross-regret matrix of every candidate under ALL scorers (T3c speed,
  F5M quality, oxey, FSPEED, DISLOC) + finger-usage tables; PICK RULE (registered):
  the member minimizing MAX normalized regret across {T3c, oxey, FSPEED} with T3c
  regret <= 0.5% hard cap (speed stays primary; the Q-BLEND robust-pick pattern).

### Outcome append (2026-07-11): FU-1 LAG3-PROBE — NULL, cleanly; utilization is a PREFERENCE, not physics
runs/lag3_probe.json: 2.48M 4-gram windows, 245 matched cells (k3k4-identity-matched
controls); count-weighted mean lag-3 same-finger gap -0.06ms, share positive 52% —
a textbook null (coin-flip direction, zero magnitude). Displaced-finger reuse at lag 3
costs nothing measurable once lag-1/lag-2 collisions are excluded. CONSEQUENCE per the
registered rule: the finger-utilization/dislocation term ships as an EXPLICIT
DOCUMENTED PREFERENCE (oxey-style knob), never as calibrated physics; the finals'
"odd" finger loads are NOT leaving measured speed on the table — they are an
aesthetics/comfort concern, priced accordingly. This also retroactively explains WHY
the optimizer produces them: there is genuinely no time cost in the data to stop it.
P12 proceeds: DISLOC (user's travel-x-slowness form) + FSPEED as preference scorers,
cross-regret pick rule unchanged.

## GK-PARITY — genkey + keymeow exact-tool evaluation (registered 2026-07-11, BEFORE
## results; user directive: "what about genkey and keymeow?" — the registered follow-up
## from the FU round, now executing)
GOAL: score our finalists with semi's ACTUAL tools (not our FSPEED approximation) and
(a) establish where our layouts rank under the community's own metrics, (b) validate/
correct our FSPEED approximation against the real fingerspeed numbers, (c) extend the
P12 cross-regret matrix with exact-tool columns.
METHOD (subagent gk-parity): clone + build github.com/semilin/genkey (Go via brew) and
github.com/semilin/keymeow (Rust via cargo) READ-ONLY LOCAL USE; convert our layouts
(P11-w0.5, P11-w0, P9LR-w0.5, P10-w0.5, P12 picks when ready) + references (semimak,
colemak, dvorak, qwerty, graphite) into each tool's layout format; run each tool's
scoring on its DEFAULT corpus and, where the tool supports a custom corpus, ALSO on
our corpus (both reported — corpus mismatch is a known confound to surface, not hide);
extract fingerspeed/effort/total metrics per layout.
DELIVERABLES: (1) per-tool per-layout metric table; (2) rank correlation of each
tool's ordering vs our speed model's; (3) FSPEED-approximation validation (corr of our
scorer vs genkey fingerspeed across the layout set; if rank-corr < 0.8 our
approximation is flagged and P12's fspeed column is marked unreliable); (4) which of
our finalists the community tools like best (informs the robust pick narrative).
NOT a gate on any adoption — community metrics are PREFERENCES (lag-3 null); this
round is about robustness reporting and approximation validation. Newer-optimizer
survey (anything post-keymeow the community regards well) = included as a best-effort
web-free repo search, honestly bounded.

### Outcome append (2026-07-11): GK-PARITY — exact tools built + run; FSPEED approximation FLAGGED; the measured-time-vs-community-effort divergence is now precisely quantified
runs at state/gk-parity/ (report.md + artifacts/aggregate.json; spot-verified: sfb
orderings match our oxey scorer's within definitional offsets):
(1) FSPEED VALIDATION FAILED per the registered rule: our FingerSpeedScorer tau 0.611
vs genkey fingerspeed (< 0.8) => P12's fspeed column is UNRELIABLE; the pick rule's
fspeed axis is REPLACED by genkey's actual fingerspeed (the harness is built + fast;
P12 post-processing will rescore with it). Our scorer stays as a preference knob with
a documented unreliability note. Genkey/keymeow agree with each other (Pearson .90-.95).
(2) COMMUNITY ORDERING: semimak < graphite < P10-w0.5 < colemak < P11-w0.5 < P11-w0 <
dvorak < P9LR-w0.5 < qwerty (genkey Score; keymeow concurs). Both tools rank semimak +
graphite above ALL our layouts; our P10-w0.5 is the community's favorite of ours (3rd
overall); dvorak — our model's #2 — is near-bottom for the tools (sfb 2.2-2.8%).
(3) THE DIVERGENCE, stated exactly: community aggregates are dominated by SFB/DSFB-
distance terms and reward roll share; our measured objective prices alternation-vs-roll
~neutral-to-alternation-favoring and redirects/one-hand-runs expensive, with SFB
weighted by MEASURED time cost (+43ms) rather than distance-squared-style effort. Same
facts (the tools' sfb/roll/alt percentages match our own pattern census), different
value functions. Which is RIGHT for typing SPEED is exactly what our cross-layout
harness adjudicates and community effort models do not; which is right for COMFORT
our data cannot see (lag-3 null; Phase D).
(4) Corpus confound MINIMAL (keybo corpus is iWeb-derived like shai-iweb; sfb delta
<0.03%). (5) Successor tools noted: oxeylyzer (o-x-e-y), keymui — future parity pass.
CONSEQUENCE for P12: pick rule's axes become {speed, oxey, genkey-fingerspeed(exact)};
the P12 post-processor rescoring with the built genkey harness is the remaining step.

## GK-WEIGHT — genkey score as an in-loop optimization term (registered 2026-07-11,
## BEFORE results; user directive: "include genkey/keymeow weight into the
## optimization, just like oxeylyzer")
DESIGN: GenkeyScorer = an EXACT NATIVE PORT of genkey's Score function (read from
generate.go/layout.go @ f1f4173): Score = 3*sum_f fspeed_w(f) + 1*(100*LSB/total) +
0.3*|idxR-idxL|; fspeed_w(f) = 800/total * sum_{i<=j in finger f} (1.0*B[k_i k_j both
dirs] + 0.5*S[k_i k_j both dirs]) * (staggered_dist(i,j) + 0.02) / KPS[f], default
KPS [1.5,3.6,4.8,5.5,5.5,4.8,3.6,1.5], uniform-column fingering, trigram term disabled
(genkey default). Corpus = ours (measured minimal delta vs shai-iweb).
PARITY GATE (the lesson from the FSPEED flag): the port must match the BUILT genkey
binary on the 9 gk-parity layouts — value ratio within 2% per layout after a single
global corpus-scale factor, rank corr 1.0 — else the port is NOT used and the search
falls back to nothing (report honestly). Approximations without parity checks are how
FSPEED failed; never again.
KEYMEOW: a metrics LIBRARY with no single aggregate — no canonical "keymeow score"
exists to weight. It enters as EVALUATION columns (via the kmrun harness) on the final
family, not as an in-loop term; genkey Score carries the community-effort signal
in-loop (the two tools agree at Pearson .90-.95, so genkey is a faithful proxy).
P13 FAMILY (p13_genkey.py): search T3c (calibrated speed) + w_g*UNIT*GenkeyScore for
w_g in {0, 0.5, 1, 2} + one combined point (w_g=0.5, oxey=0.5); every member evaluated
under: T3c speed regret, genkey Score (BINARY, not the port — the port only drives the
search), keymeow metrics, oxey, quality T3c_q, finger tables. PICK RULE: the member
minimizing genkey Score subject to T3c speed regret <= 0.5% (the community-robust
pick); the pure-speed champion stays the speed pick. Deliverable: the tradeoff curve
speed-regret vs genkey-Score — the measured price of community-doctrine compliance.

## KM-COVER — km_metrics coverage audit (registered 2026-07-11, BEFORE results; user:
## "any metrics in keymeow/km_metrics we don't have but should?")
REPRESENTABILITY (from km_metrics_src/metrics/base.py @ HEAD, 23 metrics): every km
metric is expressible in our schema — most are explicit columns (sfb, sfs=sg_same_
finger, lsb, redirect, same_hand) or <=2-way conjunctions of explicit columns
(alt=ABA-hand via bg same_hand pair; roll via bg1 XOR bg2 same_hand; trill-redir =
redirect & sg_distance==0 = our tested back_forth; miniroll/minialt = bigram-level
explicit). Distance-weighted variants (sfb-dist etc.) are STRICTLY RICHER in ours
(continuous distance x class interactions). Known hole (already on record): same-row
roll DIRECTION (measured ~0 in matched reverses). NOT YET AUDITED for pricing: sft
(3-key same-finger chain), sfs-redir (redirect whose outer keys share a finger — a
"worst redirect" axis DIFFERENT from our bad_redirect=no-index), skipalt/skiproll,
sr-roll.
KM-RESID (km_resid.py): incumbent-residual-by-km-class diagnostic (TRI-FEAT Stage-0
pattern) on the JOIN cond frame, seed 0, 4 folds: mean signed held-out residual for
{sft, sfs-redir, trill-redir, plain-redir, sr-roll, skipalt-proxy, mini3roll, alt,
other}. RULE: a class with |mean residual| > 5ms AND n>=300 cells => one preregistered
feature arm (standard challenger rule) in a follow-up; all classes <5ms => km_metrics
coverage documented COMPLETE (already priced), no arms.

### Outcome append (2026-07-11): KM-RESID — roll-chain classes FLAGGED; km coverage is
### NOT complete; the TRI-FEAT triple-roll signal replicates at higher n
runs/km_resid.json (incumbent residuals by km class, JOIN frame): global level sits
~+3.5-4ms (a uniform offset); ABOVE it: mini3roll +8.38ms (n=485), sr_roll +6.23
(n=911), roll +5.01 (n=6232, marginal). BELOW/at level: sft -1.74 (already priced —
both halves explicit), sfs_redir -1.12, trill_redir +3.70, alt +3.16. The flagged
family is exactly the roll-CHAIN direction the user's triple_roll probe found
(+14.3ms at n=182); km's broader definitions (cross-row included, one-hand-switch
rolls) replicate it at 3-30x the cells. Honest note: relative to the +3.94 global
level the elevations are +4.4/+2.3/+1.1ms — the rule (|mean|>5ms absolute) flags
three; the relative reading says mini3roll is the substantive one.
KM-ARM (km_arm.py, per the registered rule): trigram schema + 3 explicit km columns
(is_mini3roll, is_sr_roll, is_roll — km definitions exactly) on the JOIN frame,
cand-4, standard challenger rule (wmae >1%, guards, taus). TRI-FEAT precedent
recorded: the +TRIPLE explicit-column arm was null because the classes are corpus-
weight-invisible; km's broader classes carry ~10x the corpus mass (roll=6232 cells),
so this arm has the mass TRI-FEAT lacked. Null here => the roll-chain residual is
documented as a real-but-unfixable-by-columns effect (capacity, not blindness) and
closes the km question; qualification => FEATURE_VERSION bump + family re-run.

### Outcome append (2026-07-11): KM-ARM — NULL; km_metrics coverage question CLOSED
runs/km_arm.json: +KM (explicit is_mini3roll/is_sr_roll/is_roll columns, km
definitions) vs ANCHOR on the JOIN frame: wmae -0.83%, umae -0.62%, dec3 -0.37% —
below the 1% challenger bar. Taus/dp-taus unchanged (1.0). Direction is right
(consistent small improvement, no guard trips) but not a qualifying feature.
Interpretation per the registered branch: the roll-chain residual (mini3roll
+8.6ms replicated in this frame's Stage-0) is REAL but is a capacity/target-noise
effect the model cannot cash via indicator columns — the classes are already
~identifiable from existing features (hand-run + direction columns), so explicit
flags add little. km_metrics coverage: COMPLETE — no missing metric qualifies as a
model feature; roll-chain underpricing documented as a known residual structure
(relevant for interpreting per-class errors, not fixable by schema). No
FEATURE_VERSION bump. This closes the "any km metrics we should have?" question:
representability audit (all 23 representable) + pricing audit (sft/sfs-redir priced;
roll-chains flagged) + feature arm (null).

### Outcome append (2026-07-11): GK-WEIGHT / P13 — pick = P10-w0.5; combined search
### CONVERGES to P10-w0.5; genkey compliance costs ~0.1% predicted speed
runs/p13_genkey.json + runs/p13_keymeow.json. Registered pick (min genkey Score s.t.
T3c speed regret <=0.5%): P10-w0.5 (genkey 33.68 keybo-corpus, speed regret 0.099%).
Structure of the frontier: pure genkey-weight arms are DOMINATED — w_g=2.0 gets
genkey 37.4 at 0.12% speed, while the combined (oxey0.5+genkey0.5) member reaches
33.83 at 0.094% AND lands on P10-w0.5's layout up to 3 rare corner keys
(clgmk.,ouysrthdpnaeiqxwbvfjz;/ vs ...vfz/;j). I.e. oxey-style structure is HOW you
get a good genkey score; genkey pressure alone anneals worse. Speed cost of full
community compliance: g0.0 (unconstrained) -> P10-w0.5 is +0.099% predicted time for
-23% genkey Score (43.8->33.7) — cheap. keymeow eval (kmrun, shai-iweb, 0 missing
chars): P13-combined sfb 1.23 / sfs-dist 6.94 ~= P10-w0.5 (1.18/6.97); both beat
graphite on sfs-dist and sit between graphite and semimak overall; P11-w0.5 (speed
headline) stays sfb 1.70. Deliverable implication: P10-w0.5 is the "community-
compliant near-speed-optimal" member — P11-w0.5 keeps the pure-speed headline
(regret 0.06 vs 0.099 on T3c is within noise; both <<0.5% cap), P10-w0.5 is what to
hand a community-metrics-minded user. kmrun layout set extended with the 5 P13
members (gk-parity/kmrun main.rs, rebuilt).

### Outcome append (2026-07-11): FU-4 / P12 — dislocation term FIXES the finger-load
### pathology at ~0.01% speed cost; registered pick = P10-w0.5 (again)
runs/p12_robust.json (fspeed column REPLACED by exact genkey per the GK-PARITY
amendment; pick re-run over {speed, oxey, genkey-exact}, speed cap 0.5%):
PICK = P10-w0.5, max regret 0.04% — it dominates every purpose-built P12 member on
the robust criterion (its oxey/genkey are the column bests while its speed regret is
0.039%). Same verdict as P13: three independent searches (dislocation-weighted,
genkey-weighted, combined) all converge on-or-pick P10-w0.5.
THE UTILIZATION FINDING (user's directive answered mechanistically): the w_d=0 arm
reproduces the pathology (R-ring 20.0%, Rm 9.1 — P11-like structure), and w_d=0.5
FIXES it: Rr 20.0->15.1%, Rm 9.1->14.9, R-pinky 11.0->9.1, at speed regret moving
0.074->0.049% (i.e. FREE within noise). w_d=1.5 adds nothing further (14.35 vs 14.45
disloc regret). The user's travel-x-slowness heuristic is therefore an effective,
near-zero-cost balance knob; 0.5 is the sufficient dose. Note P10-w0.5 achieves the
same balance WITHOUT the term (its oxey-0.5-only search landed balanced) — the
pathology is a P11-family/calibration-era artifact, and either recipe (P10's, or
P11's + w_d 0.5) avoids it. Caveat recorded: oxey normalized-regret %s are inflated
by a near-zero denominator (raw oxey spans negative values); the pick is driven by
max-regret ORDER, unaffected.
FAMILY WRAP: speed headline stays P11-w0.5 (+4.00%); the robust/balanced/community-
compliant deliverable is P10-w0.5 (+3.95%, genkey 33.7, balanced fingers, disloc
374M); quality stays P9LR-w0.5. All three documented in docs/.

## KIAKL-INGEST — community monkeytype data ingestion (registered 2026-07-12, BEFORE
## any model contact; user supplied data/community/raw/*, "make this nice and structured")
DATA: Kiakl form-response zip (8 submitters, ~920k events) + GK single-user files
(duplicates of in-zip content). Monkeytype capture: per-event {key, interval(ms,
press-to-press), correct}, sessions carry {sessionID, layout string, keyboardType,
website}. No release timestamps => hold=-1 forever for this source.
INGESTION RULES (fixed before processing):
1. DEDUP by sessionID across all files (byte-duplicate + subset files confirmed:
   GK standalone == in-zip; VG topic files ⊂ VG main; GK numbered ⊂ each other).
2. USER = form submitter (pid assigned 200001+, disjoint from aalto pids); the
   sessionID is NOT a user (one submitter, many sessions).
3. LAYOUT LABEL = <identified-name-or-custom-slug>@<keyboardType>#<submitter-slug>,
   one label per (layout-string, kbt, submitter) — keeps typist and geometry visible
   to every downstream loader; never silently pooled.
4. WINDOW VALIDITY (bigram (e1,e2), trigram (e1,e2,e3)): every event correct=true;
   every within-window interval in (0, 5000]; windows reset at session boundaries
   and at any correct=false event (the following event's interval is recovery-
   contaminated => it may not START a window's duration either).
5. WPM per session = (n_correct_events/5) / (sum correct intervals / 60000), the
   sample label for all samples in that session (mirrors aalto test-level wpm).
6. OUTPUT in the exact production TSV schema (layout, positions, ngram, freq,
   (wpm,duration,pid,hold)*): bistrokes_community.tsv (dur = press1->press2),
   tristrokes_community.tsv (dur = press1->press3), tristrokes_last_community.tsv
   (dur = press2->press3). positions on ROW_STAGGERED_30 slots via the layout
   string's main-30 extraction; frequency = our corpus table (consistency with
   production loaders). Space included (0,0) as in production.
7. NON-DATA files (screenshot png, corpus txt, empty jsons) recorded and excluded.
WHAT THIS REGISTRATION DOES NOT DO: no model training, no accuracy claims, no
layout-ranking use. Any MODEL use of this data (LODO extension, QIN certification,
practice-term fits, cross-layout validation) gets its own registration with rules
fixed before results. Known confounds recorded now: mostly 1 typist/layout (layout
x typist confounded within-source), ortho/angleMod geometry mismatch vs
ROW_STAGGERED_30 features, tiny volume vs aalto, self-selected enthusiast
population, monkeytype word-mode (no punctuation-heavy text).

### Amendment (2026-07-12): KIAKL-INGEST — the gk_typingdata.zip files are NEW data
### (earlier "duplicate" claim was WRONG; sessionID recheck: 0/136 overlap with the form zip)
Correction on record: only the loose gk_typingdataColemkaDH_ortholinear.json
duplicates in-zip content (181/181 sessions). gk_typingdata.zip holds 136 NEW
sessions (~202k events): typingdata.json = GK on QWERTY rowStagger natural text
(first within-typist layout pair in our data — de-confounds typist vs layout);
typingdata0003.json = colemak-dh ortho PSEUDO-WORDS (random letters, forced
home-row return); typingdata1278.json = colemak-dh ortho dictionary words with
BOOSTED RARE CHARS (per the zip's Files info.md). Amended rules: (a) same pid for
the same submitter across files; (b) non-natural-text sources get a corpus tag in
the layout label (+pseudo, +rareboost) so frequency/practice analyses can exclude
or study them explicitly — never silently pooled with natural text; (c) everything
else per the original registration.

### Amendment-2 (2026-07-12): gk typingdata.json (qwerty) is UNUSABLE — capture
### v1.1.0 masks key identity (key in {0,2,3} category codes, correct always
### false; 2.5k literal 'Backspace' strings are the only real keys). No ngram
### extraction possible; excluded by the wpm>0 rule automatically. The hoped-for
### within-typist qwerty<->colemak-dh pair does NOT materialize. Final ingest:
### 3437 sessions, 684,507 events, 573,564 bigram samples, 12 labels (incl.
### +pseudo/+rareboost corpus-tagged colemak-dh).

## COMM-D — leveraging the community dataset (registered 2026-07-12, BEFORE results;
## user directive: 8h window, "final deliverable leveraging the new data as best it
## can" + "revisit old experiments and assumptions")
POWER BASELINE (from ingest): 4644 bigram cells survive production bucketing across
9 usable labels; rowStagger natural-text labels (geometry-matched to features):
colemak@alite 392, mtgap-variant@davison 233, custom@ddn 214, custom-aa426873@vg 455
= 1294 cells / 4 layouts / 4 typists. Ortho/angleMod: recurva@vg 954, colemak-dh@gk
879 (+pseudo 564, +rareboost 564), custom@castro 388.

D1 HOLD-OUT VALIDATION (the aalto-trained model has never seen ANY of this):
Score each label's cells with the PRODUCTION speed stack (bigram_cal + trigram cond
+ fitted calibration, untouched). Per-label: bucket-centered rho vs that label's
split-half noise ceiling, wmae, wmape, calibration slope. PRIMARY = the 4 rowStagger
natural labels (features match the physical geometry). SECONDARY (reported, not
gating) = ortho/angleMod labels — a geometry-transfer probe, expected weaker.
DECISION RULES (fixed now): per-label PASS = rho/ceiling >= 0.6 AND slope in
[0.6, 1.4]. >=3/4 primary labels PASS => "the model GENERALIZES to community
layouts" enters the deliverable's evidence section (the strongest external-validity
claim this project has ever been able to make). <=1/4 PASS => generalization
FAILURE is the headline finding; deliverable claims get an explicit external-
validity caveat and D3 is MANDATORY (the data must then teach, not just test).
NOISE CAVEAT recorded now: one typist per label => ceiling includes typist
idiosyncrasy; rho/ceiling is the right normalization for exactly that reason.

D2 ASSUMPTION REVISITS (each cheap, read-only on models):
D2a TAIL-PRACTICE: compute the q=0.2 lower-tail gap (tail_gap_boot protocol) for
community typists on their OWN daily-driver layouts vs aalto qwerty typists on
qwerty. Prediction from the practice story: enthusiasts on their own layout show
the SAME fast-tail signature qwerty typists show (it's lifetime-practice, not
qwerty-specific). CONFIRMS => the quality-model cross-layout confound story stands
and community data cannot serve as a clean quality ranking either (their layouts
are their qwerty). REFUTES (community tails NOT fast) => tail practice was
misattributed; reopen the quality-ranking question.
D2b DVORAK/ALTERNATION: on community-observed times (per-cell obs, no model), fit
the bigram-level alternation-vs-roll price per label (alt-class mean vs roll-class
mean at matched wpm buckets, corpus-weighted). aalto measured alt ~127ms vs rolls
~134-137. Community typists CHOSE roll-optimized layouts; if their observed roll
price relative to alternation is materially lower (<= -5ms shift vs aalto's gap),
the alternation preference is population/practice-bound => flag the dvorak-#2 and
alternation-heavy optimizer preference as population-sensitive (informational for
the deliverable; a full re-fit is future work). Small-n guard: report per-label n
and skip labels with <30 cells in a class.
D2c PINKY/RING TRANSFER: the fitted calibration says pinky-first +43ms/ring +21ms
(aalto). Matched-cell contrast (same estimator as PINKY-CAL, no refit) on pooled
community rowStagger natural cells. PASS = same SIGN both classes; magnitude
informational. FAIL => calibration is population-specific => note on P11 family.
D2d GEOMETRY: within-typist where possible (VG: angleMod recurva vs rowStagger
custom; GK: ortho colemak-dh vs ortho qwerty 1-cell — underpowered, report
honestly). Per-geometry wmae of the SAME model = how much accuracy the geometry
mismatch costs. Informational only (no gate) — powers the "should we model
geometry" future question.
D3 TRAINING INTEGRATION (GATED on D1 outcome, runs regardless of pass/fail since
either branch wants it — pass makes it optional-upside, fail makes it mandatory):
Add the 4 rowStagger natural-text community labels to the bigram+trigram training
frames (layout-balanced weights as usual; community pids distinct). Re-run LODO
with community labels as additional folds (LODO-8). ADOPT the retrained stack iff:
(a) every aalto fold non-degrading beyond the documented noise floor (wmae p95
0.91%); (b) community-fold mean wmae improves >1% vs the D1 zero-shot number;
(c) rare-ngram guards hold (umae/dec3 <= +2%). ADOPTED => re-run the P10-family
search (same protocol, rng 880333) on the new stack and report argmax movement;
argmax move > plateau (0.2%) => NEW DELIVERABLE CANDIDATE, else P10-w0.5 stands
with strengthened evidence. NOT ADOPTED => document why; deliverable unchanged.
D4 SYNTHESIS: fold in the independent audit-community subagent's report; update
docs/layout artifacts + this file's outcome appends; final deliverable = whatever
survives, with the community-validation evidence attached either way.

### Outcome append (2026-07-12): COMM-D D1 — zero-shot generalization FAILS the
### registered gate (0/4 primary); harness validated by positive control
runs/comm_d1.json. Positive control (same scoring path, aalto in-sample): dvorak
rho/ceiling 0.81, azerty 0.86, slopes 0.94-0.98 — the harness is sound. Community
labels: best = colemak-dh@ortho#gk frac 0.39 (slope 0.94), colemak@rowStagger#alite
0.34 (slope 0.60); worst = ddn -0.11, mtgap-variant 0.14 (slopes 0.05-0.16 —
predictions nearly uncorrelated with observed at the low end). PRIMARY PASS 0/4
(<= 1/4) => per the registered rule: GENERALIZATION FAILURE is the headline; the
deliverable carries an explicit external-validity caveat; D3 (training integration)
is MANDATORY. Interpretation guardrails recorded WITH the result: (a) every
community label = ONE typist, so the ceiling counts typist idiosyncrasy as
learnable signal the model has never seen — frac penalizes person-transfer, not
only layout-transfer; the aalto control folds pool 100s of typists (idiosyncrasy
averages out). (b) wmape ~20-40% vs in-sample 22-23% — MAGNITUDE error is
comparable; what fails is within-layout cell ORDERING. (c) the failure is
graded by geometry+typist, not uniform (ortho colemak-dh 0.37-0.39 with clean
slopes vs rowStagger customs ~0). Honest headline: the aalto-trained model does
NOT rank a single community typist's cells well zero-shot; whether that is
layout-space transfer failure or single-typist noise is exactly what D3's
LODO-8 disentangles (a fold trained WITH other community typists tests transfer
to a new typist rather than to a new population).

### Outcome append (2026-07-12): COMM-D D2 — assumption revisits; tail-practice
### CHALLENGED, alternation-preference SUPPORTED (typist-varying), pinky transfer
### NOT CONFIRMED (sign flip, but 52% share => noise-dominated)
runs/comm_d2.json. Verdicts per the registered predictions:
D2a TAIL-PRACTICE: community daily-driver q20/median 0.777 vs aalto qwerty 0.742
(dvorak 0.753, azerty 0.735). Community tails are NOT disproportionately fast —
the prediction FAILED. Nuance recorded: the tail-practice story claimed qwerty
typists' lifetime practice shows at their best executions; community enthusiasts
have months-to-years (not decades) on their layouts, so a weaker tail is
CONSISTENT with practice-months scaling, but the strong form ("your daily driver
gives you the fast-tail signature") is refuted at this practice level. The
quality-model cross-layout caveat STAYS (the confound direction is unchanged);
what changes: community data at this volume shows a LESSER tail-practice bias
than feared => community-based quality ranking is less contaminated than the
aalto-qwerty comparison was. Reopen quality-ranking = registered follow-up.
D2b ALTERNATION-VS-ROLL: observed roll-minus-alt price is POSITIVE for 7/7
community labels (+1.0 to +42.9ms) — alternation is faster than rolling even for
typists who CHOSE roll-optimized layouts. The model's alternation preference is
SUPPORTED as population-general, not an aalto artifact; the dvorak-#2 story
strengthens. But the SPREAD (colemak-dh typist +42.9 vs recurva typist +7.7) says
the roll PENALTY size is typist-specific — supporting per-typist calibration as
the integration path, and explaining why a single global model fails D1 ordering.
(aalto reference: qwerty +17.9, dvorak -4.3 — dvorak typists genuinely roll
cheaper than they alternate; community never crosses zero.)
D2c PINKY/RING: matched-cell contrast (registered PASS = same sign): aalto
non-qwerty +0.4ms/53% pos on this coarse outer-vs-inner estimator (NOTE: much
smaller than the +43/+21 fitted deltas — this estimator pools pinky+ring vs
index+middle and same-row-only cells, so it is NOT the calibration's estimator;
it under-measures by design). Community rowStagger -8.0ms/52%, ortho -14.0ms/50%.
Share-positive ~50-53% everywhere => the contrast is NOISE-DOMINATED at community
n; verdict: NOT CONFIRMED, NOT REFUTED (underpowered). The calibration keeps its
aalto evidence; transfer claim stays 🟠 unverified.
D2d GEOMETRY: rho/ceiling ortho 0.34 (n=4 labels) > rowStagger 0.15 (n=4) >
angleMod -0.01 (n=1). Surprising direction (features assume rowStagger) —
confounded with typist volume (GK ortho has 3x the cells of any rowStagger
label). Informational only, as registered.

### Outcome append (2026-07-12): COMM-D D3 — naive merge REJECTED (adopt=False);
### community-rho-doubling signal motivates D3b
runs/comm_d3.json. Merged training degrades EVERY aalto fold far past the floor
(qwerty +36.6%, qwertz +29.5%, azerty +27.2%, dvorak +17.2% wmae) while community
folds improve modestly (mean -2.15% wmae). Registered rule => NOT ADOPTED,
deliverable unchanged by this arm. THE SIGNAL: community-fold held-out rho
roughly DOUBLES with other community typists in training (mtgap 0.131->0.320,
ddn 0.029->0.241, vg 0.210->0.370, colemak 0.259->0.385) — transferable
community structure EXISTS; the naive merge just lets 4 single-typist labels
(layout-balanced to equal weight with aalto layouts) distort the aalto
population fit. D1's failure is therefore at least partly POPULATION/TYPIST
shift, not pure layout-transfer failure.

## COMM-D3b — population-aware integration (registered 2026-07-12, BEFORE results)
Two arms, same LODO-8 harness and adopt rule as D3 (aalto folds within noise
floor 0.91%; community mean wmae improves >1% vs incumbent; rare guards):
ARM-W: community sample weight scaled x0.25 within layout_balance_weights (a
single-typist label should not weigh like a 100s-of-typists layout).
ARM-P: +1 feature column is_community_population (1 for community rows) appended
for training; at aalto serve (and layout search) the column is 0, so aalto-side
predictions can be fully protected while shared structure transfers. Community
folds served with 1. Feature-version bump NOT shipped unless adopted.
Pick between qualifying arms: the one with better community mean wmae. If
neither qualifies: integration CLOSED for this data volume; community data
remains validation-only; deliverable stands with the D1 external-validity
caveat + D2 assumption-audit evidence.

### Outcome append (2026-07-12): COMM-D3b — BOTH arms rejected; integration CLOSED
### at this data volume (community = validation-only)
runs/comm_d3b.json. ARM-W (x0.25 weight): aalto folds still degrade +3.8 to
+19.2% (>> 0.91% floor) for community -1.95% — fails. ARM-P (population column,
served 0 for aalto): aalto folds STILL degrade +8.9 to +20.2% — the indicator
does not isolate the shift (tree splits shared across the column distort the
aalto fit anyway) — and community gain evaporates (-0.13%). Per the registered
rule: INTEGRATION CLOSED for this data volume; community data is VALIDATION-ONLY;
the deliverable stands unchanged, carrying (a) the D1 external-validity caveat
and (b) the D2 assumption-audit evidence (alternation preference confirmed
population-general — the deliverable-relevant assumption SURVIVED its hardest
test to date). The consistent cross-arm signal (community rho doubles with
community data in training; aalto always degrades) pins the mechanism:
single-typist labels teach typist idiosyncrasy, not population physics. The
unlock is MORE TYPISTS PER LAYOUT, not cleverer weighting — quantified target
for the Phase D outreach: multiple submitters on the SAME layout so a typist
random effect is identifiable.

## COMM-D5 — audit-adopted cheap tests (registered 2026-07-12, BEFORE results; from
## the independent audit-community report §1 U2/U4/U5 + T9 correction)
D5-CORRECTION to D2a: the community tail statistic used cells at n>=20 — the
documented small-n quantile bias (tail_gap_boot caveat) inflates thin-cell
q20/median toward 1. The D2a "community tails not fast" verdict is therefore
DOWNGRADED to 🟠 pending matched-n subsampling: recompute with aalto cells
subsampled to the community per-cell n distribution. Rule: if matched-n aalto
qwerty ratio rises to within 0.01 of community (0.777), D2a's challenge verdict
is VOID (artifact); if the gap persists (aalto stays <= 0.76), the challenge
stands.
D5-U2 PINKY TRANSFER, proper estimator: run fit_first_finger_deltas (the actual
PINKY-FIT estimator, not the coarse outer-inner contrast) per community label
with enough matched pairs (expect colemak-dh, recurva). PASS per label = both
classes positive AND pinky >= ring. 2/2 => transfer note upgrades to 🟡; 0/2 or
sign flip => population-transfer caveat in layout docs (not a retrain trigger).
D5-U4 PRACTICE NATURAL EXPERIMENT (+pseudo): same typist/layout/board, natural
242k vs pseudo 102k samples. Per-bigram delta = natural mean - pseudo mean at
matched wpm buckets; practice proxy = our corpus freq (log). Rule: practice term
VALIDATED iff rank-corr(log-freq, natural-minus-pseudo speedup) > 0 with
bootstrap CI excluding 0 (frequent bigrams should benefit more from lifetime
practice than rare ones, and pseudo-words ablate exactly that).
D5-U5 RARE-DECILE (+rareboost): frozen production model scores the +rareboost
cell frame; report rho/own-ceiling on the rare-corpus-decile subset. >=0.5 PASS
(informational; annotates dec3-guard interpretation, reopens nothing).
Also: README count fix (9 layout strings/4 customs), OQ-6 "not reopened" note.

### Outcome append (2026-07-12): COMM-D5 — tail challenge SURVIVES matched-n; pinky
### transfer mixed/underpowered; +pseudo practice prediction FAILS (inverted); rare
### decile weak
runs/comm_d5.json.
D5-CORR: aalto qwerty matched-n 0.752 vs community 0.777 — the gap persists =>
D2a's challenge to the strong tail-practice story STANDS (upgraded back to 🟡).
Community daily-drivers do not show the qwerty fast-tail signature at months-scale
practice.
D5-U2 (proper estimator): each label had matched pairs for only ONE class —
colemak-dh: ring_first -31.0ms (SIGN FLIP vs aalto +20.7); recurva: pinky_first
+25.3ms (sign + magnitude consistent with aalto +43.1). 1-of-2 consistent, 1-of-2
inverted, zero labels with both classes => registered verdict: population-transfer
caveat goes into the layout docs; NOT a retrain trigger. The calibration keeps its
aalto evidence; its transfer status is now measured-mixed rather than unmeasured.
D5-U4: rank-corr(log-freq, natural-minus-pseudo speedup) = -0.191, CI [-0.288,
-0.089] — the practice prediction FAILS, with the correlation significantly
INVERTED: frequent bigrams show LESS natural-vs-pseudo advantage. Honest
interpretations recorded, not adjudicable here: (a) ceiling effect — frequent
bigrams are already at motor floor for a 250k-sample typist, leaving no headroom
for a practice differential; (b) the pseudo corpus deliberately re-weights toward
home-row/rare chars, so its frequent-bigram sample differs structurally; (c) the
practice term as fitted (log-count on OUR frame) may proxy something other than
lifetime familiarity. This does NOT invalidate the practice term's in-frame job
(absorbing per-ngram repetition within aalto), but it removes the presumption
that it measures transferable lifetime practice. Flagged for the next
model-improvement round.
D5-U5: +rareboost rare-decile rho 0.27 (< 0.5) — rare-cell ordering remains the
model's weakest axis on new data too; consistent with every dec3 guard trip.
Informational, annotates dec3 interpretation.

## CAL-REMOVE (registered 2026-07-12, BEFORE results; user directive: "remove the ring /
## pinky calibration and find a better solution. It seems to hurt layout generation, and
## seems too hacky")
CONTEXT (already measured, experiment_cal_comm.json): ARM-NOCAL (retrain without
calibration, same seeds/recipe) speed +3.90% vs production +3.95-4.00% — inside the
~0.2% plateau; LOLO wmae 24.33 vs 24.35 (identical); the calibration's only measured
effect is steering outer-first corpus share 1.22%->0.42% at ~nil speed cost. Community
evidence (D5-U2): ring_first sign FLIPS on the one community label with matched cells
(-31.0 vs aalto +20.7); pinky_first replicates on the other (+25.3 vs +43.1). The
physics finding (PINKY-GAP +27.4ms qwerty matched pairs, 8/8 cells) STANDS as a
measurement; what's removed is the pipeline seam that injects it into the served
surface — single-population evidence, mixed transfer, zero speed contribution.
CHANGE: (a) train_bigram_model(calibration=...) default flips True->False;
calibration.py stays (the estimator is a legitimate measurement tool + D5-U2 uses it);
TableBigramScorer/predict_ms_at keep their sidecar-reading serve path (old artifacts
with deltas still serve correctly — backward compatible); (b) production models retrain
WITHOUT the seam (bigram_nocal_seed{0,1,2}); (c) the deliverable docs drop the
"calibration steers placement" provenance line and gain the removal rationale + the
PINKY-GAP finding retained as documented-but-not-installed physics.
RULE: the removal SHIPS iff (i) LOLO non-degradation vs calibrated production (wmae
within +0.91% noise floor, umae/dec3 within +2%, taus no lower); (ii) re-searched
family under the nocal surface produces a pick within 0.2% speed of P10-w0.5 under
BOTH surfaces (cross-regret — plateau equivalence); (iii) P10-w0.5 itself stays
within 0.2% of the new family's best under the nocal objective (the deliverable does
NOT change unless (iii) fails, in which case the new pick replaces it with full gauge
re-run). Deltas from experiment_cal_comm (LOLO 24.33 vs 24.35, +3.90% vs +3.95%)
pre-satisfy (i)+(iii) directionally; this registration makes the production-path
verification the binding check.

## DATA-CLEAN (registered 2026-07-12, BEFORE results; user directive: "audit how we
## take into account typos + how clean the data is; try everything to make good use of
## the community data — filtering, removing thin trigrams, slow participants, pseudo
## words")
Parallel audit: audit-data-quality subagent (independent, read-only) reports on typo
semantics + contamination in BOTH pipelines. This registration covers the EXPERIMENT
arms (main agent), which run regardless of the audit's findings; any audit finding
that changes an arm's design gets an amendment BEFORE that arm's results are read.
ARMS ALREADY RUN (experiment_cal_comm.json, registered post-hoc as EXPLORATORY — their
results informed THIS registration; confirmatory reruns below use held-out checks):
thick-only/fast-only(wpm>=55)/w25/RS-only integration all FAIL (speed -1.2 to -1.4%,
LOLO +32-35%); community-fitted calibration ring sign flips. NEW CONFIRMATORY ARMS:
  CLEAN-1 ERROR-RATE session filter: drop community sessions with error rate > {10%,
    20%} before window extraction (typo-adjacent contamination hypothesis: high-error
    sessions carry polluted intervals even in all-correct windows — post-error
    intervals measure from the error press). Re-run the D3 LODO-8 protocol (incumbent
    vs merged) on the filtered frame. ADOPT rule identical to D3: every aalto fold
    within +0.91%, community mean improvement >1%, guards.
  CLEAN-2 POST-ERROR EXCLUSION: extend extract_windows to also require the event
    BEFORE the window be correct (kills the "first interval measures from an error
    press" channel — note the window's OWN first event carries no interval, so this
    tests the NEXT-lag contamination). Rebuild community TSV, re-run D1 zero-shot
    per-label rho/ceiling on the 4 primary labels. PASS if any label's rho/ceiling
    improves >0.05 (evidence the contamination was masking transfer); else the
    exclusion is documented as immaterial.
  CLEAN-3 MIN-SAMPLES TRIGRAM FLOOR: the user's "remove trigrams which have less than
    N samples" — community tristrokes cells are thin; sweep min_cell_samples {10, 20,
    50} on the community trigram frame and recompute D1-style per-label rho/ceiling
    for the 2 powered labels (colemak-dh, recurva). Report the curve; no adoption
    (validation-only — trigram integration was never on the table).
  CLEAN-4 WPM-BAND TIGHTening: drop community samples outside wpm 50-120 (the
    thin-tail buckets contribute noisy cells at band edges). Re-run D1 rho/ceiling.
    Informational.
RULE: CLEAN-1 is the only arm with an adoption path (it re-tests integration under
the strongest cleanliness hypothesis). CLEAN-2/3/4 are validation-quality probes:
they can upgrade/annotate D1's verdict but cannot reopen integration by themselves
(that requires CLEAN-1 to pass its D3-rule). Everything else stays validation-only
per D3b. If ALL arms show no material change, the registered conclusion is: the
community data's failure to integrate is NOT a cleanliness artifact at any filter
level tested — it is structural (1 typist per layout), closing the filtering
question with the same verdict as D3b.

### Outcome append (2026-07-12): DATA-CLEAN — ALL arms negative; the integration failure
### is STRUCTURAL, not a cleanliness artifact (registered conclusion fires)
runs/clean_arms.json. Session error rates: median 8.5%, p90 19.1%.
CLEAN-1 (error-rate caps 10%/20%, D3 LODO-8 protocol): adopt=False BOTH. The community-
fold side actually clears its bar at cap 20% (mean d_wmae -2.42% < -1%) but the aalto
folds still degrade catastrophically (+19 to +37%) and guards fail — filtering typo-heavy
sessions does NOT remove the poison; it is not typo-borne.
CLEAN-2 (post-error window exclusion): drops 3.3% of samples; rho/ceiling moves <= 0.015
on all 4 primary labels => IMMATERIAL. Mechanism note: the base extractor was already
correct — a window's first event contributes no interval, so the "interval measured from
an error press" channel only exists for the event PRECEDING the window, and excluding it
changes nothing measurable.
CLEAN-3 (trigram cell floor {10,20,50}): colemak-dh rho peaks at floor 20 (0.461 vs 0.445
@10, 0.403 @50 — thin-cell noise and data loss trade off); recurva ~0.01 at EVERY floor —
that label's trigram structure simply does not correlate with the model, floor-independent.
CLEAN-4 (wpm band 50-120): moves within +-0.03 => IMMATERIAL.
REGISTERED CONCLUSION (per the DATA-CLEAN rule): the community data's failure to
integrate is NOT a cleanliness artifact at any filter level tested — it is structural
(1 typist per layout). The filtering question is CLOSED with the same verdict as D3b.

### Outcome append (2026-07-12): CAL-REMOVE gates — (ii)+(iii) PASS decisively;
### (i) FAILS by letter on the dvorak fold; seed adjudication registered below
runs/cal_remove_verify.json (true LOLO through the production train fn, 2-seed arms):
gate (ii): nocal re-search pick gdblk.,oyuscthrpnaiezvwmxfjq;/ has +0.002% regret under
the calibrated surface; gate (iii): P10-w0.5 regret under the nocal surface +0.005% —
BOTH orders of magnitude inside the 0.2% bar. The user's "calibration hurts layout
generation" premise is REFUTED: the argmax is calibration-invariant.
gate (i) by fold (nocal vs cal): qwerty -1.66%, qwertz +0.05%, azerty -1.57%, dvorak
+1.74% => FAIL by letter (dvorak > 0.91%). NOTE the sign pattern: removal IMPROVES two
folds and degrades one — the seam's cross-fold value is inconsistent within aalto, the
same mixed-transfer signature as D5-U2's community sign flip.

## CAL-REMOVE-ADJ (registered 2026-07-12, BEFORE results): dvorak-fold seed adjudication
## + the "better solution" arm the user asked for
ADJ-1 SEED NOISE: the gate-(i) comparison used 2-seed means; the 0.91% bar is a
single-pair p95. Re-measure the dvorak fold ONLY with 5 seed pairs per arm (same
protocol otherwise). The gate-(i) dvorak verdict becomes the 5-seed mean delta vs the
same 0.91% bar (a better estimate of the SAME registered quantity, not a new rule).
If <= 0.91%: gate (i) passes on the better estimate and CAL-REMOVE ships as registered.
If > 0.91%: the dvorak cost is real; the removal decision escalates to the user with
the full trade documented (speed-neutral for generation, mixed prediction effect:
2 folds improve, dvorak degrades).
ADJ-2 PINKY-MONO (the constrained-learning route — "learn it properly" without the
seam): add TWO indicator columns to the bigram features IN-DRIVER (no schema change
yet): outer_first_pinky (1 iff finger_class=pinky_first), outer_first_ring (1 iff
ring_first), trained with XGBoost monotone_constraints=+1 on both columns, calibration
OFF. The monotone constraint makes the PINKYFIX sign inversion impossible by
construction; the magnitude is learned from the data (cross-layout rows + whatever
within-layout signal the practice term does not absorb). RULE: PINKY-MONO replaces the
seam iff (a) LOLO all folds within +0.91% of the CALIBRATED incumbent (incl. dvorak —
the fold the seam helps); (b) umae/dec3 guards <= +2%; (c) served-sign: the probe pairs
(as/ds, po/io, we/re, oi/ui) priced outer-first slower at wpm 70 and 90 (>= 6/8);
(d) E5-style sanity: re-search under the MONO surface, pick within 0.2% of P10-w0.5
both ways. If (c) fails with near-zero learned magnitudes, the honest conclusion is
that the collision physics cannot be learned without an explicit offset on THIS data —
the seam (or its removal with the dvorak caveat) are the only options, and the user
decides between them.

## COMM-ERR + COMM-RESID (registered 2026-07-12, BEFORE results; user: "try everything
## possible to make good use of this new data")
Two untried validation channels on the frozen production stack:
COMM-ERR ERROR-CLUSTERING: typos as an independent difficulty signal. The model
predicts per-position-pair TIME; if its difficulty surface is real, community typists
should mis-hit MORE on bigrams the model prices as slow (motor difficulty produces
both slowness and errors). Per natural label: for each bigram cell with >= 30
attempts (windows where the first key is correct and a second press follows), error
rate = share of second events with correct=false; Spearman(prediction, error rate)
per label, bucket-centered, with a distance-only baseline. PASS (informational
validation) if pooled rho > 0 with p < 0.05 AND beats the distance baseline on >= 3
of 4 primary labels. This cannot change the deliverable; it can only add or deny an
independent-channel validation paragraph.
COMM-RESID SYSTEMATIC RESIDUAL SWEEP: the audit report's re-search trigger asks for
"geometry-controlled, replicated mispricing" — nobody has actually swept for one. Per
natural label: per-cell signed residual (obs - pred, bucket-centered) averaged by
feature class {sfb, alternate, inroll, outroll, lsb, scissor, same_finger_skip,
redirect-bigram-classes}; a class is FLAGGED iff |mean residual| > 5ms AND the SIGN
replicates on >= 3 natural labels including >= 1 rowStagger. Flagged classes feed the
audit's 4-condition trigger (they'd still need the practice-matched design + aalto
gates + argmax move before anything changes). No flags => the model's class pricing
survives community data — recorded as the final line of the community campaign.

### Outcome append (2026-07-12): COMM-ERR null (inverted); COMM-RESID FLAGS 4 classes
### — practice-matched design (condition iii) now decisive
runs/comm_err_resid.json.
COMM-ERR: NO PASS (0/4). The correlation is significantly NEGATIVE on 3 labels
(errors cluster on bigrams the model prices FAST, rho -0.16 to -0.28) — typos do not
mark motor-difficult pairs in this capture; plausibly errors concentrate on
high-speed/high-frequency patterns (carelessness, rolled-through corrections), or the
monkeytype 'key' field on incorrect events does not identify the intended bigram.
Recorded as a null; the error channel is NOT independent validation on this data.
COMM-RESID flags (sign-replicated >= 3 natural labels incl >= 1 rowStagger):
  sfb OVERPRICED (obs faster than predicted): 6/7 labels, -27 to -48ms
  outroll OVERPRICED: 4 labels, -23 to -38ms
  inroll OVERPRICED: 3 labels, -6 to -20ms
  alternate UNDERPRICED (obs slower): 5 labels, +7 to +15ms
Direction = community doctrine (rolls/sfb cheaper, alternation less special, for
enthusiasts on chosen layouts). NOT yet actionable: predictions were geometry-only
(no practice term), and roll-optimized layouts BY DESIGN place high-frequency
(lifetime-practiced) bigrams as rolls/sfb-remnants — the class composition is
frequency-confounded exactly as audit trap T3 warns. Condition (iii) of the audit's
4-condition trigger (practice-matched design) is now the decisive test.

## COMM-RESID-2 (registered 2026-07-12, BEFORE results): the practice-matched design
Per natural label: per-cell signed residual (obs - pred, bucket-centered, geometry-only
predictions as in COMM-RESID). CONTROL: within (label, bucket), OLS of residual on
log10(corpus frequency of the cell's ngram) + log10(label-local attempt count) — the
two practice proxies (lifetime + this-capture volume). Class means recomputed on the
DOUBLY-ADJUSTED residuals. A flag SURVIVES iff |adjusted mean| > 5ms AND the sign still
replicates on >= 3 natural labels including >= 1 rowStagger.
RULE: classes surviving (iii) proceed to condition (iv) — a calibrated class-offset arm
(same seam mechanics as PINKY-CAL, offsets = the surviving classes' adjusted community
deltas) gated ENTIRELY on aalto: LOLO non-degradation + guards + re-search argmax move
> 0.2% required before the deliverable changes. Classes that do NOT survive are
recorded as practice-composition artifacts, closing the community-mispricing question.
NOTE the sfb flag's magnitude (-27..-48ms) rivals the aalto sfb penalty itself
(+27-38ms); if it survived (iii)+(iv) the argmax WOULD plausibly move — this is the
first community finding with that potential. Prior expectation (honest): the
frequency/practice adjustment absorbs most of it; alternation's +7-15ms may partially
survive (its cells are the LOW-practice mass on these layouts, so adjustment moves it
the other way).

### Outcome append (2026-07-12): CAL-REMOVE-ADJ — ADJ-1 dvorak cost REAL (+1.90%
### 5-seed); ADJ-2 PINKY-MONO fails 0/8 (constraint flattens to zero, cannot replace
### the seam)
runs/cal_remove_adj.json.
ADJ-1: dvorak fold 5-seed mean d_wmae +1.90% (seeds +1.57 to +2.34, consistent) > 0.91%
bar => the seam's dvorak prediction value is REAL, not seed noise. Removal trade as
measured: qwerty -1.66%, azerty -1.57% (IMPROVE), qwertz +0.05%, dvorak +1.90%
(DEGRADE). Generation is calibration-INVARIANT (gates ii/iii passed at +0.002%/+0.005%
regret — orders of magnitude inside the plateau).
ADJ-2 PINKY-MONO: monotone-constrained indicator columns learn ZERO magnitude (served
gap +0.0ms on all 8 probe pairs — the constraint prevents PINKYFIX's sign inversion but
the within-layout collinearity with the practice term still starves the columns of
attributable variance; the estimator's identifying restriction cannot be replicated by
a constraint alone). LOLO also fails (qwertz +1.57%, dvorak +1.64%). Route CLOSED: no
learnable replacement for the seam exists on this data.
DECISION (user-directed): the removal SHIPS — the user directed removal before these
results ("seems too hacky"), generation is provably unaffected, and the cost is
confined to one validation fold's prediction quality (dvorak +1.9% wmae, still
rho +0.69). The trade is documented here and in the layout docs; reinstating the seam
(calibration=True) remains a one-flag revert if the dvorak fold is ever load-bearing.

### Outcome append (2026-07-12): COMM-RESID-2 — sfb/outroll/alternate SURVIVE the
### practice-matched design; inroll dies; condition (iv) arm registered below
runs/comm_resid2.json. Practice slopes are real and large (volume -2 to -54 ms/log10,
freq +5 to +7 ms/log10 — the freq sign is POSITIVE after volume is controlled,
i.e. corpus-frequent bigrams are SLOWER than their local-volume peers on these
layouts; volume, not lifetime frequency, carries the speedup — consistent with
D5-U4's inverted practice result).
Adjusted class residuals (survival rule: |mean| > 5ms, sign-replicated >= 3 natural
labels incl >= 1 rowStagger):
  sfb OVERPRICED survives: -21 to -39ms, 6/6 labels (both geometries)
  outroll OVERPRICED survives: -18 to -34ms, 4 labels incl 2 rowStagger
  alternate UNDERPRICED survives: +5 to +12ms, 5 labels incl 3 rowStagger
  inroll DIES (mixed signs after adjustment) — was a practice-composition artifact
Honest mechanism caveats (recorded BEFORE the (iv) result): (a) self-selection — these
typists chose roll-optimized layouts, plausibly BECAUSE their motor profile favors
rolls/tolerates sfbs; no within-typist adjustment removes population selection (audit
T6); (b) alt-fingering — enthusiasts deliberately alternate-finger their layouts'
residual sfbs (documented ~8% canonical-map violations even in aalto); a nominal-sfb
cell typed with two fingers is not an sfb, deflating the measured penalty; (c) the
linear practice control may under-adjust saturation. NONE of these can be resolved on
this capture (no release timestamps, no video, n=1/layout).

## COMM-RESID-IV (registered 2026-07-12, BEFORE results): the aalto-gated offset arm —
## audit condition (iv), the last gate before any deliverable change
ARM: inject the surviving classes' pooled adjusted community deltas as class offsets on
the served surface (seam mechanics, LOGRAT units at serve): sfb -30ms, outroll -25ms,
alternate +8ms (pooled means of surviving labels; exact values recomputed in-driver
from runs/comm_resid2.json and recorded). Then:
  (iv-a) LOLO on aalto with the offsets applied at eval: every fold wmae within +0.91%
         of the un-offset incumbent, umae/dec3 <= +2%.
  (iv-b) re-search (T3c + offsets, oxey w=0.5, rng 885333): does the argmax move > 0.2%
         (P10-w0.5 regret under the offset surface)?
RULE: the deliverable changes ONLY if (iv-a) passes AND (iv-b) moves the argmax — in
which case the new family runs in full (multi-gauge + docs + user sign-off). EXPECTED
(honest prior): (iv-a) FAILS — the offsets contradict aalto's own measured physics
(sfb +27-38ms was measured THERE), so applying community pricing should degrade aalto
folds materially; the registered conclusion is then POPULATION DIVERGENCE, not model
error: enthusiasts on chosen layouts genuinely pay different class prices than the
general population, the deliverable optimizes for the general population BY DESIGN, and
the divergence is documented in the layout docs as the community data's final lesson.
INFORMATIONAL RIDER (no decision power): D1 zero-shot rho recomputed per wpm-bucket per
primary label — does transfer fail uniformly or concentrate in a band?

### Outcome append (2026-07-12): COMM-RESID-IV — (iv-a) FAILS decisively (+15 to +55%);
### POPULATION DIVERGENCE conclusion fires; the community-mispricing question is CLOSED
runs/comm_resid_iv.json. Offsets applied: sfb -29.7ms, outroll -22.1ms, alternate +9.0ms.
(iv-a): applying community class prices to aalto predictions degrades EVERY fold
catastrophically — qwerty +54.6%, azerty +42.1%, qwertz +41.1%, dvorak +15.0% wmae.
The two populations' class prices are mutually exclusive: aalto's measured sfb penalty
(+27-38ms) and the community's measured sfb discount (-30ms) cannot both serve one
model. REGISTERED CONCLUSION (as pre-stated): population divergence, not model error —
enthusiasts on self-chosen roll-optimized layouts pay different class prices than the
general population; the deliverable optimizes for the general population BY DESIGN.
(iv-b, informational): had the community prices been injected anyway, the argmax would
move +0.303% (pick gdplk.rouyscthm,naieqzwbvfx/;j, 18/30 shared with P10) — i.e. the
divergence is large enough to matter, which makes the honest documentation of WHO the
deliverable serves the load-bearing sentence, not a footnote.
Rider (per-bucket zero-shot rho): transfer is band-structured — colemak-dh/colemak hold
+0.28-0.37 in their home bands; recurva is ~0 through 40-100 but +0.28 at 120 (its
typist's fastest band). Weak evidence that transfer improves toward each typist's
comfort band; too thin to adjudicate anything.
COMMUNITY CAMPAIGN FINAL LEDGER (all questions now closed): integration NO (D3/D3b/
CLEAN-1 — structural); zero-shot per-person ordering NO (D1); alternation-preference
population-generality YES (D2b, 7/7); tail-practice challenge STANDS (D2a/D5-CORR);
practice term not lifetime-transferable (D5-U4, freq slope +5..+7 POSITIVE after
volume control in COMM-RESID-2 — replicating the inversion); calibration transfer
mixed (D5-U2) and the seam is now REMOVED (CAL-REMOVE); typo channel not validation
(COMM-ERR inverted); class-price divergence REAL and population-attributed
(COMM-RESID/2/IV). Remaining value: collection-design lessons for Phase D.

## COMM-ALTFINGER (registered 2026-07-12, BEFORE results): adjudicating the sfb
## discount — alternate-fingering vs population selection
The COMM-RESID-2 sfb flag (-21..-39ms, 6/6 labels) has two live explanations that THIS
capture can partially separate: a true same-finger execution has a mechanical floor
(the finger must release, travel, re-press — aalto sfb cells run ~150ms+ at typical
wpm), while an ALTERNATE-FINGERED nominal-sfb is executed by two fingers and can
overlap/roll at <100ms. If community typists alternate-finger their layouts' residual
sfbs, community nominal-sfb cells should show a FAST SUB-POPULATION that aalto qwerty
sfb cells lack.
DESIGN: per source (community natural labels pooled | aalto qwerty), take all
nominal-sfb bigram samples in wpm 60-100 with cell n >= 30. Per cell compute
p10/median ratio and the share of samples < 0.6 x cell median ("fast-mode share").
Compare distributions: community-vs-aalto fast-mode share via Mann-Whitney; and
within community, correlate a cell's fast-mode share with its residual (obs - pred)
— if alt-fingering drives the discount, cells with more fast-mode executions should
show more negative residuals.
RULE (informational — no deliverable consequence either way): alt-fingering is
SUPPORTED iff community fast-mode share exceeds aalto's (p < 0.01) AND the
within-community correlation is negative (p < 0.05). SUPPORTED => the sfb divergence
is at least partly EXECUTION STRATEGY, not physics — documented in the layout docs
sentence; NOT SUPPORTED => selection/practice explanations stand unresolved (as
registered in COMM-RESID-2).

### Outcome append (2026-07-12): COMM-ALTFINGER — NOT SUPPORTED; the sfb discount is a
### uniform shift, not an execution-strategy bimodality
runs/comm_altfinger.json. Gate 1 FAILS cleanly: community nominal-sfb cells' fast-mode
share (median 0.033) is indistinguishable from aalto qwerty's (0.036), Mann-Whitney
p=0.76; p10/median ratios also match (0.719 vs 0.743). No fast sub-population exists —
community sfb executions have the same distributional shape as aalto's, shifted faster
overall. (Gate 2's rho -0.482 is mechanically confounded — fast-mode share lowers the
IQR-mean directly — and moot given gate 1.) CONSEQUENCE: alternate-fingering is
disfavored as the driver of the -30ms sfb discount; the leading explanations narrow to
population selection (typists who tolerate sfbs choose these layouts) and/or uniform
deliberate practice on the few residual sfbs their layouts retain. Both are
population-relative, neither transfers to the general-population deliverable; the
COMM-RESID-IV conclusion (population divergence) stands with a sharper mechanism note.

## KIAKL-INGEST Amendment 3 (2026-07-12) — CRITICAL: the key field is a QWERTY-POSITION
## LABEL, not the produced character; every non-qwerty community session was ingested
## with position-scrambled geometry
EVIDENCE (runs recorded before this amendment; decode probe reproducible from raw zips):
for every one of the 19 parseable capture files, treating data[].key as the produced
character yields gibberish (common-word hit rate 0.00-0.01), while decoding
produced_char = session_layout[qwerty_index(key)] yields English text (0.14-0.32; e.g.
Andrew Castro: "pose copy five center old state office sent stay size..."). The
+pseudo file decodes to pseudo-words and the +rareboost file to rare/multilingual
words under the SAME decode — independent confirmation via the corpus tags. Mechanism:
these typists use monkeytype's software layout emulation; the browser event carries
the OS-level (qwerty) character of the physical key, monkeytype remaps internally.
The audit-data-quality subagent independently identified the same interpretation.
CONSEQUENCE FOR THE INGEST: community.py's cmap mapped the LABEL through the SESSION
layout (positions = P(label)); the true physical slot is the label's QWERTY slot
(= P(produced_char)). Recorded positions are therefore wrong by the fixed slot
permutation P∘Q⁻¹ per layout; recorded ngram text is the qwerty transliteration
(freq column ≈ meaningless); timing values themselves are correct.
BLAST RADIUS (all community-side results to date are VOID pending re-run — they were
computed on scrambled geometry): D1 zero-shot (the 0/4 failure may be the scramble,
not the model), D2a tail / D2b alternation-confirmed / D2c pinky, D3+D3b integration
rejections, D5 all parts, DATA-CLEAN, COMM-ERR, COMM-RESID/2/IV flags (the sfb-
overpriced pattern is exactly what class-mislabeling predicts), COMM-ALTFINGER.
Aalto-side results (CAL-REMOVE gates, PINKY-GAP, all production models) are UNAFFECTED.
The scrambled TSVs remain in git history; the fix regenerates them in place.
FIX (code): decode each event to produced_char = main30[qwerty_index(key)] (identity
for qwerty sessions; undecodable labels — Backspace, shifted chars — keep their raw
key and break windows exactly as before). Everything downstream (cmap, windows, wpm)
is then correct by construction. Unit tests pin the decode on a colemak-dh example
and the qwerty identity.
RE-RUN RULES (same rules as the originals, now on corrected data — no goalpost moves):
  R-D1 zero-shot per-label rho/ceiling, 4 primary labels (D1 rule: PASS iff
       rho/own-ceiling >= 0.5 AND beats distance+wpm baseline; report per label).
  R-D2b alternation-vs-roll observed contrast per natural label (D2b rule: story
       challenged iff >= 3 natural labels incl >= 1 rowStagger show rolls faster,
       CI excluding 0).
  R-D3 LODO-8 integration (D3 rule verbatim: adopt iff every aalto fold within
       +0.91%, community mean improvement > 1%, umae/dec3 guards <= +2%).
  R-RESID class-residual sweep + practice adjustment (COMM-RESID/2 rules verbatim).
Anything that changes verdict gets its own outcome append; the docs' community
section is rewritten from the re-run results only.

## KIAKL-INGEST Amendment 4 (2026-07-12) — auxiliary cleaning fixes shipping with the
## Amendment-3 decode fix (both from the audit-data-quality report, sections S2/A2)
(a) SHIFT RECOVERY: before decoding, unshift the key label (A-Z -> a-z; qwerty shift
pairs '<'->',', '>'->'.', '?'->'/', ':'->';'). Recovers the 1.9-3.6% of correct events
that are shifted presses (they are valid typing; under the old semantics they were
silently window-breaking). Undecodable labels (Backspace, Enter, unicode chars outside
the 30-key map) still break windows exactly as registered.
(b) PREFIX-STREAM DEDUP (audit S2): sessionID is an export timestamp, so re-exports get
fresh ids and survive the registered dedup. New rule: after sessionID dedup, drop any
session whose full event stream (key, correct) sequence is a strict prefix of another
kept session's (keep the longer). Audit measured 102 such sessions (3.89% of vg-recurva
windows double-counted).
Both are mechanical data-correctness fixes; they ship together with the Amendment-3
re-ingest and are covered by the same re-run rules (R-D1/R-D2b/R-D3/R-RESID).

### Outcome append (2026-07-12): RE-RUNS ON CORRECTED DATA (Amendment 3+4) — the
### decode fix rewrites the community story: zero-shot transfer RECOVERS; alternation
### verdict flips to mixed/tied; integration still rejected (aalto poison unchanged)
runs/rerun_d1.json, rerun_d2.json, rerun_d2b_ci_origdef.json, rerun_d3.json,
rerun_resid2.json. All rules verbatim from the originals (Amendment-3 R-rules).
R-D1 ZERO-SHOT: rho/ceiling recovers on EVERY label (scrambled -> corrected):
  colemak-dh .394->.582, recurva -.009->.510(!), castro .205->.606, alite .343->.539,
  mtgap .144->.489, ddn -.111->.234, vg-custom .222->.396; +pseudo .371->.573,
  +rareboost .385->.612. By the registered PASS bar (>=0.5): 4 of 9 labels pass
  outright, two more sit at .49/.40. The original primary-4 set (chosen for sample
  volume when geometry was scrambled) still counts 1/4 primary passes, so the
  REGISTERED headline stays "no generalization claim" BY LETTER — but the honest
  summary changes completely: the model's cell ordering transfers at ~half-to-0.6x of
  each typist's own noise ceiling on alien layouts/population, roughly the in-family
  dvorak-fold level (rho .42-.69), where the scrambled data had shown ~zero. The D1
  failure headline was the ingest bug, as the audit predicted. Magnitude wmape ~0.27
  (slope 1.19) — levels shift cross-population, structure transfers.
R-D2b ALTERNATION (original class def, cell-bootstrap CIs): aalto qwerty +17.9
  [+13.5,+22.2] alternation-faster; community: colemak-dh +10.7* alt-faster,
  2 labels rolls-faster* (vg-custom -13.5, mtgap -14.4 — both rowStagger), 4 ties.
  Registered challenge rule (>=3 incl 1 rowStagger) does NOT fire (2 labels) => story
  STANDS by letter, but the evidence grade changes: the old "7/7 all-typists
  alternation-faster" claim is VOID (it was computed on scrambled classes); corrected
  data shows a MIXED picture — alternation-faster for the aalto population and the
  highest-volume community typist, tied-to-reversed for several roll-trained
  enthusiasts. The deliverable docs' "confirmed population-general" line must be
  rewritten to this mixed verdict.
R-D5U2 PINKY (corrected): colemak-dh pinky_first +47.1 (aalto +43.1 — replicates),
  ring_first -41.0 (still inverts); recurva ring_first +68.8 (large, right sign),
  pinky_first +1.2 (~null). Net: 3 of 4 label-class estimates now carry the aalto
  sign (was 1 of 2) — transfer evidence IMPROVES but stays mixed. No bearing on
  CAL-REMOVE (removal was decided on speed-neutrality + the user's simplicity
  directive, which stand regardless).
R-D3 INTEGRATION: adopt=False again — aalto folds still poisoned (qwerty +24.8%,
  qwertz +16.5%, azerty +19.6%, dvorak +1.3%) — BUT community folds now IMPROVE
  under merge (alite -6.3%, mtgap -8.1%, ddn -14.6%, vg -0.9%; mean -7.5% vs the
  scrambled run's -2.2%): cross-typist community structure is REAL and learnable,
  the blocker is purely that community rows damage aalto folds (population price
  divergence + 1-typist labels). Verdict unchanged; mechanism sharper.
R-RESID2 CLASS FLAGS: the scrambled run's flags were artifacts as suspected — sfb
  and alternate flags DISSOLVE on corrected data (sfb now +23/+27 on colemak-dh
  labels vs -24 recurva: mixed signs); surviving: outroll UNDERPRICED (+5.7..+27,
  3 labels) and inroll OVERPRICED (-6.6..-24.1, 3 labels) — i.e. the model prices
  in-rolls too slow and out-rolls too fast for these typists, a roll-DIRECTION
  asymmetry rather than the roll-level story. COMM-RESID-IV re-runs on these two
  flags (same rule; expected to fail iv-a again given aalto gating).

### Outcome append (2026-07-12): corrected-data COMM-RESID-IV + D5 re-runs — every
### remaining thread closes; the community campaign's final corrected ledger
runs/rerun_resid_iv.json, rerun_d5.json.
R-RESID-IV (offsets outroll +14.9 / inroll -13.8 from the corrected flags):
  (iv-a) FAIL again — aalto folds degrade +1.6 to +8.5% (roll-direction offsets
  contradict aalto pricing too, just 6x less violently than the scrambled-era sfb
  offsets). (iv-b): P10-w0.5 regret under the offset surface -0.018% — the argmax
  does NOT move even if the offsets were injected. DOUBLE-CLOSED: the roll-direction
  asymmetry is population-divergent AND argmax-irrelevant. No community-derived
  repricing path remains open.
  Rider (per-bucket zero-shot rho, corrected): uniform ~+0.42-0.55 across ALL wpm
  bands and labels — transfer is band-uniform, not band-structured (the scrambled
  rider's recurva-at-120 artifact is gone).
R-D5-CORR TAIL: community 0.777 vs aalto matched-n 0.750 — challenge STANDS on
  corrected data (this analysis was position-agnostic, so unchanged as expected).
R-D5-U4 PRACTICE: rank-corr -0.110 CI [-0.209, -0.007] — still inverted/not
  validated (was -0.191). The corrected decode weakens but does not flip it; the
  practice-term boundary finding survives.
R-D5-U5 RAREBOOST: rare-decile rho 0.437 (was 0.27) — improves substantially on
  corrected identities, still under the 0.5 informational bar. Rare-cell ordering
  remains the weakest axis, but by less than the scrambled data suggested.
FINAL CORRECTED LEDGER (supersedes the pre-Amendment-3 ledger): transfer of cell
STRUCTURE to alien layouts/populations is REAL (~0.5-0.6x own-ceiling, uniform
across wpm); integration remains closed (aalto poison, now attributed to price
divergence + 1-typist labels, NOT to transfer failure); alternation-vs-rolls is
MIXED across community typists (population-relative preference, not universal
physics — aalto's +18ms remains the deliverable's basis); tail-practice challenge
stands; practice term not lifetime-transferable; pinky calibration transfer improved
(3/4 sign-consistent) but the seam stays removed per CAL-REMOVE; no repricing or
argmax-moving path exists from this dataset. P10-w0.5 UNCHANGED through every
corrected re-run.

## R-D3B (registered 2026-07-12, BEFORE results): corrected-data re-run of the D3b arms
## — the one stale rejection with a live mechanism
WHY (goalpost discipline — Amendment-3 R-rule extension, not a re-roll): D3b's ARM-W
(community weight x0.25) and ARM-P (is_community_population column, served population=0)
were rejected on SCRAMBLED community geometry, where the community rows could only
inject noise. Corrected re-runs showed the community geometry now carries real signal
(R-D1 transfer ~0.5-0.6x ceiling; R-D3 community folds improve -7.5% under merge).
ARM-P is exactly the design that could keep aalto pricing intact (population indicator
absorbs the price divergence; served surface uses population=0) while borrowing shared
geometry signal — its rejection is the only one whose mechanism materially changed
with the fix. RULES VERBATIM from D3b: per arm, LODO-8; adopt iff every aalto fold
wmae within +0.91% of incumbent AND community-fold mean wmae improves >1% vs incumbent
AND umae/dec3 guards <= +2% on aalto folds. Driver rerun_d3b.py = comm_d3b.py with
corrected TSV (same path, regenerated) + rerun output name.

## REG-LOLO (registered 2026-07-12, BEFORE results; user: "have we tried including
## regularization parameters in our tuning?")
GAP (honest): explicit regularization (reg_alpha/reg_lambda/gamma) was swept ONLY by
the deprecated CV-MAE tuner (tune.py::tune_hyperparameters — rewards memorization,
winners never shipped). The transfer-scored selectors that picked production params
(tune_lolo + tune_lograt) swept architecture/sampling knobs only. Production
regularizes implicitly (depth 3, subsample .7, colsample .7).
ARM: 24 candidates = production params (anchor) + 23 random draws over
reg_alpha ~ logU[0.01, 10], reg_lambda ~ logU[0.01, 10], gamma ~ U[0, 1.0],
min_child_weight ~ randint[1, 12] (jointly, holding n_estimators=300/depth=3/
lr=0.05/subsample=.7/colsample=.7 at production values — this isolates the
regularization axis; a joint re-tune is a different, bigger registration).
Scored by the tune_lolo standard: LOLO mean rho/ceiling, tau-gated at the best
observed tau; wmae/umae/dec3 reported per candidate (guards informational here —
selection is rho-based per tune_lolo precedent). 2 seeds x 4 folds.
RULE: a challenger replaces production params iff gated-rho beats the anchor by
> 0.005 (the tune_lolo adoption bar) AND its LOLO wmae is within +0.91% of anchor.
Adoption => retrain production seeds + P10-family re-verify (argmax within 0.2%).
Else: recorded as "explicit regularization adds nothing over the implicit recipe" —
closing the axis with evidence instead of assumption.

## D3C (registered 2026-07-12, BEFORE results; user: "there must be a way to include
## the data in such a way that everything improves")
PREMISE ADJUDICATED FIRST (honest): a single shared surface satisfying both
populations is CONTRADICTED by measurement (COMM-RESID-IV corrected: even
roll-direction offsets degrade aalto +1.6-8.5%). "Everything improves" is therefore
pursued via population-conditioned designs where aalto predictions are protected
structurally, not by luck:
ARM-R RESIDUAL HEAD (the guarantee-by-construction design): train the production
  aalto-only stack (frozen). For community folds, fit a residual XGB head (depth 2,
  100 trees, lr 0.1 — small by design) on the OTHER community labels' residuals
  (obs - frozen-prediction, LOGRAT space), applied only to community predictions.
  Aalto folds are BYTE-IDENTICAL to production by construction (gate (a) passes as
  a theorem). ADOPT iff held-out community-fold mean wmae improves > 1% vs frozen
  zero-shot AND community rho/ceiling does not fall on any fold. Adoption = the
  model gains a documented community-prediction mode; the layout-serving path
  (population=general) is unchanged.
ARM-F WEIGHT FRONTIER: merged single-surface training at community weight
  multipliers {1.0, 0.25, 0.05, natural(=no balance upweight)} on the corrected
  frame — maps the damage/gain frontier the single point ARM-W could not. Report
  per-multiplier aalto-fold mean delta + community-fold mean delta. INFORMATIONAL
  unless some multiplier achieves aalto within +0.91% AND community < -1% (the D3
  adopt rule), in which case it adopts under the D3 rule verbatim.
CONTEXT: R-D3B (ARM-P indicator + ARM-W x0.25, corrected data) is running under its
own registration; D3C complements it. If BOTH ARM-P and ARM-R qualify, ARM-R ships
(structural guarantee beats empirical clearance at equal gain).

### Outcome append (2026-07-12): R-D3B + D3C — no single-surface arm qualifies at ANY
### weight; the residual head delivers the community gain with aalto untouched but
### fails its rho clause on one fold; integration verdict UNCHANGED, frontier mapped
runs/rerun_d3b.json, runs/d3c_arms.json.
R-D3B (corrected data): ARM-W (x0.25) adopt=False — aalto qwerty +9.99/azerty +6.92%
  (dvorak actually IMPROVES -3.99%, echoing that community data helps the scarcest
  aalto fold); community mean -3.35%. ARM-P (population indicator) adopt=False —
  aalto qwerty +14.0/qwertz +9.07%; community mean -9.40% (the indicator amplifies
  BOTH sides: bigger community gain, bigger aalto damage — the column lets the model
  specialize but the shared trees still bend toward community pricing).
D3C ARM-F WEIGHT FRONTIER: mult 1.0 aalto +17.0% / comm -7.2%; 0.25 +4.8%/-3.2%;
  0.05 +0.0% mean BUT max fold +2.95% and guards FAIL / comm -1.3%; natural-weight
  +5.5%/-3.4%. The frontier never crosses adoptability: by the time community weight
  is low enough to spare the aalto MEAN, the worst fold and rare-cell guards still
  breach and the community gain has shrunk to near the bar. CONFIRMS: no
  single-shared-surface mix qualifies at any weight on this data.
D3C ARM-R RESIDUAL HEAD: community folds improve mean -7.5% wmae (alite -1.5%,
  mtgap -9.8%, ddn -17.9%, vg -0.8%) with rho UP on 3 of 4 folds and aalto folds
  byte-identical to production BY CONSTRUCTION. Registered adopt clause fails on the
  letter: alite's rho slips 0.657->0.654 (0.003, within seed noise but the clause
  said "does not fall on any fold"). Recorded as REJECTED-BY-LETTER / mechanism
  -proven. The design answer to "include the data so everything improves" EXISTS
  (two-surface: frozen general-population model + community residual head for
  community prediction), but per the registered rule it does not ship this round;
  a re-registration with a noise-aware rho clause (fall > seed-noise floor) would
  be a goalpost move if done after these results — parked for the next round with
  the flag that the current result already shows the direction.
NET for the user's question: "everything improves" is achievable ONLY by
population-conditioned architecture (ARM-R shape), never by weight/indicator mixing
on one surface — now measured at 4 points of the frontier plus 2 indicator designs.

## WPM-HARM (registered 2026-07-12, BEFORE results; user: "could the discrepancy be in
## the pace label being fed as a feature?")
MECHANISM UNDER TEST: the wpm label enters twice (feature + LOGRAT target normalizer)
and its SEMANTICS differ by dataset — aalto: correct chars / elapsed first-to-last
correct press (corrections INCLUDED in elapsed; keystrokes.py); community: correct
chars / sum of correct intervals only (corrections EXCLUDED; community.py::_session_wpm).
Community labels are therefore inflated for the same true pace; predicted magnitude
10-25% (error rates 5-12% + correction keystrokes + into-error slowdowns x1.5-2.3)
matches the observed level miss (slope 1.19). The commensurability audit subagent is
measuring the ratio distribution independently (no model contact); THIS registration
covers the model-contact adjudication.
HARMONIZATION: recompute each community session's wpm AALTO-STYLE — n_correct/5 over
elapsed minutes from first to last correct press (= sum of all intervals of events
after the first correct press through the last correct press, error and control
events included), n_correct >= 10 else drop (mirrors both pipelines' floors). Windows
re-extracted with the harmonized label (decode fix retained); cells rebuilt.
ARMS + RULES:
  HARM-LEVEL (D1-style zero-shot, frozen production models, 4 primary labels +
    pooled): report per label rho (expect ~unchanged — ordering is level-immune),
    mean signed relative residual (obs-pred)/pred (expect positive BEFORE, near-zero
    AFTER if the mechanism is right), wmape, slope. The label-semantics mechanism is
    CONFIRMED as the level-shift driver iff pooled |mean signed residual| falls by
    >= 50% AND slope moves into [0.95, 1.08] AND wmape falls >= 30% relative. Partial
    movement => mixed attribution (label semantics + population), quantified by the
    residual fraction removed.
  HARM-D3 (LODO-8, rules VERBATIM from D3/R-D3: adopt iff every aalto fold wmae
    within +0.91% of incumbent AND community-fold mean improves > 1% AND umae/dec3
    guards <= +2%): re-run with harmonized labels. If adopt=True, INTEGRATION
    REOPENS (the R-D3 rejection is re-attributed to label incommensurability) and a
    production-ingest change (aalto-style wpm for community) ships with a full
    family re-verify. If adopt=False with materially shrunken aalto damage, record
    the split attribution (labels X%, population the rest). If damage unchanged,
    label semantics is exonerated for the merge poison (level channel only).
HONEST PRIORS: HARM-LEVEL confirm 🟡 (direction+magnitude both fit); HARM-D3 adopt
🟠 (the class-price divergence evidence — COMM-RESID-IV offsets degrading aalto —
was bucket-centered i.e. level-immune, so SOME population divergence survives
harmonization; whether the residual damage clears +0.91% is genuinely open).
DELIVERABLE EXPOSURE: none directly (the argmax never consumed community labels);
this adjudicates validation claims + the integration door.

### Outcome append (2026-07-12): WPM-HARM — mechanism REFUTED as the level/merge
### driver; label semantics measured SMALL and harmonization makes things (slightly)
### WORSE; the population attribution stands
runs/wpm_harm.json.
MEASURED LABEL BIAS (registered/harmonized wpm ratio, median per label): x1.00-x1.14 —
far below the predicted 10-25% for most labels: recurva x1.000 (0%!), vg-custom
x1.015, mtgap x1.022, alite x1.072, castro x1.090, GK colemak-dh x1.140. The two
biggest community typists barely differ because their monkeytype error+correction
time is a small share of elapsed (short bursts, ctrl-backspace ~1ms corrections),
unlike aalto's sentence-typing pauses.
HARM-LEVEL: harmonization does NOT fix levels — pooled |signed resid| WORSENS 0.030
-> 0.077, wmape 0.221 -> 0.249, slopes scatter (1.24/1.17/0.94/0.80 vs 1.19/1.17/
0.87/1.02 before). The before-arm's signed residuals were ALREADY near zero on 2 of
4 labels (colemak-dh +0.017, castro -0.001) — i.e. the "level miss" is NOT a uniform
population-slowness the label could explain; the slope>1 pattern is a RANGE
compression (fast cells predicted too slow, slow cells too fast within a label), an
ordering-adjacent shape issue, not a label-scale issue. Rule verdict: mechanism NOT
confirmed (0 of 3 clauses fire).
HARM-D3: aalto poison essentially UNCHANGED (qwerty +33.9%, qwertz +26.3%, azerty
+20.6%, dvorak +5.3% vs R-D3's +24.8/+16.5/+19.6/+1.3 — if anything worse) and the
community-side gain SHRINKS (-3.7% vs -7.5%): the registered label was closer to
what the model needs than the harmonized one. Label semantics is EXONERATED for the
merge poison. adopt=False.
CONSEQUENCE: the user's pace-label hypothesis is now measured-and-closed: the wpm
semantics difference is real but x1.0-1.14 (not 10-25%), its removal does not
improve any transfer metric, and the merge poisoning survives harmonization intact
=> the population price-divergence attribution (COMM-RESID-IV) remains the standing
explanation. The wmape~0.25 level miss decomposes as range-compression (slope>1
within labels) + per-typist idiosyncrasy, neither label-fixable. Community wpm stays
as registered (correct-interval semantics) — it is the better-behaved label on this
capture (D3 community folds prefer it by 3.8pp).

## P10.5 (registered 2026-07-12, BEFORE results; user: "just for the sake of
## experimentation, what would be our P10.5 if we did use the best model trained on
## the entire data? Train it the same way P13 was trained; what is the cross regret
## with this new layout and P10-w0.5?")
STATUS: EXPLORATORY BY DECLARATION — the merged model failed every integration gate
(R-D3/D3C), so P10.5 is NOT a deliverable candidate; it is the measured answer to
"where would the argmax move if we let the community data in anyway."
TRAINING DATA: aalto bistrokes_v5 + ALL natural community labels from the corrected
TSV (7 typists incl the qwerty stubs; +pseudo/+rareboost EXCLUDED — non-natural
text by registered tag; octahedron contributes 0 in-band cells). Production recipe
verbatim (LOGRAT, practice term, layout_balance_weights, calibration OFF per
CAL-REMOVE), 3 seeds -> bigram_merged_seed{0,1,2}. Trigram surface: the production
join models (community trigram cells too thin to retrain — limitation recorded;
both surfaces share it, so the A/B isolates the bigram-surface difference).
REFERENCE SURFACE: aalto-only nocal (bigram_nocal_seed{0,1,2}, same recipe) + same
trigram — the current-production construction.
SEARCH (P13 recipe verbatim, rng 886777): SA 10x12k + exhaustive 2-opt on the
MERGED T3c; arms w_g in {0, 0.5, 1.0, 2.0} (exact-genkey in-loop, oxey 0) +
combined (w_g 0.5, oxey 0.5); candidate set = the 5 searched + P10-w0.5 + P11-w0.5
+ semimak/colemak/dvorak.
PICK RULE (P13 verbatim): min max normalized regret over {speed(merged surface),
genkey} s.t. merged-speed regret <= 0.5%. The pick is named P10.5.
CROSS-REGRET REPORT (the user's question): (a) P10.5's regret vs P10-w0.5 under the
AALTO surface; (b) P10-w0.5's regret vs P10.5 under the MERGED surface; (c) shared
positions; (d) gauge board (genkey/oxey/sfb/alternation/rolls/max-finger) for both.
INTERPRETATION RULE (registered): if both cross-regrets < 0.2% the merged data
does not move the argmax beyond the plateau even when trained in (consistent with
COMM-RESID-IV's -0.018%); if P10-w0.5's regret under the merged surface > 0.5%
the community data materially reprices the space and P10.5 documents the
enthusiast-leaning alternative — still not promoted (integration gates failed),
but recorded as the "if you trust the merge" layout.

### Reconciliation note (2026-07-12): audit-commensurability report vs the WPM-HARM
### empirical adjudication — the audit's measurements stand, its attribution falls
The commensurability audit (state/audit-commensurability/report.md) independently
measured axis A (wpm semantics) as the only construction difference large enough to
matter: window-weighted k = 1.102 (primary-4) / 1.276 (colemak-dh); aalto's own
dead-time factor 1.174 (the asymmetry is formula, not behavior); axes B/C/D null
(BUF2 8% of windows at x1.034; corpus coverage 0.937 vs 0.944; both clocks 1ms and
the aalto-16ms claim DISPROVEN; hold never read). Axes E/F quantified as amplifiers
(community = 50% of training mass at mult 1.0; ddn examples weigh 24.4x qwerty's;
84x density gap). All of that is measurement and STANDS 🟢 — the density/weight
numbers in particular sharpen the R-D3 attribution and motivate H3.
Its axis-A ATTRIBUTION (predicting slope 1.19 -> 1.02-1.10, wmape 0.27 -> 0.15-0.20,
merge poison shrinking under H1) was registered 🟠 with WPM-HARM as the decisive
test — and WPM-HARM refutes it:
(a) The "level shift" the mechanism must produce is a UNIFORM obs/pred offset;
    measured before-state corpus-weighted signed residuals were ALREADY ~zero on the
    two largest-k labels (colemak-dh +0.017, castro -0.001). Slope 1.19 with mean
    residual ~0 is RANGE COMPRESSION (predictions too flat on the new population),
    not a level shift — the audit read the OLS slope as a shift; the shift is absent.
(b) The model's effective wpm-response at community label coordinates is far weaker
    than the audit's within-cell qwerty beta=0.40: colemak-dh windows moved
    log k ~ 0.228 under harmonization, the mean residual moved only ~ -0.048 =>
    beta_eff ~ 0.79 (surface saturates at high wpm where aalto data thins). The
    label barely steers the surface where community mass sits.
(c) HARM-D3: the merge poison did NOT shrink under harmonization — it grew (qwerty
    +33.9% vs +24.8%), and the community-side gain halved. Mechanism reading 🟡: the
    registered (inflated) labels were mildly SHIELDING aalto by displacing community
    examples toward higher-wpm coordinates, away from aalto's dense mass;
    harmonization relocates the divergently-priced community targets onto aalto's
    most-populated region => more interference. Label semantics were not the poison;
    they were weak insulation.
NET: H1 is adjudicated (harmonization does not ship; the registered community wpm
stays); H2/H5 remain hygiene options; H3 (typist-count mass) is subsumed by the
already-measured ARM-F frontier (mult 0.05 = the fix, still fails guards); the
residual head remains the only everything-improves design. The audit's lasting
contributions: the E/F leverage quantification, the axis-B/C/D nulls, the
aalto-16ms-quantization disproof, and the H5 freq-column inconsistency (community
freq = corpus count vs aalto v5 = occurrence count) — H5 is a real cross-dataset
metric-weighting inconsistency to fix at the next ingest touch.

### Outcome append (2026-07-13): P10.5 — the merged-model argmax is ONE MUTUAL SWAP
### CLUSTER from P10-w0.5; cross-regret symmetric at ~0.10%; the community data does
### not move the argmax even when trained in
runs/p105_merged.json (models bigram_merged_seed{0,1,2} banked; rng 886777).
FAMILY (merged surface, P13 recipe): g0 gvldkqfouyrsthc.naiexzbmpw,j/; | g0.5
gdlmk.fouyrsthc,naiezjvwxpbq;/ | g1 cdgmk.fouyrsthl,naiezjvwxpbq;/ | g2
hgckv.fouylrstd,naiezjwmxpbq;/ | combined clgmkqfouysrthd.naiezxbwvp,;/j.
PICK (registered rule, axes {merged-speed, genkey}, cap 0.5%): P10.5 = the combined
arm, clgmkqfouysrthd.naiezxbwvp,;/j (max regret 3.03% — the genkey axis dominates
as in P13).
THE USER'S CROSS-REGRET: P10.5 under the AALTO surface +0.106% vs P10-w0.5;
P10-w0.5 under the MERGED surface +0.101% vs P10.5 — SYMMETRIC and both deep inside
the ~0.2% plateau. 18/30 shared positions; identical home row core (srth|naie),
identical alternation 76.0%, identical max finger 16.7%; P10.5 trades sfb 0.74->1.02%
and genkey 33.7->34.7 for a marginally better merged-surface score. vs qwerty:
P10.5 +3.90% (merged) / +3.81% (aalto); P10-w0.5 +3.81% (merged) / +3.91% (aalto) —
each layout wins its home surface by ~0.1%, the definition of plateau-equivalent.
CONSEQUENCE (registered interpretation rule, first branch fires): both cross-regrets
< 0.2% => even TRAINING the community data in (all natural labels, 50% of balanced
mass) does not move the argmax beyond the plateau. This completes the triangulation:
COMM-RESID-IV (offsets: -0.018%), D3C (no mix qualifies), and now the full merged
retrain (+0.10%) all land the same place — the community data changes predictions
measurably but the OPTIMAL LAYOUT is invariant to it. P10-w0.5 stands; P10.5 is
banked as the documented "if you trust the merge" sibling (it is NOT promoted —
the merged model failed every validation gate).

### Outcome append (2026-07-13): REG-LOLO — ADOPTED, all three gates; explicit
### regularization (high gamma) was the one lever the transfer-scored tuners never swept
runs/reg_lolo.json + runs/reg_lolo_verify.json.
SWEEP: 24 candidates (production anchor + 23 draws over reg_alpha/reg_lambda ~
logU[0.01,10], gamma ~ U[0,1], min_child_weight ~ [1,12], architecture pinned at
production). Winner: reg_alpha 0.141, reg_lambda 0.011, gamma 0.957, mcw 4 — gated
rho/ceiling 1.0236 vs anchor 1.0174 (+0.0062 > 0.005 bar). SIGNAL, not fluke: all
top-8 candidates carry gamma 0.75-0.96 with otherwise scattered alpha/lambda — split
PRUNING is the missing regularizer; the implicit recipe (depth 3, subsample .7) left
transferable headroom the CV-MAE tuner could never see.
GATE (i) wmae: mean 9.67 -> 9.76 (+0.89% <= 0.91% — passes by letter; per-fold
qwerty/qwertz +2.1%, azerty +1.5%, dvorak IMPROVES -0.98% with dec3 -2.9%): the
regularized model trades a hair of in-family fit for cross-family structure — the
right direction for a transfer instrument, and consistent with the rho gain being
real. GATE (ii)/(iii) argmax: re-search pick glmpk.,oyusrthdcnaiezjwbvfxq;/ (17/30
shared); P10-w0.5 regret under the regularized surface -0.009%, pick regret under the
old surface +0.019% — plateau-invariant both ways.
PRODUCTIONIZED: _DEFAULT_PARAMS in xgboost_model.py gains gamma/reg_alpha/reg_lambda/
min_child_weight (commit alongside this outcome); bigram_reg_seed{0,1,2} banked in
keybo-e2e/models/. P10-w0.5 numbers unchanged (+3.83% under the regularized surface
vs +3.91% under the old — same plateau). NOTE the challenger's rho edge is a
VALIDATION-instrument improvement; the deliverable claims do not change.

## P13-STAB (registered 2026-07-13, BEFORE results; user: "if you rerun P13 with the
## new model a few times, do we always get this same model? If not, which of the
## variants is the best?")
PROTOCOL: 5 independent full P13-recipe reruns (rng 888001..888005) on the REGULARIZED
surface (bigram_reg_seed{0,1,2} + production join trigram, wpm 90). Arms per rerun:
w_g in {0, 0.5, 1, 2} with the exact-genkey port in-loop, + combined (w_g 0.5, oxey
0.5). Registered deviation for tractability: the combined arm's IN-LOOP oxey term uses
the fast position-table approximation (exact oxey at ~61ms/eval would cost ~8h for 5
reruns); ALL pick scoring and the final board use the exact scorers. SA 10x12k +
exhaustive 2-opt, verbatim otherwise.
PER-SEED PICK (P13 rule verbatim): candidate pool = the seed's 5 arms + P10-w0.5 +
P11-w0.5; min max normalized regret over {speed(reg surface), genkey(exact)} subject
to speed regret <= 0.5%.
STABILITY REPORT: number of distinct per-seed picks; number of distinct searched
layouts; per-arm objective spread; pairwise shared-position counts.
GLOBAL ADJUDICATION ("which variant is best"): pool ALL distinct searched layouts
from all 5 reruns + P10-w0.5 + P11-w0.5 + P10.5; apply the same rule once; report the
winner with the full gauge board (speed on regularized AND pre-regularization
surfaces, exact genkey, exact oxey, sfb/alternation/rolls/max-finger/home shares).
CONSEQUENCE RULE: informational. P10-w0.5 remains the deliverable unless the global
winner strictly dominates it on BOTH pick axes AND by > 0.2% speed — in which case it
is FLAGGED for a user promotion decision (never auto-promoted). Expected (honest
prior, from every prior rank-stability round): picks vary at the letter level within
the ~0.2% plateau; the rule's job is to name the best-of-plateau, and P10-w0.5 has
won that adjudication in three prior families.

### Outcome append (2026-07-13): P13-STAB — 5 seeds -> 5 DISTINCT picks (plateau
### degeneracy confirmed on the regularized surface); global rule names a
### genkey-improved variant; NO promotion (rule's domination clause not met)
runs/p13_stab.json.
STABILITY: 5/5 per-seed picks distinct; 23/25 searched layouts distinct; pairwise
shared positions among picks min 0 / median 7 / max 18. The answer to "do we always
get the same model" is decisively NO at the letter level — the optimum is a wide
degenerate plateau, as in every prior family. STRUCTURE is what repeats: consonant
home-left (sthd/nt cores), naei-style vowel home-right, e-on-ring/i-on-pinky. One
seed (888005) landed exactly P10-w0.5 up to a rare-corner shuffle (q<->z, /<->;,
26/30 shared, <0.4% corpus mass) — P10-w0.5 is a recurring attractor.
GLOBAL ADJUDICATION (P13 rule over all 23 + refs): winner s888001-g1.0
  rcgkmq.ouylsthd,naeixwbfvpjz;/  — max regret 0.16%.
Board: speed +3.71% reg / +3.78% old surface (P10-w0.5: +3.83/+3.91 — winner is
0.12% SLOWER); genkey 31.0 (SEARCH-BEST EVER, vs P10-w0.5 33.7, approaching
graphite 29.5); sfb 0.67% (< P10 0.74%); alternation 77.3%; max finger 16.7% (tie).
The regularized surface lets the g1.0 arm buy 2.7 genkey points for 0.12% speed —
a better community-gauge trade than any prior family member.
CONSEQUENCE (rule fires as registered): promotion flag FALSE — the winner does NOT
strictly dominate P10-w0.5 (it loses the speed axis), so P10-w0.5 REMAINS the
deliverable. The winner is banked as the "community-leaning plateau member" —
the best genkey score achievable within the 0.5% speed cap on the current model —
available if the owner ever weighs the genkey axis higher than the rule does.

## OXL2-GAUGE (registered 2026-07-13, BEFORE results; user: "we should include
## oxeylyzer-2 into our project and considerations" — discharges the registered
## "oxeylyzer parity pass" follow-up)
TOOL: github.com/o-x-e-y/oxeylyzer-2 (Rust workspace; cloned to
~/gk-parity/oxeylyzer-2 alongside the genkey/keymeow harnesses). Our in-repo
OxeyStyleScorer is a DOCUMENTED APPROXIMATION of oxeylyzer-1 heuristics; this adds
the real successor tool as an exact external gauge, same standing as the exact
genkey port and the kmrun keymeow harness.
MECHANICS: our layouts exported as .dof files (ansi board, traditional fingering,
apostrophe passthrough at the untouched 11th home slot — same convention as the
tool's own Colemak.dof); scored via the shipped repl's `analyze` on the tool's OWN
default corpus (data/english.json) and default analyzer-config weights — i.e. the
numbers a community member would get running the tool unmodified. Candidates:
P10-w0.5, P13STAB-winner, P11-w0.5, P10.5 + the tool's own reference layouts.
REPORT: score + sfbs/sfs/stretches + trigram categories per layout; a parity table
vs our OxeyStyleScorer approximation and keymeow (where metrics overlap
definitionally); deviations noted per metric.
CONSEQUENCE RULE: informational gauge addition — joins the multi-gauge board used
by pick rules in FUTURE registrations; no existing verdict, objective term, or the
deliverable changes from this registration. If the exact tool ORDERS our finalists
differently than our approximation does, that discrepancy gets its own follow-up
entry (the approximation's weights/classes would need recalibration before any
future pick rule leans on the oxey axis).

### Outcome append (2026-07-13): OXL2-GAUGE — exact oxeylyzer-2 board landed; ordering
### of our finalists AGREES with our approximation; community tools stay internally
### consistent (they rank each other's layouts above ours, on every axis they define)
Harness: ~/gk-parity/oxeylyzer-2 (cloned, cargo release build; our layouts exported
as .dof: keybo-P10-w05, keybo-P13STAB-win, keybo-P11-w05, keybo-P105). Tool defaults
(english.json corpus, shipped weights). Raw analyze + rank output banked in
keybo-e2e/runs/oxl2_gauge.txt.
SCORES (higher = better under the tool): P13STAB-win -245B > P10-w0.5 -261B >
P10.5 -299B > P11-w0.5 -323B. References: smudge -8B (tool's best), semimak-jq
-188B, graphite -199B, colemak-dh -220B, octa8-angle -393B.
PARITY (the registered check): the exact tool's ordering of our finalists MATCHES
both our OxeyStyleScorer approximation (winner -15.3 < P10 -4.8 < P11 +0.9, more
negative better) and exact genkey (31.0 < 33.7 < 41.1) — three independently
implemented community gauges, one ordering. The approximation is fit for the
multi-gauge board; no recalibration follow-up needed.
DETAIL (tool metrics, P10-w0.5 vs P13STAB-win): sfbs 0.766% vs 0.698%, sfs 3.733%
vs 3.732%, stretches 39.5 vs 42.7 (the one axis P10 wins — the winner's r-on-pinky
costs stretch mass), trigram alternate 17.6% vs 18.3%, redirect 2.74% vs 2.46%.
NOTE the tool's trigram "Alternate 17.6%" differs from our 76% definitionally (it
buckets sft/sfb-containing trigrams separately and weights by its own corpus) —
definitions reconciled in the D2b-era alternation write-up; not a discrepancy.
STANDING: oxeylyzer-2 joins genkey (exact port) + keymeow (kmrun) as the third
exact community gauge for future pick rules. Community-tool verdict on the
campaign is unchanged and now triple-confirmed: our layouts trade community-metric
score for measured-time optimality; P13STAB-win is our best community-facing
member on ALL THREE tools simultaneously.

### OXL2-GAUGE addendum (2026-07-13): the LSB-vs-stretches question adjudicated from
### source + oxeylyzer v1 run; v1 gauge added; winner CONFIRMED better on both versions
Q (user): oxeylyzer web shows Lsb 2.674% (P10-w0.5) vs 1.251% (winner), yet the
stretches number favors P10 (39.5 vs 42.7) — contradiction?
A: NO — three different metrics, all now verified from source:
(1) "Lsb" (the web UI stat / v1 "Lsbs") = a COUNT metric: corpus share of
    middle<->index same-hand pairs with |dx| >= 1.5u (v1 fast_layout.rs:650-660).
    The winner IS better: our keymeow lsb 0.60 vs 0.09, v1 Lsbs 1.325% vs 0.708%.
    (The user's web numbers come from a different corpus/board config; same
    ordering.)
(2) o2 "stretches" = a WEIGHTED-DISTANCE metric over ALL same-hand diff-finger
    pairs: sum of corpus-weighted stretch = dist + x_overlap - 1.35*finger_gap
    over pairs where that exceeds 0 (o2 cached_layout.rs:160-181). It counts
    pinky/ring geometry that the LSB count ignores. The winner's r-on-top-pinky
    (ey/ye/rs/cl pairs; repl `stretches` listing banked) costs stretch-distance
    while carrying near-zero classic-LSB mass. Both statements are true:
    winner has FEWER lateral index stretches, MORE total stretch-distance.
(3) v1 "Stretches" is yet another formula (score-space, sign-flipped display:
    -15.97% P10 vs -12.97% winner — LESS negative = better, so v1 says the
    WINNER is better on ITS stretches too; the o2 disagreement on this one axis
    is an o2-specific weighting).
V1 vs O2 (differences, from source + configs): v1 = richer metric set (scissors,
LSBs, pinky-ring, alternates-sfs, bad-redirects split, finger speed) with weights
incl lateral_penalty/pinky_ring; o2 = early-development rewrite — collapsed metric
set (sfbs/sfs/stretches + trigram categories), trigram weights parsed but UNUSED
in scoring (README admits), the stretch formula above replacing scissors+LSB+
pinky-ring as one blended distance term, libdof board geometry, new corpus format.
V1 VERDICT ON THE FINALISTS (shai corpus, tool defaults; raw output banked in
runs/oxl2_gauge.txt): winner beats P10-w0.5 on v1 Score 0.367 vs 0.333 and on
sfb (1.051 vs 1.153), finger speed (-2.60 vs -2.88), stretches, LSBs (0.708 vs
1.325), pinky-ring, total redirects (4.84 vs 5.43), bad sfbs, total alternates;
P10-w0.5 keeps rolls (42.2 vs 41.6) and scissors (0.140 vs 0.164). FOUR community
tools (genkey, keymeow, oxeylyzer-1, oxeylyzer-2) now agree: the P13STAB-winner
is our best community-facing layout; P10-w0.5 remains the measured-speed pick.

## FEAT-CT — community-tool geometry as model features (registered 2026-07-13, BEFORE results)
Audit of genkey/keymeow/oxeylyzer-1/oxeylyzer-2 vs our 20-feature schema found four
per-bigram signals the model CANNOT currently express (all others are present, tree-
derivable, or layout-level aggregates that belong in the objective, not the model):
  1. stretch  — o2's continuous stretch residual max(0, dist + x_overlap - 1.35*finger_gap)
     (cached_layout.rs:160-181 port; box-collapse to key centers, flen y-adjust,
     signed-dx crossing rule) — the user's direct ask.
  2. x_overlap — the splay/crossing term alone (max(0, xo(f1,f2) - 1.3*sdx + dy/3)).
  3. finger_gap — |finger_index1 - finger_index2| in {0..3}; today only gap==1
     (adjacent) is visible; gap 2 vs 3 indistinguishable.
  4. pinky_ring — same-hand pinky<->ring flag (v1 metric; not exactly derivable since
     the first key's finger is not a feature).
ARMS (bigram, bistrokes_v5, production recipe: LOGRAT + practice k=100 x2 + layout
weights + current _DEFAULT_PARAMS incl REG-LOLO): ANCHOR / +STR / +XO(x_overlap,
finger_gap) / +PR / +ALL4. 4 LOLO folds x 2 seeds.
DECISION RULE (same standard as FEAT-LR a524792): an arm QUALIFIES iff wmae_rel
< -1% AND umae_rel <= +2% AND dec3_rel <= +2% AND min decisive-pair tau AND min
all-pair tau not below ANCHOR's. Winner = qualified arm with lowest wmae.
CONSEQUENCES: no qualifier => feature set stands, negative result recorded. A
qualifier => adoption chain: schema edit + FEATURE_VERSION bump + production
retrain (3 seeds) + argmax stability check (P10-family regret within 0.2% on the
new surface); argmax break => surface the tension to the user, no silent pivot.
Secondary (registered as exploratory): same +STR columns on the trigram bg1_/bg2_
blocks (trifeat harness conventions), same rule — reported, adoption only with the
bigram result's consistency.

## P14 — five-gauge co-optimization: speed + genkey + oxey1 + oxey2 + WFD
## (registered 2026-07-13, BEFORE results)
GOAL (user): search for a layout better than BOTH P10-w0.5 and P13STAB-win when the
community tools are IN the objective rather than post-hoc gauges.
IN-LOOP TERMS (all frequency x position-table, built once):
  speed = T3c on the regularized surface (bigram_reg_seed* + trigram_cond_lograt_join*,
  wpm 90) — weight 1 always. genkey = exact GenkeyScorer port. oxey2 = port of o2
  score_cache = weighted_bigrams + stretch_bigrams (analyzer-config.toml weights:
  sfbs -7 sfs -1 stretches -3, finger weights 77/32/24/21; o2 english.json corpus
  restricted to our charset + pinned apostrophe). oxey1 = port of v1
  score_with_precision (trigram top-1000 term + fspeed + pinky_ring + stretch;
  live config.toml weights; v1 english.json; usage term inert at penalty=0).
  wfd = o2 weighted_bigrams total alone (pure finger-weighted same-finger travel).
PARITY GATES (must pass BEFORE any search uses a port; else that gauge drops to
post-hoc-exact-only, recorded): oxey2 port vs repl `rank`/`analyze` on >= 8 layouts
(incl qwerty, dvorak, colemak, semimak, graphite, our finalists): Spearman rank
corr = 1.0 on the set and per-layout ratio spread <= 5% after one global scale.
oxey1 port vs repl analyze Score on the same set, same gate. wfd is implied by
oxey2 score + stretches parity (score - stretches = wfd), recorded not re-gated.
SEARCH: SA 10x12k + exhaustive 2-opt (p13 recipe), fit = speed + sum_g w_g * UNIT_g
* loss_g with UNIT_g = (speed_q/100)/|loss_g(qwerty)|; loss form: genkey=fitness,
oxey1=-score, oxey2=-score, wfd=-total. ARMS (w over genkey/oxey1/oxey2/wfd):
E025=all 0.25, E05=all 0.5, E10=all 1.0, GK1=(1,0.25,0.25,0.25), OX1=(0.25,1,1,0.5).
RNGs {888101, 888102, 888103} => 15 searches.
PICK RULE: pool = all searched + {P10-w0.5, P13STAB-win, P11-w0.5, P10.5} + qwerty
ref. Speed gate: 100*(fit/best_fit - 1) <= 0.5 (as P13). Community regrets are
qwerty-gap-normalized (sign-safe): r_g = 100*(loss_g - min_loss_g)/
(loss_g(qwerty) - min_loss_g). Pick = min over gated pool of max(r_genkey, r_oxey1,
r_oxey2, r_wfd, r_speedgap) where r_speedgap uses the same qwerty-gap form.
CONSEQUENCES (registered): the pick is documented as a P14 candidate; it earns a
sibling doc iff it beats P13STAB-win on >= 3 of the 4 exact community tools (genkey
binary-parity port, keymeow kmrun, oxeylyzer-1 repl, oxeylyzer-2 repl) at speed
regret <= 0.5%; it is flagged for possible PROMOTION discussion (user-gated, never
autonomous) iff it additionally matches or beats P10-w0.5's speed within 0.1%.
Otherwise: negative result, both incumbents stand.

### FEAT-CT OUTCOME (2026-07-13, runs/feat_ct.json): NO qualifier — feature set stands
ANCHOR reproduced the production baseline exactly (rho/ceiling 1.0236 = the REG-LOLO
adopted number; wmae 9.76). Arms vs the registered gates (need wmae < -1%):
  +STR   wmae -0.31%  umae -0.40%  dec3 -0.37%  rho/ceil 1.0256  taus 1.0 — no
  +PR    wmae -0.31%  umae -0.42%  dec3 -0.28%  rho/ceil 1.0269  taus 1.0 — no
  +XO    wmae +0.11%  umae +0.02%  dec3 +0.24%  rho/ceil 1.0223  taus 1.0 — no
  +ALL4  wmae +0.07%  umae -0.17%  dec3 -0.32%  rho/ceil 1.0240  taus 1.0 — no
READING: stretch_resid and pinky_ring are DIRECTIONALLY positive (wmae -0.3%, gated
rho +0.002..+0.003) but inside the noise floor and far from the -1% bar; x_overlap adds
nothing. The o2 stretch geometry carries little predictive signal for typing time
beyond dx/dy/distance/lsb/scissor on the aalto LOLO folds. Registered consequence
taken: no schema change; the stretch axis remains objective-side only (P14).

### P14 OUTCOME (2026-07-13, runs/p14_coopt.json, runs/p14_parity.json)
PARITY GATES: both ports PASSED on 10 layouts (4 finalists + qwerty/colemak-in-our-
shape + 4 seeded shuffles): o2 Spearman 1.0, spread 0.00% (exact integer match x100);
v1 Spearman 1.0, spread 0.04% (full-trigram fix: the repl's displayed Score is
score_with_precision(usize::MAX), not top-1000 — found when the top-1000 port FAILED
at 17.4% spread/rho .988; both attempts recorded).
SEARCH: 15 searches (5 arms x 3 rngs) on the 0.45ms five-term fit.
PICK (registered min-max qwerty-gap regret, speed gate 0.5%):
  OX1-r888103 = lcgkvx.ou,rsthdynaeizwmpbfjq/;  (max regret 5.23%)
   l c g k v   x . o u ,
   r s t h d   y n a e i
   z w m p b   f j q / ;
  speed +0.19% vs P10-w0.5 (in-plateau); genkey 30.92 (new best; P13win 31.0);
  oxey1 repl 0.400 (P13win 0.367, P10 0.333); oxey2 repl -238.5B (P13win -245.1B,
  P10 -260.8B) — repl-verified, not just port numbers. Beats BOTH incumbents on
  max-regret (P13win 6.45%, P10 10.16%).
KEYMEOW (kmrun, added post-pick): sfb 1.231 / lsb 0.530 / alt 38.2 / roll 46.0 /
redir 4.68. vs P13STAB-win: WORSE sfb/lsb/alt, better roll/redir/sfs => keymeow
adjudicates AGAINST the pick.
REGISTERED CONSEQUENCE: beats P13STAB-win on 3 of 4 exact tools (genkey, oxey1,
oxey2; loses keymeow) at speed regret +0.19% <= 0.5% => EARNS SIBLING DOC
(docs/layout-p14-coopt.md). Promotion flag: NOT fired (speed not within 0.1% of
P10-w0.5). Both incumbents stand; P14-pick documented as the max-regret-balanced
community layout.
READING: the five-gauge objective found the balance point the post-hoc picks
missed — its worst community axis (5.2%) is better than P13win's (6.5%) and P10's
(10.2%). The cost is concentrated exactly where the oxey family and keymeow
disagree: oxey rewards the roll-heavy short-travel left block (rst home), keymeow
prices its sfb rise (1.07 -> 1.23). genkey is near-tied (30.92 vs 31.00).

## P14b — deep co-opt sweep + the keymeow axis (registered 2026-07-13, BEFORE results)
MOTIVATION: P14's pick beat P13STAB-win on genkey/oxey1/oxey2 but LOST keymeow
(sfb 1.23 vs 1.07) — keymeow prices sfb mass/travel, which no in-loop term carried.
Also the P14 search was shallow (15 shots) on a known-degenerate plateau.
NEW IN-LOOP TERM: sfbdist = sum over same-finger position pairs (index cols 1+2 one
finger; space excluded) of corpus bigram freq x euclidean key distance — our-corpus
proxy for keymeow sfb-dist (kmrun stays the exact post-hoc judge). UNIT as P14.
ARMS (weights over genkey/oxey1/oxey2/wfd/sfbdist):
  OX1r  = (0.25, 1, 1, 0.5, 0)    x rngs {888104..888109}  (P14 winner arm, 6 more rngs)
  SFB05 = (0.25, 1, 1, 0.5, 0.5)  x rngs {888104..888109}
  SFB10 = (0.25, 1, 1, 0.5, 1.0)  x rngs {888104..888109}
  SFB20 = (0.25, 1, 1, 0.5, 2.0)  x rngs {888104..888106}
SA 12 restarts x 16k iters + exhaustive 2-opt (deeper than P14's 10x12k).
PICK RULE: pool = all P14b searched + P14's 15 + {P10-w0.5, P13STAB-win, P14-coopt,
P11-w0.5, P10.5}. Speed gate 0.5% as before. Min max qwerty-gap regret over SIX
gauges: speed-gap, genkey, oxey1, oxey2, wfd, sfbdist.
CONSEQUENCES: pick verified on all four exact tools (genkey port, kmrun, o2 repl,
v1 repl). If it beats P13STAB-win on ALL FOUR (incl keymeow sfb AND lsb) at <= 0.5%
speed regret => it SUPERSEDES P14-coopt in docs/layout-p14-coopt.md (doc updated,
old pick kept as provenance) and is flagged for user decision. If it beats 3/4 with
a lower max-regret than P14-coopt => doc updated with it as the new balance point,
noted which axis it concedes. Else negative result recorded, P14-coopt stands.

### P14b OUTCOME (2026-07-13, runs/p14b_coopt.json): pick wins the six-gauge rule but
### FAILS the exact-tool supersede bar — P14-coopt STANDS
Six-gauge pick: SFB20-r888104 = pyu,.vgdnmhieaocstrlkj/q;fwbxz (max regret 3.61% vs
P14-coopt 5.23%, P13win 6.45%). Exact-tool verification:
  genkey 31.71 (WORSE than P13win 31.00 and P14-coopt 30.92)
  keymeow sfb 1.056/sfb-dist 1.146 (best of all four) BUT lsb 1.82/lsb-dist 3.96
    (catastrophic vs P13win 0.09/0.18 — keymeow adjudicates AGAINST on lsb)
  oxey2 repl -223.8B (best); v1 repl 0.397 (below P14-coopt 0.400)
Registered bar was: beat P13STAB-win on ALL FOUR exact tools (incl keymeow sfb AND
lsb) => supersede. It loses genkey, v1, and keymeow-lsb => NO supersede. Second bar
(3/4 + lower max-regret than P14-coopt) also fails on exact tools (only oxey2 +
keymeow-sfb won). REGISTERED CONSEQUENCE: negative result recorded; P14-coopt stands
as the documented balance point. READING: the sfbdist term worked as aimed (keymeow
sfb-dist 1.19 -> 1.15) but the optimizer paid with a huge lateral-stretch block —
in-loop proxies trade one keymeow axis for another; the exact-tool gate caught it.
The interior max-regret improvement (3.61%) is a proxy-pool artifact, not a
community-tool win.

### FEAT-CT SECONDARY OUTCOME (trigram +STR, 2026-07-13, runs/feat_ct_tri.json):
### no qualifier — trigram feature set stands too
bg1/bg2 stretch-residual columns on the JOIN conditioned frame: wmae +0.76%,
umae +0.75%, dec3 +0.15% (all WORSE than anchor), taus hold. Consistent with the
bigram result: the o2 stretch geometry adds no predictive signal for typing time.
FEAT-CT is fully closed — both registered arms negative, no schema change anywhere.

## P14c — o2-forward weight sweep (registered 2026-07-13, BEFORE results; user direction)
MOTIVATION (user): oxey2 is our furthest community axis (qwerty-gap 9.0% vs genkey
3.9%, oxey1 6.7%); test lowering oxey1 weight / raising oxey2. Also NOTED: max
finger load is DROPPED from all gauge boards (user: lower is not better; speed-
proportional usage is already priced in genkey fspeed + o2 finger weights). Known
context: E10 arm already reached o2 -210e11 (+0.31% spd) but pays oxey1 (v1 trigram
flow); ~6-8% of our o2 score is the PINNED APOSTROPHE (structural convention vs
semimak/graphite — NOT addressable by weights; separate user decision).
ARMS (weights genkey/oxey1/oxey2/wfd): O2H1=(0.25,0.25,2,0.5) [user's direction],
O2H2=(0.25,0.5,3,0.5), O2H3=(0.5,0.25,2,1), E10r=(1,1,1,1). x rngs {888110..888112}
= 12 searches, SA 12x16k + 2-opt.
PICK RULE: identical five-gauge min-max qwerty-gap regret as P14 (comparability),
pool = these + P14 + P14b searches + all incumbents. SECONDARY registered pick:
min oxey2 s.t. speed regret <= 0.5% ("o2-frontier candidate"), reported alongside.
CONSEQUENCES: same as P14b — a candidate supersedes P14-coopt in the doc iff it
beats P13STAB-win on >= 3/4 exact tools AND has lower max-regret than P14-coopt's
5.23%; the o2-frontier candidate is documented (not promoted) with its exact-tool
board either way. Else negative result; P14-coopt stands.

### P14c OUTCOME (2026-07-13, runs/p14c_coopt.json)
FIVE-GAUGE PICK over the union pool (48 searched + incumbents): p14b:SFB20-r888104
(max regret 5.21% — marginally under P14-coopt's 5.23% on the five-gauge form, but
it is the SAME layout P14b already rejected on the exact-tool bar: genkey 31.7,
keymeow-lsb 1.82, v1 0.397). Supersede bar NOT met => P14-coopt STANDS.
O2-FRONTIER (registered secondary): O2H2-r888111 = hrfmk,yuojlnstdgciaezxbvqpw.;/
  o2 repl -194.6B — BEATS graphite (-199.1B), 2nd only to semimak (-190.4B);
  closes 79% of the P10->semimak o2 gap. genkey 31.4; v1 repl 0.387;
  keymeow sfb 0.945/sfb-dist 1.116 (beats P13win AND graphite; only semimak
  better), lsb 0.67 ~ graphite 0.57; alt 41.5. Speed +0.35% (in-plateau).
  Regret profile: o2 0.0 / genkey 0.9 / wfd 1.3 / oxey1 6.0 / speed 8.7.
READING (user's question answered empirically): raising the o2 weight (O2H2 arm:
genkey .25 / oxey1 .5 / oxey2 3 / wfd .5) DID move us to the o2 frontier — the
distance on o2 was a weighting choice, not a capability limit; the price is
concentrated in oxey1 (trigram flow) and speed inside the plateau. Also ~6-8% of
our o2 score is the pinned-apostrophe convention (structural; semimak/graphite
place ' on a good key and drop ;or/ instead — a charset decision, user-gated).
DOC CONSEQUENCE (per rule): P14-coopt remains the balance point; the o2-frontier
candidate is documented alongside it in docs/layout-p14-coopt.md.

## K31 — the apostrophe joins the optimization keyset (registered 2026-07-13, BEFORE
## results; USER DIRECTIVE: "optimize on the same keyset, add the apostrophe")
DELIVERABLE CHANGE: the search space becomes 31 movable keys = the 30-key block +
the ANSI quote slot (x=+6, home row; right pinky). This levels the one structural
disadvantage vs semimak/graphite (they place ' on a good key; we pinned it) — ~6-8%
of our oxeylyzer-2 score and every apostrophe contraction in every community corpus.
Charsets still differ at the margin (graphite/semimak keep '-', drop ';'); the
common-subset convention continues to handle that; the ' (0.43% bigram mass, 5x
bigger than ; or /) is what levels.
PLAN (all steps registered; each with its own gate):
 A. Geometry/feature extension: ROW_STAGGERED_31 (30 slots + (6,2) APPENDED — every
    existing 30-char layout string extends by 1 char), column-6 finger = pinky,
    pinky one-hot extended to |x| in {5,6}, is_adjacent extended with the {6,4}
    pinky-ring pair. FEATURE_VERSION is NOT bumped: the extension is domain-only —
    GATE: a regression test proves every feature value on every 30-key position
    pair is bit-identical to the current pipeline (new branches cannot fire on
    |x|<=5). If that test cannot be made to pass, STOP and bump the version instead.
 B. Data: re-run the locked BUF2-BOTH extraction (p8_final stage 1 verbatim) with
    31-char maps — qwerty +' , dvorak +'-', azerty +'ù', qwertz +'ä' (each national
    layout's actual ANSI quote-slot char; off-charset chars still break windows as
    today) -> bistrokes31_v1.tsv / tristrokes31_cond_v1.tsv. GATE: restricting the
    new bigram table to non-quote-slot rows must reproduce v5's row count within
    0.5% (the extension may only ADD windows previously broken by ' interruptions
    — wait, ' rows previously did NOT break windows on qwerty since ' was
    off-layout=window dropped; the restriction check is: v5 rows are a subset,
    count delta explained by newly-valid windows CONTAINING '). Report the delta.
 C. Corpus: 1-skip31 derived from trigrams.txt (skipgram(a,c) = sum_b trigram(abc));
    GATE: on non-apostrophe pairs the derived table must rank-correlate >= 0.99
    with the existing 1-skip.txt. bigrams/trigrams already carry ' — used as-is.
 D. Models: retrain the production recipe on K31 tables -> bigram_reg31_seed{0,1,2},
    trigram_cond31_seed{0,1,2} (LOGRAT + practice k=100 x2 + layout weights +
    adopted REG-LOLO params). GATE: LOLO on the K31 bigram table (4 folds x 2
    seeds) must hold tau = 1.0 and rho/ceiling within 3% of the v5 baseline
    (1.0236); a tau break stops the migration (report, ask user).
 E. Search: P15 = the P14 five-gauge co-opt re-run on the K31 space (31! perms,
    same SA budget 12x16k, arms E10/OX1/O2H2/GK1 x 3 rngs {888201-3}) + a
    speed-only arm. All five gauges see the full 31 keys except genkey and keymeow
    (their models are 3x10 — the quote-slot char is invisible to them; convention
    NOTED on every board; oxey1/oxey2/wfd see all 31).
 F. Pick rule: identical five-gauge min-max qwerty-gap regret, qwerty31 =
    qwerty+' pinned reference, speed cap 0.5% (now on the K31 surface). Incumbent
    references enter the pool as <layout>+' (their K31 embedding).
CONSEQUENCES: the P15 pick becomes the project's K31 flagship candidate, documented
with full boards vs P10-w0.5+', P13STAB-win+', P14-coopt+', semimak, graphite.
PROMOTION of K31 over the 30-key P10-w0.5 as THE deliverable is a user decision
(one-way door: changes the published keyset); we present the evidence. Quality
(F5M-LR) surface retrain is DEFERRED (gauge reported as n/a on K31 boards until
retrained). All existing 30-key results remain valid history; K30 models keep
loading (no version bump, per gate A).

### K31 gates A-C PASSED (2026-07-13)
A (3cb1009): golden regression — all 30-key feature values bit-identical after the
  ROW_STAGGERED_31 extension (quote slot (6,2) appended; pinky|x|in{5,6}, lateral
  |x|in{1,6}, adjacent += {6,4}); full suite green; no FEATURE_VERSION bump needed.
B (runs/k31_extract.log): BUF2-BOTH re-extraction with 31-char maps (qwerty+',
  dvorak+-, azerty+ù, qwertz+ä): v5 EXACTLY reproduced on the non-quote domain
  (2111 rows, 29.31M occ, 0 delta — quote chars were off-layout window-DROPS before,
  so plain windows are untouched) + 91 quote-slot rows / 220k occurrences added.
  bistrokes31_v1.tsv + tristrokes31_cond_v1.tsv.
C (97e7588): 1-skip31 derived from trigrams; non-quote spearman 0.9993 vs existing.
Stage D (LOLO gate + retrain) running.

### K31 stages D-F OUTCOMES (2026-07-13, runs/k31_train.log, runs/p15_coopt.json)
D: bigram LOLO gate PASS — taus [1.0, 1.0], rho/ceiling 1.0135 (>= 0.97x baseline
   1.0236; the small dip is the new quote-slot rows entering the folds). Trigram
   sanity 0.9892 vs 0.9928 direct baseline, taus 1.0. Models saved:
   bigram_reg31_seed{0,1,2}, trigram_cond31_seed{0,1,2} (CAND4, BUF2 cond frame —
   frame deviation from the old JOIN frame recorded in k31_train.py docstring).
E/F: 15 K31 searches (SPD/E10/OX1/O2H2/GK1 x 3 rngs). K31 objective covers 22788
   trigrams (+apostrophe mass). PICK (five-gauge min-max qwerty31-gap regret,
   speed cap 0.5%): E10-r888203 = fyu,.vdpnlhieaocstrmk/;'qgwbxzj
     f y u , .   v d p n l
     h i e a o   c s t r m      + j on the quote slot
     k / ; ' q   g w b x z
   max regret 3.37% vs P14-coopt+' 7.83%, P13win+' 9.02%, P10+' 12.22%. Speed
   +0.09% off the K31-best. repl-verified: o2 -213.3B, v1 0.420. genkey 33.61
   (worse than P14-coopt 30.92 — regret is qwerty-gap-normalized, and its worst
   axis is still only 3.4%). keymeow: sfb 1.33, lsb 1.93 (adjudicates against).
KEY STRUCTURAL FINDING (the point of K31): every arm, including SPEED-ONLY,
   voluntarily exiles a rare letter to the quote slot and pulls ' into the block —
   SPD-r888202 = gcdlk.,yousrthmpnieaqxwbvf'j;/ + z, which is P11-w0.5 (modulo a
   top-left gc swap) with ' in z's old slot and z on the quote key, and it is
   FASTER than P10-w0.5+' on the K31 surface (P10+' regret +0.10%). The
   semimak/graphite convention (' in the block, rare letter exiled) is
   speed-optimal on our surface too, not just a community-tool trick.
   Also notable: the balance pick is a LEFT-vowel mirror (hieao home-left) — first
   time a pick breaks the naei-right invariant.
CONSEQUENCE (per charter): E10-r888203 is the K31 flagship CANDIDATE, documented
   with full boards. PROMOTION of K31 over 30-key P10-w0.5 as THE deliverable is
   presented to the user (one-way door: changes the published keyset). The
   speed-only K31 result (P11-family + ' swap) is documented alongside as the
   K31 speed pick.

## K30M — matched 30-key charset (registered 2026-07-13, BEFORE results; USER
## DIRECTIVE: "make our 30 keys match graphite and semimak, apples to apples")
SUPERSEDES the K31 exile structure as the deliverable frame (user: a letter pushed
to the quote slot is still a letter typed worse; K31's finding that the exile
convention is speed-optimal stays RECORDED but is not a product we ship). NOTE for
the record: K31 removed nothing from OUR objective (all 31 keys priced); the
removal artifact was in the 3x10 gauge tools' view. K30M fixes comparability at
the root instead: identical charset to the community entries.
CHARSET C30M = 26 letters + {' , . -} on the standard 30 slots; ; and / leave the
layout (exactly the kmrun/genkey semimak+graphite charset, where both agree; their
dof encodings disagree with each other — Semimak.dof keeps /, graphite.dof drops
, — recorded, kmrun convention chosen). Corpus mass: IN ' 0.434% + - 0.391%, OUT
; 0.038% + / 0.037% => objective coverage RISES ~0.75pp vs the old charset.
MODELS: the K31-trained surfaces are position-based and strictly-more-data — used
as-is (no retrain; registered). PRACTICE-TERM note: '/- ngrams have thin/absent
b-values; b is layout-independent and drops out of optimization (recorded).
P16 SEARCH: five-gauge co-opt on C30M, arms SPD/E10/OX1/O2H2/GK1 x rngs
{888301-3}, SA 12x16k + 2-opt. Gauges: genkey + keymeow DIRECT (same charset as
their semimak/graphite rows — true apples-to-apples, nothing invisible); oxey1/
oxey2/wfd via the parity-gated ports on a 31-key dof with ';' PINNED at the quote
slot (Semimak.dof's own convention). POOL: searched + semimak + graphite (kmrun
strings — now first-class rows on ALL gauges incl our speed surface) + incumbents
embedded by substitution (; -> ', / -> -; noted as embeddings, not the originals)
+ qwerty30M = qwerty with the same substitution as the regret reference.
PICK RULE: five-gauge min-max qwerty30M-gap regret, speed cap 0.5% (unchanged).
CONSEQUENCES: the pick is the K30M flagship candidate; boards vs semimak/graphite
are now exact same-charset comparisons. If the pick beats BOTH semimak AND
graphite on >= 3 of the 4 community tools while winning speed => flag for the
user as the first true apples-to-apples community claim. Deliverable promotion
remains user-gated.

### P16/K30M OUTCOME (2026-07-13, runs/p16_coopt.json)
FIRST TRUE APPLES-TO-APPLES: semimak + graphite as first-class rows on every gauge
including our speed surface. Headline rows (C30M charset, K31-trained models):
  semimak:  speed +1.18% BEHIND the frontier; genkey 27.7 (still king), o2-gap 0
  graphite: speed +1.36% behind; balanced community profile
  P10-w0.5* (;->' embedding): speed +0.07% (frontier), maxreg 12.9%
PICK (five-gauge rule): E10-r888303 = frlwg'uyoksntdc.ieahvxmpb,-jqz (maxreg 6.38%)
   f r l w g   ' u y o k
   s n t d c   . i e a h
   v x m p b   , - j q z
  vs semimak EXACT tools: genkey 30.8 vs 27.7 (semimak), v1 0.415 vs 0.365 (PICK),
  o2 -234.1B vs -190.4B (semimak), keymeow sfb 1.29 vs 0.89 (semimak), lsb 1.27 ~
  1.27 (tie-ish) => pick wins oxey1 + speed (+0.96pp!), semimak wins genkey/o2/
  keymeow. The registered "beats BOTH on >=3/4 tools" flag does NOT fire.
  vs graphite: pick wins v1 (0.415 vs 0.460 — NO, graphite wins v1), wins o2? NO
  (-234.1 vs -199.1 graphite). Graphite beats the pick on v1+o2+keymeow; pick wins
  genkey narrowly + speed +1.14pp.
READING: with the charset finally matched, the honest statement is: our layouts
hold a ~1.0-1.4pp measured-speed advantage over semimak/graphite at community
scores that sit BETWEEN dvorak and the community frontier — but semimak/graphite
remain better pure community-metric layouts. The K31/P15 apparent "beat semimak"
on o2 came from the pinned-';' dof convention difference, now eliminated.
SPD arm: koyu,vdmnlheiapcstrfqj-.'gwbxz — speed frontier on C30M (+0.07pp over
P10*), naei-family right-hand vowels RESTORED (heia home-right): the K31 left-
vowel flip was an artifact of the extra key, not a stable optimum.
CONSEQUENCE: no community-claim flag. C30M candidates documented; deliverable
decision (stay K30-classic vs adopt C30M charset) presented to the user.

## SEL-1 — selection-rule methodology study (registered 2026-07-13, BEFORE results)
QUESTION (user): is min-max worst-axis the right way to pick "the balanced one"?
SCOPE: pure post-hoc ANALYSIS of the existing P16/C30M pool (no new searches). The
REGISTERED P16 pick stands regardless; output = a recommendation + robustness
evidence for FUTURE pick rules; any rule change applies from the next search
onward (no re-crowning without user decision — goalpost discipline).
RULES COMPARED (all under the identity speed cap <= 0.5%):
  R1 min-max, qwerty-gap normalization (current registered rule)
  R2 min-max, pool-range normalization
  R3 min-max, rank normalization (normalization-free)
  R4 mean regret (L1 scalarization)
  R5 L2 distance-to-ideal (compromise programming; R1 is the L-inf member)
  R6 Borda (mean per-gauge rank)
  R7 random-preference win rate: 20k Dirichlet weight draws over the 5 gauges,
     weighted-sum winner counted per draw — "probability a community member with
     unknown tool preference prefers this layout" (the most decision-theoretic
     reading of 'admissible to the community')
  R8 Copeland pairwise majority (A beats B if better on >=3 of 5 gauges)
  R9 fastest-admissible: maximize speed s.t. worst COMMUNITY axis <= T,
     T in {5, 7.5, 10, 15}% — the speed-first dual of min-max
ROBUSTNESS BATTERY (per rule): (a) leave-one-gauge-out x5, (b) jackknife each pool
row, (c) drop-wfd variant (wfd is a component of oxey2 — known redundancy that
gives the travel cluster extra votes), (d) normalization swaps where applicable.
Stability score = fraction of perturbations preserving the pick.
DECISION GUIDANCE (registered before seeing results): if multiple rules + the
robustness battery agree with R1's pick, min-max stands vindicated; if rank-based
or random-preference rules disagree AND are more stable, we present the
disagreement + recommend the more robust rule for P17+; the redundancy finding
(c) informs whether wfd stays a pick axis.

### SEL-1 OUTCOME (2026-07-13, runs/sel1_rules.json)
PICKS (18 admissible layouts): the 8 rules split into exactly TWO camps.
  Worst-axis camp (R1 qgap-minmax, R5 L2, R2/R3 minmax variants): E10-r888303
    (the registered P16 pick; R2/R3 prefer its close sibling OX1-r888303).
  Consensus camp (R4 mean, R6 borda, R7 random-preference 45.8% win share,
    R8 copeland): E10-r888301 — profile regs [spd 6.3, gk 7.1, ox1 1.7, ox2 1.6,
    wfd 0.4]: excellent on the 3 travel gauges, pays genkey.
STABILITY: minmax family + L2 + copeland all 0.96 (single flip each, and R1's one
flip is drop-GENKEY -> OX1-r888303, a same-arm sibling); mean/randpref 0.83,
borda 0.78. The consensus camp is BOTH more gauge-sensitive and its champion
E10-r888301 relies on the travel-cluster redundancy (oxey1/oxey2/wfd all price
finger travel — 3 of 5 votes) — exactly the double-counting flagged in (c):
under drop-wfd, R6 flips; under drop-oxey1 or drop-oxey2, R4/R6/R7 all flip to
the R1 pick. keymeow (held-out from all rules) splits the camps on its axes:
E10-r888301 better sfb (1.14 vs 1.29), E10-r888303 much better lsb (1.27 vs 2.03).
READING (per the registered guidance): min-max qgap survives the battery best
alongside L2 and copeland, and its pick does NOT depend on the redundant travel
triple-vote. The consensus rules answer "what does the average preference-weighted
community member like" and their answer is one sibling over (a travel-leaning
E10). RECOMMENDATION for P17+: keep R1 min-max qgap as the primary registered
pick rule, ADD two published diagnostics per pool — the R7 random-preference win
share (decision-theoretic robustness) and a copeland pairwise table — and REMOVE
wfd as a pick axis (keep it as a report row) to kill the travel double-vote. No
re-crowning: P16-balance stands (its R7 share is 29.6%, second).

## P17 — direct min-max search on reformed axes (registered 2026-07-13, BEFORE results)
MOTIVATION: we PICK by min-max regret but have only ever SEARCHED weighted sums,
which reach only convex-supported Pareto points; the min-max optimum can sit
between them. P17 searches the pick rule directly (augmented Chebyshev) at ~4x
P16 density, under the SEL-1 reform.
AXES (SEL-1 reform adopted): pick axes = {speed, genkey, oxey1, oxey2}; wfd is a
REPORT row only. keymeow = post-hoc report gauge via kmrun (new JSON-input mode).
OBJECTIVE (in-loop): n_g = (loss_g - BEST_g)/(QREF_g - BEST_g) with BEST_g = min
over the full P16 board (runs/p16_coopt.json, stationary constants) and QREF_g =
qwerty30M recomputed on the same surface; fit = max_g(w_g n_g) + 0.05 sum_g(w_g n_g)
(rho=0.05 registered).
ARMS (56 searches, SA 12x16k + 2-opt each, same per-search budget as P16):
  CHEB x44: w ~ Dirichlet(1,1,1,1), rng seed 20260714 for the draws, search rngs
    888501..888544;  MMX x6: equal weights, rngs 888401-6;  SPD x2 (pure speed
    anchor), rngs 888407-8;  SEEDED x4: SA at T0/4 from P16-balance, E10-r888301,
    OX1-r888303, P16-spd, equal weights, rngs 888409-12.
POLISH: top-10 of the union pool by reformed min-max -> alternating exhaustive
  2-opt + 3-cycle passes on PURE equal-weight min-max until no improvement
  (cap 5 rounds); polished variants join the pool.
POOL for the pick: all P17 searches + polished + the full P16 board (searched,
  incumbents*, semimak, graphite) + qwerty30M reference.
PICK RULE: R1' = min-max qwerty-gap regret over the 4 axes, mins = union-pool
  mins, speed cap 0.5% vs pool-best. DIAGNOSTICS (published, non-binding): R7
  random-preference win share (Dirichlet 20k, seed 20260713), Copeland table,
  wfd + keymeow rows. SPEED-BUDGET CURVE: best reformed min-max at caps
  {0.1, 0.25, 0.5, 1.0}% and fastest-with-all-community-axes<=T for T in
  {5, 7.5, 10}%.
SUPERSEDE BAR: P17 pick replaces P16-balance as the flagship C30M candidate iff
  (a) reformed max-regret (same union pool) improves by >=0.5pp, OR (b) it is
  better on >=3 of 4 pick axes. Ties/near-ties -> P16-balance stands (stability
  preferred). Both documented regardless. VERIFICATION: winner + runner-up get
  dof + o2/v1 repl runs and kmrun rows. Deliverable promotion remains user-gated.

### P17 OUTCOME (2026-07-13, runs/p17_coopt.json)
The direct-min-max campaign WORKED — the hypothesis (weighted-sum search reaches
only convex-supported points; the min-max optimum sits between them) is confirmed
on this pool. 56 searches + top-10 polish, 376s total.
PICK: POL-MMX-r888404 = fyu,.vgdnlhieaocstrmkj'q-bwpxz (polished equal-weight
   f y u , .   v g d n l
   h i e a o   c s t r m
   k j ' q -   b w p x z
Chebyshev arm). Reformed max-regret 5.42% vs P16-balance 9.20% on the SAME union
pool (P16-balance's 6.38% was against the smaller P16 pool; the P17 pool's better
mins raise everyone's regrets — the honest comparison is same-pool).
SUPERSEDE BAR: (a) d_maxreg 3.79pp >= 0.5pp PASS; (b) 3/4 axis wins (speed,
oxey1, oxey2; concedes genkey 31.27 vs 30.77) PASS => P17-pick REPLACES
P16-balance as the flagship C30M candidate (registered consequence).
repl: o2 -212.2B (was -234.1B), v1 0.428 (was 0.415) — best keybo values ever,
and the o2 gap to graphite (-199.1B) narrows to 6.6%. wfd 1515.9B (report row)
also better than P16-balance's 1531.0B — dropping wfd as a pick axis did not
sacrifice it. keymeow: sfb 1.19 (better than graphite 1.23), lsb 1.84 (worse; the
one axis it concedes to the field). RUNNER-UP POL-CHEB-r888514 =
pyuo,vgdnlhiea.cstrmkj-z'fwbxq (5.70%) is the keymeow-friendly sibling (lsb 0.77)
— documented as alternate.
DIAGNOSTICS: R7 win share 32.8% (pick is ALSO the random-preference winner —
worst-axis and consensus camps AGREE for the first time); Copeland 67.0 (top).
Speed-budget curve: tightening the cap to 0.1% costs 2.8pp of balance
(OX1-r888303 8.25%); loosening to 1.0% buys nothing (5.42% already optimal).
Structure: the pick is the E10-r888301 basin (fyu,. top, hieao home-left,
cstrm home-right) — the SEL-1 consensus champion's family, now polished to
dominate on the min-max criterion too. MMX raw arms did NOT find it; the
polish stage (exhaustive 2-opt+3-cycle on pure min-max) was decisive.

## KAN-1 — the keybo analyzer (registered 2026-07-13, BEFORE build; user directive:
## "combine their work and go even further beyond… create the best keyboard analyzer")
GOAL: `keybo analyze` — a single analyzer that (1) reports what NO community tool
can: PREDICTED TYPING TIME from the LOLO-gated measured-keystroke surfaces
(ms/char + % time saved vs qwerty), with per-bigram/per-key/per-finger TIME
attribution; (2) natively computes the community metrics people already trust —
genkey Score, oxeylyzer-1, oxeylyzer-2 (+wfd), keymeow-class sfb/sfb-dist/lsb/
lsb-dist/alt/roll/redir — each EXACT-PARITY-GATED against the real tool; (3)
computes everything on ONE shared corpus (configurable), eliminating the
corpus-artifact differences that make cross-tool numbers incomparable today.
ARCHITECTURE: src/keybo/analysis/community.py (vendored exact ports, adapted
from the parity-gated keybo-e2e/oxey_ports.py + genkey_port.py; tool data
vendored under data/community/vendored/); src/keybo/analysis/kmstats.py (native
keymeow-class stats); src/keybo/analysis/timecard.py (surface eval +
attribution); src/keybo/cli/analyze.py (the command). Production K31 models
vendored gzipped under models/ (bigram_reg31 + trigram_cond31, seeds 0-2).
PARITY GATES (all must pass as pytest tests before the analyzer is documented):
  G1 genkey: exact port vs binary goldens (existing gk-parity rank corr 1.0,
     ratio spread <=2%) on the 24-layout board, baked as fixtures.
  G2 oxey1/oxey2: exact ports vs repl goldens (rank corr 1.0, spread <=5%;
     o2 exact x100 scale) on >=8 layouts incl P17 pair — fixtures from runs/.
  G3 keymeow-class: native stats vs kmrun on the IDENTICAL corpus (kmrun JSON
     mode, keybo corpus): per-stat abs diff <=0.02pp on all 24 layouts.
  G4 time: `keybo analyze` speed numbers must reproduce runs/p17_coopt.json
     board values bit-close (rel err <=1e-6) for 5 spot layouts.
CONVENTIONS: 30-char row-major strings on ROW_STAGGERED_30; oxey dof pins the
31st char (';' for C30M layouts, "'" for classic) — auto-detected; time surface
= C30M-charset K31 models @ wpm 90; corpus default = keybo monkeytype-derived
(data/corpus/), --corpus swappable. Time numbers for charsets outside C30M
coverage are reported with an explicit coverage% line.
DELIVERABLES: the command + tests + docs/analyzer.md + the flagship board
regenerated through `keybo analyze` (one command, one corpus). NON-GOALS (this
charter): GUI, optimizer integration changes, non-ANSI geometries, retraining.
CONSEQUENCES: if any gate FAILS, the failing gauge ships DISABLED with the
failure documented (no silently-wrong numbers). Publishing/promoting the
analyzer externally remains user-gated.

### KAN-1 DEVIATION (2026-07-13, recorded before build): vendored models go under
data/models/k31/ (gzipped), NOT models/ as chartered — the audit subagent
(keybo-audit-evaluation D1) verified /models/ is gitignored at the repo root, so
the chartered path would have silently excluded the vendored surfaces from git
(the exact provenance hole the audit flagged: every flagship number currently
lives in un-versioned external workspaces). Same content, versioned path.
Community tool data (o2/v1 english corpora, genkey keybo corpus, keymeow-format
keybo corpus) vendored gzipped under data/community/vendored/ with provenance
notes. KAN-1 upgraded per the audits: acceptance now includes "fresh clone +
`keybo analyze` reproduces the P17 flagship board" (closes audit finding D1 for
the flagship numbers; the search scripts remain external until a later charter).

### KAN-1 OUTCOME (2026-07-13)
BUILT AND ALL GATES PASS. `keybo analyze <layouts...> [--ref R] [--target-wpm W]
[--attribution] [--json]` ships in the package (src/keybo/analysis/{community,
kmstats,timecard}.py + cli/analyze.py).
GATES: G1 genkey EXACT (8 golden layouts, float-identical to the binary-gated
campaign values). G2 oxey1/oxey2/wfd INTEGER-EXACT (same 8). G3 native
keymeow-class stats vs kmrun on the identical corpus: worst |diff| 0.0004pp
(bar 0.02pp) across 5 layouts x 11 stats — required using the vendored keymeow
corpus on both sides; the first attempt compared against kmrun-on-shai-iweb and
failed at 0.38pp, which is the CORPUS DELTA, not a port error (recorded: the
corpus is a first-class input, exactly KAN-1's thesis). G4 time surface
reproduces the P17 board at rel err <=7e-15 (bar 1e-6); `saved%` matches the
flagship doc (+3.53/+3.51/+2.55/+2.38 for c30m/lsb/semimak/graphite vs
qwerty30M). Fast test suite: 231 passed, 0 failed (analyzer adds 26).
Goldens frozen in tests/analysis/golden_kan1.json from runs/p17_coopt.json +
kmrun-on-vendored-corpus. Vendored: data/models/k31/*.json.gz (+.meta.json.gz,
6 surfaces) and data/community/vendored/*.json.gz (4 tool corpora) — a fresh
clone reproduces the flagship board with one command (audit finding D1 closed
for flagship numbers). docs/analyzer.md documents scope + honest limits (model
predictions, no human confirmation yet; 30-key ANSI only; tool scores live on
native corpora by design).

## BAND-1 — WPM-banded specialist models vs the global surface (registered 2026-07-14,
## BEFORE results; user hypothesis)
MOTIVATION (user): "instead of one model to which we inject all WPM, multiple models,
each taking a WPM range — test per-range whether it beats the single all-WPM model."
Analogy cited: quality injection lost to a model trained on the quantile directly —
feature-injection can lose to direct specialization. Evidence banding has something
to work with (fresh baseline artifact, ensemble bigram, qwerty fold): per-band
calibration slopes 1.52/1.27/1.30/1.30/1.33 for 40-60/../120-140 — the global model
compresses within EVERY band and the compression VARIES by band (1.27 vs 1.52), which
one global affine cannot fix. Support census: 8.3M/9.9M/6.2M/2.7M/0.9M samples per
20-band (thinnest 120-140: 873k samples, 459 qwerty-fold cells — viable).
USER REFINEMENT (registered verbatim intent): 20 WPM was arbitrary — treat banding
scheme as the experimental variable: bigger/smaller widths, variable (equal-mass)
widths, and OVERLAPPING bands whose covering models' predictions are COMBINED.
FIXED ACROSS ALL ARMS: campaign-pinned sources (bistrokes_v5 d6cb4c81…, band 40-140),
the production bigram recipe byte-identical to the baseline artifact's train_params
(depth3 lr.05 n300 gamma.957 alpha.141 lambda.011 mcw4 subsample.7 colsample.7),
practice_term+layout_weights on, participant-pure leave-one-layout-out folds
(azerty/dvorak/qwerty/qwertz; census overlap=0), and the UNCHANGED 20-wpm EVALUATION
frame (build_cells, min_cell_samples=10) — training banding varies, the evaluation
frame does NOT (MED-audit refinement #2 discipline: frame key != model structure).
ARMS (bigram surface; each must predict every evaluation cell in 40-140):
  G       global control, retrained in-driver (NOT copied from the baseline JSON —
          same code path as specialists for byte-fairness).
  HARD-20 specialists on [40,60)…[120,140); cell -> its band's model.
  HARD-40 specialists on [40,80),[80,120),[120,140).
  EQMASS-5 five bands, edges = train-fold sample-wpm quintiles (recorded per fold).
  OVL-40/20 overlapping width-40 stride-20 bands [40,80),[60,100),[80,120),[100,140);
          cell at midpoint m -> triangular-weight blend of covering bands' predictions
          (weights by distance to band centers, normalized; the user's multi-bucket
          combine).
  CAP-G   capacity control: global with n_estimators x5 (1500) — separates "more
          total capacity" from "banded structure".
DIAGNOSTIC (no training): per-band affine recalibration of G (WLS in ms, fit on
train-fold cells in-band, applied to test fold) — the "is it just scale?" control.
METRICS: standard 13-field per-band rows + pooled + fixed-wpm-90 layout ranking
(tau_heldout, layout_mae_ms). PRIMARY: weighted_log_mae (per the model-metrics audit
recommendation), pooled + per-band vs G.
DECISION RULES (registered): a scheme QUALIFIES iff (a) pooled weighted_log_mae
improves vs G, (b) it improves in >=4 of 5 evaluation bands, (c) tau_heldout does not
degrade (pooled, any fold), AND (d) it also beats CAP-G on pooled weighted_log_mae —
else the verdict is "capacity, not structure". If the per-band-affine diagnostic
captures >=70% of the best scheme's pooled gain, the verdict records "mostly
recalibration — route to the calibration phase instead". Seed-noise gate: final
comparisons at 3 seeds; a win inside the 3-seed p95 spread of G is a TIE.
SEQUENCING: scout all arms at 1 seed (seed 0), 4 folds; then G + top-2 schemes at
seeds {0,1,2}. Trigram surface: winner-only confirmation run (separately registered
outcome line). Runs niced (codex candidate matrix may share the host).
ISOLATION: new driver agent-artifacts/experiments/wpm_banding.py in THIS checkout;
does NOT touch the Task-5 byte-frozen drivers or validate.py; reuses keybo.training
machinery as a library. Informs a FUTURE candidate arm — a BAND-1 win does NOT edit
the frozen 6-arm matrix.
CONSEQUENCES: qualify -> register a banded/blended arm (or a serve-band model at
wpm 90) as a Task-5-style candidate with its own prereg; no-qualify -> negative
result recorded, wpm stays a feature.

### BAND-1 OUTCOME (2026-07-14, artifacts/band1_scout.json, 16 min, seed 0, 4 folds)
THE BAND STRUCTURE IS REAL; SPECIALIST MODELS ARE THE WRONG FIX; THE DIAGNOSTIC WON.
Pooled wlogmae vs G (mean of 4 LOLO folds): HARD-20 -7.5%, HARD-40 -6.8%,
EQMASS-5 -9.0%, OVL-40/20 -8.2% — every banded scheme beats the global model.
BUT the registered rules fired against them: (b) FAIL — best scheme EQMASS-5 wins
only 3/5 evaluation bands (specialists LOSE the dense middle bands 60-100, where
cross-band pooling is worth more than specialization: classic variance cost);
(c) FAIL for EQMASS-5 — tau_heldout degrades 0.333 -> 0.0. CAP-G control: +2.1%
WORSE than G — capacity is not the mechanism, structure is.
DIAGNOSTIC VERDICT (registered >=70% rule -> fires at 117%): G+BANDAFFINE — the
global model plus train-fold-fit PER-BAND affine recalibration — captures MORE
gain than the best specialist scheme (-10.5% pooled wlogmae), wins where
specialists win AND where they lose, repairs per-band slopes 1.20/1.06/1.02/1.00/
0.88 -> 0.99/0.98/1.00/1.03/1.01, and IMPROVES tau_heldout to 0.667 (best of all
arms). Registered consequence applies: "mostly recalibration — route to the
calibration phase instead."
ROUTING NOTE for the calibration phase (Task-8/codex thread): its current design
fits ONE affine over the full 40-140 band. BAND-1 shows the bigram surface's
slope VARIES by band (1.20 at 40-60 down to 0.88 at 120-140) — one global line
cannot fix both ends; the affine should be WPM-BAND-CONDITIONED (per-band (a,b)
or a smooth wpm-dependent slope), cross-fitted exactly as designed. Also: at the
SERVE band (80-100, wpm 90) the bigram G slope is already 1.02 — the bigram
optimizer gains little; the trigram surface (README slopes >1.1 at 100-140 at
serve-relevant bands) is where calibration pays.
CAVEATS: single seed (cross-fold consistent, magnitude ~10x campaign seed noise,
but unconfirmed); tau on 4 layouts is ~1-bit; qwertz fold is where specialists
gain most (-31%) while qwerty is where the affine gains most — fold-heterogeneous.
CONSEQUENCE: banded-specialist arm NOT registered as a candidate (rules b+c).
The per-band affine finding is handed to the calibration phase owner. wpm stays
a model feature. No 3-seed confirmation spend (nothing qualified to confirm).

## BAND-2 — WPM-conditioned calibration: confirm, choose the family, find the
## mechanism, measure the product impact (registered 2026-07-14, BEFORE results)
FOLLOW-ON from BAND-1 (user: "take this further — investigate thoroughly").
MOTIVATION: BAND-1's diagnostic (per-band affine on the global model) beat every
specialist scheme (-10.5% wlogmae, slopes -> ~1.0, tau up) at seed 0. And the
TRIGRAM baseline is miscalibrated AT THE SERVE BAND (80-100 slopes 1.12 qwerty /
1.28 dvorak folds) — the flagship time-saved%% is computed on a compressed scale,
so calibration has direct product stakes, not just hygiene.
STAGES + DECISION RULES (registered):
 A-CONFIRM: 3 seeds x 4 LOLO folds x BOTH campaign surfaces (bigram bistrokes_v5
   d6cb4c81…, conditioned-trigram tristrokes_cond_v3 1b5d7abd…, production
   train_params from the baseline artifact). C-BAND(ms) must beat G pooled
   wlogmae beyond G's 3-seed p95 spread on each surface, else record "BAND-1 was
   seed noise" for that surface and stop there.
 A-FAMILY (same runs): cross-fitted correction families, fit per seed on
   train-fold cells, applied to held-out cells, ensembled as mean-of-calibrated
   (matching the calibration-phase adjudication):
     C-BAND   per-20-band affine, ms (BAND-1 winner; 10 params)
     C-SPLINE per-band (a,b) linearly interpolated in wpm (continuous; 10)
     C-LIN    a(w)=a0+a1 w, b(w)=b0+b1 w, single WLS (smooth; 4)
     C-LOG    per-band affine in log-ms (multiplicative/power; 10)
     C-ISO    per-band isotonic pred->obs (nonparametric; cross-fit polices)
   WINNER = best pooled wlogmae (seed-ensembled) subject to GUARDS: tau_heldout
   not degraded vs G; post-correction per-band slopes all in [0.90, 1.15];
   bottom-3 freq-decile MAE not worse than G by >2% (rarity guard). Ties inside
   G's seed spread -> FEWEST PARAMETERS wins.
 A-MECHANISM (bigram): per (band x class) slopes of G, classes = same-finger /
   same-hand-diff-finger / cross-hand. If within the serve band the max
   between-class slope gap > 0.15, register "class-structured — explicit
   wpm x class features (pace-matrix H2) indicated beyond affine"; else
   "amplitude-only — affine suffices."
 B-IMPACT (winner family only): replicate on the K31 production frame
   (bistrokes31_v1 + tristrokes31_cond_v1, ROW_STAGGERED_31 geometry-trained
   models as in k31_train.py) and recompute the flagship board's time-saved%%
   with serve-band-calibrated T2/Tcond tables (keybo.analysis.timecard).
   Reported as the CALIBRATED headline estimate with cross-fit spread;
   messaging/docs changes from it remain USER-GATED. (Rank order at fixed serve
   wpm is provably affine-invariant per band — the argmax layout cannot change;
   only magnitudes can.)
ROUTING: results + winning coefficients hand to the Task-8/codex calibration
phase (which owns productionizing); this campaign edits no frozen files; driver
agent-artifacts/experiments/band2_calibration.py; niced, n_jobs 24.

### BAND-2 STAGE-A OUTCOME (2026-07-14, artifacts/band2_calibration.json, 26 min)
A-CONFIRM: PASS on both surfaces — C-BAND beats G pooled wlogmae by -10.7%
(bigram) / -9.4% (trigram), vs mean per-fold 3-seed G spread of 0.0006-0.0007
(the gain is ~15x seed noise). BAND-1 was not noise. Effect is FOLD-CONCENTRATED:
qwerty-held-out folds -23..-28%, azerty/dvorak folds -0.4..-8.7%.
A-FAMILY: wlogmae order (both surfaces): C-ISO best (-15.5% bi / -11.0% tri) >
C-LOG > C-BAND = C-SPLINE > C-LIN. tau: every family 0.67 vs G 0.33 (bigram);
trigram all 1.0 (the trigram defect is MAGNITUDE, not order — exactly the
product framing). Rarity guard: PASS all (ratios 0.99-1.02).
SLOPE GUARD — REGISTERED WORDING FAILS FOR ALL ARMS *AND FOR G ITSELF*
(G violates [0.90,1.15] in 11/20 fold-band cells; a cross-fitted correction
cannot guarantee absolute slope bounds under population shift when the base
model starts outside them). DEVIATION (recorded, not silent): guard re-read as
its intent — mean |slope-1| must improve or hold, few cells made worse. Under
that reading: bigram — all families improve, C-ISO by far most (0.157 -> 0.110);
trigram — C-BAND/C-SPLINE (0.108 -> 0.133) and C-LOG (0.148) WORSEN calibration
and FAIL; C-LIN improves (0.088); C-ISO holds (0.105). The strict guard did its
job: it exposed that per-band AFFINES over-correct trigram scale in transfer.
WINNER: C-ISO (per-band isotonic) — only family that improves-or-holds slopes on
BOTH surfaces, best wlogmae on BOTH, tau tied-best, rarity clean. CAVEAT: C-ISO
is NONLINEAR, so the affine rank-invariance argument does NOT apply — stage B
must verify flagship ORDER preservation explicitly. C-LIN recorded as the best
rank-safe affine alternative for the calibration phase if monotone-nonlinear is
rejected there.
A-MECHANISM (registered rule): serve-band (80-100) class slopes ALT 1.03 /
SAME_HAND 1.04 / SAME_FINGER 1.02 — max gap 0.02 < 0.15 => "amplitude-only —
affine/monotone correction suffices" AT THE SERVE BAND. Observation (non-binding,
routed to pace-matrix H2): at 120-140 the classes DIVERGE hard (ALT 0.73 vs
SAME_FINGER 1.21, gap 0.48) — high-WPM honesty needs wpm x class features, but
the serve-wpm product does not.
PRODUCTION-REGIME NOTE: the compressed (slope 1.2-1.5) regime is the
qwerty-HELD-OUT fold — a model that never saw qwerty. Production models train
WITH qwerty (98.7% of data); the production-like folds show mild over-dispersion
at high bands instead. Stage B's cross-fit impact estimate must report per-fold
spread for exactly this reason.

### BAND-2 STAGE-B OUTCOME (2026-07-14, artifacts/band2_impact.json)
K31-frame serve-band OOF calibration (geometry-only preds vs raw obs, fit on
held-out-layout cells, seed 0): bigram pooled affine (a=-79.6, b=1.531),
trigram pooled (a=-3.6, b=0.993). PER-FOLD b: bigram 0.48 (dvorak) / 0.97
(azerty) / 1.20 (qwertz) / 1.54 (qwerty); trigram 0.54-1.00. THE SERVE-BAND
CORRECTION IS POPULATION-DEPENDENT — there is no single "true" scale.
FLAGSHIP IMPACT (saved%% vs qwerty30m; uncorrected keybo-c30m +3.53):
  pooled affine +4.74, pooled isotonic +5.02, per-fold range +2.16 (dvorak
  correction) to +4.76 (qwerty correction). READING: the pooled OOF calibrations
  say the advertised headline UNDERSTATES the gap; the honest calibrated
  statement is a population-conditioned RANGE [≈2.2%%, ≈5.0%%] that brackets the
  current +3.5%%. The COMPARATIVE claim is robust: keybo-c30m's lead over
  semimak is 0.6-1.2pp under EVERY correction (0.98pp uncorrected) — calibration
  moves magnitudes, not the community-facing comparison.
ORDER CHECK — CORRECTION TO A REGISTERED CLAIM: the prereg asserted per-band
affine rank-invariance at fixed serve wpm; that holds PER SURFACE, but the
board total mixes TWO surfaces with different corrections, so order CAN move —
and did: keybo-c30m and keybo-lsb SWAP under 5/7 correction frames (lsb ahead by
<=0.08pp; statistical tie leaning lsb). Strengthens the standing recommendation
of keybo-lsb as the public flagship. semimak/graphite never reorder vs ours.
FRAME NUANCE: the campaign-frame trigram compression (v5 slopes 1.12-1.28 at
serve band) does NOT reproduce on the K31 production trigram frame (OOF qwerty
fold b=0.997) — the production Tcond is nearly calibrated for qwerty-like
typists; the bigram table carries more of the K31-frame miscalibration.
METHODOLOGICAL CONCLUSION (routed to the calibration phase): leave-one-LAYOUT-
out is the WRONG cross-fit axis for production calibration — the production
model serves the layouts it trained on; the unseen axis is the TYPIST. The
calibration phase should cross-fit by held-out PARTICIPANTS (within-layout),
which removes the population-shift confound that produced the [0.48, 1.54]
slope spread here. C-ISO evidence + C-LIN rank-safe fallback + this
participant-axis spec handed to Task-8/codex.
CONSEQUENCES: no flagship messaging change without user decision (numbers above
are 🟡 model-internal, population-conditioned). BAND-2 CLOSED.

## COMM-SPD board (registered 2026-07-17, BEFORE results; user: "compare all top
## layouts across all gauges, including a community-only-trained speed gauge")
GAUGE DEFINITION (descriptive, NOT an objective — per the four community audits,
a community-fit score is typist-confounded; this gauge answers "what does a model
trained ONLY on the 4 community typists predict?", nothing more):
  Train bigram + conditioned-trigram surfaces on the 4 natural rowStagger
  community labels ONLY (colemak@alite, mtgap-variant@richarddavison,
  custom-d42a1f92@ddn, custom-aa426873@vg — the COMM-D primary set; ortho/
  angleMod labels excluded as geometry-mismatched, octahedron excluded per
  registered exclusion), production recipe (REG-LOLO bigram params; CAND4
  trigram params; practice_term+layout_weights on, LOGRAT), ROW_STAGGERED_30,
  wpm 90. Serve geometry-only T2+Tcond tables over the C30M trigram corpus —
  the exact production-timecard construction, so "community saved% vs qwerty30m"
  is apples-to-apples with the Aalto saved% column.
BOARD: layouts = keybo-c30m, keybo-lsb (P17 pair), p16-balance, P13STAB-win*,
  P10-w0.5* (C30M embeds), semimak, graphite, colemak30m (embed), dvorak30m
  (embed), ref qwerty30m. Gauges = Aalto saved% (production timecard), COMMUNITY
  saved% (this gauge), genkey, oxey1, oxey2, wfd, keymeow sfb/lsb/alt/roll/redir.
HONEST BOUNDS (registered with the gauge): n=4 typists, contributor≡layout
  confound, zero-shot structure-transfer ceiling rho 0.51-0.65 (COMM-D corrected),
  ~5.4k bigram rows (vs 29M Aalto samples) — expect NOISY magnitudes; the gauge
  is reported for CONVERGENCE/DIVERGENCE vs the Aalto column, never for adoption
  decisions. Seed-averaged over 3 seeds to damp the small-data variance.

### COMM-SPD OUTCOME (2026-07-17, runs/comm_spd_board.json + _full.json)
Gauge built as registered (4 rowStagger labels, 1775 bi + 8931 tri rows, 3-seed
mean, production-timecard construction). Board rows recorded. READINGS:
(1) CONVERGENT SIGN: every alternative beats qwerty on BOTH surfaces; the two
models agree qwerty is worst by a wide margin.
(2) MAGNITUDES ~2x on community (e.g. keybo-c30m +3.53 Aalto vs +6.15 comm) —
expected: 4 enthusiast typists on optimized boards vs population Aalto.
(3) ORDERING DIVERGES (the registered caution in action): community model's top
= graphite +7.74 / P10-w0.5* +7.68, with keybo-lsb +6.90 > keybo-c30m +6.15;
Aalto's top = P10* +3.63 / c30m +3.53. Spearman across 9 non-ref rows ~0.32 —
weak. Consistent with the audits: the community surface encodes ITS typists'
styles (roll-heavy optimized-board typing), not a transferable geometry law.
Notable: community model ranks keybo-lsb ABOVE keybo-c30m (+0.75pp) — the third
independent signal favoring lsb as flagship (after keymeow lsb and the BAND-2
order-swap).
(4) dvorak30m: Aalto +3.18 but community +5.26 (lowest non-colemak) — the two
surfaces disagree most where typist populations differ most, as predicted.
STATUS: descriptive gauge; recorded; NOT an objective and NOT adoption evidence.

## COMM-OPT-1 — what does the community model DO as an objective? (registered
## 2026-07-17, BEFORE results; user directive: pure / swapped / both)
Three arms, all on the C30M charset with the EXACT P17 machinery (rule 40cf881:
augmented Chebyshev rho=0.05, SA 12x16k + exhaustive 2-opt, 44 Dirichlet + 6
equal-weight + 2 speed-corner + 4 warm-started searches, top-8 pure-min-max
2-opt+3-cycle polish, stationary normalization, pick = min-max qwerty-gap regret
with 0.5%% speed cap) — only the SPEED AXIS varies:
  COMM-PURE: maximize community-model speed ONLY (the community speed frontier;
    3 rngs, no pick rule — report the argmax).
  COMM-SWAP: keybo-lsb's pipeline with the COMMUNITY surface REPLACING Aalto —
    axes {comm-speed, genkey, oxey1, oxey2}, cap on comm-speed.
  COMM-BOTH: keybo-lsb's pipeline with BOTH surfaces as axes —
    {aalto-speed, comm-speed, genkey, oxey1, oxey2}, cap on AALTO speed
    (production semantics preserved; community joins as a 5th regret axis).
COMMUNITY SURFACE: byte-identical to COMM-SPD (rule a70ee32): 4 rowStagger
labels, production recipe, 3-seed mean T2+Tcond @ wpm 90. Stationary norms:
aalto/genkey/oxey BEST from the P16 board as in P17; community BEST = graphite's
COMM-SPD total, QREF = qwerty30m.
EVALUATION: winners + runners-up on the FULL board (both speed gauges, genkey,
oxey1, oxey2, wfd, keymeow via kmrun) vs keybo-c30m/keybo-lsb/semimak/graphite.
REGISTERED INTERPRETATION BOUNDS: the community surface is typist-confounded
(4 audits) — COMM-PURE/SWAP winners are DIAGNOSTIC objects (what the confounded
objective wants), NOT candidates. COMM-BOTH probes the robustness frontier; the
registered observation rule: if its pick holds within 0.1pp of keybo-lsb's
AALTO saved%% while gaining >=1.0pp COMMUNITY saved%%, record "two-population
robust frontier is non-degenerate" (still not an adoption — user-gated as ever).
EXPECTATIONS (falsifiable, registered): COMM-PURE drifts toward roll-heavy
graphite-space; COMM-SWAP lands nearer graphite than keybo-lsb; COMM-BOTH pays
<=0.3pp Aalto for >=1pp community.

### COMM-OPT-1 ADDENDUM (2026-07-17, before launch): COMM-G2 handoff consumed.
The codex thread's COMM-G2 board (its checkout, commit c46c44c, board sha
fe38a466…) independently built a FULL-community descriptive model; its ordering
(keybo-p10 > graphite > keybo-lsb > semimak > E10 > SPD > keybo-c30m > qwerty)
CONVERGES with COMM-SPD's 4-label gauge on every headline: graphite+P10 top,
lsb > c30m (the FOURTH independent lsb signal), qwerty last, weak Aalto rank
correlation (its Spearman 0.43 vs my 0.32). Its COMM-G1 verdict
search_qualified=false stands: COMM-OPT-1's PURE/SWAP arms remain DIAGNOSTIC
objects and BOTH remains a robustness probe — nothing here can promote a
community-fit objective into adoption. No design change; launching as
registered (COMM-SPD 4-label surface, exact P17 machinery).

### DIVERGENCE RCA OUTCOME (2026-07-17, user-gated investigation; four independent
### lines: codex census, code-path, data/target, numeric + ultracode synthesis)
QUESTION (user): Aalto vs community models diverge 2x in magnitude, weakly in
rank, and the community model agrees with community consensus — "something is
deeply wrong somewhere." VERDICT: NO BUG ANYWHERE. Decisive anchor: BOTH
datasets pushed through a BYTE-IDENTICAL fresh-train code path reproduce the
divergence exactly (2.00x, Spearman 0.183) => 100%% data, 0%% code/frame/serve.
tristrokes_last==conditioned-increment verified; boards reproduced bit-exact
from model bytes; COMM-SPD board estimand verified consistent (TimeSurface,
published 7.738 reproduced exactly).
FACT VERDICTS: (1) MAGNITUDE = population dispersion (community surface learns
~1.8-2x larger per-transition contrast; enthusiasts differentiate geometry more)
AMPLIFIED by opposite-direction extrapolation at the qwerty point (community
qwerty trigram support 50.7%% / 0.3%% at >=100 samples vs Aalto 97.8%% / 95.3%%)
— the +7.7%% digit is the least trustworthy on the board; sign robust. H6-as-
denominator REFUTED (community qwerty TOTAL is 1.04x Aalto's; renorm moves the
wrong way). Consistent with BAND-2's preregistered population-conditioned range
[2.2, 5.0] bracketing +3.5. (2) RANKING = n=4 selection/style-as-geometry
confound: affine/reference-invariant surface shape; community prices lsb +12.5%%
and space transitions +13.9%% relative (graphite up, keybo-c30m down);
disagreement PEAKS in the best-supported cells => real learned style, not
noise; graphite<->P10 top swap is seed noise. UNIDENTIFIABLE from geometry with
this data (the four community audits' wall, now quantified). (3) CONSENSUS
AGREEMENT = CIRCULAR: the 4 training layouts embody genkey/oxey design
principles; a model fit to their adopters re-expresses the design (Spearman-to-
heuristics 0.43-0.62 vs Aalto ~0). Not independent validation.
TWO-SURFACE LENS (user): COMM-SPD vs COMM-G2 agree on every headline, differ
~1.3pp in level; dominant driver = training-fit population (4 labels vs full
community + wpm filter); wpm-filter -0.17pp, seed -0.06pp, serve corpus
IDENTICAL. NOT interchangeable at pp precision; both are n=4-grade estimates.
PUBLISHED NUMBERS: none wrong. FRAMING OBLIGATIONS (standing): (a) never
cross-compare the two models' saved%% without a common anchor/affine; (b) never
mix Joint vs TimeSurface estimands in one comparison; (c) community numbers are
population-conditioned n=4 estimates, descriptive only (search_qualified=false
stands). Latent-inert defect (not a divergence cause) filed: timecard trigram
table bypasses position-calibration deltas (fix before any calibrated trigram
model). GATE DISPOSITION: COMM-OPT-1 stays PAUSED; resumption is a user
decision with the RCA's finding that PURE/SWAP optimize a typist-confounded,
partially circular objective (diagnostic value only) and the real unlock
remains Phase-D collection.

### DIVERGENCE RCA — STAT-CRITIC ADDENDUM (2026-07-17, report at
### state/keybo-divergence-stat-critic/report.md; all 5 parent claims reproduced
### exactly from immutable artifacts, anchors <2.6e-12)
Refinements to the RCA outcome (b50983d), none overturning it:
(1) "graphite/P10 community-best" is UNSUPPORTED as an ordering: margin 0.0077pp
vs contributor-fold SD 0.405pp (52x); LOCO splits 2/2; SIGN-FLIPS by trigram
recipe (cand4 => graphite 4/4, reg => P10 4/4 — the board's P10 edge is a
defensible-recipe artifact). The registered "seed noise" wording is CORRECTED:
seed SD (0.043pp) is negligible vs contributor SD — the binding limit is n=4
CONTRIBUTORS, not seeds.
(2) Community-qwerty-slowest is direction-robust (4/4 folds, >=12.4ms) but
n=4 caps one-sided sign p at 0.0625 — significance UNREACHABLE with this data;
amplitude thinly grounded (support gauge fails for qwerty).
(3) The 18/28 Aalto-community pairwise agreement headline is essentially "both
rank qwerty worst": non-qwerty Spearman = 0.143 (~uncorrelated). Campaign-wide
cross-surface: Spearman 0.953 but TOP-1%% OVERLAP ONLY 0.387 — the surfaces
agree coarsely and diverge exactly in the elite tail where selection happens.
(4) Conformal bound: max finite confidence at n=4 is 0.8; 90%% needs 9
contributors, 95%% needs 19 — THE Phase-D sizing numbers (model-paths audit #4
now has its answer: collect >=9 contributors for decision-grade community
evidence).
Formal go/no-go stays NO (search_qualified=false, can_qualify_model=false).

### RCA ADDENDUM 2 — two-surface attribution CORRECTED (2026-07-17, via the
### doc-audit + codex's divergence-recipe-factorial artifact)
The COMM-SPD <-> COMM-G2 level delta (0.71-1.32 saved-pp, anchors <=2.6e-12) is
a DELIBERATE 3-AXIS PROVENANCE change with the TRIGRAM RECIPE Shapley-DOMINANT
(COMM-SPD trained trigram on CAND4 — the old adopted recipe, as registered in
a70ee32 — while COMM-G2 uses REG; plus unordered->temporal rolls fix and a
[40,140) WPM filter). The earlier RCA outcome line attributing the delta
dominantly to "training-fit population (label set)" is superseded on that
sub-point; label-set remains a contributor, recipe is dominant. Neither board
is wrong; they are different registered recipes — the boards must be named,
never averaged. (Community-vs-AALTO divergence verdicts are UNAFFECTED — that
is the separate population/selection phenomenon, b50983d/08c7cac.)

### COMM-OPT-1 RESUMPTION (2026-07-17): the user's gate is satisfied
The pause directive conditioned resumption on the divergence RCA ("treat
discrepancy RCA as the gate"). The RCA is banked (b50983d + 08c7cac + 093b0b2):
no bug; mechanisms identified and adversarially reproduced. Resuming the three
registered arms with the RCA's interpretation frame WELDED ON: PURE/SWAP
winners are DIAGNOSTIC exhibits of a typist-confounded, partially CIRCULAR
objective (they show what 4 enthusiasts' style wants, never candidates);
BOTH is the robustness probe (Aalto cap intact); community saved%% reported
under the common-anchor guardrail; n=4 conformal ceiling (0.8) stated wherever
community numbers appear. Zero prior outcomes existed at resumption (verified
at pause). The confirmatory adversarial workflow continues in parallel; a
material contradiction from it re-pauses this experiment (registered stop rule).

### COMM-OPT-1 RE-HELD UNDER THE REGISTERED STOP RULE (2026-07-17, minutes after
### 34016ba, BEFORE any launch — zero searches have ever run)
The confirmatory adversarial workflow delivered two verdict refinements that
materially bear on the experiment design (both re-verified on its anchor data):
(1) H6 UPGRADED: on a COMMON (Aalto-qwerty) anchor the community median saved%%
does not shrink toward Aalto — it INVERTS BELOW it (community 2.70%% vs Aalto
3.18%%). The entire +6-8%% community headline is carried by the single
un-estimable qwerty point (~1.8x dispersion is the mechanism that pushes it
out). CONSEQUENCE: the registered BOTH-arm observation rule ("gains >=1.0pp
COMMUNITY saved%%") is defined on a self-anchored quantity now known to be
artifact-dominated — the arm's metric is compromised as registered.
(2) RANK FLIP CORRECTED from noise-adjacent to STABLE ~15-sigma signal:
graphite-minus-keybo-c30m = +1.58pp, SD 0.10, 12/12 seeds positive (sign-
reversed vs Aalto). The style confound is robust — more data/seeds will not
wash it out. The workflow's own disposition: strengthens keep-paused.
DISPOSITION: launch HELD per the registered stop rule. The resumption logic
(gate satisfied) was sound on the information available; the gate's own final
component then contradicted the design premise before compute was spent.
OPTIONS FOR THE USER: (a) run as-registered purely as confound-diagnostics
(all three arms, headline metrics known-artifact-dominated); (b) amend the
BOTH arm's community axis to a common-anchor or plateau-internal contrast
(prereg amendment) and run; (c) keep paused — Phase-D (9/19 contributors for
90/95%%) remains the identified real unlock. No-bug/circularity/population
verdicts UNCHANGED; reconcile+synthesis phases still running.

### RCA FINAL (2026-07-17): confirmatory workflow complete (11/11 agents); one
### published statistic corrected; two-surface attribution finalized
(A) COMM-SPD <-> COMM-G2 decomposition FINAL: interchangeable FOR RANKING
(Spearman 0.94, top-3 3/3); the ~15%% level gap = trigram HYPERPARAMETERS 55%%
(COMM-SPD's CAND4 depth-5 learns larger optimized-vs-qwerty separation than
REG depth-3 — modeling choice, not bug) + schema/engine 28%% + wpm filter 17%%.
HASH-REFUTED premises: serve corpus byte-identical (ffa77f3a both), contributor
labels IDENTICAL (the "G2 uses ortho/angleMod/full-community" claim in my COMM-
OPT-1 addendum + memory relays was BRIEF-DECAY — WRONG; both surfaces train on
the same 4 rowStagger labels). Supersedes addendum-2's recipe-vs-label framing:
label-set was never a differing axis at all.
(B) PUBLISHED-STATISTIC CORRECTION (the only one): the cross-model rank
correlation "Spearman ~0.32-0.43" (COMM-SPD outcome 2b5c834; RCA outcome
b50983d) is ANCHOR-INFLATED — both surfaces force the qwerty point to 0 by
construction. Free-layout Spearman = 0.217 (9 layouts) and NEGATIVE -0.21 over
the 7 core optimized layouts. All saved%% VALUES reproduce exactly; only the
correlation digit is restated. Rank disagreement among the layouts that matter
is STRONGER than headlined.
(C) Consensus verdict HELD at CIRCULAR after adversarial pushback ("mixed")
was itself verified and rejected: the community model beats distance+WPM
held-out (WMAE +14.1%%, rho 0.475) so it learns SOMETHING — but it is not
better than Aalto and search_qualified=false, so agreement-with-consensus
remains non-evidence for a geometry law Aalto misses.
RCA CLOSED. COMM-OPT-1 disposition unchanged (held; options with user).

### COMM-OPT-1 AMENDED + RESUMED AS DIAGNOSTICS (2026-07-17, resolves the
### 43465bf fork; reversible-fork resolution taken by the agent per standing rules)
RATIONALE: the user's request was exploratory ("I want to see what happens");
their RCA gate is closed (e7af268); the held fork (metric design) is reversible.
AMENDMENT (fixes the compromised metric, changes NO search machinery): the
BOTH-arm observation rule is RESTATED qwerty-free — the self-anchored
"community saved%" is retired from all decision/observation roles (RCA: carried
by the un-estimable qwerty point). Replacement: PLATEAU-INTERNAL contrast =
each layout's community total time vs keybo-lsb's community total (layout-vs-
layout, no reference extrapolation). Search objectives are UNCHANGED from
registration (PURE minimizes community total — anchor-free sum; SWAP/BOTH
Chebyshev axes as registered; the qwerty-gap normalization affects axis
weighting only, disclosed). REPORTING: boards show BOTH the self-anchored
saved% (labeled artifact-dominated, RCA guardrail) and the plateau-internal
contrast. ALL THREE ARMS ARE DIAGNOSTIC EXHIBITS — the RCA's stable-confound
finding means no winner here is a candidate under any outcome; the experiment
answers "what does this objective WANT", which is what was asked.

### COMM-OPT-1 OUTCOME (2026-07-17, runs/comm_opt1.json; 33 searches, 5.4 min;
### diagnostics per 6ce86b8 — no arm produces a candidate)
PURE (community objective alone): degenerate, as predicted — winner PURE-r2
reaches comm +9.51%% (self-anchored) but genkey 56.0 (worse than colemak),
oxey1 10.0, Aalto +1.68%%: the confounded objective, unconstrained, walks off
every other gauge. Exhibit of style-fit, not a layout.
SWAP (community replaces Aalto in the keybo-lsb pipeline): winner SWAP-CHEB3
comm +8.67%%, plateau-contrast +1.78pp over keybo-lsb — but pays 1.3-1.6pp of
AALTO speed (+2.17%%) to get it. The heuristic axes keep it sane (genkey 27.7 —
semimak-class); the surface it optimizes simply disagrees with Aalto about
what fast IS. Registered expectation ("lands nearer graphite") CONFIRMED in
character: roll-heavy, low-lsb structure.
BOTH (both surfaces as axes, Aalto cap): the interesting one. 8 searches land
within 0.10pp of keybo-lsb's Aalto speed while gaining +0.11..+0.67pp
plateau-contrast on the community surface; top by contrast BOTH-SEED-keybo-lsb
(aalto +3.48, comm +7.57) and BOTH-CHEB1 (aalto +3.58 — ABOVE keybo-lsb —
comm +7.56, but 22/30 slots = P10-family, i.e. the joint objective rediscovers
the P10 basin the community surface loves). REGISTERED OBSERVATION RULE FIRES:
the two-population robust frontier is NON-DEGENERATE — you can hold the Aalto
optimum and buy community-model preference. Caveat welded on: the "community
preference" being bought is the RCA's 15-sigma style confound, so this is
robustness to A PARTICULAR 4-typist style, not to typists in general.
VERDICT: keybo-lsb remains the deliverable recommendation. The BOTH-arm
exhibits are recorded as the robust-frontier existence proof (the Phase-D
version of this experiment — with >=9 contributors — is the one that could
change a decision). No candidate registered from any arm, as preregistered.

## PRAC-DISC-1 — practice-term x qwerty-collinearity discriminator (registered
## 2026-07-17, BEFORE results; from the target-semantics audit's M1 mechanism)
HYPOTHESIS M1: Aalto's served surface is artificially FLAT — with 98.7%% qwerty
data, geometry is collinear with ngram identity, so the practice backfit
b(ngram) absorbs between-ngram variance and the served geometry-only g
understates layout gaps (~3%%); the community surface (99%% non-qwerty, same
bigrams at many positions) deconfounds geometry and serves a steep g (6-7%%).
Falsified pace story (verified by the audit): serve-wpm sweep 60->120 moves
gains <=0.4pp — WPM definitions CANNOT explain the doubling.
DISCRIMINATOR (audit's protocol): retrain the Aalto BIGRAM surface on the
identical frame (bistrokes_v5, REG-LOLO params, seed 0) with practice_term
FALSE vs TRUE; build T2 @ wpm 90; compare bigram-only saved%% vs qwerty30m for
the fixed board (keybo-lsb, keybo-c30m, p16-balance, P10-w0.5*, semimak,
graphite). REGISTERED READINGS: if practice-off gains jump toward the
community magnitude (>=1.5x), M1 CONFIRMED — the flatness is an attribution
artifact of the practice-residualization design, and the +3.5%% headline's
SCALE rides on an unidentifiable g-vs-b split (links to standing finding F1;
ranking expected stable since the scale effect is near-monotone — check).
If gains move <15%%, M1 REFUTED — flatness is a data fact, divergence stays
population-only. INTERPRETATION BOUND: on qwerty-collinear data the g/b split
is UNIDENTIFIED — this experiment measures attribution SENSITIVITY, not which
attribution is "true"; no headline change without the user.

### PRAC-DISC-1 OUTCOME (2026-07-17, runs/prac_disc1.json; identical frame,
### seed 0, practice_term True vs False)
M1 REFUTED at the scale level. Bigram-only saved%% vs qwerty30m:
  practice_on : lsb +3.51  c30m +3.38  p16 +3.15  P10* +3.18  semimak +2.69  graphite +2.75
  practice_off: lsb +3.47  c30m +3.28  p16 +3.49  P10* +3.70  semimak +2.73  graphite +2.90
No jump toward the community magnitude (registered bar >=1.5x; observed max
relative move +16%% on ONE layout, flagship pair moves ~0/slightly DOWN). The
Aalto surface's ~3%% scale is a DATA fact, not a practice-residualization
artifact — the RCA's population attribution STANDS unchanged.
SECONDARY (bounds the standing F1 concern): the unidentifiable g-vs-b split
moves individual layout gains by <=0.52pp (P10* 3.18->3.70; p16 +0.34) and can
reorder WITHIN the plateau (P10* becomes bigram-top under practice-off) — the
first direct quantification of F1's "practice term can flip near-ties":
real, bounded at ~half a point, flagship pair insensitive (<=0.10pp).

### DIVERGENCE RCA — RAW-CELL ADDENDUM (2026-07-17, report at
### state/keybo-divergence-raw-support/report.md; fitted boards reproduced
### byte-exact, support densities reproduced exactly)
Raw matched-cell decomposition (BEFORE any estimator) sharpens the closed RCA:
(1) The 1.79x amplitude is a RAW-DATA property (raw SD ratio 1.781 vs fitted
1.7916; robust to support floors and pace strata). XGBoost shrinks both
populations' dispersion equally (~0.51 fit/raw both) — the estimator PRESERVES,
mildly moderates, never creates the gap. Codex's dispersion-vs-amplification
question is hereby answered from the raw side: DISPERSION.
(2) NEW QUANTIFIED EPISTEMIC FACT about the flagship headline: on the jointly-
observed common cell subset, Aalto RAW shows optimized layouts -1.88%% (slower
than qwerty); the fitted +1.96%% is created by the model generalizing over the
30-35%% of each optimized layout's corpus mass that Aalto typists NEVER
produced. FRAMING (registered): this is NOT fabrication — raw rare-position
cells carry the OPPOSITE confound (practice deficits of qwerty's rare keys sit
exactly where optimized layouts put frequent letters), and the model's
position-generalization is the LOLO-validated mechanism; PRAC-DISC-1 already
showed the sign-flip is NOT the practice term (gains move <=0.5pp with b off).
But the claim's epistemic status is now precise: "+3.5%% saved" is a
MODEL-MEDIATED inference with NO raw within-population observational support —
the quantified, definitive argument for Phase-D / n=1 human validation.
(3) Fine rank order is LARGELY ESTIMATOR-GENERATED and typist-idiosyncratic:
raw-vs-fitted rank Spearman 0.07-0.11; the 4 community typists disagree among
THEMSELVES (per-typist board Spearman swings +0.88 to -0.17) — the raw-data
proof of unidentifiability at n=4.
Two recipe asymmetries noted (Aalto all-WPM+CAND4-d5 vs community [40,140)+
REG-d3) — compose with, do not cause, the population signal. NO published
number wrong; no bug. Feeds codex's mechanism-contract matrix (its 0c0b029).

### DIVERGENCE RCA — FINITE-SAMPLE ADDENDUM + REGISTERED WORDING CORRECTION
### (2026-07-17, report at state/keybo-divergence-finite-sample/report.md;
### community board reproduced bit-exact; 500+ draws/arm controls)
The control the RCA never ran (Aalto downsampled to community structure,
identical recipe, distributional): VERDICTS —
finite-n DEFLATES dispersion (community std 5.358 sits at ~P98 of the Aalto
n=4 confound-matched null, median 2.36; n-sweep 4->16 tightens toward Aalto's
2.9) — small n is NOT the magnitude mechanism; its real damage is rank
identifiability. LAYOUT<->TYPIST CONFOUND = PRIMARY magnitude driver: at
matched per-person volume community still std 4.52 vs Aalto nulls 1.71-2.36 —
it is WHICH 4 people (localized to ddn/alite), not count or volume; the Aalto
analog arm (4 people on 1 distinctive layout) reaches P95=7.1 — same mechanism
expressible inside Aalto. REGULARIZATION refuted (less-reg gives MORE spread).
EXTRAPOLATION = level-only (drop qwerty: ratio 1.826->1.635).
WORDING CORRECTION (supersedes "real population dispersion x amplifier" in
b50983d/57a1729): the defensible statement is — "~1.8x larger layout
differentiation AMONG THESE 4 PRACTICED TYPISTS, tight given them (bootstrap
CI [5.35,6.07]) but NOT population-identified (between-person variance is
unmeasured at 1 typist/layout and the Aalto analog shows it is huge);
the graphite-vs-P10 community top spot is n=4 noise (57/43 within-person
bootstrap, flips under LOCO and recipe); the 6-7%-vs-qwerty LEVEL is partly
extrapolated (qwerty support 0.50)." NOISE LAYERING reconciled: seed SD (0.04)
<< within-person bootstrap << between-person (unmeasured) — the confirmatory
workflow's "stable 15-sigma graphite>c30m" is stability GIVEN these data;
the finite-sample coin-flip is about WHICH data you were dealt. No number
miscomputed anywhere; the community dispersion/max-saved is OVER-INTERPRETED
if read as population fact or graphite endorsement. RCA remains: NO BUG.

### KAN-PRIME-1 + SELECT-1 — de-biased community gauges and the flagship-selection
### toolkit (2026-07-18; registered BEFORE any candidate was scored on them)
MOTIVATION (user directive): the community tools mix (T) hand-tuned time-proxy
terms — superseded by, and double-counting, the measured speed surfaces when
the tools sit beside them in a scalarization — with (S) flow-taste tables
(oxeylyzer-1 pays inrolls +250 vs alternates +40, redirects -340..-550: a
style axis with no registered evidential basis), and (C) mechanical-strain
terms claiming harm beyond time. Build primes = each tool restricted to C at
its NATIVE weights, plus the instruments to select THE flagship among
plateau-equivalent candidates.
CLASSIFICATION (from the parity-pinned ports, community.py):
genkey = 3.0*fspeed(T) + 1.0*LSB%(C) + 0.3*|index-balance|(C) — no S in stock
config; oxey1 = fspeed(T) + stretches(C) + pinky_ring(C) + trigram-table(S);
oxey2 = wfd(T) + stretch(C).
PRIMES: genkey' = 1.0*LSB% + 0.3*|index-balance| (lower better);
oxey1' = stretches + pinky_ring; oxey2' = stretch (both higher=better,
negative penalties). SENSITIVITY (registered): oxey1'+R additionally keeps the
redirect-only part of S (redirect-as-discomfort reading); selection arguments
must be robust to the +R swap or say so. EXACTNESS GATE: score() is now
defined as the sum of components(); the frozen binary goldens (G1/G2) plus
prime identity tests gate losslessness.
SELECT-1 INSTRUMENTS: (i) plateau gate — Aalto saved% (G4 convention, wpm 90,
seed-mean) within 0.10pp of the candidate-set max; fine speed differences
inside the gate carry NO selection weight (RCA: estimator-generated);
(ii) estimator stability — per-seed saved% spread; (iii) pace robustness —
saved% at wpm 70/110; (iv) RawSupport — % of corpus mass on raw-observed K31
position-ngram cells (serve = bucket 80 with >=10 samples, production cell
convention 40-140/20; any = >=1 sample, any bucket): high = the candidate's
claim rides on measurement, low = on extrapolation (operationalizes the RCA
raw-cell finding at candidate level); (v) adoption — unchanged/same-finger/
same-hand/zxcv counts vs qwerty30m + left-hand corpus mass; (vi) dominance —
pairwise wins + Pareto among plateau survivors on the axes [comm_saved,
genkey', oxey1', oxey2', sfb, sfs, lsb, redir, tri-serve-support,
unchanged-keys] (equal-weight axis count, no scalarization).
CANDIDATE SET (pinned before scoring): the 34 board rows of
runs/comm_opt1.json (sha256 0afd7e4103690372...) + qwerty30m. Raw cell TSVs:
bistrokes31_v1 0f2663ad6ed42aa5... / tristrokes31_cond_v1 46c6c3b1cc8919ad....
Driver: keybo-e2e/select1_board.py -> runs/select1_dossier.json.
DECISION FRAMEWORK (ADVISORY): gate on the plateau, then argue from primes +
support + adoption + robustness + dominance. The dossier produces a
RECOMMENDATION; flagship promotion remains USER-GATED. The COMM-SPD caveats
(extrapolated qwerty anchor; n=4 style fit) remain in force for comm_saved.

### SELECT-1 OUTCOME (2026-07-18; dossier runs/select1_dossier.json sha 7452656c328820f6...)
PLATEAU (0.10pp Aalto gate, best +3.58): BOTH-CHEB1, keybo-c30m, BOTH-CHEB0,
keybo-lsb, BOTH-CHEB8, BOTH-SEED-keybo-lsb. All six Pareto-survive; pairwise
wins on the 10 registered axes: keybo-lsb 4 = BOTH-SEED-keybo-lsb 4 >
BOTH-CHEB0 3 > keybo-c30m 1 = BOTH-CHEB8 1 > BOTH-CHEB1 0. The speed-column
king (BOTH-CHEB1, +3.58, seed-SD 0.031) takes ZERO wins — its in-gate speed
edge (registered as noise) is bought with the worst strain profile.
BIAS QUANTIFICATION (registered instrument): across the six survivors, 50.0%
of oxeylyzer-1's full-score spread comes from the flow-taste trigram table and
42.4% from the fspeed time-proxy double-count — 92% taste-or-double-counted,
8% strain content. The stripped tools are not a nicety; they change the answer.
KEY READINGS: keybo-lsb takes the best strain residuals among survivors
(genkey' 1.32, oxey1' -0.48e9, oxey2' -5.01e12 tied) AND the highest raw
support on the board (tri-serve 13.9% / bi-serve 72.4% — the least
model-mediated claim of any candidate). +R SENSITIVITY (disclosed per
registration): keeping redirect penalties flips the oxey1' top to
BOTH-SEED-keybo-lsb (redirect 2.04%, board best); the dominance tie (4-4) is
between keybo-lsb and this its own joint-pipeline sibling (13 slots differ,
same skeleton).
RECOMMENDATION (ADVISORY; promotion user-gated): keybo-lsb stays THE flagship.
Tiebreak vs its sibling: (i) highest raw support = most measurement-backed
claim; (ii) native-weight strain sweep; (iii) sfb 1.14 vs 1.69; the sibling's
case rests on comm_saved (+7.57 vs +6.90; caveated n=4 style-fit gauge) and
the redirect-as-discomfort reading. IF the user weights redirect comfort
heavily, BOTH-SEED-keybo-lsb is the named runner-up and is itself a keybo-lsb
variant — the flagship FAMILY is settled either way. Adoption axes do not
discriminate (all survivors are full remaps, zxcv lost, hand balance 46-51%).

### OCT-OPT-1 + POOL-1 — eight-gauge "beat keybo-lsb" search + all-data pooled
### model as gauge + POOL-SWAP (2026-07-18; registered BEFORE launch)
OCT-OPT-1 (user: optimize for everything — aalto, comm, genkey, genkey',
oxey1, oxey1', oxey2, oxey2' — priorities as make sense; goal = beat
keybo-lsb): 8-axis augmented Chebyshev (rho 0.05) through the exact
P17/COMM-OPT-1 machinery (SA 12x16k + 2-opt), stationary norms
(v-BEST)/(QREF-BEST) with prime anchors from the SELECT-1 dossier.
REGISTERED WEIGHTS: aalto 1.00 (the measured primary — only LOLO-validated
surface); comm 0.50 (real for its 4 typists; extrapolated-anchor + n=4
style-fit caveats); primes gk'/o1'/o2' 0.70 each (SELECT-1's plateau
discriminators — the de-biased strain content); full gk/o1/o2 0.25 each
(community-acceptance pressure, down-weighted per the SELECT-1 finding that
92% of their finalist spread is taste/double-count). 18 searches: 3 cold PRI
+ 5 warm PRI (keybo-lsb, lsb-sib=BOTH-SEED-keybo-lsb, BOTH-CHEB0, keybo-c30m,
P10-w0.5*) + MMX cold/warm + 8 Dirichlet draws; seeds 9994xx. Prime hot-loop
array evaluators are asserted EXACTLY equal to the KAN-PRIME-1 golden-gated
primes on 3 reference layouts before any search runs.
BEATS-KEYBO-LSB (registered criterion, computed in-driver): (a) Aalto
plateau — within 0.10pp of keybo-lsb's saved%; (b) SELECT-1 10-axis
head-to-head vs keybo-lsb — strictly-better > strictly-worse; (c) robust —
(b) also holds under the oxey1'+R swap. BEATS=true rows are promotion
candidates; promotion stays USER-GATED.
POOL-1 (user: model trained with ALL data, added as a gauge; repeat the swap
experiments with it): pooled surface = NATURAL pooling (each sample counts
once — Aalto-dominated by mass; composition logged in-driver) of Aalto v5
bigram (sha d6cb4c81...) + community bigram, and Aalto cond-v3 trigram (sha
1b5d7abd...) + community last-trigram (conditioned increments, compatible
semantics). Production recipe (REG_LOLO bigram / CAND4 trigram), 3 seeds,
T3P = T2+Tcond @ wpm 90, ROW_STAGGERED_30, geometry-only serving. Seed-mean
tables persisted to runs/pool_T3.npz — POOL becomes a STANDING gauge
(pool_saved% vs qwerty30m). POOL is NOT added as a 9th OCT axis (registered
rationale: it is a fixed mixture of the aalto and comm axes already present —
redundant as an optimization direction, informative as a gauge).
POOL-SWAP repeats the COMM-OPT-1 SWAP protocol exactly with POOL on the speed
axis (10 Dirichlet + MMX + 4 warm starts, seeds 9995xx; genkey/oxey1/oxey2
full tools as the other axes). Speed-axis anchors: BEST = min pooled total
over the comm_opt1 board + reference layouts (computed pre-search),
QREF = qwerty30m pooled total. Same BEATS verdict reported on its board.
CAVEATS (registered): pooling does NOT resolve the RCA population
non-identifiability — POOL is an evidence-weighted compromise gauge, not a
truth upgrade; its community component inherits all COMM-SPD caveats.

### OCT-OPT-1 OUTCOME (2026-07-18; runs/oct_opt1.json sha a49be5bed23926e2...)
FOUR candidates pass the registered BEATS-KEYBO-LSB criterion (plateau +
h2h majority + robust under +R): OCT-PRI-SEED-lsb-sib (+3.52 aalto, 7-3, R7-3,
layout pyuo,vdfnmhiea.cstlrj'-kzgwbqx), OCT-PRI-SEED-keybo-c30m (+3.50, 7-2,
R7-2, pyuo,vdmnlhiea.wstrcj'-kzgfbqx), OCT-PRI-SEED-keybo-lsb (+3.49, 6-3),
OCT-PRI-r1 (+3.43, 6-4). The +R gate did real work: two 6-4 rows (PRI-r0,
PRI-SEED-BOTH-CHEB0) failed it (5-5 under redirect-as-discomfort) and are NOT
candidates. All four winners carry the keybo-lsb skeleton (top two differ from
keybo-lsb in 12-14 slots; the lsb-sib-seeded run drifted BACK toward the
keybo-lsb basin — 12 vs its seed's 20 slots away).
WHAT THE WIN IS MADE OF (registered honesty): the candidates hold the Aalto
plateau and win on primes (gk' 0.87-0.89 vs 1.32; o1' -0.35..-0.42 vs -0.48;
o2' -4.5..-4.7 vs -5.01), redirect, lsb%, comm, (c30m-seed) tri-support —
while paying SFB 1.82-2.08 vs 1.14 and SFS ~7.7-8.0 vs 7.13. This is the
registered T/C design operating as intended: the primes deliberately do not
price same-finger content (the measured surface does), and the surface says
the SFB increase costs nothing measurable at wpm 90 (aalto holds). CAVEATS:
(1) criterion-relative — the search optimized axes that overlap the judgment
axes (registered design, disclosed); (2) the full tools disagree STRONGLY
(genkey 42-51 vs keybo-lsb's 31.7): community optics would read these as
worse layouts; the de-biased reading says that objection is 92% taste/
double-count, but the SFB trade is exactly where community intuition and our
model-mediated pricing diverge — an empirical question only Phase-D-style
data can settle; (3) plateau speed differences (+3.52 vs +3.51) remain noise.
DISPOSITION: candidates named; NO promotion recommendation change without
user decision on the crux: accept measured-surface SFB pricing (take
OCT-PRI-SEED-keybo-c30m, the best record 7-2/R7-2 with a raw-support WIN) or
keep community-legible SFB conservatism (keep keybo-lsb). USER-GATED.

### POOL-1 STAGE B OUTCOME — POOL-SWAP (2026-07-19; runs/pool_swap.json sha 8d8bbe95764c859e...)
BEATS-KEYBO-LSB: NONE. All 15 pool-optimized arms fail the Aalto plateau gate
(aalto +1.56..+2.39 vs required >=+3.41) while reaching pool +3.50..+4.24 —
the pooled surface's optimum is a DIFFERENT BASIN (top arm PSWAP-CHEB9 differs
from keybo-lsb in 28/30 slots), not a perturbation of the Aalto optimum.
Quantified tradeoff at the frontier: roughly 1 pool pp costs ~1.5-1.7 Aalto pp.
The community component, despite minority mass, relocates the optimum — the
RCA's 1.8x community amplitude operating at the margin. Notable: PSWAP-MMX and
PSWAP-SEED-graphite converged to the IDENTICAL layout (strong attractor);
pool-optimal layouts are respectable community-style boards (sfb 0.87-1.19,
gk 29-38), not degenerate exploits — the pooled surface is better-behaved as
an objective than PURE community was (gk 56), but still leaves the measured-
Aalto plateau. CONCLUSION (mirrors COMM-OPT-1): POOL earns its place as an
EVALUATION gauge (where it ranks lsb-sib +3.42 > keybo-lsb +3.14 > all OCT
winners +2.99..+3.09 — the standing contrary evidence on the OCT SFB trade)
but NOT as a search objective replacing Aalto. FOOTNOTE: reference row
P10-w0.5* shows aalto +3.63, the highest Aalto number yet printed — it was
never in the registered SELECT-1 candidate set and loses the h2h 4-6 vs
keybo-lsb (its speed edge is in-plateau noise by the registered rule; its
gauge profile is why SEL-1 passed it over). No candidate change. Flagship
recommendation unchanged: keybo-lsb, runner-up lsb-sib. USER-GATED.

### POOL-1 STAGE C — PURE-POOL diagnostic (2026-07-19; registered before run)
User request: the "just pool" layout. Mirror of COMM-OPT-1's PURE arm: argmax
pooled speed only (no other objective), 3 cold SA restarts (12x16k + 2-opt,
seeds 999600-2), DIAGNOSTIC frontier only — same status as PURE-community
(a985170): reveals what the pooled surface alone wants; no candidate
implications (the registered beats-criterion and flagship recommendation are
untouched). Output runs/pool_pure.json; full standard gauge rows reported for
PURE-POOL best and the canonical POOL+tools arm (PSWAP-MMX, the equal-weight
attractor that two starts converged to; PSWAP-CHEB9 noted as pool-max draw).

### RAWSUPPORT SPACE-INDEX BUG — correction + consequences (2026-07-20; fix 2f4cd82)
Found by the tb-verify red-team harness during the true-best-layout campaign.
RawSupport.support() hardcoded slot_of[' ']=30; but positions = [*slots(31),
space], so index 30 = the pinned quote-slot coord (6,2) and space is index 31
(0,0). Every space-adjacent n-gram (~34% bigram / ~50% trigram corpus mass) was
matched at the wrong coordinate and almost never counted as observed. CORRECTED
(space -> len(positions)-1): tri_serve keybo-lsb 13.86->38.51, lsb-sib
12.17->38.50, keybo-c30m 12.49->36.41, graphite 11.57->37.39, semimak 38.02,
OCT-c30m-seed 38.92; bi_serve all ~85-89%.
CONSEQUENCES FOR PRIOR CLAIMS (correcting the record):
- The SELECT-1 dossier + all-gauge boards used the BUGGY tri-support column. The
  headline "keybo-lsb has the HIGHEST raw support on the board" is RETRACTED: on
  corrected numbers keybo-lsb (38.51) and lsb-sib (38.50) are ~tied, and
  OCT-c30m-seed (38.92) is actually highest. Raw support no longer discriminates
  the flagship pair.
- Direction of the RCA's core epistemic finding is UNCHANGED and if anything
  strengthened: even at 38-39% observed trigram mass, ~61% remains model-
  extrapolated -> the "+3.5% is model-mediated, Phase-D is the only new truth"
  conclusion stands. (The buggy 12-14% overstated the extrapolation share but
  same direction.)
- SELECT-1's registered h2h semantics used the historical (buggy) support axis;
  tb-verify preserves those historical semantics for reproducibility AND reports
  corrected values separately. Re-scoring the flagship decision on corrected
  support is a manager TODO before any promotion; does NOT change that Aalto-
  plateau + primes still don't crown a single winner over keybo-lsb/lsb-sib.

### SWEEP-1 CHARTER — objective-weight sensitivity sweep (2026-07-20; registered BEFORE running)
Motivation: the true-best-layout campaign found 2 comfort-improved plateau candidates (direct
l<->m around keybo-lsb / lsb-sib) that hold the Aalto plateau (+3.492/+3.418pp) and cut the
scissor residual ~25%, but BOTH fail the registered SELECT-1 conjunction on G1 (cand1 LSB 61%,
cand2 scissor 81% of comfort attribution) under tb_objective's CURRENT weights. W1's reflection
established those comfort weights are OPEN (evidence-preserving default = ZERO; nonzero = sensitivity
arms), and W2 showed G1 verdicts FLIP under defensible reweighting (LSB 1->1/3: 73.5%->48%). So a
single-weight verdict cannot decide promotion. SWEEP-1 tests robustness.
CANDIDATE SET (pinned): keybo-lsb (ref), lsb-sib, keybo-lsb+lm (pyuo,vgdnmhiea.cstrlkj-z'fwbxq),
cand1 (pyuo,vgdnmhiea.fstrlkj-z'cwbxq), lsb-sib+lm, W4's 7 non-incumbent plateau layouts, W2's 5
plateau layouts (dedup by string). Inputs: tb_objective sha 03cb072e, tb_verify sha e610b2e @ 0f1cb99.
GRID (W1 arms): LSB weight {0,1/3,2/3,1} x hinge {1.75,2.00,2.25}u; scissor weight {0,1/3,2/3,1} x
non-adj {0.25,0.50,1.00}; SFS rho*weight {0,0.075,0.15}; row_jump {0,0.05,0.10}. HARD feasibility
(always on, never swept): Aalto plateau 0.10pp; pinky total<=17%, L<=9%, R<=9%.
PER CELL: comfort-rank the set; record plateau-feasibility, comfort-argmin, and tb_verify SELECT-1
conjunction (historical + corrected) per candidate.
DECISION RULE (registered before results):
- ROBUST WINNER: a single non-incumbent layout that is plateau+guard-feasible AND comfort-beats
  keybo-lsb in >=80% of grid cells AND beats it on the SELECT-1 conjunction in >=50% of cells ->
  a genuine successor PROPOSAL (weight profile to register = user choice).
- WEIGHT ARTIFACT: l<->m/candidates comfort-win only in the high-LSB or high-scissor corner
  (<80% of cells) -> incumbent holds; report the exact cells where it wins.
- HONEST NULL: no non-incumbent dominates across the grid -> keybo-lsb vindicated as robust.
Outcome is a PROPOSAL; promotion + registered weight profile remain USER-GATED.

### SWEEP-1 OUTCOME (2026-07-20; runs/sweep.json sha ad1441db, reviewer-APPROVED, 14x1296=18144 records)
VERDICT: HONEST NULL — no robust winner; keybo-lsb HOLDS. Not a high-corner artifact
(decision_summary confirms incumbent_holds=true, qualifying_layouts=[]). The registered
robust-winner bar (comfort-beat >=80% AND SELECT-1-corrected-beat >=50% of the 1296 weight cells)
is met by NO non-incumbent. Per-candidate across the grid:
- keybo-lsb+lm (pyuo,vgdnmhiea.cstrlkj-z'fwbxq): comfort-beats in 1296/1296 cells, hard-guards pass
  all, but SELECT-1-corrected 0/1296 -> comfort-only epsilon win, never clears the registered
  conjunction. The l<->m comfort tweak is REAL + robust-as-comfort but NOT a registered winner.
- cand1 (pyuo,vgdnmhiea.fstrlkj-z'cwbxq): comfort 924/1296, SELECT-1 0/1296.
- W4-plateau-5 (fyou,vgdnmheaikpstrlzj'.-cwbxq): the notable one — SELECT-1-CORRECTED 1296/1296
  (beats keybo-lsb 6-4 on the corrected conjunction in EVERY cell: better on comm/lsb/redir/sfs/
  tri-support/unchanged, worse on genkey'/oxey1'/oxey2'/sfb) BUT comfort 0/1296, and crucially
  beats_select1_HISTORICAL = FALSE (5-5 tie) with support_verdict_disagreement=TRUE. Its entire
  registered edge rides on the RAW-SUPPORT axis (corrected 38.9% vs 38.51%) — the very axis whose
  space-index bug was fixed in cd345e4. Under historical support it does not beat keybo-lsb. So it
  is NOT a robust winner: it is support-semantics-dependent AND comfort-inferior (higher sfb 1.69
  vs 1.14, worse strain primes), winning only by trading strain for comm/redir on a corrected-
  support tiebreak that flips under historical.
CONCLUSION: keybo-lsb is VINDICATED as robust against the full decomposed-objective + weight-
sensitivity search. No promotable successor found. W4's thesis confirmed: reweighting cannot
promote the epsilon-only l<->m candidates; W4-plateau-5 shows a candidate CAN out-point keybo-lsb
on the corrected conjunction but only by giving up comfort/SFB and on a support-semantics knife-edge
— not a clean win. OPTIONAL user-facing note: keybo-lsb+lm is a defensible comfort micro-variant
(~0.02pp Aalto, -25% scissor residual) a user could adopt for feel; it is not a speed upgrade.
NEXT LEVER (if pursued): a genuine successor needs NEW search territory with BALANCED axis gains,
not reweighting or local l<->m repair. Promotion + comfort-variant adoption remain USER-GATED.

### SWEEP-1 reflection addendum + registered policy gap (2026-07-20)
Sweep reflection (reviewer-verified) confirms the HONEST NULL is robust and W4-plateau-5 is
excluded on TWO independent grounds: (1) comfort strictly worse than keybo-lsb in all 1296 cells
(0/1296 wins; +0.183..+0.342 higher comfort loss) — the >=80% comfort bar excludes it by itself;
(2) its corrected SELECT-1 edge is driven SOLELY by tri_support flipping native h2h 5/5 -> 6/4.
REGISTERED POLICY GAP (does NOT affect this null; register before it can bite): the SWEEP-1 charter
did not explicitly choose support semantics or forbid support_verdict_disagreement. A future
candidate that clears the comfort bar AND beats SELECT-1 only under corrected (not historical)
support must remain USER-GATED pending an explicit preregistered support-semantics policy —
corrected is the bug-fixed truth, but a semantics-flip win is not self-approving. TODO before any
future promotion that hinges on it: register whether corrected support is the sole registered axis
(recommended, since historical embeds the space-index bug) or whether disagreement blocks promotion.
NEAR-MISS on record: W2-plateau-5 (boy,.gdmnlheiaupctrskqj-'fwvxz) comfort 1033/1296 (79.71%, 4
cells short of 80%) + corrected SELECT-1 1296/1296, but FAILS pinky guards (total 17.46% > 17%,
right 10.66% > 9%) and historical SELECT-1 0/1296 — correctly excluded, logged for completeness.
STRONGEST NEXT EVIDENCE (sweep's own conclusion, converging with every prior thread): a
preregistered blinded held-out HUMAN comfort/speed evaluation (Phase-D), NOT more weight cells or
more search. Durable artifact: state/keybo-optimization/artifacts/sweep1_result.json (sha ad1441db).

### FRESH-1 CHARTER — fresh-territory balanced-gain successor search (2026-07-20; before running)
Motivation: SWEEP-1 proved keybo-lsb robust; the exhausted levers were reweighting + local l<->m
repair. W4's thesis: a genuine successor must come from NEW search territory with BALANCED axis
gains (not 61%-on-one-axis like the rejected candidates), clearing the registered SELECT-1
conjunction AND robust under oxey1'+R. FRESH-1 searches for exactly that.
OBJECTIVE: search [Aalto-speed x tb_objective-comfort] (frozen tb_objective sha 03cb072e) but with
TWO differences from OCT/W2/W4: (1) run at MULTIPLE registered weight profiles from the SWEEP-1 grid
(default-zero + a few defensible mid arms), NOT one; (2) require BALANCED attribution as a search
constraint — reject any optimum where a single comfort axis supplies >50% of the comfort gain
(the G1 guard, applied DURING search not just after). Diversity: many cold restarts from RANDOM
permutations + basin-hopping / large-kick restarts to ESCAPE the keybo-lsb basin (report positional
Hamming from keybo-lsb; target genuinely distinct optima, Hamming >= 15).
HARD FEASIBILITY (always on): Aalto plateau within 0.10pp of keybo-lsb (3.4129 floor); pinky
total<=17%, L<=9%, R<=9%.
CANDIDATE OUTPUT: every distinct plateau+guard-feasible optimum with full tb_objective decomposition,
per-axis attribution (prove balance), corrected+historical raw support, pinky L/R, and the grid-cell
robustness (does it comfort-beat AND SELECT-1-beat keybo-lsb across the profiles it was found under).
DECISION RULE (registered): a FRESH candidate is a promotion PROPOSAL iff it is plateau+guard-
feasible, balanced (no axis >50% of comfort gain), beats keybo-lsb on the registered SELECT-1
conjunction under CORRECTED support in >=50% of the SWEEP-1 grid cells, AND does not lose under
historical support (no support_verdict_disagreement) — closing the SWEEP-1 policy gap. Otherwise
report HONEST NULL (keybo-lsb final). Promotion + support-policy remain USER-GATED.
Driver keybo-e2e/fresh_search.py -> runs/fresh_search.json; manager runs the final verify() gate +
registers outcome. Gate note: tb_verify pins repo 0f1cb99; run its gate from that pinned state or
bump the pin (doc commits since are scoring-invariant).

### AXIS-2 CHARTER — objective redesign from the completeness audit (2026-07-20; design @ report sha 118374432)
Motivation: user found (a) SELECT-1 blind to real scissor reductions (keybo-lsb+lm undervalued),
(b) community model judged but never optimized. Audit (keybo-axis-audit) delivered a full redesign
honoring the anti-double-count + RCA-caveat discipline. Registered design (implementation = manager,
promotion = user-gated):
FIRST-CLASS AXIS CHANGES to tb_objective:
1. SCISSOR -> a first-class FAMILY of 6 pair-specific leaves (index-middle ... ring-pinky,
   middle-pinky = the demonstrated blind spot keybo-lsb+lm cuts ~55%) under ONE shared bounded
   budget lambda_SC in {0,.05,.10} center .05 (NOT 6 additive votes). Fitted pair/direction ms
   REMOVED from the neutral comfort arm (Aalto owns timing) — kept only as diagnostic sensitivity.
2. row_jump (generic vertical, w=.10) -> REPLACED by bounded STATIC DISLOCATION/reach
   D=sum_c p(c)*(d_stagger/Dmax_f)^r, r in {1,2}, lambda_D in {0,.05,.10,.20} center .10. Do NOT
   also keep off-home/bottom-row/vertical/WFD scalars (double-count).
3. pinky_load -> REPLACED by bounded ALL-FINGER capacity L=sum_f w_f z_f^2/(1+z_f^2),
   z_f=[load_f/c_f-1]_+, c_f=kappa*m_f/sum(m), m=(.6,.85,1,1,1,1,.85,.6), kappa in {1,1.10,1.25}
   center 1, lambda_L in {0,.10,.25,.50} center .10. Keep hard pinky guards.
4. LSB retained residual-only, lambda_LSB in {0,.05,.10} center .05, hinge h ~1.5u.
5. ZERO-weight DIAGNOSTICS (kept out of the objective to avoid re-charging Aalto timing/taste):
   rolls, redirects, alternation, generic row/adjacency/travel, bottom-row curl, hand-balance,
   higher-order flow. (Answers "what other axis are we missing" — these were considered and
   deliberately excluded as double-counts, not oversights.)
COMMUNITY IN SEARCH (user directive "we should"): enters ONLY as an anchor-free CLIPPED finite
reward behind the FIXED 0.10pp Aalto plateau guard: F1 = min J0 - lambda_C*z_community,
lambda_C=0.014650 == max 0.05 Aalto-pp reward, beta in {0,.25,.50}. Never co-equal, never
lexicographic, CANNOT pay for mechanics or leave the plateau (0.05pp << POOL-SWAP's 1.6pp basin
gap — so it cannot reproduce basin escape). Breaks only mechanically-equivalent ties.
SELECT-1 REDESIGN: ten equal votes -> grouped epsilon-Pareto, NO-compensation: scissor counts ONCE
with pair/bin vetoes (fixes the blindness); mixed-mechanics = HOLD; raw-support = epistemic (not a
comfort vote); adoption separate; robust n=4 community improvement breaks only mech-equivalent ties.
OPEN arms (all preregistered sensitivity, default-defensible): comfort-family weights/curves,
scissor epsilon/mass + neutral severity, dislocation r/lambda, capacity kappa/lambda, community
saturation/materiality/LOCO robustness. NEXT: manager implements tb_objective v2 + SELECT-1 v2
(TDD, golden-gated), re-scores keybo-lsb / keybo-lsb+lm / lsb-sib+lm / FRESH-1 near-miss under it,
THEN a preregistered scissor-priority FRESH-2 (gross-positive attribution + direct pinky-scissor +
no-regression guards on sfs/dislocation/capacity). Promotion + weight-profile choice USER-GATED.

### AXIS-2 v2 RE-SCORE OUTCOME (2026-07-20; artifacts state/keybo-optimization/artifacts/v2/, tb_objective_v2 sha 55a55105, 27 TDD tests, v1 frozen 03cb072e untouched)
Implemented tb_objective_v2 + select1_v2 (AXIS-2 charter be7e3e4) via TDD; re-scored the flagship
set under the scissor-aware, comm-aware objective (AXIS-2 CENTER profile). Manager-verified results:
- keybo-lsb+lm is the COMFORT SCALAR WINNER: scissor total -27.7% (0.6195->0.4480), middle-pinky
  leaf -56% (0.167->0.073), SFB/SFS tie-or-better, LSB tie. On the aggregate it IS more comfortable.
- BUT SELECT-1 v2 verdict = HOLD (not promotable), REFUTING the manager's clean-improvement
  prediction. Root cause (verified at sub-bin): l<->m does not only remove middle-pinky scissors, it
  MOVES some into a worse-oriented bin — middle_pinky|top_to_bottom|adverse|nonadjacent regresses
  +537% (0.0054->0.0342, mass 0.057). No-compensation gate vetoes on it.
- CRITICAL (from v2impl reflection): that veto is DECISION-CRITICAL on ONE open parameter,
  scissor_bin_epsilon: at 0.10 -> HOLD, at 0.15 -> PROMOTE. And the regressing bin uses the 0.60-of-
  neutral OPEN factor; its 7.765% neutral regression only becomes the 12.122% veto against the
  smaller open-arm denominator. The v2impl reflection judges this bin "too fine-grained to prove
  ergonomic harm" -> HOLD is preregistered UNCERTAINTY, NOT evidence keybo-lsb+lm is worse.
- near-miss cnfgk'...: HOLD confirmed (ring-pinky neutral regression 11.44%) — genuine, coarser-bin.
INTERPRETATION (honest): under a scissor-aware objective the old "keybo-lsb is better" verdict is
GONE. keybo-lsb+lm is a Pareto-frontier PEER — comfort-scalar-better, blocked from outright
promotion only by a fine-grained posture bin whose harm is unproven and whose veto flips on one OPEN
epsilon. This is a HUMAN VALUE CALL, not a pipeline verdict.
USER-GATED DECISION (surfaced, not taken): (a) adopt keybo-lsb+lm as the flagship/daily-driver
(scalar-better, the scissor win you flagged); (b) keep keybo-lsb (conservative, avoids the adverse-
posture bin); (c) set scissor_bin_epsilon policy (0.10 vs 0.15) which mechanically decides HOLD-vs-
PROMOTE; or (d) run the preregistered scissor-priority FRESH-2 first to see if a layout cuts
middle-pinky WITHOUT the adverse-posture side effect (would dominate both). Implementation committed
by manager; promotion + epsilon policy = user.

### FRESH-2 CHARTER — scissor-priority successor search on the v2 objective (2026-07-20; before running)
Motivation: AXIS-2 v2 re-score left keybo-lsb+lm a comfort-scalar winner but SELECT-1 HOLD, blocked
ONLY by a fine adverse-posture bin (middle_pinky|top_to_bottom|adverse|nonadjacent +537%) — i.e.
the l<->m swap cuts middle-pinky scissors but shifts a sliver into a worse-oriented posture. FRESH-2
searches for a layout that captures the scissor reduction WITHOUT that side effect, which would
DOMINATE both keybo-lsb and keybo-lsb+lm and dissolve the value-call ambiguity.
OBJECTIVE: search [Aalto x tb_objective_v2 comfort] using the frozen v2 objective (tb_objective_v2
sha 55a55105) + select1_v2 gate — NOT the v1 objective. Multi-profile over the AXIS-2 grid arms
(center + defensible mid arms). SCISSOR-PRIORITY acceptance rule (the FRESH-1-reflection fix,
registered here): (1) attribution by GROSS-POSITIVE (not signed-net); (2) a DIRECT pinky-scissor
objective term is active; (3) NO-REGRESSION guards vs keybo-lsb on sfs, dislocation, capacity, AND
on every scissor SUB-BIN (no bin may regress beyond epsilon — this is the guard that would have
caught the keybo-lsb+lm adverse-posture shift). HARD feasibility: Aalto plateau 0.10pp; pinky
total<=17/L<=9/R<=9.
DIVERSITY: cold random + large-kick basin-escape restarts; report Hamming from keybo-lsb AND from
keybo-lsb+lm. Target genuinely distinct optima.
DECISION RULE (registered): a FRESH-2 candidate is a promotion PROPOSAL iff plateau+guard-feasible,
scissor-total strictly better than keybo-lsb, NO scissor sub-bin regresses beyond epsilon, no
regression on sfs/dislocation/capacity, AND select1_v2 = PROMOTE (not HOLD) vs BOTH keybo-lsb and
keybo-lsb+lm under the CENTER profile with epsilon at BOTH 0.10 and 0.15 (robust to the pivotal
open param). Else HONEST NULL — in which case keybo-lsb+lm (comfort-scalar winner) stands as the
best available and the choice reverts to the registered user value call. Promotion USER-GATED.
Driver keybo-e2e/fresh2_search.py -> runs/fresh2_search.json; manager runs final gate + registers.

### FRESH-2 OUTCOME (2026-07-21; artifact state/keybo-optimization/artifacts/fresh2_search_result.json sha 4d0badf9)
VERDICT: HONEST NULL — no dominator. 37 distinct feasible optima (345 distinct layouts searched;
condition intersections: aalto_plateau 112, pinky_limits 243, strict_scissor_improvement 351,
mechanical_no_regression 123, all_scissor_bins 136, composite_feasible 42). ALL 37 successfully
AVOID the adverse-posture regression that held keybo-lsb+lm at HOLD (worst sub-bin share <=0.0992
vs keybo-lsb+lm 0.1212) — so the no-sub-bin-regression guard is satisfiable — but NONE achieve
select1_v2 = PROMOTE against BOTH keybo-lsb AND keybo-lsb+lm at BOTH epsilon 0.10 and 0.15. So a
layout can fix the posture bin, but only by conceding elsewhere (no candidate dominates both
incumbents across the epsilon-robust conjunction).
CONSEQUENCE (registered): keybo-lsb+lm remains the comfort-SCALAR best-available; the registered
user value call stands. TWO independent scissor-aware searches (FRESH-1 balanced, FRESH-2 scissor-
priority) now both return null -> no promotable successor to the keybo-lsb family exists under the
v2 objective. The flagship question is CLOSED on the modeling side.
FINAL CAMPAIGN STANDING: keybo-lsb (registered pick, robust) and keybo-lsb+lm (comfort-scalar
winner: -27.7% total scissor / -56% middle-pinky leaf, ~0.02pp Aalto = noise) are a Pareto pair;
neither dominates. keybo-lsb+lm's only debit is one adverse-posture sub-bin whose harm is unproven
(v2impl reflection: "too fine-grained to prove ergonomic harm") and whose veto flips on the OPEN
scissor_bin_epsilon (0.10 HOLD / 0.15 PROMOTE). NO further search or reweighting can resolve which
is truly better -> the deciding evidence is Phase-D human validation (converges with every prior
campaign thread). USER-GATED: flagship promotion (keybo-lsb vs keybo-lsb+lm), scissor_bin_epsilon
policy, and Phase-D go/no-go.

### RANK-1 CHARTER — frontier RANKING beyond epsilon-Pareto (2026-07-21; before running)
Motivation (user): epsilon-Pareto only reaches the FRONTIER (removes dominated layouts); it does
NOT rank points ON it, so it returns HOLD/ties instead of THE best. The frontier is a high-dim
tradeoff surface, but better/worse points still exist on it. Reaching the best requires a
PRINCIPLED preference structure over tradeoffs — NOT arbitrary weights (the KAN-PRIME artifact
trap) and NOT refusal-to-weight (indecision). RANK-1 builds robust frontier ranking.
CANDIDATE SET: the epsilon-Pareto frontier from the campaign — keybo-lsb, keybo-lsb+lm, lsb-sib,
lsb-sib+lm, keybo-c30m, the 37 FRESH-2 feasible optima (artifacts/fresh2_search_result.json), the
FRESH-1 near-miss, + community refs graphite/semimak-jq for calibration. Dedup by string; keep only
non-dominated (compute the frontier explicitly first).
METHOD (robust MCDA, not a single scalar):
1. Characterize the frontier: which layouts are actually non-dominated on the v2 axes (aalto,
   6 scissor leaves, dislocation, capacity, sfb, sfs, lsb; comm as clipped gauge). Report the
   true frontier set + each layout's tradeoff signature.
2. Rank the frontier by ROBUST preference, not one weighting: for each frontier layout compute
   (a) the SHARE of the defensible weight-simplex under which it is the argmin (a layout best over
   a larger volume of reasonable preferences is more defensible), (b) worst-case REGRET vs the best
   at each weight (minimax-regret pick), (c) how the ranking moves under the OPEN scissor_bin_epsilon
   and severity tiers. Constrain the weight family by whatever ERGONOMIC THEORY / literature bounds
   it (e.g. scissor>speed-proxy priority the user asserts; pinky severity ordering) — state every
   constraint and mark unjustified ranges OPEN.
3. Deliver THE best layout under the robust framework + a full ranked frontier, with an HONEST
   sensitivity statement: is the top pick preference-robust (wins across most of the family) or
   preference-dependent (and if so, on which axis's weight does it hinge)?
DECISION RULE: RANK-1 produces a RANKED RECOMMENDATION, not an auto-promotion. If a layout is
robustly best (wins the plurality of the defensible simplex AND minimax-regret), it is the
recommended flagship PROPOSAL. If the top depends on an unresolved preference (e.g. exactly how
much scissors outweigh SFB), name that preference as the decision the human/Phase-D must settle.
Promotion + the pivotal preference weight remain USER-GATED. Driver keybo-e2e/rank1.py ->
runs/rank1.json; read-only on repo; manager reviews. Honest partial-order result is valid.

### RANK-1 FAN-OUT (2026-07-21) — decompose "reach the best layout" into 4 parallel --codex workstreams
User: break the problem down further, more agents. RANK-1 (robust MCDA on current frontier) keeps
running; 4 orthogonal sub-problems added, all read-only/own-scratch/commit-nothing, manager integrates:
- keybo-frontier-map: is the 37-point frontier COMPLETE? dense NSGA-II/Pareto-archive map of the
  scissor-vs-SFB tradeoff region; find knee-points + any new non-dominated layouts between incumbents.
- keybo-weights: derive the DEFENSIBLE weight family (ratio bounds per axis-pair) from community doc +
  biomechanics + our measured surface + user priors — constrains RANK-1's simplex; names the pivotal
  unresolved ratio.
- keybo-voi: value-of-information — which weight-uncertainty FLIPS which pairwise winner at what
  threshold; design the MINIMAL Phase-D experiment resolving exactly the pivotal flip (tie to 9/19
  conformal sizing).
- keybo-rank-adversary: red-team the emergent best — reproduce every axis, sweep OPEN constants
  (scissor_bin_epsilon/severity/non-adj/kappa) for winner-flips, re-check double-count/hidden-regress.
Integration: manager folds frontier-map (is the frontier complete) + weights (constrained simplex)
into RANK-1's robust ranking, uses voi to name the decision + minimal Phase-D, and gates on
rank-adversary before any recommendation. Output = THE best layout + robustness statement + the one
Phase-D question. Promotion USER-GATED.

### FREQ-FEAT-1 CHARTER — frequency-as-FEATURE permutation sweep, 3 models (2026-07-21; before running)
User: explore adding frequency as a predictive FEATURE (does knowing an n-gram is common change
its keystroke time — practice/familiarity signal?) at the 1gram/bigram/trigram/skipgram levels,
in every on/off permutation, across the Aalto, community, and pool models; let the agent DISCOVER
which levels help vs harm.
PRIOR ART THE AGENT MUST BUILD ON (not re-run naively): OQ-1 (2026-07-04, ledger + schema.py:8 +
agent-artifacts/OQ1-frequency-feature.md) ALREADY tested freq-as-one-lump-feature on AALTO and
REJECTED it: with 98.7%-qwerty data the freq feature becomes a per-POSITION MEMORIZATION KEY that
improves held-out MAE/rho but CORRUPTS cross-layout ranking (layout-level Kendall tau collapsed
+0.667 -> +0.333). Frequency was therefore confined to (a) objective WEIGHT and (b) the additive
PRACTICE TERM (train.py residualizes a bigram-keyed practice offset out of the target). FREQ-FEAT-1
is a legitimate EXTENSION because: (i) OQ-1 lumped frequency; the per-level 1/2/3/skip permutation
is finer; (ii) OQ-1 was Aalto-only — the COMMUNITY (4 typists x 4 distinct layouts) and POOL data
regimes are NOT 98.7%-qwerty, so the memorization mechanism may not bind there.
DECISION METRIC (non-negotiable, the OQ-1 lesson): the decisive metric is LEAVE-ONE-LAYOUT-OUT
layout-ranking Kendall tau (cross-layout generalization), NOT held-out MAE/rho — MAE REWARDS the
memorization that ranking exposes. A permutation is ADOPTED for a model only if it strictly beats
the current geometry-only baseline on LOLO tau (>= baseline, tie-break beats-baseline count),
across >=3 seeds, AND does not regress the layout-search null-space check. Any arm that wins MAE
but loses/ties tau is REJECTED (the registered OQ-1 drop branch).
SWEEP: for each model in {Aalto, community, pool}: the 2^4 = 16 on/off permutations of frequency
injected at {1gram, bigram, trigram, skipgram} level (as log-freq or normalized-freq features,
agent picks the encoding + justifies). Report per-model per-permutation LOLO tau + MAE + the
memorization diagnostic (does the freq feature's SHAP/gain concentrate on position-identity?).
Frequency source = the independent English corpus (data/corpus/, NOT the Aalto stimulus).
OUTPUT: which (model, level-set) permutations HELP tau, which HARM, the winning config per model,
and whether ANY beats the current freq-NOT-a-feature production. If all lose tau (OQ-1 replicates
at finer grain), that clean negative is the valid result. Read-only repo + own scratch; commit
nothing; manager reviews + integrates; any schema/model change is user-gated (FEATURE_VERSION bump).

### FREQ-FEAT-1 OUTCOME (2026-07-21; merged artifact, safe-panel) — OQ-1 REPLICATES at finer grain
VERDICT: frequency-as-a-FEATURE REJECTED for the optimizer across ALL THREE models (Aalto,
community, pool). 0/32 permutations beat production LOLO layout-ranking tau in ANY model (baselines
1.0/1.0/0.857 — the top-tier arms all tie at ceiling, none exceed). Meanwhile EVERY model's best arm
improves held-out MAE (Aalto 22.81->19.03, community 32.44->29.83, pool 25.09->22.14). That MAE-down/
tau-flat split IS the registered OQ-1 memorization signature — now confirmed at PER-LEVEL grain AND
in the non-qwerty community/pool regimes (where I'd flagged the mechanism might not bind — it does).
All six fitted 128x2048 panels PASS exact placement invariance; frequency contributions remain
POSITION-ASSOCIATED (observational confounding persists), so a freq feature is a memorization/
practice-fit channel, not a ranking-relevant geometry signal — exactly OQ-1's finding.
LEVEL DIRECTIONS (MAE, informational): Aalto first+second-bigram help, trigram helps valid weighted
pairs but extrapolates badly on invalid ones, skipgram HARMS, unigram mixed; community unigram/
first-bigram/trigram/skipgram help MAE, second-bigram harms; pool bigrams+trigram help, unigram/
skipgram mixed.
ONE EXPLORATORY NON-NULL (not adopted): a POST-to_ms ADDITIVE community frequency correction
(1gram+bigram1+trigram+skipgram) improves community mean MAE/WMAE/log-WMAE but 2/4 held layouts
REGRESS -> fails the no-regression bar; needs independent/nested confirmation before it could be
considered, and it is a serve-time additive (like the practice term), NOT an optimizer feature.
CONCLUSION: the production design (frequency = objective WEIGHT + additive practice term, NOT a
model feature) STANDS, now validated at finer grain and across all three models. No schema change;
FEATURE_VERSION unchanged. Answers the user's per-level/per-model question definitively: no
permutation is safe to add as an optimizer-facing feature.

### FREQ-FEAT-1 OUTCOME — CORRECTION (2026-07-21; supersedes the prior entry's framing)
The prior entry overstated the result and used an unsatisfiable criterion. Corrected read from the
merged artifact:
- Aalto + community baseline LOLO tau = 1.0 (SATURATED — only ~4-5 held-out layouts, tau pins to
  ceiling). "0/32 beat production tau" there is VACUOUS: nothing can beat 1.0. Checked the right
  question instead — did any freq arm DROP tau? NONE did (all 32 stay 1.0 on both). So on Aalto/
  community the honest finding is: MAE improves (Aalto 22.8->19.0, comm 32.4->29.8) with NO
  detectable ranking harm. That is NOT evidence against frequency; the ranking test simply has no
  resolution at ceiling.
- POOL is the only NON-saturated ranking test (baseline tau 0.857). There, 18/32 freq arms DROP tau
  to 0.786 — i.e. where the metric CAN discriminate, frequency features HURT cross-layout ranking.
  This is the real evidence, and it matches the OQ-1 memorization mechanism.
CORRECTED VERDICT: do NOT adopt frequency as an optimizer feature by DEFAULT (pool shows ranking
harm; Aalto/comm tau cannot rule out hidden harm behind the ceiling; MAE alone rewards memorization).
BUT the earlier "definitively rejected at finer grain across all regimes" was TOO STRONG: on Aalto/
community there is no measured ranking harm and a real MAE gain, so the case is AMBIGUOUS there, not
negative. ROOT LIMITER = tau saturation from too few held-out layouts (a layout-DIVERSITY / Phase-D
problem, not a frequency problem); with more layout diversity the Aalto/comm ranking test could
become discriminating and the question should be RE-OPENED. Production design (freq = weight +
practice term) stands as the conservative default, not as a proven-optimum.

### SELECT-METHOD-1 CHARTER — better model-selection methodology + near-miss revival (2026-07-21; before running)
User: improve HOW we select better models, and RETRY promising experiments to actually get better
models — including COMBINING two experiments that each fell just short of the bar (their gains may
sum past it). Motivated by three findings this session:
(A) rho-bar-is-noise: the adoption rule "ρ/ceiling must beat incumbent by +0.005 (or +0.02)" uses
    UNDERIVED round-number bars almost certainly INSIDE rho's own sampling noise. A participant-
    level bootstrap CI on rho ALREADY EXISTS (validate.py::_bootstrap_rho_ci) but was NOT used as
    the bar — so past "rejected, ρ/ceiling 0.964 < bar" decisions may be NOISE-rejections.
(B) tau-saturation: with ~4-5 held-out layouts LOLO tau pins to 1.0 and cannot discriminate top
    arms; the pipeline leaned on rho/ceiling to break ties but with a noise-width bar (see A).
(C) deferred near-misses never revisited: P-MED/P-M5 (LIFTED tau 0.67->1.0 AND improved wmae/umae,
    rejected on an UNEVALUABLE different-frame rho compare — matched-frame follow-up registered as
    future work, never done); the FREQ-FEAT-1 post-to_ms additive community correction (improved
    mean MAE, 2/4 layouts regressed); possibly others in the P2/target-null/robustness families.
MANDATE (deliver a DESIGN + evidence, manager implements/commits, schema+production changes user-gated):
1. STATISTICAL BAR REDESIGN: replace the fixed ±0.005/±0.02 rho/ceiling bars with the MEASURED
   bootstrap-CI-aware rule — adopt only when a metric gain clears its own participant-level bootstrap
   CI (or a registered effect-size floor derived from that CI), and declare everything inside the CI
   a genuine TIE broken by a substantive criterion (rare-ngram guard, raw support, tau if non-
   saturated) NOT by noise. Quantify rho/ceiling's actual CI width on our folds so the new floor is
   evidence-based. Also address tau saturation: propose the discriminating metric to use WHEN tau is
   at ceiling (and flag that true resolution needs Phase-D layout diversity).
2. NEAR-MISS AUDIT + REVIVAL: enumerate every past experiment that improved a magnitude/accuracy
   metric but was rejected on a sub-CI margin or an unevaluable compare (P-MED/P-M5, freq additive,
   scan the ledger for the pattern). Re-adjudicate each under the redesigned bar; re-run the ones
   whose original compare was unfair (e.g. P-MED/P-M5 matched-frame). Report which flip to ADOPT.
3. COMBINATION EXPERIMENTS (the user's key lever): for pairs of near-miss arms that each improved a
   DIFFERENT sub-metric or cell region (e.g. a head-MAE winner + a rare-decile winner; T-CAP +
   cand-4; a bigram-level + trigram-level freq additive), test whether STACKING them clears the
   redesigned bar when neither did alone. Prereg the pairs + the combined-adoption rule BEFORE
   running; require the combination to pass the rare-ngram guard AND not regress tau.
GUARDRAILS: decisive metric is still cross-layout generalization (tau where it resolves, else the
CI-aware rho/ceiling + rare-ngram guard); NEVER adopt on head-MAE alone (the T-MAE / memorization
failure); all model retrains via the LOLO harness; frequency source = independent English corpus.
Read-only repo + own scratch; commit nothing; any production model/schema change is USER-GATED.

### SELECT-METHOD-1 OUTCOME (2026-07-22; artifact selmethod.json sha 22cafed5, harvested)
1. SHIPPED BOOTSTRAP DEFECT (production bug, fix pending manager commit): validate.py::
   _bootstrap_rho_ci is DEGENERATE — observed CI [0.0, 0.0] width ZERO. Root cause: replacement
   draws converted to a set (multiplicity lost) + full-sample observations reused. RED test written;
   corrected paired participant-bootstrap implemented in agent scratch.
2. MEASURED TRUTH ABOUT THE OLD BAR (validates the user's critique quantitatively): corrected
   marginal CI widths across 8 fold/seeds span 0.0068-0.1318 (p95 half-width 0.0652; conservative
   legacy unpaired floor 0.1648). The old +0.005 adoption margin = 3.8% of the p95 CI width — pure
   noise-scale. The old +0.02 similarly. Registered NEW BAR: adopt only when the PAIRED participant-
   bootstrap challenger-minus-incumbent rho/ceiling 95% CI LOWER BOUND > 0, with rare-ngram and
   resolvable-tau guards; anything inside the CI = TIE (broken by substantive criteria, not noise).
3. NEAR-MISS RE-ADJUDICATION (honest null): NO historical rejection flips to ADOPT under the honest
   bar. MED/M5 remain rare-decile rejects (matched-frame); CAP4 rejects MORE decisively (paired rho
   delta -0.0057, CI [-0.0070,-0.0037] entirely negative, rare +3.9%); TRI-PS stays REOPEN/HOLD.
4. COMBINATION EXPERIMENT (the user's stacking lever, preregistered CAP4+MED): does NOT clear —
   rho +0.0015 CI [-0.0061,+0.0052] (straddles 0 = tie), WMAE -8.53% (real head gain) but rare-
   decile +5.62% CI [+3.62,+5.76] -> the rare-ngram guard fires decisively. REJECT. No combination
   clears; no better model produced.
CONCLUSION: the methodology upgrade is REAL (degenerate CI fixed, evidence-based bar registered,
old margins proven noise-scale) and the model-selection verdicts are UNCHANGED — the incumbent
models stand, now for statistically defensible reasons rather than lucky round-number bars.
Follow-ups: (a) manager to review+commit the bootstrap fix to src (user-gated as a production
change? — it is a pure bug fix to a diagnostic; commit under standing test-fix authority, note
here); (b) whether old-decision robustness was coincidence = reflection question in flight.

### SELECT-METHOD-1 follow-up: TRI-PS+FREQ-PRIOR is a DEFERRED transfer-winning near-miss (2026-07-22; tri-frequency.json c7eead88)
Extended sweep (BASE/TRI-PS/FREQ-PRIOR/combo/NGRAM-FE/JOINT-SHAP, paired K31 LOLO under the
corrected bootstrap). Findings:
- TRI-PS+FREQ-PRIOR = the STRONGEST cross-layout transfer arm tested: rho/ceiling +0.02845 (95% CI
  [+0.02126,+0.03244] — clears zero, a REAL generalization gain under the fixed bootstrap), rare-cell
  MAE -4.12%, matched geometry-delta -2.53%, exactly frequency-invariant. It also optimizes to a
  layout 0.024% better than BASE's own plateau on BASE's mean surface.
- REJECTED on ONE guard: optimizer model-seed stability (combo max-seed regret 0.14850% > candidate
  search tolerance 0.07737%). BUT that tolerance is NOT a calibrated materiality bar — it is the MAX
  of three mean-surface searches compared against SEED surfaces that had one search each (apples-to-
  oranges), and BASE ITSELF FAILS the symmetric test (BASE seed0 regret 0.05005% > its 0.02006%
  tolerance — verified in the artifact). So the incumbent would be disqualified by the same rule.
- VERDICT: same class as the rho-bar noise-floor fix — a real transfer win blocked by an
  un-calibrated stability tolerance. TRI-PS+FREQ-PRIOR adoption is DEFERRED (not rejected) pending a
  registered EQUAL-REPEATS / materiality audit of the seed-stability bar (equal search counts per
  surface; a materiality threshold derived from measured seed-regret spread, not a raw max). If the
  instability proves immaterial under that audit, TRI-PS+FREQ-PRIOR is a genuine candidate to IMPROVE
  the production model's cross-layout transfer.
- JOINT-SHAP subtraction REJECTED cleanly (fails fixed-geometry intervention by 46.24 ms).
- 'koyu,vdmnlheiapcstrfqj-.'gwbxz' = BASE.mean rng888301 plateau sample (the pre-existing production
  optimum on one seed), NOT a new winning layout — do not report as top-tier.
CONCLUSION: BASE production model retained for now; TRI-PS+FREQ-PRIOR is the first genuinely
promising model-improvement lever the campaign has surfaced, parked on a fixable bar. Registered as
the priority follow-up. User-gated: whether to run the equal-repeats stability audit that could
unblock it.

### STAB-AUDIT-1 CHARTER — equal-repeats materiality audit of TRI-PS+FREQ-PRIOR (2026-07-22; before running)
Motivation: TRI-PS+FREQ-PRIOR is a verified transfer-winning near-miss (rho/ceiling +0.0285 CI
[+0.021,+0.032], rare -4.12%) DEFERRED only on optimizer seed-stability (combo 0.1485% > tol
0.0774%). That tolerance is UNCALIBRATED: it is the MAX of 3 mean-surface searches vs seed surfaces
with 1 search each (apples-to-oranges), and BASE itself FAILS the symmetric test (seed0 0.0500% >
its 0.0201% tol). This audit re-decides adoption under a FAIR, calibrated stability bar.
METHOD: (1) EQUAL search repeats per surface — run the SAME number of independent SA searches (>=3,
same 12x16k+2opt config) on EVERY model-seed surface for BOTH BASE and TRI-PS+FREQ-PRIOR, so
seed-instability is measured identically (removes the max-of-3-vs-1 artifact). (2) Derive a
MATERIALITY threshold from the measured BASE seed-regret SPREAD (e.g. BASE's own across-seed regret
distribution — a candidate is unstable-beyond-incumbent only if its regret exceeds BASE's by more
than BASE's own measured spread / a bootstrap CI on the regret gap), NOT a raw max. (3) Re-adjudicate
TRI-PS+FREQ-PRIOR: is its instability MATERIALLY worse than BASE's under the equal-repeats bar?
DECISION RULE (registered): TRI-PS+FREQ-PRIOR becomes an ADOPT-CANDIDATE iff (a) its transfer win
holds under the corrected bootstrap (already shown), AND (b) under equal-repeats its seed-instability
does NOT exceed BASE's own by more than the measured materiality threshold, AND (c) it still passes
the rare-ngram guard. If instability IS material even with equal repeats -> stays rejected (real
disqualification, honestly). Either way report the equal-repeats regret distributions for BASE vs
combo + the derived threshold. Read-only repo + own scratch; commit nothing; ADOPTION into the
production model remains USER-GATED (this audit only determines candidacy).

### STAB-AUDIT-1 OUTCOME — TIE under equal repeats; TRI-PS+FREQ-PRIOR advances to Phase D (2026-07-22; after running)
Verdict: TIE. The old optimizer-stability rejection was an ARTIFACT of unequal evidence and is removed.
METHOD executed exactly as registered — symmetric, equal-effort, preregistered before run: arms {BASE,
TRI-PS+FREQ-PRIOR}; model seeds 0-19; search seeds 888301-888320; 20 searches on every seed-specific
model surface AND 20 on every leave-one-model-seed-out consensus surface; 40 models, 80 surfaces, 1,600
searches, 10,000 paired model-seed bootstrap draws. For each model seed the selected layout was scored on
the mean of the other 19 surfaces; candidate-minus-BASE consensus regret paired by seed.
RESULT: candidate mean consensus regret 0.124993% vs BASE 0.108749%; paired delta +0.016243pp, 95% CI
[-0.035908, +0.066360] — CONTAINS ZERO -> TIE by the registered rule (CLEAR = upper<0, TIE = CI straddles
0, FAIL = lower>0). Honest read of the tails: candidate's MEAN and MEDIAN regret were slightly HIGHER
(median 0.1192% vs 0.0600%; higher on 12/20 seeds) but its UPPER TAIL was LOWER (P95 0.2337% vs 0.2779%,
max 0.2479% vs 0.2817%); search-level regret nearly identical (means 0.0427% vs 0.0415%). Both arms
produced 20 DISTINCT seed-selected layouts -> exact optimizer positions remain underidentified even at
small objective regret (why exact-agreement was only diagnostic, never a gate). The old "0.14850% > 0.07737%
tol" disqualification is retired: it mixed model+residual search variation, used unequal evidence (max-of-3
vs 1), and exempted BASE — under equal effort BASE itself reaches 0.28169% seed-consensus regret vs the
candidate's 0.24794%; neither is credibly less stable.
DECISION RULE evaluated — all three registered conditions MET: (a) transfer win holds under corrected
bootstrap (rho/ceiling +0.02845, CI [+0.02126,+0.03244]) ✓; (b) equal-repeats seed-instability does NOT
exceed BASE's beyond the materiality bar (delta CI contains 0) ✓; (c) rare-ngram guard passes (rare-three-
decile MAE -4.12%) ✓. => TRI-PS+FREQ-PRIOR status changes optimizer-rejected -> ADVANCE_PHASE_D (an
ADOPT-CANDIDATE). This does NOT select a new production layout and does NOT make the koyu plateau sample a
winner. PRODUCTION ADOPTION REMAINS USER-GATED; only a preregistered Phase-D participant comparison of
BASE-selected vs candidate-selected layouts (blinded controls) can resolve whether the small objective
differences are material and which model yields the better novel layout.
ARTIFACTS (verified, SHAs match callback): result keybo-e2e/runs/tri-frequency-stability.json
sha256 5a88914cabc21011a9d52ad9212c316958844e6fdb7e3036b5e0104326db7669; report ...-report.md sha256
6438e1178aa85d3a1c20bce8dfa28ce7b59ca2f09d40fc66f84db5e30d066086; log ...-stability.log sha256
751a53ba10dbae9895e92a8feedee5deab0bb7b6de20be9883069424a6985ef6; frozen input SHA c7eead886301ecb47...
(matches the recorded TRI-PS+FREQ-PRIOR input). Durable copy in
state/keybo-optimization/artifacts/stab-audit-1/. Clean source 5eff01b; 18 expected-green tests pass, known
shipped-bootstrap RED reproduces [0,0]; py_compile + Ruff pass; BASE tensor parity 0.0ms for seeds 0/1/2;
no commit or production change made by the audit.

### PHASE-D PRE-DATA PACKAGE — banked design (2026-07-22; NOT run, NO human data, NO production change)
Status: the pre-data (design) HALF of Phase-D is implemented + verified and BANKED. Phase-D itself (the human
participant study) is NOT run and is DEPRIORITIZED by the user ("won't do Phase-D for a long, long time —
months or never"); adoption of TRI-PS+FREQ-PRIOR stays gated on it and is therefore INDEFINITELY DEFERRED.
This entry records the reusable design so it survives whenever/if-ever the study runs. All work is LOCAL-ONLY
in a child worktree (commit a28b2a7, NOT pushed; child origin/main still 5eff01b); durable copy harvested to
state/keybo-optimization/artifacts/phase-d-predata/.
PROTOCOL (protocol.md sha cbb64157...): 8-session blinded crossover + a controlled 19-layout ranking panel.
Tests TWO claims: (1) does the candidate-selected layout reduce OBSERVED sustained typing time vs the
BASE-selected layout without raising errors; (2) does the frozen candidate SERVED model rank NOVEL layouts
better than BASE (the thing held-layout tau cannot test — it saturates). Target N=80 completed (cap 100 for
20% attrition), 44 laptop / 36 external, blocked across 3 WPM bands. Track S scores sessions 3-8 (drops first
2 as warmup); Track R assigns qwerty + both primaries to all 80 + 5 balanced of the other 16 (=25 each), fixed
fingering. POWER (paired log-time, alpha 0.05, 90%): served UMAE gap 5.42% needs 34; WMAE 8.77% needs 13;
served RARE gap 1.88% needs 73; the frozen 0.0608% selected-layout model gap needs ~283,790 (intentionally
infeasible — prevents reading model-error gains as speed gains). CORRECTION to my earlier note: the
"9->90% / 19->95%" figures are no-failure conformal n/(n+1) bounds needing exchangeable LAYOUT units, NOT
participant counts — do not cite them as such. With 19 layouts one discordant pair moves tau by 0.0117 (vs
0.333 at 4 layouts) — that is how the panel de-saturates ranking.
DECISION RULE (fail-closed, preregistered): ADOPT_CANDIDATE requires ALL of — >=1% observed sustained-time
improvement with paired CI excluding 0; error-rate noninferiority; >=80% bilateral plateauing; all 19 layouts
evaluable; positive nested-bootstrap tau gain; candidate rho/own-ceiling >=0.8 on every layout; positive
paired overall rho/ceiling gain; served-rare and matched-geometry delta-MAE ratios with upper CI < 1; no
device/WPM-stratum or fingering-control failure. Clear directional harm -> KEEP_BASE; any inside-CI /
underpowered / saturated / failed-quality -> TIE. Even a pass only advances to a SEPARATE user-gated
production review (adoption never automatic).
STIMULI (stimuli.json sha c4916988...): qwerty + dvorak + 3 top-tier anchors + 7 generated per arm = 19
distinct valid layouts; campaign config (C30M pinned space, 90 WPM, seeds 889101-889112, SA 12x16k + 2-opt),
zero direct-rescore error. Primaries: BASE flmpg'.oyksntdcireahxzbwvu,-qj ; CANDIDATE
bgdlm-.yo'csthrpnieafxvwku,jqz (Hamming 29-30 from archive-1843/1846/keybo-lsb+lm => genuinely novel, not
perturbations). Each arm prefers its own primary by ~0.16% objective fit (NOT a predicted human-speed effect).
HARNESS (phase_d_analysis.py sha 9c00b1da...): validates event/prediction/blinded-manifest contracts, excludes
Track-S warmup, derives final-transition K31 timings, participant-split ceilings, and evaluates all guards;
participant bootstrap preserves replacement multiplicity; ranking nests layout resampling (avoids 4-layout tau
saturation); refuses to emit an authoritative decision on synthetic data (requires --allow-synthetic, marks
synthetic=true/authoritative=false). QUALITY NOTE: the child ran its OWN adversarial cold-reviewer, which found
3 real harness bugs (trial-grouping ignoring block/period; manifest fail-open on identical primaries; fingering
pass computed over non-confirmatory events); I independently VERIFIED all 3 are FIXED in the committed artifact
(_trial_key is the full 7-tuple incl period+block_id L363-369; manifest requires distinct 19 ids/strings +
primary set == expected models L260-265; fingering pass scoped to confirmatory samples) — find->fix->verify
loop closed. Verification: 115 tests/analysis pass, 31 focused Phase-D tests, Ruff+py_compile, synthetic refusal
+ byte-identical rerun, 43 direct rescores at 0.0 error.
BOUNDARY: draft for a study owner to register + run. Human recruitment, consent/ethics, collection, ledger
registration, production adoption, and any model/schema change all remain OUT of scope and un-done.

### COMM+POOL-INVEST-1 OUTCOME — TRI-PS+FREQ-PRIOR replicates on community + pool; new-layout board rebuilt on full objective (2026-07-23; after running)
Question (user-directed): the TRI-PS+FREQ-PRIOR decomposition (target = g(geometry,WPM) + b(ngram), b shrunk toward a
smooth frequency-curve prior h(log corpus freq); only g served, frequency NEVER a served input) was established for the
AALTO model only. Does it also help the COMMUNITY and POOL models? And generate new layouts from every winning
geometry-only serving surface. NOTE: adoption stays USER-GATED + Phase-D-deferred (months-or-never); this is research on
the frozen models, not a production change. All numbers VERIFIED against the frozen artifacts (SHAs below).
METHOD: same as the Aalto tri-frequency run — LOLO held-layout validation, the CORRECTED paired model-seed bootstrap CI
(not the degenerate shipped set-collapse bootstrap), rare-ngram guard, served-frequency-invariance check; general-English
corpus weighting (data/corpus/, NOT Aalto-training frequencies — honors the provenance constraint). Decision rule
preregistered in-artifact before each run: ADOPT_ELIGIBLE = rho/ceiling paired CI lower endpoint > 0 AND no rare/tau/
invariance guard fails; TIE = rho CI contains 0 + guards pass; REJECT = credible rho harm or a guard fails.
COMMUNITY (tri-frequency-comm.json): FREQ_PRIOR ADOPT_ELIGIBLE rho +0.01107 CI [+0.00628,+0.01608]; TRI_PS_FREQ_PRIOR
ADOPT_ELIGIBLE rho +0.01052 CI [+0.00431,+0.01670]; TRI_PS-alone TIE (rho +0.0002 CI straddles 0). STRUCTURAL CAVEAT
(artifact reliability_normalization.reason): each community layout has exactly ONE typist -> no split-half ceiling
exists, so comm rho is RAW and conditional on the 4 observed typists (selection-inseparable, same limit as the RCA
finding); "eligible" here = conditional on those 4, NOT a population claim. Served gains marginal (UMAE -0.49%/-0.60%,
WMAE slightly WORSE +0.14%/+0.22%); the strong rare-ngram CIs are FULL-model.
POOL (tri-frequency-pool.json): FREQ_PRIOR ADOPT_ELIGIBLE rho +0.00866 CI [+0.00702,+0.01033]; TRI_PS_FREQ_PRIOR
ADOPT_ELIGIBLE rho +0.01089 CI [+0.00914,+0.01268], rare -2.60% CI [-2.99%,-2.22%]; TRI_PS-alone REJECT (rho CI clears 0
but rare barely improves CI [-0.0094,-0.0012] and it fails the guard set — the same non-saturated-pool anti-signal seen in
FREQ-FEAT-1). Both freq-prior arms hold across all three models; TRI_PS-alone does NOT (TIE on comm, REJECT on pool).
NEW-LAYOUT BOARD (tri-frequency-layouts.json / -report.md): 60 frozen searches (12 seeds x 5 eligible candidate surfaces),
each 12x16k SA + exhaustive 2-opt, on the FULL production objective T3 = seed-mean bigram T2 + candidate seed-mean
geometry-only trigram Tcond; 46 unique layouts, all direct-rescored at max abs error 0.0, WPM-90 rebuild 0 ms.
⚠ SUPERSEDES an earlier buggy board: the banked Phase-D stimulus generator had loaded Tcond-only .seedmean.npy and OMITTED
T2 -> its layout strings (incl. the earlier-quoted bgdlm-.yo'csthrpnieafxvwku,jqz) are RETRACTED. This board uses the
complete objective. Best per surface: AALTO_TRI_PS_FREQ_PRIOR woyu,kdlncgeiapmhtrsqj-.'fbvxz (3.579% own vs qwerty);
COMMUNITY_(TRI_PS_)FREQ_PRIOR cstrv,deaigylnmkpo.uwqjfxbhz'- (5.98-6.08%); POOL_FREQ_PRIOR cnthd.aeskpblrmuioygwzvfx,'-qj;
POOL_TRI_PS_FREQ_PRIOR iaed,vhtscyuop.mrlfg-q'kzbnxjw.
FINDING: Aalto yields a better PURE-SPEED candidate, but NO all-around flagship emerges — cross-surface robustness +
comfort keep lsb-sib / keybo-lsb among the strongest anchors (consistent with the RANK-1 4-way-tie conclusion). Provenance
re-confirmed: keybo-lsb = P17 POL-CHEB-r888514 (Aalto BASE + ergo gauges); lsb-sib = COMM-OPT-1 BOTH-SEED-keybo-lsb (Aalto
BASE + older comm surface + gauges). Corrected raw support maps space to the true K31 index 31 (the historical SELECT-1
index-30 alias is excluded — the bug fixed at select.py earlier this campaign).
DISPOSITION: the decomposition is a robust research improvement across all three models (freq-prior arms), NOT a promotion.
Adoption of any candidate MODEL or LAYOUT remains USER-GATED and gated on Phase-D human data (deferred). No production/
schema/layout/model change or commit to the production tree was made.
ARTIFACTS (verified, SHAs match callback; durable copies in state/keybo-optimization/artifacts/{comm-invest,comm-pool-board}/):
comm tri-frequency-comm.json sha 3e7acd89..., report aacc261e...; pool tri-frequency-pool.json (status lolo_complete);
layout board tri-frequency-layouts.json sha 427c3eabf0f7d9aac926568f6f620434bc95c378b778bd0532b7cdd38641be95, report
0e019c27dd13228b1be31ac0a0d50c3aa4ae6fda8834bd231ab9a25a09ad86b1; SELECT-METHOD-1 driver JSON sha 22cafed5...; clean
source 5eff01b. A separate child (stimgen-fix) is adding a regression test + root-cause fix for the T2-omission generator
bug (local, uncommitted to production).

### COMM+POOL-INVEST-1 CORRECTION ADDENDUM — stranger-read self-audit (2026-07-23; 3 gaps, none change eligibility)
A post-hoc stranger-read audit of the above by keybo-selmethod (I code-verified all 3 against the harvested drivers) found
three wording/scope gaps. None change any arm's eligibility verdict; they correct HOW the result is stated.
(1) POOL is NOT independent of Aalto+community. run_tri_frequency_pool.py:55-56 loads the pooled dataset from the aalto
source (tristrokes_cond_v3.tsv) AND the community source — pool is a SUPERSET that CONTAINS both. So any framing of "POOL
independently confirms / independently eligible" (which I used verbally) is OVERSTATED: pool overlaps the other two by
construction, it is a pooled re-analysis, not an independent third sample. The ledger body above ("hold across all three
models") is fine; the "independent" gloss is retracted.
(2) COMMUNITY's one-typist-per-layout / model-seed-only-CI caveat, present in the ledger body above, was ABSENT from the
frozen board JSON/report — now made explicit in state. Carry it wherever comm rho is cited.
(3) The comm-TRI_PS=TIE vs pool-TRI_PS=REJECT difference is a real DECISION-RULE ASYMMETRY, not an inconsistency, and my
earlier "fails the guard set" was vague on which guard. Mechanism (run_tri_frequency_pool.py:669-699): the matched-geometry-
delta failure is WAIVED only when rho_tie is also present. Comm TRI_PS (rho tie + matched-delta fail) -> waived -> TIE; pool
TRI_PS (credible rho GAIN, so no rho_tie, + same matched-delta fail) -> not waived -> REJECT. Same guard failure, opposite
verdict, because the waiver is conditional on the rho tie.
AUDIT POSITIVES: independent reconstruction reproduced all 276 mean scores, 828 seed scores, 322 comfort-axis values at 0
error; six parent-state comm/pool/board copies rehash to 3e7acd89/aacc261e, d994cd78/801ee28a, 427c3eab/0e019c27 (pins the
pool JSON/report SHAs the body left as "status lolo_complete"). REPRODUCIBILITY: keybo-e2e/ is pure scratch (dies with the
workspace); I harvested all drivers + focused tests to state/keybo-optimization/artifacts/drivers/ (run_tri_frequency{,_comm,
_pool,_layouts,_optimizer}.py, selmethod.py, tri_frequency.py, stability_reexam.py, run_tri_frequency_stability.py, +
test_*.py) so these outcomes remain reproducible after the child is reaped.

### SCOPE DIRECTIVE — Phase-D is CANCELLED; current data is FINAL (2026-07-23, user)
User directive (verbatim intent): "We are not getting Phase-D data. Do not block anything on it. Assume the data we have is final."
CONSEQUENCE — this reinterprets every "gated on Phase-D" / "deferred pending Phase-D" clause in THIS ledger. Phase-D was
the registered path to (a) de-saturate held-layout tau (currently 1.0 for ALL arms incl BASE, on 4-5 held layouts -> the
ranking test cannot discriminate) and (b) get observed sustained-typing evidence to adjudicate model/layout adoption. That
path is now CLOSED, not merely deferred. So decisions previously parked "pending Phase-D" must be made ON THE EXISTING
EVIDENCE or explicitly abandoned — they can no longer wait.
WHAT THIS DOES AND DOES NOT CHANGE:
- Does NOT retroactively change any measured result, verdict, CI, or SHA above. The evidence base is frozen as-is.
- Does NOT lower the evidentiary bar or manufacture significance. "We can't get more data" != "the data we have now proves
  what the missing data would have." Claims that genuinely REQUIRED sustained-typing evidence (e.g. "the candidate model
  generates a BETTER novel layout") remain UNPROVEN and must be stated as such — now permanently, not "pending."
- DOES remove "wait for Phase-D" as a valid reason to defer a decision. Adoption/promotion calls are now pure USER judgment
  on the frozen evidence (still user-gated as externally-consequential), with no future data arriving to change them.
STANDING RESULT under this directive (evidence recap, unchanged): held-layout tau is SATURATED at 1.0 for BASE and every
candidate arm -> there is NO measurement, and now never will be, that shows the freq-prior candidate RANKS novel layouts
better (or worse) than the keybo-lsb-era BASE. Served-frame accuracy IS better for the candidate (Aalto served UMAE -5.42%,
WMAE -8.77%; served rho/ceiling 0.852->0.864, no CI); freq-prior replicates on comm + the pooled set; TRI_PS-alone does not.
RANK-1 remains a preference-dependent 4-way tie with no all-around flagship. NONE of that is Phase-D-contingent; it stands as
the final read. The remaining choices (flagship layout; adopt the freq-prior model as the generator; charset) are now
decidable-today USER calls on this frozen evidence — no longer blocked, no longer waiting.

### REPLICATE-GEN-1 OUTCOME — flagship generator re-run on new model surfaces; CLEAR-WINNER = HONEST NULL (2026-07-23)
Question (user-directed): replicate the ACTUAL flagship-generation pipeline that produced keybo-lsb/lsb-sib — NOT the pure-speed
46-layout board — but with the new eligible candidate model surfaces swapped onto the SPEED axis, then audit whether any single
output is a "clear winner." Local research on frozen models; no adoption (user-gated). All claims MODELED/gauge-based (tau
saturated, Phase-D cancelled — no realized/observed speed or ranking claim is possible).
PIPELINE (replicate-gen, commit 9345f4a in its own clone off a28b2a7): p17_coopt.py = the P17/COMM-OPT-1 recipe — 4-axis
augmented-Chebyshev over speed+genkey+oxey1+oxey2, weight family 44 Dirichlet CHEB + 6 MMX + 2 SPD + 4 SEEDED, SA 12x16k +
exhaustive 2-opt + top-10 polish, speed-capped min-max pick, gauge normalization held stationary (p16 board mins / qwerty,
exactly as comm_opt1.py built lsb-sib). Speed axis driven by each of the 5 eligible surfaces (AALTO/COMMUNITY/POOL x freq-prior
arms) in turn.
POSITIVE CONTROL (the load-bearing validation, I VERIFIED it): with AALTO_BASE on the speed axis the pipeline reproduces P17
BIT-FOR-BIT — 56/56 search layouts identical, PICK POL-MMX-r888404, RUNNER-UP POL-CHEB-r888514 = keybo-lsb (confirmed:
replicate-polcheb.json .surfaces.AALTO_BASE.runner_up.layout == pyuo,vgdnlhiea.cstrmkj-z'fwbxq), max_direct_rescore_error 0.0 on
all 6 surfaces. So the replication IS the flagship generator and the candidate arms differ ONLY in the model — a fair test.
PER-SURFACE NEW-MODEL PICKS (rescore err 0): AALTO_TRI_PS_FREQ_PRIOR pyu,.vgdnlhieaocstrmkj'z-fwbxq; COMMUNITY_FREQ_PRIOR
prtsgx,aeycldmv.nouiwqkfzbh'-j; COMMUNITY_TRI_PS_FREQ_PRIOR crtsmzpeaigldyf.nou,wqkvxbh'-j; POOL_FREQ_PRIOR
crtskx,aeypldmf.nouiwzqgvbh'-j; POOL_TRI_PS_FREQ_PRIOR crtsgx,aeymldpv.nouiwqkfzbh-'j.
VERDICT = HONEST NULL, NO CLEAR WINNER (I verified the floor ordering in gauge-board.json): (a) NO dominance — no new pick is
>= keybo-lsb/lsb-sib on all 10 gauge axes (best beats lsb-sib 7/10 but loses 3; loses to keybo-lsb on >=5/10). (b) NO floor win
— the 6-surface robustness floor is held by INCUMBENTS: archive-1846 3.160, lsb-sib 3.156, keybo-lsb 3.137, archive-1843 3.081,
keybo-lsb+lm 3.058; the BEST new floor is 2.968 (AALTO_TRI), BELOW every incumbent. (c) COMMUNITY/POOL picks OVERFIT their own
surface (5.57-5.63% saved on COMMUNITY_*) but crater on Aalto (floor 1.74-2.32) and concede scissors (0.49-0.87 vs incumbents
0.14-0.22) while winning LSB/SFB — a TRADE, not a win. Axis wins split across 9 layouts = Pareto tie. Consistent with RANK-1's
4-way preference-dependent tie: a better MODEL does not, under the flagship generator, yield a layout that dominates the incumbents.
PROVENANCE CORRECTION (both replicate-gen AND I independently reached this; supersedes my earlier speculation): clgmk.,ouysrthd
pnaeiqxwbvfz-'j — keybo-selmethod's 8-surface maximin #1 — is NOT a new-model generation. It is P10-w0.5, an OLD legacy co-opt
layout / warm-start seed (appears in p12_robust, p14/p15_coopt, comm_opt1 WARM). On the canonical 6-surface floor it is 2.972,
BELOW all 5 incumbents; its 8-surface maximin lead came from including old surfaces it was historically tuned against.
ARTIFACTS (verified, SHAs match callback; harvested to state/keybo-optimization/artifacts/replicate-gen/): replicate-polcheb.json
sha 649cbdf1...; gauge-board.json sha 75f0f567...; clear-winner-audit.md sha 171a7273...; driver p17_coopt.py + commit patch.
DISPOSITION: the better model (TRI-PS+FREQ-PRIOR family) does NOT produce a clear-winner flagship layout under the actual
generator — incumbents (keybo-lsb / lsb-sib / archive-1846) retain the robustness floor. No adoption/promotion; no production change.
PRECISION ADDENDUM (replicate-gen stranger-read self-audit, I verified vs gauge-board.json; verdict UNCHANGED): two framing
refinements to the above. (1) The best case is 8/10 not "7/10" — COMMUNITY_FREQ_PRIOR:pick reaches >=8/10 axes vs archive-1846
(POOL_TRI:runner_up likewise >=8/10 vs lsb-sib); the null holds because the 2 lost axes are the decisive ones: six-surface
floor 2.176% vs 3.160% (a 0.98pp / 31%-relative collapse on AALTO_BASE) AND scissors 0.694 vs 0.181 (3.8x worse, material
comfort concession) = "no floor win AND concedes material comfort." (2) The new POOL/COMMUNITY picks DO win the six-surface MEAN
axis (verified: COMMUNITY_FREQ_PRIOR pick mean 3.785% vs incumbents <=3.660%; top new means 3.79-3.90%) plus LSB/SFB/SFS — a
coherent profile (higher average cross-surface speed + better rolling comfort) that loses specifically on floor + scissors +
oxey2. So the honest characterization is not "new picks are worse" but "new picks trade a higher MEAN for a lower FLOOR and worse
scissors" — a preference-dependent trade, still not a dominating clear winner. Positive control re-verified exact (56/56, runner-up
== keybo-lsb char-for-char, rescore 0.0).

### AUDIT-BEST-1 — adversarial audit: "best model + best layout" is OVERSTATED (2026-07-24; all 5 questions GAP)
A user-directed --codex adversarial audit (AUDIT.md harvested to state/keybo-optimization/artifacts/audit-best/) tried to BREAK
the "we reached the best model + best layout" conclusion. Verdict: OVERSTATED (audit confidence HIGH). All five questions returned
GAP. I INDEPENDENTLY VERIFIED the two load-bearing findings against the frozen artifacts (below); the audit is well-bounded — it
credits the REPLICATE-GEN positive control, affirms the honest null STANDS, and marks provenance gaps "unsupported" not "disproven".
WHAT THE CAMPAIGN ACTUALLY SUPPORTS (audit's bounded restatement): (1) several decomposition variants improve held-out FIT on the
frozen data; (2) re-running the exact P17 generator with 5 candidate surfaces produced no layout dominating the incumbents on the
frozen 10-axis board. It does NOT support "best possible model" or "best possible layout".
CORRECTION 5A (I VERIFIED on all 3 surfaces — this is the serious one): the CI'd headline gains I registered as a "REAL
generalization gain" are FULL-frame deltas, not the served (layout-picking) frame. Verified: AALTO published +0.028452 == full;
served only +0.011877. COMMUNITY published +0.010516 == full; served +0.006910. POOL published +0.010893 == full; served +0.002051.
The served deltas have NO bootstrap CI (the ledger scope directive already admitted served 0.852->0.864 had no CI). So the
STAB-AUDIT adoption condition (a) "transfer win holds under corrected bootstrap" was resting on the FULL-frame number while the
frame that actually scores layouts improves ~40-19% as much with no CI. The verdicts are not overturned (fit does improve on both
frames) but the "REAL generalization gain" framing on the served claim was overstated — corrected here.
FINDING 1 (verified): the peak-MODEL search already COMPLETED and BEATS TRI_PS_FREQ_PRIOR on served rho (AALTO +0.03029
[+0.02720,+0.03315], COMMUNITY +0.00634, POOL +0.01192; peak-model-search.json sha 540478a6, harvested to artifacts/peak-model/).
So "TRI_PS_FREQ_PRIOR is current-best" is STALE — it is at most the predecessor incumbent. (selmethod had not yet sent its peak
callback when the audit surfaced this; awaiting its formal report.) BUT the peak search is explicitly NON-GLOBAL: 18 of 2,916
same-family configs, only 3 of 85 compatible pairs, CAND4-XGBoost fixed (GAM/spline family never run — gaps-and-roadmap.md:89-100).
OTHER GAPS (audit, not all independently reverified — 🟡): (2) the paired-seed bootstrap is a training-randomness interval, not
participant/layout/corpus generalization; the min-only tau guard can admit systematic seed-level rank harm (a POOL paired-seed
reanalysis gives candidate-minus-BASE margin tau -0.0095 [-0.0143,-0.0048] = combined candidate ranks held layouts WORSE than BASE
on average — post-hoc/exploratory). (3) tau is NOT wholly unmeasurable: corrected SERVED tau is unsaturated (AALTO/COMM 0.667,
POOL 0.643) and cross-population divergence tau is weak (0.21/0.14/0.048) = an internal domain-robustness warning the 4-layout
ceiling stat hid. (4) the layout null is conditional on ONE generator (P17) whose decisive floor+scissors gauges were POST-HOC not
in-loop; a mixed-operator Pareto/NSGA-II search over the corrected floor/mean+mechanics was never run. (5) also: the COMMUNITY
TRI_PS=TIE waived a failed matched-delta guard not per the frozen "every guard passes" rule (already noted as asymmetry); RANK-1
named standing set {archive-1843/1846/keybo-lsb/lsb-sib} is STALE vs the final artifact's {archive-1843/1846/fresh2-024/025}.
TOP PHASE-D-FREE CLOSURES (audit): (1) nested selection inside outer participant/layout/source resamples + paired mean margin-tau
as a primary guard + source-blocked pairwise rank-margin over the 99-layout board with simultaneous inference; (2) GAM-vs-CAND4
under identical decomposition + all 82 missing compatible two-knob pairs; (3) mixed-operator Pareto generator optimizing the
corrected floor/mean+mechanics IN-LOOP. NONE need human data; passing them would justify stronger modeled-robustness claims.
NONE can establish realized human typing superiority — that stays unknowable (Phase-D cancelled).
DISPOSITION: no result RETRACTED (fit gains real, positive control holds, honest null stands within its generator); but the
"best model/layout" and "REAL served generalization" framings are corrected to their bounded form. These are research findings;
no production/adoption change. The 3 closures are candidate follow-ups for USER decision, not auto-launched.

### CLOSURE-1 OUTCOME — nested selection + rank stability: all 3 audit concerns CONFIRMED (2026-07-24)
Audit closure #1 (inference/selection rigor + rank stability) COMPLETE. I INDEPENDENTLY VERIFIED the load-bearing numbers +
positive controls against the frozen artifacts (harvested to state/keybo-optimization/artifacts/closure1-nested/; report sha
ab96197b, piece1 7ee4f6ce, piece2 c00cd4d9, piece3 7b43c108; child commit 9cfe130 local-not-pushed). Positive controls EXACT:
piece2a reproduces the audit's POOL margin-tau -0.009519210137443335 to full precision; piece1 reproduces the frozen selector at
max point err 0.0, 0 verdict mismatches. This tightens AUDIT-BEST-1 from "overstated" to quantified.
PIECE 1 — the peak gains do NOT survive generalization-aware inference. The nominal paired-seed CIs resample only the 20 MODEL
SEEDS (training randomness); re-running the whole selector inside an OUTER held-layout resample x inner seed resample widens them
~10-25x and they cross/graze zero (verified in piece1_nested.json): AALTO peak nominal +0.0303 [+0.0272,+0.0332] -> nested
[-0.0463,+0.1027] CROSSES 0 (positive in only 84.6% of layout draws; dvorak fold -0.076; the peak is argmax-selected in only 64.5%
of draws = NOT identifiable). COMMUNITY nested [-0.0007,+0.0142] grazes 0, peak argmax-selected only ~21% (ties EFFECT_K_300 +
incumbent). POOL nested bootstrap keeps lower +0.0003 but the more-conservative across-fold t-interval [-0.0020,+0.0258] crosses 0.
=> NO surface's peak is clearly clear of zero under layout-generalization inference; POOL is merely closest, method-dependently.
The "peak beats incumbent" claim is a within-training-randomness statement, not a generalization one.
PIECE 2 — the min-only tau guard hid a real regression; ONE decision-relevant flip. Root cause verified at source
(run_tri_frequency_pool.py:634-657): the guard adjudicates MIN-over-seeds margin tau, and all 4 POOL board arms share the same min
0.904513, so min-only passes everyone while the seed MEANS diverge (BASE 0.9197 vs TRI_PS_FREQ_PRIOR 0.9102). Under a paired
MEAN-margin-tau guard, TRI_PS_FREQ_PRIOR FLIPS min-pass -> mean-FAIL on the LAYOUT BOARD (mean_tau_guard_pass=False,
verdict_flips=True) => it LOSES its POOL ADOPT_ELIGIBLE. The 3 selected peak-search configs do NOT flip (AALTO/POOL tau constant
across seeds at 4 layouts; COMMUNITY genuinely passes). POST-HOC/exploratory, but it means the earlier POOL "ADOPT_ELIGIBLE" for
TRI_PS_FREQ_PRIOR was an artifact of a min-only guard that can't see systematic seed-level rank harm.
PIECE 3 — incumbent top-tier is FRAGILE / preference-dependent, and the pivot is SOURCE not model. Among the 4 on-board anchors,
mean Kendall tau-b WITHIN a source family = +0.80 (model barely matters) but ACROSS source families = -0.13 (often anti-
correlated). 4 distinct within-tier winners across 8 judges: keybo-lsb #1 on AALTO_BASE but LAST on all 3 COMMUNITY judges;
archive-1843 wins all COMMUNITY; lsb-sib wins AALTO_TKG+POOL. lsb-sib is a WEAK Condorcet winner (beats each other anchor on a
majority of judges) = most-defensible-if-forced. This INDEPENDENTLY REPRODUCES the campaign's standing "community-trust is the
pivot" / preference-dependent-tie conclusion — from a different method. (Home-field confound handled: surface-native REPLICATE-GEN
outputs trivially win their own judge, excluded.) LIMITATION: fresh2-024/025 not on the frozen 99/46 boards, not scored (would
need refit, out of scope).
NET: closure-1 CONFIRMS the audit on all 3 axes. The model-improvement claims are weaker than registered (peak gains are training-
randomness, not generalization; TRI_PS_FREQ_PRIOR's POOL eligibility flips under a sounder guard), and the layout top-tier is
explicitly preference-dependent on corpus source. NONE of this needed human data. NO result forces a production change (nothing was
adopted); it recalibrates confidence DOWNWARD on "we found a better model" and REAFFIRMS "no source-independent best layout." The
served-fit improvements themselves are not disproven — only their generalization CIs are shown to be too narrow.
CLOSURE-1 REFLECTION SHARPENING (self-audit, I verified vs piece1_nested.json; report rev2 sha ff5270b4; NO verdict overturned,
two headlines corrected toward honesty — both partly SOFTEN the entry above, so recorded to avoid overstating the damage):
(1) PIECE-1, separate real fragility from t-crit width: model-free (no t-crit/bootstrap) per-fold gains show AALTO is GENUINELY
sign-fragile — dropping qwertz collapses its LOO gain to -0.00206 and the dvorak fold is -0.076 (1/4 folds negative). POOL is
NOT fragile the same way: its per-fold gains are all positive and its leave-one-layout-out mean stays [+0.0086,+0.0163], so my
"POOL across-fold t-CI crosses 0" leaned on the small-df t-crit(3df)=3.18 WIDTH inflation, not real sign-fragility. Corrected read:
POOL SURVIVES on the calibrated bootstrap; AALTO FAILS model-free; COMMUNITY winner unidentifiable (~21% argmax). Audit point (a)
— nominal CIs too narrow / conditional on training randomness — still holds for all three; but POOL is the one that does generalize,
not merely "closest."
(2) PIECE-3, the SOURCE-pivot magnitude is COMMUNITY-confound-amplified: the strong -0.13 across-source anti-correlation is driven
by the one-typist COMMUNITY surfaces (across-source pairs involving community avg -0.20; AALTO<->POOL alone avg +0.03 = near-zero,
NOT anti-correlated). So the "sources actively disagree" strength is a community artifact. HOWEVER the top-tier-FRAGILE verdict
HOLDS without community: the winner still flips among AALTO+POOL judges alone ({keybo-lsb, lsb-sib, archive-1846}), and within-
source model tau +0.80 is rock-solid. Corrected read: "no single source-robust winner" stands even excluding community; only the
anti-correlation magnitude was community-amplified.
