# Ideas Ledger — exhaustive map of levers, status, and interactions

Purpose (user directive 2026-07-08): make the idea space EXPLICIT so "we simply haven't
thought of it" gaps become visible, and track which closed verdicts are conditional on
assumptions that later adoptions (esp. QIN) invalidate. One line per idea; status in
{ADOPTED, REJECTED, NEAR-MISS, CLOSED-UNCERTIFIABLE, OPEN, MOOT-PENDING}; the
"conditioned on" column is what makes re-tests systematic rather than vibes.

## A. Data / cleaning (what samples enter)
| idea | status | conditioned on | re-test trigger |
|---|---|---|---|
| hesitation cap (session-relative) | NEAR-MISS (CAP3/4 blocked only by censor guard letter) | mean-target training | QIN adoption (see F1) |
| post-error+control buffer BUF2-BOTH | ADOPTED (bigram) | mean-target | F2 |
| buffer after control keys alone | REJECTED (harmful — shift deletes good data) | — | stands |
| fast-interval floor | REJECTED-DEMOTED (rollover = legit; user) | — | stands |
| session warmup drop | REJECTED (no gain) | mean-target | weak; F2 covers |
| slow-TYPIST removal (wpm floors) | REJECTED (P2: tau collapse) | mean-target | stands (ranking, not target, killed it) |
| error-rate session drop | REJECTED (P3) | mean-target | weak |
| per-participant z-trim | NOT RUN (redundant w/ CAP) | — | only if F1 flips CAP in |
| dedup/near-dup sentence filtering | OPEN (never examined) | — | E1 |
| keyboard-hardware stratification (metadata has KEYBOARD_TYPE) | OPEN | — | E2 |

## B. Target / objective (what the model predicts)
| idea | status | conditioned on | re-test trigger |
|---|---|---|---|
| IQR-mean cell target | INCUMBENT | — | — |
| median cell target (T-MED) | REJECTED | — | stands |
| MAE training objective (T-MAE) | REJECTED (rare-decile guard) | — | stands |
| per-cell quantile targets (Q20/Q25) | REJECTED (Q-OBJ: -5pp own-ceiling structure) | dedicated single-q models | SUPERSEDED by QIN (shared strength fixes exactly this) |
| fastest-fifth mean (F5M) | ADOPT-CANDIDATE (razor-thin), D4 not-moot | dedicated model | QIN q=0.2 slice likely dominates it — F3 |
| quality-as-INPUT (QIN) | GATES PASS + dp-tau flag under diagnosis | — | in flight |
| overlap-conditioned target (mechanism marker) | CLOSED-UNCERTIFIABLE (8% marker floor) | this dataset's release-timing channel | Phase D data |
| hold-duration as auxiliary target/feature | OPEN (holds are in the data, unused beyond D1') | — | E3 |

## C. Features / model inputs
| idea | status | conditioned on | re-test trigger |
|---|---|---|---|
| pace label = session mean | INCUMBENT | mean-target | F4: labels may interact w/ q-conditioning |
| session-median / M5 blind-blend labels | REJECTED (rare-decile guard; wmae -7.4% documented near-miss) | mean-target, NO q input | F4 |
| EWMA local speed (4 variants) | REJECTED (5-deep null) | mean-target | weak — F4 covers the family |
| freq feature | REJECTED (tau collapse) | — | stands |
| skill-conditioned features (wpm interactions) | PARTIAL (wpm is an input; skill strata measured) | — | covered by QIN's q x wpm surface |
| finger-map correction (~8% alternate fingering) | OPEN (side finding; static map wrong 8%) | — | E4 |
| hand/finger asymmetry features (left-right speed diff) | OPEN | — | E5 |
| key-position priors (row/column effects beyond geometry dx/dy) | PARTIAL (row/finger features exist) | — | audit in E5 |

## D. Evaluation machinery
| idea | status | notes |
|---|---|---|
| LOLO + split-half ceilings + frac-of-own-ceiling | ADOPTED | |
| decisive-pair tau (bootstrap-certified pairs) | ADOPTED | mean-frame pairs; PER-FRAME re-certification now needed (QIN diag is the first instance) |
| corpus-weighted + uniform + rare-decile MAE guards | ADOPTED | guard tolerance 2% — near-misses MED/CAP3 documented |
| frozen-frame methodology for cleaning arms | ADOPTED (user's truncation catch) | |
| censor-ratio guard | FLAWED-AS-BUILT (can't separate frequency from geometry) | registered fix: frequency-controlled construction |
| E5 optimizer-side structural gate | ADOPTED | |
| affine recalibration control | ADOPTED (QSEL) | |

## E. OPEN ideas never yet run (the "haven't thought of it" inventory)
E1. Sentence/content dedup: memorable repeated sentences may train the practice term
    wrong; never audited. CHEAP audit: sentence frequency distribution; if top-1% >>
    rest, a dedup arm on the frozen frame.
E2. Hardware stratification: full vs laptop keyboards differ in key travel/rollover;
    METADATA HAS THE FIELD, never used. CHEAP: LOLO with hardware as an extra feature
    vs without; also a laptop-only robustness check of the final family.
E3. Hold-duration channel: press-release times exist for every key; beyond the D1'
    overlap probe they are unused. Ideas: hold as a feature (predicts force/precision
    style), hold-normalized durations, bimodality by hold. MEDIUM cost.
E4. Fingering-aware features: soften same_finger by the measured 8% deviation (e.g.
    a soft-SFB feature weighted by observed overlap rate per pair). MEDIUM; Phase-D
    certified data would do it properly.
E5. Asymmetry/biomech audit: per-hand, per-finger speed asymmetries as features; check
    the model's learned finger ordering against motor-control literature. CHEAP audit.
E6. Corpus sensitivity: the objective weights are one English corpus; never stress-
    tested the family against a second corpus (code, another register). CHEAP: rescore
    the final family under an alternate corpus; layouts that flip rank = fragile.
E7. Trigram QIN: quality conditioning was built for bigrams only; the conditioned
    trigram target could take q the same way. Deferred until bigram QIN verdict.
E8. Optimizer: QAP search at larger budgets / exact-solver for the bigram component
    (GL cert is 2-3%; a tighter bound or better incumbent may exist). MEDIUM.
E9. Ergonomics beyond speed: comfort/finger-load scorers exist but weights are
    user-preference-gated; no data-driven comfort signal attempted (hold variance as
    strain proxy?). OPEN, likely Phase D.
E10. Same-key repeat handling: "ee"-type bigrams have a distinct mechanism (double-tap
    vs re-press); never examined whether the model treats them sanely. CHEAP audit.

## F. Interaction re-tests triggered by QIN adoption (user's point: "changes we
##    previously rejected might be worth it now")
F1. CAP x QIN: the hesitation filter's rare-decile damage was measured against a MEAN
    target. Under q-conditioning, hesitation mass lands in the HIGH-q region by
    construction — the q=0.2 slice may be nearly filter-invariant (the filter's gain
    may be FREE now, or unnecessary). Arm: QIN on CAP3-filtered vs unfiltered data,
    judged at q in {0.2, 0.5} own-frames + guards.
F2. BUF2-BOTH x QIN: same logic, cheaper (buffer kept unless it hurts q-slices).
F3. F5M vs QIN-q0.2 head-to-head: if QIN survives its diagnostic, F5M is redundant —
    verify QIN-q0.2 >= F5M on the q=0.2 own-frame, then retire F5M.
F4. Pace labels x QIN: MED's wmae -7.4% was bought at rare-decile cost under the mean
    target; q-conditioning changes both sides of that trade. Arm: QIN with MED label
    vs session-mean label.
F5. Guard tolerances: the 2% rare-decile guard was calibrated to mean-target noise
    levels; q-frames have different noise. Re-derive the tolerance from the q-frame
    ceilings BEFORE running F1-F4 (else near-misses are uninterpretable) — this is a
    calibration step, not a rule change after results.
