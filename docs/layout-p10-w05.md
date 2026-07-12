# P10-w0.5 — THE PRIMARY DELIVERABLE (promoted 2026-07-11; created 2026-07-10)

```
c l g m k   . , o u y
s r t h d   p n a e i
q x w b v   f z / ; j
```

String form: `clgmk.,ouysrthdpnaeiqxwbvfz/;j` (qwerty slot order, space unchanged).

## What it is

**The recommended layout.** Promoted to primary deliverable (user-approved,
2026-07-11) after three independent multi-gauge searches converged on it:

- **P12** (dislocation-weighted family, the owner's travel×slowness balance
  heuristic in-loop): registered min-max-regret pick = **P10-w0.5**, max regret
  0.04% over {speed, oxey, exact-genkey} — it beat every purpose-built member.
- **P13** (exact genkey Score in-loop): registered pick (min genkey s.t. speed
  regret ≤0.5%) = **P10-w0.5** (genkey 33.68, speed regret 0.099%).
- **P13-combined** (genkey 0.5 + oxey 0.5 search from scratch): annealed to this
  exact layout up to a shuffle of the rare corner keys `j z ; /` (<0.4% corpus
  mass) — an independent reconvergence on the same structure.

It is statistically speed-TIED with the pure-speed champion P11-w0.5 (+3.95% vs
+4.00% vs qwerty on the final calibrated objective — inside the ~0.2% plateau)
while beating it on every preference gauge measured this campaign: genkey Score
33.7 vs 41.1 (near the community frontier of semimak 27.7 / graphite 29.5),
keymeow sfb 1.18% vs 1.70%, oxey (column-best), dislocation 374M vs 382M, and —
the deciding usability point — balanced finger loads (max finger 15.9%, graded
pinky<ring<middle≈index) where P11-w0.5 concentrates 20.0% on the right ring.

| metric | value | context |
|---|---|---|
| predicted speed vs qwerty | **+3.95%** | final calibrated objective, wpm 90 (+3.91% on the pre-calibration build it was annealed under) |
| speed vs P11-w0.5 (speed champ) | −0.04% (tie) | inside the ~0.2% plateau; regret 0.039–0.099% across final-objective evaluations |
| genkey Score (exact port, keybo corpus) | **33.7** | semimak 27.7, graphite 29.5, colemak 41.4, dvorak 48.2 |
| keymeow sfb / sfs-dist | **1.18% / 6.97** | exact tool, shai-iweb corpus; graphite 1.23/7.47 |
| max finger load | **15.9%** (R-middle) | P11-w0.5 puts 20.0% on R-ring |
| dislocation (owner's travel×slowness) | 374M | P11-w0.5 382M, colemak 324M, qwerty 706M |
| sfb | **0.74%** | qwerty ~6%, colemak ~1.4% |
| dsfb (skipgram sfb) | 4.00% | |
| lsb | 0.73% | |
| scissors | 0.10% | |
| inroll / outroll | 6.43% / 2.09% | |
| onehand | 0.91% | |
| redirect / bad redirect | 3.01% / 0.62% | |
| alternation | 75.97% | the model's signature preference (measured: alternation gap +32ms, skill-invariant) |
| home row usage | 55.0% | top 22.5%, bottom 5.8% |

## Provenance (everything traceable)

- **Model stack:** bigram `bigram_logratv5_seed{0,1,2}` + conditioned trigram
  `trigram_cond_lograt_join_seed{0,1,2}` (LOGRAT target space, T-REL adopted
  2026-07-10: cross-layout wmae −37.4% bigram / −23.5% join-trigram, rare-ngram
  guards held). Objective T3c = T2 + Tcond at wpm 90; oxey term weight 0.5
  (documented preferences, `keybo.scoring.oxey`).
- **First-finger calibration REMOVED (CAL-REMOVE, 2026-07-12):** the pinky/ring
  seam was retired after measuring that generation is calibration-invariant
  (re-search cross-regret +0.002%/+0.005% — orders of magnitude inside the
  plateau) and its community transfer is mixed (ring sign flips on the one
  testable label). The PINKY-GAP physics finding (+27ms outer-first on qwerty
  matched pairs) stands as a measurement; it is documented, not installed. Cost
  of removal is confined to the dvorak validation fold (+1.9% wmae); qwerty and
  azerty folds actually improve. This layout was found WITHOUT the calibration,
  so nothing about it changes.
- **Search:** SA + exhaustive 2-opt, 12 restarts × 12k iters, rng 880333
  (`keybo-e2e/p10_family.py` → `runs/p10_family.json`); 12 distinct near-optima
  within 0.5% — a wide plateau, so treat single-swap variants as equivalent.
- **Validation behind the models:** LOLO harness (leave-one-layout-out over
  qwerty/qwertz/azerty/dvorak), pooled held-out tau 1.0, rho at/above the split-half
  noise ceiling, corpus-weighted MAE with uniform + rare-decile guards
  (PREREGISTRATIONS.md, entries 046b92e → 470776d).
- **Certificate:** the bigram component of the pure-speed sibling sits within 3.35%
  of the provable QAP optimum (Gilmore–Lawler bound); no cubic-objective certificate
  exists (Q3AP has no GL analogue).
- **Cross-objective robustness:** the P10 family's champion loses only 0.62% when
  scored under the previous (ms-era) objective, while the previous champion loses
  1.18% under the current one — the family is the robust pick under objective
  uncertainty as well as the better-validated one.
- **Hardware:** cross-checked full-size vs laptop keyboards (kb_strat) — the layout
  ranking is hardware-invariant; one layout serves both.

## Community-data validation (COMM-D, 2026-07-12)

7 community typists on 8 modern layouts (colemak-dh, recurva, mtgap-variant, colemak,
and 4 custom), collected via the Kiakl monkeytype form (573k bigram samples,
`data/community/`). Preregistered campaign:

- **External validity (D1):** the aalto-trained model fails to rank a single
  community typist's cells zero-shot (best rho/ceiling 0.39). Mechanism: each
  label is one typist, so the model has never seen that person's idiosyncrasy.
  Magnitude errors are comparable to in-sample (~wmape parity); it is the
  within-typist ordering that fails, not the absolute predictions. Training
  integration (3 arms) was rejected — community typists teach idiosyncrasy, not
  physics. The unlock for integration is multiple typists per layout (Phase D).
- **Alternation preference CONFIRMED population-general (D2b):** all 7 community
  typists show alternation faster than rolls (+1 to +43ms), including those who
  chose and trained roll-optimized layouts. The assumption this deliverable's
  structure depends on most survived its hardest-ever test.
- **First-finger calibration transfer: MIXED (D5-U2):** one label replicates the
  pinky sign (+25ms), another inverts the ring sign (−31ms). The calibration
  retains its aalto evidence but transfer is not confirmed across populations.
- **Tail practice challenged (D2a, D5-CORR):** community typists on daily-driver
  layouts do not show the fast-tail signature (survives matched-n correction).
  Implication for quality-model cross-layout ranking: the confound is less severe
  at enthusiast-level practice than at aalto-qwerty decade-scale practice.
- **Practice term (D5-U4):** the +pseudo natural experiment (same typist/layout/
  board, ablated word familiarity) shows frequent bigrams do NOT benefit more from
  natural text — the practice term captures within-frame repetition, not
  transferable lifetime familiarity.
- **Cleanliness ruled out (DATA-CLEAN, 2026-07-12):** the integration failure is
  structural, not a data-quality artifact — error-rate session caps (10%/20%),
  post-error window exclusion, trigram cell floors, and wpm-band tightening all
  leave the verdicts unchanged. The ingest's typo handling was verified correct
  (a window's first event carries no interval, so error-adjacent contamination
  was already excluded by construction).
- **Class-price divergence measured and population-attributed (COMM-RESID/2/IV,
  2026-07-12):** community enthusiasts on self-chosen layouts pay ~30ms LESS for
  sfbs, ~22ms less for outrolls, and ~9ms MORE for alternation than the model
  prices — and this survives practice matching. But applying those prices to
  aalto predictions degrades every fold by +15–55%: the two populations' class
  prices are mutually exclusive. The deliverable optimizes for the
  general-population physics BY DESIGN; a hypothetical enthusiast-priced argmax
  differs by only +0.3% (18/30 shared keys). Self-selection and deliberate
  alternate-fingering of residual sfbs are the leading explanations and cannot
  be separated on this capture.
- **Deliverable verdict:** P10-w0.5 STANDS — no integration path qualified. The
  community data's value is as validation evidence (alternation preference
  confirmed) and honest boundary documentation (single-typist ordering fails;
  class prices are population-relative).

## Honest caveats

- Speed numbers are model predictions validated cross-layout on 4 aalto layouts
  (qwerty/qwertz/azerty/dvorak); zero-shot transfer to single community typists
  fails on cell ordering (D1), constraining external-validity claims to the
  population-level physics (alternation preference, pinky cost) rather than
  per-person prediction.
- The oxey term re-introduces community heuristics the timing data measured as
  neutral (redirects, dsfb) — deliberately, as documented preferences, weight 0.5.
- The alternation-heavy structure is what the bigram+trigram physics rewards
  (alternation gap +32ms, rolls sub-additive but alternation still wins on this
  data); typists who subjectively prefer rolls may prefer the w=2 sibling
  (`hrfkv.y,oulnstdgciaezxbmqwpj;/`, sfb 0.59%, inroll 8.5%, −0.2% speed).
- First-finger calibration transfer is mixed (D5-U2): the pinky cost is the most
  robust claim; the ring cost may be population-specific.
