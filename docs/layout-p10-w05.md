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

## Community-data validation (COMM-D, 2026-07-12; CORRECTED re-runs same day)

7 community typists on 8 modern layouts (colemak-dh, recurva, mtgap-variant, colemak,
and 4 custom), collected via the Kiakl monkeytype form (573k bigram samples,
`data/community/`). **Ingest correction (KIAKL-INGEST Amendment 3):** the capture's
`key` field is the QWERTY LABEL of the physical key, not the produced character
(monkeytype layout emulation) — the first ingest permuted every non-qwerty position.
All findings below are from the preregistered RE-RUNS on corrected data; the earlier
scrambled-era verdicts are void and superseded (PREREGISTRATIONS.md carries both).

- **External validity (R-D1): cell-structure transfer is REAL.** Zero-shot, the
  aalto-trained model orders each community typist's cells at ~0.5–0.6× that
  typist's own split-half noise ceiling (colemak-dh .58, castro .61, recurva .51,
  alite .54; thinner labels .23–.49), uniform across wpm bands — comparable to the
  in-family dvorak fold. Absolute levels shift (wmape ~0.27, slope 1.19): the model
  transfers structure, not per-person magnitudes. (The pre-correction "transfer
  fails" headline was the ingest bug.)
- **Training integration still REJECTED (R-D3), with a sharper mechanism:** merged
  training now IMPROVES community folds (mean −7.5% wmae) — cross-typist community
  structure is learnable — but still poisons aalto folds (+1 to +25%). The blocker
  is population price divergence + one-typist-per-layout labels, not transfer
  failure. The unlock for integration remains multiple typists per layout (Phase D).
- **Alternation preference: MIXED across community typists (R-D2b).** aalto qwerty
  measures alternation faster by +17.9ms [CI +13.5,+22.2]; on corrected community
  data the highest-volume typist agrees (+10.7ms, CI excl 0), four are ties, and
  two rowStagger enthusiasts measure ROLLS faster (−13.5/−14.4ms, CIs excl 0). The
  registered challenge rule (≥3 labels) does not fire, so the aalto-based structure
  stands — but the earlier "confirmed 7/7 population-general" claim was a
  scrambled-data artifact and is retracted. Alternation preference is
  population-relative: general population yes, roll-trained enthusiasts sometimes no.
- **Roll-direction asymmetry (R-RESID2/IV):** corrected residuals show the model
  prices in-rolls too slow / out-rolls too fast for several community typists
  (±5–27ms, survives practice matching). Injected as offsets it degrades aalto folds
  (+1.6–8.5%) AND moves the argmax by −0.018% (nothing) — population-divergent and
  argmax-irrelevant, doubly closed.
- **First-finger calibration transfer improved on corrected data (R-D5U2):** 3 of 4
  label-class estimates carry the aalto sign (colemak-dh pinky +47 vs aalto +43;
  recurva ring +69; colemak-dh ring −41 still inverts). Moot for the pipeline — the
  seam is removed (CAL-REMOVE) — but the physics measurement gains support.
- **Tail practice challenged (D2a, R-D5-CORR):** community typists on daily-driver
  layouts do not show the fast-tail signature (survives matched-n; position-agnostic,
  so unaffected by the correction).
- **Practice term (R-D5-U4):** the +pseudo natural experiment stays inverted
  (rank-corr −0.11, CI excl 0) — the practice term captures within-frame repetition,
  not transferable lifetime familiarity.
- **Cleanliness ruled out (DATA-CLEAN):** error-rate caps, post-error exclusion,
  trigram floors, wpm tightening — all immaterial. The one real data bug was the
  key-decode semantics (found by the data-quality audit); fixed, re-ingested,
  everything re-run.
- **Deliverable verdict: P10-w0.5 UNCHANGED through every corrected re-run.** The
  community data's corrected value: first out-of-family evidence that the model's
  cell structure transfers (~0.5–0.6× ceiling), honest boundaries (per-person
  magnitudes don't transfer; class prices are population-relative), and collection
  design for Phase D (record key AND code, release timestamps, ≥2 typists/layout,
  a qwerty control run per contributor).

## Honest caveats

- Speed numbers are model predictions validated cross-layout on 4 aalto layouts
  (qwerty/qwertz/azerty/dvorak). Corrected community data adds out-of-family
  support for the model's cell ORDERING (~0.5–0.6× each typist's noise ceiling,
  zero-shot) while showing absolute magnitudes are population-shifted — claims
  about any individual's exact speedup remain extrapolation.
- The oxey term re-introduces community heuristics the timing data measured as
  neutral (redirects, dsfb) — deliberately, as documented preferences, weight 0.5.
- The alternation-heavy structure is what the aalto bigram+trigram physics rewards
  (alternation +17.9ms on qwerty). Corrected community data shows this preference
  is population-relative: two roll-trained enthusiasts measure rolls FASTER on
  their layouts. Typists who prefer (or have trained) rolls may prefer the w=2
  sibling (`hrfkv.y,oulnstdgciaezxbmqwpj;/`, sfb 0.59%, inroll 8.5%, −0.2% speed).
- First-finger calibration was REMOVED from the pipeline (CAL-REMOVE) — the
  measured pinky effect stands as documentation (and gained community support on
  corrected data, 3/4 sign-consistent), but it is not installed in the served
  surface; the layout was found without it.
