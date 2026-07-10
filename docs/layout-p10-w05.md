# P10-w0.5 — the balanced deliverable layout (2026-07-10)

```
c l g m k   . , o u y
s r t h d   p n a e i
q x w b v   f z / ; j
```

String form: `clgmk.,ouysrthdpnaeiqxwbvfz/;j` (qwerty slot order, space unchanged).

## What it is

The oxey-weight-0.5 member of the P10 family: the layout the measured speed model
picks when community comfort heuristics get half a vote. On this build it was the
best of both worlds — statistically tied with the pure-speed champion on predicted
speed (the family's speed plateau is ~0.2% wide) while carrying nearly half its
same-finger-bigram load.

| metric | value | context |
|---|---|---|
| predicted speed vs qwerty | **+3.91%** | LOGRAT corrected-trigram objective, wpm 90 |
| speed vs pure-speed champ (w=0) | −0.04% (tie) | inside search noise |
| sfb | **0.74%** | qwerty ~6%, colemak ~1.4% |
| dsfb (skipgram sfb) | 4.00% | |
| lsb | 0.73% | |
| scissors | 0.10% | |
| inroll / outroll | 6.43% / 2.09% | |
| onehand | 0.91% | |
| redirect / bad redirect | 3.01% / 0.62% | |
| alternation | 75.97% | the model's signature preference (measured: alternation gap +32ms, skill-invariant) |
| home row usage | 55.0% | top 22.5%, bottom 5.8% |
| max finger load | R-middle 13.2% | pinkies 7.6–7.9% |

## Provenance (everything traceable)

- **Model stack:** bigram `bigram_logratv5_seed{0,1,2}` + conditioned trigram
  `trigram_cond_lograt_join_seed{0,1,2}` (LOGRAT target space, T-REL adopted
  2026-07-10: cross-layout wmae −37.4% bigram / −23.5% join-trigram, rare-ngram
  guards held). Objective T3c = T2 + Tcond at wpm 90; oxey term weight 0.5
  (documented preferences, `keybo.scoring.oxey`).
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

## Honest caveats

- Speed numbers are model predictions validated cross-layout on 4 layouts (the 136M
  keystroke aalto dataset's layout diversity ceiling) — not a human trial of THIS
  layout. The +3.91% is a within-model statement.
- The oxey term re-introduces community heuristics the timing data measured as
  neutral (redirects, dsfb) — deliberately, as documented preferences, weight 0.5.
- The alternation-heavy structure is what the bigram+trigram physics rewards
  (alternation gap +32ms, rolls sub-additive but alternation still wins on this
  data); typists who subjectively prefer rolls may prefer the w=2 sibling
  (`hrfkv.y,oulnstdgciaezxbmqwpj;/`, sfb 0.59%, inroll 8.5%, −0.2% speed).
