# P13STAB-winner — the community-leaning plateau member (2026-07-13)

```
r c g k m   q . o u y
l s t h d   , n a e i
x w b f v   p j z ; /
```

String form: `rcgkmq.ouylsthd,naeixwbfvpjz;/` (qwerty slot order, space unchanged).

## What it is

**Not the primary deliverable** (that is [P10-w0.5](layout-p10-w05.md)) — this is the
best community-facing member of the speed plateau, surfaced by the P13-STAB
stability study (5 independent P13-recipe reruns on the regularized surface;
PREREGISTRATIONS.md 1314aa4 → 8e2d18b). The registered multi-gauge pick rule named
it the global winner across all 23 searched variants; the promotion clause did NOT
fire (it does not dominate P10-w0.5 — it concedes the speed axis), so it ships as a
documented sibling, the layout to reach for when community-metric standing matters
more than the last 0.1% of predicted speed.

## How it splits the board against P10-w0.5

Every axis derived from measured typing time favors P10-w0.5; every
community-doctrine axis favors this layout — measured on three independently
implemented exact community tools (genkey port, keymeow via kmrun, oxeylyzer-2),
which agree on the ordering.

| gauge | P10-w0.5 | P13STAB-winner | better |
|---|---|---|---|
| speed vs qwerty (regularized surface, wpm 90) | **+3.83%** | +3.71% | P10 |
| speed at wpm 70 / 110 | **+3.64% / +3.69%** | +3.46% / +3.60% | P10 |
| quality objective (F5M-LR) | **−2.63%** | −2.77% | P10 |
| dislocation (owner metric) | **3.74e8** | 4.04e8 | P10 |
| genkey Score (exact port) | 33.7 | **31.0** | winner |
| oxeylyzer-2 score (exact tool) | −261B | **−245B** | winner |
| oxey approximation | −4.8 | **−15.3** | winner |
| keymeow sfb / sfb-dist | 1.18% / 1.44 | **1.07% / 1.19** | winner |
| keymeow lsb / lsb-dist | 0.60 / 1.23 | **0.09 / 0.18** | winner |
| keymeow alt / redirect | 38.0 / 5.85 | **39.4 / 5.20** | winner |
| sfb (our corpus) | 0.74% | **0.67%** | winner |
| alternation | 76.0% | **77.3%** | winner |
| hand imbalance | 6.33 | **2.29** | winner |
| home row usage | **55.0%** | 52.5% | P10 |
| oxeylyzer-2 stretches | **39.5** | 42.7 | P10 |
| max finger load | 15.9–16.7% | same (tie) | — |

Its genkey 31.0 is the best any keybo search has produced (graphite: 29.5,
semimak: 27.7), bought for ~0.12% predicted speed — inside the ~0.2% plateau, so
the two layouts are speed-equivalent within search noise while this one is
measurably friendlier to community metrics.

Trigram-level trade (layout-diff, P10-w0.5 → winner): it fixes P10-w0.5's single
worst frequent pattern (`ic ` −159ms) and improves `ere`/` wh`, paying with the
`r`-family (`ir `, `ry `, `ers`, `her` — the cost of r on the top-row pinky, which
is also the stretches regression).

## Provenance

- Surface: `bigram_reg_seed{0,1,2}` (REG-LOLO regularized recipe, calibration off
  per CAL-REMOVE) + `trigram_cond_lograt_join_seed{0,1,2}`, T3c at wpm 90.
- Search: P13 recipe, rng 888001, g=1.0 arm (exact genkey in-loop at weight 1.0);
  SA 10×12k + exhaustive 2-opt (`keybo-e2e/p13_stab.py` → `runs/p13_stab.json`).
- Pick: registered min-max-regret rule over {speed, genkey}, speed cap 0.5%,
  pooled over 5 independent reruns (23 distinct layouts) + P10/P11/P10.5 refs.
- Stability context: the 5 reruns produced 5 distinct letter-level picks (plateau
  degeneracy); the recurring invariants are the structure this layout shares with
  P10-w0.5 — consonant home-left, `naei` vowel block home-right, balanced fingers.
