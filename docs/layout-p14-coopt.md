# P14-coopt — the five-gauge balance point (2026-07-13)

```
l c g k v   x . o u ,
r s t h d   y n a e i
z w m p b   f j q / ;
```

String form: `lcgkvx.ou,rsthdynaeizwmpbfjq/;` (qwerty slot order, space unchanged).

## What it is

**Not the primary deliverable** (that is [P10-w0.5](layout-p10-w05.md)) — this is the
first layout searched with the community tools IN the objective rather than applied
post-hoc: fit = T3c speed + weighted units of exact-genkey + oxeylyzer-1 + oxeylyzer-2
+ weighted-finger-distance (ports parity-gated against the real binaries: o2 exact to
the integer, v1 within 0.04%; PREREGISTRATIONS.md 01546c8 → e90e989). The registered
min-max qwerty-gap-regret rule picked it over all 15 searched candidates and both
incumbents. It beat [P13STAB-win](layout-p13stab-winner.md) on 3 of the 4 exact
community tools, earning this doc; the promotion flag did NOT fire (speed regret
+0.19% is inside the plateau but not within 0.1% of P10-w0.5).

## The one-number summary: worst community axis

Qwerty-gap-normalized regret (0% = best-in-pool on that gauge, 100% = qwerty):

| layout | worst axis (max regret) | genkey | oxey1 | oxey2 | wfd | speed |
|---|---|---|---|---|---|---|
| **P14-coopt** | **5.2%** | 4.8% | 2.6% | 5.2% | 4.4% | +0.19% |
| P13STAB-win | 6.5% | 3.0% | 6.3% | 6.5% | 5.3% | +0.12% |
| P10-w0.5 | 10.2% | 0.0%* | 10.2% | 9.3% | 9.4% | +0.00% |

*P10-w0.5's speed regret is the 0% reference; its genkey regret is 3.8%.

## Exact-tool board (repl/binary verified, not port numbers)

| gauge | P10-w0.5 | P13STAB-win | P14-coopt | best |
|---|---|---|---|---|
| genkey Score (exact port, lower better) | 33.7 | 31.0 | **30.9** | P14 |
| oxeylyzer-1 repl Score | 0.333 | 0.367 | **0.400** | P14 |
| oxeylyzer-2 repl score | −260.8B | −245.1B | **−238.5B** | P14 |
| keymeow sfb / sfb-dist | 1.18 / 1.44 | **1.07 / 1.19** | 1.23 / 1.36 | P13win |
| keymeow lsb / lsb-dist | 0.60 / 1.23 | **0.09 / 0.18** | 0.53 / 1.09 | P13win |
| keymeow alt / roll | 38.0 / 45.6 | **39.4** / 44.9 | 38.2 / **46.0** | split |
| keymeow redirect | 5.85 | 5.20 | **4.68** | P14 |
| speed vs qwerty (reg surface, wpm 90) | **+3.83%** | +3.71% | +3.64% | P10 |

Where it sits: **best on the oxeylyzer family and genkey, best redirect, most rolls;
concedes sfb/lsb to P13STAB-win and speed to P10-w0.5.** The disagreement is
structural: the oxey scores reward the short-travel roll-heavy left block (`rst`
home with `l`/`c` above), keymeow prices the sfb mass that block creates
(1.07 → 1.23). No layout in 38 searched candidates dominated the whole board.

## Which to reach for

- **Measured typing speed** → P10-w0.5 (unchanged primary).
- **Best single showing on any one community tool** → P13STAB-win (keymeow) or
  P14-coopt (genkey/oxeylyzer).
- **Never look bad anywhere** (min worst-axis) → P14-coopt.

## Provenance

- Surface: `bigram_reg_seed{0,1,2}` + `trigram_cond_lograt_join_seed{0,1,2}`, T3c at
  wpm 90 (REG-LOLO recipe, calibration off per CAL-REMOVE).
- Objective: speed + Σ w_g · UNIT_g · loss_g, UNIT_g = (speed_q/100)/|loss_g(qwerty)|;
  winning arm OX1 = (genkey 0.25, oxey1 1.0, oxey2 1.0, wfd 0.5), rng 888103.
- Search: SA 10×12k + exhaustive 2-opt, 15 searches (`keybo-e2e/p14_coopt.py` →
  `runs/p14_coopt.json`; parity gate `runs/p14_parity.json`).
- Pick rule: registered min max qwerty-gap-normalized regret over {speed, genkey,
  oxey1, oxey2, wfd}, speed cap 0.5% (PREREGISTRATIONS.md 01546c8).
