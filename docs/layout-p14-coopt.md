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

## Reference board (qwerty / dvorak / semimak / graphite added)

Same gauges with the community anchors. Exact tools use each tool's canonical
layout files; "speed" for the apostrophe-bearing references (dvorak/semimak/
graphite) is computed on the common-subset corpus (91.4% mass) — the shared
convention for cross-charset comparisons. Our-corpus pattern metrics exclude the
~5–6% of corpus mass not typeable on a given 30-key layout.

| gauge | qwerty | dvorak | semimak | graphite | P10-w0.5 | P13STAB-win | P14-coopt |
|---|---|---|---|---|---|---|---|
| speed vs qwerty, wpm 90 | 0 | +3.30% | +2.60% | +2.51% | **+3.83%** | +3.71% | +3.64% |
| genkey Score (↓) | 110.8 | 48.2 | **27.7** | 29.5 | 33.7 | 31.0 | 30.9 |
| oxeylyzer-1 Score (↑) | −0.437 | 0.211 | 0.365 | **0.460** | 0.333 | 0.367 | 0.400 |
| oxeylyzer-2 score (↑) | −721.9B | −391.0B | **−190.4B** | −199.1B | −260.8B | −245.1B | −238.5B |
| keymeow sfb / sfb-dist | 6.70/9.59 | 2.80/3.31 | **0.89/1.05** | 1.23/1.45 | 1.18/1.44 | 1.07/1.19 | 1.23/1.36 |
| keymeow lsb / lsb-dist | 2.99/6.66 | 0.93/2.01 | 1.27/2.57 | 0.57/1.24 | 0.60/1.23 | **0.09/0.18** | 0.53/1.09 |
| keymeow alt / roll / redir | 26.8/37.7/13.4 | 44.9/38.8/3.4 | 40.6/42.7/6.6 | 42.2/44.2/**3.0** | 38.0/45.6/5.9 | 39.4/44.9/5.2 | 38.2/**46.0**/4.7 |
| sfb % (our corpus) | 4.43 | 1.86 | **0.55** | 0.77 | 0.74 | 0.67 | 0.78 |
| alternation % | 68.2 | **80.2** | 77.4 | 78.9 | 76.0 | 77.3 | 76.3 |
| rolls % | 17.2 | 6.9 | 10.2 | 7.7 | 8.5 | 8.2 | 8.8 |
| hand imbalance | 12.1 | 8.1 | 7.5 | 2.6 | 5.3 | **1.9** | 2.2 |
| home-row usage % | 26.7 | **57.0** | 52.6 | 53.4 | 55.0 | 52.5 | 55.0 |

(Max finger load is deliberately not a gauge: lower is not better — fingers should
be loaded in proportion to their speed, which genkey's fspeed and the oxeylyzer
finger weights already price.)

Reading: on predicted typing speed the keybo family leads everything (semimak and
graphite give up ~1.2–1.3pp vs P10-w0.5; dvorak ~0.5pp); on the community tools our
layouts now sit between graphite/semimak and dvorak — P14-coopt is second overall
on oxeylyzer-1 (above semimak), within ~3 genkey points of semimak, and about a
third of the P10→semimak gap on oxeylyzer-2 is closed. semimak keeps the sfb
crown, P13STAB-win the lsb crown, dvorak alternation, graphite redirects.

## P14c-o2f — the oxeylyzer-2 frontier candidate (2026-07-13)

```
h r f m k   , y u o j
l n s t d   g c i a e
z x b v q   p w . ; /
```

String form: `hrfmk,yuojlnstdgciaezxbvqpw.;/`. From the P14c o2-forward weight
sweep (arm O2H2: genkey 0.25 / oxey1 0.5 / oxey2 3.0 / wfd 0.5; PREREGISTRATIONS
d30af30 → c8edffe). Answers "how close can we get on oxeylyzer-2 without leaving
the speed plateau": **o2 repl −194.6B — ahead of graphite (−199.1B), second only
to semimak (−190.4B)**, closing 79% of the P10-w0.5 → semimak o2 gap, at +0.35%
speed regret. Its keymeow sfb (0.945/1.116) beats both P13STAB-win and graphite.
It pays with oxeylyzer-1 (0.387, below P14-coopt's 0.400) and the most speed any
documented sibling concedes (+0.35%, still in-plateau). Right-hand vowel block
`ciae` with `y/u/o` above is the o2-pleasing structure; note it is a *different*
vowel arrangement than the `naei` invariant of the P10 family — the o2 weighting
is strong enough to reorganize the vowel block.

P14-coopt remains the documented balance point (P14c's five-gauge pick was the
P14b layout already rejected on exact tools). The three siblings now span the
frontier: P13STAB-win (keymeow/lsb pole), P14-coopt (balance), P14c-o2f (oxey pole).
