### T0 — POSITIVE CONTROL: does this pipeline reproduce the published cells?

| check | measured | published | abs Δ | verdict |
|---|---|---|---|---|
| `random-wide` | +0.7970054 | +0.7970054 | 0.00e+00 | **PASS** |
| `archive-x400` | +0.2184272 | +0.2184272 | 0.00e+00 | **PASS** |
| `within:random-wide` | +0.9872249 | +0.9872000 | 2.49e-05 | **PASS** |
| `within:archive-x400` | +0.9647336 | +0.9647000 | 3.36e-05 | **PASS** |

Archive bank 2860 layouts; reference bank 200000; frame `native`.

### T1 — the SLACK is the SAME algebraic identity (defect found in the brief)

Exactly, for any pool, with `q = u_A/u_B` and `k = sd(C)/sd(D)`:

```
r_Pearson = [(k^2-1)/(k^2+1)] * (1+q^2)/(2q)
SLACK     = r_Pearson - (k^2-1)/(k^2+1) = [(k^2-1)/(k^2+1)] * ((1+q^2)/(2q) - 1)
```

| published cell | k | q = u_A/u_B | predicted r | measured r | Δ | predicted SLACK | reported SLACK | Δ |
|---|---|---|---|---|---|---|---|---|
| `random-wide` | 3.063 | 0.9982 | +0.807377 | +0.807377 | +0.0e+00 | +0.00000 | +0.00000 | +5.1e-17 |
| `archive` | 1.058 | 0.2488 | +0.119216 | +0.119216 | -3.6e-16 | +0.06336 | +0.06336 | -3.6e-16 |
| `boxmatch` | 1.093 | 1.0254 | +0.089144 | +0.089144 | -1.2e-16 | +0.00003 | +0.00003 | -1.3e-16 |
| `kswap1` | 3.817 | 0.9663 | +0.872054 | +0.872054 | +0.0e+00 | +0.00051 | +0.00051 | -4.4e-17 |
| `kswap3` | 4.162 | 0.9972 | +0.890869 | +0.890869 | +3.3e-16 | +0.00000 | +0.00000 | +3.4e-16 |
| `kswap8` | 4.360 | 1.0129 | +0.900104 | +0.900104 | -1.1e-16 | +0.00007 | +0.00007 | -5.7e-17 |
| `kswap20` | 4.018 | 0.9884 | +0.883399 | +0.883399 | +4.4e-16 | +0.00006 | +0.00006 | +4.7e-16 |
| `restrictC-0.02` | 0.067 | 0.9960 | -0.991184 | -0.991184 | +2.2e-16 | -0.00001 | -0.00001 | +2.0e-16 |
| `restrictD-0.02` | 172.848 | 1.0001 | +0.999933 | +0.999933 | +2.2e-16 | +0.00000 | +0.00000 | +2.2e-16 |
| `restrictC-0.1` | 0.361 | 0.9775 | -0.769734 | -0.769734 | -2.2e-16 | -0.00020 | -0.00020 | -2.5e-16 |
| `restrictD-0.1` | 34.797 | 1.0018 | +0.998351 | +0.998351 | +1.1e-16 | +0.00000 | +0.00000 | +1.7e-16 |
| `restrictC-0.4` | 1.332 | 1.1101 | +0.280403 | +0.280403 | +1.7e-16 | +0.00152 | +0.00152 | +1.5e-16 |
| `restrictD-0.4` | 8.288 | 0.9832 | +0.971441 | +0.971441 | -1.1e-16 | +0.00014 | +0.00014 | -1.4e-16 |

**max |predicted r − measured r| = 4.441e-16** over 13 rows; **max |predicted SLACK − reported SLACK| = 4.725e-16**.

### T2 — THE PRIMARY RESULT: both agreements, archive vs the matched asymmetric random pool

| pool | lineage | n | ACHIEVED u_A | ACHIEVED u_B | ACHIEVED q | inst-vs-inst (cross) | 95% CI | inst-vs-itself (within) |
|---|---|---|---|---|---|---|---|---|
| `random-wide` | random | 400 | 0.9693 | 0.9711 | 0.9982 | **+0.7970** | [+0.750, +0.836] | **+0.9872** |
| `archive-x400` | **archive** | 400 | 0.0399 | 0.1605 | 0.2488 | **+0.2184** | [+0.114, +0.321] | **+0.9647** |
| `asym-match` (R=12, mean) | random | 400 | 0.0398 | 0.1606 | **0.2480** ± 0.0048 | **+0.0131** ± 0.0357 | (replicate sd) | **+0.6955** ± 0.0146 |
| `sym-match` (R=12, mean) | random | 400 | — | — | 0.9997 | +0.0194 ± 0.0447 | (replicate sd) | +0.2345 ± 0.0239 |

Archive target was `u_A = 0.039921`, `u_B = 0.160464` (**achieved**, not requested), `q = 0.2488`, `u_geo = 0.080037`.

### T3 — the two inferential tests

| test | pairing | statistic | value | resolution / CI | p |
|---|---|---|---|---|---|
| `asym − sym`, CROSS leg | **paired** (same bank+seed, only asymmetry differs) | mean Δrho over R=12 | -0.0063 | replicate sd of Δ = 0.0607 | Wilcoxon **0.7334** |
| `asym − sym`, WITHIN leg | **paired** (same) | mean Δrho over R=12 | **+0.4611** | replicate sd of Δ = 0.0257 | Wilcoxon **0.00049** |
| `archive − asym`, CROSS leg | unpaired (disjoint universes, different lineage) | Δrho | **+0.2303** | bootstrap CI [+0.0865, +0.3742] | **0.0015** |
| `archive − asym`, CROSS leg (replicate mean) | unpaired | Δrho | +0.2053 | 5.76 replicate sds | — |

### T4 — the q-ladder at FIXED geometric-mean narrowness (only asymmetry moves)

`u_A = √q · u_geo`, `u_B = u_geo / √q`, so `u_A·u_B = u_geo²` for every q.

| requested q | ACHIEVED q | ACHIEVED u_A | ACHIEVED u_B | cross | within |
|---|---|---|---|---|---|
| 0.0625 | 0.0631 | 0.0204 | 0.3228 | +0.0615 | +0.9161 |
| 0.1250 | 0.1253 | 0.0279 | 0.2229 | +0.0686 | +0.8285 |
| 0.2500 | 0.2506 | 0.0401 | 0.1602 | +0.0789 | +0.6940 |
| 0.5000 | 0.5003 | 0.0561 | 0.1122 | -0.0289 | +0.4892 |
| 1.0000 | 0.9769 | 0.0785 | 0.0803 | -0.1006 | +0.2515 |
| 2.0000 | 2.0142 | 0.1152 | 0.0572 | +0.0377 | -0.0363 |
| 4.0000 | 4.0539 | 0.1619 | 0.0399 | -0.1841 | -0.2280 |
| 8.0000 | 8.0076 | 0.2238 | 0.0280 | +0.0486 | -0.3379 |
| 16.0000 | 15.5639 | 0.3160 | 0.0203 | +0.0520 | -0.4004 |

### T5 — the LEVEL ladder (ratio held, overall narrowness scaled)

| cell | ACHIEVED u_A | ACHIEVED u_B | ACHIEVED q | cross | 95% CI | within |
|---|---|---|---|---|---|---|
| `level-asym-1x` | 0.0394 | 0.1593 | 0.2471 | +0.0047 | [-0.090, +0.102] | +0.6971 |
| `level-sym-1x` | 0.0804 | 0.0794 | 1.0132 | +0.0547 | [-0.043, +0.154] | +0.1916 |
| `level-asym-2x` | 0.0801 | 0.3225 | 0.2484 | +0.0588 | [-0.033, +0.150] | +0.9029 |
| `level-sym-2x` | 0.1574 | 0.1575 | 0.9994 | +0.0383 | [-0.060, +0.133] | +0.6709 |
| `level-asym-4x` | 0.1648 | 0.6006 | 0.2743 | +0.2998 | [+0.210, +0.386] | +0.9676 |
| `level-sym-4x` | 0.3242 | 0.3217 | 1.0079 | +0.2880 | [+0.197, +0.375] | +0.9018 |

### T6 — CONFIRMATORY C1: is the WITHIN-leg comparison fair? (matched on u_seed, the axis the statistic lives on)

| arm | matched on | mean u_seed_geo | archive's u_seed_geo | mean within | mean cross |
|---|---|---|---|---|---|
| `archive-x400` | — | 0.1617 | 0.1617 | **+0.9647** | **+0.2184** |
| `match_uB` (R=6) | seedMEAN `u_A`,`u_B` | 0.1815 | 0.1617 | +0.6938 ± 0.0118 | +0.0647 ± 0.0661 |
| `match_useed` (R=6) | **PER-SEED** `u_seed` | 0.1817 | 0.1617 | +0.6925 ± 0.0258 | +0.0254 ± 0.0339 |

`archive − match-uB` on WITHIN = **+0.2710**, on CROSS = **+0.1537**.  
`archive − match-useed` on WITHIN = **+0.2723**, on CROSS = **+0.1930**.

### T7 — CONFIRMATORY C2: replicated q-ladder (F4 retest)

| requested q | ACHIEVED q (mean) | cross mean | replicate sd | within mean |
|---|---|---|---|---|
| 0.0625 | 0.0620 | +0.0628 | 0.0360 | +0.9086 |
| 0.25 | 0.2506 | +0.0288 | 0.0221 | +0.6902 |
| 1 | 1.0106 | +0.0202 | 0.0574 | +0.2336 |
| 4 | 4.0213 | -0.0294 | 0.0757 | -0.2102 |
| 16 | 16.0132 | +0.0496 | 0.0472 | -0.3993 |

| F4 test: q vs 1/q | cross mean at q | cross mean at 1/q | \|Δ means\| | pooled replicate sd | Mann-Whitney p | exceeds 0.20? |
|---|---|---|---|---|---|---|
| 0.0625 vs 16 | +0.0628 | +0.0496 | 0.0132 | 0.0420 | 0.5887 | no |
| 0.2500 vs 4 | +0.0288 | -0.0294 | 0.0582 | 0.0558 | 0.0649 | no |

### T8 — CONFIRMATORY C3: does the 4× two-legged signature replicate? (F2 retest)

| arm | mean u_B | mean u_seed_geo | cross mean | replicate sd | within mean | replicate sd |
|---|---|---|---|---|---|---|
| `asym-4x` | 0.5972 | 0.6012 | +0.2457 | 0.0311 | +0.9660 | 0.0028 |
| `sym-4x` | 0.3184 | 0.3283 | +0.3155 | 0.0592 | +0.9103 | 0.0047 |

Reproduces BOTH legs (within ≥ +0.90 **and** cross ≤ +0.30): **True**. Its cross ±2 replicate sd contains the archive's +0.2184: **True**.

### T9 — BOUNDING: the per-seed-matched arm (bracket reopened) and the quantitative bound

| arm | ACHIEVED u_B | ACHIEVED u_seed_geo | within | cross |
|---|---|---|---|---|
| `archive-x400` | 0.1605 | 0.1617 | **+0.9647** | **+0.2184** |
| B1 `u_seed`-matched random (R=4) | 0.1382 | 0.1613 | +0.6259 ± 0.0174 | +0.0645 ± 0.0201 |
| B2 `within`-matched random (R=4) | **0.5991** | 0.6035 | +0.9664 ± 0.0017 | +0.2916 ± 0.0234 |

B1 bracketed: **True**, `u_seed` miss -0.0154. At MATCHED per-seed spread the archive leads by **+0.3388** on the within leg and **+0.1540** on the cross leg.

B2: a random pool needs `u_B` = **0.5991** = **3.73×** the archive's 0.1605 to reach the archive's within-reliability +0.9647 — and its cross-source rho there is +0.2916, i.e. the two legs do NOT move together.

