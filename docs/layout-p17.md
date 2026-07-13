# P17 — searching the pick rule directly (2026-07-13)

**The flagship C30M candidate.** Charter 40cf881, outcome 43aee13.

Every prior campaign searched *weighted sums* of the gauges and then picked by
*min-max regret* — an objective/pick mismatch: weighted-sum search can only reach
convex-supported Pareto points, and the min-max optimum routinely sits between
them. P17 closed the gap by searching the pick rule itself (augmented Chebyshev
scalarization, 44 random-weight + 6 equal-weight + 4 warm-started arms), then
polishing the top 10 with exhaustive 2-opt + 3-cycle passes on pure min-max. It
also adopted the SEL-1 reform: pick axes = speed, genkey, oxeylyzer-1,
oxeylyzer-2; weighted-finger-distance demoted to a report row (it double-counted
travel with the oxey scores).

Both registered supersede bars fired: max-regret improved 3.79pp on the same
union pool (bar: ≥0.5pp), and the pick beats P16-balance on 3 of 4 axes
(speed, oxey1, oxey2; concedes genkey 31.3 vs 30.8).

## keybo-c30m (`POL-MMX-r888404`)

```
f y u , .   v g d n l
h i e a o   c s t r m
k j ' q -   b w p x z
```

| gauge | semimak | graphite | P16-balance | **keybo-c30m** |
|---|---|---|---|---|
| speed: typing time saved vs qwerty30M | +2.55% | +2.38% | +3.48% | **+3.53%** |
| max qwerty-gap regret (4 axes, union pool) | 30.9% | 35.6% | 9.2% | **5.4%** |
| genkey Score (↓) | **27.7** | 29.5 | 30.8 | 31.3 |
| oxeylyzer-1 (repl, ↑) | 0.365 | 0.460 | 0.415 | **0.428** |
| oxeylyzer-2 (repl, ↑) | **−190.4B** | −199.1B | −234.1B | −212.2B |
| wfd (report row, ↓) | 1416.7B | 1589.9B | 1531.0B | 1515.9B |
| keymeow sfb | **0.89** | 1.23 | 1.29 | 1.19 |
| keymeow lsb | 1.27 | **0.57** | 1.27 | 1.84 |

Best keybo values ever on oxeylyzer-1 and oxeylyzer-2 (the o2 gap to graphite is
down to 6.6%), better sfb than graphite, and — pleasingly — better wfd than
P16-balance even though wfd no longer voted. Its one concession is keymeow lsb
(1.84); the runner-up `POL-CHEB-r888514` (`pyuo,vgdnlhiea.cstrmkj-z'fwbxq`,
max-regret 5.70%, lsb 0.77) is the documented alternate for lsb-sensitive eyes.

## Why we trust this pick more than any before it

1. **The two selection philosophies agree for the first time.** SEL-1 showed
   worst-axis rules and consensus rules (mean/Borda/random-preference/Copeland)
   picking different siblings. keybo-c30m wins *both*: min-max 5.42% (best) AND
   random-preference win share 32.8% (best) AND Copeland 67.0 (top).
2. **The speed-budget curve says the cap isn't binding:** loosening the speed
   cap from 0.5% to 1.0% buys zero additional balance; tightening to 0.1% costs
   2.8pp. The pick sits at the knee.
3. **Structure confirms the mechanism:** raw equal-weight Chebyshev arms did NOT
   find it — the min-max polish stage did (it's a polished MMX arm in the
   E10-r888301 basin). The convex-hull gap was real, and closing it was worth
   3.8pp.

## Status

keybo-c30m is the flagship **candidate** on the C30M charset. The two open
owner decisions are unchanged: (1) charset — classic 30 (`;`/`/`) vs C30M
(`'`/`-`); (2) promotion of any candidate to THE deliverable. If C30M is
adopted, deferred work: F5M-LR quality retrain, P13-style stability study on
this plateau, and a K31-informed look at whether `q`/`-` placement generalizes
across boards.
