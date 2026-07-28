# MODELNORM-1 — predictions scored against the artifacts

18 pre-registered in `PREDICTION.md` (committed `412e58f`, **before** `runs/` existed).
**Result: 11 HELD, 5 FAILED, 2 UNTESTABLE.** Every failure is reported.

All numbers: corpus **blend-v1**, `.native` frame, `TRI_PS_FREQ_PRIOR`, **BAKED 90 WPM**.
MODELLED ONLY.

| # | Prediction | Verdict | Evidence |
|---|---|---|---|
| P1 | per-model "1" beats best-of-8 by ≥0.5 pp further below the random mean | ❌ **FAILED** | COMMUNITY +2.032 pp ✓, POOL +1.384 pp ✓, **AALTO only +0.099 pp** ✗ |
| P2 | two seeds agree to within 0.10 % of span | ✅ HELD (emphatically) | gap **EXACTLY 0.0** on all three; both seeds found the *identical layout* |
| P3 | seed-disagreement spread differs ≥2× across models | ⚠ **UNTESTABLE** | all three gaps are exactly 0, so the ratio is 0/0 |
| P4 | search "1" beats ceiling "1" by >1 % of span, all three | ✅ HELD | 1.212 % / 14.736 % / 15.193 % |
| P5 | per-model rankings unchanged by normalization | ✅ HELD | asserted in code; 0 discordant pairs |
| P6 | equal-weight blend ranking differs from raw-mean ranking (1–3 transpositions) | ❌ **FAILED** | **0** discordant pairs vs raw mean-ms *and* vs raw mean-saved%; rankings identical |
| P7 | any ranking change is below the paired resolution floor | ✅ HELD | the 2 changes vs raw MIN have gaps 0.0122 / 0.0241 vs a 0.2319 conservative floor |
| P8 | blend champion is SLOWER than arm B | ✅ HELD | 256.6268 vs 253.9006 = **+2.7262** |
| P9 | blend champion in [254.0, 257.5], better than arm A and qwerty30m | ✅ HELD | 256.6268 ∈ range; beats arm A 256.8466 and qwerty30m 264.1389 |
| P10 | blend champion beats arm B on ≥1 native surface (likely COMMUNITY) | ✅ HELD | COMMUNITY 221.05 G < 227.71 G ✓ and POOL 235.78 G < 239.72 G ✓; AALTO no. COMMUNITY called correctly |
| P11 | the three solo champions are distinct, each best for its model | ✅ HELD | 3 distinct layouts; each solo cell returned own_blend = **1.000000000** at its anchor |
| P12 | pairwise Hamming between solo champions ≥ 8/30 | ✅ HELD | **24, 26, 24** of 30 — far beyond the bar |
| P13 | (1,1,1) champion within 0.05 of the best solo score on every model | ❌ **FAILED** | POOL 0.014 ✓, COMMUNITY 0.035 ✓, **AALTO 0.097** ✗ |
| P14 | (2,1,1) champion lies strictly between (1,0,0) and (1,1,1) in AALTO score | ✅ HELD | 1.00000 → **0.93740** → 0.90286, monotone |
| P15 | no 10-axis dominator; best n_ge ≤ 4 | ❌ **FAILED** (half) | no dominator ✓, but **best n_ge = 5/10** for blend-equal and **7/10** for sweep-aalto-only — above my bound |
| P16 | normalized floor POSITIVE | ✅ HELD | +0.9029 (blend-equal); every sweep cell +0.759 … +0.937 |
| P17 | all 8 candidates within a 0.25-wide normalized window on every model | ❌ **FAILED** as stated | windows 0.4230 / 0.4283 / 0.3241 — because **qwerty30m** alone spans most of it. Excluding it the other 7 sit in **0.1696 / 0.0895 / 0.0962** |
| P18 | equal-weight blend spread over the 8 candidates < 0.30 | ❌ **FAILED** as stated | 0.3796 with qwerty30m; **0.0934** without it |

## What the failures taught

**P6 is the most informative failure, and it is the direct answer to deliverable B.** I
predicted normalizing would re-order something versus a raw aggregate. It re-orders *nothing*
versus the raw mean. The reason is measurable: on these 8 candidates the three per-model
normalized scores are so strongly co-monotone that dividing by per-model spans — which is
exactly what the design does — cannot flip an adjacent pair. Normalization is doing real work
on the *interpretation* of the weight (P14 confirms it) while doing **no work at all** on the
equal-weight ranking. Both halves matter, and I would have asserted only the first.

**P1 and P13 fail on the same model for the same reason,** which makes them one finding rather
than two: **AALTO is nearly saturated by layouts that already exist.** arm B is at 0.9879 of
AALTO's own optimum, so a 10M-eval search can only find +0.099 pp more headroom (P1), and an
equal-weight blend must give up 0.097 of AALTO to buy COMMUNITY and POOL (P13). The design's
premise — that per-model optima make the scale a property of the *model* — is right, but the
*size* of the correction it buys is very unequal across the three models: large on COMMUNITY
and POOL, nearly nil on AALTO.

**P15's failure is a bound error, not a direction error.** No dominator exists (the campaign's
standing result survives), but I set the n_ge ceiling from the campaign's prior best of 3/10
without accounting for my own `floor` axis being a *different quantity* (this arm's normalized
min-over-three-models, not arm E's six-surface ceiling-fraction floor). A candidate optimized
directly against that floor naturally scores well on it. **The dominance verdict is comparable
to other arms; the n_ge number is not**, and I should have predicted a range only for the
verdict.

**P17/P18 failed on the letter and held on the mechanism.** I predicted a narrow occupancy
window and gave the right reason (random layouts are 2.5–3 sd worse than qwerty, so the scale
spends its range where no candidate lives) — but I forgot that **qwerty30m is itself one of the
8 candidates**, and it is the one layout that sits far down the scale. Excluding it, the
remaining 7 occupy 0.09–0.17 of the range, which is *tighter* than I predicted. So the defect
is real and worse than stated; my prediction was just arithmetically inconsistent with its own
candidate list.

**P3 is untestable in the best possible way.** I predicted the two seeds would converge
unevenly across models, so that the scheme's own failure mode would show up at some measurable
level. Instead all three models' seeds landed on the *identical layout*, making the ratio 0/0.
The failure mode is absent at this budget rather than merely small — which is a stronger result
than the one I predicted, and it is the reason the normalization can be called stable.
