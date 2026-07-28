# ARM E — the evidence-weight search on the ARCHIVE-fitted curves, under CLAMP

**Outcome: (E3), but at its very bottom edge — and the story is two-sided.**
Arm E's champion is **258.1803 ms/char**: **+4.2797 slower than arm B**, **+3.5495 slower than the
worst incumbent**, **+1.3337 slower than arm A** — yet **5.5338 FASTER than qwerty** and **11.0959
FASTER than arm D**.

⚠ **Two qwertys are in play and the brief's list mixes them; both are quoted throughout.**
`qwerty` **263.7141** is the CLASSIC `;./` charset (`analyze`'s own `--ref` row) and is the figure
the brief's comparison set and my pre-registration's E3 band use. `qwerty30m` **264.1389** is the
C30M-charset variant on the board. Arm E is faster than both (**−5.5338** and **−5.9586**), so the
"worse than qwerty" prediction (P5) fails on either reading.

Corpus **blend-v1 (production default)**, `md5(trigrams.txt) = c5066fa7bcc46dea1ecbc987fb465b4a`,
frame **`.native`** (asserted as a hard gate from the weights JSON), **90 WPM**. Weights
`state/evidence-scorer/artifacts/arm-archive400-native.json` (`COMMUNITY_BASE`, pool `archive-400`,
n=400), priced through the validated `LossCurve.price_many(..., policy=CLAMP)`. **MODELLED ONLY** —
every number is attribution of a *fitted* timing surface, not measured typing. tau saturated,
Phase-D cancelled. **No layout here is promoted or adopted; that is the user's gate alone.**
🟢 = verified, 🟡 = read from source, 🟠 = inferred, 🔴 = uncertain.

---

## 1. The board

| layout | **ARCH clamp** | r400 clamp | r400 extrap | **ms/char** | vs best inc | normfloor | ood_A | ood_R |
|---|---|---|---|---|---|---|---|---|
| **arm E** `ou-qdbpmlsaiehvgctnr.,y'kfwjzx` | **−2.6902** | −18.4729 | −37.0958 | **258.1803** | **+3.5495** | **+0.398631** | 6/14 | 9/14 |
| arm B *(baseline, fastest)* | −0.2324 | −18.5790 | −41.3178 | **253.9006** | −0.7302 | +0.600450 | 0/14 | 10/14 |
| keybo-lsb *(best incumbent)* | −1.1902 | −17.8939 | −37.4618 | 254.6307 | — | +0.726952 | 0/14 | 9/14 |
| keybo-lsb+lm | −0.9206 | −17.8939 | −38.3090 | 254.6847 | +0.0539 | +0.730176 | 0/14 | 9/14 |
| lsb-sib | −1.8638 | −18.4986 | −37.5827 | 254.7058 | +0.0750 | +0.739914 | 1/14 | 9/14 |
| archive-1846 | −1.2961 | −18.3966 | −39.1339 | 254.7961 | +0.1653 | +0.745186 | 1/14 | 9/14 |
| archive-1843 | −1.3652 | −18.3161 | −38.9624 | 254.8436 | +0.2129 | +0.744198 | 2/14 | 9/14 |
| flagship-c3 | −1.5899 | −18.5402 | −38.9980 | 254.9761 | +0.3454 | +0.745096 | 1/14 | 9/14 |
| arm C | −1.5063 | −18.5202 | −45.0664 | 256.0220 | +1.3912 | +0.568914 | 4/14 | 10/14 |
| arm A | −1.2976 | −18.6413 | −45.4363 | 256.8466 | +2.2158 | +0.583619 | 6/14 | 10/14 |
| semimak | −1.8358 | −18.5026 | −36.7063 | 257.3915 | +2.7608 | +0.551121 | 6/14 | 8/14 |
| graphite | −1.8973 | −18.5975 | −37.2531 | 258.1696 | +3.5388 | +0.601021 | 3/14 | 8/14 |
| **qwerty30m** | +1.9434 | −6.9764 | −6.9764 | 264.1389 | +9.5082 | 0.000000 | 14/14 | 0/14 |
| **arm D** | −0.1375 | −23.3157 | −23.5812 | **269.2762** | +14.6455 | −0.563179 | 9/14 | 3/14 |

🟢 **258.1803 verified twice** — through the fast evaluator and independently through the shipped
`keybo analyze --json` (blend-v1, 90 WPM, skipgram `1-skip31.txt`), exact to 4 dp. Set-containment
asserted, **not** row count: 2 requested → 3 returned, 0 missing, the extra being `analyze`'s own
`--ref` row `qwertyuiopasdfghjkl;zxcvbnm,./` (trap 38).

⚠ **FOUR RULERS ON TWO AXES, never mixed.** Arm E optimized **ARCH clamp**; arm D **r400 clamp**;
arms A/C **r400 extrapolate**. `ood_A` counts out-of-domain against the ARCHIVE domains, `ood_R`
against random400's. Quoting one evidence column across arms is meaningless — and here there are
*two* axes of difference (fit pool × domain policy), not one.

⚠ **arm E 258.1803 vs graphite 258.1696 is a COINCIDENCE.** The gap is 0.0107 ms/char — below the
0.4964 conservative paired floor, so it does not resolve — and the two layouts share **0 of 30**
character positions. Nothing should be read into it. 🟢

### Does the gap resolve? Yes — all 10 arm-E pairs, by 2.69× to 22.35×.
PAIRED resolution over **my** named pool, **n=11** (5 champions + 6 incumbents; qwerty/graphite/
semimak deliberately excluded): median **0.1406**, p95 0.4387, **conservative max 0.4964** ms/char.
Unpaired 0.4628 is the wrong ruler (trap 37). SS shares: layout **99.55%**, **seed 0.37%**,
residual 0.08%, n_seeds=3.

| pair | Δ ms/char | × conservative floor | resolves |
|---|---|---|---|
| arm E vs arm D | **−11.0959** | 22.35× | 🟢 yes (arm E faster) |
| arm E vs arm B | **+4.2797** | 8.62× | 🟢 yes |
| arm E vs keybo-lsb | +3.5495 | 7.15× | 🟢 yes |
| arm E vs keybo-lsb+lm | +3.4956 | 7.04× | 🟢 yes |
| arm E vs lsb-sib | +3.4745 | 7.00× | 🟢 yes |
| arm E vs archive-1846 | +3.3842 | 6.82× | 🟢 yes |
| arm E vs archive-1843 | +3.3366 | 6.72× | 🟢 yes |
| arm E vs flagship-c3 | +3.2042 | 6.46× | 🟢 yes |
| arm E vs arm C | +2.1583 | 4.35× | 🟢 yes |
| arm E vs arm A | +1.3337 | 2.69× | 🟢 yes |

⚠ **On the floor's pool, per the brief.** My n=11 max is **0.4964** — numerically identical to arm
D's n=10 figure because the widest pair is the same one; the parent's 0.2222 was n=8 and another
artifact's 0.1406 was n=11 (which happens to equal my *median*). I used my own, larger, more
conservative value. **And I computed the seed share for my own pool rather than reusing
FLAGSHIP-1's 78–83% iWeb figure: it is 0.37%** — pairing is the correct instrument but does little
work when the layout effect spans 15 ms/char. 🟢

---

## 2. Out-of-domain: **6 of 14** — and all six sit at their curve's own optimum

Arm A 10/14 (r400), arm D 3/14 (r400), arm E **6/14 (archive)**.

🟢 **The clamp binds, exactly.** On arm E's own champion, through the same `ValidatedClampedEval`
the search used: pushing **any** of the 14 gauges 50 *and* 1000 domain-widths past either edge
changes the total by **exactly 0.000e+00**. `all_bind = True`, worst |reward outside| =
**0.000e+00**. This was pre-registered abort condition **P7** and it passed on all 14.

🟢 **And every one of the 6 is clamped to that gauge's own in-domain optimum** — because **9 of the
14 archive curves are minimized AT a domain edge**, so the clamped optimum lies *on* the boundary
by construction and a layout at it reads as outside:

| gauge | level | outside by | clamped to | curve argmin | mechanism of the move |
|---|---|---|---|---|---|
| `sfb` | 1.4014 | 0.1236 | 1.5249 | 1.5249 | **right** (lower sfb = faster) |
| `sfb-dist` | 1.6459 | 0.1219 | 1.7677 | 1.7677 | **right** |
| `sfs` | 6.1922 | 0.0719 | 6.2640 | 6.2640 | **right** |
| `sfs-dist` | 7.3181 | 0.1247 | 7.4428 | 7.4428 | **right** |
| `scissor` | 0.0548 | 0.0006 | 0.0554 | 0.0554 | **right** |
| `comfort` | 4.0015 | 0.1644 | 3.8371 | 3.8371 | **WRONG** |

**This is the opposite of arm D's out-of-domain signature.** Arm D's 3 were all marginally *below*
a floor on gauges it was making worse. Arm E's 6 are 5 mechanism-**right** pushes (genuinely lower
same-finger and scissor rates) plus `comfort`. So the out-of-domain count here is not pathology —
it is the search having driven five gauges to the *good* end of their support and been correctly
stopped from paying beyond it. 🟢

---

## 3. 🔑 The mechanism: domain coverage bought **72%** of arm D's damage back, and then stalled

Arm D's excess over arm B was **+15.3756** ms/char. Arm E's is **+4.2797**. So changing *only the
fit pool* — same engine, same seed, same budget, same CLAMP — recovered **11.0959 of 15.3756 =
72%**. 🟢 That is a large, real effect and it vindicates the brief's premise that arm D was partly
an artifact of a fit whose domains did not cover the band.

**The premise itself is verified, exactly** (trap 20 — I re-derived every number rather than
trusting the brief): keybo-lsb is out-of-domain on **0 of 14** gauges under archive weights vs
**9 of 14** under random400; flagship-c3 **1 of 14** vs **9 of 14**; `comfort` ARCHIVE
[3.2519, 3.8371] vs random400 [6.5236, 11.5644]; `sr-roll` ARCHIVE [11.3865, 18.0617] vs random400
[1.9997, 8.3369]. Sharpening it: **6 of the 14 domains are FULLY DISJOINT** between the two fits
(`sfb`, `sfb-dist`, `alt`, `sr-roll`, `oxey-style`, `comfort`). 🟢

**And arm E is therefore a genuine test of the curves in a way arm D structurally could not be:**
🟢 CLAMP freezes **0.00%** of the fitted attribution into a constant across the 10-layout band
(under random400 it froze **82.55%**), with **0 exact ties**; arm B, keybo-lsb and keybo-lsb+lm have
`ev_clamp == ev_extrapolate` to the bit. The archive fit is also the better *fit*: holdout R²
**0.7591 vs 0.4286**, effective dof 3.99 vs 5.03.

### But the objective still points partly the wrong way *inside its own support*

🟢 **What the search actually did**, vs keybo-lsb — measured, not assumed:

| gauge | keybo-lsb → arm E | dir | mechanism | attribution |
|---|---|---|---|---|
| `sfs-dist` | 8.9906 → 7.3181 | down | right | 22.48% |
| **`oxey-style`** | −3.2497 → **−0.9924** | up | **WRONG** | **12.15%** |
| **`comfort`** | 3.7109 → **4.0015** | up | **WRONG** | **11.16%** |
| `sfs` | 7.6488 → 6.1922 | down | right | 9.19% |
| `sr-roll` | 12.6921 → 15.1349 | up | right | 8.62% |
| `sfb` | 1.6231 → 1.4014 | down | right | 6.26% |
| `sfb-dist` | 1.9031 → 1.6459 | down | right | 5.12% |
| **`imbalance`** | 2.0779 → **2.4345** | up | **WRONG** | 4.26% |
| **`redir`** | 3.3584 → **6.1491** | up | **WRONG** | 4.14% |
| `scissor` | 0.1429 → 0.0548 | down | right | 3.90% |
| **`lsb`** | 0.9219 → **1.6422** | up | **WRONG** | 3.71% |
| `roll` | 41.6249 → 42.7950 | up | right | 3.58% |
| **`alt`** | 45.1561 → **41.4455** | down | **WRONG** | 3.53% |
| **`lsb-dist`** | 1.8960 → **3.5100** | up | **WRONG** | 1.90% |

**7 moves mechanism-right, 7 mechanism-WRONG, and the wrong ones carry 40.84% of the attribution.**
`redir` nearly doubled (3.36 → 6.15) and `lsb-dist` nearly doubled (1.90 → 3.51) — both
mechanistically bad, both paid for by the fitted curves inside their own fitted domains.

**So the pathology is milder than arm D's but the same in kind:** the curves are mis-specified where
they are supported, and a better-covered, better-*fitting* pool reduces the damage without removing
it. 🟢

### Why 72% and not 100%: the ruler is still anti-informative in the band

🟢 On an independently constructed pool (3600 1–4-swap perturbations of the six incumbents, chosen
by **neither** objective; instrument positive control ρ = 1.0000):

| band | n | **ARCH clamp** *(arm E's ruler)* | ARCH extrap | r400 clamp *(arm D's)* | r400 extrap |
|---|---|---|---|---|---|
| all | 3600 | **+0.7272** | −0.1213 | +0.5586 | +0.9017 |
| ≤257.0 | 1010 | **+0.4195** | +0.1111 | +0.1237 | +0.5966 |
| ≤256.0 | 590 | **+0.2568** | +0.1842 | +0.0416 | +0.3184 |
| ≤255.5 | 305 | **+0.1609** | +0.1413 | −0.0692 | +0.1293 |
| ≤255.0 | 104 | **+0.0580** | +0.0100 | +0.0274 | +0.0809 |

The archive ruler is **better than arm D's in every band** (which is the 72%) and still **decays to
≈0 exactly where a search operates** (which is the remaining 28%). The sharpest single statement:
🟢 **the archive objective ranks arm B — the fastest layout the campaign has produced — 12th of 14
on its own ruler** (−0.2324, ahead of only arm D and qwerty), and ρ(ev, ms/char) **over the six
incumbents alone is −0.6000**. A good archive score is not evidence of a fast layout.

---

## 4. Which outcome? **E3**, but the narrowest possible reading of it

- **(E1) arm E ≤ 254.63 ⇒ the domains were the whole problem: 🔴 REFUTED.** +3.5495 from the best
  incumbent, 7.15× the resolution.
- **(E2) 254.6–256.9 ⇒ partial: 🔴 REFUTED**, by 1.2803 ms/char. Arm E missed the E2 band, but only
  just — and E2's *substance* ("domains mattered, curves are still weak") is closer to what the
  evidence shows than E3's stated implication.
- **(E3) ≥ 256.9 ⇒ the CURVES are the defect regardless of fit pool: 🟢 HOLDS on the number.**

⚠ **But I will not report E3 as stated.** My pre-registration wrote E3 as *"the curves are the
defect regardless of fit pool, which CONFIRMS ARMD-1 on the strongest possible evidence and closes
the evidence-weight line entirely."* The number lands in E3; **the second half of that sentence is
not what the data says**, and honouring the pre-registration means saying so rather than claiming
the stronger conclusion the band was labelled with:

- "**Regardless of fit pool**" is **wrong as an absolute**: the pool moved the result by 11.0959
  ms/char, 22× the resolution floor. The correct statement is *the fit pool matters a great deal and
  is still not sufficient.*
- "**Closes the line entirely**" is **overclaiming**: arm E is a large improvement on arm D and only
  1.33 behind arm A, obtained by fixing one thing. It is fair to say the line has not produced a
  layout competitive with the incumbents in five arms, and that its remaining defect is now
  localized (40.84% of attribution moving the wrong way in-domain). It is not fair to call it
  closed by this arm.

**ARMD-1's diagnosis needs narrowing, not confirming.** Its claim was "the curves are
mis-specified *where they are supported*". Arm E shows that is **true of both fits but to very
different degrees** — arm D's random400 curves were catastrophically wrong in-domain
(92.5% of headroom mechanism-wrong, champion slower than qwerty), the archive curves are
*partially* wrong in-domain (40.84% of attribution moved wrong, champion faster than qwerty but
slower than every incumbent). The honest form is: **mis-specification in-domain is a property of
these fitted curves generally, and its severity is a property of the fit.** 🟢

**Direct answer to the user's live question** ("can we optimize a layout now that we have greatly
improved things?"): **not with these evidence weights.** The best-posed version of that search —
better-fitting curves, domains that genuinely cover the near-optimal band, a clamp that provably
binds and freezes nothing — still yields a layout **4.28 ms/char slower than the incumbent search
already produces** and **3.55 slower than the best hand-tuned incumbent**. 🟢

⚠ **This does NOT rehabilitate the archive weights as a scorer**, and I am not reporting it as
doing so. EVIDENCE-SCORER-1 rejected them on 0 of 12 cross-source cells inside the noise band; arm
E tested a different property (domain coverage for a search) and the two verdicts are consistent —
the weights are unfit for both purposes, for the same underlying reason.

---

## 5. Optimizing the ruler, and the normalized floor

🟢 **Arm E wins its own ruler outright and loses on speed.** It scores **−2.6902** on ARCH clamp —
**better than all 13 other layouts**, beating the best incumbent keybo-lsb (−1.1902) by 1.50 units
— while being **3.5495 ms/char slower**. It beats arm B by **2.46 units** on the ruler while being
**4.28 ms/char slower**. That is the cleanest optimizing-the-ruler pair in the arm.

On the **19-gauge frame** it wins **7 of 18** scored gauges against every incumbent
(`sfb`, `sfb-dist`, `sfs`, `sfs-dist`, `scissor`, `roll`, `sr-roll`), losing 11. ⚠ Per **correlation
cluster** that is only **5 of 11** (`sfb+sfb-dist`, `sfs+sfs-dist`, `roll`, `sr-roll`, `scissor`) —
effective dof **3.99**, so the raw count over-counts independent evidence (trap 39). `sfr` excluded
as a permutation invariant (trap 23).

🟢 **And all four *independent* community gauges LOSE** — `genkey`, `oxeylyzer1`, `oxeylyzer2`,
`wfd`. Arm E's wins are concentrated exactly in the gauge families its own objective prices; the
gauges fitted by nobody in this loop all say it is worse.

⚠ **P12 FAILED — the normalized floor is POSITIVE: +0.398631.** I predicted negative, reasoning
from arm D's −0.563179. Arm E is worse than every incumbent (0.727–0.745) and worse than arm B
(0.600) and arm A (0.584), but it is *not* below qwerty on the six-surface frame the way arm D was.
Ceiling re-derivation passed its frozen-iWeb positive control (worst abs diff < 1e-9), reading iWeb
from **my own worktree** with its md5 asserted (`50cab38b…`), not from a sibling's tree (trap 35).

**Is the champion comfort-driven?** 🟢 **Partly, and via the opposite edge from arm D.** The largest
attribution is **`sfs-dist` at 22.48%**, not comfort. But `comfort` (11.16%, a hand-chosen taste
table with no fitted parameter — `DEFAULT_COMFORT`, trap 48) is **pushed past its ceiling to 4.0015
against a domain of [3.2519, 3.8371]** and clamped back to 3.8371 — which is *exactly* where the
archive comfort curve is minimized. ⚠ This is arm D's shape (pinned against the binding edge,
headroom 0.000000) at the **opposite edge**: random400's comfort curve is minimized at its **lo**
edge, the archive one at its **hi** edge. That could not be inherited from arm D's report and had to
be measured. **6 of 14 gauges sit at their own clamped optimum.**

## 6. Plateau census — **zero plateaus again**, on a different fit

🟢 Over the entire final population (40 islands × 64 = 2560 slots, epoch 49):

| quantity | arm E | arm D |
|---|---|---|
| distinct layouts | **1698** | 1730 |
| **distinct objective values** | **1698** | 1730 |
| distinct values per distinct layout | **1.0000** | 1.0000 |
| plateaus (≥2 layouts sharing a value to 12 s.f.) | **0** | 0 |
| champion's exact ties | **0** | 0 |

The clamped archive objective distinguishes **every single layout** in the final population. I did
not assume arm D's result would repeat — the brief explicitly warned against that — and it did,
independently, on a different fit with a different (much smaller) in-domain signal (6.4350 units vs
random400's 48.8093). **The flat-objective hypothesis is refuted twice now, on two fits.** The
search was well-conditioned and confident, and converged to a layout slower than every incumbent.

Also: arm E's top50 holds **50 distinct fitness values, largest plateau 1**, spread 0.024164.

## 7. Admissibility (10-axis dominance frame, with the strict-win term)

Frame is **10** axes (`floor mean wfd genkey oxey1 oxey2 lsb scissor sfb sfs`), predicate
`n_ge == 10 AND n_strict ≥ 1` (trap 33).

| champion | dominator exists | best n_ge |
|---|---|---|
| **arm E** | **no** | **3 / 10** |
| arm A | no | 3 / 10 |
| arm C | no | 3 / 10 |
| arm B | no | 1 / 10 |
| arm D | no | 1 / 10 |

🟢 Arm E is **not admissible**. It ties arm A/arm C at 3/10 — better than arm D's 1/10, consistent
with its better ms/char. **No arm has produced a dominator** — unchanged across all five.

---

## 8. Prediction scorecard — 11 of 16 correct, **5 failed, all reported**

Registered in `artifacts/PREDICTION.md` and **committed at `414f2a6` before the search ran**, so
priority is verifiable rather than asserted.

**Correct (11):** P1 (worse than arm B by > floor ✓ +4.2797 vs 0.4964) · P3 · P4 (E3 on the number)
· P7 (clamp binds, all 14, **0.000e+00**) · P8 (n_ood ≥ 4 ✓ 6) · P9 (ev in [−2.99, −1.4] ✓ −2.6902)
· P10 (zero plateaus, untied, ratio 1.0000) · P11 (wins its ruler vs all 6 incumbents AND loses
ms/char) · P13 (no dominator, n_ge 3 ≤ 4) · P14 (comfort pinned at its hi edge) · P15
(10,017,839 ≥ 9.4M).

**Failed (5), each informative:**

1. **P2 FAIL — predicted [262, 272], point estimate 268.6; actual 258.1803. Wrong by 10.4 ms/char.**
   The most instructive failure, and the one worth carrying forward. My estimate came from
   extrapolating a **42,605-eval** probe (which had reached 268.6092) to a 10M-eval budget.
   **That inference is invalid**: at 0.4% of budget the search has not yet found the in-domain
   interior, so an early champion is not a scaled-down version of the final one. A cheap probe
   bounds an objective's **rank behaviour**; it does not locate the **optimum**. My PREDICTION.md
   called that probe "the single most informative number I have" — it was informative about the
   wrong quantity.
   ⚠ Note the contrast with arm D: ARMD-1's miss came from *not* probing its objective in-band; mine
   came from probing it and over-reading the probe. **Both directions of the same error — treating a
   proxy as the thing.**
2. **P5 FAIL — predicted worse than qwerty (263.7141); arm E is 5.5338 FASTER.** I over-generalized
   arm D's slower-than-qwerty result to a fit whose curves I had already measured as 79.3%
   mechanism-right. My own §1(a) evidence contradicted this prediction and I predicted against it.
3. **P6 FAIL — predicted |arm E − arm D| < 8 ms/char; actual 11.0959.** Same root cause as P2/P5: I
   anchored on arm D.
4. **P12 FAIL — predicted a negative normalized floor; it is +0.398631.** I reasoned from arm D's
   −0.563179; but arm D's floor was negative *because* it was slower than qwerty, which arm E is
   not. The floor tracked ms/char, as it should.
5. **P16 FAIL — predicted ρ ≤ +0.2 in the final population; actual +0.3214.** Inside its own
   converged population the archive objective retains modest positive rank information — weaker
   than the +0.7272 pool-wide figure, stronger than I allowed. Consistent with 72% recovery.

⚠ **The pattern across all five: every failure was in the same direction — I predicted arm E would
be as bad as arm D.** My stated premise ("curve behaviour is a property of the curves, not the fit
pool") was **half wrong**, and it biased four of the five. The pre-run measurement that pointed the
other way (79.3% mechanism-right headroom) was in my PREDICTION.md, flagged as "the honest tension
in my own reasoning" — and it was the better predictor. **I explicitly wrote that if E1/E2 landed,
the mechanism split would be why and I would say so. E3 landed by 1.28, so I say it here: the
mechanism split was the better instrument, and the anti-correlation evidence I favoured over it
predicted the *sign* of arm E's failure but overstated its *magnitude* by ~2.6×.**

---

## 9. What this changes, and what it does not

- 🟢 **Domain coverage is a first-order factor, not a technicality.** Same engine, same seed, same
  budget, same CLAMP, *only the fit pool changed* → **11.0959 ms/char, 22× the resolution floor**.
  Any future claim that "the domains were incidental" is refuted. This also retroactively bounds how
  much of arm D's verdict was about *its* fit rather than about fitted-curve objectives in general:
  **most of it**.
- 🟢 **And it is not sufficient.** The best-posed evidence search available still loses to every
  incumbent by 3.20–3.55 ms/char and to arm B by 4.28. The residual defect is now localized:
  **40.84% of the fitted attribution moves the layout the mechanistically wrong way inside the
  curves' own support**, led by `oxey-style` (12.15%) and `comfort` (11.16%).
- 🟢 **`SEARCH_DOMAIN_POLICY` should still ship, and for a better-supported reason than arm D gave.**
  Arm D showed the clamp is *sound*; arm E shows it is *nearly inert* on a well-fitted pool
  (freezes 0.00% of attribution across the band, `ev_clamp == ev_extrapolate` bitwise for three
  layouts). A policy that costs nothing when the fit is good and bounds an unbounded objective when
  it is not is exactly what you want as a default.
- 🟠 **A caution, inferred:** arm E's champion pushed **5 of 6** out-of-domain gauges in the
  mechanism-*right* direction and was correctly stopped at the edge. So on a well-covered fit, CLAMP
  is now sometimes *blocking real improvement* rather than blocking exploitation. That is the right
  trade (an unsupported region is a guess either way), but it means the clamp's cost is no longer
  zero, and a future arm might reasonably test `REJECT`-with-repair or a widened-support refit
  instead. That would be a **new arm with its own pre-registration**.
- **Nothing here promotes or adopts any layout.** Arm E's champion is a diagnostic object.

## 10. ⚠ A defect in the validated path the parent should know about

**`LossCurve.price_many` (added in `cf5f731`) is NOT bit-exact with `LossCurve.price`, and is not
bit-exact with ITSELF across batch shapes.** 🟢 My gate 1 caught it on its first run.

`price` evaluates a length-1 array; `price_many` on n≥2 evaluates length n, and
`_design(...) @ coeffs` dispatches to a different BLAS kernel by shape — **identical design rows,
different product**. For `comfort` at its `lo`: `0.069389400121559` (n=1) vs
`0.06938940012155903` (n≥2). **7 of the 14 archive curves** show it; arm D's hand-rolled elementwise
`ClampedCurve` shows it on **0 of 14**, because elementwise arithmetic has no shape dispatch.

**Consequences:**
1. ⚠ **The brief's instruction to "pin your fast path against `price_many` at EXACT float equality"
   is unsatisfiable by construction.** No shape-invariant implementation can equal a function that
   is not equal to itself across shapes.
2. **The 4 shipped tests cannot catch it.** `test_price_many_matches_price_exactly_under_every_policy`
   uses one fixed 8-element array on both sides (same BLAS path both times); the saturation test
   uses `pytest.approx`. All 11 tests in `tests/analysis/test_domain_policy.py` pass. Same shape as
   traps 28/31: the check cannot fail in the way that matters.
3. **It is immaterial for arm E, and I proved that rather than assuming it:** over 2061 real +
   random layouts spanning 4.0811 score units, `price_many` and arm D's `ClampedCurve` totals differ
   by at most **1.332e-15** (3.26e-16 of the score range), with **identical argmin and identical
   full argsort ordering**. The search's own accept threshold is `1e-12` — **751× larger**.

**Suggested fix** (verify against current HEAD before relying): compute the linear/quadratic/hinge
form **elementwise** in `price_many`, exactly as `armd_obj.ClampedCurve._raw` does, instead of via
`design @ coeffs`. Then exact equality with `price` becomes attainable, and the tests should assert
it across **≥3 batch shapes** (1, 2, and a large n) rather than one fixed-length array.

**Methodological note on how gate 1 handles this:** ULP is the wrong criterion here. It measures
spacing at the *result's* magnitude, so a value produced by cancellation sits absurdly far away in
ULPs while being an equally-correct rounding — `sr-roll` at level 13.6116 gives
`0.015004423663657684` vs `0.015004423663658572`, an 8.9e-16 absolute difference from terms of order
1, which is **512 ULP**. Gate 1 therefore tests the **dot-product rounding bound**
(`n · eps · Σ|term|`) plus **identical induced ordering**, and includes a **mutation control**: a
knot shifted 1e-6, a domain widened 1%, no clamp at all, and a coefficient perturbed by 1e-12
relative are all caught (by 45× to 9.3e13×), while a **+1 ULP coefficient change is documented as a
necessary blind spot** rather than asserted away. **Measured sensitivity floor: it catches any
relative coefficient error ≥ 1e-13.**

---

## Run integrity (all pre-registered abort conditions met)

| condition | result |
|---|---|
| Gate 1 (policy path validated on the ARCHIVE curves) | 🟢 **113 checks, 0 failures**, `gate1-rc.txt`=0 |
| Gate 2 (engine is arm D's; only the weights differ; resume bit-exact) | 🟢 **28 checks, 0 failures**, `gate2-rc.txt`=0 |
| — positive control, arm A vs frozen `search.py` | 🟢 champion, fitness (`−43.567599284396664`), unique (45181), top50 order all identical |
| — positive control, arm D vs frozen `search_armd.py` | 🟢 champion, fitness (`−22.203251644142952`), unique (46072), top50 order all identical |
| — one variable changed | 🟢 the two workers' 14 gauges are **BITWISE identical** while scores differ by 17.1005 |
| — resume bit-exact (trap 36) | 🟢 champion, fitness AND **unique count** (42605 = 42605) |
| P7 (clamp binds on the champion) | 🟢 all 14 gauges, 50 and 1000 widths, worst **0.000e+00** |
| ≥9.0M unique evals | 🟢 **10,017,839** (arm D 10,099,380; arm A 9,434,590) |
| same seed / islands / overshoot / ga-share / polish-sweeps | 🟢 20260728 / 40 / 1.95 / 0.6 / 40 |
| per-epoch checkpointing (trap 7) | 🟢 49 epochs, budget reached early |
| `.native` frame asserted (not assumed) | 🟢 hard gate in gate 1, both weight sets |
| `analyze` set-containment (trap 38) | 🟢 0 missing; 1 extra `--ref` row, correctly ignored |
| six-surface ceiling positive control | 🟢 PASS, iWeb read from my own worktree with md5 asserted |
| all cited `judgement.json` keys present (trap 19) | 🟢 15 of 15 |
| `tests/analysis` suite green on my branch | 🟢 **300 passed, 1 skipped** of **301 collected** — reconciled against `--collect-only`, not read off a suppressed summary line (trap 2: `addopts = "-q"` makes a second `-q` into `-qq`) |
| shared clone left as found | 🟢 `~/repos/keybo` on `main`, clean. It advanced `bd68f8a`→`f87837b` during my run — verified **not mine**: a fast-forward (`merge-base --is-ancestor` passes), one commit by another agent, touching **no** arm E file. My 2 commits are on `arm-e` only (`git branch --contains` → `arm-e`), not pushed |

## Artifacts (all under `state/arme/artifacts/`)

| file | what |
|---|---|
| `PREDICTION.md` | pre-registration, P1–P16, **committed `414f2a6` before the run** |
| `judgement.json` | 14 layouts, 4 rulers, paired resolution, 19-gauge frame, dominance, clamp-binding, plateau census, in-band rank test, champion drivers + moves |
| `runs/arm-archive.json` + `.ckpt.json` + `.keys.npy` + `.log` | the run, per-epoch checkpointed |
| `gate1-policy.log`/`.json` (rc=0) | 113 checks on the real archive curves + mutation control |
| `gate2-engine.log`/`.json` (rc=0) | positive controls vs BOTH frozen engines, clamp live, resume bit-exact |
| `prerun-arme.json`/`.log` | the pre-run analysis the prediction rests on |
| `report.log`, `report-rc.txt` (=0), `arme-rc.txt` (=0) | judge output + rc sentinels |
| `profiles-and-artifacts-index.md` | durable-location index |
| `drivers/` | `arme_obj.py`, `arme_load.py`, `gate1_policy.py`, `gate2_engine.py`, `prerun_arme.py`, `search_arme.py`, `judge_arme.py`, `report_arme.py`, `run_arme.sh` |

Committed on branch **`arm-e`** in worktree `/tmp/arme` (`414f2a6` = pre-registration + gates;
`29af7d7` = the result + drivers). **Not pushed; no CR** — per scope.

## Reusable traps this arm produced

1. **`LossCurve.price_many` is batch-shape-dependent** (§10) — and a "pin at exact float equality"
   instruction against it is unsatisfiable. Its 4 tests structurally cannot catch it.
2. **ULP is the wrong metric for "same function, different association order".** It measures spacing
   at the result, so cancellation inflates it without bound (512 ULP for an 8.9e-16 difference).
   Use the dot-product rounding bound, and report the induced *ordering*, which is what a search
   consumes.
3. **A rounding-tolerant check has a mathematically necessary blind spot.** Measure and publish its
   **sensitivity floor** (here: relative coefficient error ≥ 1e-13) instead of asserting it catches
   everything.
4. **A cheap early-budget probe bounds an objective's RANK behaviour, not its OPTIMUM's location.**
   Extrapolating a 0.4%-budget champion cost me a 10.4 ms/char prediction miss (P2).
5. **Dump the artifact LAST.** Arm D's driver wrote `judgement.json` mid-function and kept adding
   sections; inherited verbatim, that silently left P14's `champion_drivers` *printed but absent
   from the JSON* (trap 19). Now the dump is last and asserts all 15 cited keys exist.
6. **When a pre-registered outcome band's LABEL overclaims relative to its numeric threshold, honour
   the number and reject the label explicitly** (§4). E3's threshold was met; E3's stated conclusion
   ("regardless of fit pool", "closes the line entirely") is contradicted by the same run's 11.0959
   ms/char pool effect.
