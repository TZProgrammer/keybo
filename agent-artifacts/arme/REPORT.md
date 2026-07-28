# ARM E — the evidence-weight search on the ARCHIVE-fitted curves, under CLAMP

> ## 🔴 POST-HOC SELF-AUDIT (2026-07-28) — FOUR CORRECTIONS, ONE OF WHICH CHANGES A VERDICT
> Registered by the parent as **ARME-1 (`571bfe9`)**, then re-audited by me as a skeptic. **Read
> §11 before citing ANY per-pair Δ, §3, §4, §6 or §9.** The 258.1803 measurement stands; four
> *attributions* do not.
>
> ### 🔴 0. THE BIGGEST ONE — I RAN A SECOND SEED AND ARM E'S SEARCH NOISE IS ~9.43 ms/char
> I had listed "arm E's own search-noise sd is unmeasured" as a gap. I then measured it. Seed
> **20260729**, everything else identical:
>
> | | ev_clamp | **ms/char** | n_ood | unique evals |
> |---|---|---|---|---|
> | seed 20260728 (reported) | −2.690226 | **258.1803** | 6/14 | 10,017,839 |
> | seed 20260729 (new) | −2.677732 | **267.6096** | 9/14 | 10,084,782 |
>
> The two seeds reach objectives **0.46% apart** and layouts **9.4293 ms/char apart**, sharing only
> **2 of 30** key positions. **So every per-pair Δ in §1 is quoted against the wrong ruler.** My
> "resolves at 8.62× the paired floor" used the *timing* floor (0.4964), which measures seed-table
> noise in the ms/char model — **not** run-to-run variation of the search itself. Against the
> ~9.43 spread: arm E vs arm B (+4.2797) is **0.5×**, vs keybo-lsb (+3.5495) **0.4×**, vs arm A
> (+1.3337) **0.1×**. **Only arm E vs arm D (−11.0959, 1.2×) still clears it, and barely.**
>
> **Verdict changes:** (i) "arm E is slower than every incumbent" is **no longer established** —
> it is one draw from a distribution ~9 ms/char wide; **the ARM-LEVEL conclusion (an evidence
> objective does not beat arm B) survives, because both seeds land far above 253.9006, but the
> specific 258.1803-vs-incumbent gaps do not.** (ii) My §6 plateau census, which I presented as
> reassuring ("the objective distinguishes every layout, so the search was well-conditioned and
> confident"), **was the wrong reassurance** — see 11.7.
>
> ⚠ And it means **arm D vs arm E — the 11.0959 that my whole "72%" rests on — is only ~1.2× the
> search spread.** The direction is probably real (both arm E seeds beat arm D's 269.2762, though
> seed 2 by only 1.67) but **"72% recovered" needs n≥3 seeds per arm before it is quotable.**
>
> ### The other three
> 1. **"Domain coverage is worth 11.0959 ms/char" is a BUNDLED attribution, not an isolated one.**
>    Arms D and E differ in **coeffs 14/14, domain 14/14, knot 13/14, form 2/14** simultaneously.
>    Worse for my headline: the archive curves are **NOT better-shaped** — 8/14 mechanism-correct
>    minima vs random400's **9/14**, and **42.5%** of their collectable units are mis-signed vs
>    random400's **17.5%**. What plausibly did the work is **SCALE**: total in-domain signal
>    **6.4349 vs 48.8090 units (7.6× smaller)**, so a *larger* mis-signed fraction buys *less*
>    damage. "Domain coverage is first-order" should read **"the fit pool is worth 11.0959 ms/char,
>    bundled; the leading candidate mechanism is objective SCALE, not coverage."**
> 2. **My in-band ρ = +0.0580 at ≤255.0 is INDISTINGUISHABLE FROM ZERO** — bootstrap CI
>    **[−0.1473, +0.2604]**, p=0.558, n=104. I reported it as a point estimate. The *decay* is
>    real (all/≤257/≤256/≤255.5 all exclude zero); the *tightest cell's value* is not a
>    measurement.
> 3. **My P2 lesson mis-credited the probe.** I wrote "a cheap probe bounds RANK behaviour, not the
>    OPTIMUM's location." It bounded **nothing** — ms/char no, objective value no, rank no (the rank
>    evidence came from a *separate* 3600-layout pool). The real error was **using an unconverged
>    run as a point estimate**, and it was cheaply detectable: epoch 1 of the real run already had
>    368,209 unique evals at best −2.204979, far past my probe's −1.4335.
>
> **Survived unchanged:** the mechanism-right classification is **NOT circular** (`EXPECTED_SIGN`
> is a hardcoded prior table, but it independently agrees with the served surface on **14/14
> in-band**, 13/14 on a wide random pool); the clamp binding (0.000e+00); the dominance frame; and
> the E3-label rejection. **E3 itself survives on both seeds** (258.1803 and 267.6096 are both
> >= 256.9).
>
> Also: my "`price_many` affects 7 of 14 curves" **understated** it. The parent found 9/14 and
> fixed it in `79cb175`; re-probing on a *grid* of levels rather than one arbitrary level shows it
> is **14 of 14** — the defect is per-**level**, not per-curve. 🟢 **And I verified the fix does not
> move this arm:** the frozen champion re-scores to **−2.690225544692558**, bit-identical, ms/char
> 258.1803 unchanged, ordering and argmin preserved.

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

### Does the gap resolve? 🔴 **AGAINST THE WRONG RULER — see §11.7.**
The table below uses the **paired timing floor**, which measures the ms/char model's seed-table
noise. It does NOT measure run-to-run variation of the SEARCH, which a second seed shows is
**~9.43 ms/char**. Against that, only arm E vs arm D still clears (1.2×); vs arm B / keybo-lsb /
arm A are 0.5× / 0.4× / 0.1× and **do not resolve**. Kept as published, with the correction.

Original text — all 10 arm-E pairs, by 2.69× to 22.35× *(paired timing floor)*:
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
> 🔴 **REFRAMED BY §11.7 — read that first.** "Zero plateaus" answers *"can the objective
> distinguish these layouts?"*, which is not the question that matters. Within 0.02 objective units
> — less than the gap between two seeds' champions — sit layouts spanning **12.24 ms/char**. The
> objective is non-degenerate in its own units and **near-degenerate with respect to speed**, which
> is the pathology rather than evidence against it.

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
4. ⚠ **SUPERSEDED by §11.4 — the correct form is stronger.** *(Original: "a cheap probe bounds an
   objective's RANK behaviour, not its OPTIMUM's location.")* The probe bounded **nothing**. The
   real rule: **never use an unconverged run as a point estimate**, and check convergence by
   whether best-fitness has stopped improving — not by budget fraction.
5. **Dump the artifact LAST.** Arm D's driver wrote `judgement.json` mid-function and kept adding
   sections; inherited verbatim, that silently left P14's `champion_drivers` *printed but absent
   from the JSON* (trap 19). Now the dump is last and asserts all 15 cited keys exist.
6. **When a pre-registered outcome band's LABEL overclaims relative to its numeric threshold, honour
   the number and reject the label explicitly** (§4). E3's threshold was met; E3's stated conclusion
   ("regardless of fit pool", "closes the line entirely") is contradicted by the same run's 11.0959
   ms/char pool effect.

---

# 11. POST-HOC SELF-AUDIT (2026-07-28)

Run as a skeptic against my own load-bearing claims, after the parent registered this arm as
**ARME-1 (`571bfe9`)** and accepted the E3-label rejection. Everything below is freshly computed on
the same corpus/frame; nothing is re-quoted from the original pass.

## 11.1 The 72% recovery is a **BUNDLED** attribution — and my headline named the wrong mechanism

**Verdict: the audit is right, and it is worse for my headline than the audit supposed.** 🟢

Arms D and E differ by more than "the fit pool". Measured, per gauge:

| what differs between the two weight sets | count |
|---|---|
| coefficients | **14 / 14** |
| valid_domain | **14 / 14** |
| knot | **13 / 14** |
| functional form | 2 / 14 (`sr-roll` hinge→quadratic, `scissor` linear→quadratic) |

So **"the fit pool is worth 11.0959 ms/char" is a bundled effect** of {domain coverage} × {curve
shape} × {coefficient magnitudes and signs} × {knot placement}. It is **not decomposable from this
arm**, and I should not have written "domain coverage is FIRST-ORDER" as though coverage were the
isolated factor.

**What would separate them** — and the two probes I *can* run without a new search:

**(a) Are the archive curves simply better-*shaped*? NO — they are marginally WORSE.** Reading the
piecewise coefficients and knot (trap 53), and asking whether each curve's in-domain minimum sits at
the end `EXPECTED_SIGN` says is *faster*:

| | mechanism-correct minima | mis-signed collectable units | mechanism-RIGHT units | **mis-signed share** |
|---|---|---|---|---|
| random400 (arm D) | **9 / 14** | 8.5623 | 40.2467 | **17.5%** |
| archive (arm E) | **8 / 14** | 2.7333 | 3.7016 | **42.5%** |

The archive fit is *proportionally more* mis-signed. "Better-shaped curves" is **refuted** as the
explanation. 🟢

**(b) The leading surviving candidate is objective SCALE.** Total in-domain signal (sum of per-gauge
ranges): **random400 48.8090 units vs archive 6.4349 — 7.6× smaller.** `comfort` alone collapses
from a 24.8168-unit range (43.55% of attribution) to 0.9456 (11.16%), and `alt` from 5.2732 to
0.1795. A separable sum of mis-signed curves does damage in proportion to *how many units the
mis-signed terms can pay*, so a 7.6× smaller objective with a *larger* wrong fraction still does far
less damage. 🟠 INFERRED — consistent with everything measured, not isolated.

**(c) The one factor I could vary independently — the POLICY, curves held fixed** — corroborates
that coverage is real but secondary. Over the same 3600-layout band:

| curves | mean \|clamp − extrapolate\| | max | mean n_ood |
|---|---|---|---|
| random400 | **12.6910** | 22.8766 | 6.57 / 14 |
| archive | **1.6381** | 35.0288 | 5.73 / 14 |

Coverage *is* materially better under the archive fit (the clamp has ~8× less work to do on average).
But mean n_ood only falls 6.57 → 5.73, which is a modest coverage gain for an 11.10 ms/char outcome
difference — again pointing at scale rather than coverage as the dominant term.

⚠ **And scale cannot be separated from coverage either**, because a narrower fit pool mechanically
produces *both* narrower domains *and* smaller ranges. **Isolating them needs an arm that holds one
fixed** — e.g. refit on the archive pool but rescale the curves to random400's total signal, or fit
both pools with a common domain. That is a new arm, not a re-reading of this one.

**Corrected headline** (please cite this form): **the fit pool is worth 11.0959 ms/char (22× the
resolution floor) as a bundled effect, and is not sufficient; the leading candidate mechanism is
objective SCALE rather than domain coverage, and this arm cannot decompose the two.** The
*negative* half of the finding — still +3.5495 behind the worst incumbent — is unaffected, since it
rests on the measured ms/char alone.

## 11.2 The tightest in-band ρ cell is indistinguishable from zero

**Verdict: the audit is right.** 🟢 I reported bare point estimates. Bootstrap, 2000 resamples,
same n=3600 pool:

| band | n | ρ | 95% CI | excludes 0? | p |
|---|---|---|---|---|---|
| all | 3600 | +0.7272 | [+0.7103, +0.7437] | ✅ yes | ~0 |
| ≤257.0 | 1010 | +0.4195 | [+0.3676, +0.4696] | ✅ yes | 2.5e−44 |
| ≤256.0 | 590 | +0.2568 | [+0.1767, +0.3321] | ✅ yes | 2.4e−10 |
| ≤255.5 | 305 | +0.1609 | [+0.0450, +0.2755] | ✅ yes | 0.0049 |
| **≤255.0** | **104** | **+0.0580** | **[−0.1473, +0.2604]** | **❌ NO** | **0.558** |

**So "+0.0580" must be reported as "indistinguishable from zero at n=104"**, not as a value. Note
this does **not** weaken the conclusion — it arguably strengthens it: "the objective carries no
detectable rank information in the tightest band" is the claim I wanted, and it is what the CI
supports. What is *not* supported is any comparison of that cell's magnitude against the sibling's
−0.0455.

**On comparability with optevidence's 36,005-perturbation figures (+0.9111 → −0.0455):** those are
for the **RAW/extrapolating random400 objective**; my +0.7272 → +0.0580 is for the **CLAMPED archive
objective**. **Different objective, different fit, different pool size — they are not the same
measurement** and I should not have implied a like-for-like decay comparison. The comparable column
in my own table is `r400 extrap` (+0.9017 → +0.0809), which *does* track the sibling's shape
closely — that agreement is the real corroboration, and at n=104 my cell cannot distinguish
+0.0809 from their −0.0455 either.

## 11.3 The mechanism-right classification is **NOT circular** — it survives

**Verdict: the audit's concern is legitimate but the classification holds.** 🟢

`EXPECTED_SIGN` (`evidence_scorer.py:121–136`) is a **hardcoded, hand-authored prior table** — not
derived from the timing surface — so the concern is well-posed: if the sign convention were merely
asserted, "5 of 6 pushed mechanism-right" would be unfalsifiable.

**But it is independently testable against the served surface, and it passes.** Rank correlation of
each raw gauge against predicted ms/char, on pools chosen by no objective:

| pool | n | EXPECTED_SIGN agrees with the surface |
|---|---|---|
| random C30M permutations | 4000 | **13 / 14** (only `sfs` disagrees, ρ = −0.0218 ≈ 0) |
| **in-band, ≤257.0** | 1010 | **14 / 14** |

In the band that matters the convention is confirmed on **every gauge** (e.g. `sfb` +0.5608,
`sfb-dist` +0.5821, `comfort` +0.6162, `alt` −0.3084, `sr-roll` −0.3525). The single wide-pool
disagreement is `sfs` at ρ = −0.0218, itself indistinguishable from zero.

So the inverse-signature claim (arm D's 3 out-of-domain gauges on axes it was *worsening*; arm E's
6 with 5 on axes it was *improving*) is **falsifiable and not refuted**. 🟢 Caveat kept: it is
verified against the *fitted served surface*, which is a model — so this is "consistent with the
timing model", not with measured typing.

## 11.4 My P2 lesson mis-credited the probe — the honest lesson is stronger

**Verdict: the audit is right; the probe bounded nothing.** 🟢

| | unique evals | ev_clamp | ms/char | n_ood |
|---|---|---|---|---|
| the probe | 42,605 | **−1.4335** | 268.6092 | 8/14 |
| the real champion | 10,017,839 | **−2.6902** | 258.1803 | 6/14 |

The probe had collected only **53%** of the objective the search eventually reached. Testing my own
stated lesson against it:

- **ms/char? Not bounded.** 268.6092 vs 258.1803 — off by 10.4 in the direction that *overstated*
  badness, so not a usable bound either way.
- **Objective value? Not bounded.** The search improved past it by 88%.
- **Rank behaviour? Not bounded — and this is where my lesson was wrong.** The probe was a *single
  layout*; it carries no rank information at all. My in-band ρ evidence came from a **separate
  3600-perturbation pool**. My lesson credited the probe with a property that belonged to a
  different artifact.

**Corrected lesson:** *never use an unconverged run as a point estimate — and diagnose convergence
by whether best-fitness has stopped improving, not by budget fraction.* 🟢 **It was cheaply
detectable in advance:** the real run's own epoch trace shows **epoch 1 alone reached 368,209 unique
evals at best −2.204979** — 8.7× my probe's budget and already far past its −1.4335 — with
improvement continuing to epoch 47. A single scaled epoch would have refuted the 268.6 estimate
before I registered it.

## 11.5 Why I recommend AGAINST arm F, in one sentence

**Because arm F (refitting on a pool that covers the band's good side) widens the support a
maximizer can exploit while leaving the sign errors in place — and this arm measured the archive
fit as *proportionally more* mis-signed than random400 (42.5% vs 17.5% of collectable units), so the
one mechanism that plausibly produced arm E's improvement (a 7.6× smaller objective) is exactly what
a wider, better-covering refit would undo.** 🟠 INFERRED — and if the parent is asked to run it
anyway, it should pre-register the *scale* control (hold total in-domain signal fixed), because
without it arm F reproduces arm E's non-identifiability at greater cost.

## 11.6 What I did NOT test — for a reviewer

1. **`price_many` count.** My "7 of 14" was **probe-dependent** — one arbitrary level per curve. On
   a 101-point in-domain grid it is **14 of 14**; the defect is per-**level**, not per-curve. The
   parent's 9/14 is likewise probe-dependent (a different sample), and the parent's `79cb175`
   elementwise fix is the correct response either way. **I did not re-run arm E against the fixed
   `price_many`** — my worktree is at `cf5f731`+3, pre-fix. Expected impact nil (the deviation was
   1.3e-15 against a 1e-12 accept threshold, identical argmin and ordering over 2061 layouts), but
   **that is an argument, not a measurement.** A cheap confirmation would be re-scoring the frozen
   champion under the fixed code and asserting the same 258.1803.
2. ✅ **CLOSED, and it was the most consequential gap — see §11.7.** I ran the second seed. Arm E's
   search spread is **~9.43 ms/char**, not the ~0.34 I guessed from OPTEVIDENCE-1, so **+4.2797 vs
   arm B is 0.5× the spread, not ~12σ.** My instinct that this was a minor loose end was wrong;
   it invalidated every per-pair Δ's precision.
3. **The 5-of-6-mechanism-right count is against `keybo-lsb` only.** I did not check whether the
   inverse-signature claim holds against a different reference incumbent.
4. **`normfloor +0.398631` uses corpus-matched ceilings derived from a 46-layout reference
   population** whose *inputs* live in another workspace's state dir (trap 14). The iWeb positive
   control passed (< 1e-9), but the blend-v1 ceilings themselves have no independent control.
5. **The plateau census reads the epoch-49 checkpoint**, i.e. the population the search *ended*
   with. I did not verify that no earlier epoch had plateaus that later resolved.
6. **Two `qwerty`s.** 263.7141 (classic `;./`) vs 264.1389 (`qwerty30m`). P5 fails on both, but any
   future comparison quoting "qwerty" from this arm should say which.

## 11.7 🔴 THE SECOND SEED — arm E's search noise is ~9.43 ms/char, and it reframes §6

**Verdict: this closes gap 11.6#2 and changes more than the gap did.** 🟢

I flagged "arm E's own search-noise sd is unmeasured" as a reviewer item, then measured it. Seed
**20260729**, every other parameter identical (40 islands, 55-epoch cap, overshoot 1.95, ga-share
0.6, polish-sweeps 40, budget 10M):

| | champion | ev_clamp | **ms/char** | n_ood | unique evals | epochs |
|---|---|---|---|---|---|---|
| seed 20260728 *(reported)* | `ou-qdbpmlsaiehvgctnr.,y'kfwjzx` | −2.690226 | **258.1803** | 6/14 | 10,017,839 | 49 |
| **seed 20260729** *(new)* | `,qkbw'juzxastgphnieromdfc.v-yl` | −2.677732 | **267.6096** | 9/14 | 10,084,782 | 50 |

**Objectives 0.46% apart; layouts 9.4293 ms/char apart; 2 of 30 shared key positions.**
Sentinel `seed2-rc.txt` = 0.

### Every per-pair Δ in §1 was quoted against the wrong ruler
The 0.4964 "conservative paired floor" is the **timing model's** seed-table noise — it does **not**
measure run-to-run variation of the *search*. Re-reading §1's claims against the ~9.43 search spread:

| pair | Δ ms/char | × paired timing floor *(what I reported)* | **× search spread** | still resolves? |
|---|---|---|---|---|
| arm E vs arm D | −11.0959 | 22.35× | **1.2×** | 🟡 marginally |
| arm E vs arm B | +4.2797 | 8.62× | **0.5×** | 🔴 **no** |
| arm E vs keybo-lsb | +3.5495 | 7.15× | **0.4×** | 🔴 **no** |
| arm E vs arm A | +1.3337 | 2.69× | **0.1×** | 🔴 **no** |

⚠ n=2 is a **spread, not an sd worth a CI** — I am not claiming 9.4293 is *the* sd. But it bounds the
order of magnitude, and it is **27× larger than OPTEVIDENCE-1's 0.3440** evidence-arm figure, so that
number cannot be borrowed for this arm either.

**What survives:** the **arm-level** conclusion. Both seeds land far above arm B's 253.9006 (+4.28
and +13.71) and above every incumbent, and **both satisfy E3** (≥256.9). So *"an evidence-weight
objective, even the best-posed one, does not produce a layout competitive with arm B"* is
**unaffected** — it is the *specific gap sizes* and the *72% recovery* that lose their precision.
**"72% recovered" needs n≥3 seeds per arm before it is quotable.**

### And it reframes §6's plateau census — I gave the wrong reassurance
§6 reported 1698/1698 distinct objective values, zero plateaus, and I framed that as *"the search
was well-conditioned and confident"*. That is true **of the objective** and it is the wrong thing to
reassure a reader about. Pooling both final populations (n=5120) and asking how much **ms/char**
varies among layouts the objective cannot meaningfully separate:

| layouts within … of the best ev | n | ms/char min | max | **ms/char SPREAD** |
|---|---|---|---|---|
| 0.001 units | 40 | 258.1803 | 258.1803 | 0.0000 |
| 0.005 units | 120 | 258.0502 | 258.1803 | 0.1301 |
| **0.010 units** | 204 | 257.9352 | 262.0906 | **4.1554** |
| 0.020 units | 1005 | 257.7526 | 269.9879 | **12.2353** |
| 0.050 units | 2274 | 257.4419 | 272.8030 | **15.3611** |

**The objective is not flat in ITSELF — it is nearly flat in the quantity we care about.** Within
0.02 objective units (smaller than the gap between the two seeds' champions) sit layouts spanning
**12.24 ms/char**. Distinguishing every layout to 12 significant figures while those layouts differ
by ~10 ms/char **is the pathology, not evidence against it** — and it is the direct cause of 11.7's
seed spread. My §6 conclusion ("the flat-objective hypothesis is refuted") should read: **the
objective is non-degenerate in its own units and near-degenerate with respect to speed, so which
champion a run returns is close to arbitrary w.r.t. ms/char.** That is a *sharper* version of the
sibling's original warning, not a refutation of it. 🟢

**Reusable trap:** *"zero plateaus" answers "can the objective distinguish these layouts?", which is
not the question. Ask instead how much the TARGET quantity varies inside a band of the objective the
search cannot resolve — and confirm it with a second seed, which is the cheapest test of whether a
champion is reproducible at all.* Two seeds cost ~9 minutes each here and would have changed how I
wrote §1 and §6.
