# ARM E — PRE-REGISTRATION

**Written BEFORE the arm-E search is launched.** Corpus **blend-v1** (production default,
`md5(trigrams.txt) = c5066fa7bcc46dea1ecbc987fb465b4a`), frame **`.native`** (asserted in gate 1
from the weights JSON), **90 WPM**. Weights
`state/evidence-scorer/artifacts/arm-archive400-native.json` (`COMMUNITY_BASE`, pool
`archive-400`, n=400), priced under `SEARCH_DOMAIN_POLICY = CLAMP` via the validated
`LossCurve.price_many`. **MODELLED ONLY** — attribution of a fitted timing surface, not measured
typing. **No layout here is promoted or adopted; that is the user's gate alone.**

Frozen comparison set (all blend-v1 @ 90 WPM, all re-derived by me from the frozen artifacts):
arm B **253.9006** · keybo-lsb 254.6307 · keybo-lsb+lm 254.6847 · lsb-sib 254.7058 ·
archive-1846 254.7961 · archive-1843 254.8436 · flagship-c3 254.9761 · arm C 256.0220 ·
arm A 256.8466 · qwerty 263.7141 · qwerty30m 264.1389 · **arm D 269.2762**.

---

## 0. MY PREMISE, STATED EXPLICITLY

Arm D's report records the most instructive failure of the campaign: its estimate and the parent's
**agreed** (255.2–255.4 vs 255.3–256.3) and were **both wrong by ~14 ms/char**, because both rested
on the same false premise — *"a bounded objective must land in the band its data came from."* Two
independent estimates agreeing was not evidence, because the premise was shared. So I state mine.

**My premise: the fitted curves' behaviour ON the near-optimal band is a property of the CURVES,
and the fit pool changes which region is *supported*, not whether the curves point the right way
inside it.** If that is right, arm E fails much as arm D did, because arm E's advantage is entirely
about domain COVERAGE, and coverage is not correctness.

**The premise is falsifiable and I can already state what would break it:** if the archive fit's
in-domain gradients were mechanism-correct where random400's were not, arm E would land in the
incumbent band and E1 would hold. So I measured the mechanism split before predicting (§1), rather
than reasoning from the domain-coverage fact alone. **That measurement is what separates my
estimate from arm D's failed one: arm D's ±14 ms/char miss came from predicting without measuring
its objective's in-band rank behaviour; I measured mine, and it is negative.**

⚠ **The counterargument I carry, per the brief:** EVIDENCE-SCORER-1 **rejected** these archive
weights as a *scorer* (0 of 12 cross-source cells, inside the noise band). Arm E asks the set that
failed as a RANKER to serve as a SEARCH objective, on the grounds that its domains cover the band.
Those are genuinely different tests and can diverge. **A good arm-E result would NOT rehabilitate
the scorer**, and I will not report it as doing so.

---

## 1. WHAT I MEASURED BEFORE PREDICTING (all 🟢, `prerun-arme.json`, `gate1-policy.json`)

**The premise the brief gave me is TRUE and I reproduced every number of it:** keybo-lsb is
out-of-domain on **0 of 14** gauges under ARCHIVE weights vs **9 of 14** under random400;
flagship-c3 **1 of 14** vs **9 of 14**; `comfort` ARCHIVE [3.2519, 3.8371] vs random400
[6.5236, 11.5644]; `sr-roll` ARCHIVE [11.3865, 18.0617] vs random400 [1.9997, 8.3369]. I can
sharpen it: **6 of the 14 domains are FULLY DISJOINT** between the two fits (`sfb`, `sfb-dist`,
`alt`, `sr-roll`, `oxey-style`, `comfort`). The archive fit is also the better *fit*: holdout
R² **0.7591 vs 0.4286**, effective dof 3.99 vs 5.03.

**And the clamp is nearly inert in the band, exactly as the brief predicted:** CLAMP freezes
**0.00%** of the fitted attribution into a constant across the 10-layout band (under random400 it
froze **82.55%**), with **0 exact ties**. arm B, keybo-lsb and keybo-lsb+lm have
`ev_clamp == ev_extrapolate` to the bit. So arm E is a *genuine* test of the curves in a way arm D
structurally could not be. **The arm is better-posed. That is established, and it is independent of
how the arm turns out.**

Then the three things that actually drive my prediction:

**(a) The in-domain headroom is mostly mechanism-RIGHT — the opposite of arm D.** From keybo-lsb,
of 1.7911 units of remaining clamped headroom, **79.3% is mechanism-RIGHT** (collectable by making
the layout genuinely faster) and only 20.7% WRONG. Arm D's figure was **92.5% WRONG**. Read off the
piecewise coefficients and knots, not the linearized weights (trap 53). **This is the strongest
pro-E signal I have, and it is why E3 is not a foregone conclusion.**

**(b) But the objective mis-ranks the band it is supposed to search.** This is the fact that
dominates (a):

| pool | n | ρ(ev_CLAMP, ms/char) — want **+1** |
|---|---|---|
| the 10-layout board | 10 | **−0.1758** |
| the six incumbents only | 6 | **−0.6000** |
| 3600 1–4-swap perturbations of the incumbents, all | 3600 | +0.7272 |
| …band ≤257.0 | 1010 | +0.4195 |
| …band ≤256.0 | 590 | +0.2568 |
| …band ≤255.5 | 305 | +0.1609 |
| …band ≤255.0 | 104 | **+0.0580** |

The same monotone decay OPTEVIDENCE-1 and arm D found for random400 reproduces for the ARCHIVE fit
on an independently built pool: informative wide, ~zero (and negative among the incumbents) exactly
where a search operates. Concretely: **the archive objective scores arm B — the FASTEST layout
known, 253.9006 — at −0.2324, nearly the WORST in the band**, while ranking `lsb-sib` (254.7058)
and `graphite` (258.1696, slow) best at −1.8638/−1.8973. In the 3600-layout pool the best-ev layout
is 255.69 ms/char while the fastest layout (254.38) scores only −0.9143 against a −2.1164 minimum.

**(c) A 40k-eval arm-E run already exists, and it is bad.** Gate 2's resume test ran arm E for
42,605 unique evals (0.4% of the real budget) and its champion is
`qjvgu-wz'xtiordphenasby.,mfklc` at **268.6092 ms/char, ev_clamp −1.4335, n_ood 8/14** — already
within 0.67 of arm D's 269.2762, on 1/237th of the budget. 🟢 This is the single most informative
number I have, and I did not have to spend the budget to get it. **It is also the check that keeps
me from repeating arm D's error**: arm D predicted without a cheap in-band probe of its own
objective. More budget optimizes the objective harder, so if the objective is anti-correlated in
the band, more budget makes ms/char *worse*, not better.

---

## 2. PRE-SPECIFIED OUTCOMES — I expect **E3**

- **(E1)** arm E ≤ **254.63** (inside the incumbent band) ⇒ the random400 DOMAINS were the whole
  problem, and ARMD-1's "mis-specified where supported" narrows to "mis-specified for that fit".
- **(E2)** arm E in **254.6–256.9** (between the incumbents and arm A) ⇒ partial: domains mattered,
  the curves are still weak.
- **(E3)** arm E ≥ **256.9**, or worse than qwerty (263.7141) ⇒ the CURVES are the defect
  regardless of fit pool, which CONFIRMS ARMD-1 on the strongest possible evidence and closes the
  evidence-weight line.

**I expect E3, and specifically the worse-than-qwerty form of it.** 🟠 INFERRED, from §1(b)+(c):
the objective is anti-informative among the incumbents (ρ = −0.60), it prefers arm B *last* among
fast layouts, and a 0.4%-budget probe already sits at 268.61.

⚠ **The honest tension in my own reasoning:** §1(a) says 79.3% of the headroom is
mechanism-RIGHT, which argues for E1/E2. I am predicting E3 anyway, because a *separable sum of
14 curves* is not steered by the aggregate mechanism split — the search collects whichever units
are cheapest, and `comfort` alone (11.16% of attribution, 0.9233 of arm B's 2.7490 headroom = 34%,
mechanism-WRONG, wants to go UP) plus `sfs-dist` (22.48% attribution) can dominate the direction.
**If arm E lands in E1/E2, the mechanism split is why, and I will say that (a) was the better
predictor than (b).**

## 3. NUMBERED PREDICTIONS

| # | prediction | basis |
|---|---|---|
| **P1** | arm E is **worse (slower) than arm B's 253.9006** by more than the paired resolution floor | §1(b) ρ<0 in-band |
| **P2** | arm E lands in **[262, 272] ms/char**; point estimate **268.6** | §1(c), the 42,605-eval probe |
| **P3** | arm E is **worse than the best incumbent** keybo-lsb 254.6307 | §1(b) |
| **P4** | **E3 holds** (≥256.9) | §1(b)+(c) |
| **P5** | arm E is **worse than qwerty** (263.7141) | §1(c) probe already 268.61 |
| **P6** | arm E is **not materially better than arm D** (269.2762): |Δ| < 8 ms/char | §1(c) within 0.67 at 0.4% budget |
| **P7** | **the clamp BINDS exactly**: worst \|reward N domain-widths outside\| = **0.000e+00** on all 14 gauges. **ABORT if not** | gate 2 check 5 |
| **P8** | champion `n_ood` is **≥ 4 of 14** — i.e. the search LEAVES the domains the incumbents sit inside | 40k probe was 8/14 |
| **P9** | `ev_clamp` of the champion lands in **[−2.99, −1.4]** (−2.9814 is the unattainable per-gauge floor; the 40k probe already hit −1.4335) | §1, `clamped_lower_bound` |
| **P10** | **plateau census: ZERO plateaus**, champion untied, distinct-values/distinct-layouts = 1.0000 | 0 ties in the band; arm D found 1730/1730 |
| **P11** | arm E **wins the clamped-archive ruler** it trained on against every incumbent while **losing on ms/char** — optimizing the ruler | arm D's §5 shape |
| **P12** | normalized floor is **NEGATIVE** | arm D's −0.563179 with a slower-than-qwerty champion |
| **P13** | **no dominator** on the 10-axis frame; champion best `n_ge` **≤ 4 of 10** | no arm has produced one |
| **P14** | the champion **IS comfort-driven**: `comfort` pinned at/near its ceiling **3.8371** (note: the archive curve is minimized at its **hi** edge, the opposite of random400's **lo**) | §1(a), 34% of arm B's headroom |
| **P15** | ≥ **9.4M** unique evals, per-epoch checkpointed | budget parity with arms A/B/D |
| **P16** | the in-band ρ decay reproduces on the FINAL population: ρ(ev, ms) ≤ +0.2 among the champion's near-ties | §1(b) |

## 4. WHAT WOULD FALSIFY MY PREMISE, AND WHAT I WILL REPORT REGARDLESS

- If **E1** holds, my premise ("curve behaviour is a property of the curves, not the fit pool") is
  **refuted**, ARMD-1's diagnosis narrows to "mis-specified for THAT fit", and the
  domain-coverage argument was the whole story. I will report that plainly.
- If **E2** holds, both factors are real and neither §1(a) nor §1(b) alone was sufficient.
- If **E3** holds, ARMD-1 is confirmed on the strongest available evidence: the curves are the
  defect regardless of fit pool, **and** — because this fit's domains genuinely cover the band and
  its clamp freezes 0.00% of the attribution — no "the domains were wrong" escape remains.

**A NEGATIVE RESULT IS FULLY ACCEPTABLE and will not be tuned away.** Any change to the objective
after this point is a NEW arm with its own pre-registration. I will report all failed predictions
explicitly, as ARMD-1 did with its 5 of 16.

## 5. Abort conditions (a run that trips these is not reported as a result)

1. **P7 fails** — the clamp does not bind at exactly 0.000e+00 through arm E's own objective.
2. Gate 1 fails (policy path not validated on the archive curves) — **PASSED: 113 checks, 0 fail**.
3. Gate 2 fails (engine not arm D's, or resume not bit-exact) — arm A AND arm D positive controls
   must both reproduce the frozen champions exactly.
4. Fewer than 9.0M unique evals, or checkpointing absent.
5. `analyze --json` set-containment fails on the requested layouts (trap 38 — **set** containment,
   not row count; `--ref` legitimately adds a row).
