# ARM D — PRE-REGISTERED PREDICTION

**Written 2026-07-28, BEFORE the arm D search was launched.** Evidence available at prediction
time: gate-1 verification (`gate1-verify-policy.log`, 1301 checks / 0 failures),
`pre-run-analysis.json`, `headroom.json`. No arm D evaluation had been run.

Corpus: **blend-v1 (production default)**, `/tmp/domainfix/data/corpus/blend-v1`. Frame:
**`.native`** (asserted from the weights JSON: `surface_frame == "native"`). Weights:
`state/evidence-scorer/artifacts/arm-random400-native.json` (source `COMMUNITY_BASE`, pool
`random-c30m-400`, n=400). All ms/char at 90 WPM. **MODELLED ONLY.**

---

## The headline prediction: **outcome (ii), with a large dose of (iii)**

> **(ii)** arm D is still well behind arm B => the weights really are uninformative in the
> near-optimal band, and the extrapolation was a separate, additive defect. CONFIRMS
> OPTEVIDENCE-1 on stronger evidence.

I predict arm D lands **well behind arm B** (not (i)), and that the *mechanism* by which it does
so is substantially the (iii) mechanism — a clamped objective that is nearly flat in the
near-optimal band, so the search is close to unguided there. I expect the two to be **jointly
observable**, which the parent's three-way framing treats as exclusive: (iii) is not a separate
fate from (ii) but the *reason* for it, and I expect to report "(ii) via (iii)".

**Why, from the pre-run evidence — this is not a guess:**

1. **The clamp freezes 82.55% of the fitted attribution into a constant across the near-optimal
   band.** Of the 14 gauges, 8 have every band layout's level clamped to the *same* domain edge:
   `sfb`, `sfb-dist`, `sfs-dist`, `alt`, `sr-roll`, `redir`, `scissor`, `comfort` — including
   `comfort` at **43.55%** of attribution, pinned at its floor 6.5236 for all 9 band layouts.
   A term that is constant over the feasible band contributes **zero** to ranking within it.
2. **The band's clamped spread collapses to 0.094x the extrapolating one** (0.7474 vs 7.9745
   units) against a total in-domain signal of 48.8093. The objective the search must climb in
   the region where champions live is ~1/11th as tall as arm A's was.
3. **The clamped objective is already NEGATIVELY rank-correlated with speed in the band.**
   `spearman(ev_clamp, ms/char)` over the 9-layout near-optimal band = **−0.4435** (want +1);
   EXTRAPOLATE was −0.5833. Clamping *improves* it by 0.14 but leaves it the **wrong sign**. So
   the clamp fixes unboundedness without making the ruler point at speed. Contrast the 400
   random permutations the weights were FITTED on: **+0.7093** — trap 52 exactly (in-sample on
   the wide pool, useless in the band of use).
4. **92.5% of the headroom the clamped search has left is mechanism-WRONG.** From arm A's
   champion, remaining clamped headroom is 7.3897 units, of which **6.8331 (92.5%)** is only
   collectable by moving a gauge in the direction that makes the layout *slower* (`scissor` up
   2.2951, `sfb-dist` up 1.8365, `sfb` up 1.4535, `oxey-style` up 0.7270, `lsb-dist` up 0.5210),
   and only **0.5567 (7.5%)** is mechanism-right. A minimizer chasing what is left is being paid
   to get worse. **This is the load-bearing prediction** — it is why I do not expect (i).
5. **The clamp already produces an exact degeneracy in the band.** `keybo-lsb` and
   `keybo-lsb+lm` are *different layouts* whose clamped scores are **bit-identical**
   (−17.89387841849961, |Δ| = 0.000e+00) while their extrapolating scores differ by 0.8472 and
   their ms/char by 0.0539. A plateau exists at 9 layouts; a 9.4M-eval search will find vastly
   larger ones.

## Falsifiable numeric predictions

| # | Prediction | Resolves if |
|---|---|---|
| P1 | Arm D's ms/char is **worse (higher) than arm B's 253.9006** by **> 0.2222** (the conservative paired resolution) | measured gap vs arm B |
| P2 | Arm D's ms/char lands in **[254.5, 257.5]**, i.e. better than arm A's 256.8466 but not reaching arm B | measured |
| P3 | Arm D **beats arm A** (256.8466) on ms/char — the clamp removes a real defect, so some of the 2.95 deficit is recovered | measured |
| P4 | Arm D recovers **> 28%** of arm A's deficit vs arm B (arm C's figure), i.e. more than the sign-constraint did — but **< 100%** | (256.8466 − D) / 2.9460 |
| P5 | Arm D's champion is **still out-of-domain on ≥ 6 of 14** gauges. NOT a wiring bug: 8 curves are minimized AT a domain edge, and 5 of those edges are unreachable from the near-optimal band, so the clamped optimum SITS on the boundary by construction | count |
| P6 | The clamp **binds**: pushing arm D's champion further out-of-domain buys **exactly 0.0** additional reward on every clamped gauge | direct measurement |
| P7 | Arm D's champion is **comfort-driven**: `comfort` is pinned at its clamped floor 6.5236 (43.55% of attribution), so the champion's `comfort` level is **≤ 6.5236** | measured level |
| P8 | Arm D's evidence score (clamped) lands **above the clamped lower bound −26.0311** and near it — I predict **[−24.5, −21.0]** | measured |
| P9 | Arm D is **NOT admissible**: 0 dominators on the 10-axis frame, best `n_ge ≤ 4` of 10 | judge |
| P10 | Arm D's normalized floor is **POSITIVE** (arm A/B/C were +0.5836/+0.6005/+0.5689) | judge |
| P11 | **Degeneracy**: arm D's top-50 archive contains ≥ 2 distinct layouts whose clamped scores are equal to < 1e-9 — the (iii) signature | archive |
| P12 | `spearman(ev_clamp, ms/char)` over the pool INCLUDING arm D stays **negative** | judge |

**P4 is the interesting one.** If arm D recovers most of the deficit but stalls short of arm B,
that is (ii) with the extrapolation quantified as the larger of two additive defects. If it
recovers almost nothing, the weights are worse than OPTEVIDENCE-1 concluded.

## What would make me wrong, and what each failure would mean

- **If arm D lands within 0.2222 of arm B (outcome (i))**: the weights were fine and unbounded
  extrapolation was the whole pathology. OPTEVIDENCE-1's "uninformative" conclusion would need
  real softening. I put this at **~10%** — the −0.4435 in-band rank correlation and the 92.5%
  wrong-direction headroom both argue against it, and both are measured, not assumed.
- **If arm D is *worse* than arm A**: the clamp made things worse, which would mean the flat
  objective actively misleads (strongest (iii)). **~15%.** Note P3 and this are complementary;
  I predict P3 but this is the live alternative, because a 0.094x-spread objective is close to
  a random walk and a random near-optimal C30M layout is not fast.
- **If arm D's champion is in-domain on ≥ 12 of 14**: my P5 reasoning about boundary optima is
  wrong and I would re-examine whether `np.clip` is reachable in the search path at all.
- **If P6 fails** (pushing further out still pays): the clamp is NOT wired into the search and
  every arm D number is void. This is the abort condition, not a finding.

## Pre-registered abort conditions (so a broken run cannot become a result)

1. Gate 1 must pass (**it did**: 1301 checks, 0 failures, on the real fitted curves).
2. P6 must hold on arm D's champion. If it fails, arm D is reported as FAILED, not as an outcome.
3. Arm D must reach **≥ 9.0M unique evals** (arm A: 9,434,590) or the comparison is
   budget-confounded and I say so rather than quoting the gap.
4. Arm D's gauge computation must be bit-identical to arm A's. `ClampedEval` wraps the same
   `FastEval`, and gate D asserts `ClampedCurve == evobj.Curve` in-domain, so any difference is
   the policy alone.

## Explicitly NOT predicted

I do not predict arm D's *layout*. I do not claim any layout should be promoted or adopted —
that is the user's gate alone. Every number here is **MODELLED ONLY** (fitted-surface
attribution at 90 WPM on blend-v1), not measured typing speed; tau is saturated and Phase-D is
cancelled.
