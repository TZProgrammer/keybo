# PREREGISTERED PREDICTION — written BEFORE any search was run

Timestamp: see events.log (`prediction-written`). Derived from the fitted curves in
`state/evidence-scorer/artifacts/arm-random400-native.json` ALONE — no search had been run,
no champion existed. Driver: `drivers/predict.py`; output `prediction-curve-analysis.json`.

## 0. Correction to the brief's premise (trap 20: re-derive every numeric claim)

The brief says FIVE signs are wrong and that a search "will drive the wrong-signed gauges to
their extremes". The five linearized weights are wrong-signed — I reproduced all five to the
digit:

    scissor -0.4723 | sfb -0.1122 | sfb-dist -0.1073 | lsb-dist -0.0486 | sfs -0.0038

But **the search does not optimize the linearized weight; it optimizes the CURVE.** 13 of 14
curves are hinges, and a hinge's slope changes sign at the knot. Taking the argmin of each
fitted curve over its own observed range:

| wrong-signed gauge | form | slope below knot | slope above knot | argmin | exploitable? |
|---|---|---|---|---|---|
| `scissor`   | linear | −0.4723 | −0.4723 | **top of range, unbounded** | **YES** |
| `sfb`       | hinge  | −0.2091 | −0.0486 | **top of range, unbounded** | **YES** |
| `lsb-dist`  | hinge  | −0.3534 | −0.0236 | **top of range, unbounded** | **YES** |
| `sfb-dist`  | hinge  | −0.1643 | **+0.0013** | interior, x\*=18.40 (≈knot) | no — self-limiting |
| `sfs`       | hinge  | **+0.7259** | −0.1054 | **bottom of range** | no — pushed the CORRECT way |

So the exploitable set is **3 of 14, not 5 of 14**, and `sfs` — one of the five "wrong signs" —
is actually driven in the mechanism-CORRECT direction by its own curve. This is a genuine
refinement of the brief, not a contradiction of it: the linearization is wrong-signed for five
gauges, but only three of those five give a maximizer an unbounded downhill direction.

## 1. What the objective is mostly ABOUT

`comfort` is 43.55% of the attribution, its curve is monotone-increasing (slopes +5.98 below
knot, +3.66 above), so the search pushes comfort **DOWN and beyond the observed minimum**.
`comfort` is `DEFAULT_COMFORT`, a hand-chosen taste table (off_home 8.0, bottom_row 10.0,
**sfb 25.0**, **scissor 15.0**, **lsb 10.0**, lag2_reuse 5.0) with no fitted parameter.

**P1. The evidence champion will be predominantly comfort-driven** — `comfort` will supply the
largest single block of the improvement in the evidence score. An "evidence-based" search is
therefore substantially a search against a rival's taste constants.

## 2. The objective contains an internal contradiction, and I predict which side wins

`comfort` (43.55%, slope ≈ +5.98/unit) *penalizes* sfb at 25.0 ms-equiv/occurrence and scissor
at 15.0, while the direct `sfb` and `scissor` terms *reward* them. Rough exchange rate: +1
percentage point of sfb share ⇒ ≈ +0.25 comfort units ⇒ **+0.9 to +1.5 ms** of price, versus the
direct sfb term's **−0.05 to −0.21 ms**. For scissor: +1 pp ⇒ ≈ +0.15 comfort units ⇒ **+0.55 to
+0.90 ms**, versus the direct term's **−0.47 ms**.

**P2. `comfort` overrides the wrong sign on `sfb` (by ≈7–30×) and on `scissor` (by only
≈1.2–1.9×, the closest contest).** So `sfb` and `scissor` should NOT blow up — sfb should come
DOWN, scissor should be near-neutral-to-mildly-elevated.

**P3. `lsb-dist` is the UNOPPOSED exploit and is my headline prediction.** `DEFAULT_COMFORT`
prices `lsb` (count) at 10.0 but has **no `lsb-dist` term at all**, and the fitted `lsb` curve
itself pushes `lsb` DOWN (slope +0.618 below knot). So the objective can lower the lsb COUNT
while widening the surviving stretches — nothing in the objective opposes it.
Concretely: **the evidence champion's `lsb-dist` will exceed the incumbent band and will sit
ABOVE the fitted valid_domain upper bound of 16.7225** (observed pool max 27.7176).

## 3. Out-of-domain extrapolation

Nine of 14 curves have their argmin exactly at an endpoint of the observed range with the
price still descending past it (`comfort`, `alt`, `imbalance`, `sr-roll`, `scissor`, `sfs`,
`redir`, `lsb`, `sfs-dist`, `sfb`, `roll`, `lsb-dist`). A maximizer therefore has an incentive to
leave the fitted support on many axes at once.

**P4. The evidence champion will be out-of-domain on ≥4 of 14 gauges** (i.e. `extrapolating:
true`), so its headline score is an extrapolation rather than an optimum. I name `comfort`,
`alt`, `sr-roll` and `lsb-dist` as the most likely out-of-domain axes.

## 4. Predicted ms/char — the served surface

**P5. The evidence champion will be WORSE (higher ms/char) than the baseline-arm champion on
the served surface, and worse than the best incumbent.** EVIDENCE-SCORER-1 measured these
weights losing 0 of 12 cross-source cells on the near-optimal band and sitting inside the
noise placebo band there — a search operates in exactly that band. I predict the evidence
champion loses to the baseline champion by **more than the paired resolution** (~0.17–0.24
ms/char), i.e. the gap RESOLVES against the evidence arm.

**P6. Optimizing-the-ruler will be confirmed:** the evidence champion wins its trained
objective by a wide margin and has a NEGATIVE normalized six-surface floor (WSCISSOR-GEN-1's
precedent), i.e. it is worse than the reference population's worst on at least one speed
surface.

**P7. The evidence champion will NOT be admissible** — it will dominate no incumbent on the
12-axis frame and will carry at least one pathological axis.

## 5. What would FALSIFY this, stated in advance

- P3 falsified if the evidence champion's `lsb-dist` lands inside the incumbent band.
- P2 falsified if `sfb` or `scissor` blows up past the observed pool maximum (that would mean
  the direct wrong-signed terms beat comfort, i.e. the brief's version is right and mine wrong).
- P5 falsified if the evidence champion's ms/char beats the baseline champion's, or ties within
  the paired resolution.
- P6 falsified by a positive normalized floor.
- If ALL of P1–P7 fail, the weights are informative and my whole framing is wrong. That is a
  publishable outcome too and I will report it as such.

## 6. Arm C (constrained) prediction

**P8.** With the five wrong-signed gauges hard-bounded at the incumbent band, arm C's evidence
score will still improve substantially over the incumbents (because `comfort` + `alt` carry
53.6% of the attribution and neither is constrained), but arm C's champion will remain WORSE
than the baseline champion on ms/char. That would mean the weights are **uninformative about
predicted time**, not merely wrong-signed — the sign errors are a symptom, not the disease.
