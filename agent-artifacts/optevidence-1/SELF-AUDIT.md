# OPTEVIDENCE-1 SELF-AUDIT — attacking my own load-bearing claims

Written after the result was registered as `9fd5c7b` and after proposal (a) was shipped as
branch `domain-hard`. Every number below is re-derived, not transcribed. Artifacts:
`self-audit.json`, `decisive-inband-rank.json`, `banded-rank.json`, `decomposition-audit.json`,
`amplification-audit.json`, `extrapolation-split.json`.

**Bottom line: 1 headline SURVIVES but for a reason I never tested (and my stated warrant for
it was indeed confounded), 1 number is CORRECTED DOWN, 1 claim is DEMOTED to a
non-generalizable observation, and 2 claims stand.**

---

## CLAIM 1 — "the weights are UNINFORMATIVE about predicted time, not merely wrong-signed"

### The confound is REAL and my stated warrant was wrong

You are right. Arm C bounded the **five sign errors** and left **extrapolation entirely free**.
I then separately attributed 96.5% of arm A's win to extrapolation. So the 72% residual I
called "evidence the weights are uninformative" is *mostly the extrapolation my own
decomposition attributes elsewhere*. Quantified for the first time here — arm C's champion is
out-of-domain on **10 of 14** gauges, exactly like arm A's, and only **3.9–10.5%** of its score
advantage survives a clamp. **Arm C did not isolate what I claimed it isolated.** The
inference "72% residual ⟹ weights uninformative" is confounded, and I should not have stated it
that way.

### But the conclusion itself survives, on evidence I did not run in round 1

I built the test that avoids arm C entirely: rank-agreement between the evidence objective and
predicted ms/char on a pool of **36,005 incumbent perturbations** (1–6 random swaps off each of
the five incumbents) — a pool selected by *neither* objective, so it cannot flatter either.
Instrument positive control: the served objective vs itself gives ρ = +1.0000, so the test can
detect agreement. Positive ρ = the objective agrees with predicted speed.

| band (ms/char) | n | sd | ρ raw | CI95 | verdict |
|---|---|---|---|---|---|
| **≤ 255.0** | 809 | 0.114 | **−0.0455** | [−0.111, +0.026] | **INDISTINGUISHABLE from 0** (inside the scorer's own noise band, p95 0.2231) |
| ≤ 255.5 | 2,260 | 0.242 | +0.2373 | [+0.199, +0.275] | agrees (barely outside noise) |
| ≤ 256.0 | 3,890 | 0.378 | +0.3896 | [+0.362, +0.417] | agrees |
| ≤ 257.0 | 6,852 | 0.651 | +0.6331 | [+0.620, +0.647] | agrees |
| ≤ 260.0 | 16,884 | 1.517 | +0.7940 | [+0.788, +0.800] | agrees |
| all | 36,005 | 4.315 | +0.9111 | [+0.909, +0.913] | agrees |

**This is the finding I should have led with.** The weights are *strongly* informative about
predicted time when the comparison spans degraded layouts (ρ +0.91 overall), and they become
**indistinguishable from noise exactly in the band the incumbents occupy** (≤255.0, where the
five incumbents span 254.6307–254.8436: ρ = −0.0455, CI straddling zero, inside the
evidence-scorer's own placebo band). Informativeness decays monotonically as the band tightens.

So: **"uninformative about predicted time" is TRUE but must be scoped — uninformative
*in-band*, highly informative *out-of-band*.** My report said it flatly, which over-claims.
The mechanism is also different from what I implied: it is not that the weights carry no signal,
it is that all their signal lives in a dynamic range coarser than the one a search resolves.

### Verdict on claim 1
**Headline SURVIVES, warrant REPLACED, scope TIGHTENED.** The arm-C-based argument in report
§5 is confounded and should be read as *suggestive only*; the in-band rank test is the sound
evidence. I would not have found this without the challenge.

### What arm D would have to show
- **Refutes my conclusion** if the clamped arm D champion lands **≤ 254.85** (inside the
  incumbent band) — that would mean the weights *plus a domain guard* are a usable search
  objective and the deficit was purely extrapolation, not in-band uninformativeness.
- **Confirms it** if arm D still lands **≥ ~255.5** despite the clamp. Note ⚠: clamping makes
  the objective **piecewise-flat outside every domain**, so a clamped search gets *no gradient*
  there — arm D may plateau early or return a near-arbitrary point among ties. That is a
  predicted artifact of CLAMP, not evidence about the weights, and arm D should check for tie
  plateaus before reading its champion.
- **My honest prior:** arm D improves on arm A's 256.8466 (removing the sr-roll/comfort
  exploit should help) but does **not** reach the incumbent band, landing ~255.3–256.3.
  🟠 INFERRED — from the ≤255.0 ρ ≈ 0 result, not measured.
- ⚠ One caveat for the sibling: I re-scored the existing champions under CLAMP and arm A's
  champion is **still ranked best** (−18.6413 vs incumbents' −17.89…−18.50), and the
  clamped objective's rank agreement with ms/char over those 8 layouts is **negative**
  (ρ −0.395). CLAMP fixes the *unbounded exploit*; on this evidence it does **not** by itself
  make the objective rank the near-optimal band correctly.

---

## CLAIM 2 — the 96.5% decomposition (comfort 57.3% + sr-roll 39.2%)

### Method, stated plainly
The evidence objective is a **sum of 14 independent univariate curves**, Σ price_m(x_m). I
computed `delta_m = price_m(champ) − price_m(ref)`. That is an **identity, not a model**.

### It is NOT order-dependent — verified two ways
- **Exactness:** Σ delta_m = −6.302379179938 vs total difference −6.302379179938, residual
  **3.6e-15**. No residual to allocate, so no ordering choice exists.
- **Shapley test:** I computed the exact Shapley value over the **11 correlation clusters**
  (2^11 subsets, full permutation weighting) with a real ablation (cluster takes champion levels,
  others frozen at reference). **Every cluster's Shapley value equals its additive delta to
  <1e-9.** For an additive objective these coincide by construction; I confirmed it rather than
  assuming it. Collinearity (VIF 12.8–119) corrupts *which gauge got which coefficient in the
  fit* — it does not corrupt the arithmetic of splitting an already-separated sum.

### But I DID inflate the denominator — number corrected
Shares were taken over the **NET** gain (−6.3024) while six gauges push the *other* way
(+0.7458 total). A share of a net quantity is inflated by **1.118×**.

| | comfort + sr-roll |
|---|---|
| as % of **net** (what I reported) | **96.5%** |
| as % of **gross improvement** (honest denominator) | **86.4%** |

**Correction: the two gauges carry 86.4% of the gross improvement, not 96.5%.** The 96.5%
figure is defensible only if stated as "of the net advantage", which my report did not make
explicit. The qualitative claim (two gauges dominate) is unaffected.

### The limitation that matters more than either
The share answers *"of the score difference, how much is booked to gauge m?"* It does **not**
answer *"if gauge m were removed, how much of the ms/char deficit would go away?"* Because the
gauges are collinear you cannot move one and hold the rest fixed on a real board, so the share
is **not a counterfactual**. That counterfactual needs a re-search per ablation — arm-D-shaped,
and the sibling's job, not mine.

### Verdict on claim 2
**Method VALID and order-independent. Headline number CORRECTED 96.5% → 86.4% of gross.
Causal reading explicitly disclaimed.**

---

## CLAIM 3 — "sr-roll: 39.2% of the win from 4.90% of attribution = 8× amplification"

### The baseline was cross-pool, and I did not say so
`shap_share_pct` is mean |SHAP| over the **fitting pool (400 random permutations)**. My win
share is over **one champion-vs-one-incumbent difference in the near-optimal band**. So "8×"
divides a share on pool X by a share on pool Y. That is not wrong, but it is a **cross-pool
ratio** and my report presented it as if both terms lived on the same footing.

### Is 8× a property of sr-roll or of hinge geometry? — NEITHER, and that kills the claim's generality
I regressed the amplification ratio on the geometric predictors across all 14 gauges:

| predictor | Spearman ρ with amplification |
|---|---|
| \|far slope\| | +0.2132 |
| distance outside domain | −0.1111 |
| **\|far slope\| × distance** (the mechanical product) | **+0.0711** |
| shap share | −0.0286 |

**Hinge geometry does not predict amplification at all** (ρ = +0.07). So 8× is not "what any
steep-far-slope hinge would show". But it is not a distinctive property of `sr-roll` either:
`comfort` has the **largest** \|far slope\| × distance (13.06, rank 1 of 14 — sr-roll is rank 2
at 4.87) and only 1.32× amplification, because its fitted share was already huge. The ratio is
just **(realized share) / (fitted share)**, and it is large for sr-roll only because the
denominator happens to be small. Five gauges exceed 1× (sr-roll 8.01, sfs 1.82, lsb-dist 1.56,
comfort 1.32, sfs-dist 1.05).

### Verdict on claim 3
**DEMOTED.** "8× amplification" is arithmetically correct but is a **ratio of two shares
measured on different pools**, is not explained by hinge geometry, and is not a property of
`sr-roll` as a gauge. It should be stated as a descriptive fact about this one comparison
("sr-roll contributed far more to this champion's score than its fitting-pool importance
suggested"), **not** as a mechanism. The load-bearing fact underneath it is unaffected and
better: `sr-roll` sits at 17.83 vs a domain ceiling of 8.34.

---

## CLAIM 4 — P6 falsification (the normalized floor is POSITIVE, not negative)

### I checked comparability before letting it propagate. The floors ARE the same quantity.
WSCISSOR-GEN-1's blend-v1 incumbent floors vs mine:

| layout | WSCISSOR-GEN-1 | mine | \|diff\| |
|---|---|---|---|
| keybo-lsb | 0.7270 | 0.726952 | 4.8e-05 |
| keybo-lsb+lm | 0.7302 | 0.730176 | 2.4e-05 |
| lsb-sib | 0.7399 | 0.739914 | 1.4e-05 |
| archive-1843 | 0.7442 | 0.744198 | 2.3e-06 |
| archive-1846 | 0.7452 | 0.745186 | 1.4e-05 |

Worst |diff| **4.8e-05** — agreement to every digit they published. Same normalization
(ceiling-fraction), same corpus (blend-v1), same 46-layout reference population. **So the
comparison is apples-to-apples and the falsification is a real falsification of the number.**

### But the *interpretation* I attached to it was too strong
Their champions: blend-v1 −0.4470 / −0.1016 / −0.0330; iWeb −0.1059 / −0.1538 / −0.4339 — all
negative. Mine: +0.5836 / +0.6005 / +0.5689. The mechanical reason is an **objective-shape
difference I did not state**: WSCISSOR-GEN-1 drove a **single narrow axis to its Pareto extreme**
under a 6-objective NSGA-II (its own report: "pushing a narrow axis to its Pareto extreme…
produces degenerate corners"), whereas my arm A optimizes a **14-gauge composite** in which
`comfort` alone prices off-home, bottom-row, sfb, scissor and lsb — i.e. my objective contains
broad positional pressure that theirs lacked.

Two further corrections to how I framed it:
- **Trap 19 applies to the precedent itself.** WSCISSOR-GEN-1 printed `(feasible)` for its
  *constrained* champion's floor — never computed, later found to be **+0.8025**. So the
  "every champion has a negative floor" precedent covers **unconstrained** champions only.
- My **arm C is the analogue of their constrained cell**, and both are positive (mine +0.5689,
  theirs +0.8025) — **consistent, not contradictory.**

### Verdict on claim 4
**Falsification of the NUMBER stands (verified same quantity). The claim "this is not a
ruler-optimizing pathology" is OVER-STATED** — it is not *their* pathology because my objective
is a composite rather than a single-axis extreme, which is a difference in objective shape, not
proof that ruler-optimizing is absent. Arm A still wins its own ruler while losing 7–9 of 18
independent gauges; that behaviour is intact.

---

## CLAIM 5 — what I did NOT test and would want a reviewer to check

1. **🔴 Untested and potentially decisive: the ARCHIVE-fitted weights cover the band.** I only
   ever used `arm-random400-native.json`. Checking `arm-archive400-native.json`, `keybo-lsb` is
   out-of-domain on **0 of 14** gauges (vs 9 of 14 under random400) — that pool's domains
   *do* cover the near-optimal band. EVIDENCE-SCORER-1 rejected those weights because they lose
   all 12 cross-source cells, so they are worse *as a scorer* — but they may be the better
   *search* objective, since a search's failure mode here is extrapolation, not ranking. **This
   is the single most valuable follow-up and it is not arm D.** Nobody has run it.
2. **🟡 One corpus, one surface.** Everything is blend-v1 + K31 @ 90 WPM. No iWeb replication.
   `arm-random400-native-iweb.json` exists and was never used by me.
3. **🟡 One objective shape per arm.** Single-objective scalarization only. A multi-objective
   (NSGA-II) evidence arm could behave differently, and given claim 4's finding — that objective
   *shape* drove the floor-sign difference — this is more load-bearing than I treated it.
4. **🟡 CLAMP's flat-region pathology is unexamined.** Clamping removes the gradient outside
   every domain. I did not test whether a clamped search plateaus on ties. Directly relevant to
   arm D's interpretability.
5. **🟠 Search-noise placebo covered 2 arms, not 3.** No band for the constrained arm, so arm C
   vs arm A (−0.8246) leans on the evidence arm's sd (0.3440) as a proxy.
6. **🟡 The 19-gauge win counts** in report §④ are uncorrected for effective dof ~4–5 (trap 39).
   Flagged in the report; still un-recomputed per-cluster.
7. **🟠 `sfs` is mis-signed in my own P0 framing.** I said `sfs` is "pushed the
   mechanism-CORRECT way", and its argmin is at the range bottom — but its realized delta is
   **−0.4928 (7.8% of the win)**, i.e. it *contributed* to the win, 69.6% of that from
   extrapolation below its domain floor. My P0 statement is about the curve's argmin and does
   not license the impression that `sfs` was inert.
8. **🔴 Reference-population dependence of the floor.** All normalized floors divide by ceilings
   from one frozen 46-layout population. I positive-controlled the *derivation* (4.4e-14 vs the
   frozen iWeb constant) but never tested sensitivity to the population choice, and every floor
   claim — including the P6 falsification — inherits it.

---

## Net effect on the report's conclusions

| conclusion | status after audit |
|---|---|
| Arm A's champion is +2.95 ms/char worse than arm B's, resolving | **unchanged** (paired, placebo'd, saturated) |
| Extrapolation, not wrong signs, is the exploit | **strengthened** — only 1.8–10.5% of arm A's score advantage survives a clamp |
| Weights uninformative about predicted time | **survives, rescoped to IN-BAND**; warrant replaced (arm C → in-band rank test) |
| 96.5% from two gauges | **corrected to 86.4% of gross** |
| 8× amplification | **demoted** to a descriptive cross-pool ratio, not a mechanism |
| P6: floor positive not negative | **number verified same quantity**; "not ruler-optimizing" over-stated |
| No champion admissible | **unchanged** |
| `valid_domain` must be a hard constraint | **strengthened, and now shipped** — but CLAMP alone does not fix in-band ranking (ρ −0.395 on the 8-layout re-score) |
