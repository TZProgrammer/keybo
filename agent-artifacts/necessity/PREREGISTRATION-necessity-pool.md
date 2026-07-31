# PREREGISTRATION — is NEAR-OPTIMALITY *necessary* for the cross-source instrument disagreement?

Written and committed **BEFORE any new score was read.** Branch `necessity-pool`, base `e6a5b9e`.
This file is local to my branch; `PREREGISTRATIONS.md` is NOT touched.

## 0. What is already established (verified by me, not accepted on trust)

Positive control first: my from-scratch pipeline reproduces poolsweep-1's published cells
**exactly** — `archive-x400` rho `+0.2184272` (published `+0.2184272`, |Δ| = 0.00e+00),
`random-wide` rho `+0.7970054` (|Δ| = 0.00e+00), within-COMMUNITY seed means `+0.9647` /
`+0.9872` (|Δ| ≤ 3.4e-05), archive bank 2860 layouts, reference bank moments identical to 6 dp.
So any difference I report downstream is a property of a pool design, not reimplementation drift.

## 1. 🔴 A DEFECT IN THE BRIEF I MUST DESIGN AROUND: the "SLACK" is *also* algebra

The ledger retracts `Spearman(rho, log C/D) = +0.999` as an identity, then promotes as *what
survives empirically*: "the closed form's SLACK is largest for the archive ALONE (+0.0634)
precisely because it is the only asymmetrically restricted pool."

**The slack is the same identity's second-order term.** Writing `q = u_A/u_B`, the exact
relation for any pool (no assumption) is

    r_Pearson = [(k² − 1)/(k² + 1)] · [(1 + q²) / (2q)]
    SLACK     = r_Pearson − (k²−1)/(k²+1) = [(k² − 1)/(k² + 1)] · [(1 + q²)/(2q) − 1]

I evaluated both against all **13** published `A1_algebra` rows:
max |predicted r − measured Pearson r| = **4.441e-16**, max |predicted slack − reported slack| =
**4.725e-16**. The slack is a deterministic function of `(k, q)`. It measures nothing beyond them.
So `+0.0634` is not the archive's empirical signature — it is `(1+q²)/(2q) − 1` at `q = 0.249`
times `(k²−1)/(k²+1)`. **Trap 11/30 recurred one level up, inside the retraction of trap 11/30.**

Consequence for the design: the second moments hold **exactly one** empirical degree of freedom.
`r_Pearson = cov(z_A,z_B)/(u_A u_B)`, and `(k, q, scale) ↔ (r, u_A, u_B)` is a reparameterization,
3 numbers ↔ 3 numbers. **So a Pearson-channel test is near-vacuous once `u_A`, `u_B` are matched,
and the probe must live in channels the identity does NOT determine:**
  (a) **Spearman** rho — rank structure, not a function of `(k, q)`;
  (b) the **within-instrument** (COMMUNITY seed-refit) reliability.

## 2. 🔴 A SECOND DEFECT: "the archive is the ONLY asymmetric pool, ~1.0 for every constructed pool" is FALSE

That holds only for `final.py`'s `boxmatch`/`curve` cells. `matched.py` already built **eleven**
asymmetric random-lineage cells. From its own artifact (`matched-blend-seed0.json`):

| constructed cell | u_A | u_B | q = u_A/u_B | rho |
|---|---|---|---|---|
| `bandrandom-A-match-archive-mid` | 0.0039 | 0.6191 | **0.0063** | −0.0140 |
| `bandrandom-A-sd10` | 0.0999 | 0.4119 | **0.2425** | **+0.2205** |
| `jointband-match-archive-mid` | 0.0427 | 0.1559 | **0.2737** | **+0.0537** |
| `jointband-match-archive-full` | 0.2205 | 0.1619 | 1.3620 | +0.2314 |
| (archive, for reference) | 0.0399 | 0.1605 | 0.2488 | +0.2184 |

Two of these bear directly on my question and **were never read as such**:
`jointband-match-archive-mid` is a *good two-sided match* (u_A 0.0427 vs 0.0399, u_B 0.1559 vs
0.1605) and collapses to **+0.0537**; `bandrandom-A-sd10` matches only the *ratio* (q 0.2425) at
~2.5× the u-levels and lands at **+0.2205**, i.e. reproduces the archive's rho almost exactly.
They straddle the archive. The ledger's dismissal of "jointband" cited `-full` (u_A 0.2205, "5.2×
the archive's") and generalized it to the `-mid` cell, which is well matched.

## 3. What I will build (the confound-breaking design)

A **two-parameter** restriction, since the restriction has two sides — the exact defect that made
P4 unreadable. Random lineage throughout: `rng.permutation(C30M)`, no search, no archive ancestry.
Selection is a 2-D box on `(y_A, y_B)` with **both** half-widths bisected until the **ACHIEVED**
`sd_A` and `sd_B` hit target. Achieved `u_A`, `u_B` are reported; requested values are never quoted
as results.

- **PRIMARY `asym-match`** — target the archive's own achieved `u_A = 0.0399`, `u_B = 0.1605`
  (⇒ q ≈ 0.249). Matched to the archive on pool size (n = 400), replicate structure, scale
  (`.native`, ms/trigram, same reference bank), and statistic (Spearman on ms/trigram).
- **PRIMARY CONTROL `sym-match`** — `u_A = u_B = √(0.0399 × 0.1605) = 0.0800`. *Identical overall
  narrowness (same geometric mean), asymmetry the ONLY thing changed.* This is the
  one-variable-at-a-time contrast that neither `boxmatch` (q = 1.025) nor `jointband` isolated.
- **q-LADDER at FIXED geometric-mean u = 0.0800**: q ∈ {1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16} via
  `u_A = √q·0.08`, `u_B = 0.08/√q`. Maps rho against asymmetry with overall narrowness held fixed —
  the curve P4 could not represent because it took one `u`.
- **LEVEL vs RATIO**: `asym`/`sym` at 1×, 2×, 4× the u-levels.
- **REPLICATES**: R = 12 independent construction seeds for `asym-match`, `sym-match`, and
  `archive-x400`, giving a replicate sd for each — **this, not any ms/char number, is my
  resolution floor** (see §5).
- **BOTH AGREEMENTS for every cell**: instrument-vs-instrument = cross-source Spearman(AALTO,
  COMMUNITY); instrument-vs-itself = mean pairwise Spearman over COMMUNITY's 3 per-seed refits.

**Stated identification limit I cannot close:** no random permutation reaches the archive's speed
(archive mean_A ≈ 254.8 vs random-bank mean_A ≈ 277.3 ms/trigram). Every random-lineage cell is
centred on the random median, so LEVEL is structurally unmatchable (trap 16). I match spread,
size, scale, statistic and replicate structure; I do not match level, and say so.

## 4. PREDICTION — committed before any new score is read

**The archive's signature is TWO-legged: HIGH within-instrument (+0.9647) + LOW cross-instrument
(+0.2184).** `boxmatch` reproduced only the second leg (within **+0.4605**), so its low cross-rho
is partly refit attenuation — it never reproduced the archive's signature at all. The sharp
question is therefore whether *asymmetric restriction alone* can produce **high within + low
cross**.

**I expect it CANNOT, and that the necessity question splits by leg:**

1. **Cross-instrument leg — asymmetric restriction is largely SUFFICIENT.** I predict
   `asym-match` rho lands in **[0.00, +0.16]**, at or below the archive's +0.2184, with the
   archive−asym difference positive but its 95% CI including zero. (Anchors: the well-matched
   `jointband-mid` gave +0.0537; ratio-only `bandrandom-A-sd10` gave +0.2205.)
2. **Within-instrument leg — asymmetric restriction is NOT sufficient.** I predict `asym-match`
   within-COMMUNITY reliability lands **below +0.80**, i.e. `boxmatch`-like rather than
   archive-like, because a narrow random window contains little true signal for independent
   refits to agree on.
3. **Asymmetry per se moves rho only weakly at fixed overall narrowness.** On the q-ladder I
   predict |rho(q = 1/4) − rho(q = 1)| < 0.15, and the ladder to be roughly **symmetric in
   log q** (the algebraic asymmetry factor `(1+q²)/(2q)` is), i.e. asymmetry is not a
   *direction*-carrying variable for Spearman.

**⇒ Pre-registered verdict if all three land: near-optimality is NOT necessary for the
cross-source collapse (asymmetric restriction reproduces it), but IS necessary for the
instrument-DISAGREEMENT reading of it — because only the archive keeps the within-instrument
leg high while the cross leg collapses.**

### FALSIFIERS (each names the observation that kills the claim)

- **F1 kills "asymmetric restriction is sufficient" (⇒ near-optimality IS necessary):**
  `asym-match` rho ≥ **+0.30** with a 95% CI excluding the archive's +0.2184.
- **F2 kills my within-instrument prediction (⇒ near-optimality NOT necessary for either leg —
  the strongest anti-necessity result available):** `asym-match` within-reliability ≥ **+0.90**
  while its cross-rho stays ≤ **+0.30**. This reproduces the archive's full two-legged signature
  from a random pool and would refute my §4.2.
- **F3 kills the "reproduces the collapse" framing:** `asym-match` rho ≤ **−0.10** (overshoot into
  anti-agreement ⇒ the construction made something structurally unlike the archive, not a match).
- **F4 kills §4.3:** the q-ladder is strongly **asymmetric** in log q (|rho(q) − rho(1/q)| > 0.20
  at any q), i.e. *which* source is squeezed matters, not just that they are squeezed unequally.
- **F5 (construction validity — checked FIRST, gates everything):** achieved |u_A/0.0399 − 1| >
  0.10 or |u_B/0.1605 − 1| > 0.10. A cell that missed its two-sided target cannot test necessity,
  and I report the miss instead of the verdict.

## 5. Resolution floor — and why I will NOT quote 0.17–0.24

My brief says to state "the paired resolution (~0.17–0.24), NOT the unpaired 0.72", and also that
a floor may be quoted **only if the quadruple (pool × replicate-structure × scale × statistic)
matches**. Those two instructions conflict here, and the second is the correct one: **0.17–0.24
and 0.72 are ms/char floors on LAYOUT TIME. My statistic is a correlation.** The quadruple fails
on its fourth element, so importing that floor would be a units error dressed as rigour.

The floor for *this* statistic is derived, not imported: the **replicate sd of rho across R = 12
independent construction seeds** (construction noise), reported alongside the **within-draw
bootstrap CI** (layout-sampling noise). Both are emitted per cell, and no difference is called
real unless it exceeds the replicate sd.

## 6. Statistical protocol, fixed in advance

- Cross-source statistic: Spearman rho on ms/trigram, `.native` frame (asserted at load).
  `.standardized` is refused: all sources carry AALTO's bigram tensor there.
- Bootstrap: 8000 resamples over layouts, percentile CI — **same protocol as the A5 test whose
  +0.1106 I am comparing against**, so the numbers are commensurable.
- `asym` vs `sym` is a **PAIRED** comparison: both are cut from the same random bank with the same
  construction seed, so replicate *r* is a matched pair, and asymmetry is the only difference →
  Wilcoxon signed-rank over R = 12 pairs. This is my primary inferential test.
- `archive` vs `asym` is **UNPAIRED** (disjoint layout universes, different lineage) → two-sample
  bootstrap difference + replicate distributions. I will label it unpaired rather than borrow the
  paired test's resolution.
- Every table in the report is machine-GENERATED from the emitted JSON; no hand-transcribed cells.
