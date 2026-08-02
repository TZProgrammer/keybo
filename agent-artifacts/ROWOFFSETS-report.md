# Report — `stagger`: why `row_offsets` is hardcoded, and whether it can be fit

**Question (human's words):** *"Why is row offset hardcoded like this? Is there a way to properly assign
the best row_offsets for the learning of the model?"*

**Target:** `src/keybo/geometry.py:65` — `row_offsets: dict[int, float] = field(default_factory=lambda:
{1: 0.5, 2: 0.0, 3: -0.25})`, consumed by `stagger_adjusted_dx` (`geometry.py:90-98`) and thence by the
features `dx`, `sg_dx`, `lsb`, `angle`, and by `analysis.lateral_span`.

**Branch:** `stagger-rowoffsets` (worktree `/tmp/stagger-wt`), off `main` @ `3c91928`.
**Pre-registration:** `/tmp/stagger-wt/agent-artifacts/ROWOFFSETS-prereg.md` — committed at `dd112d3`
BEFORE any INVARIANT B/C result existed; space-axis addendum at `dd3223b`, likewise before any space
number existed.

**Status of this document:** FINAL. All five sections complete, plus the space axis added to scope
mid-task by the parent. Every decision rule quoted was committed before the corresponding measurement.

---

## 1. INVARIANT A — the parameter count, and the identifiability proof

### 1.1 The answer: **2 free parameters on the letter block, 3 on the frame the model is actually trained on.**

The parent's expectation of 2 is **CONFIRMED for letter-letter pairs** and **incomplete for the real
training frame**, because a third parameter — space's own offset — exists and is silently pinned.

### 1.2 Uniform shift cancels on letter pairs (ledger:9313 CONFIRMED, with a caveat that matters)
🟢 VERIFIED. Adding a constant `c` to all three rows leaves the 20-column bigram feature matrix
**bit-identical** over all 900 ordered letter-letter position pairs of `ROW_STAGGERED_30`, for every
**dyadic** `c` tested (0.5, 1.0, 7.0, ±0.25, 0.125). Same on K31 over its 961 pairs, and on the trigram
frame (240 triples). Mechanism: offsets enter only as `(ax + off(ay)) − (bx + off(by))`.

⚠️ **The caveat is a real defect, not a formality.** At **non-dyadic** `c` (measured at `c = −0.30` and
`c = 1/3`) the **`lsb` indicator FLIPS on 4 pairs** — `max|Δ| = 1.0`, four orders of magnitude above the
float noise (`~4.4e-16`) seen on `dx`. Cause: `is_lsb` tests `stagger_adjusted_dx > 1.5` and 1.5 is
**exactly attainable**, so the gauge freedom is exact in real arithmetic but **not** in floating point at
the threshold. Consequence, and it constrains any future work here: **every candidate offset vector must
be checked for indicator flips, not merely for `dx` equality.** My grids are dyadic for this reason.

### 1.3 🔴 CORRECTION TO ledger:9313 AND TO THE BRIEF: the cancellation **FAILS** on space-touching pairs
🟢 VERIFIED. Space sits at `space_position = (0, 0)`; `row_offsets` has keys `{1, 2, 3}` and **no key
`0`**, so `row_offsets.get(ay, 0.0)` **pins space's offset at 0.0** and it does **not** move under the
shift. Measured: **60 of 61** space-involving ordered pairs change; the moving column is `dx`; and
`max|Δ| = c` **exactly**. The trigram frame shows the same on `sg_dx`.

The precise correct statement is therefore: *offsets enter only inside differences, **but one arm of the
difference is a hardcoded constant that no shift moves.*** That is exactly the "an offset used outside a
difference" case the brief said would be a REAL FINDING. It was found by my A2 control and independently
reproduced by the parent (`max|Δ| = 0.3700000000 == c` on 60/60 space pairs at `c = 0.37`).

### 1.4 Same-row pairs carry **exactly zero** information
🟢 VERIFIED. Under arbitrary random perturbation of all three offsets (3 trials), the 300 same-row
ordered letter pairs give `max|Δfeature| = 0.0000000000` — exact, not small. All 600/600 cross-row pairs
move. **Only cross-row (and space-touching) bigrams identify the offsets.**

### 1.5 The rank computation that settles the count
🟢 VERIFIED — numerical Jacobian of (3 offsets) → features:

| universe | rank | singular values | reading |
|---|---|---|---|
| letter-letter only | **2** | `[2.0e5, 3.0e3, 0]` | null space **exactly (1,1,1)** ⇒ 2 params, home pinnable at 0 |
| space-involving only | 3 | `[4.47, 4.47, 4.47]` | space breaks the gauge |
| all pairs | **3** | `[2.0e5, 3.0e3, **4.47**]` | 3rd direction real but **~450× weaker** than the first two |

The third parameter is **space's own offset**, and it is **exactly orthogonal** to the letter block:
setting `row_offsets[0]` moves space pairs and touches **no** letter-letter pair (`max|Δ| = 0.0e+00`,
independently reproduced by the parent at `row_offsets[0] = 0.61`). It reached 0.0 **by dict omission,
never by a decision.**

### 1.6 How much data carries the signal — **plentiful, contra the brief's worry**
🟢 VERIFIED (`census.py`). Of **5854** bigram eval cells (LOLO's unit of evaluation; wpm [40,140)×20,
cell floor 10):

| class | cells | share | corpus-freq mass |
|---|---|---|---|
| cross-row (**identifying**) | 3275 | **55.9%** | 43.1% |
| space-touching (**identifying, 3rd param**) | 876 | 15.0% | 34.3% |
| same-row (**zero information**) | 1703 | 29.1% | 22.6% |

Per-fold cross-row share: qwerty 59.9% · qwertz 57.0% · azerty 54.9% · **dvorak 42.4%** (the thinnest
fold). Row-pair contrasts: (2,3) 1595 cells · (1,3) 976 · (1,2) 704 · space↔2 348 · space↔3 330 ·
space↔1 194. **Sample size does not cap this arm** — so a wide CI downstream would be an instrument-
resolution result, not a data-volume one.

### 1.7 ⚠️ A source defect found on the way, and it invalidates the obvious experiment
🟢 VERIFIED — registered by the parent as **VALIDATE-GEOM-1**. `training/validate.py`'s fold loop calls
`train_fn(train_rows, target_wpm=…, direction=…, kitchensink=…, **params)` and **never forwards
`geometry`**, while `:797`/`:808` evaluate on the passed board. `train_bigram_model` *does* accept
`geometry` (`train.py:444`). So **`validate(geometry=X)` trains on the SHIPPED geometry and evaluates on
X** — deliberate train/serve skew, the exact failure the module docstring says the single-source pipeline
prevents. Measured on the dvorak fold:

| arm | wmae (ms/char) |
|---|---|
| train=shipped, eval=shipped | 13.2296 |
| train=shipped, eval=X — **what `validate(geometry=X)` does today** | **14.1309** |
| train=X, eval=X — the honest A/B | 13.4083 |

The skew artifact is **+0.72 ms/char = 5× my registered 0.135 bar**. A naive `row_offsets` A/B through
`validate()` would have reported an artifact as a result. **All my drivers therefore call
`train_bigram_model(..., geometry=g, ...)` directly.** (The parent scoped the bug INERT for the
K30-vs-K31 comparison, since those geometries share `row_offsets` and `space_position`; my arm is the one
case where it bites. Per instruction I do **not** land a fix — the answer is the deliverable.)

---

## 2. INVARIANT B — the fitted offsets, per fold, with a CI

### 2.1 Design (as registered at `dd112d3`)
7×7 dyadic grid, `off_top` ∈ [−1.00, +0.50] × `off_bottom` ∈ [−0.50, +1.00], step 0.25 at the coarse
stage; home **pinned at 0.0** (justified by A1.5's null space). 4 LOLO folds, seed 0, **trained AND
evaluated under each candidate geometry** (never `validate(geometry=)` — see 1.7). 2384 s.
Shipped (−0.25, +0.50) is on the grid and evaluated by the identical code path.

### 2.2 The surface (pooled wmae, ms/char; rows = `off_top`, cols = `off_bottom`)

```
           -0.500   -0.250    0.000    0.250    0.500    0.750    1.000
top-1.000  10.0058   9.9587   9.9186   9.9230   9.9003   9.9799   9.9189
top-0.750   9.9617   9.9857   9.9485   9.9277   9.9590   9.8779   9.9071
top-0.500   9.9684   9.9383   9.9171   9.9671   9.8952   9.9792   9.8879
top-0.250   9.8812   9.9116   9.9209   9.9942  [9.8948]  9.8933   9.9223   <- [ ] = SHIPPED
top+0.000   9.9173   9.8907   9.8779   9.9043   9.9051   9.9173   9.8882
top+0.250   9.9333   9.8977  <9.8571>  9.9066   9.9897   9.9664   9.9461   <- < > = pooled argmin
top+0.500   9.9029   9.8899   9.8896   9.9106   9.9467   9.9464   9.9066
```

### 2.3 Per-fold estimates — **no two folds agree** (this is the invariant's whole point)

| fold | argmin (top, bottom) | wmae | shipped wmae | delta | fold's own surface spread |
|---|---|---|---|---|---|
| azerty | (0.00, 0.00) | 9.0101 | 9.1224 | −0.1123 | 0.3955 |
| **dvorak** | **(−0.25, +0.50) = SHIPPED EXACTLY** | 13.2296 | 13.2296 | **0.0000** | 0.3368 |
| qwerty | (+0.25, 0.00) | 8.7527 | 9.0056 | −0.2529 | 0.3037 |
| qwertz | (+0.50, −0.50) | 8.1324 | 8.2216 | −0.0892 | 0.1894 |

**The four argmins scatter across the entire box.** Notably `dvorak` — the thinnest fold (42.4%
cross-row) — selects the shipped values *exactly*. And **each fold's own surface spread (0.19–0.40
ms/char) EXCEEDS the whole pooled surface's spread (0.1487)**: the between-fold noise is larger than the
entire signal being fit.

### 2.4 Pooled estimate, CI, and the registered verdict
- Pooled argmin **(top +0.25, bottom 0.00)**, wmae 9.8571 vs shipped **9.8948** ⇒ **delta −0.0377
  ms/char = 28% of the registered 0.135 bar.** **0 of 49** grid points beat shipped by more than the bar.
- Shipped ranks **11 of 49** (z = −0.854); 10 points beat it, all inside noise. Surface sd = 0.0356.
- **CI (prereg B4 — points within the seed-spread 0.1219 of the optimum): 43 of 49 grid points.**
  `off_top` spans [−1.00, +0.50] = **1.50 key widths**; `off_bottom` spans [−0.50, +1.00] = **1.50 key
  widths**. The rule fires: **≥ 1.0 key width ⇒ NOT IDENTIFIED AT THIS SAMPLE SIZE.**
- 🟢 **The shipped point (−0.25, +0.50) is INSIDE the CI**, so per registered rule B5: **THE HARDCODE IS
  VINDICATED — the data cannot distinguish it from the best fit.**
- 🔴 **And the pooled argmin has the WRONG SIGN for ANSI.** `off_top = +0.25` shifts the TOP row
  *rightward*, the opposite of the physical row stagger. A "best fit" that inverts the real geometry
  while gaining 0.28× the noise floor is a fit to noise, not to stagger. **Per prereg B4 I therefore
  quote no point estimate as the answer.**
- Distance from shipped to the pooled argmin: |Δtop| = 0.50, |Δbottom| = 0.50 key widths — reported for
  completeness, but not meaningful given the CI spans the box.

---

## 3. INVARIANT C — held-out effect, paired per-fold, vs the pre-registered bar

### 3.1 The bar (registered at `dd112d3`, before any C number existed)
ADOPT requires **all three**: (a) mean paired per-fold wmae delta ≤ **−0.135 ms/char** (the MODEL-SEED
floor for fixed boards — *not* the 0.883 SEARCH-seed spread); (b) sign holds on **≥ 3 of 4 folds**;
(c) `require_no_high_wpm_regression_in_report` PASSES. Plus (C4) a mandatory same-width **placebo**.

### 3.2 Results — 4 folds × 3 seeds per arm, paired by (fold, seed) per MOR-FIX-1

| arm | offsets | mean paired Δwmae | sd | folds improving | high-wpm gate |
|---|---|---|---|---|---|
| SHIPPED | (−0.25, 0.0, +0.50) | — (baseline) | — | — | PASS |
| **PLACEBO** | shipped **+0.5 uniform** | **+0.1142** | 0.1210 | 1/4 | PASS |
| ZERO | (0, 0, 0) | **−0.0603** | 0.0750 | 4/4 | **FAIL (structural)** |
| **SEEDNOISE** | **SHIPPED again**, seeds 3-5 | **+0.0026** | **0.1219** | 3/4 | **FAIL (structural)** |

### 3.3 The three readings that decide it
1. 🔴 **NOTHING MEETS THE BAR.** The best candidate anywhere in this arm is ZERO at −0.0603 ms/char —
   **45% of the 0.135 bar.** The B-grid pooled argmin is −0.0377 (28%). The space axis is −0.0212 (16%).
2. 🔴 **THE INSTRUMENT'S OWN NOISE IS LARGER THAN EVERY EFFECT MEASURED.** `SEEDNOISE` re-runs the
   **identical shipped geometry** with different seeds: **sd 0.1219**, with per-fold swings up to
   **+0.2237** (azerty) and −0.2092 (dvorak). That sd **exceeds ZERO's entire effect (0.0603), the
   B-grid argmin's (0.0377) and the space axis's (0.0212)**. Any of those "wins" is reproducible by
   changing a random seed and touching no geometry at all.
3. 🔴 **THE PLACEBO CONFIRMS C4's WARNING AND THEN SOME.** A uniform +0.5 shift carries **zero new
   information on letter pairs** (bit-identical by A1.2) yet moves wmae by **+0.1142** — nearly 2× ZERO's
   effect and 3× the B argmin's, in the *worse* direction. Its whole footprint is the 9.8% space-touching
   rows (A1.3) plus refit noise. **Reading a candidate against SHIPPED without this placebo would have
   mistaken a nuisance channel for a geometry result** — the DIRECTION-1 lesson, reproduced exactly.

### 3.4 The high-WPM gate, and why its failures are *not* attributable to geometry
Run through the repo's own `require_no_high_wpm_regression_in_report`, baseline = SHIPPED's per-fold
seed-mean bucket rhos. ZERO and SEEDNOISE both FAIL **structurally** (azerty bucket 120, 3/3 seeds).
🟢 **But SEEDNOISE is the shipped geometry itself** — so this gate failure is produced by *reseeding
alone*. The correct inference is not "ZERO regresses high-wpm"; it is that **azerty's 120-wpm bucket is
not stable enough at 3 seeds to support a gated verdict.** Reported honestly rather than used to condemn
ZERO. SHIPPED and PLACEBO pass with scattered single-seed noise buckets only.

### 3.5 Registered conclusion for C
**DO NOT ADOPT.** Every candidate is measurable but **below the instrument's floor**, and the floor is
set not by my chosen bar but by the harness's own reseeding spread. This is the pre-registered
"measurable but below the instrument's floor ⇒ DO NOT ADOPT" branch, fired as written.

---

## 3b. The space axis (`row_offsets[0]`) — registered at `dd3223b` before any space number existed

**Why it exists:** space sits at `(0,0)`, `row_offsets` has keys `{1,2,3}`, so `.get(ay, 0.0)` supplies
**0.0 for space by dict omission — nobody ever chose it** (A1.5). 876 of 5854 eval cells (15.0%) touch it.

1-D dyadic scan over [−1, +1] step 0.125, letters pinned at shipped, 4 folds, seed 0, 816 s:

| off_space | −0.50 | −0.25 | −0.125 | **0.0 (shipped)** | **+0.125** | +0.25 | +0.50 | +0.75 | +1.00 |
|---|---|---|---|---|---|---|---|---|---|
| pooled wmae | 10.0260 | 9.9372 | 9.9220 | **9.8948** | **9.8736** | 9.9003 | 9.9251 | 9.8816 | 9.9231 |

- **argmin `off_space` = +0.125**, delta vs the accidental 0.0 = **−0.0212 ms/char = 16% of the bar** ⇒
  **DOES NOT MEET** it.
- Per-fold argmins (prereg S2 — all four, never pooled alone): azerty **+0.500** (−0.0702) · dvorak
  **+0.125** (−0.0331) · qwerty **+0.750** (−0.0742) · qwertz **+0.750** (−0.0929). **Again no two agree**,
  and the per-fold spreads (0.119–0.470) again exceed the whole 1-D surface spread (0.179).
- **CI (prereg S4): 13 of 17 grid points, spanning [−1.00, +1.00] = 2.00 key widths ⇒ NOT IDENTIFIED AT
  THIS SAMPLE SIZE.** The shipped implicit 0.0 **is inside the CI**.
- 🟡 Weak, honest positive: the scan is *mildly* consistent with a small **positive** space offset (every
  negative value is worse than 0.0; the four per-fold argmins are all positive, +0.125 to +0.75). That is
  a **direction**, not a value, and it is below the floor. **Registered as an open question, not a
  finding.** It is also physically plausible — a wide centred spacebar has no single column, and the
  right-thumb contact point sits right of centre for most typists — but I did not test that and do not
  claim it.

---

## 4. INVARIANT D — the mirror interaction  ✅ FINAL (re-scoped to descriptive by the parent)

**Scope note, stated because it changes the reading:** the T2 surface comes from the **pinned shipped k31
models**, which were *trained* under the shipped offsets. Substituting offsets at scoring time therefore
measures the model's **sensitivity surface**, not a refit. That is the right object for D's question
("would a mis-specified stagger MANUFACTURE apparent asymmetry?") but is not a claim about a refit model.

### 4.1 Baseline reproduced exactly — the parent's and `mirror`'s numbers are RIGHT
🟢 VERIFIED, independent third party (870 ordered distinct pairs, wpm 90, seed-mean over the 3 shipped
models): mean **1.9624** · median **0.0739** · p90 **5.8208** · max **42.2665** · >1 ms **238** ·
>5 ms **112** · 540 rows change under mirroring. All six match the brief to 4 dp.

### 4.2 The decisive result: mirror asymmetry is **provably blind to the SIGN of the stagger**
Algebra: `dx(a,b) = |Δx + Δoff|` and `dx(mir a, mir b) = |Δx − Δoff|`. Negating **all** offsets swaps
those two values, i.e. it maps `(a,b)` onto `(mir a, mir b)` featurewise.
🟢 VERIFIED at feature level: `max|feat(a,b | +off) − feat(mir a, mir b | −off)| = 0.000e+00`, **0 of 870
pairs differ**, at four distinct offset vectors (independently reproduced by the parent). At model level
the 870-asymmetry vector is **elementwise identical** under negation (mean and max agree to 1e-12).

### 4.3 The consequence: minimizing mirror asymmetry selects a **flat ortho board**
🟢 VERIFIED on a 9×9 dyadic surface of mean asymmetry: **argmin = 0.000000 at (top 0.0, bottom 0.0)**, and
the surface is **symmetric about the origin** (e.g. (+0.25,+0.25) ties (−0.25,−0.25) at 0.563). It is
monotone in |stagger|: scale 0 → **0.0000**, 0.125 → 1.2042, 0.25 → 1.2082, 0.5 → 1.3200, 0.75 → 1.4991,
**1.0 → 1.9624 (shipped)**, 1.5 → 2.4987, 2.0 → 2.9154. Rows-changed jumps 0 → 540 the instant any
stagger exists.

⇒ Mirror asymmetry is a stagger-**magnitude penalty**, not a fitting objective. Because it is an *even
function* of the stagger, **a fit against it could not even recover the SIGN of the ANSI stagger**, let
alone its magnitude. Zero stagger is physically wrong for a row-staggered ANSI board.

### 4.4 The placebo confirms A2's gauge result at the model level
A uniform +0.5 shift of all three rows gives mean asymmetry **1.9624 — identical to shipped to 4 dp**
(the letter-letter block is bit-identical). Note the 870-pair asymmetry universe excludes space, so it is
structurally blind to the space channel that same shift *does* move (60/61 pairs).

### 4.5 The one descriptive number at the fitted offsets (re-scoped ask, NOT a criterion)
Per the parent's re-scope, reported as a datapoint only:

| offsets | mean | median | p90 | max | >1 ms | >5 ms |
|---|---|---|---|---|---|---|
| SHIPPED (−0.25, 0.0, +0.50) | 1.9624 | 0.0739 | 5.8208 | 42.2665 | 238 | 112 |
| B-grid pooled argmin (+0.25, 0.0, 0.00) | **0.8621 (0.439×)** | 0.0000 | 2.1393 | 27.1186 | 110 | 62 |
| space-axis argmin (letters shipped, `off_space` +0.125) | 1.9624 (1.000×) | 0.0739 | 5.8208 | 42.2665 | 238 | 112 |

**It does move — down 0.439× — and that number must not be read as support for the B argmin.** It falls
for exactly the mechanism 4.2/4.3 proved: that argmin has `off_bottom = 0.00`, i.e. a *smaller-magnitude*
stagger (one row's offset deleted), and asymmetry is a monotone magnitude penalty. It is the *same*
non-result as the 0.439× "top only" row of my sweep. Since the B argmin fails C's bar and inverts the
physical sign, a shrink here is evidence about *magnitude*, not about correctness. **The space row is
identical to shipped by construction** — the 870-pair universe excludes space, so the statistic is
structurally blind to `off_space` (my A1.3/4.4 result).

### 4.6 ⇒ ANSWER TO D, and it refutes the hypothesis the arm was built on
**Stagger mis-specification is the MECHANISM of the sibling's asymmetry but cannot be a FIX.** 100% of
the asymmetry mass sits in the 540 stagger-changed rows (`mirror`'s result, which I reproduce), and there
is **no** stagger value that both reduces the asymmetry and remains physically ANSI — only zero does. The
asymmetry is the deterministic image of a **real physical asymmetry of the row-staggered board**, and it
should not be minimized away. My registered prediction D2 ("won't shrink unless the fit lands near zero
stagger") was directionally right; the sharper truth is that the objective always points at zero.

The parent has accepted this, retracted their H-stagger hypothesis, and registered it as MIRROR-SCOPE-1 /
STAGGER-D-1. Per their re-scope, the asymmetry at my fitted offsets is reported below as **one descriptive
number and explicitly NOT a success criterion** — pending B.

---

## 5. VERDICT

**NO — the hardcode does not need to change, and this is a positive result rather than a failure to find
one.** Evidence: the offsets are **not identified at this sample size** on either axis (CI spans 1.50 key
widths in both letter coordinates, 43/49 grid points; 2.00 key widths on the space axis, 13/17 points) and
**the shipped values lie inside both CIs**; no candidate anywhere reaches the pre-registered −0.135 ms/char
bar (best −0.0603 = 45%, B argmin −0.0377 = 28%, space +0.125 −0.0212 = 16%); the harness's **own reseeding
noise (sd 0.1219) is larger than every effect measured**, and a **zero-information placebo moved wmae by
+0.1142 — more than any real candidate**; the pooled argmin **inverts the physical ANSI stagger sign**; and
the per-fold argmins **scatter across the whole box**, with the thinnest fold (dvorak) selecting the
shipped values exactly.

### 5.1 Direct answers to the human's two questions
**"Why is row offset hardcoded like this?"** Because it encodes a *physical measurement* of the ANSI
board, not a learned parameter — and the values are right: top −0.25 / home 0.0 / bottom +0.50 is the
correct ANSI stagger direction and magnitude. What was never justified is subtler and is worth fixing in
*documentation*: (i) only **2** of the 3 letter numbers are free (a uniform shift is a gauge freedom, so
home = 0.0 is a *choice of origin*, not a measurement); and (ii) a **4th, invisible parameter** — space's
own offset — was set to 0.0 **by dict omission rather than by any decision**.

**"Is there a way to properly assign the best row_offsets for the learning of the model?"** Yes, and I
built it: profile held-out LOLO error over a dyadic grid, training *and* evaluating under each candidate
geometry (`agent-artifacts/drivers/inv_b.py`, `inv_s.py`). **The method works; the data cannot support the
answer.** At 4 layouts the fold-to-fold spread (0.19–0.47 ms/char) exceeds the entire offset surface
(0.149), so the fit resolves noise. This is a **resolution** limit, not a data-volume one: identifying
data is plentiful (3275 cross-row + 876 space-touching of 5854 cells). Converging it needs *more
layouts* (more folds), not more samples per layout.

### 5.2 What I would change (all documentation/robustness, none of it a value change)
1. **`validate()` must forward `geometry` to `train_fn`** (VALIDATE-GEOM-1) or reject a non-default
   geometry loudly. A documented parameter that silently applies to half the pipeline is worse than none:
   it produced a **+0.72 ms/char** artifact, 5× my bar. *Not landed — per parent instruction the answer,
   not the fix, is the deliverable.*
2. **Give `row_offsets` an explicit `0: 0.0` entry** (a pure no-op) so space's offset is a visible,
   reviewable choice instead of a `.get` default, and pin it in `tests/test_geometry.py`.
3. **Pin `ROW_STAGGERED_31.row_offsets`** — already an open ledger hole; the K31 offsets are what the
   *shipped* models were trained under, and nothing gates them today.
4. **Document the gauge**: home = 0.0 is an origin choice; only 2 letter parameters are identified; and
   `is_lsb`'s `dx > 1.5` threshold makes the gauge freedom exact only at **dyadic** offsets.
5. **Do not use mirror asymmetry as a stagger objective** — provably sign-blind, minimized by a flat
   ortho board.

### 5.3 Confidence
🟢 VERIFIED: A (all of §1, incl. the `lsb` flip and VALIDATE-GEOM-1), D (§4, incl. the sign-blindness
proof), the B and space surfaces and their CIs, the C arms and the gate. 🟡 HIGH: the reading that C's
gate failures reflect azerty-b120 instability rather than geometry (grounded in SEEDNOISE failing
identically, but 3 seeds is thin). 🟠 INFERRED: that more *layouts* rather than more samples would resolve
the fit. 🔴 UNCERTAIN / registered as an open question, not a finding: the weak hint of a small **positive**
space offset (all four per-fold argmins positive, every negative value worse than 0.0) — below the floor.

### 5.4 Where the artifacts are
- Report: `/local/home/zegertho/agent/state/stagger/report.md`
- Artifacts index: `/local/home/zegertho/agent/state/stagger/artifacts/profiles-and-artifacts-index.md`
- Pre-registration: `/tmp/stagger-wt/agent-artifacts/ROWOFFSETS-prereg.md` (`dd112d3`, addendum `dd3223b`)
- Drivers + results: `/tmp/stagger-wt/agent-artifacts/drivers/` on branch `stagger-rowoffsets`
- Nothing pushed, nothing merged to `main`, no layout adopted, `data/models/k31/` untouched.
