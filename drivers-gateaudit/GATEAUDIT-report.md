# gateaudit — decision audit of the calibration gate (branch `calib` @ c28b37e)

**ADOPT WITH A NAMED MODIFICATION: land the report exactly as written, but change the scope of
`passed` from every-slice to `{pooled, bucket_centered}` and drop the `band` default from
`calibration_report()`.** The gate's *instrument* is sound and the landing is clean (3 files, 260
insertions, **zero deletions**, suite 1290 passed / 3 skipped / 0 failed), but `passed = not
out_of_band` over all 7 slices is dominated by sampling noise in the thin high-wpm buckets — at the
repo's own r≈0.658 a **perfectly calibrated** 64-cell bucket lands outside [0.90, 1.10] **49% of the
time by chance alone** — while the two slices that carry real signal would have caught the actual
historical defect on their own.

> Status: COMPLETE. Nothing pushed, merged, or landed — the modification is coded and suite-green on
> a local branch in my own worktree for the human to review and land.

---

## 0. CORRECTIONS TO THE BRIEF — read these first

### 0.1 🟢 VERIFIED — `calibration_report()` **does** default `band`. The brief's central design claim is wrong as stated.

The brief says: *"`band` is a REQUIRED argument with NO default, and
CALIBRATION_SLOPE_RECOMMENDED_BAND = (0.90, 1.10) is offered to a human, not installed."* That is
true of `require_calibration()` and **false of `calibration_report()`**, which is the function
`validate()` actually calls:

```
verdicts.py:363  def calibration_report(
verdicts.py:367      band: tuple[float, float] = CALIBRATION_SLOPE_RECOMMENDED_BAND,   # <-- DEFAULTED
verdicts.py:422  def require_calibration(
verdicts.py:426      band: tuple[float, float],                                        # <-- required
```

And `validate()` (`validate.py:886-908`) passes `slopes`, `what`, and `support` — **no `band`**. So
every artifact the pipeline writes carries `"band": [0.9, 1.1]` and a hard `passed: false`, taken
from the constant. The ledger's own registration of this arm (line 12214) repeats the same wrong
claim: *"the band is a REQUIRED argument with NO default."*

This matters because it is the difference between the design the ledger registered and the code on
the branch. The number **is** installed on the only path production uses; it is absent only on the
path nothing calls (H2). The docstring on the constant even explains why installing it would be
wrong — *"a REPORTED RECOMMENDATION and NOT a default threshold"* — while the signature three lines
below installs it. 🟢 VERIFIED by reading both signatures and the sole call site; reproduced
empirically in every fold block of the parent's own artifact (`"band": [0.9, 1.1]`, no caller
supplying it).

### 0.2 🟢 VERIFIED — a THIRD ledger misreading, previously unreported, and it is the sentence that argues the surface is fine

The brief names two known misreadings (ledger:98, ledger:356). There is a third, and it is in
CALIB-1's **own** prereg — twice:

- `PREREGISTRATIONS.md:11919`: *"Measured in `agent-artifacts/results_bigram.json`: per-fold pooled
  **0.914–0.999** … By the repo's own metric and its own docstring, **the surface does not
  compress.**"*
- `PREREGISTRATIONS.md:12013`: *"since the shipped folds sit at **0.914–0.999**…"*

**MEASURED from that exact file** (`narrow` frame, 12 fold×seed cells, `g02_historical.py`):

| fold | pooled slope, per seed | mean |
|---|---|---|
| azerty | 0.9993 · 0.9957 · 0.9981 | 0.9977 |
| dvorak | 0.9232 · 0.9138 · 0.9196 | 0.9189 |
| qwertz | 0.9641 · 0.9636 · 0.9624 | 0.9634 |
| **qwerty** | **1.2356 · 1.2199 · 1.2295** | **1.2283** |

Full range over all 12 cells: **0.9138 … 1.2356**. Range **excluding qwerty**: **0.9138 … 0.9993** —
which reproduces the quoted "0.914–0.999" exactly. ⇒ **The quoted range is 3 of 4 folds presented as
all 4, with the one out-of-band fold omitted, in support of the conclusion "the surface does not
compress."** The `widened` frame behaves identically (qwerty 1.2268; others 0.9223–0.9950).

This is the same failure mode as ledger:98/:356, one layer deeper: not a number nobody read, but a
number read with the disconfirming fold dropped. It also **strengthens** the case for the gate — an
automatic per-fold emission cannot quietly omit a fold — which is why I report it under corrections
rather than as an argument against adoption.

### 0.3 🟢 VERIFIED — the brief's per-fold numbers are right, but they are a DIFFERENT slice from CALIB-1's registered numbers, and the two disagree

| fold | CALIB-1 registered (ledger:12092, :12208) | parent's e2e run (`e2e_full_slices.json`) | Δ |
|---|---|---|---|
| azerty | 1.0423 | 1.0158 | −0.027 |
| dvorak | 0.9248 | 0.9150 | −0.010 |
| qwertz | 1.0217 | 0.9997 | −0.022 |
| qwerty | **1.4067** | **1.3116** | **−0.095** |

Both were produced by the same `validate()` seams on the same data, so the gap is a seed/aggregation
choice rather than a defect — but **the headline 1.4067 is not reproducible as stated**, and the
qwerty figure is the single number this arm rests on. ⏳ My own LOLO resolves it in §(b).

### 0.4 🟢 VERIFIED — the brief's "1276 passed" is the *branch's* suite, 14 tests short because the branch deletes `test_los.py`

Branch `calib`: 1276 passed. My reconstructed tree (same gate, cherry-picked onto current
`origin/main`): **1290 passed / 3 skipped / 0 failed** (197.70 s, rc=0). The 14-test gap is the LOS
instrument the branch would revert. This is H1 quantified in the currency that matters.

### 0.5 🟠 INFERRED — H2 is real but is the *lesser* half of the enforcement problem

`require_calibration()` having no callers is true (verified: `git grep` finds it only in
`verdicts.py` and `tests/test_verdicts.py`). But it is **not** inert as the brief suggests, because
§0.1 shows `calibration_report()` already writes a hard `passed: false` into every artifact using
the defaulted band. So the landed state is not "reports without enforcing" — it is "**asserts a
failing verdict against a band no human chose**", which is worse than either intended option.

---

## (a) INVARIANT 1 — the estimand interrogation

*All figures below are 🟢 VERIFIED on constructed cases with known ground truth
(`g03_estimand.py`, `g04_estimand_fix.py`) — demonstrations, not assertions.*

### The target of 1.0 is not a matter of taste — confirmed to 1e-16

| r | slope(obs~pred) | slope(pred~obs) | product | r² | \|product − r²\| |
|---|---|---|---|---|---|
| 0.300 | 1.000000 | 0.086255 | 0.086255 | 0.086255 | 1.4e-17 |
| 0.500 | 1.000000 | 0.252457 | 0.252457 | 0.252457 | 5.6e-17 |
| **0.657889** | **1.000000** | **0.436812** | 0.436812 | 0.436812 | **1.1e-16** |
| 0.900 | 1.000000 | 0.811277 | 0.811277 | 0.811277 | 1.1e-16 |

The r=0.657889 row is the repo's own correlation and reproduces the parent's 0.296/1.4618 identity
from the other side. ⇒ **A low r² does not license a slope away from 1.** Choosing `slope(obs~pred)`
over `slope(pred~obs)` is correct and not arbitrary: only the former has a parameter-free target.

### What the gate MISSES

1. **Monotone nonlinearity — completely invisible.** A tanh-warped predictor rescaled to
   `slope(obs~pred) = 1.000000` (asserted to 1e-12) sits **inside** the band with **ρ = τ = +1.0**,
   while its local exchange rate is wrong by **1623×** between centre and edge (local slope 2.7266
   vs 0.00168; max absolute gap error 0.761 units; r² = 0.890). Fitness is a weighted sum, so this
   is exactly the defect class the gate exists to catch — and one OLS slope cannot see it. An
   isotonic or piecewise check is the instrument for this; the gate has none.
2. **Level / intercept — invisible by construction.** A uniform **+50 ms** offset leaves the slope
   at exactly 1.0. Irrelevant to rankings (affine-invariant), but it is precisely what
   ms-denominated published claims depend on.
3. **Scatter / r² — out of scope**, correctly (that is the ρ/ceiling instrument's job), but worth
   stating so the gate is not read as a general fit check.
4. **Opposite-direction slices, if pooled alone is gated** — see below.

### What the gate could FALSELY FLAG

1. **THIN-SLICE SAMPLING NOISE — the dominant effect, and it decides INVARIANT 2.** Floor MEASURED,
   not borrowed (20,000 trials/cell; ground-truth slope exactly 1; band [0.90, 1.10]; measured sd
   matches analytic `sd_e/√(n−2)` to 3 dp, which is the control on the simulation):

   | n_cells | P(false flag) r=0.5 | **r=0.658 (repo's own)** | r=0.8 | band half-width in sds (r=.658) |
   |---|---|---|---|---|
   | 12 | 85.1% | **77.2%** | 67.1% | 0.26 |
   | 20 | 80.7% | **70.6%** | 57.3% | 0.36 |
   | 40 | 71.9% | **59.0%** | 41.0% | 0.53 |
   | **64** | 65.1% | **49.2%** | 29.3% | **0.68** |
   | 100 | 56.8% | **38.8%** | 18.5% | 0.86 |
   | 200 | 41.6% | **22.1%** | 5.9% | 1.23 |
   | 400 | 25.0% | **8.0%** | 0.9% | 1.75 |
   | 900 | 8.4% | **0.9%** | 0.0% | 2.62 |

   ⇒ At the repo's own correlation, **[0.90, 1.10] is ±0.68 sd wide at n=64**. A gate at that width
   on that slice is close to a coin flip on a perfect surface.
2. **Errors-in-variables.** With ground-truth slope 1.0 and noise sd 0.4 on `pred`, OLS reads
   **0.8608** (false flag) while Deming (λ=1) reads **0.9662**. So a slope *below* 1 can be a
   property of the estimator. ⚠ **But** CALIB-1 already measured the raw side's split-half
   reliability at **0.9860** with noise sd only 4.8 ms (ledger:12080) — the target is *not* noisy, so
   errors-in-variables owns little here and **OLS is the right functional for this data**. Deming is
   the wrong recommendation; I checked before proposing it and am **not** proposing it.

### Is `bucket_centered` the right slice to privilege? — YES, and it is measurable, not a judgement

Constructed case: 5 wpm buckets with a realistic ramp (means 55–120 ms), true **within-bucket
compression of exactly 1.45×** in every bucket:

| slice | reading | in band? |
|---|---|---|
| pooled | **1.0176** | **✅ PASSES (blind)** |
| bucket_centered | **1.4500** | ❌ fails (correct) |
| each of the 5 buckets | 1.4500 | ❌ fails (correct) |

Masking scales with the ramp exactly as the mechanism predicts:

| between-bucket ramp sd | pooled | bucket_centered |
|---|---|---|
| 0 ms | 1.4500 | 1.4500 |
| 3 ms | 1.2718 | 1.4500 |
| 6 ms | 1.1260 | 1.4500 |
| 12 ms | 1.0378 | 1.4500 |
| 25 ms | 1.0107 | 1.4500 |
| 50 ms | 1.0024 | 1.4500 |

A grid search additionally found **10,322** (k₁,k₂) factor pairs where pooled is inside [0.90, 1.10]
while **both** buckets are outside (best: buckets 0.8352 / 2.6105, pooled 1.0022). ⇒ **A pooled-only
gate is demonstrably blind to real, large, within-bucket compression.** This validates the branch's
stated reason for the `bucket_centered` slice and rules out "pooled-only" on measured grounds.

### Verdict on INVARIANT 1

`slope(obs~pred)` is the **right estimand**, OLS is the **right functional for this data** (the
target is not noisy: reliability 0.9860), and 1.0 is the right target. Two slices are needed, not
one: **`pooled`** (what ms-denominated claims rest on) and **`bucket_centered`** (what the search
consumes). The gate's real blind spot is **monotone nonlinearity**, which no slope test can see; that
is a separate future instrument (isotonic / piecewise), not a reason to reject this one.

---

## (b) INVARIANT 2 — the scope comparison, MEASURED on my own LOLO

Independent LOLO on the reconstructed tree, 3 seeds × 4 folds, config identical to the campaign's
(seeds [0,1,2], wpm [40,140) ×20, cell floor 10, n_boot 50, 2202 rows from
`/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv`). Provenance printed and asserted:
`keybo.__file__ = /local/home/zegertho/repos/keybo-wt-gateaudit/src/keybo/__init__.py`.

**A fold passes iff ALL THREE seeds pass** — the campaign's own "conclusions must hold across seeds"
convention.

| scope | # folds passing | passing | failing |
|---|---|---|---|
| **every_slice (as written on the branch)** | **0 / 4** | — | azerty, dvorak, qwerty, qwertz |
| buckets_only | 0 / 4 | — | all four |
| support_gated_n100 | 0 / 4 | — | all four |
| support_gated_n200 | 1 / 4 | qwertz | azerty, dvorak, qwerty |
| bucket_centered_only (**the brief's hypothesis**) | **3 / 4** | azerty, dvorak, qwertz | **qwerty** |
| pooled_only | 3 / 4 | azerty, dvorak, qwertz | qwerty |
| **structural_pair {pooled, bucket_centered}** (my candidate) | **3 / 4** | azerty, dvorak, qwertz | **qwerty** |
| support_gated_n400 | 3 / 4 | azerty, dvorak, qwertz | qwerty |

🟢 **The brief's hypothesis is CONFIRMED on the which-folds-pass axis** — `bucket_centered`-only
passes 3/4 and flags only qwerty, exactly as predicted. I recommend the *pair* rather than
`bucket_centered` alone only because pooled costs nothing here (it agrees on all 4 folds) and it is
the slice that ms-denominated claims actually rest on, so gating it keeps those claims covered.

### The support behind every failing bucket, with the noise it must clear

*`n_cells` is identical across seeds; seed 0 shown. "P(false flag)" is interpolated from the measured
floor at that n. "dev/sd" is |slope − 1| in units of the analytic sampling sd at that n.*

| fold / slice | slope | n_cells | n_participants | in band | P(false flag) | **dev / sd** |
|---|---|---|---|---|---|---|
| azerty/pooled | 0.9993 | 1001 | 166 | ✅ | 0.9% | 0.02 |
| azerty/bucket_centered | 1.0405 | *(1001)* | *(166)* | ✅ | — | — |
| azerty/bucket_40 | 1.1622 | 282 | 143 | ❌ | 15.1% | 2.37 |
| azerty/bucket_60 | 1.0425 | 280 | 157 | ✅ | 15.3% | 0.62 |
| azerty/bucket_80 | 0.8434 | 222 | 98 | ❌ | 20.0% | 2.03 |
| azerty/bucket_100 | 0.8182 | 153 | 51 | ❌ | 28.5% | 1.95 |
| **azerty/bucket_120** | **0.7226** | **64** | **23** | ❌ | **49.1%** | **1.91** |
| dvorak/pooled | 0.9232 | 799 | 64 | ✅ | 1.9% | 1.89 |
| dvorak/bucket_centered | 0.9324 | *(799)* | *(64)* | ✅ | — | — |
| dvorak/bucket_40 | 1.0284 | 170 | 40 | ✅ | 26.0% | 0.32 |
| dvorak/bucket_60 | 0.8774 | 212 | 59 | ❌ | 20.9% | 1.55 |
| dvorak/bucket_80 | 0.8920 | 193 | 51 | ❌ | 22.9% | 1.30 |
| dvorak/bucket_100 | 0.9071 | 145 | 34 | ✅ | 29.8% | 0.97 |
| **dvorak/bucket_120** | **0.8259** | **79** | **19** | ❌ | **44.2%** | **1.33** |
| qwertz/pooled | 0.9641 | 1406 | 485 | ✅ | 0.9% | 1.18 |
| qwertz/bucket_centered | 1.0229 | *(1406)* | *(485)* | ✅ | — | — |
| qwertz/bucket_40 | 1.0627 | 366 | 413 | ✅ | 9.8% | 1.04 |
| qwertz/bucket_60 | 0.9939 | 367 | 457 | ✅ | 9.8% | 0.10 |
| qwertz/bucket_80 | 0.9875 | 323 | 314 | ✅ | 12.4% | 0.20 |
| qwertz/bucket_100 | 1.0045 | 242 | 161 | ✅ | 18.2% | 0.06 |
| **qwertz/bucket_120** | **0.7625** | **108** | **54** | ❌ | **36.9%** | **2.14** |
| **qwerty/pooled** | **1.2356** | **2648** | 54689 | ❌ | **0.9%** | **10.58** |
| **qwerty/bucket_centered** | **1.4145** | *(2648)* | *(54689)* | ❌ | — | — |
| **qwerty/bucket_40** | **1.6060** | **541** | 43467 | ❌ | 5.4% | **12.29** |
| **qwerty/bucket_60** | **1.3405** | **555** | 51224 | ❌ | 5.1% | **6.99** |
| **qwerty/bucket_80** | **1.2897** | **548** | 40472 | ❌ | 5.2% | **5.91** |
| **qwerty/bucket_100** | **1.2475** | **527** | 24015 | ❌ | 5.6% | **4.95** |
| **qwerty/bucket_120** | **1.2146** | **477** | 10811 | ❌ | 6.5% | **4.09** |

### 🟢 THE DECIDING MEASUREMENT — the two failure populations do not overlap

| population | slices | deviation in sds | n_cells | P(false flag) |
|---|---|---|---|---|
| non-qwerty out-of-band | 8 | **1.30 – 2.37** | 64 – 366 | 9.8% – 49.1% |
| qwerty out-of-band | 6 | **4.09 – 12.29** | 477 – 2648 | 0.9% – 6.5% |

**max non-qwerty 2.37 sd < min qwerty 4.09 sd — a clean separation with no overlap.** The 3-of-4
"failures" the branch's every-slice rule produces are indistinguishable from sampling noise at the
measured floor; qwerty's are 4–12 sd events on well-supported slices. This is what makes the scope
change a *correction* rather than a weakening.

### The false-flag budget over the campaign's 12 fold×seed cells

| scope | cells failing (of 12) | **expected noise-only slice flags** |
|---|---|---|
| **every_slice** | **12 / 12** | **11.74** |
| support_gated_n400 | 3 / 12 | 0.97 |
| **structural_pair** | **3 / 12** | **0.14** |
| pooled_only | 3 / 12 | 0.14 |
| bucket_centered_only | 3 / 12 | 0.00 |

A gate that fails 12 of 12 cells, ~99% of it attributable to noise, is a gate its readers will learn
to ignore. That is the GATESUPPORT-1 hazard turned up to maximum: the azerty b120 cell (**64 cells /
23 participants**) the brief names as having *already* single-handedly decided four arms is the same
cell here, and at n=64 it is a coin flip.

### Two gaps in the every-slice rule worth naming

1. 🟡 **`bucket_centered` ships with NO support recorded.** `validate.py` builds the support map for
   `pooled` and each bucket but omits `bucket_centered` — so the slice I recommend gating on arrives
   with `support` absent. It spans the same cells as `pooled` (centering removes bucket means, it
   drops no cell), so the fix is two lines. Values in *(parentheses)* above are mine, not the gate's.
2. 🟡 **`support` is collected and provably never consulted.** That is deliberate per GATESUPPORT-1,
   and I am **not** proposing a support floor — a minimum-n would retroactively decide which past
   verdicts stand, exactly the objection GATESUPPORT-1 raises. Note the measured consequence:
   `support_gated_n400` reaches the same 3/4 verdict as `structural_pair`, so **the support route is
   not needed** — scoping to the structural slices achieves it without inventing a floor.

---

## (c) INVARIANT 3 — the retroactive-force price, per configuration

The brief's key question, answered directly: **for every configuration, adopting this gate reopens
NO ranking-based conclusion, because none of them depends on the slope.** This is not my inference —
it is CALIB-1's registered and verified blast-radius result, which I use rather than re-derive:
ms/char is linear in the tables, so orderings are invariant under any positive affine map, verified
across 39 equal-coverage board pairs to **2.164e-12**.

| configuration | folds failed | what it reopens |
|---|---|---|
| **every_slice** (as written) | **all 4** | Nothing *rank*-based. But it labels the **entire shipped surface** failing on all four folds, 3 of them on noise — so it reopens **nothing substantive while impeaching everything**, the worst of both. |
| bucket_centered_only | qwerty | The qwerty-fold ms magnitudes only. |
| pooled_only | qwerty | Same. |
| **structural_pair** | **qwerty** | Same — see the itemised list below. |
| support_gated_n400 | qwerty | Same, plus it implies a floor whose value is a new unregistered decision. |

**What a qwerty-fold failure DOES overturn (the same list under all three 3/4 configurations):**

- 🟢 **NOT overturned — `candidate`'s survival**: rank 3/13 with zero losses under all four pricings,
  **0 rank changes field-wide** (CALIB-1, 39 pairs, 2.164e-12).
- 🟢 **NOT overturned — all 78 TOURNAMENT-1 pair verdicts**, the 5-board cluster equivalence,
  "qwerty is slowest", and every ρ/τ-based arm verdict including LATSPAN-1's nine nulls.
- 🔴 **SENSITIVE — the qwerty-vs-field gap in percent**: 3.6845% → 5.6001% (absolute 9.7147 →
  14.2013 ms). ⚠ Direction is **favourable**: the correction makes qwerty look *worse*, so published
  ~3.4–3.7% figures are **lower bounds** and no conclusion flips.
- 🔴 **SENSITIVE — every "X beats Y by N ms/char" figure** (scales by exactly the affine factor).
- 🔴 **SENSITIVE — PRICEBAND-1's ms sfb price**: its *location* in sfb units is unchanged
  (`sf_share` is a pure corpus/board quantity) but the **ms price** is understated by 1.05–1.46×.

⇒ 🟢 **ADOPTION IS CHEAP.** Every load-bearing *verdict* in this campaign is rank- or sign-based and
therefore untouched; the only casualties are ms- and percent-denominated *magnitudes*, and the one
measured casualty moves in the direction that makes the published claim conservative. **The honest
answer the brief asked for — "adopting this reopens nothing because every load-bearing conclusion is
rank-based" — is very nearly right.** The precise version: it reopens no verdict, and re-prices a
known, already-registered set of magnitudes whose direction is favourable.

One genuine asymmetry, and it is an argument *for* the scope change: `every_slice` prices in the
*appearance* of a surface-wide failure. Anyone reading four consecutive `passed: false` folds would
reasonably conclude the instrument is broken, when three of those failures are noise on thin buckets.

---

## (d) INVARIANT 4 — would it have caught the thing it exists for?

### Q1: the qwerty-fold compression — 🟢 YES, and with numbers that already existed

Tested against the **shipped historical artifact** `agent-artifacts/results_bigram.json` (dated
2026-07-31, on disk during the campaign). That artifact predates the branch and therefore has **no
`bucket_centered` key** — so the honest counterfactual uses only `pooled` and the per-bucket slopes:

| scope available at the time | azerty | dvorak | qwertz | qwerty |
|---|---|---|---|---|
| **pooled_only** | ✅ pass | ✅ pass | ✅ pass | **❌ FAIL** |
| every_available_slice | ❌ | ❌ | ❌ | ❌ |
| buckets_only | ❌ | ❌ | ❌ | ❌ |
| support_gated_n400 | ✅ | ✅ | ✅ | **❌ FAIL** |

qwerty's pooled slope in that artifact: **1.2356 / 1.2199 / 1.2295** — out of band on all three
seeds. ⇒ **A `pooled`-scoped gate would have flagged exactly the right fold, and only that fold,
from data that was already on disk.** The every-slice rule would have flagged all four and told the
reader nothing. This is the strongest single argument for adoption *and* for the scope change: the
gate justifies itself, but only at the narrower scope.

### Q2: the ledger's misreadings — 🟢 YES for two of the three, structurally

- **ledger:11919 / :12013** (*"pooled 0.914–0.999 … the surface does not compress"*): **YES,
  decisively.** That sentence is only writable if the qwerty fold is omitted. An always-emitted
  per-fold gate block makes fold-omission structurally visible — you cannot quote a 3-of-4 range when
  the 4th fold's block sits in the same artifact carrying `passed: false`.
- **ledger:356** (*"calibration slopes ~1.0 per fold"*): **YES.** False of 1 of 4 folds (pooled
  1.2283, bucket_centered 1.4067), and that fold's block would say so.
- **ledger:98** (*"calibration slope 1.04 on qwerty (no compression)"*): 🟠 **NOT DIRECTLY.** This is
  a **trigram** claim (`runs/lolo_trigram_v1.json`) and I did not re-run the trigram LOLO, so I do
  not know the trigram slope and will not guess. The gate would apply there identically, but I cannot
  claim it would have fired.

⇒ The gate catches the class of error that produced all three: a prose summary that diverges from the
per-fold numbers. It does not *prevent* misreading, but it removes the ability to do it silently.

---

## (e) INVARIANT 5 — the minimal auditable landing plan, with the deletions audit

### H1 quantified: what a naive merge of `calib` would revert

`git diff --stat origin/main..calib` — the branch is **5 commits** off a merge-base of `b87beb9`,
while `origin/main` is now at `8701c00`:

| file | effect of a naive merge | status |
|---|---|---|
| `src/keybo/analysis/los.py` | **−405 lines (DELETED)** | shipped at `1a10450` |
| `tests/analysis/test_los.py` | **−228 lines (DELETED)** | shipped at `1a10450` |
| `drivers-los/` (4 files) | −632 lines | shipped |
| `drivers-losvar/` (4 files) | −929 lines | shipped |
| `src/keybo/training/train.py` | −33/+? — reverts the `8701c00` docstring correction | just landed |
| `src/keybo/layouts.py` | **+25 — reintroduces the colemak-dh pin** | ⚠ **user dropped this deliberately** |
| `tests/test_layout.py` | +41 — the pin's tests | ⚠ same |

Total: **1595 insertions, 9513 deletions.** Three separate hazards, not one: the LOS instrument, the
`train.py` docstring correction that is `origin/main`'s current HEAD commit, and the colemak-dh pin
the user explicitly decided to drop. ⇒ 🔴 **`calib` must not be merged. Cherry-pick only.**

### The minimal landing: 3 files from ONE commit

The gate lives entirely in `9b22a91`. Of that commit's 9 files, 6 are `drivers-calib/` analysis
scaffolding with **no gate dependency** (verified: nothing in `drivers-calib/` references
`calibration_report`, `require_calibration`, or `calibration_gate`). The minimal landing is:

```bash
git checkout -b <branch> origin/main
git checkout calib -- src/keybo/verdicts.py src/keybo/training/validate.py tests/test_verdicts.py
```

**Deletions audit on the reconstructed tree — 🟢 CLEAN:**

```
$ git diff --diff-filter=D --name-only origin/main
(empty — ZERO files deleted)

$ git diff origin/main --stat
 src/keybo/training/validate.py |  30 +++++++++-
 src/keybo/verdicts.py          | 133 +++++++++++++++++++++++++++++++++++++++++
 tests/test_verdicts.py         |  99 ++++++++++++++++++++++++++++
 3 files changed, 260 insertions(+), 2 deletions(-)
```

The 2 deletions are **both inside one comment** in `validate.py` (the TAUGATE-1 note, rewritten to
mention calibration) — 🟢 verified line-by-line, no code removed.

- 🟢 `src/keybo/analysis/los.py` — **untouched, 405 lines**
- 🟢 `tests/analysis/test_los.py` — **untouched, 228 lines**
- 🟢 `src/keybo/layouts.py` — **NOT in the diff.** The colemak-dh pin is not reintroduced; the
  user's decision stands.
- 🟢 `src/keybo/training/train.py` — **NOT in the diff.** The `8701c00` docstring correction survives.
- 🟢 `drivers-los/`, `drivers-losvar/` — untouched.

**Full suite on the reconstructed tree: 🟢 1290 passed / 3 skipped / 0 failed, rc=0, 197.70 s**
(`artifacts/suite-reconstructed-tree.log`). Note **1290**, not the branch's 1276 — the 14-test
difference is the LOS suite the branch would have deleted.

### The modification, implemented and tested

I coded the recommended change rather than only describing it (branch `gateaudit-proposal` in my own
worktree — **not pushed, not landed**):

1. `calibration_report(band=None)` — **the default removed.** No band ⇒ `gated: False`,
   `passed: None`, `band: None`, **and every slope still reported**. The number is the useful part;
   the adjudication is what needs a human.
2. `deciding: Sequence[str] | None` on both functions + a new
   `CALIBRATION_DECIDING_SLICES_RECOMMENDED = ("pooled", "bucket_centered")` constant carrying the
   measured justification in its docstring.
3. `out_of_band` still lists **every** out-of-band slice regardless of scope; the difference lands in
   the new `out_of_band_advisory`. **Narrowing the scope cannot hide a slice** — otherwise `deciding`
   would itself become a way to silence an inconvenient verdict, the exact failure the gate exists
   against.
4. `validate(calibration_band=None, calibration_deciding=...)` — the band now arrives from the caller.
5. `bucket_centered` support recorded (gap 1 above, fixed).

**Full suite on the proposal: 🟢 1295 passed / 3 skipped / 0 failed, rc=0, 174.08 s**
(`artifacts/suite-proposal.log`). Two tests initially failed and **both were the intended behaviour
changes, not bugs** — I checked which failure before touching either: `calibration_report(...)` with
no band now returns `gated: False` (the whole point), and the raise message now says "1 of 2
**DECIDING** slices". Both updated, plus 5 new tests including one asserting a narrowed scope
**cannot silence** a slice.

**End-to-end through `validate()` on real data (3 arms, 8/8 checks pass** —
`artifacts/g07_e2e_proposal.json`):

| arm | azerty | dvorak | qwertz | qwerty |
|---|---|---|---|---|
| 1. `band=None` (new default) | `None` | `None` | `None` | `None` |
| 2. band (0.90,1.10) + recommended scope | ✅ True | ✅ True | ✅ True | **❌ False** |
| 3. band (0.90,1.10) + every slice (the branch) | ❌ False | ❌ False | ❌ False | ❌ False |

Checks verified: arm 1 still reports **all 7 slopes** per fold (the measurement is not lost) with
`gated: False` / `band: None`; arm 2 fails qwerty, passes the other three, **still reports** their
out-of-band thin buckets and relegates them to `out_of_band_advisory`; arm 3 **reproduces the branch
exactly**, which is what makes the comparison in §(b) apples-to-apples; and `bucket_centered` support
is now recorded on every fold.

---

## (f) INVARIANT 6 — the recommendation

# ADOPT WITH MODIFICATION

**Land the gate. Do not land it as written.**

| | value | MEASURED or JUDGEMENT |
|---|---|---|
| **Band** | **(0.90, 1.10)** | **MEASURED as appropriate for the recommended scope.** At pooled n=799–2648 a 5%-false-flag band is [0.921,1.079]…[0.956,1.044], so 0.10 half-width is correct-to-slightly-conservative there. The *target* 1.0 is MEASURED (identity to 1e-16). The exact width remains the human's call — but it is no longer an arbitrary one. |
| **Scope** | **`{pooled, bucket_centered}`** decides; every other slice reported as advisory | **MEASURED.** Thin-bucket failures are 1.30–2.37 sd against a 9.8–49.1% false-flag floor; qwerty's are 4.09–12.29 sd. No overlap. |
| **Band default** | **remove it** from `calibration_report()` | **MEASURED defect** (§0.1): the current default writes `passed: false` into every artifact against a band no human chose. |
| **Enforcement (H2)** | **leave `require_calibration()` uncalled for now** | **JUDGEMENT.** See below. |

**Expected effect of adopting exactly this:** the shipped surface passes on azerty, dvorak, qwertz and
**fails on qwerty** — 🟢 verified end-to-end, not predicted. That is the correct outcome: it is the
one fold where a real 4–12 sd magnitude defect exists, it is already registered (CALIB-1), and it
reopens **no rank-based conclusion**.

**On H2 (no production callers) — JUDGEMENT: correct-by-design *for now*, but only because of the
band fix.** The brief frames this as "report now, enforce when a human picks a band" vs "a gate nobody
calls is TAUGATE-1 in a new hat". With the defaulted band, it was neither — it was silently asserting
`passed: false`. Once the band arrives from the caller, `passed` becomes a genuine tri-state
(`None` = un-adjudicated, `True`/`False` = adjudicated against a registered band) and reporting is
*sufficient*, because a `passed: false` in an artifact is now a real, readable verdict. **Wire
`require_calibration()` into the model-adoption path only after the human registers a band** — and
note it would then refuse the incumbent surface on qwerty, which is exactly the GATESUPPORT-1
"the gate refuses the incumbent" situation the brief flags. That is a decision to make deliberately,
not a side effect of landing a report.

**What I did NOT recommend, and why (each checked before discarding):**

- **Deming/TLS instead of OLS** — considered and **rejected on measurement**: the raw side's
  split-half reliability is 0.9860 (noise sd 4.8 ms), so errors-in-variables owns almost nothing and
  OLS is right for this data.
- **A support floor (minimum n)** — **rejected on the GATESUPPORT-1 precedent**, and it is
  *unnecessary*: `support_gated_n400` reaches the identical 3/4 verdict, so scoping achieves the same
  end without inventing a threshold that retroactively decides which past verdicts stand.
- **`bucket_centered` alone** (the brief's hypothesis) — **confirmed correct on which-folds-pass**;
  I add `pooled` only because it is free here (agrees on all 4 folds) and it covers the
  ms-denominated claims. If the user prefers the brief's narrower version, it costs nothing measured.
- **A wider band for thin buckets** — arithmetically possible ([0.715,1.285] at n=64) but it makes the
  band n-dependent, which is a much larger design change for no measured gain.

---

## (g) What would change my mind

| Evidence | Which way it moves me |
|---|---|
| A **trigram** LOLO showing the trigram surface's pooled/centered slopes badly out of band | Strengthens adoption *and* would let me answer ledger:98, which I currently cannot. **This is the top open item.** |
| A demonstration that the optimizer's *reachable* search space is affected by within-bucket compression at magnitudes the qwerty fold shows | Would push me from "adopt" to "adopt **and** wire `require_calibration()` immediately" — the compression would then be a live defect, not a bookkeeping one. |
| A thin bucket whose slope deviation exceeds ~4 sd (i.e. crosses into the qwerty population) | Would make me add that bucket to the deciding scope — the separation I rely on is empirical, not structural, and it could stop holding on new data. |
| Evidence that the raw side IS noisy (reliability well below 0.9860) at the slice level | Would move me to Deming/TLS, which I currently reject. |
| A published claim that depends on a thin high-wpm bucket's *magnitude* | Would argue for gating buckets after all, accepting the false-flag cost. |
| The user deciding that a gate refusing the incumbent is unacceptable | Would move me to "adopt at report-only, never enforce" — still worth landing for the fold-omission property alone (§d/Q2). |

**What would NOT change my mind:** an argument that "3 of 4 folds failing proves the surface is
broken". That is the noise floor, and it is measured.

---

## (h) Negative controls

**Two full controls plus one internal, all 🟢 PASSED** — and one of them *caught a real error*, which
is what a control is for.

1. **Reproduce a published quantity — the CALIB-1 per-fold registration.** My independent LOLO
   reproduces all four registered bucket-centered slopes to **|diff| = 0.0000**:

   | fold | my per-seed | my mean | ledger | \|diff\| |
   |---|---|---|---|---|
   | azerty | 1.0405 · 1.0449 · 1.0415 | **1.0423** | 1.0423 | **0.0000** |
   | dvorak | 0.9324 · 0.9130 · 0.9289 | **0.9248** | 0.9248 | **0.0000** |
   | qwertz | 1.0229 · 1.0212 · 1.0210 | **1.0217** | 1.0217 | **0.0000** |
   | qwerty | 1.4145 · 1.3945 · 1.4112 | **1.4067** | 1.4067 | **0.0000** |

   My pooled values also match the shipped `results_bigram.json` per-seed values exactly. This is
   what licenses §0.3's correction: the ledger is right, the parent's e2e run is the outlier.
2. **The attenuation identity, as an independent check on the estimand.** An MSE-optimal predictor
   yields `slope(obs~pred) = 1.000000` at r = 0.3 / 0.5 / 0.657889 / 0.9, with
   `slope_fwd × slope_rev = r²` to **1.1e-16**. Confirms the brief's identity from the other side.
3. **Internal control on the simulation** (a control on the control): the measured slope sd across
   20,000 trials matches the analytic `sd_e/√(n−2)` to 3 decimals at every (r, n) cell — so the
   false-flag floor is not a simulation artifact.

**The control that fired:** my first negative-control run reported `passes: false` — measured max
1.2355 vs the ledger's stated 0.999. Rather than adjusting my reader, I checked, and the *ledger* was
wrong (§0.2). Two further errors of mine were caught the same way and fixed before use: a rescale
inverted in probe B (I divided where I should have multiplied, giving b² = 1.536 instead of 1.0), and
an assumption that mirror factors cancel in a pooled slope — they don't, because pooling is
variance-weighted, so I solved for the hiding pair instead.

---

## (i) What remains open

1. 🔴 **The trigram surface is unmeasured.** ledger:98's "slope 1.04 on qwerty (no compression)" is a
   *trigram* claim and I did not run the trigram LOLO. I do not know the trigram slope. **This is the
   highest-value follow-up** (~10 min: same driver, `ngram="trigram"`), and it is the one place I
   cannot say whether the gate would have caught the misreading.
2. 🟡 **Monotone nonlinearity has no instrument.** Probe B shows a surface can sit at slope exactly
   1.0 with a 1623× local-exchange-rate error. An isotonic or piecewise-slope check would close it;
   nothing in the repo does today. Not a blocker — a named gap.
3. 🟡 **The 2.37-vs-4.09 sd separation is empirical, not structural.** It holds cleanly on this data.
   New data (a 5th layout, more high-wpm participants) could produce a genuine 4 sd thin-bucket
   failure, which my scope would miss. The mitigation is that such a slice is still *reported* in
   `out_of_band`, so it is visible — but it would not gate.
4. 🟠 **Why qwerty compresses is unresolved** — CALIB-1 found corr(slope, b-exposure) = −0.869 vs
   corr(slope, train/test ratio) = −0.892, only 0.023 apart and unresolvable at n=4 folds. Adopting
   the gate does not resolve it; it makes it visible on every run.
5. 🟡 **The parent's e2e-vs-ledger 0.095 discrepancy on qwerty** is unexplained beyond
   "seed/aggregation choice". My run matches the ledger, so I did not pursue it — but something in
   that run differed and the cause is not identified.
6. 🟢 **Not open, stated for completeness:** I did **not** push, merge, land, or delete anything.
   All work is on `gateaudit-audit` / `gateaudit-proposal` in
   `/local/home/zegertho/repos/keybo-wt-gateaudit`, unpushed. `src/keybo/layouts.py` untouched;
   `data/models/k31/` never written.

---

## Artifacts

All under `/local/home/zegertho/agent/state/gateaudit/artifacts/` (292 K, pointers-only compliant),
and committed in-repo at `drivers-gateaudit/` on branch `gateaudit-audit` (commits `f230c4d`,
`e776614`, `4288f45`) so they survive a workspace loss:

| file | what |
|---|---|
| `/local/home/zegertho/agent/state/gateaudit/artifacts/g01_scope.json` | my LOLO, 3 seeds × 4 folds, every gate block verbatim **with support** |
| `/local/home/zegertho/agent/state/gateaudit/artifacts/g02_historical.json` | historical replay + negative control #1 |
| `/local/home/zegertho/agent/state/gateaudit/artifacts/g03_estimand.json` | identity control, misses, **the measured false-flag floor** |
| `/local/home/zegertho/agent/state/gateaudit/artifacts/g04_estimand_fix.json` | corrected probes B/C + the pooled-blindness demonstration |
| `/local/home/zegertho/agent/state/gateaudit/artifacts/g05_scope_table.json` | the scope comparison + support/noise table |
| `/local/home/zegertho/agent/state/gateaudit/artifacts/g06_invariant4.json` | would-it-have-caught-it + the false-flag budget |
| `/local/home/zegertho/agent/state/gateaudit/artifacts/g07_e2e_proposal.json` | end-to-end proposal verification, 8/8 checks |
| `/local/home/zegertho/agent/state/gateaudit/artifacts/suite-reconstructed-tree.log` | 1290 passed / 3 skipped / 0 failed |
| `/local/home/zegertho/agent/state/gateaudit/artifacts/suite-proposal.log` | 1295 passed / 3 skipped / 0 failed |

Plus every driver (`g01`–`g07`) alongside its output.
