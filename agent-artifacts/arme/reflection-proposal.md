# Reflection proposal — arme (ARM E)

**For the parent to register.** I do not edit `PREREGISTRATIONS.md` or the shared knowledge base
(per my scope). These are drafts; the parent decides what lands.

---

## A. Proposed TOOLING-TRAPS entries (6 candidates, most valuable first)

### 1. `LossCurve.price_many` is batch-shape-dependent — and its tests structurally cannot see it
⚠ **Count corrected by A10 below: it is 14 of 14 curves on a level GRID, not the "7 of 14" written
here (nor the parent's 9/14) — both were single-sample probes. The defect is per-LEVEL. Fixed by the
parent in `79cb175`, and I verified the fix leaves arm E bit-identical.** Entry otherwise stands.

**Verified 🟢 (arm E gate 1, first run).** `cf5f731` added `price_many` to be *the* validated
vectorized path. It is **not bit-exact with `LossCurve.price` for n≥2, and not bit-exact with itself
across batch shapes**: `price` evaluates a length-1 array while `price_many` evaluates length n, and
`_design(...) @ coeffs` dispatches to a different BLAS kernel by shape. Identical design rows,
different product — `comfort` at its `lo`: `0.069389400121559` (n=1) vs `0.06938940012155903`
(n≥2). **7 of 14 archive curves** show it; arm D's elementwise `ClampedCurve` shows it on **0 of
14**.

Consequences: (a) an instruction to "pin your fast path against `price_many` at EXACT float
equality" is **unsatisfiable by construction**; (b) the 4 shipped tests pass because
`test_price_many_matches_price_exactly_under_every_policy` uses one fixed 8-element array on *both*
sides and the saturation test uses `pytest.approx` — same family as traps 28/31, the check cannot
fail in the way that matters.

**Fix** (unverified 🟡 — I did not change shipped code, per scope): compute the
linear/quadratic/hinge form **elementwise** in `price_many`, as `armd_obj.ClampedCurve._raw` does,
instead of `design @ coeffs`. Then exact equality becomes attainable and the tests should assert it
across **≥3 batch shapes** (1, 2, large n), not one fixed-length array.

### 2. ULP is the wrong metric for "same function, different association order"
**Verified 🟢.** ULP measures spacing at the *result's* magnitude, so a value arising by
cancellation sits absurdly far away while being an equally-correct rounding: `sr-roll` at level
13.6116 gives `0.015004423663657684` vs `0.015004423663658572` — an **8.9e-16** absolute difference
between terms of order 1, which is **512 ULP**. A ULP tolerance is therefore simultaneously too
tight (flags correct arithmetic near a zero-crossing) and, at large magnitudes, too loose.

**Use the dot-product rounding bound** `n · eps · Σ|term|` — it scales with the TERMS, which is what
cancellation makes large relative to the result — **plus the induced ORDERING**, since an optimizer
only ever compares scores. Arm E's gate 1 does both: worst bound ratio 0.017, identical argmin and
identical full argsort over 2061 layouts.

### 3. A rounding-tolerant check has a mathematically necessary blind spot — publish its floor
**Verified 🟢.** My mutation control initially "failed" on a **+1 ULP coefficient** perturbation,
and that failure was *correct*: a perturbation smaller than the arithmetic's own rounding is
indistinguishable from a different rounding of the same function, by any test whatsoever. Asserting
it must be caught is asserting the impossible.

So: assert the mutations that ARE detectable (knot 1e-6 → 1.3e8×, domain 1% → 4.6e11×, no-clamp →
9.3e13×, coefficient 1e-12 relative → 45×), **document the sub-bound case as a blind spot**, and
**measure and publish the sensitivity floor** — arm E's is *any relative coefficient error ≥ 1e-13*.
A tolerance without a measured floor is a tolerance nobody can audit.

### 4. ⚠ SUPERSEDED BY A1 — DO NOT REGISTER THIS FORM
**The re-audit refuted this entry's own framing: the probe bounded NOTHING, including rank. Register
A1 (post-hoc addendum, below) instead.** Kept only so the correction is traceable.

<details><summary>original text (wrong)</summary>

**Verified 🟢, and it cost me a prediction.** Gate 2's resume test incidentally produced a
42,605-eval arm-E champion at **268.6092** ms/char. I made that my point estimate (P2) for the
10M-eval run. Actual: **258.1803** — wrong by **10.4 ms/char**. At 0.4% of budget the search has not
yet found the in-domain interior, so an early champion is **not** a scaled-down version of the final
one.

The transferable rule: a cheap probe is valid evidence about *whether the objective ranks the band
correctly* (which it was — ρ decays to ≈0) and invalid evidence about *where the optimum lands*.
⚠ Note the symmetry with ARMD-1: its ±14 ms/char miss came from **not** probing its objective
in-band; mine came from probing it and **over-reading** the probe. Both are "a label is not its
referent" — treating a proxy as the thing.

</details>

### 5. Write the artifact LAST — a mid-function dump silently drops later sections
**Verified 🟢.** Arm D's `report_armd.py` dumps `judgement.json` mid-function and then keeps adding
sections. Inherited verbatim, that left arm E's P14 `champion_drivers` **printed to stdout but
absent from the JSON** — exactly trap 19 ("a metric absent from a published JSON was never computed
— check keys, not prose"), except worse, because the prose *was* correct and only the artifact was
incomplete. Anything computed after the dump is quotable from a log but unreconcilable from the
artifact.

**Fix, applied in `report_arme.py`:** dump last, then **enumerate the keys a reader will cite and
assert each exists** (`assert not missing_keys`). Arm E asserts 15.

### 6. When a pre-registered band's LABEL overclaims relative to its numeric threshold, honour the number and reject the label
**Verified 🟢, methodological.** My E3 band was defined as "≥256.9" but *labelled* "the curves are
the defect **regardless of fit pool**, which … **closes the evidence-weight line entirely**." Arm E
landed at 258.1803 — E3 on the number — while the same run showed the fit pool worth **11.0959
ms/char (22× the resolution floor)**, which directly contradicts "regardless of fit pool".

Pre-registration binds you to the *threshold*, not to a conclusion you wrote before seeing the data.
The honest move is to report the band as met **and say explicitly which part of its label the data
refutes** — otherwise pre-registration becomes a device for laundering an overclaim.

---

## B. Proposed follow-up arm (NOT run — needs its own pre-registration)

🟠 **INFERRED.** Arm E's champion pushed **5 of 6** out-of-domain gauges in the mechanism-*right*
direction (lower `sfb`/`sfb-dist`/`sfs`/`sfs-dist`/`scissor`) and CLAMP correctly stopped it at the
edge. So on a well-covered fit, **the clamp is now sometimes blocking real improvement** rather than
blocking exploitation — its cost is no longer zero, as it effectively was for arm D.

That suggests **arm F: refit the curves on a pool that COVERS the near-optimal band's good side**
(e.g. the archive pool plus the incumbents' 1–4-swap neighbourhood), so the region the search wants
to reach is *supported* rather than clamped. This is different from arm E: arm E changed which
region is supported; arm F would change the support to include where the optimum actually lies.

⚠ **Do not run it on this arm's authority.** It needs its own pre-registration, and there is a
strong prior against it: the residual defect arm E measured is **40.84% of attribution moving the
mechanistically wrong way** — a wider fit does not obviously fix a sign error, and could make it
worse by giving the maximizer more supported room to exploit. My honest expectation is that arm F
also fails; I flag it because it is the one remaining structural change nobody has tested, not
because I predict it works.

---

## C. What I would tell the next agent on this repo, in one line each

1. The five evidence arms have now produced **0 layouts competitive with the incumbents**; arm B
   (plain ms/char minimization) remains the fastest thing the campaign has made at **253.9006**.
2. ⚠ **CORRECTED by A7/A8:** the **fit pool** is worth 11.10 ms/char as a **BUNDLED** effect
   (coeffs/domain/knot/form all differ), and it is **still not sufficient** (+3.55 vs the worst
   incumbent). Do NOT say "domain coverage is first-order" — the archive curves are proportionally
   MORE mis-signed (42.5% vs 17.5% of collectable units) and the leading candidate mechanism is
   objective SCALE (6.43 vs 48.81 units). Both halves are load-bearing; quoting either alone
   misrepresents the result.
3. 🔴 **CORRECTED by A12 — do NOT register the old form.** I had written "the flat-objective
   hypothesis is refuted twice". Zero plateaus is the wrong test: within 0.02 objective units sit
   layouts spanning 12.24 ms/char, and two seeds of the SAME arm differ by 9.43 ms/char. The
   objective is non-degenerate in its own units and near-degenerate w.r.t. speed. Old text:
   **The flat-objective hypothesis is now refuted twice**, on two independent fits (arm D
   1730/1730, arm E 1698/1698, zero plateaus both). Stop proposing it.
4. `keybo analyze --json` adds a `--ref` row — **assert set-containment, never row count** — and it
   independently reproduced arm E's ms/char to 4 dp, which is the check worth doing on any champion.

---

# POST-HOC AUDIT ADDENDUM (2026-07-28)

The parent registered arm E as **ARME-1 (`571bfe9`)**, accepted the E3-label rejection, confirmed
the `price_many` defect and fixed it in **`79cb175`**. I then re-audited my own claims as a skeptic.
**Three of my attributions were wrong; every measurement stands.** These supersede/extend §A above.

## A1 (SUPERSEDES A4) — "a cheap probe bounds rank, not the optimum" is WRONG; the probe bounded nothing
🟢 My 42,605-eval probe had **ev −1.4335 vs the final −2.6902 (53% of the objective)**. It bounded
neither ms/char (268.6092 vs 258.1803 — off by 10.4 in the *overstating* direction, so not a usable
bound either way), nor the objective value (the search improved 88% past it), nor rank (it was **one
layout**; my rank evidence came from a *separate* 3600-perturbation pool, so my lesson credited the
probe with another artifact's property).

**Register this instead: never use an unconverged run as a point estimate — and diagnose convergence
by whether best-fitness has STOPPED IMPROVING, not by budget fraction.** It was detectable for ~10 s:
epoch 1 of the real run already had **368,209 unique evals at best −2.204979**, 8.7× the probe's
budget and far past its value, with improvement continuing to epoch 47.

## A7 (NEW, and the most valuable one) — a cross-arm "only X changed" claim needs a PER-PARAMETER DIFF before the causal sentence
🟢 I wrote "domain coverage is FIRST-ORDER" from a single arm-D-vs-arm-E delta. The two weight sets
differ in **coefficients 14/14, valid_domain 14/14, knot 13/14, functional form 2/14** — four
factors at once. The correct form is **"the fit pool is worth 11.0959 ms/char, bundled"**, with the
mechanism named as a candidate, not a finding.

The cheap check that would have caught it: **diff the two objects field-by-field and print the
counts before writing the causal sentence.** It is ~10 lines and it is the difference between a
finding and an attribution error. Same family as trap 17/18 (a delta cannot attribute a multi-factor
change) and trap 16 (a contrast that cannot isolate the axis it names), but the failure here is that
I named an axis the design could not isolate *and had the data to know it*.

## A8 (NEW) — a fitted objective's DAMAGE scales with its total collectable units, so a proportionally WORSE fit can search BETTER
🟢 The finding that most surprised me on re-audit. I assumed arm E did better because its curves
were better. They are not:

| | mechanism-correct minima | mis-signed units | RIGHT units | **mis-signed share** | total in-domain signal |
|---|---|---|---|---|---|
| random400 (arm D) | **9 / 14** | 8.5623 | 40.2467 | **17.5%** | **48.8090** |
| archive (arm E) | **8 / 14** | 2.7333 | 3.7016 | **42.5%** | **6.4349** |

The archive fit is *proportionally more* mis-signed and has *fewer* correct minima, yet its champion
is 11.10 ms/char faster — because a separable sum of mis-signed curves pays damage in proportion to
**absolute** collectable units, and it is **7.6× smaller**. **So "is this fitted objective safer?"
must be asked about its SCALE, not only its sign-correctness.** (Read the piecewise coeffs and knot
for the minimum's location — trap 53 — then multiply by the range.)

⚠ **Scale and coverage are mutually non-identifiable from any single fit pool**, because a narrower
pool mechanically yields both narrower domains and smaller ranges. Isolating them needs a fit that
holds one fixed (refit on one pool but rescale to the other's total signal, or fit both pools under
a common domain).

## A9 (NEW) — report a rank correlation in a thin band with a CI, or it is not a measurement
🟢 My in-band decay table quoted bare point estimates. Bootstrapped (2000 resamples, n=3600 pool):
all `[+0.7103,+0.7437]` · ≤257 `[+0.3676,+0.4696]` · ≤256 `[+0.1767,+0.3321]` · ≤255.5
`[+0.0450,+0.2755]` — all exclude zero — but **≤255.0 is `[−0.1473,+0.2604]`, p=0.558, n=104**:
**indistinguishable from zero.** The decay is real; the tightest cell's *value* is not.

Corollary, and it retroactively affects a cross-agent comparison: **optevidence's 36,005-perturbation
+0.9111 → −0.0455 is the RAW random400 objective**, not my CLAMPED archive one. The like-for-like
column is my own `r400 extrap` (**+0.9017 → +0.0809**), which *does* track theirs closely — that
agreement is the real corroboration. At n=104 my cell cannot distinguish +0.0809 from −0.0455
either, so no magnitude comparison between the two campaigns' tightest cells is supportable.

## A10 (NEW) — a "N of 14 curves affected" count from ONE probe level is probe-dependent
🟢 I reported the `price_many` defect as **7 of 14** curves; the parent measured **9 of 14**. Both are
right for their sample and both understate: on a **101-level in-domain grid it is 14 of 14**. The
defect is per-**LEVEL**, not per-curve. **When counting "how many objects exhibit a numerical
defect", sweep the input domain and say which sample the count is over** — otherwise the count is a
property of your probe.

🟢 **And the fix is verified not to move this arm:** re-scoring the frozen arm-E champion against
`79cb175`'s module (loaded standalone; no sibling worktree touched) gives ev_clamp
**−2.690225544692558** — bit-identical to the frozen artifact — ms/char **258.1803** unchanged, board
ordering and argmin identical, worst board diff 4.441e-16. The fixed version is **0/14**
shape-dependent with worst |price_many − price| = **0.000e+00**, so exact equality is now attainable
and the previously-unsatisfiable pin instruction becomes satisfiable.

## A11 — the one claim that SURVIVED an attempt to break it
🟢 The audit asked whether "mechanism-right" is circular. It is **not**, though the concern is
well-posed: `EXPECTED_SIGN` (`evidence_scorer.py:121–136`) is a hardcoded, hand-authored prior table,
not derived from the surface. But it is **independently testable and passes**: rank correlation of
each raw gauge against predicted ms/char agrees with the table on **14/14 in-band (≤257.0, n=1010)**
and 13/14 on 4000 random permutations (sole disagreement `sfs`, ρ=−0.0218 ≈ 0). So the
inverse-signature claim is falsifiable and not refuted. **Worth registering as a positive: this repo
has a hardcoded sign convention that the served surface corroborates, so it can be cited without
circularity — but cite the corroboration, not the table.**

## Corrected recommendation on arm F (§B above stands, with one addition)
Still **AGAINST**, for the one-sentence reason in report §11.5. **New addition from A8:** if it is run
anyway, it MUST pre-register a **scale control** (hold total in-domain signal fixed while widening
the support), because the mechanism that plausibly produced arm E's improvement is exactly the one a
wider refit would undo — and without that control arm F reproduces arm E's non-identifiability at
greater cost.

## A12 (NEW — the highest-value entry in this whole document) — "zero plateaus" answers the wrong question; run a second seed
🟢 I reported 1698/1698 distinct objective values, zero plateaus, and framed it as *"the search was
well-conditioned and confident"* — presenting it as evidence **against** the sibling's
flat-objective warning. That framing was wrong, and a **second seed (~9 min) proved it**:

| | ev_clamp | **ms/char** | shared key positions |
|---|---|---|---|
| seed 20260728 | −2.690226 | **258.1803** | — |
| seed 20260729 | −2.677732 | **267.6096** | **2 / 30** |

**Objectives 0.46% apart, layouts 9.4293 ms/char apart.** Pooling both final populations (n=5120),
the ms/char spread among layouts the objective cannot meaningfully separate:

| within … of best ev | n | ms/char spread |
|---|---|---|
| 0.005 units | 120 | 0.1301 |
| **0.010** | 204 | **4.1554** |
| **0.020** | 1005 | **12.2353** |
| 0.050 | 2274 | 15.3611 |

**Register two things:**
1. **"Does the objective distinguish the layouts?" is not the question. Ask how much the TARGET
   quantity varies inside a band of the objective the search cannot resolve.** An objective can be
   perfectly non-degenerate in its own units and near-degenerate w.r.t. what you care about — which
   is *worse* than a plateau, because it looks healthy. My "zero plateaus, refuted twice" line was
   a sharper *confirmation* of the sibling's warning dressed as a refutation.
2. **A second seed is the cheapest possible check that a champion is reproducible, and this campaign
   has not been running one.** ~9 min per arm here. It invalidated the precision of every per-pair Δ
   I published: arm E vs arm B is **0.5×** the search spread (I reported 8.62× the *timing* floor),
   vs keybo-lsb **0.4×**, vs arm A **0.1×**. ⚠ **The paired timing floor and the search spread are
   different rulers, and the campaign has been quoting the former for claims about the latter.**
   OPTEVIDENCE-1's 0.3440 is **27× too small** for arm E, so it must not be borrowed across arms.

**What survives, and should be stated this way:** the **arm-level** conclusion is unaffected — both
seeds land far above arm B (+4.28, +13.71), above every incumbent, and both satisfy E3. What loses
precision is every *specific gap size*, including **the 11.0959 that "72% recovered" rests on (only
~1.2× the spread)**. **"72%" needs n≥3 seeds per arm before it is quotable.**

⚠ **This also retroactively questions every single-seed champion comparison in the campaign** (arms
A/B/C/D are all n=1 as far as I can see). I have NOT verified their seed sensitivity — that is a
parent-level call, not mine, but it is the obvious next check and it is cheap.
