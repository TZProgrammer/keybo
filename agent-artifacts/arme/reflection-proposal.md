# Reflection proposal — arme (ARM E)

**For the parent to register.** I do not edit `PREREGISTRATIONS.md` or the shared knowledge base
(per my scope). These are drafts; the parent decides what lands.

---

## A. Proposed TOOLING-TRAPS entries (6 candidates, most valuable first)

### 1. `LossCurve.price_many` is batch-shape-dependent — and its tests structurally cannot see it
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

### 4. A cheap early-budget probe bounds an objective's RANK behaviour, not its OPTIMUM's location
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
2. **Domain coverage is first-order** (11.10 ms/char) and **still not sufficient** (+3.55 vs the
   worst incumbent). Both halves are load-bearing; quoting either alone misrepresents the result.
3. **The flat-objective hypothesis is now refuted twice**, on two independent fits (arm D
   1730/1730, arm E 1698/1698, zero plateaus both). Stop proposing it.
4. `keybo analyze --json` adds a `--ref` row — **assert set-containment, never row count** — and it
   independently reproduced arm E's ms/char to 4 dp, which is the check worth doing on any champion.
