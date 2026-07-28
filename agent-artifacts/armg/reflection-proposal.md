# ARM G — reflection proposal (reusable learnings)

Observations only. **Not written to the shared KB** — that needs explicit user approval.
Branch `optimize-arm-g` tip **`f43701d`**.

---

## 1. NEW TRAP — a SEARCH region set from a BORROWED ruler can be LOOSER than the VERDICT region computed from your own

**Shape.** When an experiment must *constrain* a search using a quantity it will only *measure*
after the search runs, the constraint necessarily comes from a borrowed figure. If the borrowed
figure is larger than the measured one, the search region is **wider than the judging region**,
and a competent optimizer will spend exactly that difference — producing champions that sit in
the gap and fail a condition they were never given a chance to satisfy.

**Instance (ARM G).** Search band `EPS = 0.1234` (2 × borrowed 0.0617) → edge 254.0240. Verdict
band 2 × measured `sd_G` = 0.0983 → edge 253.9989. Gap **0.0251**. All five ARM G champions
landed in the gap; verdict FAILURE by F1.

**Why it survived review:** my prereg *named* this risk and reasoned it would be conservative
"if my own sd comes out LARGER." It came out **smaller (1.255×)** — the anti-conservative
direction. **Naming a risk is not bounding it: state which sign is safe AND enforce it.**

**Fix, cheapest first:**
1. Run a **3-seed pilot** to measure the sd *before* fixing the constraint (~2 min at 1M evals
   — I had the budget and did not do this).
2. Or set the search constraint from the **tighter of (borrowed, own-once-known)**, and re-run
   only if the tighter one binds.
3. Or make the search constraint **strictly tighter** than any plausible verdict band and let
   the verdict band be the loose one — errs the safe way by construction.

**Detector, one line:** list every threshold with both a *run-time* and a *judge-time* role and
assert they are the same object or the run-time one is tighter. (I did this sweep — see §4 —
and the band was the *only* divergent threshold, because every other constant is imported by
the judge from the search module and so cannot diverge. That import discipline is worth
copying.)

---

## 2. NEW TRAP — a range-normalized SUM makes the WIDEST axis the CHEAPEST to trade away

**Shape.** An objective of the form `Σ_g max(0, (g − ref_g)/s_g)` with `s_g` = the pool range
makes every axis equally *tradeable*. If you care about **one** axis, this is the wrong
objective: the optimizer will buy cheap wins on narrow axes and pay on the widest one.

**Instance (ARM G).** Built to collect `oxey-style` headroom; its champion is **worse on
`oxey-style` (11.3958) than the reference it was collecting against (arm B, 8.6110)**, while
every incumbent is far better (flagship-c3 −7.8749). `oxey-style` (s = 13.15, the widest axis)
ended as only **20.0 %** of my own final deficit. The parent quantified it further and the
stronger form belongs in the ledger: **`oxey-style` is 48.5 % of the whole board's gauge range,
5.3× the next widest axis.**

**Compounding factor:** `oxey-style` is R² = 0.9937 on {sfb, lsb, scissor, imbalance, redir,
alt} (trap 27), so a sum over all 14 axes **double-counts** that cluster — rewarding sfb/lsb/
scissor gains while charging the composite that restates them. The objective was partly
fighting itself. **I flagged this in my own prereg §5 and shipped the summed form anyway** —
identifying a defect in writing is not the same as designing around it.

**Detector:** compute each axis's **share of the final deficit**. If the axis the experiment is
*about* is a small share, the objective is not testing the hypothesis. **Fix:** lexicographic
order, or a hard per-axis constraint — never a summed penalty (see §3).

---

## 3. Sharpening of trap 51 — "a maximizer does not read flags" extends to SOFT PENALTIES

Trap 51 says an `extrapolating: true` flag is worthless because a maximizer does not read
flags. ARM G shows the next step: **a summed/soft penalty is a flag.** My quadratic speed
penalty (λ = 1000, normalized by EPS²) *did* hold the constraint in effect — but only because
I tuned λ against the D range. A penalty is a price, and a price is negotiable; a constraint is
not. **If a boundary must not be crossed, encode it as a hard rejection, not as a term in a
sum.**

---

## 4. Confirmed-good practices worth copying (cheap, high yield)

- **Commit the JUDGE while the runs are still executing.** Stronger evidence of non-tuning than
  any prose claim, and it costs nothing. (ARM G: `ceb85cd` landed mid-flight.)
- **Have the judge IMPORT its constants from the search module** rather than restating them.
  Verified: `ARMG_REF`/`ARMG_SCALE`/`ARMG_DIR` are the *same objects* in both, so they cannot
  drift. This is what limited my threshold defect to exactly one constant.
- **Ship a constants-assert that refuses to run on drift.** `armg_assert_constants()`
  re-derives REF/SCALE/REF_MS from live code and raises; wired into `main()` so it cannot be
  skipped, and its result is recorded in every output blob.
- **Designate LOAD-BEARING vs BULK artifacts in the index.** My set is 82 MB but the verdict is
  re-derivable from ~180 KB; the sidecars carry ~99.8 % of the bytes and **no claim**.

## 5. NEW TRAP (candidate) — hand-transcribing constants out of an artifact is a defect generator

2 of 2 of my attempts were wrong: all 14 `ARMG_SCALE` constants off by ~1e-5, and one
**invented** 30-char layout string that exists nowhere. **Generate the dict programmatically
from the artifact, or ship an assert that re-derives from live code.** Never retype.

⚠ **And note the near-miss in my own diagnosis:** my first hypothesis for the 1e-5 drift was
BLAS batch-shape dependence (the MODELNORM-1 class). I *measured* it rather than asserting it
and refuted my own hypothesis — the real effect is ~1e-15, roughly **ten orders too small**.
The boring cause (my typing) was correct. **When a plausible documented failure class is
available, measure before adopting it; a known trap makes an attractive wrong answer.**

---

## 6. Process gap worth a rule — ROBUSTNESS OF THE VERDICT TO THE REGISTERED STATISTIC

My prereg fixed "sd, ddof=1, n=5" and I applied it faithfully. It did **not** ask whether the
verdict is *stable* across defensible alternatives. On audit: **F1 fires under 4 of 8 rulers
(4 of 6 computed from my own replicates)** — it does *not* fire under a **range**-based
statistic, and does not fire under either borrowed figure. By contrast the headline Q1 answer
("nothing was faster than arm B") is **ruler-invariant: 0 champions qualify under all 8.**

**Proposed rule:** a prereg that fixes a statistic should also pre-register **one sensitivity
check** over the defensible alternatives, and the write-up should say which conclusions are
statistic-invariant and which are statistic-contingent. Cost: one loop. It converts "my
registered rule said FAILURE" into "FAILURE under sd-family rulers, indeterminate under
range-family ones" — a materially more honest claim.
