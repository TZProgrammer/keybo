"""ARM D gate 1 — verify the parent's domain-policy plumbing on the REAL fitted curves.

The parent's own `tests/analysis/test_domain_policy.py` passes, but every one of its 7 tests
builds a SYNTHETIC hinge (`coeffs=[0,-1,-2]`) rather than loading the fitted weights arm D will
actually search against. A test that constructs its own expectation can be green while the real
object misbehaves (trap 31), so this gate re-runs the two required checks against
`arm-random400-native.json` itself, over all 14 live gauges:

  A. IN-DOMAIN PRICES ARE BIT-IDENTICAL across EXTRAPOLATE / CLAMP / REJECT.
     Not `approx` — `==` on the float, because the whole warrant for the fix is that it
     perturbs no supported level, and a frozen number that moves by 1e-16 is still moved.
  B. UNDER CLAMP, PUSHING A GAUGE 50x PAST ITS CEILING BUYS EXACTLY NOTHING.
     Also checked below the floor, and checked that EXTRAPOLATE *does* keep paying — because
     if extrapolation were already bounded there would have been no defect to fix and the
     clamp would be untestable (the check must be able to fail).

Plus two checks the parent's suite cannot express, and which arm D's validity rests on:

  C. The SEARCH-PATH evaluator (`evobj.Curve`) is a hand-rolled vectorized reimplementation of
     `LossCurve.price` (trap 28: a reimplementation loses the validation). Under EXTRAPOLATE the
     two must agree to float round-off on real levels, or arm D is not comparable to arm A.
  D. `armd_obj.ClampedCurve` — my clamped search-path curve — must equal `LossCurve.price(...,
     policy=CLAMP)` on in-domain, out-of-domain-high and out-of-domain-low levels.

MODELLED ONLY: these are fitted-surface curves, not measurements of realized typing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from armd_load import load_curves  # noqa: E402
from keybo.analysis.evidence_scorer import (  # noqa: E402
    CLAMP,
    EXTRAPOLATE,
    LIVE_GAUGES,
    REJECT,
    SEARCH_DOMAIN_POLICY,
    OutOfDomainError,
)

ARM_JSON = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"

FAILURES: list[str] = []
CHECKS = 0


def check(ok: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(label)
        print(f"  FAIL  {label}")


def main() -> int:
    print(f"SEARCH_DOMAIN_POLICY = {SEARCH_DOMAIN_POLICY!r}")
    check(SEARCH_DOMAIN_POLICY == CLAMP, "SEARCH_DOMAIN_POLICY is CLAMP")

    curves = load_curves(ARM_JSON)
    print(f"loaded {len(curves)} fitted curves from {ARM_JSON}")
    print(f"{'gauge':<12} {'form':<10} {'domain':<26} knot")
    for name in LIVE_GAUGES:
        c = curves[name]
        knot = "-" if c.knot is None else f"{c.knot:.4f}"
        print(f"{name:<12} {c.form:<10} [{c.domain[0]:9.4f}, {c.domain[1]:9.4f}]   {knot}")

    # ---- A. in-domain prices bit-identical across all three policies --------------------
    print("\nA. in-domain prices, EXTRAPOLATE vs CLAMP vs REJECT (exact float equality)")
    n_levels = 0
    for name, c in curves.items():
        lo, hi = c.domain
        levels = list(np.linspace(lo, hi, 41))
        if c.knot is not None and lo <= c.knot <= hi:
            levels.append(float(c.knot))  # the knot is where a hinge is most fragile
        for level in levels:
            n_levels += 1
            e = c.price(float(level), policy=EXTRAPOLATE)
            k = c.price(float(level), policy=CLAMP)
            r = c.price(float(level), policy=REJECT)
            check(k == e, f"{name} @ {level:.6f}: CLAMP {k!r} != EXTRAPOLATE {e!r}")
            check(r == e, f"{name} @ {level:.6f}: REJECT {r!r} != EXTRAPOLATE {e!r}")
    print(f"   {n_levels} in-domain levels x 2 comparisons, all exact")

    # ---- B. clamp saturates; extrapolate does not ---------------------------------------
    print("\nB. out-of-domain: CLAMP saturates, EXTRAPOLATE keeps moving")
    print(f"{'gauge':<12} {'x1=hi+w':>12} {'x50=hi+50w':>12} {'clamp equal?':>13} "
          f"{'extrap moves?':>14} {'extrap drift':>14}")
    for name, c in curves.items():
        lo, hi = c.domain
        width = hi - lo
        x1, x50 = hi + width, hi + 50.0 * width
        c1, c50 = c.price(x1, policy=CLAMP), c.price(x50, policy=CLAMP)
        e1, e50 = c.price(x1, policy=EXTRAPOLATE), c.price(x50, policy=EXTRAPOLATE)
        at_hi = c.price(hi, policy=EXTRAPOLATE)
        check(c1 == c50, f"{name}: CLAMP at hi+w {c1!r} != at hi+50w {c50!r}")
        check(c50 == at_hi, f"{name}: CLAMP at hi+50w {c50!r} != price(hi) {at_hi!r}")
        # below the floor too
        y1, y50 = lo - width, lo - 50.0 * width
        d1, d50 = c.price(y1, policy=CLAMP), c.price(y50, policy=CLAMP)
        at_lo = c.price(lo, policy=EXTRAPOLATE)
        check(d1 == d50 == at_lo, f"{name}: CLAMP below floor not pinned to price(lo)")
        moves = e1 != e50
        print(f"{name:<12} {x1:12.4f} {x50:12.4f} {str(c1 == c50):>13} "
              f"{str(moves):>14} {e50 - e1:14.4f}")
        # A curve whose extrapolation does NOT move would make the clamp untestable here.
        # `quadratic` and `hinge` always move; a pure `linear` with slope 0 would not.
        check(moves, f"{name}: EXTRAPOLATE did not move over 49 domain-widths "
                     f"(the check cannot fail => it tests nothing)")
    # REJECT must actually raise out of domain
    for name, c in curves.items():
        lo, hi = c.domain
        try:
            c.price(hi + (hi - lo), policy=REJECT)
            check(False, f"{name}: REJECT did not raise above the ceiling")
        except OutOfDomainError:
            check(True, f"{name}: REJECT raised")
    print("   REJECT raises for all 14 above the ceiling")

    # ---- C. search-path evaluator agrees with LossCurve under EXTRAPOLATE ---------------
    print("\nC. evobj.Curve (search path, hand-rolled) vs LossCurve, EXTRAPOLATE")
    sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
    import evobj as EV  # noqa: E402

    worst_c = 0.0
    for name, c in curves.items():
        ev_curve = EV.Curve(metric=name, form=c.form, coeffs=np.asarray(c.coeffs, dtype=float),
                            knot=c.knot, domain=c.domain)
        lo, hi = c.domain
        xs = np.linspace(lo - (hi - lo), hi + (hi - lo), 61)
        a = ev_curve.price(xs)
        b = np.array([c.price(float(x), policy=EXTRAPOLATE) for x in xs])
        d = float(np.max(np.abs(a - b)))
        worst_c = max(worst_c, d)
        check(d < 1e-12, f"{name}: evobj.Curve vs LossCurve EXTRAPOLATE max|diff| = {d:.3e}")
    print(f"   max|evobj - LossCurve| over 14 gauges x 61 levels = {worst_c:.3e}")

    # ---- D. my clamped search-path curve equals LossCurve under CLAMP -------------------
    print("\nD. armd_obj.ClampedCurve vs LossCurve, CLAMP")
    import armd_obj as AD  # noqa: E402

    worst_d = 0.0
    for name, c in curves.items():
        ad_curve = AD.ClampedCurve(metric=name, form=c.form,
                                   coeffs=np.asarray(c.coeffs, dtype=float),
                                   knot=c.knot, domain=c.domain)
        lo, hi = c.domain
        width = hi - lo
        xs = np.concatenate([
            np.linspace(lo, hi, 41),                      # in domain
            np.linspace(hi + 1e-9, hi + 50 * width, 20),  # above the ceiling
            np.linspace(lo - 50 * width, lo - 1e-9, 20),  # below the floor
        ])
        a = ad_curve.price(xs)
        b = np.array([c.price(float(x), policy=CLAMP) for x in xs])
        d = float(np.max(np.abs(a - b)))
        worst_d = max(worst_d, d)

        # `d` is NOT zero, and the reason must be RECONCILED rather than tolerated (trap 43).
        # `LossCurve.price` evaluates `_design(...) @ coeffs` (a matmul, which reassociates)
        # while the search path evaluates `c0 + c1*x + c2*max(x-knot,0)` term by term. The two
        # are algebraically identical and differ only in float round-off — the SAME round-off
        # gate C measures under EXTRAPOLATE (7.105e-15 there).
        #
        # The exact identity that proves it is a pricing-path artifact and not a clamp bug:
        # ClampedCurve.price(x) evaluates the explicit form at clip(x), and
        # LossCurve.price(x, CLAMP) evaluates the matmul form at the same clip(x). So the
        # residual at x must EQUAL the EXTRAPOLATE residual at clip(x) — bit for bit, for
        # every x, on both sides of the domain. That is an equality, not a tolerance.
        ev_curve = EV.Curve(metric=name, form=c.form, coeffs=np.asarray(c.coeffs, dtype=float),
                            knot=c.knot, domain=c.domain)
        clipped = np.clip(xs, lo, hi)
        resid_clamped = a - b
        resid_extrap = ev_curve.price(clipped) - np.array(
            [c.price(float(x), policy=EXTRAPOLATE) for x in clipped]
        )
        check(np.array_equal(resid_clamped, resid_extrap),
              f"{name}: CLAMP residual != EXTRAPOLATE residual at the clipped level "
              f"(so the gap is NOT just the pricing path)")
        check(d < 1e-12, f"{name}: ClampedCurve vs LossCurve CLAMP max|diff| = {d:.3e} "
                         f"exceeds the matmul round-off scale")

        # In-domain, both use the SAME explicit arithmetic, so exact equality IS achievable
        # here and is the load-bearing claim: arm D perturbs no supported level of arm A's
        # objective. This one stays at `== 0.0`.
        xin = np.concatenate([np.linspace(lo, hi, 41), [c.knot] if c.knot is not None
                              and lo <= c.knot <= hi else []])
        din = float(np.max(np.abs(ad_curve.price(xin) - ev_curve.price(xin))))
        check(din == 0.0, f"{name}: ClampedCurve != evobj.Curve IN DOMAIN (diff {din:.3e})")
    print(f"   max|ClampedCurve - LossCurve(CLAMP)| = {worst_d:.3e} "
          f"(= the matmul-vs-explicit round-off gate C measures; residuals match EXACTLY)")
    print("   in-domain: ClampedCurve == evobj.Curve bit-for-bit on all 14 gauges")

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES[:20]:
            print(f"  - {f}")
        return 1
    print("GATE 1 PASS — the domain policy is sound on the real fitted curves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
