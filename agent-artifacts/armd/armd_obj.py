"""ARM D objective — arm A's evidence score with `valid_domain` as a HARD CONSTRAINT.

Arm D is arm A in EVERY respect (same weights JSON, same island seeds, same unique-eval budget,
same corpus) except one: the loss curves are priced under `SEARCH_DOMAIN_POLICY` (= `CLAMP`)
instead of extrapolating. That is the whole experiment, so the ONLY thing this module may change
about the objective is what happens OUTSIDE a curve's fitted domain.

WHY A SEPARATE CURVE CLASS EXISTS AT ALL. `keybo.analysis.evidence_scorer.LossCurve.price` now
takes a `policy`, but the search does not go through it: `evobj.Curve.price` is a hand-rolled
vectorized reimplementation over `(B,)` arrays (trap 28 — a reimplementation loses the
validation, and here it also loses the policy). Adding the policy to `LossCurve` therefore does
NOT clamp the search; the clamp has to exist on the vectorized path too. `ClampedCurve` is that
path, and `verify_policy.py` gate D pins it against `LossCurve.price(..., policy=CLAMP)` at
exact float equality, in-domain and on both sides of the domain.

CLAMPING THE INPUT, NOT THE OUTPUT. `np.clip(level, lo, hi)` then evaluating the curve is what
`LossCurve` does, and it is the right shape: it prices the layout AS IF the gauge sat at the
nearest supported level, so the price SATURATES. Clamping the output instead would bound the
total but leave the gradient pointing out of the domain, which is the defect.

⚠ THE CLAMP IS NOT A CONSTRAINT ON THE SEARCH SPACE. It removes the INCENTIVE to leave the
domain (past the edge, nothing more is paid) but a layout may still sit outside it — reaching an
in-domain gauge vector may be infeasible for a C30M permutation, and 5 of the 14 curves are
minimized at an out-of-domain edge to begin with. So "arm D's champion is still out of domain"
is NOT by itself evidence of broken wiring; the diagnostic that separates the two is whether
pushing FURTHER out still pays, which `report.py` measures directly (`clamp_binding`).

MODELLED ONLY: fitted-surface attribution, not measured typing speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClampedCurve:
    """`evobj.Curve` with the fitted domain enforced as a hard constraint.

    In-domain this is bit-identical to `evobj.Curve` (the clamp is a no-op there, and
    `verify_policy.py` gate D asserts exact equality). Outside, the price is the price at the
    nearest domain edge, so it stops rewarding further travel.
    """

    metric: str
    form: str
    coeffs: np.ndarray
    knot: float | None
    domain: tuple[float, float]

    def _raw(self, x: np.ndarray) -> np.ndarray:
        """The curve as fitted — byte-for-byte the arithmetic in `evobj.Curve.price`."""
        c = self.coeffs
        if self.form == "linear":
            return c[0] + c[1] * x
        if self.form == "quadratic":
            return c[0] + c[1] * x + c[2] * x * x
        return c[0] + c[1] * x + c[2] * np.maximum(x - self.knot, 0.0)

    def price(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.domain
        return self._raw(np.clip(x, lo, hi))

    def price_extrapolating(self, x: np.ndarray) -> np.ndarray:
        """Arm A's pricing, kept so one pass can report both totals for the same layout."""
        return self._raw(x)


class ClampedEval:
    """`evobj.FastEval` with the evidence score computed under CLAMP.

    Wraps rather than subclasses so the gauge computation is *literally* the frozen arm-A code:
    every gauge value, denominator and kernel comes from the same `FastEval` instance arm A used,
    and only `evidence_score` is replaced.
    """

    def __init__(self, fe, policy: str = "clamp"):
        from keybo.analysis.evidence_scorer import CLAMP, EXTRAPOLATE

        if policy not in (CLAMP, EXTRAPOLATE):
            raise ValueError(f"arm D prices under CLAMP or EXTRAPOLATE only, got {policy!r}")
        self.fe = fe
        self.policy = policy
        self._clamp = policy == CLAMP
        assert fe.curves is not None, "load_weights() first"
        self.curves = [
            ClampedCurve(metric=c.metric, form=c.form, coeffs=np.asarray(c.coeffs, dtype=float),
                         knot=c.knot, domain=tuple(c.domain))
            for c in fe.curves
        ]

    # -- pass-throughs so this is a drop-in for FastEval in the search ---------------------
    def gauges(self, perms: np.ndarray):
        return self.fe.gauges(perms)

    def out_of_domain(self, gauges):
        return self.fe.out_of_domain(gauges)

    @property
    def corpus_dir(self):
        return self.fe.corpus_dir

    @property
    def weights_meta(self):
        return self.fe.weights_meta

    # -- the one thing arm D changes -------------------------------------------------------
    def evidence_score(self, gauges: dict[str, np.ndarray]) -> np.ndarray:
        total = np.zeros_like(gauges[self.curves[0].metric])
        for curve in self.curves:
            level = gauges[curve.metric]
            total = total + (curve.price(level) if self._clamp
                             else curve.price_extrapolating(level))
        return total

    def evidence_score_extrapolating(self, gauges: dict[str, np.ndarray]) -> np.ndarray:
        """Arm A's total for the same layout — the side-by-side the artifact needs."""
        total = np.zeros_like(gauges[self.curves[0].metric])
        for curve in self.curves:
            total = total + curve.price_extrapolating(gauges[curve.metric])
        return total

    def evaluate(self, perms: np.ndarray):
        g = self.gauges(perms)
        return self.evidence_score(g), g.get("_ms_per_char"), g
