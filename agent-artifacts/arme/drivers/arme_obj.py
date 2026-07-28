"""ARM E objective — the ARCHIVE-fitted evidence weights priced under CLAMP.

Arm E is arm D in EVERY respect (same engine, same island seeds, same budget, same corpus, same
`SEARCH_DOMAIN_POLICY`) except ONE: the weights JSON is `arm-archive400-native.json` instead of
`arm-random400-native.json`. That is the whole experiment.

WHY THIS FILE DOES NOT HAND-ROLL THE CURVE. Arm D had to write its own vectorized `ClampedCurve`
because `evobj.Curve.price` is a hand-rolled reimplementation that never calls `LossCurve`, so the
policy plumbing and the code the optimizer runs were two different implementations (trap 28). The
fix landed in `cf5f731`: `LossCurve.price_many(levels, policy=...)` is a VALIDATED vectorized
entry point with semantics identical to `LossCurve.price` element-wise. So arm E calls **that**
and keeps no arithmetic of its own:

    ValidatedClampedEval.evidence_score  ->  LossCurve.price_many(level, policy=CLAMP)

`gate1_policy.py` pins this against BOTH the scalar `LossCurve.price(..., policy=CLAMP)` and arm
D's frozen hand-rolled `ClampedCurve` at exact float equality, in-domain and on both sides of every
domain — so arm E's objective is provably the same function arm D optimized, evaluated on different
curves.

CLAMPING THE INPUT, NOT THE OUTPUT (inherited from `LossCurve`): the level is clipped to the
nearest supported value and the curve evaluated there, so the price SATURATES and the gradient out
of the domain is exactly zero. Clamping the output would bound the total while still pointing out
of the domain, which is the defect `SEARCH_DOMAIN_POLICY` exists to remove.

⚠ THE CLAMP IS NOT A CONSTRAINT ON THE SEARCH SPACE. It removes the INCENTIVE to leave a domain,
but a C30M permutation may still land outside one — so "the champion is out of domain" is not by
itself evidence of broken wiring. The diagnostic that separates the two is whether pushing FURTHER
out still pays; `judge_arme.py` measures that directly (`clamp_binding`, abort if != 0.0).

MODELLED ONLY: fitted-surface attribution, not measured typing speed.
"""

from __future__ import annotations

import numpy as np

from keybo.analysis.evidence_scorer import CLAMP, EXTRAPOLATE, LossCurve


class ValidatedClampedEval:
    """`evobj.FastEval` with the evidence score computed through `LossCurve.price_many`.

    Wraps rather than subclasses so the gauge computation is *literally* the frozen arm-A/arm-D
    code: every gauge value, denominator and kernel comes from the same `FastEval` instance, and
    only `evidence_score` is replaced. The curves come from the weights JSON via
    `arme_load.load_curves`, which round-trip-asserts every serialized field.
    """

    def __init__(self, fe, curves: dict[str, LossCurve], policy: str = CLAMP):
        if policy not in (CLAMP, EXTRAPOLATE):
            raise ValueError(f"arm E prices under CLAMP or EXTRAPOLATE only, got {policy!r}")
        self.fe = fe
        self.policy = policy
        # Order is pinned to the evaluator's own curve order so a gauge can never be scored
        # against another gauge's curve.
        assert fe.curves is not None, "load_weights() first"
        metrics = [c.metric for c in fe.curves]
        missing = set(metrics) - set(curves)
        if missing:
            raise AssertionError(f"weights JSON is missing curves for {sorted(missing)}")
        self.metrics = metrics
        self.curves = [curves[m] for m in metrics]

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

    # -- the objective ---------------------------------------------------------------------
    def _score(self, gauges: dict[str, np.ndarray], policy: str) -> np.ndarray:
        total = np.zeros_like(gauges[self.metrics[0]])
        for curve in self.curves:
            total = total + curve.price_many(gauges[curve.metric], policy=policy)
        return total

    def evidence_score(self, gauges: dict[str, np.ndarray]) -> np.ndarray:
        return self._score(gauges, self.policy)

    def evidence_score_extrapolating(self, gauges: dict[str, np.ndarray]) -> np.ndarray:
        """The unbounded total for the same layout — the side-by-side the artifact needs."""
        return self._score(gauges, EXTRAPOLATE)

    def evaluate(self, perms: np.ndarray):
        g = self.gauges(perms)
        return self.evidence_score(g), g.get("_ms_per_char"), g
