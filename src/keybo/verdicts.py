"""Guards for the one failure mode that keeps producing a confident wrong answer here.

Three defects found in this repo within two days shared a shape:

* ``keybo analyze`` invoked as ``python -m keybo.cli.analyze`` exited 0 with ZERO output —
  the module has no ``__main__`` guard, so it imported and returned success.
* a pod-liveness probe reported "done" for a process name that never existed, because
  absence-of-process and never-started are the same observation.
* a paired A/B printed ``ARGMAX MOVES: False`` over two arms in which every candidate
  scored ``-inf``, because the objective was never evaluable on that data.

In each case a comparison whose operands were never computed returned the answer that
means "no difference" / "no problem". The verdict was true and empty, and reading it was
indistinguishable from reading a real result.

The guard is mechanical: assert the OPERANDS are finite before you compare them, and make
the "not measured" state a distinct value from any legitimate score. Prefer
:func:`require_finite` at the point where numbers become a claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


class EmptyComparison(ValueError):
    """A verdict was requested from operands that were never computed.

    Distinct from "the comparison ran and found no difference", which is a RESULT. This is
    the absence of a measurement, and the two must not share a return value.
    """


def require_finite(values: Iterable[float], what: str) -> list[float]:
    """Return ``values`` as floats, or raise :class:`EmptyComparison` if any is not finite.

    ``what`` names the quantity for the error message — make it specific enough that a
    reader can tell a data problem from a config problem without rerunning.

    >>> require_finite([1.0, 2.0], "gauge spread")
    [1.0, 2.0]
    >>> require_finite([1.0, float("-inf")], "candidate scores")
    Traceback (most recent call last):
        ...
    keybo.verdicts.EmptyComparison: candidate scores: 1 of 2 values are not finite ...
    """
    out = [float(v) for v in values]
    if not out:
        raise EmptyComparison(f"{what}: no values at all, so no verdict is available")
    bad = [i for i, v in enumerate(out) if not math.isfinite(v)]
    if bad:
        shown = ", ".join(f"[{i}]={out[i]!r}" for i in bad[:4])
        raise EmptyComparison(
            f"{what}: {len(bad)} of {len(out)} values are not finite ({shown}"
            f"{', ...' if len(bad) > 4 else ''}). A comparison over non-finite operands "
            f"returns the answer that means 'no difference' — fix the operands, do not "
            f"read the verdict."
        )
    return out


def compare_finite(
    a: Iterable[float], b: Iterable[float], what: str
) -> tuple[list[float], list[float]]:
    """Guard BOTH sides of a paired comparison, then hand them back.

    Use for before/after and arm-vs-arm work: the degenerate case that bit this campaign was
    two arms which agreed only because neither had been evaluated.
    """
    va = require_finite(a, f"{what} (side A)")
    vb = require_finite(b, f"{what} (side B)")
    if len(va) != len(vb):
        raise EmptyComparison(
            f"{what}: paired comparison needs equal lengths, got {len(va)} vs {len(vb)}"
        )
    return va, vb


def argmax_finite(scores: Sequence[float], what: str) -> int:
    """``argmax`` that refuses when no score is finite, instead of returning index 0.

    ``max``/``np.argmax`` over an all-``-inf`` sequence returns the FIRST element, which
    then reads as a selected winner. That is exactly how an unevaluated objective promotes a
    champion silently.
    """
    require_finite(scores, what)
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


class MarginTooSmall(ValueError):
    """A winner was selected by less than the margin the scoring rule can actually resolve.

    ``argmax`` answers "which is largest", never "is the difference real". When the score is
    a mean over folds of ``rho / ceiling``, a change in how the ceiling is computed reweights
    the folds — so a selection decided by a margin smaller than that reweighting can move is
    an artifact of the denominator convention, not a measurement.
    """


def reweighting_margin_bound(weights: Iterable[float]) -> float:
    """Largest RELATIVE shift a per-fold reweighting can induce in a mean-of-ratios score.

    Closed form rather than sampled: if each fold's contribution is scaled by ``w_f``, the
    achievable relative change in a mean-over-folds is bounded by the weights' relative
    half-range ``(max - min) / (max + min)`` — attained when the two candidates put their
    advantage on opposite extremes of the weight range.

    For the Spearman-Brown length correction the weight is ``(1 + c) / 2`` per fold, so pass
    ``[(1 + c) / 2 for c in ceilings]``. On this ledger's registered ceilings
    ([0.709, 0.815]) that gives 0.0301, and a 400k-pair random search found no flip at a
    margin above 0.0056 — i.e. the closed form is the conservative side of the empirical one,
    which is the direction a guard should err in.
    """
    w = require_finite(weights, "reweighting weights")
    lo, hi = min(w), max(w)
    if lo <= 0.0:
        raise EmptyComparison(
            f"reweighting weights must be positive to bound a ratio shift, got min {lo!r}"
        )
    return (hi - lo) / (hi + lo)


def require_margin(
    scores: Sequence[float], what: str, *, min_margin: float, relative: bool = True
) -> int:
    """``argmax_finite`` that also refuses when the top two are closer than ``min_margin``.

    ``relative=True`` (default) compares the gap against ``min_margin * |winner|``, matching
    :func:`reweighting_margin_bound`, which is a relative quantity. Pass ``relative=False``
    for an absolute threshold in the score's own units.

    Raises :class:`MarginTooSmall` rather than returning a winner, for the same reason
    ``tune_lolo`` raises rather than tie-breaking: a champion chosen inside the noise floor is
    indistinguishable from a real one in the output.
    """
    if min_margin < 0.0:
        raise ValueError(f"min_margin must be non-negative, got {min_margin!r}")
    best = argmax_finite(scores, what)
    if len(scores) < 2:
        return best
    runner_up = max(s for i, s in enumerate(scores) if i != best)
    gap = float(scores[best]) - float(runner_up)
    threshold = min_margin * abs(float(scores[best])) if relative else min_margin
    if gap < threshold:
        kind = "relative" if relative else "absolute"
        raise MarginTooSmall(
            f"{what}: winner beats the runner-up by {gap:.6g}, below the {kind} "
            f"minimum-resolvable margin {threshold:.6g} (min_margin={min_margin:.6g}). "
            f"The selection is inside what the scoring rule can resolve — widen the margin, "
            f"add folds/seeds, or report a tie; do not read it as a winner."
        )
    return best


def all_distinct(values: Sequence[float], what: str, *, tol: float = 0.0) -> bool:
    """Whether ``values`` are pairwise distinct — the cheap test for a hidden invariant.

    ``alt``, ``imbalance``, ``sfr`` and ``Genkey.index_imbalance_pct`` all turned out to be
    invariants that TIE layouts while reading as agreement. A distinctness check over a
    perturbed set is how each was caught; run it before crediting a per-gauge win count.
    """
    vals = require_finite(values, what)
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if abs(vals[i] - vals[j]) <= tol:
                return False
    return True


#: Lowest wpm bucket the non-regression gate protects.
#:
#: The ask was "high WPM buckets" -- PLURAL -- and a top-bucket-only gate has a measured hole: a
#: -0.20 collapse in 100-120 passed cleanly while 120-140 was held flat. So the scope is a FLOOR,
#: not a single bucket. 80 is where it sits because the campaign's own posture data puts the touch
#: typists there: 74.8% of participants at >=80 wpm use 9-10 fingers, rising to 84.7% at >=120,
#: while the 40-60 band is where the slow-typing regime dominates. Below the floor a slow-for-fast
#: trade stays a separate decision this gate must not settle.
HIGH_WPM_FLOOR = 80

#: Tolerance for the high-wpm non-regression gate, in rho units.
#:
#: Derived from what the gate must and must not catch, not chosen for roundness. It has to pass
#: search/seed wobble and still refuse the measured blend regression, and HIGHWPM-1 measured that
#: regression at 0.0733 in the 120-140 bucket -- an order of magnitude above this floor. 0.005 is
#: also the ranking-degradation bar the arm gates already use, so a reader meets one number twice
#: rather than two numbers once.
HIGH_WPM_TOLERANCE = 0.005


class HighWpmRegression(ValueError):
    """A candidate lost accuracy in the FASTEST wpm bucket, so it is refused rather than ranked.

    Fast and slow typing are different motor regimes, and a layout objective is aimed at people
    who have stopped being slow. HIGHWPM-1 measured a blended objective giving up rho in every
    bucket against the shipped ``ms/char`` -- **worst in the fastest** (120-140: -0.0733) -- while
    ``ms/char`` got *better* with speed. That structure had been computed all along by
    :func:`keybo.training.validate._per_bucket_rho` and never gated on, so it took a human noticing
    to surface it.

    Raised (not returned) for the same reason :class:`MarginTooSmall` is: a candidate that regresses
    expert typing is indistinguishable from a good one once it is a row in a leaderboard.
    """


def bucket_regression_report(
    candidate: Mapping[int, float],
    baseline: Mapping[int, float],
    what: str,
    *,
    tolerance: float = HIGH_WPM_TOLERANCE,
    floor: int = HIGH_WPM_FLOOR,
) -> dict:
    """The gate's verdict as a serializable dict — including when it did NOT run.

    ``gated`` says whether a verdict was reachable at all; ``passed`` is ``None`` when it was not.
    Both are explicit because an artifact that merely omits a verdict reads identically whether the
    gate ran and passed or never ran — the ambiguity that let the ``lolo`` tau gate report a pass
    while checking nothing (TAUGATE-1).

    Never raises: use it for reporting, and :func:`require_no_high_wpm_regression` to enforce.
    """
    deltas = {
        bucket: float(candidate[bucket]) - float(baseline[bucket])
        for bucket in sorted(baseline)
        if bucket in candidate
        and math.isfinite(float(baseline[bucket]))
        and math.isfinite(float(candidate[bucket]))
    }
    top = max(baseline) if baseline else None
    high = {b: d for b, d in deltas.items() if b >= floor}
    gated = bool(baseline) and top in deltas and bool(high)
    regressing = sorted(b for b, d in high.items() if d < -tolerance)
    report: dict = {
        "candidate": what,
        "gated": gated,
        "passed": None,
        "tolerance": float(tolerance),
        "high_wpm_floor": int(floor),
        "gated_buckets": sorted(high),
        "regressing_high_buckets": regressing,
        "top_bucket": top,
        "top_bucket_delta": deltas.get(top) if gated else None,
        "worst_bucket": min(deltas, key=lambda b: deltas[b]) if deltas else None,
        "worst_high_bucket": min(high, key=lambda b: high[b]) if high else None,
        "deltas": deltas,
    }
    if gated:
        report["passed"] = not regressing
    return report


def require_no_high_wpm_regression(
    candidate: Mapping[int, float],
    baseline: Mapping[int, float],
    what: str,
    *,
    tolerance: float = HIGH_WPM_TOLERANCE,
    floor: int = HIGH_WPM_FLOOR,
) -> dict:
    """Refuse ``candidate`` if it regresses rho in ANY bucket at or above ``floor``.

    Both maps are ``bucket start wpm -> rho`` (the shape ``_per_bucket_rho`` returns). Returns the
    :func:`bucket_regression_report` on success so a caller can serialize the passing verdict too.

    Deliberately scoped to the TOP bucket. A candidate that trades slow-typist accuracy for fast is
    a different decision, and folding it in here would make one gate quietly settle two questions.
    The full per-bucket ``deltas`` ride along in the report for whoever wants that argument.

    An absent or non-finite top bucket is REFUSED, not passed: "not measured" is not "did not
    regress" — the same absence-is-not-disproof rule that a wrongly-closed line of inquiry in this
    campaign was built on.
    """
    report = bucket_regression_report(candidate, baseline, what, tolerance=tolerance, floor=floor)
    if not report["gated"]:
        top = max(baseline) if baseline else None
        raise HighWpmRegression(
            f"{what}: the top wpm bucket ({top}) was not measured for this candidate, so the "
            f"high-wpm gate could not run. 'Not measured' is not 'did not regress' — supply the "
            f"bucket (lower min_bucket_cells, or widen the fold) or state explicitly that the "
            f"result is ungated."
        )
    if not report["passed"]:
        offenders = report["regressing_high_buckets"]
        detail = ", ".join(f"{b}: {report['deltas'][b]:+.4g}" for b in offenders)
        raise HighWpmRegression(
            f"{what}: rho regresses in {len(offenders)} of {len(report['gated_buckets'])} high-wpm "
            f"buckets (at or above {floor} wpm), beyond the {tolerance:.6g} tolerance — {detail}. "
            f"Worst high bucket: {report['worst_high_bucket']}. Fast and slow typing are different "
            f"regimes and this objective is aimed at people who have stopped being slow, so a "
            f"high-wpm regression is refused rather than ranked. All per-bucket deltas: "
            f"{ {b: round(d, 4) for b, d in report['deltas'].items()} }."
        )
    return report
