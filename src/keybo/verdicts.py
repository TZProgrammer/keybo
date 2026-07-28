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
from collections.abc import Iterable, Sequence


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
