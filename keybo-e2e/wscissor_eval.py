"""WSCISSOR-GEN — the corpus board extended with an IN-LOOP wide-support scissor axis.

Wraps :mod:`corpus_eval` (the campaign's established evaluator, reused verbatim — the six
objectives, the ceiling-fraction normalized floor, the arm A/B kmstats tabling, the invariant
community gauges) and adds the two severity-gauge shares as fast bilinear slot-pair forms so
they can be an EA objective rather than a post-hoc score:

    wscissor  = ScissorSeverity.share(layout, SeverityWeights(support="wide",  ...))
    nscissor  = ScissorSeverity.share(layout, SeverityWeights(support="narrow", ...))

Both are computable side by side on every layout, which is what the optimizing-the-ruler guard
needs: the ARM optimizes one of them, and BOTH are read back out.

WHY A NEW FAST PATH IS NEEDED, AND WHAT IT IS NOT
-------------------------------------------------
The board's existing served ``scissor`` axis is ``tb_objective_ref.scissor_event_cost``, which is
*already* a wide-support gauge (no adjacency gate; a 0.60 factor for non-adjacency) carrying
biomechanical/direction/orientation factors in heterogeneous comfort units. The severity gauge is
a different object: a **corpus share** in percent, with the adjacency gate either hard-on
(narrow) or hard-off (wide), weighted only by the declared preference. So this is not a
re-implementation of the tb axis; it is the SCISSOR-SEVERITY gauge made batch-evaluable.

Exactness, not approximation
----------------------------
For a fixed 30-char charset every layout covers exactly the same bigrams, so the
layout-restricted denominator is a CONSTANT and the numerator is bilinear in the placement:

    share(layout) = 100 * sum_{a,b} F[a,b] * W[slot(a), slot(b)] / T

where ``W[i,j]`` is the severity of a bigram whose first char sits in slot i and second in slot j
(zero off the support) and ``T`` is the constant on-key bigram mass. That is an identity, not an
approximation, and ``test_wscissor_eval.py`` pins it against ``ScissorSeverity.share`` at 0.0.

MODELED/gauge only. No realized or observed-speed claim. Every number is conditional on a stated
preference and weighted by a named corpus.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus_eval as CE  # noqa: E402

from keybo.geometry import ROW_STAGGERED_30 as GEOM  # noqa: E402
from keybo.scoring.scissor_severity import (  # noqa: E402
    SUPPORTS,
    ScissorSeverity,
    SeverityWeights,
    bigram_severity,
)

#: The preregistered headline preference, on each support. `down`/`pinky`/`ratio` are
#: SCISSOR-SEVERITY-1's registered P (docs/scissor-severity-preregistration.md section 3); only
#: the support differs between the two.
P_WIDE = SeverityWeights(pinky=2.0, ring_ratio=0.5, down=1.5, support="wide")
P_NARROW = SeverityWeights(pinky=2.0, ring_ratio=0.5, down=1.5, support="narrow")
FLAT_WIDE = SeverityWeights(support="wide")
FLAT_NARROW = SeverityWeights(support="narrow")

#: The seven objectives of a wscissor-graded arm. Index 6 is the ADDED axis.
OBJECTIVE_ORDER_7 = [
    "neg_normfloor",
    "neg_mean",
    "scissor",
    "lsb",
    "sfb",
    "sfs",
    "wscissor",
]


def severity_slot_matrix(weights: SeverityWeights) -> np.ndarray:
    """(31,31) severity of an ordered bigram placed at (slot_i, slot_j); 0 off the support.

    Row/col 30 is the space slot, which carries no key, so it is identically zero — matching
    :class:`ScissorSeverity`'s convention that a pair touching an absent char contributes to
    neither numerator nor denominator.
    """
    positions = list(GEOM.slots)
    if len(positions) != 30:
        raise ValueError(f"expected the 30-slot board, got {len(positions)}")
    matrix = np.zeros((CE.NSLOT, CE.NSLOT), dtype=np.float64)
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            matrix[i, j] = bigram_severity(GEOM, a, b, weights)
    return matrix


def bigram_pair_mass(bigrams: dict[str, int]) -> tuple[np.ndarray, float]:
    """((31,31) char-pair mass over the C30M charset + space, constant total).

    The total INCLUDES pairs touching space. That is not a stylistic choice: the reference
    denominator is :meth:`ScissorSeverity._totals`, which admits a bigram when
    ``all(layout.has_key(ch) for ch in bg)`` — and ``Layout.has_key(' ')`` is TRUE, because
    space IS a key on this board (``geometry.space_position == (0, 0)``, thumb). So
    space-touching bigrams are layout-covered and belong in the denominator.

    This deliberately differs from ``corpus_eval.build_kmstats_matrices``, whose ``bi_total``
    masks space out — kmstats' sfb/lsb convention. Matching the WRONG one of the two is silent:
    the numerator agrees to the last digit either way (space is in no scissor support, see
    :func:`severity_slot_matrix`), so only the denominator moves and the share is off by a
    constant factor (~1.5x on iWeb) that looks like a plausible share. The
    ``test_flat_narrow_reproduces_incumbent_oxey_scissor_share`` positive control is what
    distinguishes them, which is why it is pinned at exactly 0.0.
    """
    index = {c: i for i, c in enumerate(CE.C30M)}
    index[" "] = CE.SPACE
    mass = np.zeros((CE.NSLOT, CE.NSLOT), dtype=np.float64)
    for ngram, freq in bigrams.items():
        if len(ngram) != 2:
            continue
        if ngram[0] not in index or ngram[1] not in index:
            continue
        mass[index[ngram[0]], index[ngram[1]]] += freq
    return mass, float(mass.sum())


class SeverityGauge:
    """Batch-evaluable severity share on one corpus, for an arbitrary preference point."""

    def __init__(self, bigrams: dict[str, int]) -> None:
        self.mass, self.total = bigram_pair_mass(bigrams)
        if self.total <= 0:
            raise ValueError("corpus has no C30M-covered bigram mass")
        self._cache: dict[str, np.ndarray] = {}

    def matrix(self, weights: SeverityWeights) -> np.ndarray:
        key = weights.label()
        if key not in self._cache:
            self._cache[key] = severity_slot_matrix(weights)
        return self._cache[key]

    def share(self, perm: np.ndarray, weights: SeverityWeights) -> float:
        placed = self.matrix(weights)[perm[:, None], perm[None, :]]
        return 100.0 * float((self.mass * placed).sum()) / self.total

    def share_batch(self, perms: np.ndarray, weights: SeverityWeights) -> np.ndarray:
        """perms: (B,31) -> (B,) shares. Same identity as :meth:`share`, vectorized."""
        placed = self.matrix(weights)[perms[:, :, None], perms[:, None, :]]
        return 100.0 * np.einsum("ab,iab->i", self.mass, placed) / self.total


class WScissorBoard(CE.ArmBoard):
    """The campaign board plus the two severity shares, on ONE corpus.

    ``objective`` picks which severity share (if any) enters the EA's minimization vector:

      * ``"wide"``   -> 7 objectives, index 6 = wide-support severity share  (the wscissor arm)
      * ``"narrow"`` -> 7 objectives, index 6 = narrow-support severity share (the control arm)
      * ``"none"``   -> the campaign's original 6 objectives, byte-for-byte (the baseline arm)

    The two non-optimized shares are still READABLE via :meth:`severity_axes` on any layout, so
    the ruler guard never has to re-run a search to score the other gauge.
    """

    def __init__(
        self,
        corpus: str = "iweb",
        arm: str = "A",
        ceilings=None,
        objective: str = "wide",
        weights: SeverityWeights | None = None,
        wfd_mode: str = "corrected",
    ) -> None:
        super().__init__(corpus=corpus, arm=arm, ceilings=ceilings, wfd_mode=wfd_mode)
        if objective not in ("wide", "narrow", "none"):
            raise ValueError(f"objective must be wide|narrow|none, got {objective!r}")
        self.objective = objective
        bigrams, _skip, _tri = CE.corpus_tables(corpus)
        self.severity = SeverityGauge(bigrams)
        base = weights or SeverityWeights(pinky=2.0, ring_ratio=0.5, down=1.5)
        # The SAME preference on both supports — only the predicate differs, which is the
        # whole point of the comparison (SCISSOR-SEVERITY-1 proved the weights cannot move a
        # relative per-bin verdict; the support can).
        self.w_wide = SeverityWeights(
            pinky=base.pinky, ring_ratio=base.ring_ratio, down=base.down, support="wide"
        )
        self.w_narrow = SeverityWeights(
            pinky=base.pinky, ring_ratio=base.ring_ratio, down=base.down, support="narrow"
        )
        self.nobj = 6 if objective == "none" else 7

    # -- the added axes -------------------------------------------------------------------
    def severity_axes(self, layout: str) -> dict[str, float]:
        """All four severity readings for one layout: {wide,narrow} x {P, flat}."""
        perm = CE.perm_of(layout)
        return {
            "wscissor_P": self.severity.share(perm, self.w_wide),
            "nscissor_P": self.severity.share(perm, self.w_narrow),
            "wscissor_flat": self.severity.share(perm, FLAT_WIDE),
            "nscissor_flat": self.severity.share(perm, FLAT_NARROW),
        }

    def severity_axes_slow(self, layout: str) -> dict[str, float]:
        """The same four readings via ``ScissorSeverity.share`` — the SLOW reference path, with
        zero fast-path reuse. Used to verify every reported layout at zero error."""
        from keybo.layout import Layout

        bigrams, _skip, _tri = CE.corpus_tables(self.corpus)
        gauge = ScissorSeverity(bigrams)
        obj = Layout(layout, GEOM)
        return {
            "wscissor_P": gauge.share(obj, self.w_wide),
            "nscissor_P": gauge.share(obj, self.w_narrow),
            "wscissor_flat": gauge.share(obj, FLAT_WIDE),
            "nscissor_flat": gauge.share(obj, FLAT_NARROW),
        }

    def axes12(self, layout: str, floor_kind: str = "norm") -> dict[str, float]:
        """The 10 board axes plus the 2 severity-P axes — the full reporting frame."""
        out = dict(self.axes(layout, floor_kind))
        sev = self.severity_axes(layout)
        out["wscissor"] = sev["wscissor_P"]
        out["nscissor"] = sev["nscissor_P"]
        return out

    def axes12_slow(self, layout: str, floor_kind: str = "norm") -> dict[str, float]:
        """The 12 axes via the SLOW reference paths only — zero fast-path reuse.

        Composes ``ArmBoard.axes_slow`` (KmStats.stats, ComfortObjective.values, the zero-reuse
        wfd reference) with ``severity_axes_slow`` (``ScissorSeverity.share`` on a real
        ``Layout``). This is what a REPORTED dominator must reproduce.
        """
        out = dict(self.axes_slow(layout, floor_kind))
        sev = self.severity_axes_slow(layout)
        out["wscissor"] = sev["wscissor_P"]
        out["nscissor"] = sev["nscissor_P"]
        return out

    # -- EA inner loop --------------------------------------------------------------------
    def evaluate_batch(self, movables: np.ndarray) -> np.ndarray:
        """(B,30) char->slot -> (B,nobj) MINIMIZATION objectives.

        Columns 0..5 are the campaign's originals, delegated to ``ArmBoard.evaluate_batch`` so
        this arm cannot silently diverge from the established six. Column 6, when present, is
        the severity share on the arm's support (a percentage, minimized).
        """
        base = super().evaluate_batch(movables)
        if self.objective == "none":
            return base
        batch = movables.shape[0]
        perms = np.empty((batch, CE.NSLOT), dtype=np.int64)
        perms[:, :30] = movables
        perms[:, 30] = CE.SPACE
        weights = self.w_wide if self.objective == "wide" else self.w_narrow
        added = self.severity.share_batch(perms, weights)
        out = np.empty((batch, 7), dtype=np.float64)
        out[:, :6] = base
        out[:, 6] = added
        return out


#: The 10-axis dominance frame plus the two severity axes. Both severity shares are
#: LOWER-better (they are strain shares), like `scissor`/`lsb`/`sfb`/`sfs`.
AXES12 = [*CE.AXES, "wscissor", "nscissor"]
SIGN12 = {**CE.SIGN, "wscissor": -1, "nscissor": -1}


def oriented12(axes: dict) -> np.ndarray:
    """The 12 axes as an all-'higher-is-better' vector."""
    return np.array([SIGN12[a] * axes[a] for a in AXES12])


def dominates12(cand: dict, inc: dict, atol: float = 1e-9) -> tuple[bool, int, int]:
    """Dominance on all 12 axes. Returns (is_dominator, n_ge, n_strictly_gt)."""
    cv, iv = oriented12(cand), oriented12(inc)
    n_ge = int(np.sum(cv >= iv - atol))
    n_gt = int(np.sum(cv > iv + atol))
    return (n_ge == len(AXES12) and n_gt >= 1), n_ge, n_gt


__all__ = [
    "AXES12",
    "FLAT_NARROW",
    "FLAT_WIDE",
    "OBJECTIVE_ORDER_7",
    "P_NARROW",
    "P_WIDE",
    "SIGN12",
    "SUPPORTS",
    "SeverityGauge",
    "WScissorBoard",
    "bigram_pair_mass",
    "dominates12",
    "oriented12",
    "severity_slot_matrix",
]
