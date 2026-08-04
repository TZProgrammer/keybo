"""FRAME COLLAPSE — what a feature frame CANNOT distinguish, with no model and no training.

INTERPFRAME-1 discovered this arm's most transferable result and left it as prose plus a throwaway
driver (``agent-artifacts/interpframe/resolution.py``). This module is that diagnostic, promoted to a
frame-agnostic surface. The question it answers, in ~2 seconds, with NO model, NO SHAP and NO
training:

> Given a featurizer and a geometry, how many of the geometry's position cells does the featurizer
> make INDISTINGUISHABLE from some other cell — and what is the best-case error ANY model on that
> frame could achieve against a known per-cell target?

WHY IT IS WORTH ANYTHING
------------------------
Cells with byte-identical feature rows are indistinguishable to every possible model on that frame.
They MUST receive the same prediction — not "usually do", not "do unless the model is very good": the
model's input is literally the same vector. So the WITHIN-GROUP dispersion of the true target across
such a group is an error no amount of training, tuning or capacity can remove. It is a property of
the FRAME, measurable before a model exists.

INTERPFRAME-1 spent 34 minutes of LOLO establishing that its 10-column ``interp.1`` frame kept RANK
accuracy (Δρ −0.00047, τ 1.0) while losing MAGNITUDE accuracy (wmae +58%). This diagnostic predicts
exactly that signature in two seconds: the collapsed groups are still ORDERED correctly relative to
one another, but every cell inside a group can only be predicted at the group's central value.
**Rank-preserving, magnitude-destroying.** On the campaign's own frames, over the 961 position cells
of the production surface:

===================  ========  ==============  =================  ==========  ==================
frame                 columns  distinct rows   collapsed cells    mass share  floor (group mean)
===================  ========  ==============  =================  ==========  ==================
served bigram              20             765                380      53.51%          ~0 ms
``interp.1``               10             378                817      93.19%       2.2399 ms
``interp.1`` + ``wpm``     11             378                817      93.19%       2.2399 ms
===================  ========  ==============  =================  ==========  ==================

The third row is the diagnostic earning its keep: adding ``wpm`` back changes the distinct-row count
by ZERO, so it adds no RESOLUTION and therefore cannot buy magnitude accuracy — which is precisely
what INTERPFRAME-1's ``interp-wpm`` arm found the expensive way (Δwmae +0.005, i.e. nothing).

THE WITHIN-GROUP FLOOR, AND THE ESTIMATOR CORRECTION THIS MODULE SHIPS
---------------------------------------------------------------------
Every cell in a collapse group receives ONE prediction ``p_g``. The floor is the error of the BEST
possible choice of those predictions:

    floor(L) = min over {p_g}  sum_i w_i L(t_i, p_g(i)) / sum_i w_i

⚠ **The minimizer depends on the loss, and the two are NOT interchangeable.** ``resolution.py`` used
the weighted MEAN for a ``wmae`` floor. Absolute error is minimized by the weighted MEDIAN; the mean
minimizes SQUARED error. A mean-based ``wmae`` number is therefore the achieved error of one specific
predictor (the group-mean predictor) — a perfectly valid "a model can do at least this well" UPPER
bound, but NOT the greatest lower bound it was published as. So this module reports BOTH, named for
what they are:

* :attr:`FrameCollapse.floor_wmae` — median-based. The true L1 floor: no model on this frame can beat
  it. **This is the number to quote as a floor.**
* :attr:`FrameCollapse.floor_wmae_at_group_mean` — mean-based. Exactly INTERPFRAME-1's published
  quantity, kept so the ledger stays checkable, and it is genuinely the right number if you want "how
  well would a least-squares-fitted model do", since that is what such a model converges to per group.
* :attr:`FrameCollapse.floor_wrmse` — the L2 floor, for which the mean IS the minimizer.

On the campaign's real frames the two L1 numbers agree to 4 dp (see
:mod:`tests.analysis.test_frame_collapse`), so the published 2.2399 ms survives as a floor — but that
is a measured coincidence of these frames' group structure, not an identity, and a synthetic 2-cell
group with unequal weights separates them by 50%.

⚠ THE TARGET'S PROVENANCE DECIDES WHETHER A ZERO FLOOR MEANS ANYTHING
--------------------------------------------------------------------
**A frame's floor against a target THAT FRAME GENERATED is an IDENTITY, not a measurement, and it is
always exactly zero.** If ``t`` is the output of a deterministic model fed this frame's own rows, then
two cells with identical rows are fed the identical vector to the identical model and MUST get the
identical target. The floor is then structurally incapable of being non-zero, whatever the frame's
resolution.

🔴 **This is not hypothetical: it is the correct reading of INTERPFRAME-1's published served-frame
floor.** The measurement is exact — ``max|T2 - mean_seeds(model.predict_ms(served_X))| =
0.000000e+00`` over all 961 cells, because :class:`~keybo.analysis.timecard.TimeSurface` builds
``_T2`` from ``TableBigramScorer(bigram_reg31_seed*)``, which featurizes with the **served** frame.
All 184 of the served frame's collapse groups have target spread exactly 0. So "served floor EXACTLY
0.0000 ms" is an identity, and the ledger's served-vs-interp floor pair is a one-sided measurement
rather than a two-frame contrast. ⇒ :attr:`FrameCollapse.target_is_self_generated` FLAGS this
signature, with :attr:`~FrameCollapse.groups_with_target_spread` /
:attr:`~FrameCollapse.max_group_target_spread` beside it so a reader can judge rather than trust a
boolean. It is a flag and not an exception because a genuinely perfectly-resolved frame produces the
same signature — the flag says "this zero is uninformative *unless you know where the target came
from*", which only the caller does.

**What that does NOT retract.** The collapse structure itself — 765 vs 378 distinct rows, 53.5% vs
93.2% collapsed mass — is TARGET-FREE, so INTERPFRAME-1's direction of conclusion (``interp.1``
destroys magnitude resolution the served frame keeps) stands untouched. And the ``interp.1`` floor IS
a real measurement: ``T2`` is not interp-generated, and 137 of its 234 collapse groups have non-zero
spread (largest 93.17 ms). The instrument can also be shown to FAIL, which is what makes those
survivals meaningful: scored against ``mean_c Tcond[a,b,c]`` — a target the TRIGRAM model produced
from trigram features — the served frame's floor is **0.5522 ms**, comfortably non-zero.

⇒ **A fair frame-VS-frame floor needs a target NEITHER frame generated.** The campaign has no such
held-out per-cell millisecond table on disk (every surface is model-generated from one of these
frames), so that remains open; see the report's open items.

FLOAT TOLERANCE IS A FIRST-CLASS PARAMETER (and the 765-vs-775 story)
--------------------------------------------------------------------
Grouping needs an equality rule on float rows. ``tol=0.0`` — the DEFAULT — means EXACT bitwise
equality, the only rule that is transitive, reproducible across BLAS builds, and free of a tuned
constant. ``tol>0`` quantizes each column to ``round(x / tol)``. Exactly one coarsening guarantee
holds, and the difference between it and the one you might assume is load-bearing:

1. **``distinct_feature_rows(tol) <= distinct_feature_rows(exact)`` for EVERY ``tol >= 0``** — TRUE,
   and pinned by a test. Exact-equal rows have equal quantizations, so every quantized partition is a
   coarsening OF THE EXACT ONE. ⇒ **A tolerance can never RAISE the count above the exact count.** If
   two runs of this diagnostic disagree in the direction of MORE rows, the cause is not the tolerance.
2. ⚠ **``distinct_feature_rows`` is NOT monotone in ``tol`` between two nonzero tolerances** — a
   COARSER grid can SPLIT a pair a FINER grid merged, because the bin BOUNDARIES move with ``tol``.
   The quantized partitions are therefore not a nested refinement chain. Measured on the served frame:
   ``tol=0.5`` gives 701 rows but ``tol=0.75`` gives **709**, a rise of 8. Minimal instance, and a test
   pins it so this docstring cannot be "corrected" into the false monotone claim: ``round(0.3/0.5) ==
   round(0.4/0.5) == 1`` (merged), while ``round(0.3/0.75) = 0 != 1 = round(0.4/0.75)`` (split).
3. **A ``tol>0`` grouping is a QUANTIZATION, not an equivalence "up to ``tol``".** Two rows within
   ``tol`` of each other may land in different bins, and two rows nearly ``2*tol`` apart may share one.
   The relation that IS monotone in ``tol`` is single-linkage-within-``tol``, but linkage CHAINS (``a~b``
   and ``b~c`` merges ``a`` with ``c`` however far apart they are) so it is not an equality rule, and at
   ``O(n^2 C)`` it is infeasible on the 29791-cell trigram space. Deliberately not shipped.
4. Quantization is per-column on RAW units and is therefore NOT scale-invariant across columns (a
   ``dx`` in key-widths and a ``wpm`` in words/minute get the same absolute step). ``tol=0.0`` avoids
   all four questions, which is why it is the default; a caller passing ``tol>0`` accepts them and
   must report the ``tol`` beside any number it produced.

⚠ **A worked instance of (1), because it cost a real reconciliation.** INTERPFRAME-1 published 765
distinct served rows; an independent reproduction got 775 and attributed the gap to "rounding
features at 12 decimals vs an exact comparison". That attribution is impossible on arithmetic alone,
and measurement confirms it: on this frame ``distinct_feature_rows`` is 765 at EVERY tolerance from
exact through ``1e-3`` — twelve orders of magnitude, not one row of movement. The real cause is that
the two runs used **different cell spaces that both happen to contain 961 cells**:

* ``ROW_STAGGERED_30.slots`` (30 letter keys) **+ ``space_position``** = 31 positions → **765**. This
  is the surface's own cell space (:class:`~keybo.analysis.timecard.TimeSurface` builds its tables
  over ``[*geometry.slots, geometry.space_position]``), and it is the correct one for a floor, since
  it is the space the target table, the corpus weights and the models all live on.
* ``ROW_STAGGERED_31.slots`` (31 letter keys, including the ANSI quote slot) with **no space** = 31
  positions → **775** (and ``interp.1`` 422, not 378).

⇒ **Always report the cell space, not just the cell count** — :attr:`FrameCollapse.n_positions` and
:attr:`FrameCollapse.includes_space` exist for exactly that reason.

WHAT THIS DIAGNOSTIC CANNOT TELL YOU
------------------------------------
It is a **NECESSARY-condition** instrument. Zero collapse does not imply a good frame.

1. **``resolution == 1.0`` says only that no two cells are FORCED to share a prediction.** Twenty
   columns of pure noise plus one cell-index column has perfect resolution and no predictive value.
2. **The floor is a LOWER bound on error, never a prediction OF the error.** ``interp.1``'s measured
   held-out wmae was 15.70 ms against a 2.24 ms floor: the floor explained 38.9% of the *gap* to the
   served frame, and nothing about the level.
3. **It says nothing about GENERALIZATION** — it is computed on the cell space, not on held-out data,
   so it cannot see overfitting, extrapolation or train/serve skew.
4. **It says nothing about the OPTIMIZER.** A frame with low collapse can still contain a null space a
   search exploits; see ``agent-artifacts/goodhart-row-blindness.md``, where the optimizer parked junk
   on the home row through a blindness LOLO could not see. High resolution is not safety.
5. **It is TARGET-RELATIVE.** Change the target (a different surface, corpus or WPM) and the floor
   changes on the same frame. Only the collapse structure is target-free.
6. **It cannot rank two frames' accuracy.** It bounds one frame's best case; a frame with the lower
   floor can still train worse.
7. **The mass share is CORPUS-relative** and inherits that corpus's biases.
8. **It sees only cells the featurizer is asked about.** Collapse is a property of the frame *over a
   cell space*; a frame that separates every cell of one geometry may collapse another's.
9. **A ``tol>0`` result is not "cells equal to within ``tol``"** — see tolerance point 3 above. Only
   ``tol=0`` supports the plain reading "these cells are indistinguishable".

Registered design and decision rules: ``agent-artifacts/framediag/FRAMEDIAG-preregistration.md``
(committed BEFORE any number this module emits existed).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from keybo.geometry import Geometry, Position

#: A featurizer: takes the geometry and one ORDERED tuple of positions (length = the frame's order)
#: and returns that cell's feature row. Both the ``np.ndarray`` and ``dict[str, float]`` shapes
#: :mod:`keybo.features.ngram` emits are accepted; a dict is read in ITS OWN key order, so a
#: featurizer whose key order varies between calls is a caller bug (every frame in this repo emits a
#: fixed order, and a test pins it).
Featurizer = Callable[[Geometry, tuple[Position, ...]], "np.ndarray | dict[str, float]"]


def cell_positions(geometry: Geometry, *, include_space: bool = True) -> list[Position]:
    """The position list a frame's cell space is built from.

    ``include_space=True`` (the default) appends :attr:`Geometry.space_position`, reproducing the
    space :class:`~keybo.analysis.timecard.TimeSurface` builds its tables over — so a floor computed
    against one of those tables lines up cell-for-cell. Pass ``False`` for a letters-only space.

    ⚠ This is the knob behind the 765-vs-775 reconciliation in the module docstring: on
    ``ROW_STAGGERED_30`` with space and on ``ROW_STAGGERED_31`` without it, this returns 31 positions
    EITHER WAY — same cell COUNT, different cell SPACE, different answers. Report which you used.
    """
    slots = list(geometry.slots)
    return [*slots, geometry.space_position] if include_space else slots


def _row_values(row: np.ndarray | dict[str, float]) -> np.ndarray:
    """One cell's feature row as a float vector, accepting both shapes a featurizer may return."""
    if isinstance(row, dict):
        return np.asarray(list(row.values()), dtype=np.float64)
    return np.asarray(row, dtype=np.float64).ravel()


def feature_matrix(
    featurizer: Featurizer,
    geometry: Geometry,
    *,
    order: int = 2,
    positions: Sequence[Position] | None = None,
    include_space: bool = True,
) -> np.ndarray:
    """The ``(P**order, C)`` feature matrix: one row per ordered cell, in odometer order.

    Cell ``i`` is ``itertools.product(positions, repeat=order)[i]``, which is C-order over the
    ``(P,) * order`` index grid — the SAME ordering ``np.ndarray.ravel()`` produces on the surface's
    own tables, so a target flattened from ``TimeSurface._T2`` (or from
    :meth:`~keybo.analysis.timecard.TimeSurface.triple_ms_table`) aligns with these rows WITHOUT a
    permutation. Getting that wrong would silently pair each cell with another cell's target, so it
    is asserted by a test rather than left to a comment.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    pos = (
        list(positions)
        if positions is not None
        else cell_positions(geometry, include_space=include_space)
    )
    if not pos:
        raise ValueError("empty position list: no cells to diagnose")
    return np.vstack(
        [_row_values(featurizer(geometry, cell)) for cell in itertools.product(pos, repeat=order)]
    )


def group_cells(X: np.ndarray, *, tol: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Group cells by feature row. Returns ``(group_index_per_cell, group_sizes)``.

    ``tol=0.0`` is EXACT bitwise equality. ``tol>0`` quantizes each column to ``round(x / tol)``
    first, which coarsens the EXACT partition — so ``len(sizes) <= len(sizes at tol=0)`` always — but
    is NOT monotone between two nonzero tolerances (module docstring point 2). Both are pinned by
    tests.

    ``NaN`` in a feature row would make exact grouping non-reflexive (``nan != nan``), which would
    silently report every NaN-carrying cell as its own group — i.e. as perfectly resolved, the exact
    opposite of the truth. Refused loudly instead.
    """
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).any(axis=1).sum())
        raise ValueError(
            f"featurizer emitted non-finite values in {bad} of {X.shape[0]} cells; grouping is "
            "undefined (nan != nan would report each such cell as perfectly resolved)"
        )
    keys = X if tol == 0.0 else np.round(X / tol)
    _uniq, inverse, sizes = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return np.asarray(inverse).ravel(), sizes


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The LOWER weighted median: the smallest value whose cumulative weight reaches half the total.

    The L1 minimizer over a weighted point set is an interval whenever the weight splits exactly in
    half; every point in it has identical cost, so this tie convention cannot change any floor —
    only which minimizer is named. Registered in the prereg §2 so the choice is not silent.
    """
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    total = w.sum()
    if total <= 0:
        return float(v[0])
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * total, side="left")])


@dataclass(frozen=True)
class FrameCollapse:
    """What one feature frame cannot distinguish over one cell space, and the cost of that.

    Floors are ``None`` when no ``target`` was supplied: the collapse structure is target-free, the
    floor is not (non-claim 5). ``None`` rather than ``0.0`` or ``nan`` because a zero floor is a
    REAL and meaningful answer (the served frame's), so a sentinel that could be mistaken for one
    would be a lie about the strongest result this diagnostic can return.
    """

    n_cells: int
    n_columns: int
    order: int
    n_positions: int
    includes_space: bool
    tol: float
    distinct_feature_rows: int
    collapsed_cells: int
    largest_group: int
    #: ``distinct_feature_rows / n_cells`` — the headline single number. 1.0 = nothing collapsed.
    resolution: float
    #: Share of CELLS in a group of size > 1.
    collapsed_share: float
    #: Share of WEIGHT in a group of size > 1. Equals ``collapsed_share`` under uniform weights.
    mass_share_collapsed: float
    weighted: bool
    #: The true L1 floor: weighted MAD about each group's weighted MEDIAN. No model can beat it.
    floor_wmae: float | None = None
    #: INTERPFRAME-1's published quantity: the same error AT THE GROUP MEAN. An achievable error of
    #: one predictor, hence an UPPER bound on ``floor_wmae`` — see the module docstring.
    floor_wmae_at_group_mean: float | None = None
    #: The L2 floor (root weighted MSE about each group's weighted mean, which IS the L2 minimizer).
    floor_wrmse: float | None = None
    #: ``floor_wmae`` with uniform weights, whatever ``weights`` was — the unweighted companion
    #: ``resolution.py`` reported as ``floor_umae`` (at the group mean; this one is at the median).
    floor_umae: float | None = None
    floor_umae_at_group_mean: float | None = None
    #: ⚠ ``True`` when EVERY collapse group has EXACTLY zero target spread while at least one group is
    #: non-trivial — the signature of a target the frame ITSELF generated, in which case the zero floor
    #: is an IDENTITY and says nothing about the frame. See :func:`frame_collapse` and the
    #: self-generated-target note in the module docstring.
    target_is_self_generated: bool = False
    #: Number of collapse groups whose target spread is non-zero, over the number of collapse groups.
    #: ``0/N`` with ``N>0`` is what raises the flag above; it is reported as a number so a reader can
    #: judge for themselves rather than trusting a boolean.
    groups_with_target_spread: int | None = None
    n_collapse_groups: int | None = None
    #: The largest within-group target spread (``max - min``), in target units. 0.0 with a non-trivial
    #: group structure is the tautology signature.
    max_group_target_spread: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-ready, in report order. ``None`` floors stay ``None`` (see the class docstring)."""
        return {
            "n_cells": self.n_cells,
            "n_columns": self.n_columns,
            "order": self.order,
            "n_positions": self.n_positions,
            "includes_space": self.includes_space,
            "tol": self.tol,
            "distinct_feature_rows": self.distinct_feature_rows,
            "resolution": self.resolution,
            "collapsed_cells": self.collapsed_cells,
            "collapsed_share": self.collapsed_share,
            "mass_share_collapsed": self.mass_share_collapsed,
            "largest_group": self.largest_group,
            "weighted": self.weighted,
            "floor_wmae": self.floor_wmae,
            "floor_wmae_at_group_mean": self.floor_wmae_at_group_mean,
            "floor_wrmse": self.floor_wrmse,
            "floor_umae": self.floor_umae,
            "floor_umae_at_group_mean": self.floor_umae_at_group_mean,
            "target_is_self_generated": self.target_is_self_generated,
            "groups_with_target_spread": self.groups_with_target_spread,
            "n_collapse_groups": self.n_collapse_groups,
            "max_group_target_spread": self.max_group_target_spread,
            **({"extra": self.extra} if self.extra else {}),
        }


def _floors(
    target: np.ndarray, weights: np.ndarray, inverse: np.ndarray, n_groups: int
) -> tuple[float, float, float]:
    """``(floor_wmae_median, floor_wmae_at_group_mean, floor_wrmse)`` for one weighting.

    Both L1 numbers are computed here rather than in two places: the ONLY difference between them is
    which within-group constant is used, and computing them side by side is what makes the
    mean-vs-median distinction auditable instead of a comment.
    """
    total = weights.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    gw = np.bincount(inverse, weights=weights, minlength=n_groups)
    gwt = np.bincount(inverse, weights=weights * target, minlength=n_groups)
    mean_of = np.divide(gwt, gw, out=np.zeros_like(gwt), where=gw > 0)

    median_of = np.zeros(n_groups)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(n_groups + 1))
    for g in range(n_groups):
        idx = order[bounds[g] : bounds[g + 1]]
        if idx.size:
            median_of[g] = _weighted_median(target[idx], weights[idx])

    wmae = float((weights * np.abs(target - median_of[inverse])).sum() / total)
    wmae_mean = float((weights * np.abs(target - mean_of[inverse])).sum() / total)
    wrmse = float(np.sqrt((weights * (target - mean_of[inverse]) ** 2).sum() / total))
    return wmae, wmae_mean, wrmse


def _group_target_spreads(
    target: np.ndarray, inverse: np.ndarray, sizes: np.ndarray, n_groups: int
) -> np.ndarray:
    """``max - min`` of the target within each COLLAPSE group (size > 1); empty if none collapsed.

    Computed with two grouped reductions rather than a loop so it stays cheap at the 29791-cell
    trigram space. Singleton groups are excluded because their spread is trivially 0 and would dilute
    the self-generated-target signal into meaninglessness (on the served frame 777 of 961 cells are
    singletons).
    """
    hi = np.full(n_groups, -np.inf)
    lo = np.full(n_groups, np.inf)
    np.maximum.at(hi, inverse, target)
    np.minimum.at(lo, inverse, target)
    collapsed_groups = np.flatnonzero(sizes > 1)
    if collapsed_groups.size == 0:
        return np.zeros(0)
    return hi[collapsed_groups] - lo[collapsed_groups]


def frame_collapse(
    featurizer: Featurizer,
    geometry: Geometry,
    *,
    order: int = 2,
    include_space: bool = True,
    positions: Sequence[Position] | None = None,
    target: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    tol: float = 0.0,
) -> FrameCollapse:
    """Diagnose what ``featurizer`` cannot distinguish over ``geometry``'s cell space.

    Args:
        featurizer: called as ``featurizer(geometry, cell)`` where ``cell`` is a tuple of ``order``
            positions; returns that cell's feature row (array or dict). Any frame works, including
            ones not yet written — that is the point of the diagnostic (predict a frame's cost BEFORE
            training it). For this repo's frames, wrap the ``*_from_positions`` entry points, e.g.
            ``lambda g, c: bigram_features_from_positions(g, c, wpm=90.0)``.
        order: the frame's n-gram order. 2 = bigram (``P**2`` cells), 3 = trigram (``P**3``).
        include_space: whether the cell space includes ``geometry.space_position``. See
            :func:`cell_positions`; this is a REPORTED field because it changes the answer.
        positions: an explicit position list, overriding ``geometry``/``include_space`` selection.
        target: the TRUE per-cell value the frame would have to predict (ms), flattened in the same
            odometer order as :func:`feature_matrix`. Omit it and the floors are ``None``. For this
            repo: ``surface._T2.ravel()`` at ``order=2``, ``surface.triple_ms_table().ravel()`` at 3.
        weights: per-cell weight (corpus mass). Omit for uniform. Negative weights are refused —
            they would let one cell's error CANCEL another's and turn a floor into a fiction.
        tol: 0.0 (default) = exact bitwise grouping; >0 quantizes to ``round(x / tol)``.

    Returns:
        :class:`FrameCollapse`.

    Costs one featurizer call per cell and one ``np.unique`` — ~2 s for a 961-cell bigram frame.
    """
    pos = (
        list(positions)
        if positions is not None
        else cell_positions(geometry, include_space=include_space)
    )
    X = feature_matrix(featurizer, geometry, order=order, positions=pos)
    n_cells = X.shape[0]
    inverse, sizes = group_cells(X, tol=tol)
    n_groups = int(len(sizes))
    is_collapsed = sizes[inverse] > 1
    collapsed_cells = int(is_collapsed.sum())

    if weights is None:
        w = np.ones(n_cells, dtype=np.float64)
        weighted = False
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != n_cells:
            raise ValueError(f"weights has {w.shape[0]} entries, expected {n_cells}")
        if (w < 0).any():
            raise ValueError("weights must be >= 0 (a negative weight lets errors cancel)")
        weighted = True
    w_total = w.sum()
    mass_share = float(w[is_collapsed].sum() / w_total) if w_total > 0 else 0.0

    floors: dict[str, float | None] = dict.fromkeys(
        (
            "floor_wmae",
            "floor_wmae_at_group_mean",
            "floor_wrmse",
            "floor_umae",
            "floor_umae_at_group_mean",
        )
    )
    provenance: dict = {"target_is_self_generated": False}
    if target is not None:
        t = np.asarray(target, dtype=np.float64).ravel()
        if t.shape[0] != n_cells:
            raise ValueError(f"target has {t.shape[0]} entries, expected {n_cells}")
        if not np.isfinite(t).all():
            raise ValueError("target contains non-finite values; the floor would be undefined")
        wmae, wmae_mean, wrmse = _floors(t, w, inverse, n_groups)
        floors["floor_wmae"] = wmae
        floors["floor_wmae_at_group_mean"] = wmae_mean
        floors["floor_wrmse"] = wrmse
        if weighted:
            umae, umae_mean, _ = _floors(t, np.ones(n_cells), inverse, n_groups)
        else:
            umae, umae_mean = wmae, wmae_mean
        floors["floor_umae"] = umae
        floors["floor_umae_at_group_mean"] = umae_mean

        spreads = _group_target_spreads(t, inverse, sizes, n_groups)
        n_cg = int(spreads.size)
        n_spread = int((spreads > 0).sum())
        provenance = {
            "target_is_self_generated": bool(n_cg > 0 and n_spread == 0),
            "groups_with_target_spread": n_spread,
            "n_collapse_groups": n_cg,
            "max_group_target_spread": float(spreads.max()) if n_cg else 0.0,
        }

    return FrameCollapse(
        n_cells=n_cells,
        n_columns=int(X.shape[1]),
        order=order,
        n_positions=len(pos),
        includes_space=geometry.space_position in pos,
        tol=float(tol),
        distinct_feature_rows=n_groups,
        collapsed_cells=collapsed_cells,
        largest_group=int(sizes.max()) if n_groups else 0,
        resolution=n_groups / n_cells,
        collapsed_share=collapsed_cells / n_cells,
        mass_share_collapsed=mass_share,
        weighted=weighted,
        **floors,
        **provenance,
    )


def tolerance_sweep(
    featurizer: Featurizer,
    geometry: Geometry,
    *,
    tols: Sequence[float] = (0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3),
    **kwargs,
) -> list[FrameCollapse]:
    """:func:`frame_collapse` at each tolerance — the answer to "is this number tolerance-sensitive?".

    A frame whose ``distinct_feature_rows`` is flat across this sweep has a tolerance-INDEPENDENT
    headline number (every bigram frame in this repo is flat out to ``tol=0.25``, a quarter key
    width). A frame that moves has a number that depends on a parameter, which belongs in whatever
    writes it down.

    ⚠ The sequence is bounded above by its ``tol=0`` entry but is NOT sorted: a rise between two
    nonzero tolerances is REAL behaviour, not a bug (module docstring point 2). ``exceeds_exact`` /
    ``rises`` in :func:`sweep_verdict` report the two separately for exactly that reason.
    """
    return [frame_collapse(featurizer, geometry, tol=t, **kwargs) for t in tols]


def sweep_verdict(results: Sequence[FrameCollapse]) -> dict:
    """Read a :func:`tolerance_sweep` for the two DIFFERENT things that can happen in it.

    Returns ``{"flat", "exceeds_exact", "rises", "counts"}``:

    * ``flat`` — every tolerance gave the same count, i.e. the frame's headline number does not
      depend on the tolerance at all.
    * ``exceeds_exact`` — some ``tol>0`` count is ABOVE the ``tol=0`` count. This is the guarantee
      from module docstring point 1 and it CANNOT happen: ``True`` here is a genuine bug.
    * ``rises`` — the ``(tol_lo, tol_hi)`` pairs where a coarser tolerance produced MORE rows than a
      finer one. Expected, real, and not a bug (point 2); reported so a reader is not surprised.

    The two are separated because conflating them is how the false monotone claim survived: a rise
    looks like a violated invariant, and treating it as one would send someone hunting a bug that
    isn't there — while the invariant that IS real (``exceeds_exact``) would go unchecked.
    """
    counts = [r.distinct_feature_rows for r in results]
    exact = next((r.distinct_feature_rows for r in results if r.tol == 0.0), None)
    return {
        "flat": len(set(counts)) == 1,
        "exceeds_exact": bool(exact is not None and any(c > exact for c in counts)),
        "rises": [
            (results[i].tol, results[i + 1].tol)
            for i in range(len(results) - 1)
            if counts[i + 1] > counts[i]
        ],
        "counts": dict(zip([r.tol for r in results], counts, strict=True)),
    }


def format_report(results: dict[str, FrameCollapse], *, target_name: str = "") -> str:
    """The human table: one row per named frame, floors omitted when no target was supplied."""
    if not results:
        return "no frames diagnosed\n"
    first = next(iter(results.values()))
    has_floor = first.floor_wmae is not None
    width = max(len(k) for k in results) + 2

    lines = [
        f"FRAME COLLAPSE — {first.n_cells} cells "
        f"({first.n_positions} positions{' incl. space' if first.includes_space else ', no space'}, "
        f"order {first.order}, tol {first.tol:g})",
    ]
    if has_floor and target_name:
        lines.append(f"  floors against: {target_name}")
    head = (
        f"{'frame':<{width}}{'cols':>5}{'distinct':>10}{'resolution':>12}"
        f"{'collapsed':>11}{'mass':>9}{'max grp':>9}"
    )
    if has_floor:
        head += f"{'FLOOR wmae':>13}{'at grp mean':>13}{'FLOOR wrmse':>13}"
    lines += [head, "-" * len(head)]
    for name, r in results.items():
        row = (
            f"{name:<{width}}{r.n_columns:>5}{r.distinct_feature_rows:>10}{r.resolution:>11.1%} "
            f"{r.collapsed_cells:>10}{r.mass_share_collapsed:>9.1%}{r.largest_group:>9}"
        )
        if r.floor_wmae is not None:
            row += f"{r.floor_wmae:>13.4f}{r.floor_wmae_at_group_mean:>13.4f}{r.floor_wrmse:>13.4f}"
        lines.append(row)
    lines += [
        "",
        "resolution = distinct feature rows / cells (100% = nothing collapsed). Cells sharing a row",
        "are indistinguishable to EVERY model on the frame and must receive one prediction; FLOOR is",
        "the best error achievable under that constraint. NECESSARY condition only: high resolution",
        "does NOT imply a good frame, and the floor bounds error from below, never predicts it.",
    ]
    if has_floor:
        lines.append(
            "FLOOR wmae is at the within-group weighted MEDIAN (the true L1 minimizer); 'at grp mean'"
        )
        lines.append(
            "is the same error at the group MEAN — an achievable error, so an UPPER bound on the floor."
        )
        for name, r in results.items():
            if r.target_is_self_generated:
                lines += [
                    "",
                    f"⚠ {name}: ALL {r.n_collapse_groups} collapse groups have EXACTLY zero target "
                    "spread, so this",
                    "  floor of 0 is an IDENTITY, not a measurement — the signature of a target THIS",
                    "  FRAME GENERATED (identical rows -> identical model output -> identical target).",
                    "  It says nothing about the frame's resolution. Score against a target the frame",
                    "  did not produce, or read the distinct-row/mass columns, which are target-free.",
                ]
    return "\n".join(lines) + "\n"
