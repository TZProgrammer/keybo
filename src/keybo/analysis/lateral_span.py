"""``lateral-span`` — the graded lateral-stretch gauge, and the per-cell support behind it.

An ADDITIVE diagnostic for the defect ``state/closeout-unknown`` D7 registered:
:func:`keybo.features.classify.is_lsb` hardcodes the ``('index','middle')`` finger pair, so it
can flag only 32 of the 204 same-hand two-finger position pairs of ``ROW_STAGGERED_30`` that
have a stagger-adjusted span over 1.5. The 172 it can never flag are index-pinky (72),
index-ring (64), middle-pinky (28), middle-ring (4) and ring-pinky (4).

**Why that is the whole argument.** The flagged subset captures a *different fraction of the
phenomenon on each layout* — 11.15%-24.52% of positive-span corpus mass across the five
``NAMED_LAYOUTS``, a 2.20x fold spread (D7 reports the same defect as 3.23x on qwerty rising to
13.50x on graphite against its own wider set and a space-including denominator; the ratio is
denominator-invariant, so the two agree). A subset whose coverage moves with the layout cannot
rank layouts consistently *even in principle*. That argument was already accepted once for
``is_scissor``, which is why :func:`keybo.features.classify.is_row_skip` exists; this is the
same defect shape with one instance left unfixed. ``lsb`` is simultaneously a
``FEATURE_VERSION``-stamped model input, a weighted ``comfort.py`` term (10.0) and a weighted
``oxey.py`` term (3.0), so the blindness propagates into the speed model, the comfort axis and
the community crosswalk at once.

**The measure is GRADED, and that is the fix rather than a presentation choice.** Any
*thresholded* widening keeps a sub-threshold blind spot, and that residual is itself
layout-dependent: the banded reading still measures a 1.73x coverage fold spread. The
continuous :func:`~keybo.features.classify.lateral_span` has no threshold, so it prices every
positive-span bigram on every layout — :meth:`LateralSpan.coverage` is 1.0 everywhere and its
fold spread is exactly 1.00x. A measure with no blind spot cannot have a layout-dependent one.

**Not redundant** (DIST-1's bar: ``rho >= 0.93`` against the incumbent share means relabelling,
not information): over 200 random layouts ``rho(is_lsb, lateral_span) = +0.5142``.

**Denominator: layout-restricted, space-EXCLUDED bigram mass** — the ``kmstats``/``sfb``/``lsb``
convention, and :mod:`keybo.analysis.bad_scissor`'s. ``Layout.has_key(" ")`` is True, so the
``oxey.pattern_shares`` convention silently counts space-touching bigrams in the denominator.
Space is in no lateral-span pair (``hand(0) == 0``), so choosing wrong leaves the **numerator
bit-identical** and only deflates every share (trap #9); ``exclude_space=False`` exists so the
regression test can measure that, and production callers leave it alone.

⚠ **This is a MEASUREMENT/DIAGNOSIS gauge, not a search objective and not a severity model.**
Two limits, both measured, both load-bearing:

* **No severity weighting is offered, deliberately.** The graded *total* is coverage-invariant,
  but the per-cell *mix* is not: the middle-pinky share of graded mass moves 10.69x across the
  five layouts and ring-pinky 17.39x. So a per-cell severity surface would reintroduce exactly
  the layout-dependence this gauge removes. Pricing these cells needs a measured per-cell cost,
  which requires the raw keystroke frame — not in this repo — so :meth:`LateralSpan.support`
  reports each cell's *corpus support* and refuses to invent the rest.
* **Optimizing a strain axis is optimizing the ruler** (WSCISSOR-GEN-1, ledger ``44d282b``).

Nothing here changes a shipped value: ``is_lsb`` is byte-identical and no ``FEATURE_VERSION``
list is touched, so no trained model is invalidated. Whether any weight should be attached to
this measure is a scoring-policy decision, and is not taken here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from keybo.features import classify as C
from keybo.geometry import Geometry, Position
from keybo.layout import Layout

#: Dexterity rank, least -> most dextrous. Matches :mod:`keybo.analysis.bad_scissor`'s ``_DEX``
#: so "the weaker finger" means the same thing across the campaign's artifacts.
_DEX: Mapping[str, int] = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}

#: Minimum DISTINCT bigram identities before a cell's mass is called supported. Reused from
#: ``run_scissor_estimation.py`` (SCISSOR-2), whose own docstring records why a raw-keystroke
#: floor is not enough: its first version reported ``ring-pinky:top-bottom`` at +128.6% from a
#: cell that was 100% the single bigram ``p.``, i.e. one bigram's idiosyncrasy dressed up as a
#: class cost.
MIN_DISTINCT_BIGRAMS = 3

#: Herfindahl-Hirschman ceiling on a cell's frequency concentration. The companion guard the
#: same docstring says a count floor cannot provide: three identities still fail if one of them
#: carries nearly all the mass (HHI 1.0 is a single bigram; 0.5 is two equal ones).
MAX_CELL_HHI = 0.5

#: A pair predicate/weight: ``(geometry, a, b) -> nonnegative value``.
Weight = Callable[[Geometry, Position, Position], float]


@dataclass(frozen=True)
class CellSupport:
    """Whether a cell's corpus mass can carry a per-cell statement, and why not if it cannot.

    ``status`` is ``MEASURED`` only when the cell clears BOTH floors. An unsupported cell is
    reported as ``UNMEASURED`` with a reason rather than silently filled — a labelled prior
    beats an invented number, which is the stance ``run_scissor_estimation.py`` takes.
    """

    cell: str
    n_distinct: int
    hhi: float
    top_share: float
    mass: float
    status: str
    reason: str | None


def lateral_span_cell(geometry: Geometry, a: Position, b: Position) -> str | None:
    """The pair's ``"<finger-pair>"`` cell, e.g. ``"index-pinky"``, or ``None`` off support.

    Finger pairs are named most-dextrous-first so the label matches the campaign's tables
    (``index-pinky``, not ``pinky-index``), and the label is symmetric in ``a``/``b``.
    """
    if not C.lateral_span(geometry, a, b):
        return None
    first, second = sorted(
        (_finger_name(geometry, a[0]), _finger_name(geometry, b[0])),
        key=lambda kind: -_DEX[kind],
    )
    return f"{first}-{second}"


def _finger_name(geometry: Geometry, x: int) -> str:
    """``'right-index' -> 'index'``. Hand-independent."""
    return geometry.finger(x).value.split("-")[1]


def _graded(geometry: Geometry, a: Position, b: Position) -> float:
    return C.lateral_span(geometry, a, b)


class LateralSpan:
    """The graded ``lateral-span`` share, its cell partition, and its per-cell support."""

    #: Every same-hand two-finger class that can carry lateral span, in report order. Declared
    #: rather than discovered so :meth:`by_cell` can REFUSE a drifted label (see
    #: :meth:`_partition`).
    CELLS: tuple[str, ...] = (
        "index-middle",
        "index-ring",
        "index-pinky",
        "middle-ring",
        "middle-pinky",
        "ring-pinky",
    )

    def __init__(self, bigram_freqs: Mapping[str, int]) -> None:
        self._bg = {bg: freq for bg, freq in bigram_freqs.items() if len(bg) == 2}

    # -- gauges ---------------------------------------------------------------------------

    def share(self, layout: Layout, *, exclude_space: bool = True) -> float:
        """Graded lateral-span mass as a percent of layout-restricted bigram mass.

        Units are column-stretches per bigram: a pair stretched two columns beyond rest counts
        twice a pair stretched one, which is what makes the measure continuous and so
        blind-spot-free. ``exclude_space=False`` selects the wrong (``oxey``) denominator and
        exists only for the trap-#9 regression test.
        """
        return self.share_of(layout, _graded, exclude_space=exclude_space)

    def share_of(
        self,
        layout: Layout,
        weight: Weight,
        *,
        exclude_space: bool = True,
    ) -> float:
        """This gauge's scoring loop and denominator, run on an arbitrary pair weight.

        Exposed so the incumbent ``is_lsb`` indicator can be driven through the SAME
        denominator as the graded measure. Comparing coverage across two denominators would
        confound the widening with trap #9, so the comparison has to share this loop.
        """
        numerator, denominator = self._totals(layout, weight, exclude_space)
        return 100.0 * numerator / denominator if denominator else 0.0

    def coverage(self, layout: Layout, weight: Weight | None = None) -> float:
        """Fraction of the lateral-stretch PHENOMENON that ``weight`` prices on ``layout``.

        The phenomenon is the mass of every bigram with positive lateral span; the numerator is
        the mass ``weight`` gives a nonzero value to. This is the quantity the whole deliverable
        is about: it is 1.0 on every layout for the graded measure (no threshold, so nothing is
        left unpriced) and 0.11-0.25, layout-dependently, for ``is_lsb``.

        Deliberately mass-of-flagged-bigrams, not sum-of-weights: it asks "what fraction of the
        phenomenon does this measure SEE", so a graded weight must not be able to inflate its
        own coverage by pricing a few bigrams heavily.
        """
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        priced = phenomenon = 0.0
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram, exclude_space=True):
                continue
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            if not C.lateral_span(geometry, a, b):
                continue
            phenomenon += freq
            if weight is None or weight(geometry, a, b):
                priced += freq
        return priced / phenomenon if phenomenon else 0.0

    def by_cell(self, layout: Layout, *, exclude_space: bool = True) -> dict[str, float]:
        """Graded share per finger-pair cell — an exact partition of :meth:`share`.

        Read as "where the stretched mass sits", NOT as "which pair is most strained": no
        severity is applied, and the module docstring records why none is offered.
        """
        return self._partition(layout, lateral_span_cell, self.CELLS, exclude_space)

    def support(self, layout: Layout, *, exclude_space: bool = True) -> dict[str, CellSupport]:
        """Per-cell corpus support, with both floors applied and the failure named.

        The guard the register mandated: a per-cell claim needs several distinct bigram
        identities AND bounded concentration, because a handful of high-frequency bigrams can
        otherwise carry a cell that a raw count floor calls well-supported.
        """
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        per_cell: dict[str, dict[str, float]] = {cell: {} for cell in self.CELLS}
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram, exclude_space):
                continue
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            cell = lateral_span_cell(geometry, a, b)
            if cell is None:
                continue
            per_cell[cell][bigram] = per_cell[cell].get(bigram, 0.0) + freq
        return {cell: self._judge(cell, mass) for cell, mass in per_cell.items()}

    # -- internals ------------------------------------------------------------------------

    @staticmethod
    def _judge(cell: str, mass: Mapping[str, float]) -> CellSupport:
        total = sum(mass.values())
        if not total:
            return CellSupport(cell, 0, 0.0, 0.0, 0.0, "UNMEASURED", "no mass")
        hhi = sum((freq / total) ** 2 for freq in mass.values())
        top = max(mass.values()) / total
        if len(mass) < MIN_DISTINCT_BIGRAMS:
            reason = "too few distinct bigrams"
        elif hhi > MAX_CELL_HHI:
            reason = "mass too concentrated in few bigrams"
        else:
            reason = None
        status = "MEASURED" if reason is None else "UNMEASURED"
        return CellSupport(cell, len(mass), hhi, top, total, status, reason)

    def _totals(self, layout: Layout, weight: Weight, exclude_space: bool) -> tuple[float, float]:
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        numerator = denominator = 0.0
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram, exclude_space):
                continue
            denominator += freq
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            value = weight(geometry, a, b)
            if value:
                numerator += value * freq
        return numerator, denominator

    @staticmethod
    def _counts(layout: Layout, bigram: str, exclude_space: bool) -> bool:
        """Whether a bigram is inside the denominator (and so eligible for the numerator)."""
        if exclude_space and " " in bigram:
            return False
        return all(layout.has_key(character) for character in bigram)

    def _partition(
        self,
        layout: Layout,
        classifier: Callable[[Geometry, Position, Position], str | None],
        preset_keys: tuple[str, ...],
        exclude_space: bool,
    ) -> dict[str, float]:
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        charged = dict.fromkeys(preset_keys, 0.0)
        denominator = 0.0
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram, exclude_space):
                continue
            denominator += freq
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            key = classifier(geometry, a, b)
            if key is None:
                continue
            if key not in charged:
                # REFUSE rather than append (the BSAUDIT-1 D4 lesson): silently growing the
                # dict makes a partition that is PRINTED against a fixed column list stop
                # summing to the share, and every exact-partition test still passes because
                # they sum `.values()` and never the printed columns.
                raise ValueError(
                    f"{getattr(classifier, '__name__', classifier)} returned {key!r}, which is "
                    f"not one of this partition's declared keys {sorted(charged)}. Either the "
                    f"classifier's label set drifted from the caller's column list, or a new "
                    f"class was added without extending the presets — both silently break the "
                    f"partition when it is printed."
                )
            charged[key] += C.lateral_span(geometry, a, b) * freq
        if not denominator:
            return dict.fromkeys(preset_keys, 0.0)
        return {key: 100.0 * value / denominator for key, value in charged.items()}

    @staticmethod
    def _check_geometry(geometry: Geometry) -> None:
        """Refuse a board whose columns have no declared neutral (rest) position.

        The measure is an excess over the two fingers' rest separation, read from
        ``classify._HOME_COLUMN``. A board with a column that table does not know about would
        otherwise score against an invented rest posture, so refuse instead — the same stance
        :mod:`keybo.scoring.scissor_severity` and :mod:`keybo.analysis.bad_scissor` take.
        """
        unknown = sorted({abs(x) for x, _y in geometry.slots} - set(C._HOME_COLUMN))
        if unknown:
            raise ValueError(
                "lateral span is defined for boards whose columns have a declared neutral "
                f"position ({sorted(C._HOME_COLUMN)}); got unknown column(s) {unknown}."
            )
