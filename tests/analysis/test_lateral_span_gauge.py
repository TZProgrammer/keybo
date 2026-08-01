"""``LateralSpan`` — the additive lateral-stretch gauge, its guards, and its evidence.

Written before the implementation (LSBWIDEN-1, 2026-08-01).

The four things this gauge is most likely to get wrong, each pinned:

* **the invariant it exists for** — coverage of the phenomenon must be layout-INDEPENDENT.
  ``test_the_graded_gauge_coverage_is_layout_independent`` asserts fold spread exactly 1.0
  for the graded measure while the incumbent's is over 2x, which is the whole deliverable;
* **the denominator** — layout-restricted, space-EXCLUDED (the ``kmstats``/``sfb``/``lsb``
  convention, and ``bad_scissor``'s). Borrowing ``oxey``'s space-including denominator leaves
  the numerator bit-identical and moves every share by ~1.5x (trap #9);
* **the support guards** — a cell needs several DISTINCT bigram identities *and* a bounded
  concentration (HHI). A raw count floor is not sufficient: a handful of high-frequency
  bigrams can otherwise carry a "well-supported" cell. Both floors are
  ``run_scissor_estimation.py``'s, reused rather than reinvented;
* **the exact partition** — ``by_cell`` must sum to ``share`` and must REFUSE an undeclared
  label rather than silently growing its dict (the failure ``bad_scissor`` documents).
"""

from __future__ import annotations

import pytest

from keybo.analysis.lateral_span import (
    MAX_CELL_HHI,
    MIN_DISTINCT_BIGRAMS,
    CellSupport,
    LateralSpan,
    lateral_span_cell,
)
from keybo.data.corpus import load_frequencies
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as GEOM
from keybo.geometry import Geometry
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS

_CORPUS = load_frequencies("data/corpus/bigrams.txt")


@pytest.fixture(scope="module")
def gauge() -> LateralSpan:
    return LateralSpan(_CORPUS)


# --- the cell label ----------------------------------------------------------------------


def test_the_cell_label_names_the_finger_pair_most_dextrous_first() -> None:
    assert lateral_span_cell(GEOM, (-1, 2), (-5, 2)) == "index-pinky"
    assert lateral_span_cell(GEOM, (-5, 2), (-1, 2)) == "index-pinky"  # symmetric
    assert lateral_span_cell(GEOM, (-1, 1), (-3, 3)) == "index-middle"


def test_the_cell_label_is_None_off_support() -> None:
    assert lateral_span_cell(GEOM, (-2, 2), (2, 2)) is None  # cross-hand
    assert lateral_span_cell(GEOM, (-2, 2), (-3, 2)) is None  # at neutral: no stretch


# --- the gauge ---------------------------------------------------------------------------


def test_the_share_is_positive_and_finite_on_every_named_layout(gauge: LateralSpan) -> None:
    for name, chars in NAMED_LAYOUTS.items():
        value = gauge.share(Layout(chars, GEOM))
        assert 0.0 < value < 100.0, name


def test_the_graded_share_strictly_exceeds_the_incumbent_lsb_share(gauge: LateralSpan) -> None:
    """It prices strictly more of the phenomenon on every layout — the point of widening."""
    for name, chars in NAMED_LAYOUTS.items():
        layout = Layout(chars, GEOM)
        assert gauge.share(layout) > gauge.share_of(layout, _lsb_indicator), name


def test_the_space_including_denominator_moves_every_share(gauge: LateralSpan) -> None:
    """Trap #9: no lateral-span pair contains space (``hand(0) == 0``), so choosing the wrong
    denominator leaves the NUMERATOR bit-identical and only deflates the share."""
    for name, chars in NAMED_LAYOUTS.items():
        layout = Layout(chars, GEOM)
        excluded = gauge.share(layout)
        included = gauge.share(layout, exclude_space=False)
        assert included < excluded, name
        assert 1.3 < excluded / included < 1.7, name


def test_by_cell_is_an_exact_partition_of_the_share(gauge: LateralSpan) -> None:
    for name, chars in NAMED_LAYOUTS.items():
        layout = Layout(chars, GEOM)
        assert sum(gauge.by_cell(layout).values()) == pytest.approx(gauge.share(layout)), name


def test_by_cell_refuses_a_label_outside_its_declared_keys(gauge: LateralSpan) -> None:
    """The ``bad_scissor`` lesson: silently growing the dict makes a PRINTED partition stop
    summing to the share, and no downstream test can catch it."""
    layout = Layout(NAMED_LAYOUTS["qwerty"], GEOM)
    with pytest.raises(ValueError, match="not one of this partition's declared keys"):
        gauge._partition(layout, lambda g, a, b: "not-a-cell", ("index-middle",), True)


# --- THE INVARIANT -----------------------------------------------------------------------


def test_the_incumbent_predicates_coverage_IS_layout_dependent(gauge: LateralSpan) -> None:
    """The defect, as a gauge-level test: ``is_lsb`` captures a different fraction of the
    lateral-stretch mass on each layout, so it cannot rank layouts consistently."""
    coverage = {
        name: gauge.coverage(Layout(chars, GEOM), _lsb_indicator)
        for name, chars in NAMED_LAYOUTS.items()
    }
    assert max(coverage.values()) / min(coverage.values()) > 2.0
    assert all(value < 0.30 for value in coverage.values())


def test_the_graded_gauge_coverage_is_layout_independent(gauge: LateralSpan) -> None:
    """THE DELIVERABLE. The graded measure prices every positive-span bigram, so its coverage
    is 1.0 on every layout and the fold spread is exactly 1.0 — no blind spot, hence no
    LAYOUT-DEPENDENT blind spot. This is what the incumbent cannot do at any threshold."""
    coverage = {name: gauge.coverage(Layout(chars, GEOM)) for name, chars in NAMED_LAYOUTS.items()}
    assert set(coverage) == set(NAMED_LAYOUTS)
    for name, value in coverage.items():
        assert value == pytest.approx(1.0), name
    assert max(coverage.values()) / min(coverage.values()) == pytest.approx(1.0)


def test_a_thresholded_measure_keeps_a_layout_dependent_residual(gauge: LateralSpan) -> None:
    """Why the shipped measure is graded, not banded: the banded reading's coverage still
    moves across layouts, strictly between the incumbent's and the graded measure's."""
    banded = {
        name: gauge.coverage(Layout(chars, GEOM), _banded_indicator)
        for name, chars in NAMED_LAYOUTS.items()
    }
    incumbent = {
        name: gauge.coverage(Layout(chars, GEOM), _lsb_indicator)
        for name, chars in NAMED_LAYOUTS.items()
    }
    banded_fold = max(banded.values()) / min(banded.values())
    incumbent_fold = max(incumbent.values()) / min(incumbent.values())
    assert 1.0 < banded_fold < incumbent_fold


# --- the support guards ------------------------------------------------------------------


def test_the_guards_reuse_the_scissor_estimations_floors() -> None:
    """Not reinvented: ``run_scissor_estimation.py``'s distinct-bigram floor, plus the
    concentration ceiling its own docstring says a raw floor fails to provide."""
    assert MIN_DISTINCT_BIGRAMS == 3
    assert 0.0 < MAX_CELL_HHI <= 0.5


def test_every_cell_reports_its_support_status(gauge: LateralSpan) -> None:
    support = gauge.support(Layout(NAMED_LAYOUTS["qwerty"], GEOM))
    assert set(support) == set(LateralSpan.CELLS)
    for cell, entry in support.items():
        assert isinstance(entry, CellSupport)
        assert entry.status in ("MEASURED", "UNMEASURED"), cell
        assert entry.n_distinct >= 0


def test_a_cell_carried_by_too_few_bigram_identities_is_UNMEASURED() -> None:
    """A raw-count floor alone would call this supported: one bigram, enormous frequency.

    ``qt`` is an index-pinky pair on qwerty (``q`` pinky top, ``t`` index stretch column).
    """
    gauge = LateralSpan({"qt": 10_000_000})
    layout = Layout(NAMED_LAYOUTS["qwerty"], GEOM)
    entry = gauge.support(layout)["index-pinky"]
    assert entry.n_distinct == 1
    assert entry.status == "UNMEASURED"
    assert entry.reason == "too few distinct bigrams"


def test_a_cell_dominated_by_one_bigram_is_UNMEASURED_even_with_enough_identities() -> None:
    """The HHI ceiling doing the work a count floor cannot: three identities, but one carries
    99.97% of the mass, so the 'class cost' would be one bigram's idiosyncrasy."""
    gauge = LateralSpan({"qt": 10_000_000, "tq": 1_500, "qf": 1_500})
    entry = gauge.support(Layout(NAMED_LAYOUTS["qwerty"], GEOM))["index-pinky"]
    assert entry.n_distinct >= MIN_DISTINCT_BIGRAMS
    assert entry.hhi > MAX_CELL_HHI
    assert entry.status == "UNMEASURED"
    assert entry.reason == "mass too concentrated in few bigrams"


def test_a_well_supported_cell_is_MEASURED(gauge: LateralSpan) -> None:
    entry = gauge.support(Layout(NAMED_LAYOUTS["qwerty"], GEOM))["index-middle"]
    assert entry.status == "MEASURED"
    assert entry.reason is None
    assert entry.n_distinct >= MIN_DISTINCT_BIGRAMS
    assert entry.hhi <= MAX_CELL_HHI


def test_an_empty_cell_is_UNMEASURED_not_an_error() -> None:
    entry = LateralSpan({}).support(Layout(NAMED_LAYOUTS["qwerty"], GEOM))["index-middle"]
    assert entry.status == "UNMEASURED"
    assert entry.reason == "no mass"


# --- refusals ----------------------------------------------------------------------------


def test_the_gauge_refuses_a_board_its_neutral_columns_are_undefined_for() -> None:
    """A column with no declared rest position must fail loudly, not score against an invented
    neutral. The board is the standard one with a single column moved out to |x| == 9, so the
    layout is still well formed and it is the GAUGE that refuses it."""
    slots = list(GEOM.slots)
    slots[0] = (-9, 3)
    weird = Geometry(slots=tuple(slots))
    with pytest.raises(ValueError, match="lateral span"):
        LateralSpan(_CORPUS).share(Layout(NAMED_LAYOUTS["qwerty"], weird))


def _lsb_indicator(geometry: Geometry, a, b) -> float:
    return 1.0 if C.is_lsb(geometry, a, b) else 0.0


def _banded_indicator(geometry: Geometry, a, b) -> float:
    return float(C.lateral_span_class(geometry, a, b))
