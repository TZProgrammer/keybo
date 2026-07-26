"""Tests for the in-loop wide-support scissor axis.

The load-bearing property is that the FAST bilinear batch path returns exactly what
``ScissorSeverity.share`` returns. If it does not, every number in the report is measured on a
gauge nobody declared, and the optimizing-the-ruler guard compares two things that are not the
two supports. So the identity is pinned at 0.0 (not a tolerance) on real corpora.

Also pinned: that adding the 7th objective leaves the campaign's original six BIT-IDENTICAL, so
a wscissor arm and a baseline arm differ in exactly one axis and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus_eval as CE  # noqa: E402
import wscissor_eval as WE  # noqa: E402

from keybo.data.corpus import load_frequencies  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as GEOM  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.scoring.oxey import OxeyStyleScorer  # noqa: E402
from keybo.scoring.scissor_severity import ScissorSeverity, SeverityWeights  # noqa: E402

CORPORA = ("iweb", "blend", "noanchor")
LAYOUTS = tuple(CE.INCUMBENTS.values())
PREFS = (
    SeverityWeights(support="wide"),
    SeverityWeights(support="narrow"),
    WE.P_WIDE,
    WE.P_NARROW,
    SeverityWeights(pinky=4.0, ring_ratio=1.0, down=3.0, support="wide"),
)


@pytest.fixture(scope="module")
def gauges() -> dict:
    out = {}
    for corpus in CORPORA:
        bigrams = load_frequencies(str(CE.CORPUS_DIRS[corpus] / "bigrams.txt"))
        out[corpus] = (WE.SeverityGauge(bigrams), ScissorSeverity(bigrams))
    return out


# -- the identity ---------------------------------------------------------------------------
@pytest.mark.parametrize("corpus", CORPORA)
def test_fast_share_equals_slow_share_exactly(gauges, corpus):
    """Fast bilinear share == ScissorSeverity.share, on every layout x preference, err 0.0.

    Exact rather than approximate because both sides sum the same finite set of products; a
    non-zero error would mean the two are summing different sets.
    """
    fast, slow = gauges[corpus]
    worst = 0.0
    for string in LAYOUTS:
        perm = CE.perm_of(string)
        layout = Layout(string, GEOM)
        for weights in PREFS:
            got = fast.share(perm, weights)
            want = slow.share(layout, weights)
            worst = max(worst, abs(got - want))
    assert worst == 0.0, f"fast/slow severity share disagree by {worst!r} on {corpus}"


@pytest.mark.parametrize("corpus", CORPORA)
def test_share_batch_matches_share_one_at_a_time(gauges, corpus):
    fast, _slow = gauges[corpus]
    perms = np.array([CE.perm_of(s) for s in LAYOUTS])
    for weights in PREFS:
        batch = fast.share_batch(perms, weights)
        one = np.array([fast.share(p, weights) for p in perms])
        assert np.max(np.abs(batch - one)) == 0.0


@pytest.mark.parametrize("corpus", CORPORA)
def test_flat_narrow_reproduces_incumbent_oxey_scissor_share(gauges, corpus):
    """The positive control the severity gauge is built around, through the FAST path.

    At all weights 1.0 on the narrow support the gauge must reproduce the incumbent flat
    ``oxey.pattern_shares()["scissor"]`` exactly. This is what makes the gauge a strict
    generalization rather than a rival metric — and the fast path must inherit it.
    """
    fast, _slow = gauges[corpus]
    bigrams = load_frequencies(str(CE.CORPUS_DIRS[corpus] / "bigrams.txt"))
    oxey = OxeyStyleScorer(bigrams, bigrams, {})
    worst = 0.0
    for string in LAYOUTS:
        want = oxey.pattern_shares(Layout(string, GEOM))["scissor"]
        got = fast.share(CE.perm_of(string), WE.FLAT_NARROW)
        worst = max(worst, abs(got - want))
    assert worst == 0.0, f"fast flat-narrow != oxey flat scissor by {worst!r} on {corpus}"


# -- the support relation -------------------------------------------------------------------
def test_narrow_support_is_a_strict_subset_of_wide():
    """Exhaustive over all 900 ordered slot pairs. Guarantees wide >= narrow pointwise, so
    ``unflagged = wide - narrow`` is a set difference and can never be negative."""
    narrow = WE.severity_slot_matrix(SeverityWeights(support="narrow"))
    wide = WE.severity_slot_matrix(SeverityWeights(support="wide"))
    assert np.all((narrow > 0) <= (wide > 0))
    assert np.all(wide >= narrow)
    assert int((narrow > 0).sum()) < int((wide > 0).sum())


def test_narrow_support_contains_no_middle_pinky_pair():
    """The structural fact that motivates the task: the incumbent predicate cannot reach
    middle-pinky mass at all, so no weighting on the narrow support can price it."""
    positions = list(GEOM.slots)
    narrow_hits = 0
    wide_hits = 0
    for a in positions:
        for b in positions:
            kinds = {GEOM.finger(a[0]).name, GEOM.finger(b[0]).name}
            if not (any(k.endswith("M") for k in kinds) and any(k.endswith("P") for k in kinds)):
                continue
            narrow_hits += bool(C.is_scissor(GEOM, a, b))
            wide_hits += bool(ScissorSeverity._in_wide_support(GEOM, a, b))
    assert narrow_hits == 0
    assert wide_hits > 0


@pytest.mark.parametrize("corpus", CORPORA)
def test_denominator_includes_space_touching_bigrams(gauges, corpus):
    """Regression pin for the trap this driver actually fell into.

    The severity gauge's denominator admits a bigram when the layout has BOTH chars, and
    ``Layout.has_key(' ')`` is True (space is a real key at ``geometry.space_position``). So
    space-touching bigrams count. ``corpus_eval``'s kmstats ``bi_total`` masks them OUT, and
    borrowing that convention here inflates every severity share by a constant ~1.5x while the
    numerator stays bit-exact — a wrong number that looks entirely plausible.
    """
    fast, _slow = gauges[corpus]
    bigrams = load_frequencies(str(CE.CORPUS_DIRS[corpus] / "bigrams.txt"))
    kmstats_total = CE.build_kmstats_matrices(bigrams, bigrams)["bi_total"]
    assert fast.total > kmstats_total, "severity denominator must be the space-INCLUSIVE one"
    space_mass = fast.mass[CE.SPACE, :].sum() + fast.mass[:, CE.SPACE].sum()
    assert fast.total == pytest.approx(kmstats_total + space_mass, rel=0, abs=1e-6)


def test_space_slot_carries_no_severity():
    """Space has no key, so no pair touching slot 30 may ever be charged."""
    for weights in PREFS:
        matrix = WE.severity_slot_matrix(weights)
        assert np.all(matrix[CE.SPACE, :] == 0.0)
        assert np.all(matrix[:, CE.SPACE] == 0.0)


# -- FULL positive control: EVERY axis this round cites, not a sample -------------------------
# TOOLING-TRAPS.md trap 3: three metrics reproducing bit-for-bit does not imply the fourth will
# (a sibling had umae/wmae/rho all exactly 0.0 while freq_decile_mae was 0.03 off, because a
# kind="stable" sort over tied values reordered deciles). So the controls below cover ALL TWELVE
# axes the report quotes — the subclass is verified as a whole, not just the two axes it adds.
@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_every_cited_axis_matches_the_slow_reference(corpus, arm):
    """All 12 report axes via the fast path == the zero-reuse slow path, on every incumbent.

    `WScissorBoard` inherits ten axes from `ArmBoard`, and `test_corpus_eval.py` verifies those
    on `ArmBoard` — but on `ArmBoard`, not on this subclass. A subclass that shadowed a cached
    table, or an `evaluate_batch` override that mutated shared state, would leave the parent's
    tests green and every number in this round wrong. This closes that gap explicitly.
    """
    ceilings = CE.SixSurface(corpus).ceiling_map
    board = WE.WScissorBoard(corpus=corpus, arm=arm, ceilings=ceilings, objective="wide")
    worst = {}
    for string in CE.INCUMBENTS.values():
        fast = board.axes12(string)
        slow = board.axes_slow(string) | board.severity_axes_slow(string)
        slow["wscissor"] = slow.pop("wscissor_P")
        slow["nscissor"] = slow.pop("nscissor_P")
        for axis in WE.AXES12:
            scale = max(abs(slow[axis]), 1.0)
            worst[axis] = max(worst.get(axis, 0.0), abs(fast[axis] - slow[axis]) / scale)
    # Every axis, named, so a failure says WHICH one drifted rather than "something did".
    bad = {a: e for a, e in worst.items() if e > 1e-9}
    assert not bad, f"{corpus}/arm{arm}: fast != slow on {bad}"
    assert set(worst) == set(WE.AXES12), "an axis was silently skipped by the control"


@pytest.mark.parametrize("corpus", CORPORA)
def test_evaluate_batch_columns_match_the_per_layout_path(corpus):
    """The EA sees `evaluate_batch`; the report reads `axes12`. If they disagree, the search
    optimized something other than what is reported — the most dangerous possible drift here.
    """
    ceilings = CE.SixSurface(corpus).ceiling_map
    board = WE.WScissorBoard(corpus=corpus, arm="A", ceilings=ceilings, objective="wide")
    strings = list(CE.INCUMBENTS.values())
    movables = np.array([[s.index(c) for c in CE.C30M] for s in strings])
    batch = board.evaluate_batch(movables)
    for row, string in zip(batch, strings, strict=True):
        axes = board.axes12(string)
        assert row[0] == pytest.approx(-axes["floor"], rel=0, abs=1e-9)
        assert row[1] == pytest.approx(-axes["mean"], rel=0, abs=1e-9)
        assert row[2] == pytest.approx(axes["scissor"], rel=0, abs=1e-9)
        assert row[3] == pytest.approx(axes["lsb"], rel=0, abs=1e-9)
        assert row[4] == pytest.approx(axes["sfb"], rel=0, abs=1e-9)
        assert row[5] == pytest.approx(axes["sfs"], rel=0, abs=1e-9)
        assert row[6] == pytest.approx(axes["wscissor"], rel=0, abs=1e-9)


def test_iweb_ceilings_reproduce_the_frozen_campaign_constant():
    """A positive control on the constants the normalized floor divides by. If these drifted,
    every floor number would be incomparable with the campaign's, silently."""
    six = CE.SixSurface("iweb")
    for surface, frozen in CE.FROZEN_IWEB_CEILINGS.items():
        assert six.ceiling_map[surface] == pytest.approx(frozen, rel=0, abs=1e-9)


# -- the board wiring ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def boards() -> dict:
    ceilings = CE.SixSurface("iweb").ceiling_map
    return {
        objective: WE.WScissorBoard(corpus="iweb", arm="A", ceilings=ceilings, objective=objective)
        for objective in ("wide", "narrow", "none")
    }


def test_adding_the_seventh_objective_leaves_the_first_six_bit_identical(boards):
    """A wscissor arm must differ from the baseline arm in EXACTLY one axis.

    Otherwise a movement attributed to 'optimizing wide' could be an artifact of the
    objectives changing underneath, and the guard would be measuring the wrong contrast.
    """
    rng = np.random.default_rng(20260725)
    movables = np.array([rng.permutation(30) for _ in range(24)])
    base = boards["none"].evaluate_batch(movables)
    for objective in ("wide", "narrow"):
        seven = boards[objective].evaluate_batch(movables)
        assert seven.shape == (24, 7)
        assert np.max(np.abs(seven[:, :6] - base)) == 0.0


def test_seventh_objective_is_the_named_support(boards):
    rng = np.random.default_rng(4242)
    movables = np.array([rng.permutation(30) for _ in range(16)])
    perms = np.empty((16, CE.NSLOT), dtype=np.int64)
    perms[:, :30] = movables
    perms[:, 30] = CE.SPACE
    wide_board = boards["wide"]
    narrow_board = boards["narrow"]
    wide_col = wide_board.evaluate_batch(movables)[:, 6]
    narrow_col = narrow_board.evaluate_batch(movables)[:, 6]
    assert np.max(np.abs(wide_col - wide_board.severity.share_batch(perms, WE.P_WIDE))) == 0.0
    assert np.max(np.abs(narrow_col - narrow_board.severity.share_batch(perms, WE.P_NARROW))) == 0.0
    # And they are genuinely different rulers on random layouts (else the contrast is vacuous).
    assert np.max(np.abs(wide_col - narrow_col)) > 0.0


def test_severity_axes_fast_matches_slow_reference(boards):
    """The reporting path (fast) equals the zero-reuse slow path on every incumbent."""
    board = boards["wide"]
    worst = 0.0
    for string in CE.INCUMBENTS.values():
        fast = board.severity_axes(string)
        slow = board.severity_axes_slow(string)
        for key, value in fast.items():
            worst = max(worst, abs(value - slow[key]))
    assert worst == 0.0


def test_axes12_extends_the_ten_axis_frame(boards):
    board = boards["wide"]
    string = CE.INCUMBENTS["keybo-lsb"]
    ten = board.axes(string)
    twelve = board.axes12(string)
    assert set(twelve) == set(WE.AXES12)
    for axis, value in ten.items():
        assert twelve[axis] == value


def test_dominates12_is_reflexive_false_and_orientation_correct(boards):
    board = boards["wide"]
    a = board.axes12(CE.INCUMBENTS["keybo-lsb"])
    is_dom, n_ge, n_gt = WE.dominates12(a, a)
    assert (is_dom, n_ge, n_gt) == (False, 12, 0)
    better = dict(a)
    better["wscissor"] = a["wscissor"] - 1.0  # lower is better on a strain share
    is_dom, n_ge, n_gt = WE.dominates12(better, a)
    assert is_dom and n_ge == 12 and n_gt == 1
    worse = dict(a)
    worse["wscissor"] = a["wscissor"] + 1.0
    assert not WE.dominates12(worse, a)[0]


def test_both_severity_signs_are_lower_better():
    assert WE.SIGN12["wscissor"] == -1
    assert WE.SIGN12["nscissor"] == -1
    # and the ten inherited signs are untouched
    for axis in CE.AXES:
        assert WE.SIGN12[axis] == CE.SIGN[axis]
