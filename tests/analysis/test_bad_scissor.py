"""`bad-scissor`: the predicate, its denominator, and its per-finger attribution.

Every expected value here comes from the `badscissor` agent's specification
(``state/badscissor/badscissor-spec.md``, 2026-07-26), which derived them independently on
the Aalto keystroke frame and pinned them before this implementation existed. They are
therefore genuine positive controls, not self-consistency checks.

The three things this gauge is most likely to get wrong, each pinned:

* **the predicate's support** — §4.1's exhaustive 900-pair census, including that it is a
  CROSS-CUT of the incumbent supports rather than a superset;
* **the denominator** — space-EXCLUDED (the kmstats/sfb convention). Borrowing oxey's
  space-including denominator leaves the numerator bit-identical and inflates every share
  by ~1.497x (trap #9). The `sfb` positive control isolates exactly this;
* **the attribution rule** — the whole of a pair's mass to the DESCENDING (weaker) finger,
  whose structural consequence is that both index fingers are always 0.0.
"""

from __future__ import annotations

import itertools

import pytest

from keybo.analysis.bad_scissor import (
    ATTRIBUTION_RULE,
    BadScissor,
    bad_scissor,
    bad_scissor_finger,
)
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as GEOM
from keybo.geometry import Geometry
from keybo.layout import Layout

LAYOUTS = {
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "qwerty": "qwertyuiopasdfghjkl;zxcvbnm,./",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "dvorak": "',.pyfgcrlaoeuidhtns;qjkxbmwvz",
}

#: SPEC §5, flat weights, iWeb, layout-restricted space-EXCLUDED denominator.
SPEC_SHARES = {
    "lsb-sib": 2.49802,
    "archive-1843": 2.95116,
    "flagship-c3": 3.46985,
    "archive-1846": 3.64068,
    "keybo-lsb": 3.71019,
    "semimak": 3.92014,
    "keybo-lsb+lm": 4.11684,
    "graphite": 4.66037,
    "dvorak": 5.80304,
    "qwerty": 12.49998,
}

#: SPEC §5.1, per-finger flat attribution (both index fingers structurally 0.0).
SPEC_BY_FINGER = {
    "lsb-sib": {
        "L-pinky": 0.8108,
        "L-ring": 0.6089,
        "L-middle": 0.2913,
        "R-middle": 0.0667,
        "R-ring": 0.6611,
        "R-pinky": 0.0592,
    },
    "flagship-c3": {
        "L-pinky": 1.4975,
        "L-ring": 0.4597,
        "L-middle": 0.3190,
        "R-middle": 0.0667,
        "R-ring": 0.6611,
        "R-pinky": 0.4658,
    },
    "qwerty": {
        "L-pinky": 4.8837,
        "L-ring": 4.2784,
        "L-middle": 0.9725,
        "R-middle": 0.1595,
        "R-ring": 2.1945,
        "R-pinky": 0.0114,
    },
    "graphite": {
        "L-pinky": 1.5853,
        "L-ring": 0.3277,
        "L-middle": 0.2111,
        "R-middle": 0.3635,
        "R-ring": 0.7120,
        "R-pinky": 1.4607,
    },
}

#: SPEC §5.2, per-cell flat decomposition ("<pair> dy<n>").
SPEC_BY_CELL = {
    "qwerty": {
        "index-pinky dy1": 3.57668,
        "middle-ring dy1": 3.29886,
        "index-ring dy1": 2.81550,
        "middle-pinky dy1": 0.82107,
        "index-middle dy2": 0.79813,
        "ring-pinky dy1": 0.43289,
        "index-middle dy1": 0.33388,
        "middle-ring dy2": 0.21152,
        "index-ring dy2": 0.14702,
        "middle-pinky dy2": 0.05550,
        "index-pinky dy2": 0.00583,
        "ring-pinky dy2": 0.00311,
    },
    "flagship-c3": {
        "index-pinky dy1": 0.39396,
        "middle-ring dy1": 0.46268,
        "index-ring dy1": 0.53308,
        "middle-pinky dy1": 0.96957,
        "index-middle dy2": 0.00969,
        "ring-pinky dy1": 0.46892,
        "index-middle dy1": 0.37593,
        "middle-ring dy2": 0.05434,
        "index-ring dy2": 0.07077,
        "middle-pinky dy2": 0.10713,
        "index-pinky dy2": 0.01044,
        "ring-pinky dy2": 0.01333,
    },
}

#: SPEC §5.2: the dy==2 subtotal, the number that carries the argument for the predicate.
SPEC_DY2_SUBTOTAL = {"qwerty": 1.22111, "flagship-c3": 0.26570}


def _layout(lay30: str) -> Layout:
    return Layout(lay30, GEOM)


# --- §4.1 exhaustive predicate proof -----------------------------------------------------


@pytest.fixture(scope="module")
def pair_census() -> dict:
    pairs = list(itertools.product(GEOM.slots, repeat=2))
    bad = {(a, b) for a, b in pairs if bad_scissor(GEOM, a, b)}
    narrow = {(a, b) for a, b in pairs if C.is_scissor(GEOM, a, b)}
    wide = {
        (a, b)
        for a, b in pairs
        if C.same_hand(GEOM, a, b) and not C.same_finger(GEOM, a, b) and abs(a[1] - b[1]) == 2
    }
    return {"pairs": pairs, "bad": bad, "narrow": narrow, "wide": wide}


def test_ordered_pair_count_and_bad_scissor_support_size(pair_census):
    assert len(pair_census["pairs"]) == 900
    assert len(pair_census["bad"]) == 108


def test_row_span_census_is_72_dy1_and_36_dy2(pair_census):
    census: dict[int, int] = {}
    for a, b in pair_census["bad"]:
        span = abs(a[1] - b[1])
        census[span] = census.get(span, 0) + 1
    assert census == {1: 72, 2: 36}


def test_twelve_middle_pinky_pairs_are_in_support(pair_census):
    def kinds(a, b):
        return {GEOM.finger(a[0]).value.split("-")[1], GEOM.finger(b[0]).value.split("-")[1]}

    assert sum(1 for a, b in pair_census["bad"] if kinds(a, b) == {"middle", "pinky"}) == 12


def test_predicate_is_symmetric_over_every_pair(pair_census):
    violations = [
        (a, b)
        for a, b in pair_census["pairs"]
        if bad_scissor(GEOM, a, b) != bad_scissor(GEOM, b, a)
    ]
    assert violations == []


def test_bad_scissor_is_a_CROSS_CUT_of_narrow_and_wide(pair_census):
    """Neither subset nor superset — the property that makes agreement checks meaningful."""
    bad, narrow, wide = pair_census["bad"], pair_census["narrow"], pair_census["wide"]
    assert len(narrow) == 24
    assert len(wide) == 72
    assert len(narrow - bad) == 12
    assert len(bad - narrow) == 96
    assert len(wide - bad) == 36
    assert len(bad - wide) == 72


def test_every_excluded_narrow_and_wide_pair_is_weak_finger_on_TOP(pair_census):
    """The exclusions are exactly the classes the spec's fit measured as NOT strained."""
    dex = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}

    def weak_is_on_top(a, b):
        fa = GEOM.finger(a[0]).value.split("-")[1]
        fb = GEOM.finger(b[0]).value.split("-")[1]
        weak = min((fa, fb), key=lambda f: dex[f])
        weak_y = a[1] if fa == weak else b[1]
        strong_y = b[1] if fa == weak else a[1]
        return weak_y > strong_y

    for a, b in pair_census["narrow"] - pair_census["bad"]:
        assert weak_is_on_top(a, b)
    for a, b in pair_census["wide"] - pair_census["bad"]:
        assert weak_is_on_top(a, b)


def test_adjacency_split_is_36_adjacent_and_72_nonadjacent(pair_census):
    adjacent = sum(1 for a, b in pair_census["bad"] if C.is_adjacent(GEOM, a, b))
    assert adjacent == 36
    assert len(pair_census["bad"]) - adjacent == 72


def test_the_index_finger_is_never_attributed(pair_census):
    """Structural: the index is the most dextrous, so it is never the WEAKER member."""
    for a, b in pair_census["pairs"]:
        assert not (bad_scissor_finger(GEOM, a, b) or "").endswith("index")


def test_a_four_row_geometry_is_refused_loudly():
    """A board the spec's support and expected values were never derived on must fail loudly."""
    four_rows = Geometry(slots=(*GEOM.slots, (-5, 4)))
    chars31 = "qwertyuiopasdfghjkl'zxcvbnm,.-;"
    with pytest.raises(ValueError, match="three-row"):
        BadScissor({"qw": 1}).share(Layout(chars31, four_rows))


# --- §4.2 the sfb positive control (isolates the DENOMINATOR) ----------------------------


@pytest.mark.parametrize("label", sorted(LAYOUTS))
def test_sfb_reproduces_kmstats_exactly_through_our_own_denominator(corpora, label):
    """Our scoring loop + our denominator, run on the `sfb` predicate, must match kmstats.

    `sfb` is definitionally identical in kmstats and classify and its support is DISJOINT
    from bad-scissor's, so any disagreement isolates the denominator — which is the whole
    point (trap #9), and is not the nested-guard mistake of trap #11.
    """
    from keybo.analysis.kmstats import KmStats

    bigrams, skipgrams, trigrams = corpora
    lay30 = LAYOUTS[label]
    expected = KmStats(bigrams, skipgrams, trigrams).stats(lay30)["sfb"]
    got = BadScissor(bigrams).share_of(
        _layout(lay30),
        lambda geometry, a, b: a != b and geometry.same_finger(a[0], b[0]),
    )
    assert got == expected, f"{label}: denominator mismatch"


# --- §5 expected values ------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(SPEC_SHARES))
def test_share_matches_the_specification(corpora, label):
    bigrams, *_ = corpora
    got = BadScissor(bigrams).share(_layout(LAYOUTS[label]))
    assert got == pytest.approx(SPEC_SHARES[label], abs=5e-5)


@pytest.mark.parametrize("label", sorted(SPEC_BY_FINGER))
def test_per_finger_attribution_matches_the_specification(corpora, label):
    bigrams, *_ = corpora
    got = BadScissor(bigrams).by_finger(_layout(LAYOUTS[label]))
    for finger, expected in SPEC_BY_FINGER[label].items():
        assert got[finger] == pytest.approx(expected, abs=5e-5), f"{label}/{finger}"


@pytest.mark.parametrize("label", sorted(LAYOUTS))
def test_per_finger_attribution_is_an_exact_partition(corpora, label):
    bigrams, *_ = corpora
    scorer = BadScissor(bigrams)
    layout = _layout(LAYOUTS[label])
    assert sum(scorer.by_finger(layout).values()) == pytest.approx(
        scorer.share(layout), rel=0, abs=1e-9
    )


@pytest.mark.parametrize("label", sorted(LAYOUTS))
def test_both_index_fingers_are_structurally_zero(corpora, label):
    bigrams, *_ = corpora
    got = BadScissor(bigrams).by_finger(_layout(LAYOUTS[label]))
    assert got["L-index"] == 0.0 and got["R-index"] == 0.0


@pytest.mark.parametrize("label", sorted(SPEC_BY_CELL))
def test_per_cell_decomposition_matches_the_specification(corpora, label):
    bigrams, *_ = corpora
    got = BadScissor(bigrams).by_cell(_layout(LAYOUTS[label]))
    for cell, expected in SPEC_BY_CELL[label].items():
        assert got[cell] == pytest.approx(expected, abs=5e-5), f"{label}/{cell}"


@pytest.mark.parametrize("label", sorted(SPEC_DY2_SUBTOTAL))
def test_dy2_subtotal_matches_the_specification(corpora, label):
    """The number that carries the argument: the incumbent dy==2 gate sees <10% of this."""
    bigrams, *_ = corpora
    cells = BadScissor(bigrams).by_cell(_layout(LAYOUTS[label]))
    dy2 = sum(value for cell, value in cells.items() if cell.endswith("dy2"))
    assert dy2 == pytest.approx(SPEC_DY2_SUBTOTAL[label], abs=5e-5)


@pytest.mark.parametrize("label", sorted(LAYOUTS))
def test_per_cell_decomposition_is_an_exact_partition(corpora, label):
    bigrams, *_ = corpora
    scorer = BadScissor(bigrams)
    layout = _layout(LAYOUTS[label])
    assert sum(scorer.by_cell(layout).values()) == pytest.approx(
        scorer.share(layout), rel=0, abs=1e-9
    )


# --- §4.3 / §4.4 ordering and the denominator regression ---------------------------------


def test_qwerty_is_the_worst_of_all_ten_layouts(corpora):
    bigrams, *_ = corpora
    scorer = BadScissor(bigrams)
    shares = {label: scorer.share(_layout(lay)) for label, lay in LAYOUTS.items()}
    assert max(shares, key=shares.__getitem__) == "qwerty"


def test_qwerty_is_worse_than_flagship_c3(corpora):
    bigrams, *_ = corpora
    scorer = BadScissor(bigrams)
    assert scorer.share(_layout(LAYOUTS["qwerty"])) > scorer.share(_layout(LAYOUTS["flagship-c3"]))


@pytest.mark.parametrize("label", sorted(LAYOUTS))
def test_the_space_including_denominator_moves_every_share_by_about_1_497x(corpora, label):
    """TRAP #9 REGRESSION: borrowing oxey's denominator keeps the numerator bit-identical.

    Space is in no bad-scissor pair (``hand(0) == 0``), so the wrong denominator is
    invisible to any numerator check and shows up only as a plausible ~1.497x constant on
    every share. This asserts the shipped number is the space-EXCLUDED one.

    **Direction, measured:** the space-INCLUDING denominator is the LARGER one (space-
    touching bigrams are 33.8% of the corpus mass), so the wrong choice **deflates** every
    share — ``correct / wrong`` is the 1.4961..1.4999 factor. The spec's §0/§2.5 wording
    ("inflates every share by ... 1.4961-1.4999x") has the direction backwards while the
    magnitude is exactly right; the ratio is asserted in the measured direction here.
    """
    bigrams, *_ = corpora
    layout = _layout(LAYOUTS[label])
    scorer = BadScissor(bigrams)
    correct = scorer.share(layout)
    wrong = scorer.share(layout, exclude_space=False)
    assert correct > wrong, f"{label}: the space-including denominator must be the larger one"
    assert 1.4961 <= correct / wrong <= 1.4999, f"{label}: ratio {correct / wrong}"


def test_empty_corpus_returns_zeros_not_a_crash():
    scorer = BadScissor({})
    layout = _layout(LAYOUTS["qwerty"])
    assert scorer.share(layout) == 0.0
    assert set(scorer.by_finger(layout)) >= {"L-pinky", "R-index"}
    assert all(value == 0.0 for value in scorer.by_finger(layout).values())
    assert scorer.by_cell(layout) == {}


def test_attribution_rule_is_named_in_the_module():
    assert ATTRIBUTION_RULE == "all-to-descending-weaker-finger"
