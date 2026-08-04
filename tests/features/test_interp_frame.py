"""INTERPFRAME-1's 10-column interpretability frame: schema, semantics, and ISOLATION.

The isolation tests are the load-bearing ones. This frame is a fourth model population and the
only *narrowing* one in the codebase, so the risks it introduces are (a) silently changing a
shipped frame and (b) a model being scored on a matrix it was not fitted for. Both get an
explicit test rather than a comment.
"""

import numpy as np
import pytest

from keybo.features import (
    BIGRAM_DIRECTION_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES,
    BIGRAM_INTERP_FEATURE_NAMES,
    BIGRAM_INTERP_MONOTONE,
    BIGRAM_KITCHENSINK_FEATURE_NAMES,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    FEATURE_VERSION_INTERP,
    FEATURE_VERSION_KITCHENSINK,
    TRIGRAM_FEATURE_NAMES,
    interp_features_from_positions,
    interp_row_from_positions,
)
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31

G = ROW_STAGGERED_30
G31 = ROW_STAGGERED_31
LETTERS = list(G.slots)
PAIRS = [(a, b) for a in LETTERS for b in LETTERS]


# --- schema ------------------------------------------------------------------------------


def test_row_keys_match_the_schema_in_order():
    row = interp_row_from_positions(G, (-4, 3), (-5, 2))
    assert list(row) == BIGRAM_INTERP_FEATURE_NAMES


def test_names_are_unique_and_there_are_ten():
    assert len(BIGRAM_INTERP_FEATURE_NAMES) == len(set(BIGRAM_INTERP_FEATURE_NAMES)) == 10


def test_the_monotone_tuple_is_one_entry_per_column_and_only_plus_or_minus_one():
    assert len(BIGRAM_INTERP_MONOTONE) == len(BIGRAM_INTERP_FEATURE_NAMES)
    assert set(BIGRAM_INTERP_MONOTONE) <= {1, -1}


def test_the_stamp_is_distinct_from_all_three_other_populations():
    """The load guard in models/base.py distinguishes populations BY THIS STRING alone."""
    others = {FEATURE_VERSION, FEATURE_VERSION_DIRECTION, FEATURE_VERSION_KITCHENSINK}
    assert FEATURE_VERSION_INTERP not in others
    assert len({*others, FEATURE_VERSION_INTERP}) == 4


def test_there_is_no_wpm_column_and_that_is_deliberate():
    """The whole point of the frame: wpm is CONSTANT at serve yet SHAP credited it -0.0922
    ms/char (9.2% of the T2 gap). A frame that keeps it cannot score CONSTFRAC == 0."""
    assert "wpm" not in BIGRAM_INTERP_FEATURE_NAMES


# --- ISOLATION: no shipped frame may change -----------------------------------------------


def test_the_interp_frame_shares_no_column_name_with_any_served_frame():
    """A shared NAME is the failure path: every downstream lookup in shap_diff and in the
    scorers is by column name, so a name present in two frames with two meanings would
    attribute one frame's number to the other frame's column while reconciling."""
    served = set(BIGRAM_FEATURE_NAMES) | set(TRIGRAM_FEATURE_NAMES)
    served |= set(BIGRAM_DIRECTION_FEATURE_NAMES) | set(BIGRAM_KITCHENSINK_FEATURE_NAMES)
    assert not (set(BIGRAM_INTERP_FEATURE_NAMES) & served)


def test_the_served_bigram_frame_is_untouched_at_twenty_columns_ending_in_wpm():
    assert len(BIGRAM_FEATURE_NAMES) == 20
    assert BIGRAM_FEATURE_NAMES[-1] == "wpm"


def test_the_served_trigram_frame_is_untouched_at_fortysix_columns():
    assert len(TRIGRAM_FEATURE_NAMES) == 46


# --- semantics: every column means what its name says --------------------------------------


def test_hand_conflict_is_the_bigramclass_ordering():
    for a, b in PAIRS:
        got = interp_row_from_positions(G, a, b)["hand_conflict"]
        cls = C.classify_positions(G, a, b)
        want = {
            C.BigramClass.ALTERNATE: 0.0,
            C.BigramClass.SAME_HAND: 1.0,
            C.BigramClass.SAME_FINGER: 2.0,
        }[cls]
        assert got == want, (a, b, cls)


def test_row_span_subsumes_scissor_exactly():
    """`scissor` is dy==2 on ADJACENT fingers; row_span==2 must fire on every scissor."""
    scissors = [(a, b) for a, b in PAIRS if C.is_scissor(G, a, b)]
    assert scissors, "fixture guard: no scissors on this geometry"
    for a, b in scissors:
        assert interp_row_from_positions(G, a, b)["row_span"] == 2.0


def test_row_span_also_fires_where_scissor_is_blind():
    """The point of grading: a two-row jump on NON-adjacent fingers (`row_skip`) is a real
    contortion that `scissor` cannot flag, and row_span prices it identically."""
    blind = [
        (a, b)
        for a, b in PAIRS
        if C.is_row_skip(G, a, b) and not C.is_scissor(G, a, b) and not C.same_finger(G, a, b)
    ]
    assert blind, "fixture guard: expected non-adjacent two-row jumps to exist"
    for a, b in blind:
        assert interp_row_from_positions(G, a, b)["row_span"] == 2.0


def test_row_span_is_zero_cross_hand_and_same_finger():
    for a, b in PAIRS:
        row = interp_row_from_positions(G, a, b)
        if C.classify_positions(G, a, b) is not C.BigramClass.SAME_HAND:
            assert row["row_span"] == 0.0, (a, b)


def test_lateral_span_is_the_gauges_own_predicate_bit_for_bit():
    """Failure mode 4 (name collisions) fixed BY CONSTRUCTION: the feature named
    `lateral_span` IS `keybo.features.classify.lateral_span`, the `lat-span` gauge's quantity."""
    for a, b in PAIRS:
        assert interp_row_from_positions(G, a, b)["lateral_span"] == C.lateral_span(G, a, b)


def test_same_hand_travel_is_distance_gated_on_same_hand():
    """The mechanistic fix: unconditioned `distance` prices long travel CHEAPER because long
    travel proxies for cross-hand. Gated, it can only rise with a real same-hand reach."""
    for a, b in PAIRS:
        row = interp_row_from_positions(G, a, b)
        if C.same_hand(G, a, b):
            assert row["same_hand_travel"] == G.distance(a, b)
        else:
            assert row["same_hand_travel"] == 0.0


def test_row_load_and_row_arrival_are_a_rotation_of_the_two_deviations():
    """(sum, difference) recovers (dev_a, dev_b) exactly — i.e. no information is lost by the
    rotation, which is what makes it an orthogonalization rather than a compression."""
    for a, b in PAIRS:
        row = interp_row_from_positions(G, a, b)
        dev_a, dev_b = abs(a[1] - 2), abs(b[1] - 2)
        assert row["row_load"] == pytest.approx(dev_a + dev_b)
        assert row["row_arrival"] == pytest.approx(dev_b - dev_a)
        # invertibility
        assert (row["row_load"] - row["row_arrival"]) / 2 == pytest.approx(dev_a)
        assert (row["row_load"] + row["row_arrival"]) / 2 == pytest.approx(dev_b)


def test_row_load_and_row_arrival_are_UNCORRELATED_over_the_grid():
    """The orthogonality claim, MEASURED not asserted: over the full ordered-pair enumeration
    the sum and difference of two identically-distributed deviations are uncorrelated."""
    rows = [interp_row_from_positions(G, a, b) for a, b in PAIRS]
    load = np.array([r["row_load"] for r in rows])
    arrival = np.array([r["row_arrival"] for r in rows])
    assert abs(np.corrcoef(load, arrival)[0, 1]) < 1e-12


def test_bottom_bias_is_signed_and_names_the_expensive_direction():
    for a, b in PAIRS:
        row = interp_row_from_positions(G, a, b)
        want = sum(1 for p in (a, b) if p[1] < 2) - sum(1 for p in (a, b) if p[1] > 2)
        assert row["bottom_bias"] == float(want)


def test_finger_load_rises_with_weakness():
    """Two pinky keys must score strictly worse than two index keys, or the +1 constraint's
    mechanism claim is backwards."""
    pinkies = [p for p in LETTERS if C.finger_kind(G, p[0]) == 0]
    indices = [p for p in LETTERS if C.finger_kind(G, p[0]) == 3]
    weak = interp_row_from_positions(G, pinkies[0], pinkies[-1])["finger_load"]
    strong = interp_row_from_positions(G, indices[0], indices[-1])["finger_load"]
    assert weak == 6.0 and strong == 0.0
    assert weak > strong


def test_off_home_column_counts_stretch_columns_not_stretches():
    """Renamed from `lateral` precisely because it measures a COLUMN, not a span. K30 has no
    |x|==6, so only the inner index column can fire here; K31 adds the quote slot."""
    for a, b in PAIRS:
        row = interp_row_from_positions(G, a, b)
        want = sum(1 for p in (a, b) if abs(p[0]) in (1, 6))
        assert row["off_home_column"] == float(want)
    quote = (6, 2)
    assert interp_row_from_positions(G31, quote, quote)["off_home_column"] == 2.0


def test_roll_inward_is_ANTISYMMETRIC_which_the_served_columns_are_not():
    """The honest direction of travel. `inwards`/`outwards` are swap-INVARIANT (0 of 870 pairs
    change under reversal); this column must NEGATE under reversal on every roll-eligible pair,
    which is the property that makes it a direction at all."""
    flipped = 0
    for a, b in PAIRS:
        fwd = interp_row_from_positions(G, a, b)["roll_inward"]
        rev = interp_row_from_positions(G, b, a)["roll_inward"]
        assert fwd == -rev, (a, b)
        if fwd != 0.0:
            flipped += 1
    assert flipped > 0, "fixture guard: no roll-eligible pairs found"


def test_roll_inward_partitions_the_roll_eligible_set():
    """+1 and -1 must exactly cover the eligible pairs, so one signed column loses nothing
    relative to the two ordered predicates it replaces."""
    inward = sum(
        1 for a, b in PAIRS if interp_row_from_positions(G, a, b)["roll_inward"] == 1.0
    )
    outward = sum(
        1 for a, b in PAIRS if interp_row_from_positions(G, a, b)["roll_inward"] == -1.0
    )
    eligible = sum(
        1
        for a, b in PAIRS
        if C.is_inwards_ordered(G, a, b) or C.is_outwards_ordered(G, a, b)
    )
    assert inward + outward == eligible
    assert inward == outward  # the enumeration contains both orderings of every pair


# --- the space key ------------------------------------------------------------------------


def test_every_per_key_column_is_zero_for_the_space_slot():
    """Space is thumb-pressed: no home column, no finger rank. Treated once, in one predicate,
    so the treatment cannot drift between columns."""
    space = G.space_position
    row = interp_row_from_positions(G, space, space)
    for name in ("row_load", "row_arrival", "bottom_bias", "finger_load", "off_home_column"):
        assert row[name] == 0.0, name


def test_no_column_is_nan_or_infinite_anywhere_on_the_full_grid():
    """Including space, both orderings — a nan would propagate silently through a weighted sum."""
    positions = [*G31.slots, G31.space_position]
    for a in positions:
        for b in positions:
            vec = interp_features_from_positions(G31, (a, b), wpm=90.0)
            assert np.all(np.isfinite(vec)), (a, b, vec)


# --- the vector path ----------------------------------------------------------------------


def test_the_vector_is_the_row_in_schema_order():
    row = interp_row_from_positions(G, (-4, 3), (2, 1))
    vec = interp_features_from_positions(G, ((-4, 3), (2, 1)), wpm=90.0)
    assert list(vec) == [row[n] for n in BIGRAM_INTERP_FEATURE_NAMES]


def test_wpm_is_accepted_and_IGNORED_not_appended():
    """Call-shape parity with the served featurizer, without a wpm column in the output: if
    wpm leaked in, the frame would be 11 columns and CONSTFRAC could not reach 0."""
    lo = interp_features_from_positions(G, ((-4, 3), (2, 1)), wpm=60.0)
    hi = interp_features_from_positions(G, ((-4, 3), (2, 1)), wpm=120.0)
    assert len(lo) == 10
    assert np.array_equal(lo, hi)
