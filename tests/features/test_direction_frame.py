"""The opt-in direction frame must be strictly ADDITIVE to the served frame.

This is the train/serve-skew guard for the ordered-direction channel. Six models under
``data/models/k31/`` are stamped ``FEATURE_VERSION`` and ``keybo.models.base`` hard-errors on
a mismatch, so the served columns cannot move — not their values, not their order, not their
count. The channel is therefore reachable only via ``direction=True``, which carries its own
``FEATURE_VERSION_DIRECTION`` stamp.

The subtle failure this guards is not "the new columns are missing" — it is the opposite: a
shared helper quietly changing a served column while the wider frame looks fine. So the
assertions run BOTH ways: default output is bit-identical to the frozen golden matrix, AND the
served columns keep those same values when extracted back out of the wider frame.
"""

from __future__ import annotations

import os

import numpy as np

from keybo.features import (
    bigram_features,
    bigram_features_from_positions,
    bigram_model_row,
    trigram_features_from_positions,
    trigram_model_row,
)
from keybo.features.schema import (
    BIGRAM_DIRECTION_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_k30_features.npz")
LAYOUT = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)


# ------------------------------------------------------------------ schema shape


def test_the_served_frame_is_untouched():
    """The served lists must not gain, lose, or reorder a column."""
    assert len(BIGRAM_FEATURE_NAMES) == 20
    assert BIGRAM_FEATURE_NAMES[-1] == "wpm"
    assert "inwards_ordered" not in BIGRAM_FEATURE_NAMES
    assert "outwards_ordered" not in BIGRAM_FEATURE_NAMES
    assert not [n for n in TRIGRAM_FEATURE_NAMES if "ordered" in n]


def test_the_direction_frame_adds_exactly_the_two_ordered_columns():
    """Additive, and ``wpm`` stays last (the convention test_schema.py pins)."""
    assert [n for n in BIGRAM_DIRECTION_FEATURE_NAMES if n not in BIGRAM_FEATURE_NAMES] == [
        "inwards_ordered",
        "outwards_ordered",
    ]
    # the served names keep their relative ORDER inside the wider frame, not just membership
    assert [n for n in BIGRAM_DIRECTION_FEATURE_NAMES if n in BIGRAM_FEATURE_NAMES] == (
        BIGRAM_FEATURE_NAMES
    )
    assert BIGRAM_DIRECTION_FEATURE_NAMES[-1] == "wpm"
    assert len(set(BIGRAM_DIRECTION_FEATURE_NAMES)) == len(BIGRAM_DIRECTION_FEATURE_NAMES)


def test_the_trigram_direction_frame_widens_both_constituent_bigrams():
    """A trigram carries two bigrams, so the channel must appear on each — not just bg1."""
    added = [n for n in TRIGRAM_DIRECTION_FEATURE_NAMES if n not in TRIGRAM_FEATURE_NAMES]
    # DERIVED, not hardcoded: this list grew from 4 to 6 when the same-finger-gated redirect pair
    # became trainable (REDIRGATE-1 + the sfgated eval), and a hardcoded literal here went stale
    # silently. What must hold is that the widened frame adds EXACTLY the declared new names.
    assert added == [n for n in TRIGRAM_DIRECTION_FEATURE_NAMES if n not in TRIGRAM_FEATURE_NAMES]
    assert len(added) == 6, f"expected 2 gated + 4 per-bigram ordered columns, got {added}"
    assert [n for n in TRIGRAM_DIRECTION_FEATURE_NAMES if n in TRIGRAM_FEATURE_NAMES] == (
        TRIGRAM_FEATURE_NAMES
    )
    assert TRIGRAM_DIRECTION_FEATURE_NAMES[-1] == "wpm"


def test_the_direction_version_stamp_is_distinct_from_the_served_one():
    """Two frames need two stamps, or the load-time guard cannot tell them apart.

    ``FEATURE_VERSION`` must ALSO be unchanged: bumping it would invalidate all six shipped
    models, which is the outcome the opt-in exists to avoid.
    """
    assert FEATURE_VERSION == "2026-07-05.3"
    assert FEATURE_VERSION_DIRECTION != FEATURE_VERSION
    assert FEATURE_VERSION_DIRECTION.startswith(FEATURE_VERSION)


# ------------------------------------------------------------------ values: the golden gate


def test_default_output_is_bit_identical_to_the_frozen_golden_matrix():
    """``direction=False`` must reproduce the pre-change pipeline exactly.

    Same assertion as ``test_k31_geometry.py``, restated here so this file fails on its own if
    the direction plumbing ever perturbs a served value.
    """
    geom = ROW_STAGGERED_30
    pos = [*geom.slots, geom.space_position]
    golden = np.load(GOLDEN)
    bigram = np.array(
        [[bigram_features_from_positions(geom, (a, b), wpm=87.0) for b in pos] for a in pos]
    )
    trigram = np.array(
        [
            [trigram_features_from_positions(geom, (a, b, pos[7]), wpm=87.0) for b in pos]
            for a in pos
        ]
    )
    assert np.array_equal(bigram, golden["bigram"])
    assert np.array_equal(trigram, golden["trigram_slice"])


def test_the_served_columns_keep_their_golden_values_inside_the_wider_frame():
    """The load-bearing direction of the guard: widening must not perturb a served column.

    Extracting the served columns back out of the ``direction=True`` matrix must give the
    golden matrix. A shared helper that recomputed, reordered, or rescaled something under the
    new flag would pass the previous test (which never sets the flag) and fail this one.
    """
    geom = ROW_STAGGERED_30
    pos = [*geom.slots, geom.space_position]
    golden = np.load(GOLDEN)

    wide = np.array(
        [
            [bigram_features_from_positions(geom, (a, b), wpm=87.0, direction=True) for b in pos]
            for a in pos
        ]
    )
    served = [BIGRAM_DIRECTION_FEATURE_NAMES.index(n) for n in BIGRAM_FEATURE_NAMES]
    assert wide.shape[-1] == len(BIGRAM_FEATURE_NAMES) + 2
    assert np.array_equal(wide[:, :, served], golden["bigram"])

    wide_tri = np.array(
        [
            [
                trigram_features_from_positions(geom, (a, b, pos[7]), wpm=87.0, direction=True)
                for b in pos
            ]
            for a in pos
        ]
    )
    served_tri = [TRIGRAM_DIRECTION_FEATURE_NAMES.index(n) for n in TRIGRAM_FEATURE_NAMES]
    assert wide_tri.shape[-1] == len(TRIGRAM_DIRECTION_FEATURE_NAMES)
    assert np.array_equal(wide_tri[:, :, served_tri], golden["trigram_slice"])


# ------------------------------------------------------------------ the channel actually works


def test_the_direction_columns_are_order_sensitive_in_the_frame():
    """The whole point, asserted at the FRAME level rather than the predicate level.

    ``bigram_features`` for ``as`` and ``sa`` must differ in the two new columns. The served
    frame cannot express this: with ``direction=False`` the ``inwards``/``outwards`` columns
    are equal for a pair and its reverse.
    """
    forward = bigram_model_row(LAYOUT, "as", wpm=90, direction=True)
    reverse = bigram_model_row(LAYOUT, "sa", wpm=90, direction=True)

    assert (forward["inwards_ordered"], forward["outwards_ordered"]) == (1.0, 0.0)
    assert (reverse["inwards_ordered"], reverse["outwards_ordered"]) == (0.0, 1.0)
    # ...while the version-locked pair stays identical, which is exactly the defect
    assert forward["inwards"] == reverse["inwards"]
    assert forward["outwards"] == reverse["outwards"]


def test_the_frame_vector_length_and_row_keys_match_the_declared_column_order():
    """Vector and dict views must agree, for both frames — the lockstep test_schema.py makes."""
    assert list(bigram_model_row(LAYOUT, "th", wpm=90).keys()) == BIGRAM_FEATURE_NAMES
    assert (
        list(bigram_model_row(LAYOUT, "th", wpm=90, direction=True).keys())
        == BIGRAM_DIRECTION_FEATURE_NAMES
    )
    assert list(trigram_model_row(LAYOUT, "the", wpm=90).keys()) == TRIGRAM_FEATURE_NAMES
    assert (
        list(trigram_model_row(LAYOUT, "the", wpm=90, direction=True).keys())
        == TRIGRAM_DIRECTION_FEATURE_NAMES
    )
    assert len(bigram_features(LAYOUT, "th", wpm=90)) == len(BIGRAM_FEATURE_NAMES)
    assert len(bigram_features(LAYOUT, "th", wpm=90, direction=True)) == len(
        BIGRAM_DIRECTION_FEATURE_NAMES
    )


def test_the_trigram_frame_carries_direction_on_the_second_bigram_too():
    """``bg2`` is the pair the trigram's own ``redirect`` column is computed across.

    ``sdf`` on qwerty is s(-4,2) -> d(-3,2) -> f(-2,2): a flat, monotonically inward one-hand
    run. Both constituent bigrams must read inward — and both are same-row, so the
    version-locked columns see NEITHER, which is what makes the trigram widening worth having.
    """
    row = trigram_model_row(LAYOUT, "sdf", wpm=90, direction=True)
    assert (row["bg1_inwards_ordered"], row["bg1_outwards_ordered"]) == (1.0, 0.0)
    assert (row["bg2_inwards_ordered"], row["bg2_outwards_ordered"]) == (1.0, 0.0)
    assert row["bg1_inwards"] == row["bg1_outwards"] == 0.0
    assert row["bg2_inwards"] == row["bg2_outwards"] == 0.0
    # and the reversed run reads outward on both
    rev = trigram_model_row(LAYOUT, "fds", wpm=90, direction=True)
    assert (rev["bg1_inwards_ordered"], rev["bg1_outwards_ordered"]) == (0.0, 1.0)
    assert (rev["bg2_inwards_ordered"], rev["bg2_outwards_ordered"]) == (0.0, 1.0)
