"""K31 gate A (PREREGISTRATIONS 2542bc4): the quote-slot extension must not move any
feature value on the 30-key domain, because K30-trained models keep loading without a
FEATURE_VERSION bump. The golden matrix was frozen from the pre-K31 pipeline."""

import os

import numpy as np
import pytest

from keybo.features import bigram_features_from_positions, trigram_features_from_positions
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_k30_features.npz")


def test_k30_features_bit_identical_to_golden():
    geom = ROW_STAGGERED_30
    pos = [*geom.slots, geom.space_position]
    g = np.load(GOLDEN)
    bi = np.array(
        [[bigram_features_from_positions(geom, (a, b), wpm=87.0) for b in pos] for a in pos]
    )
    tri = np.array(
        [
            [trigram_features_from_positions(geom, (a, b, pos[7]), wpm=87.0) for b in pos]
            for a in pos
        ]
    )
    assert np.array_equal(bi, g["bigram"])
    assert np.array_equal(tri, g["trigram_slice"])


def test_k31_slot_appended_not_inserted():
    assert ROW_STAGGERED_31.slots[:30] == ROW_STAGGERED_30.slots
    assert ROW_STAGGERED_31.slots[30] == (6, 2)


@pytest.mark.parametrize(
    ("a", "b", "pred", "expected"),
    [
        # quote slot is the right pinky: column 5 + 6 = one finger
        ((5, 2), (6, 2), "same_finger", True),
        ((6, 2), (5, 3), "same_finger", True),
        # ring -> quote is an adjacent (pinky-ring) two-finger pair, not a gap-2 skip
        ((4, 2), (6, 2), "is_adjacent", True),
        ((4, 3), (6, 2), "is_scissor", False),  # dy=1 is not a scissor
        ((4, 1), (6, 2), "is_scissor", False),
        # middle -> quote is NOT adjacent
        ((3, 2), (6, 2), "is_adjacent", False),
        # cross-hand never adjacent/same-finger
        ((-6, 2), (6, 2), "same_finger", False),
    ],
)
def test_k31_quote_slot_classification(a, b, pred, expected):
    fn = getattr(C, pred)
    assert fn(ROW_STAGGERED_31, a, b) is expected


def test_k31_quote_slot_feature_row():
    row = dict(
        zip(
            [
                "bottom",
                "home",
                "top",
                "pinky",
                "ring",
                "middle",
                "index",
                "lateral",
                "same_hand",
                "same_finger",
                "adjacent",
                "scissor",
                "lsb",
                "dx",
                "dy",
                "distance",
                "angle",
                "inwards",
                "outwards",
                "wpm",
            ],
            bigram_features_from_positions(ROW_STAGGERED_31, ((5, 2), (6, 2)), wpm=90.0),
            strict=False,
        )
    )
    assert row["pinky"] == 1.0
    assert row["lateral"] == 1.0  # quote column is the pinky's off-home stretch column
    assert row["same_finger"] == 1.0
    assert row["distance"] == 1.0
