"""Feature-pipeline tests — the heaviest coverage in the suite.

This is the single code path shared by data processing, training, and scoring, so a bug
here is a train/serve-skew bug. Includes regression guards (with negative controls) for:
  - bug #3: SFB must mean *same finger*, not index-vs-middle.
  - bug #2: features must respond to *every* key, including the punctuation keys the old
    28-character scorer ignored.
"""

import numpy as np
import pytest

from keybo.features import bigram_features, bigram_model_row, trigram_features
from keybo.features.classify import BigramClass, classify_bigram
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout

LAYOUT = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", ROW_STAGGERED_30)


# --- classification: same-finger (SFB) is the headline fix ----------------------------


def test_regression_bug3_index_columns_are_a_single_finger_bigram():
    """'ft': f at (-2,2) index, t at (-1,3) index -> same finger, a true SFB."""
    assert classify_bigram(LAYOUT, "ft") == BigramClass.SAME_FINGER


def test_regression_bug3_index_and_middle_are_not_a_single_finger_bigram():
    """'fd': f at (-2,2) index, d at (-3,2) middle -> different fingers, NOT an SFB."""
    assert classify_bigram(LAYOUT, "fd") != BigramClass.SAME_FINGER


def test_negative_control_old_sfb_formula_would_misclassify():
    """Proves the two tests above are not vacuous by reproducing the OLD formula.

    Old code: sfb = (ax == bx) or (same_hand and ((index1 and middle2) or (index2 and middle1)))
    That flags 'fd' (index+middle) as SFB and misses 'ft' (index+index). Assert the new
    classifier disagrees with the old one on exactly these cases.
    """

    def old_sfb(bg):
        (ax, _), (bx, _) = LAYOUT.pos(bg[0]), LAYOUT.pos(bg[1])
        index1, index2 = abs(ax) in (1, 2), abs(bx) in (1, 2)
        middle1, middle2 = abs(ax) == 3, abs(bx) == 3
        same_hand = ax != 0 and bx != 0 and (ax // abs(ax)) == (bx // abs(bx))
        return (ax == bx) or (same_hand and ((index1 and middle2) or (index2 and middle1)))

    new_ft = classify_bigram(LAYOUT, "ft") == BigramClass.SAME_FINGER
    new_fd = classify_bigram(LAYOUT, "fd") == BigramClass.SAME_FINGER
    assert old_sfb("ft") is False and new_ft is True  # old missed it, new catches it
    assert old_sfb("fd") is True and new_fd is False  # old false-positive, new rejects


def test_same_column_is_same_finger():
    # 'qa': both column -5 (left pinky), rows 3 and 2.
    assert classify_bigram(LAYOUT, "qa") == BigramClass.SAME_FINGER


def test_alternating_hands_is_alternate_class():
    # 'jf': j right index (1,2)? j is at home col... 'j' -> (1,2) right index, 'f' -> (-2,2) left.
    assert classify_bigram(LAYOUT, "jf") == BigramClass.ALTERNATE


def test_same_hand_different_finger_is_same_hand_class():
    # 'as': a (-5,2) pinky, s (-4,2) ring -> same hand, different finger.
    assert classify_bigram(LAYOUT, "as") == BigramClass.SAME_HAND


# --- bigram feature vector ------------------------------------------------------------


def test_bigram_features_returns_1d_float_array():
    vec = bigram_features(LAYOUT, "th")
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float64
    assert vec.ndim == 1


def test_bigram_same_finger_flag_matches_classifier():
    row = bigram_model_row(LAYOUT, "ft", freq=100, wpm=90)
    assert row["same_finger"] == 1.0
    row2 = bigram_model_row(LAYOUT, "fd", freq=100, wpm=90)
    assert row2["same_finger"] == 0.0


def test_bigram_dx_dy_use_stagger_offsets():
    # 'ft': f(-2,2) home index, t(-1,3) top index. dy=1.
    # dx = |(-2 + 0.0) - (-1 + -0.25)| = |-0.75| = 0.75
    row = bigram_model_row(LAYOUT, "ft", freq=100, wpm=90)
    assert row["dy"] == pytest.approx(1.0)
    assert row["dx"] == pytest.approx(0.75)


def test_bigram_row_features_are_one_hot_on_second_key():
    # 'ft' second key t is top row.
    row = bigram_model_row(LAYOUT, "ft", freq=100, wpm=90)
    assert row["top"] == 1.0
    assert row["home"] == 0.0
    assert row["bottom"] == 0.0


def test_bigram_frequency_and_wpm_are_passed_through():
    row = bigram_model_row(LAYOUT, "th", freq=1234, wpm=88)
    assert row["freq"] == 1234.0
    assert row["wpm"] == 88.0


# --- geometric predicate coverage (hand-verified against the geometry) -----------------


def test_scissor_flagged_for_adjacent_two_row_reach():
    # 'qx': q(-5,3) pinky top, x(-4,1) ring bottom -> adjacent fingers, 2 rows apart.
    row = bigram_model_row(LAYOUT, "qx", freq=1, wpm=90)
    assert row["scissor"] == 1.0
    assert row["adjacent"] == 1.0


def test_lsb_flagged_for_index_middle_wide_stretch():
    # 'et': e(-3,3) middle, t(-1,3) index, stagger dx = 2.0 (> 1.5) -> lateral stretch.
    row = bigram_model_row(LAYOUT, "et", freq=1, wpm=90)
    assert row["lsb"] == 1.0


def test_alternating_hand_bigram_has_no_same_hand_geometry():
    # 'jf': right index to left index -> alternate; same-hand-only flags must be 0.
    row = bigram_model_row(LAYOUT, "jf", freq=1, wpm=90)
    assert row["same_hand"] == 0.0
    assert row["adjacent"] == 0.0
    assert row["scissor"] == 0.0
    assert row["lsb"] == 0.0
    assert row["inwards"] == 0.0
    assert row["outwards"] == 0.0


def test_inwards_and_outwards_are_mutually_exclusive_same_hand_rolls():
    # 'as': a(-5,2) pinky, s(-4,2) ring -> same row, no roll direction.
    row = bigram_model_row(LAYOUT, "as", freq=1, wpm=90)
    assert row["inwards"] == 0.0 and row["outwards"] == 0.0
    # 'qs': q(-5,3) pinky top, s(-4,2) ring home. Outer=q higher row -> inwards roll.
    row2 = bigram_model_row(LAYOUT, "qs", freq=1, wpm=90)
    assert (row2["inwards"], row2["outwards"]) == (1.0, 0.0)


def test_trigram_skipgram_same_finger_detected():
    from keybo.features import trigram_model_row

    # 'tat': t(-1,3) and t... skipgram is first+third = 't','t' same key -> same finger.
    row = trigram_model_row(LAYOUT, "tat", tg_freq=1, bg1_freq=1, bg2_freq=1, sg_freq=1, wpm=90)
    assert row["sg_same_finger"] == 1.0


def test_trigram_embeds_both_constituent_bigrams():
    from keybo.features import trigram_model_row

    row = trigram_model_row(LAYOUT, "the", tg_freq=1, bg1_freq=1, bg2_freq=1, sg_freq=1, wpm=90)
    # bg1 = 'th', bg2 = 'he'; both prefixes present.
    assert any(name.startswith("bg1_") for name in row)
    assert any(name.startswith("bg2_") for name in row)
    # bg1's second key is 'h', bg2's second key is 'e'; their row one-hots should reflect that.
    assert row["bg1_home"] == 1.0  # h is home row


def test_alternate_class_when_hands_differ():
    from keybo.features import trigram_model_row

    # A cross-hand trigram is not a same-hand trigram.
    row = trigram_model_row(LAYOUT, "the", tg_freq=1, bg1_freq=1, bg2_freq=1, sg_freq=1, wpm=90)
    assert row["same_hand_trigram"] == 0.0
    assert row["redirect"] == 0.0


# --- REGRESSION bug #2: every key affects the objective's inputs ----------------------


def test_regression_bug2_punctuation_keys_are_not_invisible():
    """Bigrams over the apostrophe/hyphen keys must produce real, distinct features.

    The old scorer only formed bigrams from a hardcoded 28-char set that excluded ' and -,
    so those keys were invisible to the objective. Here the pipeline must feature them.
    """
    # A bigram involving the apostrophe must differ from one that swaps its position.
    base = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", ROW_STAGGERED_30)
    swapped = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", ROW_STAGGERED_30)
    swapped.swap("'", "-")
    # "'a" bigram: same characters, but ' is in a different physical spot after the swap.
    v_base = bigram_features(base, "'a")
    v_swapped = bigram_features(swapped, "'a")
    assert not np.array_equal(v_base, v_swapped)


def test_no_character_identity_features():
    """Raw key-ID features are intentionally dropped (they drove layout overfitting)."""
    row = bigram_model_row(LAYOUT, "th", freq=100, wpm=90)
    assert "k1_id" not in row
    assert "k2_id" not in row
    # Two different bigrams whose keys sit at identical positions/classes must map to
    # identical features (they cannot, if identity leaked in). Build a symmetric case:
    # 'ft' and 'jy' are both index home->top same-finger reaches on opposite hands, but
    # mirror geometry differs, so instead assert directly that no feature name encodes a
    # character identity or index.
    assert not any(name.endswith("_id") or name in {"k1", "k2"} for name in row)


# --- trigram feature vector -----------------------------------------------------------


def test_trigram_features_returns_1d_float_array():
    vec = trigram_features(LAYOUT, "the")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1


def test_trigram_same_hand_trigram_flag():
    from keybo.features import trigram_model_row

    # 'sad' on qwerty-ish: s(-4,2), a(-5,2), d(-3,2) all left hand -> same-hand trigram.
    row = trigram_model_row(LAYOUT, "sad", tg_freq=10, bg1_freq=5, bg2_freq=5, sg_freq=3, wpm=90)
    assert row["same_hand_trigram"] == 1.0


def test_trigram_redirect_detected_on_direction_change():
    from keybo.features import trigram_model_row

    # 'sad': columns -4, -5, -3. |−4|>|−5|? no. Direction: a->d outward? Uses abs columns.
    # s(4)->a(5) outward, a(5)->d(3) inward => redirect.
    row = trigram_model_row(LAYOUT, "sad", tg_freq=10, bg1_freq=5, bg2_freq=5, sg_freq=3, wpm=90)
    assert row["redirect"] == 1.0
