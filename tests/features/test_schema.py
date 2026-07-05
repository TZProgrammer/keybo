"""The feature schema must stay in lockstep with what the pipeline actually produces.

If these fail after a feature change, that is the intended signal to bump FEATURE_VERSION
(which in turn invalidates saved models trained on the old features).
"""

from keybo.features import bigram_model_row, trigram_model_row
from keybo.features.schema import (
    BIGRAM_FEATURE_NAMES,
    FEATURE_VERSION,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout

LAYOUT = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", ROW_STAGGERED_30)


def test_feature_version_is_a_nonempty_string():
    assert isinstance(FEATURE_VERSION, str)
    assert FEATURE_VERSION


def test_bigram_feature_names_are_unique():
    assert len(BIGRAM_FEATURE_NAMES) == len(set(BIGRAM_FEATURE_NAMES))


def test_trigram_feature_names_are_unique():
    assert len(TRIGRAM_FEATURE_NAMES) == len(set(TRIGRAM_FEATURE_NAMES))


def test_bigram_row_keys_match_schema_in_order():
    row = bigram_model_row(LAYOUT, "th", wpm=90)
    assert list(row.keys()) == BIGRAM_FEATURE_NAMES


def test_trigram_row_keys_match_schema_in_order():
    row = trigram_model_row(LAYOUT, "the", wpm=90)
    assert list(row.keys()) == TRIGRAM_FEATURE_NAMES


def test_wpm_is_the_last_model_feature():
    assert BIGRAM_FEATURE_NAMES[-1] == "wpm"
    assert TRIGRAM_FEATURE_NAMES[-1] == "wpm"


# --- OQ-1 consequence: frequency is NOT a feature (weight-only + practice term) ---------


def test_freq_is_not_a_bigram_feature():
    """OQ-1 closed weight-only: the measured LOLO A/B showed freq-as-feature corrupts
    cross-layout ranking (tau +0.333 vs +0.667/+1.0 without it)."""
    assert "freq" not in BIGRAM_FEATURE_NAMES


def test_no_freq_features_anywhere_in_trigram_schema():
    """The constituent-frequency landmine (audit #5) dies with the schema, not a default."""
    assert not [n for n in TRIGRAM_FEATURE_NAMES if "freq" in n]


def test_feature_version_bumped_past_freq_era():
    """Models trained with the freq column must refuse to load (train/serve skew guard)."""
    assert FEATURE_VERSION > "2026-07-03.1"


def test_bigram_schema_keeps_row_and_finger_onehots():
    """Goodhart regression (2026-07-05, agent-artifacts/goodhart-row-blindness.md): a
    feature round removed these because held-out LOLO rho improved — and the next layout
    search parked junk on the home row: without the row one-hot, same-row bigrams are
    featurewise identical ACROSS rows, a null space real-layout evaluation can't see but
    the optimizer immediately exploits. They are load-bearing for the optimizer."""
    for name in ("bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral"):
        assert name in BIGRAM_FEATURE_NAMES, name
    for name in ("same_hand", "same_finger", "adjacent", "scissor", "lsb", "distance", "wpm"):
        assert name in BIGRAM_FEATURE_NAMES, name


def test_trigram_schema_keeps_its_constituent_placement_features():
    assert "bg1_home" in TRIGRAM_FEATURE_NAMES and "bg2_pinky" in TRIGRAM_FEATURE_NAMES
