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
    row = bigram_model_row(LAYOUT, "th", freq=100, wpm=90)
    assert list(row.keys()) == BIGRAM_FEATURE_NAMES


def test_trigram_row_keys_match_schema_in_order():
    row = trigram_model_row(
        LAYOUT, "the", tg_freq=100, bg1_freq=50, bg2_freq=40, sg_freq=30, wpm=90
    )
    assert list(row.keys()) == TRIGRAM_FEATURE_NAMES


def test_wpm_is_the_last_model_feature():
    assert BIGRAM_FEATURE_NAMES[-1] == "wpm"
    assert TRIGRAM_FEATURE_NAMES[-1] == "wpm"
