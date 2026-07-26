"""The shared n-gram feature pipeline.

Import feature builders from here; :mod:`keybo.features.schema` owns the column order and
version, and :mod:`keybo.features.classify` owns the geometric predicates.
"""

from keybo.features.ngram import (
    bigram_feature_names,
    bigram_features,
    bigram_features_from_positions,
    bigram_model_row,
    feature_version,
    trigram_feature_names,
    trigram_features,
    trigram_features_from_positions,
    trigram_model_row,
)
from keybo.features.schema import (
    BIGRAM_DIRECTION_NAMES,
    BIGRAM_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES_DIRECTION,
    BIGRAM_FEATURE_NAMES_PLACEBO,
    BIGRAM_PLACEBO_NAMES,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    FEATURE_VERSION_PLACEBO,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES_DIRECTION,
    TRIGRAM_FEATURE_NAMES_PLACEBO,
)

__all__ = [
    "BIGRAM_DIRECTION_NAMES",
    "BIGRAM_FEATURE_NAMES",
    "BIGRAM_FEATURE_NAMES_DIRECTION",
    "BIGRAM_FEATURE_NAMES_PLACEBO",
    "BIGRAM_PLACEBO_NAMES",
    "FEATURE_VERSION",
    "FEATURE_VERSION_DIRECTION",
    "FEATURE_VERSION_PLACEBO",
    "TRIGRAM_FEATURE_NAMES",
    "TRIGRAM_FEATURE_NAMES_DIRECTION",
    "TRIGRAM_FEATURE_NAMES_PLACEBO",
    "bigram_feature_names",
    "bigram_features",
    "bigram_features_from_positions",
    "bigram_model_row",
    "feature_version",
    "trigram_feature_names",
    "trigram_features",
    "trigram_features_from_positions",
    "trigram_model_row",
]
