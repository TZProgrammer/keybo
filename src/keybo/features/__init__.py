"""The shared n-gram feature pipeline.

Import feature builders from here; :mod:`keybo.features.schema` owns the column order and
version, and :mod:`keybo.features.classify` owns the geometric predicates.
"""

from keybo.features.ngram import (
    bigram_features,
    bigram_features_from_positions,
    bigram_model_row,
    trigram_features,
    trigram_features_from_positions,
    trigram_model_row,
)
from keybo.features.schema import (
    BIGRAM_DIRECTION_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES,
    BIGRAM_KITCHENSINK_FEATURE_NAMES,
    BIGRAM_LATSPAN_ADD_FEATURE_NAMES,
    BIGRAM_LATSPAN_REPLACE_FEATURE_NAMES,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    FEATURE_VERSION_KITCHENSINK,
    FEATURE_VERSION_LATSPAN_ADD,
    FEATURE_VERSION_LATSPAN_REPLACE,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_KITCHENSINK_FEATURE_NAMES,
    TRIGRAM_LATSPAN_ADD_FEATURE_NAMES,
    TRIGRAM_LATSPAN_REPLACE_FEATURE_NAMES,
)

__all__ = [
    "BIGRAM_DIRECTION_FEATURE_NAMES",
    "BIGRAM_FEATURE_NAMES",
    "BIGRAM_KITCHENSINK_FEATURE_NAMES",
    "BIGRAM_LATSPAN_ADD_FEATURE_NAMES",
    "BIGRAM_LATSPAN_REPLACE_FEATURE_NAMES",
    "FEATURE_VERSION",
    "FEATURE_VERSION_DIRECTION",
    "FEATURE_VERSION_KITCHENSINK",
    "FEATURE_VERSION_LATSPAN_ADD",
    "FEATURE_VERSION_LATSPAN_REPLACE",
    "TRIGRAM_DIRECTION_FEATURE_NAMES",
    "TRIGRAM_FEATURE_NAMES",
    "TRIGRAM_KITCHENSINK_FEATURE_NAMES",
    "TRIGRAM_LATSPAN_ADD_FEATURE_NAMES",
    "TRIGRAM_LATSPAN_REPLACE_FEATURE_NAMES",
    "bigram_features",
    "bigram_features_from_positions",
    "bigram_model_row",
    "trigram_features",
    "trigram_features_from_positions",
    "trigram_model_row",
]
