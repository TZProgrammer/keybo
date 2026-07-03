"""The shared n-gram feature pipeline.

Import feature builders from here; :mod:`keybo.features.schema` owns the column order and
version, and :mod:`keybo.features.classify` owns the geometric predicates.
"""

from keybo.features.ngram import (
    bigram_features,
    bigram_model_row,
    trigram_features,
    trigram_model_row,
)
from keybo.features.schema import (
    BIGRAM_FEATURE_NAMES,
    FEATURE_VERSION,
    TRIGRAM_FEATURE_NAMES,
)

__all__ = [
    "BIGRAM_FEATURE_NAMES",
    "FEATURE_VERSION",
    "TRIGRAM_FEATURE_NAMES",
    "bigram_features",
    "bigram_model_row",
    "trigram_features",
    "trigram_model_row",
]
