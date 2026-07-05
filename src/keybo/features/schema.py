"""Canonical feature order and version.

``FEATURE_VERSION`` is stamped into every trained model's metadata. Any change to the
feature set (order, names, or meaning) MUST bump it; loading a model whose stored version
differs from this one is a hard error, which is what prevents silently scoring with a model
trained on a different feature layout (train/serve skew).

Frequency is deliberately NOT a feature (2026-07-05.1): the real-data LOLO A/B (OQ-1)
showed freq-as-feature corrupts cross-layout ranking — with 98.7%-qwerty data it acts as a
per-position memorization key. Frequency lives in exactly two places instead: the objective
WEIGHT (fitness = sum time*freq) and the identity key of the additive practice term the
training pipeline residualizes out (see keybo.training.train).

The bigram set is the RELATIONAL + GEOMETRIC core only (2026-07-05.2): the feature-arm
matrix (agent-artifacts/bigram-experiment-backlog.md; runs/feature_arms{,2}.json) measured
that the second-key row/finger one-hots were memorization capacity, not transferable
signal — dropping them plus capping tree depth at 3 raised held-out rho from .94 to ~1.0
of the noise ceiling at layout-ranking tau +1.0. The trigram set still carries the full
per-constituent placement features: the C2A5 evidence is bigram-only, and the trigram
world gets its own harness round before its schema moves.

The name lists here are the single source of truth for column order. ``keybo.features.ngram``
produces rows keyed by (a superset of) exactly these names, and a test asserts the two stay
in lockstep.
"""

FEATURE_VERSION = "2026-07-05.2"

# The bigram feature set: relational + geometric core (C2A5). Character identity and
# absolute key position are deliberately absent — position enters only through relations.
_BIGRAM_RELATIONAL_GEOMETRIC = [
    # relational
    "same_hand",
    "same_finger",
    "adjacent",
    "scissor",
    "lsb",
    # geometry
    "dx",
    "dy",
    "distance",
    "angle",
    "inwards",
    "outwards",
]

BIGRAM_FEATURE_NAMES = [*_BIGRAM_RELATIONAL_GEOMETRIC, "wpm"]

# The full placement row (second-key one-hots + the core) — still produced by the pipeline
# and still consumed by the TRIGRAM constituents below.
_BIGRAM_PLACEMENT_NAMES = [
    # second-key row (one-hot)
    "bottom",
    "home",
    "top",
    # second-key finger (one-hot; index covers columns 1 and 2)
    "pinky",
    "ring",
    "middle",
    "index",
    "lateral",
    *_BIGRAM_RELATIONAL_GEOMETRIC,
]

# Trigram-level features, then the skipgram (first+third key) features, then the two
# constituent bigrams' placement features (prefixed), then wpm.
_TRIGRAM_LEVEL_NAMES = [
    "same_hand_trigram",
    "redirect",
    "bad_redirect",
    "sg_same_finger",
    "sg_dx",
    "sg_dy",
    "sg_distance",
]

TRIGRAM_FEATURE_NAMES = [
    *_TRIGRAM_LEVEL_NAMES,
    *(f"bg1_{n}" for n in _BIGRAM_PLACEMENT_NAMES),
    *(f"bg2_{n}" for n in _BIGRAM_PLACEMENT_NAMES),
    "wpm",
]
