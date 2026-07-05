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

The row/finger one-hots are LOAD-BEARING for the optimizer (2026-07-05.3): a feature-arm
round briefly removed them (version .2) because held-out LOLO rho improved — and the very
next layout search exposed the trap: without the second-key row one-hot, every same-row
bigram is featurewise IDENTICAL across rows (home a-s == bottom z-x), so the optimizer
parked junk on the home row and vowels on the bottom row, exploiting a null space the
harness cannot see (LOLO evaluates on real layouts, which all use rows sensibly; the
optimizer queries OFF that distribution). Full placement features restored; the measured
transfer win is kept via tree depth 3 instead (same LOLO gain, no information deleted).
See agent-artifacts/goodhart-row-blindness.md.

The name lists here are the single source of truth for column order. ``keybo.features.ngram``
produces rows keyed by (a superset of) exactly these names, and a test asserts the two stay
in lockstep.
"""

FEATURE_VERSION = "2026-07-05.3"

# Placement / relational / geometry features for a single bigram, in order. Row and finger
# one-hots describe the *second* (landing) key; the first key enters through the relational
# and geometric features. Character identity is deliberately absent.
_BIGRAM_PLACEMENT_NAMES = [
    # second-key row (one-hot) — REQUIRED: without it, same-row bigrams are identical
    # across rows and the optimizer exploits the blindness (see module docstring).
    "bottom",
    "home",
    "top",
    # second-key finger (one-hot; index covers columns 1 and 2)
    "pinky",
    "ring",
    "middle",
    "index",
    "lateral",
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

BIGRAM_FEATURE_NAMES = [*_BIGRAM_PLACEMENT_NAMES, "wpm"]

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
