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

# --- the ORDER-AWARE direction channel (opt-in, additive) ---------------------------------
#
# ``inwards``/``outwards`` above are SWAP-INVARIANT: over all 870 ordered position pairs of
# ROW_STAGGERED_30, the number whose value changes when the pair is reversed is 0 (they sort
# the pair by column magnitude, then compare rows — see keybo.features.classify). So the
# served frame carries no direction-of-travel channel at all: direction enters it only
# through the landing-key one-hots, computed from the second key alone.
#
# These two columns are the honest channel. They are kept OUT of BIGRAM_FEATURE_NAMES and
# behind an explicit opt-in for one reason: the six models under data/models/k31/ are stamped
# FEATURE_VERSION above, and keybo.models.base hard-errors on a mismatch. Widening the served
# list would invalidate all six; redefining the existing columns in place would be worse —
# they would keep loading and silently score on a frame whose columns 18/19 no longer mean
# what they meant at training time. Additive-and-opt-in is the only option that neither
# breaks a shipped artifact nor lies about one.
#
# A model trained on the wider frame records FEATURE_VERSION_DIRECTION instead, so the two
# populations can never be confused for one another.
_BIGRAM_DIRECTION_NAMES = [
    "inwards_ordered",
    "outwards_ordered",
]

#: The served bigram frame plus the ordered-direction channel. ``wpm`` stays last (a
#: convention ``tests/features/test_schema.py`` pins), so the new columns are inserted before
#: it rather than appended to the end.
BIGRAM_DIRECTION_FEATURE_NAMES = [
    *_BIGRAM_PLACEMENT_NAMES,
    *_BIGRAM_DIRECTION_NAMES,
    "wpm",
]

#: Stamped instead of :data:`FEATURE_VERSION` by anything trained on the wider frame. It must
#: never equal ``FEATURE_VERSION``, or the load-time guard could not tell the frames apart.
FEATURE_VERSION_DIRECTION = f"{FEATURE_VERSION}+direction.1"

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

#: The trigram frame with the ordered-direction channel on BOTH constituent bigrams. The
#: trigram-level ``redirect``/``bad_redirect`` columns are already order-aware (they compare
#: ``|column|`` between successive keys), so what this adds is per-bigram direction, not a
#: first direction signal at the trigram level.
#: The same-finger-GATED redirect pair (REDIRGATE-1). Appended to the WIDENED trigram list only —
#: never to :data:`TRIGRAM_FEATURE_NAMES`, because that list IS the version-locked served frame all
#: three shipped ``trigram_cond31`` models carry. ``_TRIGRAM_LEVEL_NAMES`` is shared by both lists,
#: so these must NOT go there either: adding a column to the shared prefix would silently widen the
#: served frame.
_TRIGRAM_GATED_NAMES = [
    "redirect_sfgated",
    "bad_redirect_sfgated",
]

TRIGRAM_DIRECTION_FEATURE_NAMES = [
    *_TRIGRAM_LEVEL_NAMES,
    *_TRIGRAM_GATED_NAMES,
    *(f"bg1_{n}" for n in (*_BIGRAM_PLACEMENT_NAMES, *_BIGRAM_DIRECTION_NAMES)),
    *(f"bg2_{n}" for n in (*_BIGRAM_PLACEMENT_NAMES, *_BIGRAM_DIRECTION_NAMES)),
    "wpm",
]
