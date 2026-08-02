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

# --- the KITCHEN-SINK block: external-project channels we provably lacked (opt-in, additive) ---
#
# The 12 definitions that survived the KITCHEN-SINK candidate audit
# (``agent-artifacts/kitchensink_audit.py``, registered in
# ``agent-artifacts/KITCHENSINK-preregistration.md``). Every one was checked over the FULL
# enumeration of ``ROW_STAGGERED_30`` — 870 ordered pairs, 24,360 ordered triples — for how often it
# fires, whether it can see stroke order, and how much of it OLS can already recover from the
# columns we serve. Two candidates were REJECTED by that measurement rather than added:
#
# * keycraft's ``RED-WEAK`` came back R2 = 1.0000 against the widened frame, and that is an
#   IDENTITY: it is bit-identical to ``bad_redirect_sfgated`` on all 24,360 triples. It is
#   REDIRGATE-1's gated bad-redirect under another name, and that column already measured NULL.
# * ``LSB-dist`` (the graded lateral stretch) came back R2 = 0.9768 on the smallest support of any
#   candidate, 32 of 870 pairs — the same verdict KEYCRAFT-1 applied to ``2RL-IN + 2RL-OUT``.
#
# Third stamp, third population. These names are kept OUT of both lists above for exactly the
# reason ``_BIGRAM_DIRECTION_NAMES`` is: ``_BIGRAM_PLACEMENT_NAMES`` and ``_TRIGRAM_LEVEL_NAMES``
# are the SHARED PREFIXES of the version-locked served frames, so a name added there would widen
# the served frame for all six shipped ``data/models/k31`` artifacts silently.
_BIGRAM_KITCHENSINK_NAMES = [
    # HSB — keycraft splits the ONE-row adjacent-finger reach from the two-row one; our
    # ``scissor`` gates on ``dy == 2``, so every half scissor is invisible to the served frame.
    "half_scissor",
    # A two-row jump on the same hand regardless of finger ADJACENCY (``scissor`` requires
    # adjacent fingers, so a pinky->index two-row jump is unflagged today).
    "row_skip",
    # POH — keycraft's pinky-off-home penalty. We serve a ``pinky`` one-hot and a ``home`` one-hot
    # but no INTERACTION, which a tree can only build by spending depth.
    "pinky_off_home",
    # keycraft's RED-WEAK gate applied at bigram level: both keys on the two least-dextrous
    # fingers. The served finger one-hot describes only the LANDING key, so pinky->ring and
    # index->ring are identical in that block.
    "weak_finger_pair",
    # The SIGNED finger-rank step: the graded form of ``inwards_ordered``, which is binary (as is
    # keycraft's own IN/OUT). Order-aware: 324 of 870 pairs change under reversal.
    "finger_step",
]

_TRIGRAM_KITCHENSINK_NAMES = [
    # 3RL — keycraft's monotonic one-hand roll. The served frame names ``redirect`` (the
    # NON-monotonic case) but has no column for the smoothest trigram class.
    "onehand",
    # 3RL-IN — that roll travelling toward the index. Order-aware (756 of 24,360 triples).
    "onehand_in",
    # RED-SFS / ALT-SFS — keycraft prices a redirect and an alternation whose OUTER two keys share
    # a finger apart from the plain forms. We have neither split.
    "red_sfs",
    "alt_sfs",
    # FSS / HSS / LSS — the scissor and lateral-stretch predicates across the SKIPGRAM. The served
    # trigram frame carries sg_dx/sg_dy/sg_distance/sg_same_finger but no sg_scissor and no sg_lsb.
    "sg_full_scissor",
    "sg_half_scissor",
    "sg_lsb",
]

#: The widened bigram frame plus the kitchen-sink block. ``wpm`` stays last (the convention
#: ``tests/features/test_schema.py`` pins).
BIGRAM_KITCHENSINK_FEATURE_NAMES = [
    *_BIGRAM_PLACEMENT_NAMES,
    *_BIGRAM_DIRECTION_NAMES,
    *_BIGRAM_KITCHENSINK_NAMES,
    "wpm",
]

#: The widened trigram frame plus the kitchen-sink block. The five bigram-level definitions enter
#: TWICE, once per constituent bigram, which is why 12 definitions produce 17 new trigram columns.
TRIGRAM_KITCHENSINK_FEATURE_NAMES = [
    *_TRIGRAM_LEVEL_NAMES,
    *_TRIGRAM_GATED_NAMES,
    *_TRIGRAM_KITCHENSINK_NAMES,
    *(
        f"bg1_{n}"
        for n in (
            *_BIGRAM_PLACEMENT_NAMES,
            *_BIGRAM_DIRECTION_NAMES,
            *_BIGRAM_KITCHENSINK_NAMES,
        )
    ),
    *(
        f"bg2_{n}"
        for n in (
            *_BIGRAM_PLACEMENT_NAMES,
            *_BIGRAM_DIRECTION_NAMES,
            *_BIGRAM_KITCHENSINK_NAMES,
        )
    ),
    "wpm",
]

#: Stamped by anything trained on the kitchen-sink frame. Must equal neither
#: :data:`FEATURE_VERSION` nor :data:`FEATURE_VERSION_DIRECTION`, or the load-time guard in
#: ``keybo.models.base`` could not tell the three model populations apart.
FEATURE_VERSION_KITCHENSINK = f"{FEATURE_VERSION}+kitchensink.1"

# --- the LATERAL-SPAN frames (LATSPAN-1, opt-in, additive) ---------------------------------
#
# A REPRESENTATION experiment, not a new-information one: ``classify.lateral_span`` is fully
# determined by the served bigram row (0 of 699 K30 / 759 K31 buckets ambiguous — verified),
# so these frames hand the model an assembled quantity it could in principle synthesize. The
# question is whether making the graded stretch EXPLICIT changes held-out transfer at the
# shipped depth/regularization, measured against the served frame as a clean single-variable
# A/B (LATSPAN-1 preregistration, subagent ``latspan``).
#
# Two designs, each its own stamp so the three lateral-span model populations (served, add,
# replace) can never be confused by the ``models/base.py`` load-time guard:
#
# * ADD — the served frame plus a ``lateral_span`` column. The graded stretch measured on
#   ALL 204 same-hand two-finger pairs, alongside the narrow ``lsb`` (index/middle-only, blind
#   to 172 of them) it does NOT remove. Bigram 20 -> 21 columns; trigram 46 -> 48 (the column
#   is bigram-level, so it enters once per constituent, like ``lsb`` -> ``bg1_lsb``/``bg2_lsb``).
# * REPLACE — the served frame with the narrow ``lsb`` column SWAPPED for ``lateral_span`` at
#   the same position (same width: bigram 20, trigram 46). A like-for-like swap of a
#   provably-blind predicate for a blind-spot-free graded one — the more interesting design if
#   representation, not information, is the lever.
#
# Kept OUT of the served lists and behind an explicit opt-in for the same reason
# ``_BIGRAM_DIRECTION_NAMES`` is: ``_BIGRAM_PLACEMENT_NAMES`` is the shared prefix of the
# version-locked served frames, so a name added there would silently widen the served frame for
# all six shipped ``data/models/k31`` artifacts.

#: The served placement block with a ``lateral_span`` column appended (ADD).
_BIGRAM_PLACEMENT_LATSPAN_ADD = [*_BIGRAM_PLACEMENT_NAMES, "lateral_span"]
#: The served placement block with ``lsb`` swapped for ``lateral_span`` IN PLACE (REPLACE).
_BIGRAM_PLACEMENT_LATSPAN_REPLACE = [
    "lateral_span" if n == "lsb" else n for n in _BIGRAM_PLACEMENT_NAMES
]

BIGRAM_LATSPAN_ADD_FEATURE_NAMES = [*_BIGRAM_PLACEMENT_LATSPAN_ADD, "wpm"]
BIGRAM_LATSPAN_REPLACE_FEATURE_NAMES = [*_BIGRAM_PLACEMENT_LATSPAN_REPLACE, "wpm"]

TRIGRAM_LATSPAN_ADD_FEATURE_NAMES = [
    *_TRIGRAM_LEVEL_NAMES,
    *(f"bg1_{n}" for n in _BIGRAM_PLACEMENT_LATSPAN_ADD),
    *(f"bg2_{n}" for n in _BIGRAM_PLACEMENT_LATSPAN_ADD),
    "wpm",
]
TRIGRAM_LATSPAN_REPLACE_FEATURE_NAMES = [
    *_TRIGRAM_LEVEL_NAMES,
    *(f"bg1_{n}" for n in _BIGRAM_PLACEMENT_LATSPAN_REPLACE),
    *(f"bg2_{n}" for n in _BIGRAM_PLACEMENT_LATSPAN_REPLACE),
    "wpm",
]

#: Stamped by anything trained on a lateral-span frame. Each must be distinct from every other
#: known stamp so the load-time guard can tell the populations apart.
FEATURE_VERSION_LATSPAN_ADD = f"{FEATURE_VERSION}+latspan-add.1"
FEATURE_VERSION_LATSPAN_REPLACE = f"{FEATURE_VERSION}+latspan-replace.1"
