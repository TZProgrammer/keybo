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

# --- the MIRROR-SYMMETRIC frame (MIRROR-1, 2026-08-02): same columns, same width -----------
#
# An EXPRESSIVITY-REMOVING frame, and the only one here that is not additive: it keeps all 20
# bigram column names but forces three of them (``dx``, ``angle``, ``lsb``) to mirror-invariant
# values, so a bigram and its left/right mirror image get an IDENTICAL row and the model
# provably cannot price them differently. The thesis it tests: "there should be no
# biomechanical reason for 'sd' to be any faster than 'lk'".
#
# ⚠ READ THIS BEFORE USING IT — the constraint is TRUE on only 330 of the 870 ordered pairs.
# ``ROW_STAGGERED_30.row_offsets = {1: 0.5, 2: 0.0, 3: -0.25}`` applies to BOTH hands
# identically, so the physical coordinate ``x + off(y)`` is not antisymmetric and x-negation is
# NOT an isometry of the board: it changes ``stagger_adjusted_dx`` on 540 of 900 ordered
# position pairs, and ALL 540 are CROSS-row (same-row pairs share the offset, so it cancels).
# An exhaustive search over vertical-axis reflections finds NONE that maps the board onto
# itself — the row-staggered board's mirror symmetry group is trivial. So on the 270 same-row
# plus 60 same-column pairs mirroring IS an exact board symmetry and the served frame already
# satisfies it (measured: the shipped seed-averaged T2 has exactly 0.0 ms mirror asymmetry on
# all 330 — ``sd``/``lk``, the motivating example, included); on the other 540 the difference is
# real geometry of a staggered board, and forcing it away imposes a FALSE constraint.
#
# A separate STAMP rather than a flag on the served frame, for the reason
# :data:`FEATURE_VERSION_DIRECTION` exists: the six models under ``data/models/k31`` carry
# :data:`FEATURE_VERSION`, and ``keybo.models.base`` hard-errors on a version MISMATCH but not
# on a column whose MEANING changed. Symmetrizing these three in place would leave all six
# loading fine while scoring a frame whose ``dx``/``angle``/``lsb`` no longer mean what they
# meant at training time — the exact train/serve skew the stamp prevents. Column NAMES and
# ORDER are deliberately unchanged, so no widening/narrowing confound can enter an A/B: the
# served and mirror frames are both 20 columns and differ only in three columns' VALUES.
FEATURE_VERSION_MIRROR = f"{FEATURE_VERSION}+mirror.1"

#: The BIGRAM-level columns the mirror frame symmetrizes. All three are mirror-variant purely
#: through the row stagger; ``lsb`` inherits it from ``stagger_adjusted_dx``, which it
#: thresholds at 1.5. NOTE ``dx`` is ALREADY non-negative (``stagger_adjusted_dx`` returns
#: ``abs(...)``), so the obvious recipe "replace dx with |dx|" is a NO-OP — the stagger, not a
#: sign, is what breaks the symmetry.
MIRROR_SYMMETRIZED_COLUMNS = ("dx", "angle", "lsb")

#: The TRIGRAM-LEVEL columns the mirror frame symmetrizes, on top of the two constituent
#: bigrams' :data:`MIRROR_SYMMETRIZED_COLUMNS`. Only the SKIPGRAM span is mirror-variant, for
#: the same stagger reason: ``sg_dx`` is ``stagger_adjusted_dx`` across keys 1 and 3.
#: ``redirect``/``bad_redirect``/``sg_same_finger``/``sg_dy``/``sg_distance`` are already
#: mirror-invariant (they read ``abs(x)``, a row difference, or a hand-independent finger map),
#: so symmetrizing the trigram frame means these plus ``bg1_``/``bg2_``-prefixed bigram columns.
MIRROR_SYMMETRIZED_TRIGRAM_COLUMNS = ("sg_dx",)
