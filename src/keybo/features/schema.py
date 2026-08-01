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

# --- the SAME-ROW ROLL block: keymeow's roll classes, made EXPLICIT (opt-in, additive) --------
#
# SR-ROLL-1. ``sr-roll`` is one of the 15 analyze GAUGES (``analysis/kmstats.py``, keymeow's base
# metric set) and one of BALL-1's largest gauge advantages over keybo-lsb (17.81 vs 12.69). No
# keybo frame has ever carried a column for its class. The audit that justifies these two columns
# is ``state/srroll/drivers/srroll_audit.py``; over the full 24,360-triple enumeration of
# ``ROW_STAGGERED_30`` it found:
#
# * ``sr_roll`` is bit-identical to NONE of the 46 served columns, and its overlap with every
#   existing trigram-class column is exactly ZERO — ``same_hand_trigram``, ``redirect``,
#   ``bad_redirect``, and even kitchen-sink's ``onehand``/``onehand_in``/``red_sfs``/``alt_sfs``.
#   That is structural, not incidental: keymeow's roll requires the OUTER keys on OPPOSITE hands
#   (``a.hand != c.hand``) while every one of those columns requires a single hand, and kmstats'
#   ``alt`` requires ``a.hand == c.hand``. So this class is unnamed in every frame we have.
# * It is nonetheless a DETERMINISTIC FUNCTION of the served frame (0 ambiguous groups over 23,250
#   distinct served rows), so it adds no INFORMATION — only explicitness. In served-column terms it
#   is a five-way conjunction whose first term is an XOR:
#       XOR(bg1_same_hand, bg2_same_hand) AND NOT bg1_same_finger AND NOT bg2_same_finger
#       AND bg1_dy == 0 AND bg2_dy == 0
#   OLS R2 against the served frame is 0.3321, and the shallowest single tree that fits it exactly
#   needs depth 11 — 5 once the outer-hand XOR is supplied, which identifies the XOR as the
#   obstruction rather than the same-row gate.
#
# ``roll`` accompanies ``sr_roll`` because sr_roll is a strict SUBSET of it (1,080 of 9,720 firing
# triples). Without the superset column the model cannot separate "flat roll" from "roll at all",
# so the increment attributable to SAME-ROW would not be identifiable — the same reason
# ``is_row_skip`` was added alongside ``is_scissor`` in the kitchen-sink block.
#
# Fourth stamp, fourth population. Kept OUT of every list above for the reason
# ``_BIGRAM_DIRECTION_NAMES`` is: ``_TRIGRAM_LEVEL_NAMES`` is the SHARED PREFIX of the
# version-locked served frames, so a name added there would silently widen the served frame for all
# six shipped ``data/models/k31`` artifacts.
_TRIGRAM_SRROLL_NAMES = [
    # keymeow's ``roll``: outer keys on opposite hands, no repeated finger across either step.
    "roll",
    # keymeow's ``sr-roll``: that roll with all three keys on ONE row.
    "sr_roll",
]

#: The served trigram frame plus the same-row-roll block. Built on the SERVED (narrow) frame, not
#: the widened one, because the A/B this exists for tests the two new columns against the incumbent
#: that is actually shipped — adding the direction/kitchen-sink columns too would confound the
#: single variable under test with two prior arms' worth of columns.
TRIGRAM_SRROLL_FEATURE_NAMES = [
    *_TRIGRAM_LEVEL_NAMES,
    *_TRIGRAM_SRROLL_NAMES,
    *(f"bg1_{n}" for n in _BIGRAM_PLACEMENT_NAMES),
    *(f"bg2_{n}" for n in _BIGRAM_PLACEMENT_NAMES),
    "wpm",
]

#: Stamped by anything trained on the same-row-roll frame. Must equal none of the three stamps
#: above, or ``keybo.models.base``'s load-time guard could not tell the populations apart.
FEATURE_VERSION_SRROLL = f"{FEATURE_VERSION}+srroll.1"
