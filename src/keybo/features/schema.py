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

# --- the INTERPRETABILITY frame (INTERPFRAME-1, opt-in, a REPLACEMENT basis not an addition) ---
#
# The first frame in this file that is NARROWER than the served one — 10 columns, not 20 — and the
# first whose objective is not accuracy. Every other block above widens the frame hoping the model
# learns more; this one re-expresses the SAME geometry in a basis chosen so that a per-feature SHAP
# number MEANS what its name says. CLOSING-2 measured nine widening arms to NULL on accuracy, so
# accuracy-neutral-to-slightly-worse is the REGISTERED expectation here, not a hope
# (``agent-artifacts/interpframe/INTERPFRAME-preregistration.md`` §0).
#
# The five measured failure modes of the served frame that the columns below answer:
#
# 1. COUPLED-COLUMN CREDIT SPLITTING. ``dx``/``dy``/``distance`` are mutually functionally
#    dependent and the ROW one-hot is exactly collinear (``bottom+home+top == 1`` on every letter
#    key), so TreeSHAP's split of credit between them is not unique. The symptom SHAPDIFF-1
#    measured: ``wpm`` is a CONSTANT column at a fixed scoring WPM yet carried -0.0922 ms/char,
#    9.2% of the T2 gap — interaction credit booked as a main effect.
# 2. ONE MECHANISM SHATTERED ACROSS COLUMNS THAT THEN FIGHT. "this stroke leaves the home row" is
#    spread over three one-hot columns; ``scissor`` (dy==2 on adjacent fingers) is a THRESHOLDED
#    slice of a graded row span whose sub-threshold remainder lands in ``dy``.
# 3. NON-MECHANISTIC FEATURES. ``distance`` is the clearest: the model prices LONG travel CHEAPER,
#    because long travel proxies for CROSS-HAND, which is fast. So "distance explains X" has the
#    WRONG SIGN as a physical story. The fix is not a constraint on ``distance`` — it is to
#    CONDITION the column on same-hand, after which farther really is slower.
# 4. NAME COLLISIONS. ``lateral`` the COLUMN (a key in an off-home stretch column, |x| in {1,6})
#    is not ``lat-span`` the GAUGE (graded hand stretch beyond rest). ``inwards``/``outwards`` are
#    SWAP-INVARIANT — 0 of 870 ordered pairs change under reversal (pinned in
#    ``tests/features/test_roll_direction_order.py``) — so those two names LIE about being
#    directions of travel.
# 5. AGGREGATION HIDING SIGN FLIPS. Trigram-level (the ``redir`` gauge inverts on disaggregation),
#    so a BIGRAM frame cannot address it. Registered as NOT fixed rather than claimed fixed.
#
# ⚠ Kept OUT of every list above for the reason ``_BIGRAM_DIRECTION_NAMES`` is: this is a fourth
# model population with its own stamp, and ``_BIGRAM_PLACEMENT_NAMES`` is the SHARED PREFIX of the
# version-locked served frames. Nothing here is a subset of anything there — the columns are new
# definitions, so no shipped artifact's column can change meaning.
_BIGRAM_INTERP_NAMES = [
    # ORDINAL, replacing the nested {same_hand ⊃ adjacent ⊃ scissor} + same_finger ladder with the
    # one axis those four columns were jointly encoding. 0 alternate / 1 same hand, two fingers /
    # 2 same finger — ``BigramClass``'s own documented speed ordering, so +1 IS the mechanism.
    "hand_conflict",
    # GRADED vertical span (0/1/2), subsuming ``scissor`` (dy==2 adjacent), keycraft's HSB (dy==1)
    # and ``row_skip`` (dy==2 any finger) without a threshold — so it has no sub-threshold blind
    # spot, and therefore no LAYOUT-DEPENDENT one (the LSBWIDEN-1 argument).
    "row_span",
    # THE GAUGE'S OWN PREDICATE, unchanged: ``classify.lateral_span``. The feature named
    # ``lateral_span`` IS the gauge named ``lat-span``, which is failure mode 4 fixed by
    # construction rather than by convention.
    "lateral_span",
    # ``distance`` CONDITIONED on same-hand. This is the non-mechanistic fix: unconditioned
    # distance is a cross-hand PROXY (cross-hand pairs are both far apart and fast), which is why
    # the served column's learned sign contradicts its physical story.
    "same_hand_travel",
    # These two are a 45° ROTATION of the two keys' home-row deviations: the sum and difference of
    # two equal-variance quantities are orthogonal by construction, which is what "orthogonalized
    # basis" means operationally. ``row_load`` is the magnitude ("how far off home was this
    # stroke"), ``row_arrival`` the SIGNED order ("did it end further off home than it started") —
    # and the difference is what keeps stroke ORDER, which a bare sum destroys.
    "row_load",
    "row_arrival",
    # The up/down ASYMMETRY as its own signed axis. Not folded into ``row_load`` because bottom is
    # measured costlier than top (158.670 ms vs 137.0/140.2, SHAPDIFF-TCOND) and a magnitude-only
    # column cannot say which direction was expensive.
    "bottom_bias",
    # ORDINAL finger weakness, replacing the 5-column FINGER block — which is not even a one-hot:
    # ``lateral`` co-fires with ``index`` (|x|==1) and with ``pinky`` (|x|==6), so five columns
    # encode four fingers plus an overlapping flag.
    "finger_load",
    # What the served ``lateral`` column actually MEASURES, under a name that says so: a key in a
    # finger's off-home stretch column. Failure mode 4, the cheap pure win.
    "off_home_column",
    # The HONEST direction of travel: +1 toward the index, -1 toward the pinky, 0 for a stroke with
    # no horizontal direction. ``is_inwards_ordered`` is the exact complement of
    # ``is_outwards_ordered`` on the roll-eligible set, so ONE signed column carries what the two
    # swap-invariant served columns could not carry at all.
    "roll_inward",
]

#: The interpretability frame. ``wpm`` is DELIBERATELY ABSENT and the absence IS the point: it is a
#: CONSTANT column at a fixed scoring WPM, so every ms/char it is credited with is interaction
#: credit booked as a main effect (-0.0922, 9.2% of the T2 gap, measured). This is the only frame
#: in this module without a ``wpm`` column, so ``tests/features/test_schema.py``'s
#: "wpm is last" convention test must not be pointed at it.
#: ⚠ CONSEQUENCE, stated rather than hidden: a model on this frame cannot span a WPM range — it is
#: valid at the ONE scoring WPM it was trained for. That is what the attribution lens uses, and it
#: is a large part of why this is a PROOF-OF-CONCEPT frame for INTERPRETATION rather than a
#: candidate to replace the served frame.
BIGRAM_INTERP_FEATURE_NAMES = [*_BIGRAM_INTERP_NAMES]

#: Monotone constraint per column of :data:`BIGRAM_INTERP_FEATURE_NAMES`, in the SAME ORDER — the
#: tuple XGBoost's ``monotone_constraints`` wants. Every column is constrained, which is what makes
#: ``MONOFRAC`` (the share of attribution mass on monotone columns) reach 1.0 BY CONSTRUCTION;
#: whether each constraint is HONORED by the trained booster is a separate, measured question
#: (INTERPFRAME-1 §5 — present is not effective, and ADJ-2 PINKY-MONO measured a constrained
#: column learning EXACTLY ZERO magnitude on this repo's data).
#:
#: Each sign is the MECHANISM its name asserts, so a violated constraint would mean the NAME is
#: wrong. All +1 (rising => slower) except ``roll_inward``, where travelling toward the index is
#: the community's canonical easy direction.
BIGRAM_INTERP_MONOTONE = (1, 1, 1, 1, 1, 1, 1, 1, 1, -1)

#: Stamped by anything trained on the interpretability frame — the FOURTH population. Must equal
#: none of the other three, or ``keybo.models.base``'s load guard could not tell them apart.
FEATURE_VERSION_INTERP = f"{FEATURE_VERSION}+interp.1"
