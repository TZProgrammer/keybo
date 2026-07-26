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

DIRECTION-OF-TRAVEL (2026-07-26.1, additive): the shipped 20-column vector cannot express
which key you came FROM versus which you went TO. Exhaustively verified (THEORY-1, ledger
f4d126e): over all 870 ordered distinct position pairs the max absolute difference between
``features(a,b)`` and ``features(b,a)`` on any NON-landing feature is 0.000e+00 — direction
enters only through the landing key's own row/finger/lateral one-hots. ``angle``,
``inwards`` and ``outwards`` are the trap: they read as directional, are called with an
ordered pair, and are provably swap-invariant (all three are defined outer-key-to-inner-key,
which is an unordered notion).

:data:`BIGRAM_DIRECTION_NAMES` is the OPT-IN fix. It is strictly ADDITIVE: the v1 lists are
untouched, so every shipped artifact keeps loading and scoring bit-identically, and the
direction columns exist only for a model trained with ``direction=True`` (which stamps
:data:`FEATURE_VERSION_DIRECTION` instead). Two things measured before adding a column,
both of which a swap test alone would have missed:

- **Swap-dependence is necessary but NOT sufficient.** Group the 870 pairs by their exact
  v1 vector; a candidate that is constant inside every collision group is a deterministic
  function of features the model already has. ``signed_dy`` and an origin-ROW one-hot both
  fail this way — the origin row is ALREADY recoverable, because ``dx`` is stagger-adjusted
  and the per-row offsets {bottom:+0.5, home:0.0, top:-0.25} differ, so ``dx`` leaks a's
  row (a=(-5,1)->b=(5,2) gives dx=9.50, a=(-5,3)->b=(5,2) gives dx=10.25 at identical dy
  and distance). Neither is included; adding them would have been a null column dressed as
  a direction channel.
- **The genuinely missing quantity is the SIGN of travel, and it is a small channel.** Only
  30 of 870 ordered pairs (15 unordered) have a featurewise-identical reverse under v1, and
  every one is a cross-hand mirror pair. So a direction feature can only re-rank the pairs
  it can newly distinguish — which bounds the achievable fit gain a priori.
"""

#: v1 — the 20-column vector every shipped artifact in data/models/ was fit on. UNCHANGED.
FEATURE_VERSION = "2026-07-05.3"

#: v2 — v1 plus :data:`BIGRAM_DIRECTION_NAMES`. A real version bump: a v2 model's feature
#: matrix is wider, so serving a v1 artifact against it (or the reverse) is train/serve skew
#: and ``TypingModel.load`` refuses it. Kept as a SEPARATE constant rather than a mutated
#: ``FEATURE_VERSION`` precisely so the v1 artifacts stay valid and loadable.
FEATURE_VERSION_DIRECTION = "2026-07-26.1"

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

# --- direction of travel (v2, opt-in) ---------------------------------------------------
# The origin-dependent columns. Every one is order-DEPENDENT *and* carries information the
# v1 vector does not already determine (both verified exhaustively; see module docstring).
# Deliberately excluded: signed_dy and an origin-row one-hot (already determined via the
# stagger-adjusted dx), and o_lateral (likewise determined).
#
# Three encodings of the same underlying sign are offered rather than one, because they
# differ in which pairs they can separate and the trees may use them differently: the raw
# signed column displacement (finest, separates 270 of 870 pairs), the hand-relative
# inward/outward column step (coarser, 24 pairs), and the a->b roll angle whose sign is the
# direction (24 pairs). ``dir_inwards``/``dir_outwards`` are the honestly-named counterparts
# of the swap-invariant ``inwards``/``outwards``: they mean "this bigram MOVES toward the
# index finger", which is what the community's inroll/outroll argument is actually about.
_BIGRAM_DIRECTION_NAMES = [
    "signed_dx",  # (bx+off[by]) - (ax+off[ay]): signed, stagger-adjusted, hand-agnostic
    "dir_dx_inward",  # |ax| - |bx|, same hand only: >0 = travelling toward the index
    "dir_angle",  # a->b roll angle in degrees; sign IS the direction (0 when undefined)
    "dir_inwards",  # 1 when |bx| < |ax| (same hand, two fingers): a TRUE inward roll
    "dir_outwards",  # 1 when |bx| > |ax|: a TRUE outward roll
    # origin FINGER one-hot. The origin ROW is deliberately absent (already determined);
    # the origin finger is not — it varies inside 3 v1-collision groups (12 pairs).
    "o_pinky",
    "o_ring",
    "o_middle",
    "o_index",
]

#: The direction columns, exported for callers that need to name them (SHAP, ablations).
BIGRAM_DIRECTION_NAMES = list(_BIGRAM_DIRECTION_NAMES)

#: v2 bigram column order: v1's placement features, then the direction block, then wpm.
#: wpm stays LAST (tests and ``TypingModel.to_ms`` locate it by name, but the invariant is
#: asserted, and keeping it last means the v1 prefix is a contiguous slice).
BIGRAM_FEATURE_NAMES_DIRECTION = [
    *_BIGRAM_PLACEMENT_NAMES,
    *_BIGRAM_DIRECTION_NAMES,
    "wpm",
]

# --- the same-width PLACEBO frame (measurement only, never served) -----------------------
# Going v1 -> v2 changes TWO things at once: direction information is added AND the feature
# frame grows by 9 columns. Frame width alone moves an XGBoost fit (it changes the
# colsample_bytree draws, the split search, and the effective regularization), so a v1->v2
# delta cannot be attributed to direction (TOOLING-TRAPS #17: a nested-frame attribution
# needs a same-SIZE placebo).
#
# This frame adds exactly as many columns, built ONLY from quantities the v1 vector already
# determines: the origin ROW one-hot and signed_dy (both proven redundant — the
# stagger-adjusted dx leaks the origin row) and o_lateral, plus duplicate copies to reach 9.
# It therefore carries ZERO new information at the same width. The attributable direction
# effect is PLACEBO -> v2; v1 -> PLACEBO measures the width artifact by itself.
#
# Per TOOLING-TRAPS #17 the placebo's axis is deliberately NESTED in the real frame's
# information, which is the conservative choice: sharing structure with v2 UNDERSTATES v2's
# marginal effect, so an "inert" verdict survives the bias.
_BIGRAM_PLACEBO_NAMES = [
    "p_o_bottom",
    "p_o_home",
    "p_o_top",
    "p_signed_dy",
    "p_o_lateral",
    "p_o_bottom2",
    "p_o_home2",
    "p_o_top2",
    "p_signed_dy2",
]

#: The placebo columns, exported so a driver can assert the frame widths match.
BIGRAM_PLACEBO_NAMES = list(_BIGRAM_PLACEBO_NAMES)

#: Placebo bigram column order. Same width as :data:`BIGRAM_FEATURE_NAMES_DIRECTION`.
BIGRAM_FEATURE_NAMES_PLACEBO = [
    *_BIGRAM_PLACEMENT_NAMES,
    *_BIGRAM_PLACEBO_NAMES,
    "wpm",
]

#: Stamp for a placebo-trained model. It is a MEASUREMENT artifact: the placebo frame must
#: never be served, and a distinct stamp is what stops one being loaded by a serving path.
FEATURE_VERSION_PLACEBO = "2026-07-26.1-placebo"

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

#: v2 trigram order: both constituent bigrams carry the direction block too. Note the
#: trigram vector was ALREADY order-sensitive at the trigram level (``redirect`` compares
#: the two bigrams' directions of travel), which is why the community's TRIGRAM inroll /
#: outroll classes are genuinely directional while the BIGRAM ones are not — the
#: no-direction result is about the bigram feature vector only. Keep that boundary.
TRIGRAM_FEATURE_NAMES_DIRECTION = [
    *_TRIGRAM_LEVEL_NAMES,
    *(f"bg1_{n}" for n in (*_BIGRAM_PLACEMENT_NAMES, *_BIGRAM_DIRECTION_NAMES)),
    *(f"bg2_{n}" for n in (*_BIGRAM_PLACEMENT_NAMES, *_BIGRAM_DIRECTION_NAMES)),
    "wpm",
]

#: Placebo trigram order — same width as the direction trigram frame, no new information.
TRIGRAM_FEATURE_NAMES_PLACEBO = [
    *_TRIGRAM_LEVEL_NAMES,
    *(f"bg1_{n}" for n in (*_BIGRAM_PLACEMENT_NAMES, *_BIGRAM_PLACEBO_NAMES)),
    *(f"bg2_{n}" for n in (*_BIGRAM_PLACEMENT_NAMES, *_BIGRAM_PLACEBO_NAMES)),
    "wpm",
]
