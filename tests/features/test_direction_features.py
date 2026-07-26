"""The direction-of-travel feature surface (v2, 2026-07-26.1) — additive and order-dependent.

Three properties are load-bearing and each has a test here:

1. **v1 parity.** ``direction=False`` reproduces the 20-column vector BIT-IDENTICALLY. The
   shipped models in ``data/models/`` were fit on it, so any drift is silent train/serve
   skew. This is the positive control for the whole change.
2. **Order-dependence.** Every v2 column is order-dependent somewhere, and the v2 row as a
   whole distinguishes reversed bigrams that v1 cannot — which is the point. THEORY-1
   (ledger f4d126e) established the v1 blindness exhaustively; these tests are the fix's
   counterpart, and the v1 half doubles as a regression guard on that result.
3. **New information, not a restatement.** A column that is order-dependent but fully
   determined by the v1 vector adds no channel. ``signed_dy`` and an origin-row one-hot both
   fail that way (the stagger-adjusted ``dx`` already leaks the origin row), which is why
   neither is in the schema. The test pins the mechanism so a future "obvious" addition of
   them gets caught.
"""

import numpy as np
import pytest

from keybo.features import (
    BIGRAM_DIRECTION_NAMES,
    BIGRAM_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES_DIRECTION,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES_DIRECTION,
    bigram_features,
    bigram_features_from_positions,
    bigram_model_row,
    trigram_features_from_positions,
    trigram_model_row,
)
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31
from keybo.layout import Layout

LAYOUT = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", ROW_STAGGERED_30)

#: All ordered DISTINCT position pairs on the 30-key board: 30*29 = 870. (The "900" frame
#: quoted elsewhere is 30*30, i.e. including a == b; those 30 diagonal cells are trivially
#: swap-identical, so both counts describe the same result.)
PAIRS_30 = [(a, b) for a in ROW_STAGGERED_30.slots for b in ROW_STAGGERED_30.slots if a != b]
PAIRS_31 = [(a, b) for a in ROW_STAGGERED_31.slots for b in ROW_STAGGERED_31.slots if a != b]

#: Landing-key one-hots: the ONLY channel through which order entered the v1 vector.
_LANDING = ("bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral")


def test_pair_count_is_870_distinct_ordered():
    assert len(PAIRS_30) == 870
    assert len(PAIRS_31) == 930


# --- 1. v1 parity: the shipped surface must not move -------------------------------------


def test_v1_vector_is_unchanged_by_default():
    """direction=False is the shipped 20-column vector, checked against the FROZEN GOLDEN
    matrix — not against v2 computed by this same code.

    This is the load-bearing positive control, and it has to come from an independent
    evidence path: comparing v1 to v2 out of one code path would pass even if a shared v1
    feature drifted (verified — perturbing ``dy`` by 0.001 leaves every self-referential
    assertion green while this one fails). ``tests/features/test_k31_geometry.py`` owns the
    same fixture for the K31 gate; this asserts it for the direction change too, on the
    exact grid and wpm the golden file was frozen at.
    """
    import os

    assert len(BIGRAM_FEATURE_NAMES) == 20
    assert BIGRAM_FEATURE_NAMES[-1] == "wpm"

    golden = np.load(os.path.join(os.path.dirname(__file__), "golden_k30_features.npz"))
    geom = ROW_STAGGERED_30
    pos = [*geom.slots, geom.space_position]
    bi = np.array(
        [[bigram_features_from_positions(geom, (a, b), wpm=87.0) for b in pos] for a in pos]
    )
    tri = np.array(
        [
            [trigram_features_from_positions(geom, (a, b, pos[7]), wpm=87.0) for b in pos]
            for a in pos
        ]
    )
    assert np.array_equal(bi, golden["bigram"])
    assert np.array_equal(tri, golden["trigram_slice"])


def test_v2_is_a_strict_prefix_extension_of_v1():
    """v2 = v1's placement block, then direction, then wpm. The v1 values must be reused
    unchanged, so a v1 model's learned splits stay meaningful in interpretation."""
    assert BIGRAM_FEATURE_NAMES_DIRECTION[:19] == BIGRAM_FEATURE_NAMES[:19]
    assert BIGRAM_FEATURE_NAMES_DIRECTION[19:-1] == BIGRAM_DIRECTION_NAMES
    assert BIGRAM_FEATURE_NAMES_DIRECTION[-1] == "wpm"
    for a, b in PAIRS_30:
        v1 = bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0)
        v2 = bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0, direction=True)
        assert np.array_equal(v2[:19], v1[:19]), (a, b)
        assert v2[-1] == v1[-1]


def test_v2_trigram_is_a_superset_carrying_both_bigrams_direction():
    assert len(TRIGRAM_FEATURE_NAMES_DIRECTION) == len(TRIGRAM_FEATURE_NAMES) + 2 * len(
        BIGRAM_DIRECTION_NAMES
    )
    for name in TRIGRAM_FEATURE_NAMES:
        assert name in TRIGRAM_FEATURE_NAMES_DIRECTION
    for name in BIGRAM_DIRECTION_NAMES:
        assert f"bg1_{name}" in TRIGRAM_FEATURE_NAMES_DIRECTION
        assert f"bg2_{name}" in TRIGRAM_FEATURE_NAMES_DIRECTION


def test_trigram_v1_values_survive_in_v2():
    tri = trigram_features_from_positions(ROW_STAGGERED_30, ((-4, 3), (-3, 2), (2, 1)), wpm=90.0)
    tri_d = trigram_features_from_positions(
        ROW_STAGGERED_30, ((-4, 3), (-3, 2), (2, 1)), wpm=90.0, direction=True
    )
    by_name = dict(zip(TRIGRAM_FEATURE_NAMES_DIRECTION, tri_d, strict=True))
    for name, value in zip(TRIGRAM_FEATURE_NAMES, tri, strict=True):
        assert by_name[name] == value, name


def test_row_keys_match_the_selected_schema_in_order():
    assert list(bigram_model_row(LAYOUT, "th", wpm=90).keys()) == BIGRAM_FEATURE_NAMES
    assert (
        list(bigram_model_row(LAYOUT, "th", wpm=90, direction=True).keys())
        == BIGRAM_FEATURE_NAMES_DIRECTION
    )
    assert list(trigram_model_row(LAYOUT, "the", wpm=90).keys()) == TRIGRAM_FEATURE_NAMES
    assert (
        list(trigram_model_row(LAYOUT, "the", wpm=90, direction=True).keys())
        == TRIGRAM_FEATURE_NAMES_DIRECTION
    )


def test_layout_and_position_paths_agree_on_both_surfaces():
    """The train/serve parity guarantee, extended to v2."""
    for direction in (False, True):
        from_layout = bigram_features(LAYOUT, "th", wpm=90.0, direction=direction)
        from_positions = bigram_features_from_positions(
            LAYOUT.geometry, (LAYOUT.pos("t"), LAYOUT.pos("h")), wpm=90.0, direction=direction
        )
        assert np.array_equal(from_layout, from_positions)


def test_version_stamps_are_distinct_and_ordered():
    """A v2 model must not be loadable as v1: the stamps differ, and TypingModel.load
    compares them. The date ordering also keeps the existing `> "2026-07-03.1"` guard true."""
    assert FEATURE_VERSION_DIRECTION != FEATURE_VERSION
    assert FEATURE_VERSION_DIRECTION > FEATURE_VERSION


# --- 2. the v1 blindness (THEORY-1 regression guard) and the v2 fix ----------------------


@pytest.mark.parametrize(
    ("geometry", "pairs"), [(ROW_STAGGERED_30, PAIRS_30), (ROW_STAGGERED_31, PAIRS_31)]
)
def test_v1_non_landing_features_are_exactly_swap_invariant(geometry, pairs):
    """THEORY-1's exhaustive result, as a test: max abs diff on any non-landing v1 feature
    between features(a,b) and features(b,a) is EXACTLY 0. This is the fact being fixed, and
    it must keep holding for the v1 surface (the shipped models depend on it)."""
    from keybo.features.ngram import _placement_row_from_positions

    worst = 0.0
    for a, b in pairs:
        ab = _placement_row_from_positions(geometry, a, b)
        ba = _placement_row_from_positions(geometry, b, a)
        for name, value in ab.items():
            if name in _LANDING:
                continue
            worst = max(worst, abs(value - ba[name]))
    assert worst == 0.0


def test_the_named_direction_features_of_v1_are_the_trap():
    """angle/inwards/outwards read as directional, take an ordered pair, and are provably
    swap-invariant — the reason the blindness went unnoticed for a whole campaign."""
    for a, b in PAIRS_30:
        assert C.rotation_angle(ROW_STAGGERED_30, a, b) == C.rotation_angle(ROW_STAGGERED_30, b, a)
        assert C.is_inwards(ROW_STAGGERED_30, a, b) == C.is_inwards(ROW_STAGGERED_30, b, a)
        assert C.is_outwards(ROW_STAGGERED_30, a, b) == C.is_outwards(ROW_STAGGERED_30, b, a)


@pytest.mark.parametrize("name", BIGRAM_DIRECTION_NAMES)
def test_every_direction_column_is_order_dependent_somewhere(name):
    """A swap-invariant column would be worthless as a direction channel, so each one must
    actually differ under reversal on at least one pair."""
    idx = BIGRAM_FEATURE_NAMES_DIRECTION.index(name)
    differing = 0
    for a, b in PAIRS_30:
        ab = bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0, direction=True)
        ba = bigram_features_from_positions(ROW_STAGGERED_30, (b, a), wpm=90.0, direction=True)
        if ab[idx] != ba[idx]:
            differing += 1
    assert differing > 0, f"{name} is swap-invariant — it cannot express direction"


def test_direction_swap_counts_are_pinned():
    """The exhaustive per-column swap-difference census (of 870 ordered distinct pairs).

    Pinned because these counts BOUND what a direction feature can do: a column can only
    re-rank the pairs it distinguishes. signed_dx separates the most (all 870, since every
    pair has some horizontal displacement under stagger); the roll-shaped columns separate
    324 (same hand, two fingers, distinct columns); the origin-finger one-hots 288-432.
    """
    expected = {
        "signed_dx": 870,
        "dir_dx_inward": 360,
        "dir_angle": 324,
        "dir_inwards": 324,
        "dir_outwards": 324,
        "o_pinky": 288,
        "o_ring": 288,
        "o_middle": 288,
        "o_index": 432,
    }
    assert set(expected) == set(BIGRAM_DIRECTION_NAMES)
    counts = dict.fromkeys(expected, 0)
    for a, b in PAIRS_30:
        ab = dict(
            zip(
                BIGRAM_FEATURE_NAMES_DIRECTION,
                bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0, direction=True),
                strict=True,
            )
        )
        ba = dict(
            zip(
                BIGRAM_FEATURE_NAMES_DIRECTION,
                bigram_features_from_positions(ROW_STAGGERED_30, (b, a), wpm=90.0, direction=True),
                strict=True,
            )
        )
        for name in expected:
            if ab[name] != ba[name]:
                counts[name] += 1
    assert counts == expected


def test_v2_separates_reversed_pairs_that_v1_cannot():
    """The headline: under v1 exactly 30 ordered pairs (15 unordered, all cross-hand mirror
    pairs) have a featurewise-identical reverse. v2 must leave FEWER — that reduction is the
    new channel. It cannot reach zero: a cross-hand mirror pair like (-5,2) <-> (5,2) is
    genuinely symmetric under every hand-relative feature, and signed_dx separates it only
    because it is hand-agnostic."""

    def n_identical_reverses(direction):
        n = 0
        for a, b in PAIRS_30:
            ab = bigram_features_from_positions(
                ROW_STAGGERED_30, (a, b), wpm=90.0, direction=direction
            )
            ba = bigram_features_from_positions(
                ROW_STAGGERED_30, (b, a), wpm=90.0, direction=direction
            )
            if np.array_equal(ab, ba):
                n += 1
        return n

    assert n_identical_reverses(False) == 30
    assert n_identical_reverses(True) == 0


# --- 3. rejected candidates: order-dependent but already determined ----------------------


def test_origin_row_and_signed_dy_are_already_determined_by_v1():
    """Why no origin-row one-hot and no signed_dy: the stagger-adjusted dx leaks the origin
    row, so both are functions of features the model already has. Adding them would be a
    null column wearing a direction-shaped name.

    Method: group the 870 pairs by their exact v1 vector; a quantity that is constant inside
    every group is determined by v1.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for a, b in PAIRS_30:
        key = tuple(bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0))
        groups[key].append((a, b))

    row_varies = [g for g in groups.values() if len({a[1] for a, _ in g}) > 1]
    dy_varies = [g for g in groups.values() if len({b[1] - a[1] for a, b in g}) > 1]
    assert row_varies == [], "origin row is NOT determined by v1 — revisit the schema note"
    assert dy_varies == [], "signed_dy is NOT determined by v1 — revisit the schema note"

    # And the mechanism, concretely: same landing key, same |row span|, same distance —
    # different stagger-adjusted dx, so dx alone identifies the origin row.
    b = (5, 2)
    rows = {
        a: dict(
            zip(
                BIGRAM_FEATURE_NAMES,
                bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0),
                strict=True,
            )
        )
        for a in [(-5, 1), (-5, 3)]
    }
    lo, hi = rows[(-5, 1)], rows[(-5, 3)]
    assert lo["dy"] == hi["dy"] and lo["distance"] == hi["distance"]
    assert lo["dx"] != hi["dx"]


def test_origin_finger_is_not_determined_by_v1():
    """The counterpart: the origin FINGER does vary inside v1 collision groups, which is why
    it earns a column while the origin row does not."""
    from collections import defaultdict

    groups = defaultdict(list)
    for a, b in PAIRS_30:
        key = tuple(bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0))
        groups[key].append((a, b))
    varying = [g for g in groups.values() if len({ROW_STAGGERED_30.finger(a[0]) for a, _ in g}) > 1]
    assert varying, "origin finger looks determined by v1 — the column would be inert"


# --- the honestly-named roll predicates -------------------------------------------------


def test_directed_rolls_mean_what_their_names_say():
    """dir_inwards means the MOTION ran toward the index finger — unlike is_inwards, which
    is an orientation of the unordered pair."""
    g = ROW_STAGGERED_30
    outer, inner = (-4, 2), (-2, 2)  # left ring -> left index, same row: a true inroll
    assert C.is_directed_inwards(g, outer, inner)
    assert not C.is_directed_inwards(g, inner, outer)
    assert C.is_directed_outwards(g, inner, outer)
    assert not C.is_directed_outwards(g, outer, inner)
    # ... and the swap-invariant pair says nothing about which was typed first:
    assert C.is_inwards(g, outer, inner) == C.is_inwards(g, inner, outer)


def test_directed_angle_flips_by_180_under_reversal():
    g = ROW_STAGGERED_30
    a, b = (-4, 3), (-2, 1)
    fwd, rev = C.directed_angle(g, a, b), C.directed_angle(g, b, a)
    assert fwd != rev
    assert abs(abs(fwd - rev) - 180.0) < 1e-9


def test_directed_angle_is_undefined_exactly_where_rotation_angle_is():
    """Both return 0.0 on the pairs where a roll is undefined (cross-hand, same finger, same
    column), so the new column introduces no case the old one did not already have."""
    g = ROW_STAGGERED_30
    for a, b in PAIRS_30:
        defined = C.same_hand(g, a, b) and not C.same_finger(g, a, b) and abs(a[0]) != abs(b[0])
        if not defined:
            assert C.directed_angle(g, a, b) == 0.0
            assert C.rotation_angle(g, a, b) == 0.0


def test_directed_angle_strictly_refines_rotation_angle():
    """directed_angle is non-zero on 270 of 870 pairs where rotation_angle manages 216.

    The extra 54 are the FLAT (dy == 0) rolls: rotation_angle measures outer->inner, which on
    a flat pair is always atan2(0, +x) = 0, so it collapses flat-inward and flat-outward onto
    the same value. Measuring a->b separates them — exactly the inroll/outroll distinction
    the community argues about, and precisely what v1 could not state.

    SIGN CONVENTION, derived from reference motions rather than assumed (campaign rule):
    directed_angle inherits rotation_angle's hand frame, in which +x points OUTWARD (toward
    the pinky). So a flat outward roll is 0 degrees and a flat inward roll is 180, on BOTH
    hands. Verified against left/right ring<->pinky above, and cos(directed_angle) agrees in
    sign with ``is_directed_outwards`` on all 324 roll pairs.
    """
    g = ROW_STAGGERED_30
    nz_rot = sum(1 for a, b in PAIRS_30 if C.rotation_angle(g, a, b) != 0.0)
    nz_dir = sum(1 for a, b in PAIRS_30 if C.directed_angle(g, a, b) != 0.0)
    assert (nz_rot, nz_dir) == (216, 270)
    # every pair rotation_angle can see, directed_angle can see too
    for a, b in PAIRS_30:
        if C.rotation_angle(g, a, b) != 0.0:
            assert C.directed_angle(g, a, b) != 0.0
    # the concrete flat-roll case, on both hands
    for ring, pinky in [((-4, 2), (-5, 2)), ((4, 2), (5, 2))]:
        assert C.rotation_angle(g, ring, pinky) == 0.0
        assert C.rotation_angle(g, pinky, ring) == 0.0
        assert C.directed_angle(g, ring, pinky) == 0.0  # outward
        assert C.directed_angle(g, pinky, ring) == 180.0  # inward


def test_directed_angle_sign_agrees_with_the_directed_roll_predicates():
    """The angle and the boolean columns must tell the same story: +x is outward, so
    cos(directed_angle) > 0 exactly when the motion ran outward."""
    from math import cos, radians

    g = ROW_STAGGERED_30
    checked = 0
    for a, b in PAIRS_30:
        if not (C.same_hand(g, a, b) and not C.same_finger(g, a, b) and abs(a[0]) != abs(b[0])):
            continue
        checked += 1
        assert (cos(radians(C.directed_angle(g, a, b))) > 0) == C.is_directed_outwards(g, a, b)
    assert checked == 324


def test_signed_dx_is_the_signed_version_of_dx():
    g = ROW_STAGGERED_30
    for a, b in PAIRS_30:
        assert abs(C.signed_dx(g, a, b)) == pytest.approx(g.stagger_adjusted_dx(a, b))
        assert C.signed_dx(g, a, b) == pytest.approx(-C.signed_dx(g, b, a))


def test_dir_dx_inward_is_zero_across_hands_and_signed_within():
    g = ROW_STAGGERED_30
    assert C.dir_dx_inward(g, (-4, 2), (4, 2)) == 0.0  # cross-hand: undefined
    assert C.dir_dx_inward(g, (-4, 2), (-2, 2)) > 0  # left ring -> left index: inward
    assert C.dir_dx_inward(g, (4, 2), (2, 2)) > 0  # right ring -> right index: inward too
    assert C.dir_dx_inward(g, (-2, 2), (-4, 2)) < 0  # outward


def test_origin_finger_onehot_is_exclusive_and_complete():
    """Exactly one origin-finger column fires per pair (every letter column maps to a
    finger), mirroring the landing-key block's convention."""
    cols = ["o_pinky", "o_ring", "o_middle", "o_index"]
    idx = [BIGRAM_FEATURE_NAMES_DIRECTION.index(c) for c in cols]
    for a, b in PAIRS_30:
        v = bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0, direction=True)
        assert sum(v[i] for i in idx) == 1.0, (a, b)


# --- the same-width PLACEBO frame (attribution control) ----------------------------------


def test_placebo_frame_is_the_same_width_as_the_direction_frame():
    """TOOLING-TRAPS #17: a nested-frame attribution needs a same-SIZE placebo, or the
    v1->v2 delta mixes 'direction was added' with 'the frame grew'."""
    from keybo.features import BIGRAM_FEATURE_NAMES_PLACEBO, BIGRAM_PLACEBO_NAMES

    assert len(BIGRAM_PLACEBO_NAMES) == len(BIGRAM_DIRECTION_NAMES)
    assert len(BIGRAM_FEATURE_NAMES_PLACEBO) == len(BIGRAM_FEATURE_NAMES_DIRECTION)
    assert BIGRAM_FEATURE_NAMES_PLACEBO[:19] == BIGRAM_FEATURE_NAMES[:19]
    # disjoint column names, so no accidental sharing
    assert not set(BIGRAM_PLACEBO_NAMES) & set(BIGRAM_DIRECTION_NAMES)


def test_placebo_columns_carry_no_information_v1_lacks():
    """The defining property: every placebo column is constant inside every v1 collision
    group, i.e. it is a deterministic function of the v1 vector. If this ever fails, the
    placebo is smuggling in real information and the attribution is void."""
    from collections import defaultdict

    from keybo.features import BIGRAM_FEATURE_NAMES_PLACEBO, BIGRAM_PLACEBO_NAMES

    groups = defaultdict(list)
    for a, b in PAIRS_30:
        key = tuple(bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0))
        groups[key].append((a, b))

    idx = {n: BIGRAM_FEATURE_NAMES_PLACEBO.index(n) for n in BIGRAM_PLACEBO_NAMES}
    for pairs in groups.values():
        vecs = [
            bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0, placebo=True)
            for a, b in pairs
        ]
        for name, i in idx.items():
            assert len({v[i] for v in vecs}) == 1, f"{name} varies within a v1 group"


def test_placebo_cannot_separate_any_reversed_pair_v1_cannot():
    """Corollary: the placebo leaves the same 30 featurewise-identical reverses v1 had, while
    v2 leaves 0. That contrast IS the direction channel."""

    def n_identical(**kw):
        return sum(
            1
            for a, b in PAIRS_30
            if np.array_equal(
                bigram_features_from_positions(ROW_STAGGERED_30, (a, b), wpm=90.0, **kw),
                bigram_features_from_positions(ROW_STAGGERED_30, (b, a), wpm=90.0, **kw),
            )
        )

    assert n_identical() == 30
    assert n_identical(placebo=True) == 30
    assert n_identical(direction=True) == 0


def test_direction_and_placebo_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        bigram_features_from_positions(
            ROW_STAGGERED_30, ((-4, 2), (-2, 2)), wpm=90.0, direction=True, placebo=True
        )


def test_placebo_version_stamp_is_distinct():
    """A placebo model must never be servable: distinct stamp, so load() refuses it."""
    from keybo.features import FEATURE_VERSION_PLACEBO

    assert FEATURE_VERSION_PLACEBO not in (FEATURE_VERSION, FEATURE_VERSION_DIRECTION)
