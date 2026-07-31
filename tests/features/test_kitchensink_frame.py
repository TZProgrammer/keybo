"""The KITCHEN-SINK frame: additive, order-correct, and provably non-disruptive.

The load-bearing test in this file is :func:`test_narrow_and_widened_frames_are_byte_identical`.
Everything else describes the new columns; that one proves the two SHIPPED frames did not move,
which is the whole basis on which a third frame is allowed to exist at all (six models under
``data/models/k31/`` are stamped ``FEATURE_VERSION`` and three more carry the trigram served
frame; ``keybo.models.base`` errors on a version MISMATCH, not on a column whose MEANING changed,
so a silent redefinition would leave every model loading fine while scoring the wrong matrix).

The predicate tests pin the numbers the KITCHEN-SINK candidate audit measured
(``agent-artifacts/kitchensink_audit.py``, registered in
``agent-artifacts/KITCHENSINK-preregistration.md``) so a later edit cannot quietly change what a
column counts while the registered table still claims the old figure.
"""

from itertools import permutations

import numpy as np
import pytest

from keybo.features import (
    bigram_features_from_positions,
    bigram_model_row,
    trigram_features_from_positions,
    trigram_model_row,
)
from keybo.features import classify as C
from keybo.features.ngram import trigram_direction_row, trigram_kitchensink_row
from keybo.features.schema import (
    BIGRAM_DIRECTION_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES,
    BIGRAM_KITCHENSINK_FEATURE_NAMES,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    FEATURE_VERSION_KITCHENSINK,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_KITCHENSINK_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31
from keybo.layout import Layout

G = ROW_STAGGERED_30
LAYOUT = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", G)

PAIRS = list(permutations(G.slots, 2))
TRIPLES = list(permutations(G.slots, 3))


@pytest.fixture(scope="module")
def matrices():
    """The six frames over the FULL enumeration — built once, they are the scope of every claim."""
    return {
        "bn": np.vstack([bigram_features_from_positions(G, p, 90.0) for p in PAIRS]),
        "bw": np.vstack(
            [bigram_features_from_positions(G, p, 90.0, direction=True) for p in PAIRS]
        ),
        "bk": np.vstack(
            [bigram_features_from_positions(G, p, 90.0, kitchensink=True) for p in PAIRS]
        ),
        "tn": np.vstack([trigram_features_from_positions(G, t, 90.0) for t in TRIPLES]),
        "tw": np.vstack(
            [trigram_features_from_positions(G, t, 90.0, direction=True) for t in TRIPLES]
        ),
        "tk": np.vstack(
            [trigram_features_from_positions(G, t, 90.0, kitchensink=True) for t in TRIPLES]
        ),
    }


# --- the guard that licenses the whole frame ----------------------------------------------


def test_narrow_and_widened_frames_are_byte_identical(matrices):
    """Adding the kitchen-sink block moved NO pre-existing column, at max abs diff 0.000e+00.

    Read the narrow and widened columns back OUT of the kitchen-sink matrix by NAME and compare
    against the matrices those frames produce on their own. This is stronger than comparing
    widths: it would catch a reordering, a renamed column landing in the wrong slot, or a
    predicate that changed value only on pairs the shipped tests happen not to cover.
    """
    for wide_names, narrow_names, kitchen, widened, narrow in [
        (
            BIGRAM_DIRECTION_FEATURE_NAMES,
            BIGRAM_FEATURE_NAMES,
            matrices["bk"],
            matrices["bw"],
            matrices["bn"],
        ),
        (
            TRIGRAM_DIRECTION_FEATURE_NAMES,
            TRIGRAM_FEATURE_NAMES,
            matrices["tk"],
            matrices["tw"],
            matrices["tn"],
        ),
    ]:
        kitchen_names = (
            BIGRAM_KITCHENSINK_FEATURE_NAMES
            if kitchen is matrices["bk"]
            else TRIGRAM_KITCHENSINK_FEATURE_NAMES
        )
        index = {name: i for i, name in enumerate(kitchen_names)}
        assert np.abs(kitchen[:, [index[n] for n in wide_names]] - widened).max() == 0.0
        assert np.abs(kitchen[:, [index[n] for n in narrow_names]] - narrow).max() == 0.0


def test_kitchensink_is_a_strict_superset_of_the_widened_frame():
    """Every widened name survives, in order, as a prefix-preserving subsequence."""
    for wide, kitchen in [
        (BIGRAM_DIRECTION_FEATURE_NAMES, BIGRAM_KITCHENSINK_FEATURE_NAMES),
        (TRIGRAM_DIRECTION_FEATURE_NAMES, TRIGRAM_KITCHENSINK_FEATURE_NAMES),
    ]:
        assert set(wide) < set(kitchen)
        # relative order preserved
        positions = [kitchen.index(n) for n in wide]
        assert positions == sorted(positions)
        assert kitchen[-1] == "wpm"  # the pinned convention


def test_frame_widths_are_exactly_as_registered(matrices):
    """20/22/27 bigram and 46/52/69 trigram — the numbers the preregistration committed to."""
    assert (matrices["bn"].shape[1], matrices["bw"].shape[1], matrices["bk"].shape[1]) == (
        20,
        22,
        27,
    )
    assert (matrices["tn"].shape[1], matrices["tw"].shape[1], matrices["tk"].shape[1]) == (
        46,
        52,
        69,
    )
    # 12 definitions -> 17 new trigram columns, because the 5 bigram-level ones enter twice.
    assert len(TRIGRAM_KITCHENSINK_FEATURE_NAMES) - len(TRIGRAM_DIRECTION_FEATURE_NAMES) == 17
    assert len(BIGRAM_KITCHENSINK_FEATURE_NAMES) - len(BIGRAM_DIRECTION_FEATURE_NAMES) == 5


def test_the_three_stamps_are_pairwise_distinct():
    """The load-time guard can only tell the populations apart if the stamps differ."""
    stamps = [FEATURE_VERSION, FEATURE_VERSION_DIRECTION, FEATURE_VERSION_KITCHENSINK]
    assert len(set(stamps)) == 3
    assert FEATURE_VERSION_KITCHENSINK.startswith(FEATURE_VERSION)


def test_names_are_unique_within_each_frame():
    for names in (BIGRAM_KITCHENSINK_FEATURE_NAMES, TRIGRAM_KITCHENSINK_FEATURE_NAMES):
        assert len(names) == len(set(names))


def test_row_keys_match_schema_in_order():
    """Emission order must equal the declared order, or the vector is silently permuted."""
    assert (
        list(bigram_model_row(LAYOUT, "th", wpm=90, kitchensink=True))
        == BIGRAM_KITCHENSINK_FEATURE_NAMES
    )
    assert (
        list(trigram_model_row(LAYOUT, "the", wpm=90, kitchensink=True))
        == TRIGRAM_KITCHENSINK_FEATURE_NAMES
    )


def test_kitchensink_implies_direction():
    """There is no fourth frame: asking for kitchensink alone still yields the ordered columns."""
    row = bigram_model_row(LAYOUT, "th", wpm=90, kitchensink=True)
    assert "inwards_ordered" in row and "outwards_ordered" in row


# --- the audit's measured numbers, pinned -------------------------------------------------


def test_bigram_predicate_firing_counts():
    """Counts over all 870 ordered pairs, as reported in the registered audit table."""
    counts = {
        "half_scissor": sum(C.is_half_scissor(G, a, b) for a, b in PAIRS),
        "row_skip": sum(C.is_row_skip(G, a, b) for a, b in PAIRS),
        "pinky_off_home": sum(C.is_pinky_off_home(G, a, b) for a, b in PAIRS),
        "weak_finger_pair": sum(C.is_weak_finger_pair(G, a, b) for a, b in PAIRS),
        "finger_step": sum(C.finger_step(G, a, b) != 0 for a, b in PAIRS),
    }
    assert counts == {
        "half_scissor": 48,
        "row_skip": 100,
        "pinky_off_home": 116,
        "weak_finger_pair": 60,
        "finger_step": 324,
    }
    assert len(PAIRS) == 870


def test_trigram_predicate_firing_counts():
    """Counts over all 24,360 ordered triples, as reported in the registered audit table."""
    rows = [trigram_kitchensink_row(G, a, b, c) for a, b, c in TRIPLES]
    fired = {name: sum(r[name] != 0 for r in rows) for name in rows[0]}
    assert fired == {
        "onehand": 756,
        "onehand_in": 378,
        "red_sfs": 972,
        "alt_sfs": 1440,
        "sg_full_scissor": 672,
        "sg_half_scissor": 1344,
        "sg_lsb": 896,
    }
    assert len(TRIPLES) == 24360


def test_half_scissor_and_scissor_are_disjoint_and_neither_subsumes_the_other():
    """HSB is one row, ``is_scissor`` is two — a pair cannot be both, so it is genuinely new."""
    both = [(a, b) for a, b in PAIRS if C.is_half_scissor(G, a, b) and C.is_scissor(G, a, b)]
    assert both == []
    assert sum(C.is_scissor(G, a, b) for a, b in PAIRS) == 24  # the served column, unchanged


def test_row_skip_is_a_strict_superset_of_scissor():
    """Dropping the finger-adjacency requirement can only ADD firings (100 vs 24)."""
    for a, b in PAIRS:
        if C.is_scissor(G, a, b):
            assert C.is_row_skip(G, a, b)
    assert sum(C.is_row_skip(G, a, b) for a, b in PAIRS) > sum(
        C.is_scissor(G, a, b) for a, b in PAIRS
    )


def test_finger_step_is_exactly_antisymmetric_on_the_roll_eligible_set():
    """The graded direction column reverses sign under reversal — the property the SERVED
    ``inwards``/``outwards`` provably lack (they change on 0 of 870 pairs)."""
    eligible = [(a, b) for a, b in PAIRS if C.is_inwards_ordered(G, a, b) or C.is_outwards_ordered(G, a, b)]
    assert len(eligible) == 324
    for a, b in eligible:
        assert C.finger_step(G, a, b) == -C.finger_step(G, b, a) != 0
    # and it is 0 off that set, so it never invents a step for a same-finger reposition
    for a, b in PAIRS:
        if (a, b) not in set(eligible):
            assert C.finger_step(G, a, b) == 0.0


def test_finger_step_agrees_in_SIGN_with_the_binary_ordered_predicates():
    """The graded column must not contradict the binary one it generalizes."""
    for a, b in PAIRS:
        step = C.finger_step(G, a, b)
        if C.is_inwards_ordered(G, a, b):
            assert step > 0
        elif C.is_outwards_ordered(G, a, b):
            assert step < 0


def test_onehand_in_is_order_aware_but_onehand_is_not():
    """The direction split is where the order information lives; the roll itself is symmetric."""
    sym = sum(
        trigram_kitchensink_row(G, a, b, c)["onehand"]
        != trigram_kitchensink_row(G, c, b, a)["onehand"]
        for a, b, c in TRIPLES
    )
    asym = sum(
        trigram_kitchensink_row(G, a, b, c)["onehand_in"]
        != trigram_kitchensink_row(G, c, b, a)["onehand_in"]
        for a, b, c in TRIPLES
    )
    assert sym == 0
    assert asym == 756


def test_onehand_and_redirect_are_disjoint():
    """Monotonic and non-monotonic partition the distinct-finger one-hand triples, so the new
    column is the complement the served frame could only express as a conjunction of negatives."""
    from keybo.features.ngram import _trigram_level_from_positions

    for a, b, c in TRIPLES:
        if trigram_kitchensink_row(G, a, b, c)["onehand"]:
            assert not _trigram_level_from_positions(G, a, b, c)["redirect"]


def test_red_weak_was_rejected_because_it_is_an_identity_not_an_approximation():
    """The audit's headline rejection, pinned as a test so nobody re-adds keycraft's RED-WEAK.

    Over all 24,360 triples it is bit-identical to ``bad_redirect_sfgated`` — a column REDIRGATE-1
    already built and ``sfgated-eval`` already measured NULL. Its R2 = 1.0000 against the widened
    frame is therefore an identity. It looks novel (R2 = 0.685) only against the NARROW frame,
    because the SERVED ``bad_redirect`` fires 648 times, 216 more than the gated form.
    """

    def red_weak(a, b, c):
        ha, hb, hc = G.hand(a[0]), G.hand(b[0]), G.hand(c[0])
        if not (ha != 0 and ha == hb == hc):
            return 0.0
        ka, kb, kc = (C.finger_kind(G, p[0]) for p in (a, b, c))
        if C.same_finger(G, a, b) or C.same_finger(G, b, c):
            return 0.0
        if (ka < kb) == (kb < kc):
            return 0.0
        return float(ka != 3 and kb != 3 and kc != 3)

    mine = np.array([red_weak(*t) for t in TRIPLES])
    theirs = np.array(
        [trigram_direction_row(G, *t)["bad_redirect_sfgated"] for t in TRIPLES]
    )
    assert np.array_equal(mine, theirs)
    assert int(mine.sum()) == 432


def test_finger_kind_ranks_the_k31_quote_slot_as_a_pinky():
    """Column |x| = 6 is the right pinky, so the rank must not need a special case downstream."""
    assert C.finger_kind(ROW_STAGGERED_31, 6) == C.finger_kind(ROW_STAGGERED_31, 5) == 0
    assert [C.finger_kind(G, x) for x in (1, 2, 3, 4, 5)] == [3, 3, 2, 1, 0]
    assert [C.finger_kind(G, -x) for x in (1, 2, 3, 4, 5)] == [3, 3, 2, 1, 0]
    assert C.finger_kind(G, 0) == -1  # thumb: no dexterity rank
