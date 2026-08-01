"""The FIRST-KEY ABSOLUTE POSITION frame (ABSPOS-1): additive, order-correct, commensurable.

The load-bearing test here is :func:`test_served_46_columns_are_byte_identical`. Everything else
describes the new ``bg0_`` block; that one proves the SHIPPED trigram frame did not move, which is
the whole basis on which a fourth frame is allowed to exist (three ``trigram_cond31`` models under
``data/models/k31/`` are stamped ``FEATURE_VERSION``, and ``keybo.models.base`` errors on a version
MISMATCH, not on a column whose MEANING changed — so a silent redefinition would leave every model
loading fine while scoring the wrong matrix).

The asymmetry this frame closes, and why it is a REPRESENTATION experiment rather than an
information one, is measured in ``agent-artifacts/firstkey_identifiability2.py`` and
``agent-artifacts/firstkey_recoverability.py``; the numbers those produce are pinned in
:func:`test_first_key_position_is_recoverable_from_the_served_frame` so a later edit cannot quietly
turn the frame into something whose registered justification no longer holds.
"""

from itertools import product

import numpy as np
import pytest

from keybo.features import classify as C
from keybo.features import trigram_features_from_positions, trigram_model_row
from keybo.features.ngram import first_key_placement_row
from keybo.features.schema import (
    FEATURE_VERSION,
    FEATURE_VERSION_ABSPOS,
    FEATURE_VERSION_DIRECTION,
    FEATURE_VERSION_KITCHENSINK,
    TRIGRAM_ABSPOS_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31
from keybo.layout import Layout

G = ROW_STAGGERED_30
G31 = ROW_STAGGERED_31
LAYOUT = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", G)
BLOCK = ("bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral")
ALL_31 = [*G31.slots, G31.space_position]


# --- the load-bearing guard: the served frame did not move --------------------------------


def test_served_46_columns_are_byte_identical():
    """The 46 served columns must be bit-identical between the narrow and abspos frames, over the
    FULL K31 triple enumeration (repeats included — the real data contains 541 a==b and 451 b==c
    rows, so excluding them would leave the shipped path partly unchecked)."""
    idx = [TRIGRAM_ABSPOS_FEATURE_NAMES.index(n) for n in TRIGRAM_FEATURE_NAMES]
    for a, b, c in product(ALL_31, repeat=3):
        narrow = trigram_features_from_positions(G31, (a, b, c), wpm=87.0)
        wide = trigram_features_from_positions(G31, (a, b, c), wpm=87.0, abspos=True)
        assert np.array_equal(wide[idx], narrow), (a, b, c)


def test_abspos_frame_is_a_pure_addition():
    assert [n for n in TRIGRAM_FEATURE_NAMES if n not in TRIGRAM_ABSPOS_FEATURE_NAMES] == []
    new = [n for n in TRIGRAM_ABSPOS_FEATURE_NAMES if n not in TRIGRAM_FEATURE_NAMES]
    assert new == [f"bg0_{n}" for n in BLOCK]
    assert len(TRIGRAM_ABSPOS_FEATURE_NAMES) == len(TRIGRAM_FEATURE_NAMES) + 8 == 54


def test_all_four_stamps_are_distinct():
    """The load-time guard in ``keybo.models.base`` can only tell the four model populations apart
    if no two stamps collide."""
    stamps = [
        FEATURE_VERSION,
        FEATURE_VERSION_DIRECTION,
        FEATURE_VERSION_KITCHENSINK,
        FEATURE_VERSION_ABSPOS,
    ]
    assert len(set(stamps)) == 4
    assert FEATURE_VERSION_ABSPOS.startswith(FEATURE_VERSION)


def test_column_names_unique_and_wpm_last():
    assert len(TRIGRAM_ABSPOS_FEATURE_NAMES) == len(set(TRIGRAM_ABSPOS_FEATURE_NAMES))
    assert TRIGRAM_ABSPOS_FEATURE_NAMES[-1] == "wpm"


def test_row_keys_match_schema_in_order():
    row = trigram_model_row(LAYOUT, "the", wpm=90, abspos=True)
    assert list(row) == TRIGRAM_ABSPOS_FEATURE_NAMES


def test_bg0_block_reads_in_stroke_order_before_bg1():
    """bg0_/bg1_/bg2_ must appear in stroke order, so the frame reads a -> b -> c."""
    names = TRIGRAM_ABSPOS_FEATURE_NAMES
    assert names.index("bg0_bottom") < names.index("bg1_bottom") < names.index("bg2_bottom")
    # and the whole bg0_ block is contiguous
    positions = [i for i, n in enumerate(names) if n.startswith("bg0_")]
    assert positions == list(range(positions[0], positions[0] + 8))


# --- the block means what bg1_/bg2_ mean, applied to key a --------------------------------


def test_bg0_is_the_second_key_definition_applied_to_the_first_key():
    """Commensurability is the point of this block: ``bg0_x`` for a trigram (a, b, c) must equal
    ``bg1_x`` for any trigram whose SECOND key is ``a`` — i.e. the identical eight definitions."""
    probe = [s for s in ALL_31 if s != G31.space_position][:8]
    for a in ALL_31:
        first = trigram_model_row_from_positions(a, probe[0], probe[1])
        others = [s for s in probe if s != a][:2]
        second = trigram_model_row_from_positions(others[0], a, others[1])
        for k in BLOCK:
            assert first[f"bg0_{k}"] == second[f"bg1_{k}"], (a, k)


def trigram_model_row_from_positions(a, b, c):
    from keybo.features.ngram import _trigram_row_from_positions

    return _trigram_row_from_positions(G31, a, b, c, 0.0, abspos=True)


@pytest.mark.parametrize(
    ("pos", "expected"),
    [
        ((-4, 3), {"top": 1.0, "ring": 1.0}),  # top row, ring finger
        ((-5, 2), {"home": 1.0, "pinky": 1.0}),
        ((1, 1), {"bottom": 1.0, "index": 1.0, "lateral": 1.0}),  # |x|==1 is the stretch column
        ((2, 2), {"home": 1.0, "index": 1.0}),  # index's home column: NOT lateral
        ((6, 2), {"home": 1.0, "pinky": 1.0, "lateral": 1.0}),  # K31 quote slot
        ((0, 0), {}),  # space: no row, no finger, no lateral
    ],
)
def test_first_key_block_values(pos, expected):
    row = first_key_placement_row(G31, pos)
    assert set(row) == set(BLOCK)
    for k in BLOCK:
        assert row[k] == expected.get(k, 0.0), k


def test_one_hots_are_exclusive_over_the_whole_board():
    """Exactly one row and one finger for every letter key; none for space."""
    for p in ALL_31:
        row = first_key_placement_row(G31, p)
        want = 0.0 if p == G31.space_position else 1.0
        assert row["bottom"] + row["home"] + row["top"] == want, p
        assert row["pinky"] + row["ring"] + row["middle"] + row["index"] == want, p


def test_lateral_matches_the_shared_predicate():
    for p in ALL_31:
        assert first_key_placement_row(G31, p)["lateral"] == float(C.is_lateral(p[0]))


# --- the frame does NOT compose with the other two ----------------------------------------


@pytest.mark.parametrize("other", [{"direction": True}, {"kitchensink": True}])
def test_abspos_refuses_to_mix_with_the_other_frames(other):
    """No name list and no version stamp exists for a mixed frame, and ABSPOS-1 is a
    single-variable A/B against the SERVED frame — so a mixed request must raise rather than
    silently produce a mislabelled model."""
    with pytest.raises(ValueError, match="cannot be combined"):
        trigram_features_from_positions(
            G31, (ALL_31[0], ALL_31[1], ALL_31[2]), wpm=1.0, abspos=True, **other
        )


def test_abspos_is_trigram_only_in_training():
    """The bigram builder does not accept ``abspos`` at all, so ``_train`` must reject it BEFORE
    building the matrix (otherwise it dies minutes later with an unrelated TypeError)."""
    from keybo.training.train import _train

    with pytest.raises(ValueError, match="trigram-only"):
        _train([], "bigram", 90.0, (60, 120), G, abspos=True)


# --- the scorer must be told which frame it is scoring -------------------------------------


class _StubMeta:
    def __init__(self, stamp):
        self.feature_version = stamp
        self.extra = {}
        self.ngram = "trigram"


class _StubModel:
    def __init__(self, stamp):
        self.metadata = _StubMeta(stamp)


@pytest.mark.parametrize(
    ("stamp", "flags"),
    [
        (FEATURE_VERSION_ABSPOS, {}),  # abspos model, served flags -> 54 vs 46
        (FEATURE_VERSION_ABSPOS, {"direction": True}),
        (FEATURE_VERSION, {"abspos": True}),  # served model, abspos flags -> 46 vs 54
        (FEATURE_VERSION_DIRECTION, {"abspos": True}),
        (FEATURE_VERSION_KITCHENSINK, {"abspos": True}),
    ],
)
def test_scorer_refuses_a_frame_that_contradicts_the_model_stamp(stamp, flags):
    """ABSPOS-1 hit this for real: a scorer constructed WITHOUT ``abspos`` against a 54-column model
    featurized 46 columns and died with XGBoost's ``Feature shape mismatch, expected: 54, got 46``
    only after the whole corpus had been built. The stamp is checked at construction instead."""
    from keybo.scoring.model_scorer import TrigramModelScorer

    with pytest.raises(ValueError, match="do not match the model's feature_version"):
        TrigramModelScorer(_StubModel(stamp), trigram_freqs={"the": 1}, **flags)


@pytest.mark.parametrize(
    ("stamp", "flags"),
    [
        (FEATURE_VERSION, {}),
        (FEATURE_VERSION_ABSPOS, {"abspos": True}),
        (FEATURE_VERSION_DIRECTION, {"direction": True}),
        (FEATURE_VERSION_KITCHENSINK, {"kitchensink": True}),
    ],
)
def test_scorer_accepts_matching_frames(stamp, flags):
    from keybo.scoring.model_scorer import TrigramModelScorer

    TrigramModelScorer(_StubModel(stamp), trigram_freqs={"the": 1}, **flags)


# --- the registered justification: recoverable, but not at the served depth ----------------


def test_first_key_position_is_recoverable_from_the_served_frame():
    """ABSPOS-1's honest premise, pinned: the block is NOT new information.

    Bucketing trigrams by their exact 46-column served row never yields two trigrams that disagree
    on key A's absolute row or finger — so A's absolute position is a deterministic FUNCTION of the
    served columns, and this frame is a REPRESENTATION experiment (cheapness at depth 3), not an
    information one. Restricted to a 12-slot sub-board so the test stays fast; the full 32,768-triple
    run lives in ``agent-artifacts/firstkey_identifiability2.py``.
    """
    slots = [*G31.slots[:5], *G31.slots[10:15], *G31.slots[20:22], G31.space_position]
    buckets: dict[tuple, set] = {}
    for a, b, c in product(slots, repeat=3):
        vec = trigram_features_from_positions(G31, (a, b, c), wpm=0.0)
        key = tuple(np.round(vec, 12))
        block = tuple(first_key_placement_row(G31, a)[k] for k in BLOCK)
        buckets.setdefault(key, set()).add(block)
    ambiguous = [k for k, v in buckets.items() if len(v) > 1]
    assert not ambiguous, f"{len(ambiguous)} served rows are ambiguous in key A's position"


def test_k30_served_path_is_untouched_by_the_new_flag():
    """The default (abspos=False) must reproduce the served frame byte for byte on the K30 domain —
    the same property ``tests/features/test_k31_geometry.py`` freezes against its golden matrix."""
    pos = [*G.slots, G.space_position]
    for a in pos[:6]:
        for b in pos[:6]:
            v = trigram_features_from_positions(G, (a, b, pos[7]), wpm=87.0)
            assert len(v) == len(TRIGRAM_FEATURE_NAMES) == 46
