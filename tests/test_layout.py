"""Tests for Layout: character<->position mapping, rendering, and swap/undo.

The swap/undo tests include the regression guard for the original 3-opt corruption bug:
the old ``undo_swap`` only remembered the *last* swapped pair, so undoing a multi-swap
sequence silently left the layout in a wrong (but still valid-looking) permutation.
"""

import random

import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout

QWERTY = "qwertyuiopasdfghjkl;zxcvbnm,./"
# 30 distinct chars incl. the two punctuation keys the old scorer ignored.
LAYOUT30 = "qwertyuiopasdfghjkl'zxcvbnm,.-"


def make(chars=LAYOUT30) -> Layout:
    return Layout(chars, ROW_STAGGERED_30)


# --- construction & lookup ------------------------------------------------------------


def test_maps_chars_to_canonical_slots():
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    assert lay.pos("q") == (-5, 3)  # top-left
    assert lay.pos("p") == (5, 3)  # top-right
    assert lay.pos("a") == (-5, 2)  # home-left
    assert lay.pos("/") == (5, 1)  # bottom-right


def test_key_at_is_the_inverse_of_pos():
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    for ch in QWERTY:
        assert lay.key_at(*lay.pos(ch)) == ch


def test_rejects_wrong_length():
    with pytest.raises(ValueError):
        Layout("abc", ROW_STAGGERED_30)


def test_rejects_duplicate_characters():
    dup = "q" + "wertyuiopasdfghjkl'zxcvbnm,.-"  # 'q' then 29 others, but duplicate a letter
    dup = "qq" + LAYOUT30[2:]
    with pytest.raises(ValueError):
        Layout(dup, ROW_STAGGERED_30)


def test_rejects_fixed_space_in_assignable_slots():
    with pytest.raises(ValueError, match="space"):
        Layout(" " + LAYOUT30[1:], ROW_STAGGERED_30)


def test_render_round_trips_to_three_rows():
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    rendered = lay.render()
    lines = rendered.splitlines()
    assert len(lines) == 3
    assert lines[0].split() == list("qwertyuiop")
    assert lines[1].split() == list("asdfghjkl;")
    assert lines[2].split() == list("zxcvbnm,./")


# --- swap / undo ----------------------------------------------------------------------


def test_swap_exchanges_two_keys():
    lay = make()
    qpos, wpos = lay.pos("q"), lay.pos("w")
    lay.swap("q", "w")
    assert lay.pos("q") == wpos
    assert lay.pos("w") == qpos


def test_single_swap_then_undo_restores():
    lay = make()
    before = dict(lay.as_dict())
    lay.swap("q", "w")
    lay.undo()
    assert lay.as_dict() == before


def test_regression_multi_swap_undo_restores_exactly():
    """REGRESSION for bug #1: two chained swaps must undo to the original layout.

    The old ``undo_swap`` reversed only the last-remembered pair, so this sequence left
    q/w/e rotated. With a swap stack, N undos restore N swaps in LIFO order.
    """
    lay = make()
    before = dict(lay.as_dict())
    lay.swap("q", "w")
    lay.swap("w", "e")
    lay.undo()
    lay.undo()
    assert lay.as_dict() == before


def test_negative_control_single_undo_after_two_swaps_does_not_restore():
    """Negative control proving the regression test above can actually fail.

    This reproduces the OLD behavior (one undo after two swaps) and asserts it does NOT
    restore — i.e. the bug was real and the guard above is not vacuous.
    """
    lay = make()
    before = dict(lay.as_dict())
    lay.swap("q", "w")
    lay.swap("w", "e")
    lay.undo()  # only one undo -- mimics the old last-pair-only behavior
    assert lay.as_dict() != before


def test_undo_on_empty_stack_raises():
    lay = make()
    with pytest.raises(IndexError):
        lay.undo()


def test_layout_stays_a_valid_permutation_after_many_swaps_and_undos():
    lay = make()
    original_chars = set(LAYOUT30)
    rng = random.Random(0)
    depth = 0
    for _ in range(200):
        if depth == 0 or rng.random() < 0.5:
            lay.random_swap(rng)
            depth += 1
        else:
            lay.undo()
            depth -= 1
        # No key is ever lost or duplicated.
        assert set(lay.as_dict()) == original_chars
        assert len({lay.pos(c) for c in original_chars}) == 30


# --- seedable random swap -------------------------------------------------------------


def test_random_swap_is_reproducible_with_seeded_rng():
    a = make()
    b = make()
    a.random_swap(random.Random(42))
    b.random_swap(random.Random(42))
    assert a.as_dict() == b.as_dict()
