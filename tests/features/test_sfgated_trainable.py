"""The gated redirect columns must be DECLARED and EMITTED, and only on the widened frame.

REDIRGATE-1 implemented `redirect_sfgated`/`bad_redirect_sfgated` but they were in no trainable name
list, so RETRAIN-DIRECTION-1 could not test them — it measured the ordered-roll channel only. A name
in a list that the row builder never emits is the same class of defect: the declaration and the thing
must agree, and a length check alone would not catch a column silently filled with the wrong value.
"""

from __future__ import annotations

from keybo.features.ngram import _trigram_row_from_positions, trigram_direction_row
from keybo.features.schema import (
    _TRIGRAM_LEVEL_NAMES,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30

_G = ROW_STAGGERED_30
_S = tuple(_G.slots)
_GATED = ("redirect_sfgated", "bad_redirect_sfgated")


def test_the_gated_columns_are_DECLARED_on_the_widened_list() -> None:
    for name in _GATED:
        assert name in TRIGRAM_DIRECTION_FEATURE_NAMES


def test_they_are_ABSENT_from_the_locked_served_list_and_its_shared_prefix() -> None:
    """The served frame must not grow. `_TRIGRAM_LEVEL_NAMES` is shared by BOTH lists, so putting
    them there would widen the locked frame silently — the exact skew DIRECTION-1 refused."""
    for name in _GATED:
        assert name not in TRIGRAM_FEATURE_NAMES
        assert name not in _TRIGRAM_LEVEL_NAMES


def test_the_widened_ROW_actually_EMITS_them_declared_is_not_emitted() -> None:
    row = _trigram_row_from_positions(_G, _S[0], _S[3], _S[6], 90.0, direction=True)
    for name in _GATED:
        assert name in row, f"{name} is declared in the name list but the row builder omits it"


def test_the_narrow_row_does_NOT_emit_them() -> None:
    row = _trigram_row_from_positions(_G, _S[0], _S[3], _S[6], 90.0)
    for name in _GATED:
        assert name not in row


def test_row_keys_match_the_widened_name_list_EXACTLY_and_in_order() -> None:
    row = _trigram_row_from_positions(_G, _S[0], _S[3], _S[6], 90.0, direction=True)
    assert list(row) == list(TRIGRAM_DIRECTION_FEATURE_NAMES)


def test_the_emitted_VALUES_equal_the_standalone_function_not_just_the_keys() -> None:
    """A column present but wrongly filled passes a key/length check and fails silently."""
    checked = 0
    for a in _S[:8]:
        for b in _S[:8]:
            for c in _S[:8]:
                if b in (a, c) or a == c:
                    continue
                row = _trigram_row_from_positions(_G, a, b, c, 90.0, direction=True)
                want = trigram_direction_row(_G, a, b, c)
                for name in _GATED:
                    assert row[name] == want[name], f"{name} mismatch at {(a, b, c)}"
                checked += 1
    assert checked > 100, "the sweep must actually cover triples"


def test_the_gated_column_FIRES_somewhere_in_the_widened_row() -> None:
    """A column that is always 0.0 would pass every test above and teach a model nothing."""
    fired = 0
    for a in _S:
        for b in _S:
            for c in _S:
                if b in (a, c) or a == c:
                    continue
                if _trigram_row_from_positions(_G, a, b, c, 90.0, direction=True)[
                    "redirect_sfgated"
                ]:
                    fired += 1
    assert fired == 3600 - 1116, (
        f"expected 2484 firings (3600 ungated minus 1116 gated), got {fired}"
    )
