"""The trigram redirect step must not fire across a same-finger bigram.

`_trigram_level_from_positions` derives its direction step from ``abs(b[0]) < abs(a[0])`` with NO
same-finger gate, so it treats a same-finger move as a direction change. The parity-gated
``_v1_pattern`` port excludes those as Sfb, and ``scoring/oxey.py`` records having fixed exactly this
in its own trigram path — the served feature column never got the same treatment.

Measured on ROW_STAGGERED_30 over the 24,360 ordered triples with all three slots distinct
(DIRECTION-1's frame; note a wider ``a!=b, b!=c`` frame gives 25,230 / 1,152 and the same 216):

    frame                     triples   redirect  (same-finger)   bad_redirect  (same-finger)
    all three distinct         24,360      3,600        1,116            648           216
    a!=b and b!=c              25,230      3,960        1,152            756           216

    Both frames are pinned below because the two are easy to mix up -- I did, and wrote the wider
    frame's 3,960/756 next to the all-distinct 1,116/216 in a first draft. The same-finger overfire
    is 1,116 in the all-distinct frame and 1,152 in the wider one; ONLY bad_redirect's 216 is the
    same in both, which is exactly the coincidence that hides the mistake.

ADDITIVE, exactly as DIRECTION-1 was: ``redirect`` and ``bad_redirect`` are columns of the
version-locked 46-feature trigram frame that all three ``trigram_cond31`` models carry, so
redefining them in place would leave every model loading fine while scoring a frame whose columns
changed meaning — silent train/serve skew, not a publishable renumbering. The gated variants are new
columns; the old ones are untouched and pinned bit-identical here.
"""

from __future__ import annotations

from keybo.features import classify as C
from keybo.features.ngram import _trigram_level_from_positions, trigram_direction_row
from keybo.geometry import ROW_STAGGERED_30

_G = ROW_STAGGERED_30
_SLOTS = tuple(_G.slots)


def _triples():
    for a in _SLOTS:
        for b in _SLOTS:
            for c in _SLOTS:
                if a != b and b != c and a != c:
                    yield a, b, c


def _has_same_finger_bigram(a, b, c) -> bool:
    return C.same_finger(_G, a, b) or C.same_finger(_G, b, c)


def test_the_overfire_is_exactly_what_the_ledger_registered() -> None:
    """Pin the defect's size, so a future migration has its before-numbers."""
    total = redirect = bad = redirect_sf = bad_sf = 0
    for a, b, c in _triples():
        total += 1
        row = _trigram_level_from_positions(_G, a, b, c)
        sf = _has_same_finger_bigram(a, b, c)
        if row["redirect"]:
            redirect += 1
            redirect_sf += sf
        if row["bad_redirect"]:
            bad += 1
            bad_sf += sf
    assert total == 24360
    assert (redirect, redirect_sf) == (3600, 1116)
    assert (bad, bad_sf) == (648, 216)


def test_the_GATED_column_excludes_every_same_finger_case() -> None:
    """The whole point: the gated variant must fire on none of the 1,116."""
    leaked = [
        (a, b, c)
        for a, b, c in _triples()
        if trigram_direction_row(_G, a, b, c)["redirect_sfgated"]
        and _has_same_finger_bigram(a, b, c)
    ]
    assert leaked == [], f"{len(leaked)} same-finger triples still fire the gated redirect"


def test_the_gated_column_is_a_STRICT_SUBSET_of_the_ungated_one() -> None:
    """A gate may only ever remove. If it adds a firing, it is a different feature."""
    added = 0
    removed = 0
    for a, b, c in _triples():
        row = _trigram_level_from_positions(_G, a, b, c)
        gated_row = trigram_direction_row(_G, a, b, c)
        if gated_row["redirect_sfgated"] and not row["redirect"]:
            added += 1
        if row["redirect"] and not gated_row["redirect_sfgated"]:
            removed += 1
    assert added == 0, f"the gate ADDED {added} firings — it is not a gate"
    assert removed == 1116, f"expected to remove exactly the 1,116 overfires, removed {removed}"


def test_bad_redirect_is_gated_too_and_removes_exactly_216() -> None:
    added = removed = 0
    for a, b, c in _triples():
        row = _trigram_level_from_positions(_G, a, b, c)
        gated_row = trigram_direction_row(_G, a, b, c)
        if gated_row["bad_redirect_sfgated"] and not row["bad_redirect"]:
            added += 1
        if row["bad_redirect"] and not gated_row["bad_redirect_sfgated"]:
            removed += 1
    assert added == 0
    assert removed == 216


def test_the_OLD_columns_are_BIT_IDENTICAL_so_no_published_number_moves() -> None:
    """The safety property that licenses this change, checked the way DIRECTION-1 checked its own.

    The gated columns are additive; the version-locked ones must be untouched. If this fails, the
    change has become silent train/serve skew against all three trigram_cond31 models.
    """
    # Recompute the ungated columns from first principles and require exact agreement.
    for a, b, c in _triples():
        row = _trigram_level_from_positions(_G, a, b, c)
        ha, hb, hc = _G.hand(a[0]), _G.hand(b[0]), _G.hand(c[0])
        same_hand_tri = ha != 0 and ha == hb == hc
        expected = False
        if same_hand_tri:
            expected = (abs(b[0]) < abs(a[0])) != (abs(c[0]) < abs(b[0]))
        assert row["redirect"] == float(expected)


def test_a_concrete_same_finger_case_reads_correctly() -> None:
    """`ded` on qwerty: d->e and e->d are both the SAME finger, so no direction step exists."""
    from keybo.layout import Layout

    lay = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", _G)
    d, e = lay.pos("d"), lay.pos("e")
    f = lay.pos("f")
    assert C.same_finger(_G, d, e), "d and e must be the same finger for this case to bite"
    ungated = _trigram_level_from_positions(_G, d, e, f)
    gated = trigram_direction_row(_G, d, e, f)
    # The contrast is the point: the shipped column FIRES here and the gated one does not, so this
    # single trigram is a worked example of the 1,116.
    assert ungated["redirect"] == 1.0, "the shipped column must fire, or this case proves nothing"
    assert gated["redirect_sfgated"] == 0.0, "a same-finger prefix cannot be a redirect"


def test_the_WIDER_frame_census_is_pinned_too_because_the_two_get_conflated() -> None:
    """`a != b and b != c` (ABA trigrams allowed) is a different frame with different totals.

    Pinned separately because only ``bad_redirect``'s 216 same-finger count is shared between the
    two frames -- a coincidence that makes a mixed-frame quotation look self-consistent.
    """
    total = redirect = bad = redirect_sf = bad_sf = 0
    for a in _SLOTS:
        for b in _SLOTS:
            for c in _SLOTS:
                if a == b or b == c:
                    continue
                total += 1
                row = _trigram_level_from_positions(_G, a, b, c)
                sf = _has_same_finger_bigram(a, b, c)
                if row["redirect"]:
                    redirect += 1
                    redirect_sf += sf
                if row["bad_redirect"]:
                    bad += 1
                    bad_sf += sf
    assert total == 25230
    assert (redirect, redirect_sf) == (3960, 1152)
    assert (bad, bad_sf) == (756, 216)
