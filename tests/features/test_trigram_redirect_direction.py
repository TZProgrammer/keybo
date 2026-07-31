"""The served trigram ``redirect`` column IS order-aware — and diverges from the port.

Two facts about ``features/ngram.py``'s trigram-level direction logic, found while adding the
ordered bigram channel and pinned here so neither is rediscovered.

**1. It is genuinely order-dependent.** Unlike the bigram ``inwards``/``outwards`` columns, the
trigram ``redirect`` column compares ``|column|`` between successive keys in STROKE ORDER, so
it already does what this branch adds to the bigram frame. That is worth pinning: it means the
served trigram frame is not direction-blind, and a reader who generalises "the roll predicates
are swap-invariant" to the whole feature pipeline would be wrong.

**2. It fires on 1,116 triples the parity-gated classifier excludes as same-finger.** Its step
is ``abs(b[0]) < abs(a[0])`` with no same-finger gate, so an index-finger move across the
index's OWN two columns (columns 1 and 2 — one finger, per
:meth:`keybo.geometry.Geometry.same_finger`) counts as a direction step.
:func:`keybo.analysis.community._v1_pattern` — the oxeylyzer-1 port that
``tests/analysis/test_kan1_parity.py`` gates integer-exact against the real upstream repl —
classifies those as Sfb and assigns them no trigram class at all.

This is the SAME defect class :mod:`keybo.scoring.oxey`'s docstring records fixing in ITS
trigram path ("the direction step was ``abs(b.column) - abs(a.column)``, which reads the index
finger's two columns as a direction STEP"). It was fixed there by delegating to the port; the
FEATURE column was never migrated.

⚠️ **NOT fixed here, deliberately.** ``redirect`` is column 2 of the 46-column trigram frame
and all three ``trigram_cond31`` models under ``data/models/k31/`` are stamped
``FEATURE_VERSION 2026-07-05.3`` (verified: their ``feature_names`` contains ``redirect``).
Changing the predicate would silently redefine a served column under a frozen stamp — the exact
train/serve skew this branch's bigram design exists to avoid, and it would need its own
retraining round to land honestly. Pinned as characterisation so the divergence is a known,
measured quantity rather than a latent surprise; the counts are exact, so a future migration has
its before-number ready.
"""

from __future__ import annotations

from itertools import permutations

from keybo.analysis.community import _v1_pattern
from keybo.features import trigram_model_row
from keybo.features.ngram import _trigram_level_from_positions
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.oxey import _LIBDOF_FINGER

LAYOUT = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
_REDIRECT_LABELS = frozenset({"redirects", "redirects_sfs", "bad_redirects", "bad_redirects_sfs"})
_BAD_LABELS = frozenset({"bad_redirects", "bad_redirects_sfs"})


def _v1_label(geometry, a, b, c) -> str | None:
    fingers = [_LIBDOF_FINGER.get(geometry.finger(p[0])) for p in (a, b, c)]
    if any(f is None for f in fingers):
        return None
    return _v1_pattern(*fingers)


def test_the_trigram_redirect_column_is_order_dependent():
    """``sdf`` (monotone inward) is not a redirect; ``sfd`` (in then out) is.

    The bigram roll columns cannot make this distinction; the trigram column can, because it
    reads successive ``|column|`` values in stroke order.
    """
    assert trigram_model_row(LAYOUT, "sdf", wpm=90)["redirect"] == 0.0
    assert trigram_model_row(LAYOUT, "sfd", wpm=90)["redirect"] == 1.0
    # and reversing a redirect keeps it a redirect while reversing a one-hand run does not
    # create one -- so it is order-DEPENDENT without being order-ANTISYMMETRIC
    assert trigram_model_row(LAYOUT, "dfs", wpm=90)["redirect"] == 1.0
    assert trigram_model_row(LAYOUT, "fds", wpm=90)["redirect"] == 0.0


def test_trigram_redirect_is_order_dependent_over_the_whole_slot_grid():
    """Exhaustive counterpart to the spot-checks: the column genuinely varies with order."""
    geometry = ROW_STAGGERED_30
    moved = sum(
        1
        for a, b, c in permutations(geometry.slots, 3)
        if _trigram_level_from_positions(geometry, a, b, c)["redirect"]
        != _trigram_level_from_positions(geometry, c, b, a)["redirect"]
    )
    assert moved > 0, "the trigram redirect column would be swap-invariant like the bigram pair"


def test_served_trigram_redirect_overcounts_versus_the_parity_gated_port():
    """The exact divergence, pinned as a number so a future migration has its baseline.

    Direction of the disagreement matters: the served column is a strict SUPERSET (0 triples go
    the other way), and every extra one has a same-finger constituent bigram. So this is
    precisely the missing Sfb exclusion, not a second unrelated difference.
    """
    geometry = ROW_STAGGERED_30
    triples = list(permutations(geometry.slots, 3))
    assert len(triples) == 24360

    both = served_only = port_only = same_finger_among_extras = 0
    for a, b, c in triples:
        served = bool(_trigram_level_from_positions(geometry, a, b, c)["redirect"])
        port = _v1_label(geometry, a, b, c) in _REDIRECT_LABELS
        if served and port:
            both += 1
        elif served:
            served_only += 1
            if geometry.same_finger(a[0], b[0]) or geometry.same_finger(b[0], c[0]):
                same_finger_among_extras += 1
        elif port:
            port_only += 1

    assert (both, served_only, port_only) == (2484, 1116, 0)
    assert same_finger_among_extras == served_only, "every extra is a same-finger step"


def test_served_trigram_bad_redirect_overcounts_the_same_way():
    """Same divergence on the ``bad_redirect`` column: 216 extras, none the other way."""
    geometry = ROW_STAGGERED_30
    both = served_only = port_only = 0
    for a, b, c in permutations(geometry.slots, 3):
        served = bool(_trigram_level_from_positions(geometry, a, b, c)["bad_redirect"])
        port = _v1_label(geometry, a, b, c) in _BAD_LABELS
        if served and port:
            both += 1
        elif served:
            served_only += 1
        elif port:
            port_only += 1
    assert (both, served_only, port_only) == (432, 216, 0)


def test_a_pure_index_column_move_is_counted_as_a_direction_step():
    """The mechanism, on one concrete triple — the readable form of the 1,116.

    Columns 3 -> 1 -> 2 on the left hand: the 1 -> 2 leg is the index finger moving between its
    own two columns, which ``same_finger`` calls ONE finger. The served column reads it as an
    outward step after an inward one and reports a redirect; the port calls the whole triple an
    Sfb and assigns no class.
    """
    geometry = ROW_STAGGERED_30
    a, b, c = (-3, 2), (-1, 2), (-2, 2)
    assert geometry.same_finger(b[0], c[0]) is True
    assert _trigram_level_from_positions(geometry, a, b, c)["redirect"] == 1.0
    assert _v1_label(geometry, a, b, c) is None
