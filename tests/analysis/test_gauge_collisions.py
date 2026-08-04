"""FM4 — the NAME-COLLISION classification, re-derived here rather than trusted.

The map in :mod:`keybo.analysis.shap_diff` claims, per served column, that the column's predicate
either EQUALS or DIFFERS FROM the same-named reported gauge. Every one of those claims is a
measurement, so every one is re-measured here **exhaustively** over the full enumeration of
``ROW_STAGGERED_30`` — all 900 ordered position pairs, all 27,000 ordered triples — against the
gauge's OWN code path, never a re-derivation of it.

Three things this file is deliberately structured to catch, because each is a way the map could be
wrong while looking right:

* a column listed as colliding that is actually predicate-EQUAL (a rename that would destroy a
  true correspondence — the failure mode the parent brief named);
* a column NOT listed that actually differs (a collision left un-annotated);
* the map drifting out of sync with the served schema, so it annotates a column that no longer
  exists or misses one that appeared.

The counts are asserted as EXACT integers, not inequalities. An inequality would pass while the
predicate changed underneath it, which is the same "green test that does not test its own name"
failure PRODUCTIZE-1 found three of.
"""

from __future__ import annotations

import itertools

import pytest

from keybo.analysis import kmstats as KM
from keybo.analysis.community import _v1_pattern
from keybo.analysis.shap_diff import (
    _COLLISION_COLUMNS,
    _GAUGE_SIDE,
    GAUGE_COLLISIONS,
    display_name,
    gauge_collision_notes,
    gauge_side_collision_notes,
)
from keybo.features import classify as C
from keybo.features.ngram import _placement_row_from_positions, _trigram_level_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES, TRIGRAM_FEATURE_NAMES
from keybo.geometry import ROW_STAGGERED_30

G = ROW_STAGGERED_30
SLOTS = tuple(G.slots)
#: The gauge modules index keys by SLOT, this module by POSITION; this is the bridge.
_KM_KEY = {pos: KM._KEYS[i] for i, pos in enumerate(SLOTS)}
_FAM_FINGER = {pos: KM._KEYS[i].finger for i, pos in enumerate(SLOTS)}

_REDIRECT_FAMILY = ("redirects", "redirects_sfs", "bad_redirects", "bad_redirects_sfs")
_BAD_FAMILY = ("bad_redirects", "bad_redirects_sfs")


@pytest.fixture(scope="module")
def pairs():
    return tuple(itertools.product(SLOTS, repeat=2))


@pytest.fixture(scope="module")
def triples():
    return tuple(itertools.product(SLOTS, repeat=3))


def _frame(a, b, name):
    return bool(_placement_row_from_positions(G, a, b)[name])


def _frame_tri(a, b, c, name):
    return bool(_trigram_level_from_positions(G, a, b, c)[name])


# --- the EQUAL verdicts: columns that must NOT be renamed ----------------------------------


def test_scissor_column_equals_the_scissor_gauge_exactly(pairs):
    """``scissor`` is the one shared name that is TRUTHFUL, so it must stay un-annotated.

    Both ``oxey.pattern_shares`` (oxey.py, the ``scissor`` gauge) and ``comfort.fitness`` call
    ``classify.is_scissor`` directly, so this asserts the delegation has not been replaced by a
    hand-rolled copy — the defect ``oxey`` records having fixed in its own trigram path.
    """
    disagree = [(a, b) for a, b in pairs if _frame(a, b, "scissor") != C.is_scissor(G, a, b)]
    assert disagree == [], f"{len(disagree)} of {len(pairs)} pairs disagree"
    assert sum(1 for a, b in pairs if _frame(a, b, "scissor")) == 24, "the firing count itself"
    # The EXACT-two-row gate, pinned as the definition rather than left to the count. ``dy == 2``
    # and ``dy >= 2`` are indistinguishable on a 3-row board (mutation M14 survived on counts
    # alone), so the discriminating assertion is over SYNTHETIC positions with a 3-row span: the
    # one-row and three-row reaches must NOT be scissors, only the two-row one.
    left_ring, left_middle = -4, -3
    assert C.is_scissor(G, (left_ring, 3), (left_middle, 1)), "two rows apart IS a scissor"
    assert not C.is_scissor(G, (left_ring, 2), (left_middle, 1)), "one row apart is NOT"
    assert not C.is_scissor(G, (left_ring, 4), (left_middle, 1)), "THREE rows apart is NOT"
    assert not C.is_scissor(G, (left_ring, 1), (left_middle, 1)), "same row is NOT"


def test_scissor_is_absent_from_the_collision_map():
    """The EQUAL verdict must be expressed as ABSENCE, or the annotation becomes noise.

    Annotating a truthful shared name would train a reader to ignore the annotation, which
    destroys the value of the ones that matter.
    """
    assert "scissor" not in GAUGE_COLLISIONS
    assert display_name("scissor") == "scissor"
    assert display_name("bg1_scissor") == "bg1_scissor"


def test_inwards_outwards_are_absent_because_they_match_their_gauge(pairs):
    """``inwards``/``outwards`` DO lie about direction, but they lie IDENTICALLY to their gauge.

    ``oxey``'s ``inroll``/``outroll`` delegate to the same ``classify.is_inwards``, so the
    column and the gauge agree on all 900 pairs. The names mislead about PHYSICS (both are
    swap-invariant — asserted below), which is FM4's other half and is already disclosed by
    ``effect_curves``' ``outer_high``/``outer_low`` and by ``oxey``'s own honesty note. It is
    NOT a column-vs-gauge collision, so it does not belong in this map.
    """
    assert [(a, b) for a, b in pairs if _frame(a, b, "inwards") != C.is_inwards(G, a, b)] == []
    assert [(a, b) for a, b in pairs if _frame(a, b, "outwards") != C.is_outwards(G, a, b)] == []
    assert "inwards" not in GAUGE_COLLISIONS
    assert "outwards" not in GAUGE_COLLISIONS
    # and the swap-invariance that makes the NAMES (not the collision) the problem
    ordered = [(a, b) for a, b in pairs if a != b]
    assert sum(1 for a, b in ordered if _frame(a, b, "inwards") != _frame(b, a, "inwards")) == 0
    assert sum(1 for a, b in ordered if _frame(a, b, "outwards") != _frame(b, a, "outwards")) == 0


def test_sg_distance_column_equals_the_sg_dist_gauge_per_cell(triples):
    """``sg_distance`` vs the ``sg_dist`` gauge: a NEAR-name that is the same quantity.

    ``skipgram_span.sg_dist`` is the corpus-weighted mean of ``distance(first, third)``, which is
    exactly this column. Asserted so the near-name is on record as CHECKED-AND-EQUAL rather than
    merely unlisted.
    """
    bad = [
        (a, b, c)
        for a, b, c in triples
        if _trigram_level_from_positions(G, a, b, c)["sg_distance"] != G.distance(a, c)
    ]
    assert bad == []
    assert "sg_distance" not in GAUGE_COLLISIONS


# --- the DIFFERENT verdicts: exact disagreement counts -------------------------------------


def test_lsb_column_is_a_strict_superset_of_the_lsb_gauge(pairs):
    """``lsb`` the column (dx > 1.5) vs ``lsb`` the frozen keymeow gauge (|dx| >= 2).

    Both are index/middle on this board, so the difference is purely the THRESHOLD — which is why
    the disagreement is one-directional and lands exactly on the dx == 1.75 pairs.
    """
    frame_only, gauge_only = [], []
    for a, b in pairs:
        f = _frame(a, b, "lsb")
        g = bool(KM._is_lsb(_KM_KEY[a], _KM_KEY[b]))
        if f and not g:
            frame_only.append((a, b))
        elif g and not f:
            gauge_only.append((a, b))
    assert len(frame_only) == 8, "the column fires where the gauge does not"
    assert gauge_only == [], "STRICT superset: the gauge never fires alone"
    assert sum(1 for a, b in pairs if _frame(a, b, "lsb")) == 32
    assert sum(1 for a, b in pairs if KM._is_lsb(_KM_KEY[a], _KM_KEY[b])) == 24
    # the MECHANISM, so a threshold change cannot pass by keeping the count
    assert {G.stagger_adjusted_dx(a, b) for a, b in frame_only} == {1.75}


def test_lateral_column_and_lat_span_gauge_are_different_shapes(pairs):
    """``lateral`` is a ONE-KEY one-hot; the ``lat-span`` gauge is a graded PAIRWISE stretch.

    The two structural properties are asserted directly, because they are why no threshold choice
    could reconcile the two: one is a function of ``b`` alone, the other is symmetric in ``a, b``.
    """
    only_lateral = only_span = 0
    for a, b in pairs:
        lat = _frame(a, b, "lateral")
        span = C.lateral_span(G, a, b) > 0.0
        only_lateral += lat and not span
        only_span += span and not lat
    assert only_lateral == 126
    assert only_span == 126
    assert sum(1 for a, b in pairs if _frame(a, b, "lateral")) == 180
    # `lateral` reads the LANDING key only
    assert all(
        _frame(a, b, "lateral") == _frame(a2, b, "lateral")
        for b in SLOTS
        for a, a2 in ((SLOTS[0], SLOTS[-1]),)
    )
    # `lat-span` is symmetric, so it cannot be a landing-key property
    assert all(C.lateral_span(G, a, b) == C.lateral_span(G, b, a) for a, b in pairs)


def test_lateral_column_covers_the_K31_pinky_stretch_column_too():
    """``lateral`` fires on |x| == 6 as well as |x| == 1 — and only K31 HAS an |x| == 6 slot.

    Asserted on ``ROW_STAGGERED_31`` specifically because K30 carries no such column, so a
    K30-only test cannot see the second half of ``is_lateral``'s definition at all: dropping the
    ``6`` leaves every K30 number unchanged. (Found by mutation M13, which survived a K30-only
    suite.) The K31 disagreement count against ``lat-span`` also differs from K30's, which is the
    positive signal that the extra column is live.
    """
    from keybo.geometry import ROW_STAGGERED_31

    g31 = ROW_STAGGERED_31
    slots31 = tuple(g31.slots)
    assert 6 in {abs(p[0]) for p in slots31}, "K31 must carry the quote-slot column"
    assert 6 not in {abs(p[0]) for p in SLOTS}, "K30 must not — that is why this test exists"

    pairs31 = tuple(itertools.product(slots31, repeat=2))
    lateral = sum(1 for a, b in pairs31 if _placement_row_from_positions(g31, a, b)["lateral"])
    # 31 landing keys, of which those with |x| in {1, 6} fire: 6 inner-index + 1 quote slot = 7,
    # each reachable from all 31 first keys.
    assert lateral == 217
    assert sum(1 for p in slots31 if C.is_lateral(p[0])) == 7
    assert C.is_lateral(6) and C.is_lateral(-1)
    # and the collision against `lat-span` is DIFFERENT on K31 than on K30 (252 there)
    disagree = sum(
        1
        for a, b in pairs31
        if bool(_placement_row_from_positions(g31, a, b)["lateral"])
        != (C.lateral_span(g31, a, b) > 0.0)
    )
    assert disagree == 283


def test_redirect_column_is_a_strict_superset_of_both_redirect_gauges(triples):
    """``redirect`` the column has NO same-finger gate; both reported gauges do.

    THE CORRECTION THIS TEST EXISTS FOR: ``analysis/redirects.py`` documents an exhaustive
    equality over all 30**3 triples, and that equality is between the TWO GAUGES
    (``kmstats._is_redirect`` and ``_v1_pattern``) — neither of which is this column. Both
    relations are asserted here so the distinction cannot be re-lost: the gauges agree with each
    other EXACTLY, and the column is a strict superset of both.
    """
    col = km = v1 = 0
    col_not_km = km_not_col = km_not_v1 = v1_not_km = 0
    for a, b, c in triples:
        f = _frame_tri(a, b, c, "redirect")
        k = bool(KM._is_redirect(_KM_KEY[a], _KM_KEY[b], _KM_KEY[c]))
        v = _v1_pattern(_FAM_FINGER[a], _FAM_FINGER[b], _FAM_FINGER[c]) in _REDIRECT_FAMILY
        col += f
        km += k
        v1 += v
        col_not_km += f and not k
        km_not_col += k and not f
        km_not_v1 += k and not v
        v1_not_km += v and not k
    # the two GAUGES are equal -- redirects.py's own documented claim, re-derived
    assert (km_not_v1, v1_not_km) == (0, 0), "the two gauges must agree exactly"
    assert km == v1 == 2808
    # the COLUMN is a strict superset of them
    assert col == 4320
    assert col_not_km == 1512
    assert km_not_col == 0, "STRICT superset: no gauge firing is missed by the column"


def test_the_redirect_gap_is_exactly_the_missing_same_finger_gate(triples):
    """The 1512 extra firings are precisely the same-finger-constituent ones (REDIRGATE-1).

    Naming the mechanism, not just the count: a future change that altered the predicate while
    coincidentally preserving 1512 would still fail this.
    """
    extra = [
        (a, b, c)
        for a, b, c in triples
        if _frame_tri(a, b, c, "redirect")
        and not KM._is_redirect(_KM_KEY[a], _KM_KEY[b], _KM_KEY[c])
    ]
    assert len(extra) == 1512
    assert all(C.same_finger(G, a, b) or C.same_finger(G, b, c) for a, b, c in extra)


def test_bad_redirect_column_is_a_strict_superset_of_the_bad_redirect_gauge(triples):
    """Same defect, same mechanism, on the no-index-finger sub-class."""
    col = gauge = col_not_gauge = gauge_not_col = 0
    for a, b, c in triples:
        f = _frame_tri(a, b, c, "bad_redirect")
        g = _v1_pattern(_FAM_FINGER[a], _FAM_FINGER[b], _FAM_FINGER[c]) in _BAD_FAMILY
        col += f
        gauge += g
        col_not_gauge += f and not g
        gauge_not_col += g and not f
    assert col == 864
    assert gauge == 540
    assert col_not_gauge == 324
    assert gauge_not_col == 0


def test_the_gated_columns_ARE_the_gauge_predicate(triples):
    """``redirect_sfgated``/``bad_redirect_sfgated`` equal the gauges EXACTLY.

    This is what makes the chosen display names evidence-driven rather than invented: the gauge's
    predicate already HAS a column name in this repo (REDIRGATE-1's gated pair), so the served
    columns are correctly described as the UNGATED ones.
    """
    from keybo.features.ngram import trigram_direction_row

    d_gated = d_bad = 0
    for a, b, c in triples:
        gate = trigram_direction_row(G, a, b, c)
        pat = _v1_pattern(_FAM_FINGER[a], _FAM_FINGER[b], _FAM_FINGER[c])
        d_gated += bool(gate["redirect_sfgated"]) != (pat in _REDIRECT_FAMILY)
        d_bad += bool(gate["bad_redirect_sfgated"]) != (pat in _BAD_FAMILY)
    assert d_gated == 0, "redirect_sfgated IS the gauge's redirect predicate"
    assert d_bad == 0, "bad_redirect_sfgated IS the gauge's bad-redirect predicate"


# --- the map's own integrity ---------------------------------------------------------------


def test_every_annotated_column_exists_in_a_served_frame():
    """A map entry for a column no frame serves would annotate nothing, silently."""
    served = set(BIGRAM_FEATURE_NAMES) | set(TRIGRAM_FEATURE_NAMES)
    for column in GAUGE_COLLISIONS:
        assert column in served, f"{column!r} is annotated but not served"
    for column in _COLLISION_COLUMNS:
        assert column in served, f"{column!r} is annotated but not served"


def test_the_bg_prefixed_mirrors_are_generated_not_hand_written():
    """The trigram frame carries each BIGRAM-level column twice; both must inherit the annotation.

    And the TRIGRAM-level ones must NOT be mirrored: ``redirect``/``bad_redirect`` come from
    ``_trigram_level_from_positions``, so ``bg1_redirect`` is not a column of any served frame and
    annotating one would describe a name that never prints.
    """
    for column, (display, gauge, what) in GAUGE_COLLISIONS.items():
        bigram_level = f"bg1_{column}" in TRIGRAM_FEATURE_NAMES
        for prefix in ("bg1_", "bg2_"):
            mirrored = f"{prefix}{column}"
            if bigram_level:
                assert _COLLISION_COLUMNS[mirrored] == (f"{prefix}{display}", gauge, what)
                assert display_name(mirrored) == f"{prefix}{display}"
            else:
                assert mirrored not in _COLLISION_COLUMNS
                assert display_name(mirrored) == mirrored, "identity for a non-existent column"
    # the split itself, pinned so a schema move between levels is caught
    assert {c for c in GAUGE_COLLISIONS if f"bg1_{c}" in TRIGRAM_FEATURE_NAMES} == {
        "lateral",
        "lsb",
    }
    assert {c for c in GAUGE_COLLISIONS if f"bg1_{c}" not in TRIGRAM_FEATURE_NAMES} == {
        "redirect",
        "bad_redirect",
    }


def test_display_names_do_not_collide_with_a_served_column_or_each_other():
    """A display name that IS another served column's name would create a NEW collision."""
    served = set(BIGRAM_FEATURE_NAMES) | set(TRIGRAM_FEATURE_NAMES)
    displays = [display for display, _, _ in GAUGE_COLLISIONS.values()]
    assert len(displays) == len(set(displays)), "display names must be unique"
    for display in displays:
        assert display not in served, f"{display!r} is already a served column name"


def test_display_names_do_not_collide_with_a_KNOWN_OTHER_FRAME_column():
    """A display name must not be a column name used by ANOTHER frame in the campaign.

    Not hypothetical: the ``interpframe`` branch's opt-in ``interp.1`` frame serves a column named
    ``off_home_column`` that COUNTS BOTH KEYS (0/1/2), whereas the served ``lateral`` is a 0/1
    one-hot on the LANDING key — they disagree on 180 of the 900 K30 pairs. Printing ``lateral``
    as ``off_home_column`` would therefore have manufactured a FRESH collision the moment the two
    branches merged, which is the defect this whole map exists to remove.

    Pinned as a literal list rather than imported, because these frames live on a branch this one
    does not contain; the list is the interface, and a new frame must be added to it on purpose.
    """
    other_frame_columns = {
        # interp.1 (branch `interpframe`, BIGRAM_INTERP_FEATURE_NAMES)
        "hand_conflict",
        "row_span",
        "lateral_span",
        "same_hand_travel",
        "row_load",
        "row_arrival",
        "bottom_bias",
        "finger_load",
        "off_home_column",
        "roll_inward",
        # the widened / kitchen-sink frames, which THIS branch already serves
        "inwards_ordered",
        "outwards_ordered",
        "redirect_sfgated",
        "bad_redirect_sfgated",
        "half_scissor",
        "row_skip",
        "pinky_off_home",
        "weak_finger_pair",
        "finger_step",
        "onehand",
        "onehand_in",
        "red_sfs",
        "alt_sfs",
        "sg_full_scissor",
        "sg_half_scissor",
        "sg_lsb",
    }
    # The widened/kitchen-sink half of that set IS importable here, so assert the literal covers
    # it rather than trusting the transcription.
    from keybo.features.schema import (
        BIGRAM_KITCHENSINK_FEATURE_NAMES,
        TRIGRAM_KITCHENSINK_FEATURE_NAMES,
    )

    live = set(BIGRAM_KITCHENSINK_FEATURE_NAMES) | set(TRIGRAM_KITCHENSINK_FEATURE_NAMES)
    live -= set(BIGRAM_FEATURE_NAMES) | set(TRIGRAM_FEATURE_NAMES)
    live = {n.removeprefix("bg1_").removeprefix("bg2_") for n in live}
    assert live <= other_frame_columns, (
        f"unlisted live opt-in columns: {sorted(live - other_frame_columns)}"
    )

    for column, (display, _, _) in GAUGE_COLLISIONS.items():
        assert display not in other_frame_columns, (
            f"{display!r} (for served {column!r}) is a column name another frame already uses "
            "for a DIFFERENT predicate — that is a new name collision, not a fix"
        )


def test_display_names_do_not_collide_with_a_reported_gauge():
    """The whole point is to STOP colliding with gauge names — so the new names must not."""
    from keybo.analysis.redirects import REDIRECT_CLASSES
    from keybo.cli.analyze import GAUGE_NAMES
    from keybo.scoring.comfort import DEFAULT_COMFORT
    from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS

    gauges = {
        *GAUGE_NAMES,
        *REDIRECT_CLASSES,
        *DEFAULT_COMFORT,
        *DEFAULT_OXEY_WEIGHTS,
    }
    normalized = {g.replace("-", "_") for g in gauges}
    for display, _, _ in GAUGE_COLLISIONS.values():
        assert display not in gauges, f"{display!r} is a gauge name"
        assert display.replace("-", "_") not in normalized, f"{display!r} normalizes onto a gauge"


def test_notes_are_emitted_for_exactly_the_colliding_columns_present():
    """The note list must track the FRAME, not be a constant block."""
    bigram = gauge_collision_notes(BIGRAM_FEATURE_NAMES)
    assert len(bigram) == 2, "the bigram frame carries `lateral` and `lsb`"
    assert any("lateral" in n for n in bigram)
    assert any("`lsb`" in n for n in bigram)
    trigram = gauge_collision_notes(TRIGRAM_FEATURE_NAMES)
    # redirect + bad_redirect at trigram level, plus lateral/lsb under BOTH bg prefixes
    assert len(trigram) == 6
    assert gauge_collision_notes(["dx", "dy", "distance", "wpm"]) == []


def test_gauge_side_notes_mirror_the_column_side_and_name_real_gauges():
    """The gauge-side index must point at gauges ``analyze`` actually prints."""
    from keybo.cli.analyze import GAUGE_NAMES
    from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS

    printed = {*GAUGE_NAMES, *DEFAULT_OXEY_WEIGHTS}
    for gauge, column in _GAUGE_SIDE.items():
        assert gauge in printed, f"{gauge!r} is not a reported gauge"
        assert column in GAUGE_COLLISIONS, f"{gauge!r} points at an un-annotated column"
    notes = gauge_side_collision_notes(GAUGE_NAMES)
    assert len(notes) == 2, "the analyze board prints `lsb` and `lat-span`"
    assert gauge_side_collision_notes(["sfb", "alt", "comfort"]) == []


def test_every_map_entry_names_a_measured_verdict_not_an_opinion():
    """Each note must quote THE MEASURED FIRING COUNTS, not merely contain some digit.

    Asserting "contains a digit" was not enough: mutation M8 replaced the counts with vaguer prose
    that still had a digit in it and the test stayed green. So each entry's expected counts —
    themselves re-derived by the exhaustive tests above — must appear verbatim in its own note.
    """
    required = {
        "lsb": ("32", "24"),  # column firings vs gauge firings, 900 K30 pairs
        "redirect": ("4320", "2808"),  # column vs gauge, 27,000 K30 triples
        "bad_redirect": ("864", "540"),
        "lateral": ("126",),  # each-way disagreement against lat-span
    }
    assert set(required) == set(GAUGE_COLLISIONS), "a new entry needs its counts registered here"
    for column, (display, gauge, what) in GAUGE_COLLISIONS.items():
        assert display and gauge and what
        for number in required[column]:
            assert number in what, f"{column!r}'s note must quote the measured count {number}"
