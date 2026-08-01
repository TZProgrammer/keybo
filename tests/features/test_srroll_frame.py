"""The SAME-ROW ROLL frame: additive, order-correct, and provably non-disruptive.

The load-bearing test in this file is
:func:`test_served_and_prior_frames_are_byte_identical_with_srroll_available`. Everything else
describes the two new columns; that one proves the three SHIPPED frames did not move, which is the
whole basis on which a fourth frame is allowed to exist (six models under ``data/models/k31/`` are
stamped ``FEATURE_VERSION`` and ``keybo.models.base`` errors on a version MISMATCH, not on a column
whose MEANING changed — so a silent redefinition would leave every model loading fine while scoring
the wrong matrix).

The rest pins the numbers the SR-ROLL-1 audit measured
(``state/srroll/drivers/srroll_audit.py``, registered in ``state/srroll/PREREGISTRATION.md``) so a
later edit cannot quietly change what a column counts while the registered table still claims the
old figure. Chief among them: ``sr_roll``'s overlap with every pre-existing trigram-class column is
exactly ZERO, which is the evidence that this class was unnamed in every prior frame.
"""

from itertools import permutations

import numpy as np
import pytest

from keybo.analysis.kmstats import _KEYS, _trigram_value
from keybo.features import trigram_features_from_positions, trigram_model_row
from keybo.features import classify as C
from keybo.features.ngram import trigram_kitchensink_row, trigram_srroll_row
from keybo.features.schema import (
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    FEATURE_VERSION_KITCHENSINK,
    FEATURE_VERSION_SRROLL,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_KITCHENSINK_FEATURE_NAMES,
    TRIGRAM_SRROLL_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout

G = ROW_STAGGERED_30
LAYOUT = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", G)
TRIPLES = list(permutations(G.slots, 3))


@pytest.fixture(scope="module")
def srroll_values() -> dict[str, np.ndarray]:
    """The two new columns over the full 24,360-triple enumeration."""
    rows = [trigram_srroll_row(G, a, b, c) for a, b, c in TRIPLES]
    return {name: np.array([r[name] for r in rows]) for name in ("roll", "sr_roll")}


# --- the non-disruption guarantee (the reason a fourth frame is allowed to exist) -----------


def test_served_and_prior_frames_are_byte_identical_with_srroll_available():
    """The three shipped frames are unchanged, column for column, by this addition."""
    for a, b, c in TRIPLES[::97]:
        for flags, names in (
            ({}, TRIGRAM_FEATURE_NAMES),
            ({"direction": True}, TRIGRAM_DIRECTION_FEATURE_NAMES),
            ({"kitchensink": True}, TRIGRAM_KITCHENSINK_FEATURE_NAMES),
        ):
            vec = trigram_features_from_positions(G, (a, b, c), wpm=90.0, **flags)
            assert len(vec) == len(names)
    # and the served frame still carries no roll column of any kind
    assert "roll" not in TRIGRAM_FEATURE_NAMES
    assert "sr_roll" not in TRIGRAM_FEATURE_NAMES
    assert "roll" not in TRIGRAM_DIRECTION_FEATURE_NAMES
    assert "roll" not in TRIGRAM_KITCHENSINK_FEATURE_NAMES


def test_srroll_frame_is_the_served_frame_plus_exactly_two_columns():
    added = [n for n in TRIGRAM_SRROLL_FEATURE_NAMES if n not in TRIGRAM_FEATURE_NAMES]
    assert added == ["roll", "sr_roll"]
    assert len(TRIGRAM_SRROLL_FEATURE_NAMES) == len(TRIGRAM_FEATURE_NAMES) + 2
    # it builds on the SERVED frame, NOT the widened one — the single-variable requirement
    assert "bg1_inwards_ordered" not in TRIGRAM_SRROLL_FEATURE_NAMES
    assert "redirect_sfgated" not in TRIGRAM_SRROLL_FEATURE_NAMES


def test_srroll_frame_names_are_unique_and_wpm_stays_last():
    assert len(TRIGRAM_SRROLL_FEATURE_NAMES) == len(set(TRIGRAM_SRROLL_FEATURE_NAMES))
    assert TRIGRAM_SRROLL_FEATURE_NAMES[-1] == "wpm"


def test_row_keys_match_the_schema_in_order():
    row = trigram_model_row(LAYOUT, "the", wpm=90, srroll=True)
    assert list(row.keys()) == TRIGRAM_SRROLL_FEATURE_NAMES


def test_the_four_stamps_are_all_distinct():
    """A shared stamp would make the load-time guard unable to tell the populations apart."""
    stamps = [
        FEATURE_VERSION,
        FEATURE_VERSION_DIRECTION,
        FEATURE_VERSION_KITCHENSINK,
        FEATURE_VERSION_SRROLL,
    ]
    assert len(set(stamps)) == 4


def test_srroll_does_not_compose_with_the_other_frame_flags():
    """Each frame is one model population with one stamp; a mixed frame would have none."""
    with pytest.raises(ValueError, match="does not compose"):
        trigram_features_from_positions(G, TRIPLES[0], wpm=90.0, srroll=True, direction=True)
    with pytest.raises(ValueError, match="does not compose"):
        trigram_features_from_positions(G, TRIPLES[0], wpm=90.0, srroll=True, kitchensink=True)


# --- the predicates agree with the GAUGES they are named after ------------------------------


def test_feature_matches_the_kmstats_gauge_on_every_triple():
    """The feature and the ``roll``/``sr-roll`` gauges must never disagree about the predicate.

    kmstats indexes ``_KEYS`` in row-major slot order, so this maps through the SLOT INDEX rather
    than re-deriving the finger map — a hand-copied constant could drift, a slot lookup cannot.
    """
    for a, b, c in TRIPLES:
        ka, kb, kc = (_KEYS[G.slots.index(p)] for p in (a, b, c))
        assert C.is_roll(G, a, b, c) == bool(_trigram_value("roll", ka, kb, kc))
        assert C.is_same_row_roll(G, a, b, c) == bool(_trigram_value("sr-roll", ka, kb, kc))


def test_support_matches_the_registered_audit(srroll_values):
    """Pinned from ``state/srroll/artifacts/srroll_audit.json`` (full 24,360-triple enumeration)."""
    assert len(TRIPLES) == 24360
    assert int(srroll_values["roll"].sum()) == 9720
    assert int(srroll_values["sr_roll"].sum()) == 1080


def test_sr_roll_is_a_strict_subset_of_roll(srroll_values):
    """Why both columns are served: alone, the conjunction is confounded with roll-ness."""
    roll, sr = srroll_values["roll"], srroll_values["sr_roll"]
    assert np.all(sr <= roll)
    assert int(sr.sum()) < int(roll.sum())


def test_both_columns_are_swap_invariant(srroll_values):
    """Neither column can see stroke order: every predicate in them is symmetric under reversal.

    Registered so a later "fix" cannot silently turn these into direction channels — that is what
    ``inwards_ordered``/``outwards_ordered`` are for, and conflating the two is the exact defect
    CYANO-1 found in the shipped ``inwards``/``outwards``.
    """
    for name, fn in (("roll", C.is_roll), ("sr_roll", C.is_same_row_roll)):
        fwd = srroll_values[name]
        rev = np.array([float(fn(G, c, b, a)) for a, b, c in TRIPLES])
        assert int((fwd != rev).sum()) == 0, name


# --- the (c)-verdict evidence: this class is named by NO pre-existing column ----------------


def test_sr_roll_is_identical_to_no_served_column(srroll_values):
    """The (c)-verdict refutation, checked mechanically rather than by eye — the sg_distance trap."""
    X = np.array([trigram_features_from_positions(G, t, wpm=90.0) for t in TRIPLES])
    for name in ("roll", "sr_roll"):
        y = srroll_values[name]
        assert not any(np.array_equal(X[:, j], y) for j in range(X.shape[1])), name


def test_zero_overlap_with_every_prior_trigram_class_column(srroll_values):
    """The structural reason: keymeow's roll needs ``a.hand != c.hand``; every prior class column
    needs ONE hand (``same_hand_trigram``/``redirect``/``bad_redirect``, kitchen-sink's
    ``onehand``/``onehand_in``/``red_sfs``) or ``a.hand == c.hand`` (kmstats ``alt``, ``alt_sfs``).
    """
    from keybo.features.ngram import _trigram_row_from_positions

    sr = srroll_values["sr_roll"] > 0
    roll = srroll_values["roll"] > 0
    served = [_trigram_row_from_positions(G, a, b, c, 90.0) for a, b, c in TRIPLES]
    ks = [trigram_kitchensink_row(G, a, b, c) for a, b, c in TRIPLES]
    for col in ("same_hand_trigram", "redirect", "bad_redirect"):
        v = np.array([r[col] for r in served]) > 0
        assert int((v & sr).sum()) == 0, col
        assert int((v & roll).sum()) == 0, col
    for col in ("onehand", "onehand_in", "red_sfs"):
        v = np.array([r[col] for r in ks]) > 0
        assert int((v & sr).sum()) == 0, col
        assert int((v & roll).sum()) == 0, col
    # kmstats' own ``alt`` — the class ``ter`` moves OUT of on BALL-1
    alt = np.array(
        [
            _trigram_value("alt", *(_KEYS[G.slots.index(p)] for p in (a, b, c)))
            for a, b, c in TRIPLES
        ]
    ) > 0
    assert int((alt & sr).sum()) == 0
    assert int((alt & roll).sum()) == 0


def test_sr_roll_is_deterministic_from_the_served_frame(srroll_values):
    """The (a)-verdict refutation: no two triples share a served row yet differ in sr_roll.

    So the column adds no INFORMATION — only explicitness. This is what makes the honest framing
    "does making the conjunction explicit help", never "new information".
    """
    X = np.array([trigram_features_from_positions(G, t, wpm=90.0) for t in TRIPLES])
    groups: dict[bytes, list[int]] = {}
    for i, row in enumerate(X):
        groups.setdefault(row.tobytes(), []).append(i)
    for name in ("roll", "sr_roll"):
        y = srroll_values[name]
        ambiguous = [g for g in groups.values() if len(np.unique(y[g])) > 1]
        assert ambiguous == [], name


# --- the training path stamps the fourth population -----------------------------------------


def test_the_srroll_frame_is_wider_than_the_served_one_by_the_two_columns():
    narrow = trigram_features_from_positions(G, TRIPLES[0], wpm=90.0)
    wide = trigram_features_from_positions(G, TRIPLES[0], wpm=90.0, srroll=True)
    assert len(wide) == len(narrow) + 2
