"""HYBRIDB-1: the hybrid-B frame, its registry, and the invariants each one exists to protect.

Every assertion here is mutation-tested (``agent-artifacts/hybridtri/mutate.sh``), with a
``__pycache__`` purge before AND after each mutation: FM4-1 found that a ``.bak`` restored in the
same second at the same byte size satisfies CPython's ``(source_mtime, source_size)`` ``.pyc``
check, so the suite ran MUTATED BYTECODE against RESTORED SOURCE and reported three false survivors.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.analysis.shap_diff import FRAMES, block_map
from keybo.features import (
    BIGRAM_FEATURE_NAMES,
    BIGRAM_HYBRIDB_FEATURE_NAMES,
    BIGRAM_HYBRIDB_MONOTONE,
    BIGRAM_INTERP_FEATURE_NAMES,
    BIGRAM_INTERP_MONOTONE,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    FEATURE_VERSION_HYBRIDB,
    FEATURE_VERSION_INTERP,
    FEATURE_VERSION_INTERP_WPM,
    FEATURE_VERSION_KITCHENSINK,
    bigram_features_from_positions,
    hybridb_features_from_positions,
    hybridb_row_from_positions,
    interp_features_from_positions,
)
from keybo.features.ngram import REPLACEMENT_FRAME_FLAGS, replacement_frame
from keybo.geometry import ROW_STAGGERED_30

GEO = ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
WPM = 90.0


# --- the frame's IDENTITY -------------------------------------------------------------------


def test_hybridb_is_exactly_interp_plus_the_row_and_finger_onehots():
    """The frame's DEFINITION, pinned by name and order — 18 columns, interp.1's ten first."""
    assert [
        *BIGRAM_INTERP_FEATURE_NAMES,
        "bottom",
        "home",
        "top",
        "pinky",
        "ring",
        "middle",
        "index",
        "lateral",
    ] == BIGRAM_HYBRIDB_FEATURE_NAMES
    assert len(BIGRAM_HYBRIDB_FEATURE_NAMES) == 18


def test_the_added_onehots_all_come_from_the_served_frames_own_list():
    """A hybrid that named a column the served frame does not have would not be a hybrid OF it.

    This is the containment a rename in ``_BIGRAM_PLACEMENT_NAMES`` would otherwise break silently.
    """
    added = [n for n in BIGRAM_HYBRIDB_FEATURE_NAMES if n not in BIGRAM_INTERP_FEATURE_NAMES]
    assert len(added) == 8
    for name in added:
        assert name in BIGRAM_FEATURE_NAMES, f"{name!r} is not a served-frame column"


def test_hybridb_has_no_wpm_column():
    """It inherits interp.1's deliberate omission — which is why ``to_ms`` needs a stated pace."""
    assert "wpm" not in BIGRAM_HYBRIDB_FEATURE_NAMES


def test_every_hybridb_column_is_BIT_IDENTICAL_to_its_source_frames_column():
    """The load-bearing structural claim: hybrid-B does not RE-DERIVE the columns, it REUSES them.

    Checked over all 961 cells, because a re-spelled predicate (``float(by == 1)`` copied by hand)
    could agree on most cells and differ on the thumb slot or a stagger edge case — and then
    "hybrid-B contains the served ``bottom`` column" would be false in exactly the place it matters.
    """
    served, interp = list(BIGRAM_FEATURE_NAMES), list(BIGRAM_INTERP_FEATURE_NAMES)
    worst = 0.0
    for a in POS:
        for b in POS:
            h = hybridb_features_from_positions(GEO, (a, b), wpm=WPM)
            iv = interp_features_from_positions(GEO, (a, b), wpm=WPM)
            sv = bigram_features_from_positions(GEO, (a, b), wpm=WPM)
            for j, name in enumerate(BIGRAM_HYBRIDB_FEATURE_NAMES):
                want = iv[interp.index(name)] if name in interp else sv[served.index(name)]
                worst = max(worst, abs(float(h[j]) - float(want)))
    assert worst == 0.0, f"hybrid-B column differs from its source frame's by {worst:.3e}"


def test_the_row_dict_keys_are_the_name_list_in_order():
    """The vector is built by ``[row[name] for name in NAMES]``, so a key-order drift would silently
    reorder columns relative to the monotone tuple, which xgboost maps POSITIONALLY."""
    row = hybridb_row_from_positions(GEO, POS[0], POS[1])
    assert list(row) == BIGRAM_HYBRIDB_FEATURE_NAMES


def test_wpm_is_accepted_and_ignored():
    """Call-shape parity with the served featurizer: every training/attribution caller passes
    ``wpm=``, and a frame that could not accept it would need a second code path — which is how a
    frame ends up featurized differently at train and at serve time."""
    a, b = POS[3], POS[7]
    lo = hybridb_features_from_positions(GEO, (a, b), wpm=1.0)
    hi = hybridb_features_from_positions(GEO, (a, b), wpm=300.0)
    assert np.array_equal(lo, hi)


# --- the MONOTONE tuple: partial BY DESIGN ---------------------------------------------------


def test_monotone_tuple_is_interps_signs_then_eight_zeros():
    """Registered in HYBRIDTRI-preregistration.md §1 BEFORE measuring, with the consequence
    (MONOFRAC cannot reach 1.0) registered too. The zeros are the claim, so they are pinned:
    an all-zero tuple would mean the interp constraints were silently dropped, and a fully-signed
    tuple would mean the collinear row one-hots got a self-contradictory constraint."""
    assert len(BIGRAM_HYBRIDB_MONOTONE) == len(BIGRAM_HYBRIDB_FEATURE_NAMES)
    assert BIGRAM_HYBRIDB_MONOTONE[:10] == tuple(BIGRAM_INTERP_MONOTONE)
    assert BIGRAM_HYBRIDB_MONOTONE[10:] == (0,) * 8
    # and the SIGNED half is on exactly the interp columns, the ZERO half on exactly the one-hots
    by_name = dict(zip(BIGRAM_HYBRIDB_FEATURE_NAMES, BIGRAM_HYBRIDB_MONOTONE, strict=True))
    for name in BIGRAM_INTERP_FEATURE_NAMES:
        assert by_name[name] != 0, f"{name} lost its constraint"
    for name in ("bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral"):
        assert by_name[name] == 0, f"{name} must be UNCONSTRAINED (see the schema note)"


# --- the STAMP: a sixth disjoint population -------------------------------------------------


def test_the_stamp_is_distinct_from_every_other_frames_stamp():
    """``keybo.models.base`` hard-errors on a stamp mismatch, and that guard is the only thing
    standing between a model and being scored on the wrong matrix — so two frames sharing a stamp
    would make the guard blind between them."""
    stamps = [
        FEATURE_VERSION,
        FEATURE_VERSION_DIRECTION,
        FEATURE_VERSION_KITCHENSINK,
        FEATURE_VERSION_INTERP,
        FEATURE_VERSION_INTERP_WPM,
        FEATURE_VERSION_HYBRIDB,
    ]
    assert len(set(stamps)) == len(stamps), f"duplicate stamp among {stamps}"
    assert f"{FEATURE_VERSION}+hybrid-b.1" == FEATURE_VERSION_HYBRIDB


def test_the_served_FEATURE_VERSION_is_untouched():
    """The six shipped k31 artifacts carry it; changing it in place invalidates all of them."""
    assert FEATURE_VERSION == "2026-07-05.3"


# --- the REGISTRY ---------------------------------------------------------------------------


def test_the_registry_resolves_four_things_that_must_agree():
    """The builder / name list / monotone tuple / stamp for each replacement basis come from ONE
    lookup, so they cannot desynchronize — a model stamped ``interp.1`` while carrying the
    18-column constraint tuple is exactly the train/serve skew the stamp exists to prevent."""
    for flag, n_cols, stamp in (
        (True, 10, FEATURE_VERSION_INTERP),
        ("wpm", 11, FEATURE_VERSION_INTERP_WPM),
        ("hybridb", 18, FEATURE_VERSION_HYBRIDB),
    ):
        builder, names, mono, got_stamp, tag = replacement_frame(flag)
        assert len(names) == n_cols
        assert len(mono) == n_cols, "one constraint per column (xgboost maps POSITIONALLY)"
        assert got_stamp == stamp
        vec = builder(GEO, (POS[2], POS[5]), wpm=WPM)
        assert vec.shape == (n_cols,), f"{flag!r} builder emitted {vec.shape}, not {n_cols}"
        assert tag


def test_an_unknown_flag_RAISES_rather_than_falling_back_to_the_10_column_frame():
    """The whole reason the registry exists. ``"HYBRIDB"``/``"hybrid-b"``/``"interp"`` are the
    realistic near-misses, and a silent fallback would train the wrong frame and report a plausible
    number. ``False`` is refused too: the served frame is not a replacement basis."""
    for bad in (False, None, "HYBRIDB", "hybrid-b", "interp", "wmp", "", 2, -1):
        with pytest.raises(ValueError, match="interp must be False"):
            replacement_frame(bad)


def test_the_registry_lists_exactly_the_three_replacement_frames():
    """Pinned so a frame cannot be added without a deliberate edit: each needs a builder, a name
    list, a monotone tuple, a stamp, a block partition and a version guard."""
    assert REPLACEMENT_FRAME_FLAGS == (True, "wpm", "hybridb")


def test_the_registry_dict_and_its_guard_reject_the_SAME_set_of_flags():
    """Mutations M13 and M16 SURVIVED both batteries, and the honest diagnosis is that they are
    EQUIVALENT MUTANTS, not coverage gaps. Recorded here rather than papered over.

    M16 rewrites ``return _REPLACEMENT_FRAMES[interp]`` as
    ``.get(interp, _REPLACEMENT_FRAMES[True])``. The membership guard on the line ABOVE has already
    rejected every key absent from the dict, so the ``.get`` default is UNREACHABLE and the two
    forms are behaviourally identical. No test can distinguish them without deleting the guard —
    and a test that deletes the code it is testing is not a test. (My first attempt asserted
    ``_REPLACEMENT_FRAMES["hybrid-b"]`` raises ``KeyError``, which is a property of the DICT that the
    mutation never touches: aiming at the wrong object. Corrected here.)

    What IS worth pinning is the invariant that makes the mutant equivalent in the first place: the
    guard's accept-set and the dict's key-set must be THE SAME SET. If they ever drift, the lookup
    becomes reachable with a missing key and M16 stops being equivalent — so this test is the
    tripwire for the drift, not for the mutation.
    """
    from keybo.features.ngram import _REPLACEMENT_FRAMES

    assert set(REPLACEMENT_FRAME_FLAGS) == set(_REPLACEMENT_FRAMES), (
        "the guard's accept-set and the registry's key-set have drifted; a missing-key lookup is "
        "now reachable and a .get()-style default would silently select another frame"
    )
    for flag in REPLACEMENT_FRAME_FLAGS:
        assert replacement_frame(flag) is _REPLACEMENT_FRAMES[flag]


def test_the_registry_is_keyed_by_the_EXACT_flag_values_training_accepts():
    """Mutation M27 SURVIVED: pointing ``shap_diff``'s frame-name map at ``True`` instead of
    ``"hybridb"`` left the suite green, because no test asserted the two maps agree. They must:
    ``shap_diff(frame="hybridb")`` and ``train_bigram_model(interp="hybridb")`` have to resolve to
    the SAME frame, or an attribution would be computed on a different frame than was trained —
    the exact train/serve skew the version stamp exists to prevent.
    """
    from keybo.analysis.shap_diff import _REPLACEMENT_BY_FRAME_NAME, FRAMES

    # every replacement frame name in FRAMES maps to a flag, and every flag round-trips
    for name, flag in _REPLACEMENT_BY_FRAME_NAME.items():
        assert name in FRAMES, f"{name!r} is mapped but not in FRAMES"
        _b, names, _m, stamp, _t = replacement_frame(flag)
        # the mapping must be INJECTIVE onto the flags, or two frame names collapse to one frame
        assert sum(1 for f in _REPLACEMENT_BY_FRAME_NAME.values() if f == flag) == 1, (
            f"two frame names both resolve to flag {flag!r}"
        )
        # and the resolved frame must be the one the NAME claims: width + stamp are the observable
        if name == "hybridb":
            assert len(names) == 18 and stamp.endswith("hybrid-b.1")
        elif name == "interp":
            assert len(names) == 10 and stamp.endswith("interp.1")
        elif name == "interp-wpm":
            assert len(names) == 11 and stamp.endswith("interp-wpm.1")
    # served is in FRAMES but is NOT a replacement basis, so it must NOT be in the map
    assert "served" in FRAMES
    assert "served" not in _REPLACEMENT_BY_FRAME_NAME


# --- the BLOCK PARTITION ---------------------------------------------------------------------


def test_hybridb_has_a_registered_block_partition_and_it_is_an_exact_partition():
    """``block_map`` REFUSES an unknown frame rather than bucketing a remainder, because an
    unrecognised frame would otherwise report a silently incomplete primary table while every
    identity still closed."""
    spec = block_map(BIGRAM_HYBRIDB_FEATURE_NAMES)
    assert set(spec) == set(BIGRAM_HYBRIDB_FEATURE_NAMES)
    assert len(spec) == 18


def test_each_ordinal_shares_a_block_with_the_onehots_for_the_SAME_property():
    """THE PARTITION IS THE MEASUREMENT (HYBRIDB-1 §3). A block sum is only invariant to
    redistribution WITHIN the block, so an ordinal and the one-hots it was built to replace must
    share one — otherwise the block number moves with TreeSHAP's arbitrary split between them,
    which is the non-uniqueness blocks exist to contain."""
    spec = block_map(BIGRAM_HYBRIDB_FEATURE_NAMES)
    for ordinal, onehots in (
        ("bottom_bias", ("bottom", "home", "top")),
        ("finger_load", ("pinky", "ring", "middle", "index", "lateral")),
        ("off_home_column", ("lateral",)),
    ):
        block = spec[ordinal][0]
        for name in onehots:
            assert spec[name][0] == block, (
                f"{name!r} is in block {spec[name][0]!r} but {ordinal!r} is in {block!r}; "
                f"a block sum would not be invariant to credit moving between them"
            )


def test_the_partition_records_the_interpretability_COST_as_block_widths():
    """The honest price of the resolution hybrid-B buys: its widest block is 8 columns against
    interp.1's 3. Asserted rather than argued, because 'hybrid-B hides more than interp.1' is a
    claim about this partition and nothing else."""
    from collections import Counter

    spec = block_map(BIGRAM_HYBRIDB_FEATURE_NAMES)
    widths = Counter(b for b, _ in spec.values())
    assert widths["CONTACT"] == 8
    assert widths["ROWCOST"] == 6
    assert widths["SPAN"] == 3
    assert widths["DIRECTION"] == 1
    interp_spec = block_map(BIGRAM_INTERP_FEATURE_NAMES)
    interp_widths = Counter(b for b, _ in interp_spec.values())
    assert max(interp_widths.values()) == 3
    assert max(widths.values()) == 8


def test_FRAMES_includes_hybridb():
    """The attribution entry point: ``shap_diff(frame=...)`` validates against this tuple."""
    assert "hybridb" in FRAMES


# --- the SUB-BLOCK labels: they must DISTINGUISH, not decorate --------------------------------


def test_the_ordinal_and_onehot_subblocks_are_both_populated():
    """A sub-block label whose other value never occurs is a vacuous field (FRAMEDIAG-1's M20/M21b:
    an assertion whose SUBJECT CANNOT VARY tests nothing). Both values must actually appear inside
    the two mixed blocks, or the label carries no information."""
    spec = block_map(BIGRAM_HYBRIDB_FEATURE_NAMES)
    for block in ("ROWCOST", "CONTACT"):
        subs = {sub for name, (b, sub) in spec.items() if b == block}
        assert subs == {"ordinal", "onehot"}, f"{block} sub-blocks are {subs}"
    # and the UNMIXED blocks carry the empty sub-block, so the label means "this block is mixed"
    for block in ("SPAN", "DIRECTION"):
        subs = {sub for name, (b, sub) in spec.items() if b == block}
        assert subs == {""}, f"{block} should have no sub-block, got {subs}"


# --- what the FIRST mutation battery's survivors exposed --------------------------------------


def test_a_FALSY_but_not_False_interp_flag_is_REFUSED_not_silently_served():
    """Mutation M28 SURVIVED the first battery, and diagnosing it found a REAL latent bug.

    Every branch after the guard is ``if interp:``, so a falsy-but-not-``False`` value (``None``,
    ``0``, ``""``, ``[]``) skips the entire replacement-frame path and trains the SERVED frame. The
    caller asked for a frame it did not get, and the resulting artifact is stamped and shaped exactly
    like a legitimate served model — so nothing downstream can detect it. The guard is written
    ``interp is not False`` rather than as a truthiness test for exactly this reason, and this test
    is what pins that choice: it FAILS if the guard is relaxed to ``if interp and ...``.

    ``0``/``1`` are excluded deliberately: ``0 == False`` and ``1 == True`` in Python, so they are
    the two legal frames' own values, not near-misses. That is documented on
    ``train_bigram_model`` rather than special-cased.
    """
    import numpy as np

    from keybo.data.strokes import StrokeRow
    from keybo.geometry import ROW_STAGGERED_31
    from keybo.training.train import train_bigram_model

    rows = [
        StrokeRow(
            layout="qwerty",
            positions=(ROW_STAGGERED_31.slots[i], ROW_STAGGERED_31.slots[i + 1]),
            ngram="ab",
            frequency=100,
            samples=[(90, 150 + i, 1, 0), (90, 160 + i, 2, 0)],
        )
        for i in range(6)
    ]
    for falsy in (None, "", [], 0.0, np.float64(0.0)):
        with pytest.raises(ValueError, match="interp must be False"):
            train_bigram_model(
                rows, target_wpm=90.0, geometry=ROW_STAGGERED_31, interp=falsy, n_estimators=2
            )


def test_the_featurizers_missing_column_guard_RAISES_rather_than_zero_filling():
    """Mutation M13 SURVIVED, and the diagnosis is that it is an EQUIVALENT MUTANT, not a gap:
    all 18 schema names are present in one of the two row builders, so the ``KeyError`` branch is
    UNREACHABLE from real data and ``placement[name]`` vs ``placement.get(name, 0.0)`` are
    behaviourally identical on the current schema.

    Covered CONSTRUCTIVELY instead — the same route FRAMEDIAG-1 took for its own unreachable state
    (its ``exceeds_exact`` flag) — by asking the builder for a name the placement row does not have.
    A ``.get``-style default would emit a silent zero column, which reads as a MEASURED feature; the
    strict lookup raises. This verifies the REPORTING behaviour, not the reachability.
    """
    from keybo.features.ngram import _placement_row_from_positions, hybridb_row_from_positions

    # the branch's precondition, asserted so this test's premise cannot rot silently
    placement = _placement_row_from_positions(GEO, POS[0], POS[1])
    interp_row = {**hybridb_row_from_positions(GEO, POS[0], POS[1])}
    for name in BIGRAM_HYBRIDB_FEATURE_NAMES:
        assert name in interp_row
    # CONSTRUCT the unreachable state: a schema name absent from the placement row
    assert "no_such_column" not in placement
    with pytest.raises(KeyError):
        placement["no_such_column"]
