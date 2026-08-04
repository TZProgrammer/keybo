"""The n-gram feature pipeline: (layout, n-gram) -> feature row.

This is the ONE place features are computed. Data processing, model training, and layout
scoring all call these functions, so the features a model is trained on are exactly the
features it is later scored with. Rows are returned as ordered dicts keyed by the names in
:mod:`keybo.features.schema`; :func:`bigram_features` / :func:`trigram_features` return the
same values as a plain float vector for the model.

Frequency is NOT an input here (OQ-1, 2026-07-05): features are pure geometry + wpm.
Frequency enters the system only as the objective weight and as the identity key of the
additive practice term (see :mod:`keybo.training.train`).
"""

from __future__ import annotations

import numpy as np

from keybo.features import classify as C
from keybo.features.schema import (
    BIGRAM_DIRECTION_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES,
    BIGRAM_HYBRIDB_FEATURE_NAMES,
    BIGRAM_HYBRIDB_MONOTONE,
    BIGRAM_INTERP_FEATURE_NAMES,
    BIGRAM_INTERP_MONOTONE,
    BIGRAM_INTERP_WPM_FEATURE_NAMES,
    BIGRAM_INTERP_WPM_MONOTONE,
    BIGRAM_KITCHENSINK_FEATURE_NAMES,
    FEATURE_VERSION_HYBRIDB,
    FEATURE_VERSION_INTERP,
    FEATURE_VERSION_INTERP_WPM,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_KITCHENSINK_FEATURE_NAMES,
)
from keybo.geometry import Geometry, Position
from keybo.layout import Layout


def _placement_row_from_positions(
    geometry: Geometry,
    a: Position,
    b: Position,
    direction: bool = False,
    kitchensink: bool = False,
) -> dict[str, float]:
    """The placement/relational/geometry features for one bigram, from key positions.

    Positions are the fundamental input: both scoring (positions from a layout) and training
    (positions recorded in the data) route through here, so the two can never diverge.

    ``direction=False`` (the default) produces exactly the served frame, byte for byte —
    guarded by the frozen golden matrix in ``tests/features/test_k31_geometry.py``.
    ``direction=True`` appends the two ORDER-AWARE roll columns
    (:data:`~keybo.features.schema.BIGRAM_DIRECTION_FEATURE_NAMES`).

    The opt-in exists because the served ``inwards``/``outwards`` columns are swap-invariant
    (0 of 870 ordered pairs change under reversal) and cannot be fixed in place: six shipped
    models are stamped with the current ``FEATURE_VERSION`` and would keep loading while
    scoring on a frame whose columns had silently changed meaning. See
    :mod:`keybo.features.schema`.

    ``kitchensink=True`` appends the five external-project bigram columns on top of that
    (:data:`~keybo.features.schema.BIGRAM_KITCHENSINK_FEATURE_NAMES`). It IMPLIES ``direction``:
    the kitchen-sink frame is defined as the widened frame plus this block, so the two flags
    compose into three legal frames (narrow, widened, kitchen-sink) rather than four — there is
    no "kitchen-sink without direction" model population and no stamp for one.
    """
    g = geometry
    bx, by = b
    cls = C.classify_positions(g, a, b)
    abs_bx = abs(bx)

    row = {
        # second-key row one-hot
        "bottom": float(by == 1),
        "home": float(by == 2),
        "top": float(by == 3),
        # second-key finger one-hot (index = columns 1 and 2; K31 pinky = 5 and 6)
        "pinky": float(abs_bx in (5, 6)),
        "ring": float(abs_bx == 4),
        "middle": float(abs_bx == 3),
        "index": float(abs_bx in (1, 2)),
        "lateral": float(C.is_lateral(bx)),
        # relational
        "same_hand": float(cls is not C.BigramClass.ALTERNATE),
        "same_finger": float(cls is C.BigramClass.SAME_FINGER),
        "adjacent": float(C.is_adjacent(g, a, b)),
        "scissor": float(C.is_scissor(g, a, b)),
        "lsb": float(C.is_lsb(g, a, b)),
        # geometry
        "dx": g.stagger_adjusted_dx(a, b),
        "dy": float(abs(a[1] - b[1])),
        "distance": g.distance(a, b),
        "angle": C.rotation_angle(g, a, b),
        # ⚠ swap-INVARIANT (see keybo.features.classify): these two describe the key PAIR,
        # not the stroke. The direction-of-travel channel is the opt-in block below.
        "inwards": float(C.is_inwards(g, a, b)),
        "outwards": float(C.is_outwards(g, a, b)),
    }
    if direction or kitchensink:
        row["inwards_ordered"] = float(C.is_inwards_ordered(g, a, b))
        row["outwards_ordered"] = float(C.is_outwards_ordered(g, a, b))
    if kitchensink:
        # Key ORDER matters: the schema puts this block straight after the direction columns and
        # before wpm, and a test pins list(row) == the name list.
        row["half_scissor"] = float(C.is_half_scissor(g, a, b))
        row["row_skip"] = float(C.is_row_skip(g, a, b))
        row["pinky_off_home"] = float(C.is_pinky_off_home(g, a, b))
        row["weak_finger_pair"] = float(C.is_weak_finger_pair(g, a, b))
        row["finger_step"] = C.finger_step(g, a, b)
    return row


def _placement_row(
    layout: Layout, bigram: str, direction: bool = False, kitchensink: bool = False
) -> dict[str, float]:
    """Placement features for a bigram on a layout (looks up positions, then delegates)."""
    return _placement_row_from_positions(
        layout.geometry,
        layout.pos(bigram[0]),
        layout.pos(bigram[1]),
        direction=direction,
        kitchensink=kitchensink,
    )


def _bigram_column_names(direction: bool, kitchensink: bool = False) -> list[str]:
    """The canonical column order for the frame these flags select."""
    if kitchensink:
        return BIGRAM_KITCHENSINK_FEATURE_NAMES
    return BIGRAM_DIRECTION_FEATURE_NAMES if direction else BIGRAM_FEATURE_NAMES


def bigram_model_row(
    layout: Layout,
    bigram: str,
    wpm: float,
    direction: bool = False,
    kitchensink: bool = False,
) -> dict[str, float]:
    """Full ordered bigram feature row (placement features + wpm)."""
    row = _placement_row(layout, bigram, direction=direction, kitchensink=kitchensink)
    row["wpm"] = float(wpm)
    return row


def bigram_features(
    layout: Layout,
    bigram: str,
    wpm: float = 0.0,
    direction: bool = False,
    kitchensink: bool = False,
) -> np.ndarray:
    """Bigram feature vector in canonical column order."""
    row = bigram_model_row(layout, bigram, wpm, direction=direction, kitchensink=kitchensink)
    return np.array(
        [row[name] for name in _bigram_column_names(direction, kitchensink)], dtype=np.float64
    )


def bigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position],
    wpm: float = 0.0,
    direction: bool = False,
    kitchensink: bool = False,
) -> np.ndarray:
    """Bigram feature vector from recorded key positions (training path)."""
    row = _placement_row_from_positions(
        geometry, positions[0], positions[1], direction=direction, kitchensink=kitchensink
    )
    row["wpm"] = float(wpm)
    return np.array(
        [row[name] for name in _bigram_column_names(direction, kitchensink)], dtype=np.float64
    )


# --- the INTERPRETABILITY frame (INTERPFRAME-1) -------------------------------------------
#
# A REPLACEMENT basis for the served bigram columns, not an addition: 10 columns instead of 20,
# chosen so a per-feature SHAP number means what its name says. See
# :data:`keybo.features.schema.BIGRAM_INTERP_FEATURE_NAMES` for which failure mode each column
# fixes, and ``agent-artifacts/interpframe/INTERPFRAME-preregistration.md`` §4 for the design.
#
# Emitted by its OWN function rather than as a flag on ``_placement_row_from_positions``: every
# other frame in this module is that function's output PLUS extra keys, so a fourth flag there
# would have to SUBTRACT columns — and a subtracting flag on the function that feeds the
# version-locked served frame is exactly the shape of edit that silently changes a shipped
# frame. A separate function cannot.

#: Absolute column -> off-home stretch column. Mirrors
#: :data:`keybo.features.classify._HOME_COLUMN`'s reading of columns 1 and 6 as the index's and
#: pinky's off-home columns, kept as its own literal so a change to the graded lateral-span table
#: cannot silently re-define this frame's ``off_home_column``.
_OFF_HOME_COLUMNS = (1, 6)

#: The home row's ``y``. Named rather than inlined, because ``row_load``, ``row_arrival`` and
#: ``bottom_bias`` must all measure deviation from the SAME origin or the 45-degree rotation that
#: makes ``row_load``/``row_arrival`` orthogonal does not hold.
_HOME_ROW_Y = 2


def _is_letter_key(position: Position) -> bool:
    """Whether a position is an assignable letter key rather than the thumb/space slot.

    Space is at ``(0, 0)`` and pressed by the thumb, which has no home column and no finger rank
    (``Geometry.hand(0) == 0``; ``classify.finger_kind`` returns -1). Every per-KEY term in the
    interp frame therefore contributes 0 for it — written as ONE predicate rather than re-derived
    per column, so the treatment cannot drift between columns.
    """
    return position[0] != 0


def interp_row_from_positions(geometry: Geometry, a: Position, b: Position) -> dict[str, float]:
    """The ten interpretability-frame features for one ORDERED bigram ``a -> b``.

    Returns an ordered dict keyed by :data:`~keybo.features.schema.BIGRAM_INTERP_FEATURE_NAMES`
    exactly (a test pins ``list(row) == the name list``).

    ⚠ There is NO ``wpm`` key, deliberately — see the schema note. A caller that needs a
    WPM-spanning model wants a different frame.
    """
    g = geometry
    cls = C.classify_positions(g, a, b)
    same_hand = cls is not C.BigramClass.ALTERNATE
    two_finger = same_hand and cls is not C.BigramClass.SAME_FINGER

    # Per-key terms, zero for the thumb/space slot (see _is_letter_key).
    dev_a = float(abs(a[1] - _HOME_ROW_Y)) if _is_letter_key(a) else 0.0
    dev_b = float(abs(b[1] - _HOME_ROW_Y)) if _is_letter_key(b) else 0.0
    # float() on every sum: a bare ``sum(1.0 for ...)`` over an EMPTY generator returns the int 0,
    # so a space-only pair would emit ints where every other pair emits floats. Harmless in the
    # numpy vector, but the dict is also read directly (by the report and by tests), and a column
    # whose dtype depends on its value is the kind of thing that reads as a bug later.
    bottom = float(sum(1 for p in (a, b) if _is_letter_key(p) and p[1] < _HOME_ROW_Y))
    top = float(sum(1 for p in (a, b) if _is_letter_key(p) and p[1] > _HOME_ROW_Y))
    # 3 - finger_kind: index 0 (strongest) .. pinky 3 (weakest), so the column RISES with weakness
    # and +1 is the mechanism. finger_kind returns -1 for the thumb, which the letter-key gate
    # excludes before it could contribute a spurious 4.
    weakness = float(sum(3 - C.finger_kind(g, p[0]) for p in (a, b) if _is_letter_key(p)))

    return {
        "hand_conflict": float(0 if not same_hand else (1 if two_finger else 2)),
        # Gated on TWO-FINGER same-hand: a single finger travelling between its own rows is a
        # same-finger reach already priced by ``hand_conflict``, and a cross-hand pair spans no row
        # at all (the two hands move independently). Ungated, this column would fire on cross-hand
        # pairs and stop being a hand-contortion measure — the same error DIST-1 caught in its own
        # first widening convention.
        "row_span": float(abs(a[1] - b[1])) if two_finger else 0.0,
        "lateral_span": C.lateral_span(g, a, b),
        "same_hand_travel": g.distance(a, b) if same_hand else 0.0,
        "row_load": dev_a + dev_b,
        "row_arrival": dev_b - dev_a,
        "bottom_bias": bottom - top,
        "finger_load": weakness,
        "off_home_column": float(
            sum(1 for p in (a, b) if _is_letter_key(p) and abs(p[0]) in _OFF_HOME_COLUMNS)
        ),
        # +1 inward / -1 outward / 0 not roll-eligible. The two ordered predicates partition the
        # eligible set exactly (162/162 on K30), so ONE signed column loses nothing.
        "roll_inward": (
            1.0
            if C.is_inwards_ordered(g, a, b)
            else (-1.0 if C.is_outwards_ordered(g, a, b) else 0.0)
        ),
    }


def interp_wpm_row_from_positions(
    geometry: Geometry, a: Position, b: Position, wpm: float
) -> dict[str, float]:
    """:func:`interp_row_from_positions` plus ``wpm`` — the pace-adapting variant (§11).

    Exists because the 10-column frame's high-wpm cost traces to exactly one dropped column; see
    :data:`keybo.features.schema.BIGRAM_INTERP_WPM_FEATURE_NAMES` for the trade it makes.
    """
    row = interp_row_from_positions(geometry, a, b)
    row["wpm"] = float(wpm)
    return row


def interp_features_from_positions(
    geometry: Geometry, positions: tuple[Position, Position], wpm: float = 0.0
) -> np.ndarray:
    """Interp-frame feature vector from recorded key positions (training AND serving path).

    ``wpm`` is accepted and IGNORED, with the signature kept parallel to
    :func:`bigram_features_from_positions` on purpose: every caller in the training and
    attribution stack passes ``wpm=``, and a frame that could not drop into those call sites would
    have to be threaded through a second code path — which is how a frame ends up featurized
    differently at train and at serve time. Accepting-and-ignoring keeps ONE call shape; the column
    is absent from the OUTPUT, which is what matters.
    """
    del wpm  # not a column in this frame (see schema); accepted for call-shape parity only
    row = interp_row_from_positions(geometry, positions[0], positions[1])
    return np.array([row[name] for name in BIGRAM_INTERP_FEATURE_NAMES], dtype=np.float64)


def interp_features(layout: Layout, bigram: str, wpm: float = 0.0) -> np.ndarray:
    """Interp-frame feature vector for a bigram on a layout."""
    return interp_features_from_positions(
        layout.geometry, (layout.pos(bigram[0]), layout.pos(bigram[1])), wpm
    )


def interp_wpm_features_from_positions(
    geometry: Geometry, positions: tuple[Position, Position], wpm: float = 0.0
) -> np.ndarray:
    """Interp+wpm feature vector from recorded key positions.

    Unlike :func:`interp_features_from_positions`, ``wpm`` is a REAL column here, so the LOGRAT->ms
    conversion recovers the pace from the matrix exactly as it does on the served frame.
    """
    row = interp_wpm_row_from_positions(geometry, positions[0], positions[1], wpm)
    return np.array([row[name] for name in BIGRAM_INTERP_WPM_FEATURE_NAMES], dtype=np.float64)


# --- hybrid-B (HYBRIDB-1) -----------------------------------------------------------------


def hybridb_row_from_positions(geometry: Geometry, a: Position, b: Position) -> dict[str, float]:
    """hybrid-B's eighteen features for one ORDERED bigram ``a -> b``.

    interp.1's ten ordinals PLUS the served ROW and FINGER one-hots. Both halves are taken from
    the EXISTING row builders rather than re-derived — :func:`interp_row_from_positions` for the
    ordinals and :func:`_placement_row_from_positions` for the one-hots — so a hybrid-B column is
    provably the SAME quantity the frame it came from carries. Re-spelling ``float(by == 1)`` here
    would make "hybrid-B contains the served ``bottom`` column" a claim rather than a fact.

    Returns an ordered dict keyed by
    :data:`~keybo.features.schema.BIGRAM_HYBRIDB_FEATURE_NAMES` exactly (a test pins
    ``list(row) == the name list``).

    ⚠ There is NO ``wpm`` key, for the same reason interp.1 has none — so, like interp.1, a model
    on this frame cannot span a WPM range and the LOGRAT->ms conversion cannot recover the pace
    from the matrix. Callers must state it.
    """
    row = interp_row_from_positions(geometry, a, b)
    placement = _placement_row_from_positions(geometry, a, b)
    for name in BIGRAM_HYBRIDB_FEATURE_NAMES:
        if name not in row:
            # KeyError-by-construction if the schema's one-hot list ever names a column
            # ``_placement_row_from_positions`` does not build. Explicit, because the silent
            # version (``placement.get(name, 0.0)``) would emit a zero column that reads as a
            # measured feature.
            row[name] = placement[name]
    return row


def hybridb_features_from_positions(
    geometry: Geometry, positions: tuple[Position, Position], wpm: float = 0.0
) -> np.ndarray:
    """hybrid-B feature vector from recorded key positions (training AND serving path).

    ``wpm`` is accepted and IGNORED, exactly as :func:`interp_features_from_positions` does it and
    for the same reason: every caller in the training and attribution stack passes ``wpm=``, and a
    frame that could not drop into those call sites would have to be threaded through a second code
    path — which is how a frame ends up featurized differently at train and at serve time.
    """
    del wpm  # not a column in this frame (see schema); accepted for call-shape parity only
    row = hybridb_row_from_positions(geometry, positions[0], positions[1])
    return np.array([row[name] for name in BIGRAM_HYBRIDB_FEATURE_NAMES], dtype=np.float64)


def hybridb_features(layout: Layout, bigram: str, wpm: float = 0.0) -> np.ndarray:
    """hybrid-B feature vector for a bigram on a layout."""
    return hybridb_features_from_positions(
        layout.geometry, (layout.pos(bigram[0]), layout.pos(bigram[1])), wpm
    )


# --- the REPLACEMENT-basis frame registry -------------------------------------------------
#
# ``direction`` and ``kitchensink`` WIDEN the served frame; the frames below REPLACE it, and are
# selected by the string-or-bool ``interp`` flag threaded through training and validation.
#
# Resolved in ONE place because the four things that must agree — the builder, the name list, the
# monotone tuple and the version stamp — were previously re-derived at each of three call sites by
# a chain of ``if interp == "wpm"`` tests. Two of these frames differ by a SINGLE column, so a
# wrong pick at one site produces a plausible-looking number rather than an error; the guard in
# ``keybo.training.validate._predict_cells`` exists because exactly that happened once and surfaced
# only as an xgboost shape error, which was luck. A dict makes the four values impossible to
# desynchronize and makes an unknown flag a KeyError at the boundary instead of a silent fallback
# to the 10-column frame.
_REPLACEMENT_FRAMES: dict[object, tuple] = {
    True: (
        interp_features_from_positions,
        BIGRAM_INTERP_FEATURE_NAMES,
        BIGRAM_INTERP_MONOTONE,
        FEATURE_VERSION_INTERP,
        "interp",
    ),
    "wpm": (
        interp_wpm_features_from_positions,
        BIGRAM_INTERP_WPM_FEATURE_NAMES,
        BIGRAM_INTERP_WPM_MONOTONE,
        FEATURE_VERSION_INTERP_WPM,
        "interp-wpm",
    ),
    "hybridb": (
        hybridb_features_from_positions,
        BIGRAM_HYBRIDB_FEATURE_NAMES,
        BIGRAM_HYBRIDB_MONOTONE,
        FEATURE_VERSION_HYBRIDB,
        "hybrid-b",
    ),
}

#: The legal values of the ``interp`` flag, ``False`` (the served frame) excluded.
REPLACEMENT_FRAME_FLAGS = tuple(_REPLACEMENT_FRAMES)


def replacement_frame(interp) -> tuple:
    """``(builder, names, monotone, version_stamp, tag)`` for one ``interp`` flag value.

    ``interp=True`` selects INTERPFRAME-1's 10-column frame, ``"wpm"`` its 11-column pace-adapting
    variant, ``"hybridb"`` HYBRIDB-1's 18-column frame. ``False`` is NOT a member: the served frame
    is not a replacement basis and callers branch on it before reaching here.

    Raises :class:`ValueError` on anything else, listing the legal values — a typo must not resolve
    to a frame.
    """
    if interp is False or interp not in _REPLACEMENT_FRAMES:
        raise ValueError(
            f"interp must be False (the served frame) or one of "
            f"{list(REPLACEMENT_FRAME_FLAGS)!r}; got {interp!r}"
        )
    return _REPLACEMENT_FRAMES[interp]


def _trigram_level_from_positions(
    geometry: Geometry, a: Position, b: Position, c: Position
) -> dict[str, float]:
    """Trigram-level and skipgram features, from the three key positions."""
    g = geometry
    ha, hb, hc = g.hand(a[0]), g.hand(b[0]), g.hand(c[0])
    same_hand_tri = ha != 0 and ha == hb == hc

    redirect = False
    bad_redirect = False
    if same_hand_tri:
        # Direction reverses between the two constituent bigrams (using |column|).
        going_in_1 = abs(b[0]) < abs(a[0])
        going_in_2 = abs(c[0]) < abs(b[0])
        redirect = going_in_1 != going_in_2
        # "bad" when no index finger is involved to absorb the redirect.
        bad_redirect = redirect and not any(abs(p[0]) in (1, 2) for p in (a, b, c))

    return {
        "same_hand_trigram": float(same_hand_tri),
        "redirect": float(redirect),
        "bad_redirect": float(bad_redirect),
        "sg_same_finger": float(C.same_finger(g, a, c)),
        "sg_dx": g.stagger_adjusted_dx(a, c),
        "sg_dy": float(abs(a[1] - c[1])),
        "sg_distance": g.distance(a, c),
    }


def _trigram_row_from_positions(
    geometry: Geometry,
    a: Position,
    b: Position,
    c: Position,
    wpm: float,
    direction: bool = False,
    kitchensink: bool = False,
) -> dict[str, float]:
    """Assemble the full trigram row from the three positions (the shared core).

    ``direction=True`` widens both constituent bigrams' placement blocks, so the trigram
    frame gains ``bg1_/bg2_inwards_ordered`` and ``..._outwards_ordered``. Same opt-in
    contract as the bigram frame: the default is byte-identical to the served columns.

    ``kitchensink=True`` adds the seven trigram-level external columns
    (:func:`trigram_kitchensink_row`) AND widens both constituent bigrams by the five
    bigram-level ones — which is why 12 candidate definitions become 17 new trigram columns.
    It implies ``direction``, so the three legal frames stay narrow / widened / kitchen-sink.
    """
    row = _trigram_level_from_positions(geometry, a, b, c)
    if direction or kitchensink:
        # The same-finger-gated redirect pair (REDIRGATE-1), declared in
        # TRIGRAM_DIRECTION_FEATURE_NAMES. Emitted here and NOT in
        # _trigram_level_from_positions, because that function feeds the version-locked served
        # frame: adding a key there would widen it silently for all three shipped
        # trigram_cond31 models. Key ORDER matters -- the schema puts these straight after the
        # trigram-level block, and a test pins list(row) == the name list.
        row.update(trigram_direction_row(geometry, a, b, c))
    if kitchensink:
        row.update(trigram_kitchensink_row(geometry, a, b, c))
    for name, value in _placement_row_from_positions(
        geometry, a, b, direction=direction, kitchensink=kitchensink
    ).items():
        row[f"bg1_{name}"] = value
    for name, value in _placement_row_from_positions(
        geometry, b, c, direction=direction, kitchensink=kitchensink
    ).items():
        row[f"bg2_{name}"] = value
    row["wpm"] = float(wpm)
    return row


def trigram_direction_row(
    geometry: Geometry, a: Position, b: Position, c: Position
) -> dict[str, float]:
    """The same-finger-GATED ``redirect`` pair — the widened trigram channel, opt-in.

    ``redirect``/``bad_redirect`` in :func:`_trigram_level_from_positions` derive their direction
    step from ``abs(b[0]) < abs(a[0])`` with NO same-finger gate, so a finger REPOSITIONING reads as
    a change of direction. The parity-gated ``_v1_pattern`` port excludes those as Sfb, and
    :mod:`keybo.scoring.oxey` records having fixed exactly this in its own trigram path — this
    served column never got the same treatment.

    Measured on ``ROW_STAGGERED_30`` over the 24,360 all-distinct ordered triples: ``redirect``
    fires 3,600 times of which **1,116** have a same-finger constituent bigram, and
    ``bad_redirect`` 648 of which **216** do. (On the wider ``a != b and b != c`` frame the totals
    are 3,960/1,152 and 756/216 — only the 216 is shared between the two frames, which is what
    makes a mixed-frame quotation look self-consistent. Both are pinned in
    ``tests/features/test_redirect_samefinger_gate.py``.)

    Returned SEPARATELY rather than added to the served row, for the same reason
    :data:`~keybo.features.schema.FEATURE_VERSION_DIRECTION` exists: those two columns belong to the
    version-locked trigram frame all three ``trigram_cond31`` models carry, and ``models/base.py``
    errors on a version MISMATCH — not on a column whose MEANING changed. Redefining them in place
    would leave every model loading fine while scoring a frame that no longer matches its training
    data. A retraining round can adopt these under the widened stamp.

    The gate can only ever REMOVE a firing (asserted exhaustively), so it is a strict subset of the
    ungated column rather than a differently-shaped feature.
    """
    ungated = _trigram_level_from_positions(geometry, a, b, c)
    step_is_real = not (C.same_finger(geometry, a, b) or C.same_finger(geometry, b, c))
    gated = ungated["redirect"] > 0.0 and step_is_real
    return {
        "redirect_sfgated": float(gated),
        "bad_redirect_sfgated": float(gated and ungated["bad_redirect"] > 0.0),
    }


def trigram_kitchensink_row(
    geometry: Geometry, a: Position, b: Position, c: Position
) -> dict[str, float]:
    """The seven trigram-level KITCHEN-SINK columns — external channels the served frame lacks.

    Reimplemented from keycraft's definitions (BSD-3-Clause; read, not vendored) and expressed in
    this module's vocabulary. Returned SEPARATELY, never merged into
    :func:`_trigram_level_from_positions`, for the same reason
    :func:`trigram_direction_row` is: that function feeds the version-locked served frame all three
    shipped ``trigram_cond31`` models carry, and ``models/base.py`` errors on a version MISMATCH,
    not on a column whose MEANING changed.

    Two families, both measured over the 24,360 ordered triples of ``ROW_STAGGERED_30``:

    * **one-hand flow** — ``onehand`` is keycraft's 3RL, the MONOTONIC one-hand roll. The served
      frame names ``redirect`` (the non-monotonic case) and so can only express the smoothest
      trigram class as a conjunction of negatives. ``onehand_in`` splits it by direction of travel
      and is order-aware (756 of 24,360 triples change under reversal), unlike ``onehand`` itself.
    * **the SFS splits and the skipgram predicates** — ``red_sfs``/``alt_sfs`` are keycraft's
      separately-priced redirect and alternation whose OUTER two keys share a finger.
      ``sg_full_scissor``/``sg_half_scissor``/``sg_lsb`` are the scissor and lateral-stretch
      predicates across the SKIPGRAM (keys 1 and 3): the served frame carries ``sg_dx``, ``sg_dy``,
      ``sg_distance`` and ``sg_same_finger`` but no sg_scissor and no sg_lsb, and the audit found
      these three the LEAST recoverable of all candidates (R2 0.149-0.190 against the served frame).

    keycraft's own RED-WEAK is deliberately absent: the audit found it bit-identical to
    ``bad_redirect_sfgated`` on all 24,360 triples, so it is a column REDIRGATE-1 already built and
    ``sfgated-eval`` already measured NULL.
    """
    g = geometry
    ha, hb, hc = g.hand(a[0]), g.hand(b[0]), g.hand(c[0])
    ka, kb, kc = (C.finger_kind(g, p[0]) for p in (a, b, c))

    one_hand = ha != 0 and ha == hb == hc
    # A same-finger constituent is neither a roll nor a redirect (keycraft routes it to 3RL-SFB),
    # and it is the same exclusion _roll_eligible makes at bigram level: the index finger's two
    # columns are ONE finger, so a reposition is not a direction step.
    distinct_fingers = one_hand and not C.same_finger(g, a, b) and not C.same_finger(g, b, c)
    monotonic = distinct_fingers and (ka < kb) == (kb < kc)
    # An alternation: outer two keys on one hand, middle key on the other.
    alternating = ha != 0 and hc != 0 and ha == hc and ha != hb
    outer_sfs = C.same_finger(g, a, c) and a != c

    return {
        "onehand": float(monotonic),
        "onehand_in": float(monotonic and kc > ka),
        "red_sfs": float(distinct_fingers and not monotonic and outer_sfs),
        "alt_sfs": float(alternating and outer_sfs),
        "sg_full_scissor": float(C.is_adjacent(g, a, c) and abs(a[1] - c[1]) == 2),
        "sg_half_scissor": float(C.is_half_scissor(g, a, c)),
        "sg_lsb": float(C.is_lsb(g, a, c)),
    }


def _trigram_column_names(direction: bool, kitchensink: bool = False) -> list[str]:
    if kitchensink:
        return TRIGRAM_KITCHENSINK_FEATURE_NAMES
    return TRIGRAM_DIRECTION_FEATURE_NAMES if direction else TRIGRAM_FEATURE_NAMES


def trigram_model_row(
    layout: Layout,
    trigram: str,
    wpm: float,
    direction: bool = False,
    kitchensink: bool = False,
) -> dict[str, float]:
    """Full ordered trigram feature row: trigram-level + both bigrams + wpm."""
    return _trigram_row_from_positions(
        layout.geometry,
        layout.pos(trigram[0]),
        layout.pos(trigram[1]),
        layout.pos(trigram[2]),
        wpm,
        direction=direction,
        kitchensink=kitchensink,
    )


def trigram_features(
    layout: Layout,
    trigram: str,
    wpm: float = 0.0,
    direction: bool = False,
    kitchensink: bool = False,
) -> np.ndarray:
    """Trigram feature vector in canonical column order."""
    row = trigram_model_row(layout, trigram, wpm, direction=direction, kitchensink=kitchensink)
    return np.array(
        [row[name] for name in _trigram_column_names(direction, kitchensink)], dtype=np.float64
    )


def trigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position, Position],
    wpm: float = 0.0,
    direction: bool = False,
    kitchensink: bool = False,
) -> np.ndarray:
    """Trigram feature vector from recorded key positions (training path)."""
    a, b, c = positions
    row = _trigram_row_from_positions(
        geometry, a, b, c, wpm, direction=direction, kitchensink=kitchensink
    )
    return np.array(
        [row[name] for name in _trigram_column_names(direction, kitchensink)], dtype=np.float64
    )
