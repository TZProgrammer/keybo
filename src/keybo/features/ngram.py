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
    BIGRAM_KITCHENSINK_FEATURE_NAMES,
    MIRROR_SYMMETRIZED_COLUMNS,
    MIRROR_SYMMETRIZED_TRIGRAM_COLUMNS,
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


def mirror_position(position: Position) -> Position:
    """The left/right mirror image of a key: negate the signed column, keep the row.

    ⚠ A mirror of the COLUMN INDEX, not an isometry of the physical board. The row stagger
    (``Geometry.row_offsets``) applies identically to both hands, so the physical coordinate
    ``x + off(y)`` is not antisymmetric: this map preserves ``stagger_adjusted_dx`` on the 330
    same-row-or-same-column pairs of ``ROW_STAGGERED_30`` and CHANGES it on the other 540 (all
    cross-row). No vertical-axis reflection maps the staggered board onto itself, so there is no
    better definition to choose — see :data:`~keybo.features.schema.FEATURE_VERSION_MIRROR`.

    On ``ROW_STAGGERED_31`` it is not even a permutation of the slots: the quote slot ``(6, 2)``
    maps to ``(-6, 2)``, which is not a key. The image's feature row is still COMPUTABLE
    (``Geometry.finger(-6)`` resolves to the left pinky), so a symmetrized frame is well
    defined on K31 — but for that column it symmetrizes against a hypothetical key.
    """
    x, y = position
    return (-x, y)


def _mirror_symmetrized_row(
    geometry: Geometry,
    a: Position,
    b: Position,
    direction: bool = False,
    kitchensink: bool = False,
) -> dict[str, float]:
    """The placement row with :data:`MIRROR_SYMMETRIZED_COLUMNS` forced mirror-invariant.

    Averages each of ``dx``/``angle``/``lsb`` with its value on the mirrored pair. The mean is
    the symmetrization (any symmetric function of the two would do; the mean is the one that
    keeps the column's scale and leaves the 330 already-symmetric pairs BIT-IDENTICAL, so the
    frame change is confined to the 540 pairs whose two hands' geometry genuinely differs).

    The other 17 placement columns are untouched because they are already mirror-invariant:
    each is built from ``abs(x)``, from a row/column difference, or is hand-normalized. That is
    also why this frame carries no hand-identity channel — and hence why a model trained on it
    cannot represent handedness even in principle.
    """
    row = _placement_row_from_positions(
        geometry, a, b, direction=direction, kitchensink=kitchensink
    )
    mirrored = _placement_row_from_positions(
        geometry,
        mirror_position(a),
        mirror_position(b),
        direction=direction,
        kitchensink=kitchensink,
    )
    for name in MIRROR_SYMMETRIZED_COLUMNS:
        row[name] = 0.5 * (row[name] + mirrored[name])
    return row


def placement_row_from_positions(
    geometry: Geometry,
    a: Position,
    b: Position,
    direction: bool = False,
    kitchensink: bool = False,
    mirror: bool = False,
) -> dict[str, float]:
    """The placement row for one bigram, in whichever frame the flags select.

    The single dispatch point for the ``mirror`` frame, so the training, scoring and trigram
    paths cannot disagree about what it means.
    """
    if mirror:
        return _mirror_symmetrized_row(
            geometry, a, b, direction=direction, kitchensink=kitchensink
        )
    return _placement_row_from_positions(
        geometry, a, b, direction=direction, kitchensink=kitchensink
    )


def _placement_row(
    layout: Layout,
    bigram: str,
    direction: bool = False,
    kitchensink: bool = False,
    mirror: bool = False,
) -> dict[str, float]:
    """Placement features for a bigram on a layout (looks up positions, then delegates)."""
    return placement_row_from_positions(
        layout.geometry,
        layout.pos(bigram[0]),
        layout.pos(bigram[1]),
        direction=direction,
        kitchensink=kitchensink,
        mirror=mirror,
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
    mirror: bool = False,
) -> dict[str, float]:
    """Full ordered bigram feature row (placement features + wpm)."""
    row = _placement_row(
        layout, bigram, direction=direction, kitchensink=kitchensink, mirror=mirror
    )
    row["wpm"] = float(wpm)
    return row


def bigram_features(
    layout: Layout,
    bigram: str,
    wpm: float = 0.0,
    direction: bool = False,
    kitchensink: bool = False,
    mirror: bool = False,
) -> np.ndarray:
    """Bigram feature vector in canonical column order."""
    row = bigram_model_row(
        layout, bigram, wpm, direction=direction, kitchensink=kitchensink, mirror=mirror
    )
    return np.array(
        [row[name] for name in _bigram_column_names(direction, kitchensink)], dtype=np.float64
    )


def bigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position],
    wpm: float = 0.0,
    direction: bool = False,
    kitchensink: bool = False,
    mirror: bool = False,
) -> np.ndarray:
    """Bigram feature vector from recorded key positions (training path)."""
    row = placement_row_from_positions(
        geometry,
        positions[0],
        positions[1],
        direction=direction,
        kitchensink=kitchensink,
        mirror=mirror,
    )
    row["wpm"] = float(wpm)
    return np.array(
        [row[name] for name in _bigram_column_names(direction, kitchensink)], dtype=np.float64
    )


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
    mirror: bool = False,
) -> dict[str, float]:
    """Assemble the full trigram row from the three positions (the shared core).

    ``direction=True`` widens both constituent bigrams' placement blocks, so the trigram
    frame gains ``bg1_/bg2_inwards_ordered`` and ``..._outwards_ordered``. Same opt-in
    contract as the bigram frame: the default is byte-identical to the served columns.

    ``kitchensink=True`` adds the seven trigram-level external columns
    (:func:`trigram_kitchensink_row`) AND widens both constituent bigrams by the five
    bigram-level ones — which is why 12 candidate definitions become 17 new trigram columns.
    It implies ``direction``, so the three legal frames stay narrow / widened / kitchen-sink.

    ``mirror=True`` forces the frame mirror-invariant: both constituent bigrams' placement
    blocks are symmetrized (:data:`MIRROR_SYMMETRIZED_COLUMNS`) and so is the trigram-level
    skipgram span (:data:`MIRROR_SYMMETRIZED_TRIGRAM_COLUMNS`). Column names and width are
    unchanged; only three-per-bigram-plus-one values move. See
    :data:`~keybo.features.schema.FEATURE_VERSION_MIRROR` for why the constraint is true on only
    a subset of pairs.
    """
    row = _trigram_level_from_positions(geometry, a, b, c)
    if mirror:
        # Only the skipgram SPAN is mirror-variant at trigram level (stagger, via
        # stagger_adjusted_dx across keys 1 and 3); the rest read abs(x) or a row difference.
        mirrored_level = _trigram_level_from_positions(
            geometry, mirror_position(a), mirror_position(b), mirror_position(c)
        )
        for name in MIRROR_SYMMETRIZED_TRIGRAM_COLUMNS:
            row[name] = 0.5 * (row[name] + mirrored_level[name])
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
    for name, value in placement_row_from_positions(
        geometry, a, b, direction=direction, kitchensink=kitchensink, mirror=mirror
    ).items():
        row[f"bg1_{name}"] = value
    for name, value in placement_row_from_positions(
        geometry, b, c, direction=direction, kitchensink=kitchensink, mirror=mirror
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
    mirror: bool = False,
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
        mirror=mirror,
    )


def trigram_features(
    layout: Layout,
    trigram: str,
    wpm: float = 0.0,
    direction: bool = False,
    kitchensink: bool = False,
    mirror: bool = False,
) -> np.ndarray:
    """Trigram feature vector in canonical column order."""
    row = trigram_model_row(
        layout, trigram, wpm, direction=direction, kitchensink=kitchensink, mirror=mirror
    )
    return np.array(
        [row[name] for name in _trigram_column_names(direction, kitchensink)], dtype=np.float64
    )


def trigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position, Position],
    wpm: float = 0.0,
    direction: bool = False,
    kitchensink: bool = False,
    mirror: bool = False,
) -> np.ndarray:
    """Trigram feature vector from recorded key positions (training path)."""
    a, b, c = positions
    row = _trigram_row_from_positions(
        geometry, a, b, c, wpm, direction=direction, kitchensink=kitchensink, mirror=mirror
    )
    return np.array(
        [row[name] for name in _trigram_column_names(direction, kitchensink)], dtype=np.float64
    )
