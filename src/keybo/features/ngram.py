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
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.geometry import Geometry, Position
from keybo.layout import Layout


def _placement_row_from_positions(
    geometry: Geometry, a: Position, b: Position, direction: bool = False
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
    if direction:
        row["inwards_ordered"] = float(C.is_inwards_ordered(g, a, b))
        row["outwards_ordered"] = float(C.is_outwards_ordered(g, a, b))
    return row


def _placement_row(layout: Layout, bigram: str, direction: bool = False) -> dict[str, float]:
    """Placement features for a bigram on a layout (looks up positions, then delegates)."""
    return _placement_row_from_positions(
        layout.geometry, layout.pos(bigram[0]), layout.pos(bigram[1]), direction=direction
    )


def _bigram_column_names(direction: bool) -> list[str]:
    """The canonical column order for the frame ``direction`` selects."""
    return BIGRAM_DIRECTION_FEATURE_NAMES if direction else BIGRAM_FEATURE_NAMES


def bigram_model_row(
    layout: Layout, bigram: str, wpm: float, direction: bool = False
) -> dict[str, float]:
    """Full ordered bigram feature row (placement features + wpm)."""
    row = _placement_row(layout, bigram, direction=direction)
    row["wpm"] = float(wpm)
    return row


def bigram_features(
    layout: Layout, bigram: str, wpm: float = 0.0, direction: bool = False
) -> np.ndarray:
    """Bigram feature vector in canonical column order."""
    row = bigram_model_row(layout, bigram, wpm, direction=direction)
    return np.array([row[name] for name in _bigram_column_names(direction)], dtype=np.float64)


def bigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position],
    wpm: float = 0.0,
    direction: bool = False,
) -> np.ndarray:
    """Bigram feature vector from recorded key positions (training path)."""
    row = _placement_row_from_positions(geometry, positions[0], positions[1], direction=direction)
    row["wpm"] = float(wpm)
    return np.array([row[name] for name in _bigram_column_names(direction)], dtype=np.float64)


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
) -> dict[str, float]:
    """Assemble the full trigram row from the three positions (the shared core).

    ``direction=True`` widens both constituent bigrams' placement blocks, so the trigram
    frame gains ``bg1_/bg2_inwards_ordered`` and ``..._outwards_ordered``. Same opt-in
    contract as the bigram frame: the default is byte-identical to the served columns.
    """
    row = _trigram_level_from_positions(geometry, a, b, c)
    if direction:
        # The same-finger-gated redirect pair (REDIRGATE-1), declared in
        # TRIGRAM_DIRECTION_FEATURE_NAMES. Emitted here and NOT in
        # _trigram_level_from_positions, because that function feeds the version-locked served
        # frame: adding a key there would widen it silently for all three shipped
        # trigram_cond31 models. Key ORDER matters -- the schema puts these straight after the
        # trigram-level block, and a test pins list(row) == the name list.
        row.update(trigram_direction_row(geometry, a, b, c))
    for name, value in _placement_row_from_positions(geometry, a, b, direction=direction).items():
        row[f"bg1_{name}"] = value
    for name, value in _placement_row_from_positions(geometry, b, c, direction=direction).items():
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


def _trigram_column_names(direction: bool) -> list[str]:
    return TRIGRAM_DIRECTION_FEATURE_NAMES if direction else TRIGRAM_FEATURE_NAMES


def trigram_model_row(
    layout: Layout, trigram: str, wpm: float, direction: bool = False
) -> dict[str, float]:
    """Full ordered trigram feature row: trigram-level + both bigrams + wpm."""
    return _trigram_row_from_positions(
        layout.geometry,
        layout.pos(trigram[0]),
        layout.pos(trigram[1]),
        layout.pos(trigram[2]),
        wpm,
        direction=direction,
    )


def trigram_features(
    layout: Layout, trigram: str, wpm: float = 0.0, direction: bool = False
) -> np.ndarray:
    """Trigram feature vector in canonical column order."""
    row = trigram_model_row(layout, trigram, wpm, direction=direction)
    return np.array([row[name] for name in _trigram_column_names(direction)], dtype=np.float64)


def trigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position, Position],
    wpm: float = 0.0,
    direction: bool = False,
) -> np.ndarray:
    """Trigram feature vector from recorded key positions (training path)."""
    a, b, c = positions
    row = _trigram_row_from_positions(geometry, a, b, c, wpm, direction=direction)
    return np.array([row[name] for name in _trigram_column_names(direction)], dtype=np.float64)
