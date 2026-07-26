"""The n-gram feature pipeline: (layout, n-gram) -> feature row.

This is the ONE place features are computed. Data processing, model training, and layout
scoring all call these functions, so the features a model is trained on are exactly the
features it is later scored with. Rows are returned as ordered dicts keyed by the names in
:mod:`keybo.features.schema`; :func:`bigram_features` / :func:`trigram_features` return the
same values as a plain float vector for the model.

Frequency is NOT an input here (OQ-1, 2026-07-05): features are pure geometry + wpm.
Frequency enters the system only as the objective weight and as the identity key of the
additive practice term (see :mod:`keybo.training.train`).

Every builder takes a ``direction`` flag (default ``False``). ``False`` reproduces the v1
20-column vector BIT-IDENTICALLY — the shipped models in ``data/models/`` depend on that, so
the direction columns are appended, never interleaved. ``True`` selects the v2 surface
(``schema.FEATURE_VERSION_DIRECTION``), which adds the order-dependent columns the v1 vector
provably cannot express. See :mod:`keybo.features.schema` for why v1 is blind and which
candidate columns were REJECTED for being already-determined.
"""

from __future__ import annotations

import numpy as np

from keybo.features import classify as C
from keybo.features.schema import (
    BIGRAM_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES_DIRECTION,
    BIGRAM_FEATURE_NAMES_PLACEBO,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES_DIRECTION,
    TRIGRAM_FEATURE_NAMES_PLACEBO,
)
from keybo.geometry import Geometry, Position
from keybo.layout import Layout


def _check_surface(direction: bool, placebo: bool) -> None:
    if direction and placebo:
        raise ValueError(
            "direction and placebo are mutually exclusive feature surfaces: the placebo "
            "exists to isolate frame WIDTH from direction information, so combining them "
            "would measure neither"
        )


def bigram_feature_names(direction: bool = False, placebo: bool = False) -> list[str]:
    """The bigram column order for the requested feature surface."""
    _check_surface(direction, placebo)
    if direction:
        return BIGRAM_FEATURE_NAMES_DIRECTION
    return BIGRAM_FEATURE_NAMES_PLACEBO if placebo else BIGRAM_FEATURE_NAMES


def trigram_feature_names(direction: bool = False, placebo: bool = False) -> list[str]:
    """The trigram column order for the requested feature surface."""
    _check_surface(direction, placebo)
    if direction:
        return TRIGRAM_FEATURE_NAMES_DIRECTION
    return TRIGRAM_FEATURE_NAMES_PLACEBO if placebo else TRIGRAM_FEATURE_NAMES


def feature_version(direction: bool = False, placebo: bool = False) -> str:
    """The ``FEATURE_VERSION`` a model trained on the requested surface must stamp."""
    from keybo.features.schema import (
        FEATURE_VERSION,
        FEATURE_VERSION_DIRECTION,
        FEATURE_VERSION_PLACEBO,
    )

    _check_surface(direction, placebo)
    if direction:
        return FEATURE_VERSION_DIRECTION
    return FEATURE_VERSION_PLACEBO if placebo else FEATURE_VERSION


def _direction_row_from_positions(geometry: Geometry, a: Position, b: Position) -> dict[str, float]:
    """The order-DEPENDENT features for one bigram: which way the motion ran, and from where.

    Separate from :func:`_placement_row_from_positions` so the v1 block cannot drift: v1 is
    what every shipped artifact was fit on, and a change there is a silent train/serve skew.
    """
    g = geometry
    abs_ax = abs(a[0])
    return {
        "signed_dx": C.signed_dx(g, a, b),
        "dir_dx_inward": C.dir_dx_inward(g, a, b),
        "dir_angle": C.directed_angle(g, a, b),
        "dir_inwards": float(C.is_directed_inwards(g, a, b)),
        "dir_outwards": float(C.is_directed_outwards(g, a, b)),
        # ORIGIN-key finger one-hot. Mirrors the landing-key block's column convention
        # (index = columns 1 and 2; K31 pinky = 5 and 6). The origin ROW is deliberately
        # NOT here: the stagger-adjusted dx already determines it (schema docstring).
        "o_pinky": float(abs_ax in (5, 6)),
        "o_ring": float(abs_ax == 4),
        "o_middle": float(abs_ax == 3),
        "o_index": float(abs_ax in (1, 2)),
    }


def _placebo_row_from_positions(geometry: Geometry, a: Position, b: Position) -> dict[str, float]:
    """The SAME-WIDTH placebo block: 9 columns carrying NO information v1 lacks.

    Every value here is a deterministic function of the v1 vector (the origin row and
    signed_dy are recoverable from the stagger-adjusted dx; o_lateral likewise), so this
    frame isolates the effect of frame WIDTH alone. Measurement only — never served.
    """
    ay = a[1]
    signed_dy = float(b[1] - a[1])
    o_bottom, o_home, o_top = float(ay == 1), float(ay == 2), float(ay == 3)
    o_lateral = float(C.is_lateral(a[0]))
    return {
        "p_o_bottom": o_bottom,
        "p_o_home": o_home,
        "p_o_top": o_top,
        "p_signed_dy": signed_dy,
        "p_o_lateral": o_lateral,
        "p_o_bottom2": o_bottom,
        "p_o_home2": o_home,
        "p_o_top2": o_top,
        "p_signed_dy2": signed_dy,
    }


def _placement_row_from_positions(geometry: Geometry, a: Position, b: Position) -> dict[str, float]:
    """The placement/relational/geometry features for one bigram, from key positions.

    Positions are the fundamental input: both scoring (positions from a layout) and training
    (positions recorded in the data) route through here, so the two can never diverge.
    """
    g = geometry
    bx, by = b
    cls = C.classify_positions(g, a, b)
    abs_bx = abs(bx)

    return {
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
        "inwards": float(C.is_inwards(g, a, b)),
        "outwards": float(C.is_outwards(g, a, b)),
    }


def _bigram_row_from_positions(
    geometry: Geometry,
    a: Position,
    b: Position,
    direction: bool = False,
    placebo: bool = False,
) -> dict[str, float]:
    """The placement block, plus the direction (or placebo) block when one is requested."""
    _check_surface(direction, placebo)
    row = _placement_row_from_positions(geometry, a, b)
    if direction:
        row.update(_direction_row_from_positions(geometry, a, b))
    elif placebo:
        row.update(_placebo_row_from_positions(geometry, a, b))
    return row


def _placement_row(
    layout: Layout, bigram: str, direction: bool = False, placebo: bool = False
) -> dict[str, float]:
    """Placement features for a bigram on a layout (looks up positions, then delegates)."""
    return _bigram_row_from_positions(
        layout.geometry,
        layout.pos(bigram[0]),
        layout.pos(bigram[1]),
        direction=direction,
        placebo=placebo,
    )


def bigram_model_row(
    layout: Layout, bigram: str, wpm: float, direction: bool = False, placebo: bool = False
) -> dict[str, float]:
    """Full ordered bigram feature row (placement features [+ direction] + wpm)."""
    row = _placement_row(layout, bigram, direction=direction, placebo=placebo)
    row["wpm"] = float(wpm)
    return row


def bigram_features(
    layout: Layout,
    bigram: str,
    wpm: float = 0.0,
    direction: bool = False,
    placebo: bool = False,
) -> np.ndarray:
    """Bigram feature vector in canonical column order."""
    row = bigram_model_row(layout, bigram, wpm, direction=direction, placebo=placebo)
    return np.array(
        [row[name] for name in bigram_feature_names(direction, placebo)], dtype=np.float64
    )


def bigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position],
    wpm: float = 0.0,
    direction: bool = False,
    placebo: bool = False,
) -> np.ndarray:
    """Bigram feature vector from recorded key positions (training path)."""
    row = _bigram_row_from_positions(
        geometry, positions[0], positions[1], direction=direction, placebo=placebo
    )
    row["wpm"] = float(wpm)
    return np.array(
        [row[name] for name in bigram_feature_names(direction, placebo)], dtype=np.float64
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
    placebo: bool = False,
) -> dict[str, float]:
    """Assemble the full trigram row from the three positions (the shared core)."""
    row = _trigram_level_from_positions(geometry, a, b, c)
    for name, value in _bigram_row_from_positions(
        geometry, a, b, direction=direction, placebo=placebo
    ).items():
        row[f"bg1_{name}"] = value
    for name, value in _bigram_row_from_positions(
        geometry, b, c, direction=direction, placebo=placebo
    ).items():
        row[f"bg2_{name}"] = value
    row["wpm"] = float(wpm)
    return row


def trigram_model_row(
    layout: Layout, trigram: str, wpm: float, direction: bool = False, placebo: bool = False
) -> dict[str, float]:
    """Full ordered trigram feature row: trigram-level + both bigrams + wpm."""
    return _trigram_row_from_positions(
        layout.geometry,
        layout.pos(trigram[0]),
        layout.pos(trigram[1]),
        layout.pos(trigram[2]),
        wpm,
        direction=direction,
        placebo=placebo,
    )


def trigram_features(
    layout: Layout,
    trigram: str,
    wpm: float = 0.0,
    direction: bool = False,
    placebo: bool = False,
) -> np.ndarray:
    """Trigram feature vector in canonical column order."""
    row = trigram_model_row(layout, trigram, wpm, direction=direction, placebo=placebo)
    return np.array(
        [row[name] for name in trigram_feature_names(direction, placebo)], dtype=np.float64
    )


def trigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position, Position],
    wpm: float = 0.0,
    direction: bool = False,
    placebo: bool = False,
) -> np.ndarray:
    """Trigram feature vector from recorded key positions (training path)."""
    a, b, c = positions
    row = _trigram_row_from_positions(geometry, a, b, c, wpm, direction=direction, placebo=placebo)
    return np.array(
        [row[name] for name in trigram_feature_names(direction, placebo)], dtype=np.float64
    )
