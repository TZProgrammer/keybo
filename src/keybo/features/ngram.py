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
from keybo.features.schema import BIGRAM_FEATURE_NAMES, TRIGRAM_FEATURE_NAMES
from keybo.geometry import Geometry, Position
from keybo.layout import Layout


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


def _placement_row(layout: Layout, bigram: str) -> dict[str, float]:
    """Placement features for a bigram on a layout (looks up positions, then delegates)."""
    return _placement_row_from_positions(
        layout.geometry, layout.pos(bigram[0]), layout.pos(bigram[1])
    )


def bigram_model_row(layout: Layout, bigram: str, wpm: float) -> dict[str, float]:
    """Full ordered bigram feature row (placement features + wpm)."""
    row = _placement_row(layout, bigram)
    row["wpm"] = float(wpm)
    return row


def bigram_features(layout: Layout, bigram: str, wpm: float = 0.0) -> np.ndarray:
    """Bigram feature vector in canonical column order."""
    row = bigram_model_row(layout, bigram, wpm)
    return np.array([row[name] for name in BIGRAM_FEATURE_NAMES], dtype=np.float64)


def bigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position],
    wpm: float = 0.0,
) -> np.ndarray:
    """Bigram feature vector from recorded key positions (training path)."""
    row = _placement_row_from_positions(geometry, positions[0], positions[1])
    row["wpm"] = float(wpm)
    return np.array([row[name] for name in BIGRAM_FEATURE_NAMES], dtype=np.float64)


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
) -> dict[str, float]:
    """Assemble the full trigram row from the three positions (the shared core)."""
    row = _trigram_level_from_positions(geometry, a, b, c)
    for name, value in _placement_row_from_positions(geometry, a, b).items():
        row[f"bg1_{name}"] = value
    for name, value in _placement_row_from_positions(geometry, b, c).items():
        row[f"bg2_{name}"] = value
    row["wpm"] = float(wpm)
    return row


def trigram_model_row(layout: Layout, trigram: str, wpm: float) -> dict[str, float]:
    """Full ordered trigram feature row: trigram-level + both bigrams + wpm."""
    return _trigram_row_from_positions(
        layout.geometry,
        layout.pos(trigram[0]),
        layout.pos(trigram[1]),
        layout.pos(trigram[2]),
        wpm,
    )


def trigram_features(layout: Layout, trigram: str, wpm: float = 0.0) -> np.ndarray:
    """Trigram feature vector in canonical column order."""
    row = trigram_model_row(layout, trigram, wpm)
    return np.array([row[name] for name in TRIGRAM_FEATURE_NAMES], dtype=np.float64)


def trigram_features_from_positions(
    geometry: Geometry,
    positions: tuple[Position, Position, Position],
    wpm: float = 0.0,
) -> np.ndarray:
    """Trigram feature vector from recorded key positions (training path)."""
    a, b, c = positions
    row = _trigram_row_from_positions(geometry, a, b, c, wpm)
    return np.array([row[name] for name in TRIGRAM_FEATURE_NAMES], dtype=np.float64)
