"""The n-gram feature pipeline: (layout, n-gram) -> feature row.

This is the ONE place features are computed. Data processing, model training, and layout
scoring all call these functions, so the features a model is trained on are exactly the
features it is later scored with. Rows are returned as ordered dicts keyed by the names in
:mod:`keybo.features.schema`; :func:`bigram_features` / :func:`trigram_features` return the
same values as a plain float vector for the model.
"""

from __future__ import annotations

import numpy as np

from keybo.features import classify as C
from keybo.features.schema import BIGRAM_FEATURE_NAMES, TRIGRAM_FEATURE_NAMES
from keybo.layout import Layout


def _placement_row(layout: Layout, bigram: str, freq: float) -> dict[str, float]:
    """The 20 placement/relational/geometry features for one bigram (no wpm)."""
    g = layout.geometry
    a = layout.pos(bigram[0])
    b = layout.pos(bigram[1])
    bx, by = b

    cls = C.classify_bigram(layout, bigram)
    abs_bx = abs(bx)

    return {
        "freq": float(freq),
        # second-key row one-hot
        "bottom": float(by == 1),
        "home": float(by == 2),
        "top": float(by == 3),
        # second-key finger one-hot (index = columns 1 and 2)
        "pinky": float(abs_bx == 5),
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


def bigram_model_row(layout: Layout, bigram: str, freq: float, wpm: float) -> dict[str, float]:
    """Full ordered bigram feature row (placement features + wpm)."""
    row = _placement_row(layout, bigram, freq)
    row["wpm"] = float(wpm)
    return row


def bigram_features(layout: Layout, bigram: str, freq: float = 1.0, wpm: float = 0.0) -> np.ndarray:
    """Bigram feature vector in canonical column order."""
    row = bigram_model_row(layout, bigram, freq, wpm)
    return np.array([row[name] for name in BIGRAM_FEATURE_NAMES], dtype=np.float64)


def _trigram_level(layout: Layout, trigram: str, tg_freq: float, sg_freq: float) -> dict[str, float]:
    """Trigram-level and skipgram features."""
    g = layout.geometry
    a = layout.pos(trigram[0])
    b = layout.pos(trigram[1])
    c = layout.pos(trigram[2])

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
        "tg_freq": float(tg_freq),
        "same_hand_trigram": float(same_hand_tri),
        "redirect": float(redirect),
        "bad_redirect": float(bad_redirect),
        "sg_freq": float(sg_freq),
        "sg_same_finger": float(C.same_finger(g, a, c)),
        "sg_dx": g.stagger_adjusted_dx(a, c),
        "sg_dy": float(abs(a[1] - c[1])),
        "sg_distance": g.distance(a, c),
    }


def trigram_model_row(
    layout: Layout,
    trigram: str,
    tg_freq: float,
    bg1_freq: float,
    bg2_freq: float,
    sg_freq: float,
    wpm: float,
) -> dict[str, float]:
    """Full ordered trigram feature row: trigram-level + both bigrams + wpm."""
    row = _trigram_level(layout, trigram, tg_freq, sg_freq)
    for name, value in _placement_row(layout, trigram[:2], bg1_freq).items():
        row[f"bg1_{name}"] = value
    for name, value in _placement_row(layout, trigram[1:], bg2_freq).items():
        row[f"bg2_{name}"] = value
    row["wpm"] = float(wpm)
    return row


def trigram_features(
    layout: Layout,
    trigram: str,
    tg_freq: float = 1.0,
    bg1_freq: float = 1.0,
    bg2_freq: float = 1.0,
    sg_freq: float = 1.0,
    wpm: float = 0.0,
) -> np.ndarray:
    """Trigram feature vector in canonical column order."""
    row = trigram_model_row(layout, trigram, tg_freq, bg1_freq, bg2_freq, sg_freq, wpm)
    return np.array([row[name] for name in TRIGRAM_FEATURE_NAMES], dtype=np.float64)
