"""Structural diagnostics for a layout — the objective's blind spots, made visible.

Born from the finger-utilization gap (gaps-and-roadmap.md axis 1.1): the speed objective
prices finger reuse only at lag 1, so properties like per-finger duty cycle — the design
principle behind semimak-generation layouts — must at least be measurable at a glance,
for our layouts and the named ones alike. These numbers deliberately mirror what community
analyzers report, so disagreements with their judgments can be localized.

All shares are corpus-weighted (Σ over bigrams of freq × indicator / total included
weight). Bigrams containing a character the layout cannot type are excluded and their
weight reported (`excluded_weight_share`), never silently dropped.
"""

from __future__ import annotations

from collections.abc import Mapping

from keybo.features import classify as C
from keybo.geometry import Finger
from keybo.layout import Layout

_FINGER_LABEL = {
    Finger.LP: "L-pinky",
    Finger.LR: "L-ring",
    Finger.LM: "L-middle",
    Finger.LI: "L-index",
    Finger.RI: "R-index",
    Finger.RM: "R-middle",
    Finger.RR: "R-ring",
    Finger.RP: "R-pinky",
    Finger.THUMB: "thumb",
}
_ROW_LABEL = {3: "top", 2: "home", 1: "bottom", 0: "space"}


def layout_diagnostics(layout: Layout, bigram_freqs: Mapping[str, int]) -> dict:
    """Corpus-weighted structural profile of ``layout`` under ``bigram_freqs``."""
    g = layout.geometry
    finger_load: dict[str, float] = {label: 0.0 for label in _FINGER_LABEL.values()}
    row_share: dict[str, float] = {label: 0.0 for label in _ROW_LABEL.values()}
    motion_share = {"alternate": 0.0, "same_hand": 0.0, "same_finger": 0.0}
    sfb_weight = 0.0
    sfb_by_finger: dict[str, float] = {label: 0.0 for label in _FINGER_LABEL.values()}
    scissor_weight = 0.0
    lsb_weight = 0.0
    included = 0.0
    excluded = 0.0

    for bg, freq in bigram_freqs.items():
        if len(bg) != 2 or not all(layout.has_key(c) for c in bg):
            excluded += freq
            continue
        included += freq
        a, b = layout.pos(bg[0]), layout.pos(bg[1])
        # Per-KEY loads: each keystroke of the bigram contributes half the bigram weight.
        for pos in (a, b):
            finger_load[_FINGER_LABEL[g.finger(pos[0])]] += freq / 2
            row_share[_ROW_LABEL[pos[1]]] += freq / 2
        cls = C.classify_positions(g, a, b)
        if cls is C.BigramClass.ALTERNATE:
            motion_share["alternate"] += freq
        elif cls is C.BigramClass.SAME_FINGER:
            motion_share["same_finger"] += freq
            if a != b:  # a repeated key is reuse but not a movement conflict
                sfb_weight += freq
                sfb_by_finger[_FINGER_LABEL[g.finger(a[0])]] += freq
        else:
            motion_share["same_hand"] += freq
        if C.is_scissor(g, a, b):
            scissor_weight += freq
        if C.is_lsb(g, a, b):
            lsb_weight += freq

    total = included if included else 1.0
    sfb_total = sfb_weight if sfb_weight else 1.0
    return {
        "finger_load": {k: v / total for k, v in finger_load.items()},
        "row_share": {k: v / total for k, v in row_share.items()},
        "motion_share": {k: v / total for k, v in motion_share.items()},
        "sfb_share": sfb_weight / total,
        "sfb_by_finger": {k: v / sfb_total for k, v in sfb_by_finger.items()},
        "scissor_share": scissor_weight / total,
        "lsb_share": lsb_weight / total,
        "excluded_weight_share": excluded / (included + excluded) if (included + excluded) else 0.0,
    }
