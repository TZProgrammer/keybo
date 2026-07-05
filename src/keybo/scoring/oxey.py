"""Oxeylyzer-style heuristic scorer — the community's judgment as an IScorer (7.2).

An explicit, documented APPROXIMATION of the pattern-count scoring used by the
oxeylyzer/genkey family of community analyzers: corpus-weighted percentages of
same-finger bigrams, disjoint SFBs (skipgrams), lateral stretches, scissors, rolls
(rewarded), onehands (rewarded), redirects, and finger imbalance — combined with signed
weights into one scalar (lower = better, matching our fitness convention).

Two honesty notes, load-bearing:

1. This is a PREFERENCE term, not a measurement. Our own data measured redirects as
   time-NEUTRAL (roll_error_probe: redirect contrast == roll contrast at every skill
   band) and lag-2/disjoint same-finger reuse as speed-neutral — patterns this scorer
   penalizes because the community dislikes them. Jointly optimizing speed + oxey score
   (via CompositeScorer / ``optimize --oxey-weight``) deliberately re-introduces that
   doctrine at a user-chosen weight; at weight 0 the measured objective is untouched.
2. It is an approximation, not a port: weight VALUES are chosen to reproduce the
   community's layout ORDERING (tested: colemak/semimak must beat qwerty), not any
   specific analyzer's exact numbers. For the roadmap-7.2 crosswalk (score our layouts
   under THEIR judges), run the real analyzers; this scorer is for optimization and
   directional agreement checks.

Units: dimensionless pattern score scaled so qwerty ≈ O(100); the ``--oxey-weight`` knob
maps it into fitness-comparable magnitude the same way the comfort knob does.
"""

from __future__ import annotations

from collections.abc import Mapping

from keybo.features import classify as C
from keybo.layout import Layout
from keybo.scoring.base import IScorer

#: name -> (signed weight per corpus-share PERCENT, why). Positive = penalty, negative =
#: reward, mirroring community analyzers' sign conventions. Opinions, documented.
DEFAULT_OXEY_WEIGHTS: dict[str, tuple[float, str]] = {
    "sfb": (
        12.0,
        "same-finger bigrams: the community's cardinal sin; our data agrees it is the "
        "largest measured bigram penalty (+27..38ms by skill)",
    ),
    "dsfb": (
        5.0,
        "disjoint SFBs (same finger at distance 2, skipgram): penalized by every "
        "community analyzer; our lag-2 probe measured it speed-NEUTRAL — kept because "
        "this scorer reproduces community judgment, not our measurements",
    ),
    "lsb": (3.0, "lateral stretch bigrams pull the hand off anchor"),
    "scissor": (4.0, "adjacent-finger two-row reaches"),
    "inroll": (
        -2.0,
        "inward rolls rewarded: community prizes them; our data shows same-hand "
        "continuation is genuinely sub-additive (-22ms pooled, skill-scaled)",
    ),
    "outroll": (-1.0, "outward rolls rewarded, less than inward (community convention)"),
    "onehand": (-1.5, "three keys, one hand, one direction — the smoothest trigram class"),
    "redirect": (
        2.0,
        "same-hand direction reversal: penalized by all community analyzers; our data "
        "measured it time-NEUTRAL beyond its bigrams — kept as community judgment",
    ),
    "bad_redirect": (4.0, "redirect with no index finger involved — community's worst trigram"),
    "alternate": (-0.5, "hand alternation mildly rewarded (dvorak-school value)"),
    "imbalance": (
        1.5,
        "hand-load imbalance percent (|left-right| share): balanced hands preferred",
    ),
}


class OxeyStyleScorer(IScorer):
    """Community-heuristic pattern score (lower = better)."""

    def __init__(
        self,
        bigram_freqs: Mapping[str, int],
        skipgram_freqs: Mapping[str, int],
        trigram_freqs: Mapping[str, int],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self._bg = dict(bigram_freqs)
        self._sg = dict(skipgram_freqs)
        self._tg = dict(trigram_freqs)
        self._w = {name: w for name, (w, _why) in DEFAULT_OXEY_WEIGHTS.items()}
        if weights:
            unknown = set(weights) - set(self._w)
            if unknown:
                raise ValueError(f"unknown oxey weight(s): {sorted(unknown)}")
            self._w.update(weights)

    def pattern_shares(self, layout: Layout) -> dict[str, float]:
        """Corpus-share percentages per pattern class (the analyzer-style stat block)."""
        g = layout.geometry
        shares = {name: 0.0 for name in self._w}
        # --- bigram patterns ---
        bg_total = 0.0
        hand_load = {-1: 0.0, 1: 0.0}
        for bg, f in self._bg.items():
            if len(bg) != 2 or not all(layout.has_key(c) for c in bg):
                continue
            a, b = layout.pos(bg[0]), layout.pos(bg[1])
            bg_total += f
            for pos in (a, b):
                h = g.hand(pos[0])
                if h:
                    hand_load[h] += f / 2
            cls = C.classify_positions(g, a, b)
            if cls is C.BigramClass.SAME_FINGER and a != b:
                shares["sfb"] += f
            elif cls is C.BigramClass.ALTERNATE:
                shares["alternate"] += f
            if C.is_lsb(g, a, b):
                shares["lsb"] += f
            if C.is_scissor(g, a, b):
                shares["scissor"] += f
            if C.is_inwards(g, a, b):
                shares["inroll"] += f
            if C.is_outwards(g, a, b):
                shares["outroll"] += f
        # --- skipgram patterns (disjoint sfb) ---
        sg_total = 0.0
        for sg, f in self._sg.items():
            if len(sg) != 2 or not all(layout.has_key(c) for c in sg):
                continue
            a, b = layout.pos(sg[0]), layout.pos(sg[1])
            sg_total += f
            if g.same_finger(a[0], b[0]) and a != b:
                shares["dsfb"] += f
        # --- trigram patterns ---
        tg_total = 0.0
        for tg, f in self._tg.items():
            if len(tg) != 3 or not all(layout.has_key(c) for c in tg):
                continue
            a, b, c3 = (layout.pos(ch) for ch in tg)
            tg_total += f
            ha, hb, hc = g.hand(a[0]), g.hand(b[0]), g.hand(c3[0])
            if ha == hb == hc and ha != 0:
                d1 = abs(b[0]) - abs(a[0])
                d2 = abs(c3[0]) - abs(b[0])
                if d1 and d2 and (d1 > 0) == (d2 > 0):
                    shares["onehand"] += f
                elif d1 and d2:
                    shares["redirect"] += f
                    if not any(abs(p[0]) in (1, 2) for p in (a, b, c3)):
                        shares["bad_redirect"] += f
        # normalize to percents of their own corpus
        for k in ("sfb", "alternate", "lsb", "scissor", "inroll", "outroll"):
            shares[k] = 100.0 * shares[k] / bg_total if bg_total else 0.0
        shares["dsfb"] = 100.0 * shares["dsfb"] / sg_total if sg_total else 0.0
        for k in ("onehand", "redirect", "bad_redirect"):
            shares[k] = 100.0 * shares[k] / tg_total if tg_total else 0.0
        total_hand = hand_load[-1] + hand_load[1]
        shares["imbalance"] = (
            100.0 * abs(hand_load[-1] - hand_load[1]) / total_hand if total_hand else 0.0
        )
        return shares

    def fitness(self, layout: Layout) -> float:
        shares = self.pattern_shares(layout)
        return sum(self._w[name] * share for name, share in shares.items())
