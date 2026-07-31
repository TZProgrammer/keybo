"""Oxeylyzer-style heuristic scorer — the community's judgment as an IScorer (7.2).

An explicit, documented APPROXIMATION of the pattern-count scoring used by the
oxeylyzer/genkey family of community analyzers: corpus-weighted percentages of
same-finger bigrams, disjoint SFBs (skipgrams), lateral stretches, scissors, rolls
(rewarded), onehands (rewarded), redirects, and finger imbalance — combined with signed
weights into one scalar (lower = better, matching our fitness convention).

Three honesty notes, load-bearing:

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
3. The WEIGHTS are ours; the trigram CLASSES are upstream's. The approximation in note 2
   is a licence to choose prices, not to invent a partition: which trigrams count as a
   redirect is a definition the community owns. So the trigram classifier is not written
   here — :func:`_trigram_class` delegates to :func:`keybo.analysis.community._v1_pattern`,
   the oxeylyzer-1 port pinned integer-exact to the real repl by
   ``tests/analysis/test_kan1_parity.py``. Its four redirect classes are rolled onto this
   scorer's two weight keys (``redirects``/``redirects_sfs`` -> ``redirect``,
   ``bad_redirects``/``bad_redirects_sfs`` -> ``bad_redirect``); the roll-up is the same one
   :mod:`keybo.analysis.redirects` publishes as ``bad_redirects_total``, and that module
   remains the place to read upstream's four classes separately.

   ⚠️ This scorer previously carried its OWN trigram predicates, and they were wrong three
   ways against the port one module away: the ``bad_redirect`` counter was NESTED inside
   ``redirect`` (so a bad redirect paid 2.0 + 4.0 = 6.0, not the 4.0 the dict displays,
   where upstream dispatches exactly one class per trigram); the direction step was
   ``abs(b.column) - abs(a.column)``, which reads the index finger's two columns as a
   direction STEP even though :meth:`keybo.geometry.Geometry.same_finger` calls them one
   finger; and Sft/Sfb triples were not excluded at all. Over the 27,000 ordered slot
   triples of ``ROW_STAGGERED_30`` that inflated ``onehand`` by 1.4286x (1080 vs 756) and
   the ``redirect`` term by 432 triples, and double-charged 540. The bad-redirect SUPPORT
   was already right (540 either way): the old ``abs(column) in (1, 2)`` test happens to be
   equivalent to "no index finger" on this geometry, so it was a fragile proxy rather than a
   live error. Corpus consequence of the whole repair, on blend-v1: every ``oxey-style``
   score drops by 0.42 to 1.50 (−1.6% of |score| on qwerty30m, −152% on the near-zero arm E),
   spearman(before, after) is 0.997059 over 16 layouts, and the nine layouts of the published
   adoption tables keep an IDENTICAL ordering (spearman 1.000000, 0 of 36 pairwise
   inversions). See ``tests/scoring/test_oxey_trigram_partition.py``, which asserts class
   membership against ``_v1_pattern`` triple-for-triple rather than eyeballing totals.

4. ⚠️ **``inroll``/``outroll`` are NOT stroke directions, and the honest pair sits beside
   them.** Both delegate to :func:`keybo.features.classify.is_inwards` /
   :func:`~keybo.features.classify.is_outwards`, which sort the two keys by column magnitude
   and compare ROWS — discarding which key was struck first. Measured on
   ``ROW_STAGGERED_30``: over all 870 ordered position pairs, the number whose verdict
   changes when the pair is swapped is **0**, for both predicates. So a corpus in which every
   n-gram is reversed — every inward stroke turned outward — produces ``inroll`` and
   ``outroll`` values that are **bit-identical**, moving by exactly 0.00e+00. They measure
   *outer-key-on-the-higher-row*, which is a real geometric distinction wearing a
   direction-of-travel name (:mod:`keybo.analysis.effect_curves` renamed its own copies
   ``outer_high``/``outer_low`` for exactly this reason).

   :data:`ORDERED_ROLL_SHARES` (``inroll_ordered``/``outroll_ordered``) are the honest
   counterparts, from :func:`~keybo.features.classify.is_inwards_ordered` /
   :func:`~keybo.features.classify.is_outwards_ordered`. They read ``|column|`` in stroke
   order, so reversing the corpus swaps them, and they also cover the 108 same-row rolls the
   row comparison drops (162 inward / 162 outward, partitioning all 324 roll-eligible K30
   pairs, against 108 / 108 / 108-unclassed for the pair above).

   The old names are kept and the new shares added ALONGSIDE, at **weight 0** — they are
   absent from :data:`DEFAULT_OXEY_WEIGHTS` by design. Two reasons, one per direction: the
   published ``inroll``/``outroll`` numbers throughout this project's ledger stay
   reproducible (a silent renumbering of shipped gauges is worse than a badly-named gauge),
   and the weight table is a *community-preference* statement whose values were chosen to
   reproduce the community's layout ORDERING — pricing a new class is a separate decision
   from measuring it. ``tests/scoring/test_oxey_corpus_reversal.py`` pins both halves: the
   ordered shares must move under reversal, the weighted ones must not, and ``fitness`` must
   be unchanged.

Units: dimensionless pattern score scaled so qwerty ≈ O(100); the ``--oxey-weight`` knob
maps it into fitness-comparable magnitude the same way the comfort knob does.
"""

from __future__ import annotations

from collections.abc import Mapping

from keybo.analysis.community import _v1_pattern
from keybo.features import classify as C
from keybo.geometry import Finger, Geometry, Position
from keybo.layout import Layout
from keybo.scoring.base import IScorer

#: :class:`keybo.geometry.Finger` -> the libdof finger enum ``_v1_pattern`` speaks
#: (LP=0 LR=1 LM=2 LI=3 RI=6 RM=7 RR=8 RP=9). The thumb has no libdof letter-key value and
#: cannot appear in a same-hand letter trigram, so it maps to ``None`` and short-circuits.
_LIBDOF_FINGER: dict[Finger, int] = {
    Finger.LP: 0,
    Finger.LR: 1,
    Finger.LM: 2,
    Finger.LI: 3,
    Finger.RI: 6,
    Finger.RM: 7,
    Finger.RR: 8,
    Finger.RP: 9,
}

#: ``_v1_pattern``'s four redirect labels rolled onto this scorer's two weight keys, plus
#: its one-hand label. The classes are MUTUALLY EXCLUSIVE (``_v1_pattern`` returns one label
#: per trigram), so this dict is a partition, not a set of overlapping tests — see the
#: module docstring's honesty note 3. Every other ``_v1_pattern`` label (rolls, alternation,
#: Sft/Sfb -> ``None``) is not a class this scorer weighs and is dropped.
_TRIGRAM_CLASS: dict[str, str] = {
    "onehands": "onehand",
    "redirects": "redirect",
    "redirects_sfs": "redirect",
    "bad_redirects": "bad_redirect",
    "bad_redirects_sfs": "bad_redirect",
}

#: Order-aware roll shares, reported by :meth:`OxeyStyleScorer.pattern_shares` but
#: deliberately ABSENT from :data:`DEFAULT_OXEY_WEIGHTS`, so they are measured at weight 0
#: and cannot move any published ``oxey-style`` score. See the module docstring's fourth
#: honesty note for why they are separate from ``inroll``/``outroll`` rather than replacing
#: them.
ORDERED_ROLL_SHARES: tuple[str, ...] = ("inroll_ordered", "outroll_ordered")

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
    "bad_redirect": (
        4.0,
        "redirect with no index finger involved — community's worst trigram; EXCLUSIVE of "
        "`redirect` (upstream assigns one class per trigram), so 4.0 is the whole price",
    ),
    "alternate": (-0.5, "hand alternation mildly rewarded (dvorak-school value)"),
    "imbalance": (
        1.5,
        "hand-load imbalance percent (|left-right| share): balanced hands preferred",
    ),
}


def _trigram_class(g: Geometry, a: Position, b: Position, c: Position) -> str | None:
    """This scorer's trigram class for three key positions, or ``None`` for no class.

    Delegates the classification to :func:`keybo.analysis.community._v1_pattern` — the
    oxeylyzer-1 port that ``tests/analysis/test_kan1_parity.py`` gates integer-exact against
    the real upstream repl — and rolls its four redirect labels onto this scorer's two
    redirect weights via :data:`_TRIGRAM_CLASS`.

    Delegating rather than re-deriving is the point: this module used to carry its own
    predicates, and they disagreed with the parity-gated ones three ways (nested rather than
    exclusive redirect counters; a direction step computed on ``abs(column)``, which reads the
    index finger's two columns as a step; and no Sft/Sfb exclusion). A hand-rolled
    reimplementation of a validated classifier loses the validation, so there is deliberately
    only one classifier in the repo and this is a translation layer into it.
    """
    fingers = []
    for position in (a, b, c):
        finger = _LIBDOF_FINGER.get(g.finger(position[0]))
        if finger is None:  # thumb/space: not a letter-key trigram upstream classifies
            return None
        fingers.append(finger)
    return _TRIGRAM_CLASS.get(_v1_pattern(*fingers))


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
        """Corpus-share percentages per pattern class (the analyzer-style stat block).

        Includes the two unweighted :data:`ORDERED_ROLL_SHARES` alongside the weighted
        classes. They are reported because they are the only roll shares in this block that
        respond to stroke ORDER — reversing every n-gram in the corpus leaves
        ``inroll``/``outroll`` bit-identical and swaps these two.
        """
        g = layout.geometry
        shares = {name: 0.0 for name in (*self._w, *ORDERED_ROLL_SHARES)}
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
            # UNORDERED (outer-key-on-the-higher-row): the weighted, published pair.
            if C.is_inwards(g, a, b):
                shares["inroll"] += f
            if C.is_outwards(g, a, b):
                shares["outroll"] += f
            # ORDERED (direction of travel): unweighted, additive, order-sensitive.
            if C.is_inwards_ordered(g, a, b):
                shares["inroll_ordered"] += f
            if C.is_outwards_ordered(g, a, b):
                shares["outroll_ordered"] += f
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
            name = _trigram_class(g, a, b, c3)
            if name is not None:
                shares[name] += f
        # normalize to percents of their own corpus
        for k in ("sfb", "alternate", "lsb", "scissor", "inroll", "outroll", *ORDERED_ROLL_SHARES):
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
        """Weighted sum over the WEIGHTED classes only (lower = better).

        Iterates ``self._w`` rather than the share dict, so the unweighted
        :data:`ORDERED_ROLL_SHARES` are reported by :meth:`pattern_shares` without entering
        any score. Summing over the shares instead would either raise ``KeyError`` on the new
        keys or — worse, if defaulted to 0.0 — silently start pricing them the moment someone
        added a weight, renumbering every published ``oxey-style`` value.
        """
        shares = self.pattern_shares(layout)
        return sum(weight * shares[name] for name, weight in self._w.items())
