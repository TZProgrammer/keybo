"""The ``sg_dist`` gauge: the corpus-weighted first-to-third-key span of a trigram (BUILDMETRIC-1).

``sg_dist(L) = Σ_{abc} f(abc)·distance(a, c) / M`` — the frequency-weighted mean Euclidean
distance from a trigram's FIRST key to its THIRD, over the layout-restricted trigram mass. It
answers, for the average three-key burst: *does the hand come back near where it started (small
span — a roll that returns) or sweep away and land far off (large span)?*

Why this is a distinct gauge and not a restatement of an existing one (BUILDMETRIC-1, verified
on 1,549 unselected perturbation boards): the 15 shipped gauges meter the first-to-third span
**only for the same-finger skip** (``sfs`` / ``sfs-dist``). ``sg_dist`` meters it for ALL
trigrams, and it is genuinely three-key — the a→c span is 98.8% NOT determined by the two
constituent hops ``distance(a,b)`` + ``distance(b,c)``, so it is not a convexity restatement of
the pairwise distance gauges. Its maximum absolute correlation with any of the 15 gauges is 0.54
(``sfs``). It is the layout-level shadow of the model's served ``sg_distance`` trigram feature
(``keybo.features.ngram`` / ``schema``): identical per-trigram quantity ``distance(a, c)``, same
geometry, aggregated to the layout with the corpus weights.

Three conventions this module commits to, each because getting it wrong changes the number:

**Geometry: ROW_STAGGERED_30, space-aware.** ``distance(a, c)`` is the board's plain Euclidean
key distance (``keybo.geometry.Geometry.distance``, ex=2), the SAME distance the served
``sg_distance`` feature uses — NOT the keymeow ANSI board ``kmstats`` scores its 11 statistics on
(different staggers and a different centre gap, so a different number). This is why the gauge is
assembled here on the ``Layout`` object rather than added to ``kmstats``: it is a
``ROW_STAGGERED_30`` quantity, like ``scissor`` / ``comfort`` / ``oxey-style``.

**Denominator (trap #9): the layout-restricted trigram mass, INCLUDING space.** A trigram counts
toward numerator and denominator iff all three of its characters sit on the layout, where space
is a real key at the thumb position. That is byte-identically the denominator ``ms/char`` divides
by (``keybo.analysis.timecard.TimeSurface.card``: ``ms_per_char = total / covered`` over the same
space-inclusive trigram set) — so ``sg_dist`` "uses the same denominator as ms/char" rather than
inventing a fourth mass convention. Space is load-bearing: the space bar sits far from the letter
block, so space-containing trigrams carry large a→c spans; excluding them moves keybo-lsb from
3.836 to 3.900 on blend-v1.

**Units: key-widths, not a percentage.** Every ``kmstats`` gauge is a percent share; ``sg_dist``
is a mean distance, reported raw (the value ``3.836`` a reader sees, not ``383.6``). The caller's
table notes the unit so the column is not misread as a share.
"""

from __future__ import annotations

from collections.abc import Mapping

from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.layout import Layout


def sg_dist(layout: Layout, trigram_freqs: Mapping[str, int]) -> float:
    """Corpus-weighted mean first-to-third-key span of ``layout`` over ``trigram_freqs``.

    ``trigram_freqs`` is a char-trigram -> count mapping (the corpus ``trigrams.txt`` table).
    Only trigrams whose three characters all sit on the layout participate — space included, at
    the geometry's fixed thumb position — and the weighted mean of ``distance(first, third)`` is
    taken over exactly that layout-restricted mass. Returns ``0.0`` for an empty support (a
    charset that covers no trigram), never a divide-by-zero.
    """
    geometry = layout.geometry
    # Positions memoized per character (incl. space) so a corpus scan does not re-walk the
    # layout's char->slot dict once per trigram.
    pos = {char: layout.pos(char) for char in layout.chars}
    pos[" "] = geometry.space_position

    weighted_span = 0.0
    mass = 0
    for ngram, freq in trigram_freqs.items():
        if len(ngram) != 3:
            continue
        first, _middle, third = ngram
        if first not in pos or third not in pos or ngram[1] not in pos:
            continue
        weighted_span += freq * geometry.distance(pos[first], pos[third])
        mass += freq
    return weighted_span / mass if mass else 0.0


# --- diagnostic FOILS (labelled; deliberately NOT shipped as gauges) ------------------------
#
# BUILDMETRIC-1 measured two sibling trigram-distance quantities and found them NOT
# trigram-authentic, so they are the ``reach`` category error's cousins and must never sit in
# the shipped gauge frame beside ``sg_dist``:
#
# * ``path_len_sq`` = Σf·(distance(a,b)² + distance(b,c)²)/M is 95% bigram-decomposable — a
#   convexity restatement of the existing per-bigram distance gauges, carrying no genuinely
#   three-key information.
# * ``max_hop`` = Σf·max(distance(a,b), distance(b,c))/M is 69% bigram-decomposable.
#
# They are provided ONLY so a caller can reproduce the trigram-authenticity contrast (the 85%
# NOT-explained-by-the-two-hops bar ``sg_dist`` clears and these fail); they are kept out of the
# gauge frame by construction — this module exposes them under an explicit ``_foil`` name and the
# analyze CLI never adds them to ``GAUGE_NAMES``.


def _foil_spans(layout: Layout, trigram_freqs: Mapping[str, int]) -> dict[str, float]:
    """The two bigram-decomposable trigram-distance foils, for the distinctness diagnostic only.

    NOT gauges: ``path_len_sq`` and ``max_hop`` are 95% / 69% recoverable from the two
    constituent hops (BUILDMETRIC-1), so shipping them would re-commit the ``reach`` category
    error. Same denominator convention as :func:`sg_dist`.
    """
    geometry = layout.geometry
    pos = {char: layout.pos(char) for char in layout.chars}
    pos[" "] = geometry.space_position

    path_len_sq = 0.0
    max_hop = 0.0
    mass = 0
    for ngram, freq in trigram_freqs.items():
        if len(ngram) != 3:
            continue
        a, b, c = ngram
        if a not in pos or b not in pos or c not in pos:
            continue
        hop1 = geometry.distance(pos[a], pos[b])
        hop2 = geometry.distance(pos[b], pos[c])
        path_len_sq += freq * (hop1 * hop1 + hop2 * hop2)
        max_hop += freq * max(hop1, hop2)
        mass += freq
    if not mass:
        return {"path_len_sq": 0.0, "max_hop": 0.0}
    return {"path_len_sq": path_len_sq / mass, "max_hop": max_hop / mass}


def sg_dist_from_string(
    lay30: str, trigram_freqs: Mapping[str, int], geometry: Geometry = ROW_STAGGERED_30
) -> float:
    """Convenience wrapper: build the ``Layout`` from a 30-char string, then score it."""
    return sg_dist(Layout(lay30, geometry), trigram_freqs)
