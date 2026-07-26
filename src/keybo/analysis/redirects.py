"""The oxeylyzer redirect family as reportable columns, including bad-redirect (ALLGAUGE-1).

``kmstats`` reports one aggregate ``redir``. oxeylyzer-1 splits the same event into four
classes, and the community's settled definition of the worst of them already lives in this
repo — :func:`keybo.analysis.community._v1_pattern`, with ``_BAD = {0, 1, 2, 7, 8, 9}``
(every finger except the two indices; the thumb never appears in a letter trigram):

============================  ==============================================  ======
class                         same-hand trigram, direction reversed, and...   weight
============================  ==============================================  ======
``redirects``                 at least one index finger involved               -340
``redirects_sfs``             ...and the first and third key share a finger    -420
``bad_redirects``             **no** index finger involved                     -490
``bad_redirects_sfs``         ...and the first and third key share a finger    -550
============================  ==============================================  ======

(weights: ``Oxeylyzer1.WT``, community.py — oxeylyzer-1's own taste table, quoted here
only to document which class the community considers worse. This module reports **mass**,
not weighted score; no severity scheme is invented.)

**The four classes are mutually exclusive, including the ``_sfs`` pairs.**
``_v1_pattern`` returns exactly one label per trigram — ``"bad_redirects_sfs" if sfs else
"bad_redirects"`` — so ``bad_redirects_sfs`` is a *sibling* of ``bad_redirects``, not a
subset of it. (Reading the ``_sfs`` rows as a refinement *inside* the plain class is the
natural mistake, and it is wrong in the direction that matters: on qwerty
``bad_redirects_sfs`` = 1.008% actually EXCEEDS ``bad_redirects`` = 0.425%.) So the total
bad-redirect mass a reader wants is ``bad_redirects_total`` — the two summed — and
``redirects_family_total`` is the sum of all four.

**Relationship to kmstats ``redir`` — verified, not assumed.** The two predicates are
written differently (``kmstats._is_redirect`` compares ``_direction`` on the two bigrams;
``_v1_pattern`` compares ``(f1 < f2) == (f2 > f3)``), and the natural guess is that the
oxeylyzer family is a *subset*. It is not a subset — it is **equal**. Exhaustively over all
30**3 = 27000 slot triples on this board: 2808 triples satisfy both predicates, 0 satisfy
only one, and the two finger maps (``kmstats._COL_FINGER`` and
``community.FINGERS[SLOT2DOF[slot]]``) are identical. So on a shared denominator

    redir == redirects + redirects_sfs + bad_redirects + bad_redirects_sfs

exactly, and ``bad_redirects <= redir`` is the weak corollary rather than the fact.
``tests/cli/test_analyze_allgauge.py`` asserts the equality (and
``tests/analysis/test_redirects.py`` re-derives the exhaustive enumeration), because "the
family is nested inside redir" would be a *plausible* claim that happens to understate the
truth — and a nested pair of legs is one leg, not two.

**Denominator (trap #9): the layout-restricted TRIGRAM mass** — only trigrams whose all
three characters sit on the layout count, toward numerator and denominator alike. That is
exactly :meth:`keybo.analysis.kmstats.KmStats.stats`' denominator for its trigram metrics
(``tri_total``), which is what makes the equality above hold cell-for-cell against the
reported ``redir`` rather than up to a constant. Note it is NOT the denominator
``oxey.pattern_shares`` uses for its own ``redirect`` share (that one counts the same
trigrams but its predicate is a third, simpler one), so the two are not interchangeable.
"""

from __future__ import annotations

from collections.abc import Mapping

from keybo.analysis.community import _v1_pattern
from keybo.analysis.kmstats import _KEYS

#: The four redirect classes oxeylyzer-1 distinguishes, worst last.
REDIRECT_CLASSES: tuple[str, ...] = (
    "redirects",
    "redirects_sfs",
    "bad_redirects",
    "bad_redirects_sfs",
)

#: oxeylyzer-1's taste weights for the four classes (documentation only — this module
#: reports mass, and inventing a severity scheme for redirects is explicitly out of scope).
CLASS_WEIGHTS: dict[str, int] = {
    "redirects": -340,
    "redirects_sfs": -420,
    "bad_redirects": -490,
    "bad_redirects_sfs": -550,
}


class RedirectFamily:
    """Corpus mass per oxeylyzer redirect class, for 30-char layouts over one corpus."""

    def __init__(self, trigram_freqs: Mapping[str, int]) -> None:
        self._tri = {ng: freq for ng, freq in trigram_freqs.items() if len(ng) == 3}
        # slot -> finger enum, from the kmstats board (identical to community's map;
        # asserted in tests/analysis/test_redirects.py).
        self._finger = [key.finger for key in _KEYS]

    def shares(self, lay30: str) -> dict[str, float]:
        """Per-class share, in percent of layout-restricted trigram mass.

        Also returns two roll-ups, because the four classes are mutually exclusive and a
        reader almost always wants a sum rather than one leg:

        * ``bad_redirects_total`` = ``bad_redirects + bad_redirects_sfs`` — all
          no-index-finger redirects, the community's worst trigram class;
        * ``redirects_family_total`` = all four classes, which equals the ``redir`` share
          ``kmstats`` reports (see the module docstring).
        """
        slot_of = {character: slot for slot, character in enumerate(lay30)}
        sums = dict.fromkeys(REDIRECT_CLASSES, 0.0)
        total = 0
        for ngram, freq in self._tri.items():
            slots = [slot_of.get(character) for character in ngram]
            if any(slot is None for slot in slots):
                continue
            total += freq
            pattern = _v1_pattern(*(self._finger[slot] for slot in slots))
            if pattern in sums:
                sums[pattern] += freq
        if not total:
            return {
                **dict.fromkeys(REDIRECT_CLASSES, 0.0),
                "bad_redirects_total": 0.0,
                "redirects_family_total": 0.0,
            }
        out = {name: 100.0 * value / total for name, value in sums.items()}
        out["bad_redirects_total"] = out["bad_redirects"] + out["bad_redirects_sfs"]
        out["redirects_family_total"] = sum(out[name] for name in REDIRECT_CLASSES)
        return out
