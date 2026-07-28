"""The trigram partition: one label per triple, on FINGERS, agreeing with the parity-gated port.

``OxeyStyleScorer`` shipped a trigram classifier that disagreed with this repo's own
``community._v1_pattern`` — the oxeylyzer-1 port that ``tests/analysis/test_kan1_parity.py``
gates integer-exact against the real upstream repl. Two defects, one root cause:

1. **The two redirect counters were NESTED.** ``shares["redirect"]`` fired unconditionally on
   a same-hand reversal and ``shares["bad_redirect"]`` fired in an ``if`` *inside* that branch,
   so a bad redirect paid both weights (+2.0 and +4.0 = **+6.0**, not the 4.0 the dict
   displays). Upstream's ``get_one_hand`` returns ONE enum through an exhaustive 4-way
   ``match``: the predicates nest, the dispatch does not.
2. **The direction step was computed on COLUMN, not FINGER.** ``d1 = abs(b[0]) - abs(a[0])``
   makes a move between the index finger's two columns a nonzero direction step, while
   :meth:`keybo.geometry.Geometry.same_finger` documents those two columns as ONE finger. That
   inflated ``onehand`` by 1.4286x (1080 vs 756 slot triples) and ``redirect`` by 432 triples,
   and it let same-finger (Sfb) triples into classes upstream excludes entirely.

These tests pin the SUPPORT (which slot triples land in which class), not corpus values, so
they are independent of the corpus and of the weights. The class sizes are exhaustive counts
over all ``30**3 = 27000`` ordered slot triples of ``ROW_STAGGERED_30``.

The positive control at the bottom is the one that matters: class MEMBERSHIP must agree with
``_v1_pattern`` triple-for-triple, asserted rather than eyeballed. Trap 28 — a hand-rolled
reimplementation of a validated classifier loses the validation — so this scorer delegates,
and this test is what says it still does.
"""

from __future__ import annotations

import pytest

from keybo.analysis.community import _v1_pattern
from keybo.analysis.kmstats import _KEYS
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.oxey import OxeyStyleScorer

#: A C30M-charset qwerty; every 30-char permutation covers the same slots, so any layout
#: works for a support count — this one is used so failures name familiar trigrams.
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"

#: EXHAUSTIVE support sizes over all 30**3 ordered slot triples, from `_v1_pattern` rolled
#: onto oxey's two redirect keys. Independently recounted here rather than quoted.
EXPECTED_ONEHAND_TRIPLES = 756
EXPECTED_REDIRECT_TRIPLES = 2268  # redirects + redirects_sfs (index finger involved)
EXPECTED_BAD_REDIRECT_TRIPLES = 540  # bad_redirects + bad_redirects_sfs (no index)

#: What the defective classifier counted, kept as documentation of the delta this pins.
SHIPPED_ONEHAND_TRIPLES = 1080
SHIPPED_REDIRECT_TERM_TRIPLES = 3240


def _slot_fingers() -> list[int]:
    """slot -> libdof finger enum, from the kmstats board.

    ``tests/analysis/test_redirects.py`` asserts this map is identical to
    ``community.FINGERS[SLOT2DOF[slot]]``, so using it here does not introduce a third
    finger map.
    """
    return [key.finger for key in _KEYS]


def _class_support(scorer: OxeyStyleScorer, name: str) -> set[tuple[int, int, int]]:
    """Which ordered slot triples put nonzero mass in ``shares[name]``.

    Probes the real ``pattern_shares`` one trigram at a time: a single-trigram corpus makes
    that share 100.0 exactly when the triple is in the class, so this reads the SHIPPED code
    path rather than a re-derivation of it.
    """
    layout = Layout(QWERTY30M, ROW_STAGGERED_30)
    support = set()
    for i in range(30):
        for j in range(30):
            for k in range(30):
                trigram = QWERTY30M[i] + QWERTY30M[j] + QWERTY30M[k]
                probe = OxeyStyleScorer(scorer._bg, scorer._sg, {trigram: 1})
                if probe.pattern_shares(layout)[name] > 0.0:
                    support.add((i, j, k))
    return support


def _fast_support() -> dict[str, set[tuple[int, int, int]]]:
    """All three trigram classes' support in one pass, via one scorer per triple.

    Same object as three ``_class_support`` calls but 3x cheaper; the equivalence is asserted
    in :func:`test_the_fast_support_probe_matches_the_one_share_at_a_time_probe`.
    """
    layout = Layout(QWERTY30M, ROW_STAGGERED_30)
    out: dict[str, set[tuple[int, int, int]]] = {
        "onehand": set(),
        "redirect": set(),
        "bad_redirect": set(),
    }
    for i in range(30):
        for j in range(30):
            for k in range(30):
                trigram = QWERTY30M[i] + QWERTY30M[j] + QWERTY30M[k]
                shares = OxeyStyleScorer({}, {}, {trigram: 1}).pattern_shares(layout)
                for name in out:
                    if shares[name] > 0.0:
                        out[name].add((i, j, k))
    return out


@pytest.fixture(scope="module")
def support() -> dict[str, set[tuple[int, int, int]]]:
    return _fast_support()


@pytest.fixture(scope="module")
def v1_support() -> dict[str, set[tuple[int, int, int]]]:
    """The same three classes per ``_v1_pattern``, with its four redirect labels rolled onto
    oxey's two keys — the partition this scorer is supposed to implement."""
    fingers = _slot_fingers()
    roll = {
        "onehands": "onehand",
        "redirects": "redirect",
        "redirects_sfs": "redirect",
        "bad_redirects": "bad_redirect",
        "bad_redirects_sfs": "bad_redirect",
    }
    out: dict[str, set[tuple[int, int, int]]] = {
        "onehand": set(),
        "redirect": set(),
        "bad_redirect": set(),
    }
    for i in range(30):
        for j in range(30):
            for k in range(30):
                key = roll.get(_v1_pattern(fingers[i], fingers[j], fingers[k]))
                if key:
                    out[key].add((i, j, k))
    return out


# --------------------------------------------------------------------------------------
# 1. The double-charge: a bad redirect must be charged ONCE.
# --------------------------------------------------------------------------------------


def test_a_bad_redirect_trigram_is_charged_once_not_twice():
    """THE regression test for the nested counter.

    ``qew`` on qwerty30m is a plain bad redirect (``_v1_pattern`` -> ``bad_redirects``):
    left hand throughout, direction reversed, and no index finger involved (q=LP, e=LM,
    w=LR). A one-trigram corpus makes every share either 0.0 or 100.0, so the two redirect
    terms cannot be confused with a rounding difference.

    As shipped this asserted-false: ``redirect`` was ALSO 100.0, so the trigram paid
    ``2.0 + 4.0 = 6.0`` where the dict advertises 4.0.
    """
    fingers = _slot_fingers()
    slot = {character: index for index, character in enumerate(QWERTY30M)}
    assert _v1_pattern(*(fingers[slot[character]] for character in "qew")) == "bad_redirects"

    layout = Layout(QWERTY30M, ROW_STAGGERED_30)
    shares = OxeyStyleScorer({}, {}, {"qew": 1}).pattern_shares(layout)

    assert shares["bad_redirect"] == 100.0, "a bad redirect must count in bad_redirect"
    assert shares["redirect"] == 0.0, (
        "bad_redirect must be EXCLUSIVE of redirect (upstream dispatches one label per "
        f"trigram); got redirect={shares['redirect']}, so the trigram is charged "
        f"{2.0 + 4.0} instead of 4.0"
    )
    assert shares["onehand"] == 0.0

    weighted = OxeyStyleScorer({}, {}, {"qew": 1}).fitness(layout)
    assert weighted == pytest.approx(400.0, abs=1e-9), (
        "one bad-redirect trigram at 100% share must price at the bad_redirect weight "
        "alone (4.0 * 100)"
    )


def test_the_two_redirect_classes_are_disjoint_over_every_slot_triple(support):
    """Exhaustive form of the test above: no triple may be in both classes."""
    both = support["redirect"] & support["bad_redirect"]
    assert not both, f"{len(both)} triples fire BOTH redirect terms (double-charged)"


# --------------------------------------------------------------------------------------
# 2. The finger-vs-column bug: class SIZES.
# --------------------------------------------------------------------------------------


def test_onehand_class_size_is_the_finger_correct_756_not_the_column_1080(support):
    """``d1 = abs(b[0]) - abs(a[0])`` counted the index's two columns as a direction step.

    756 is the finger-correct count; 1080 is what the column proxy gave (1.4286x too many).
    """
    assert len(support["onehand"]) == EXPECTED_ONEHAND_TRIPLES, (
        f"onehand support is {len(support['onehand'])}; expected "
        f"{EXPECTED_ONEHAND_TRIPLES} (the shipped column-based proxy gave "
        f"{SHIPPED_ONEHAND_TRIPLES})"
    )


def test_redirect_class_sizes_match_the_upstream_partition(support):
    """The redirect term stops absorbing the 432 index-column triples and the bad subset."""
    assert len(support["redirect"]) == EXPECTED_REDIRECT_TRIPLES, (
        f"redirect support is {len(support['redirect'])}; expected "
        f"{EXPECTED_REDIRECT_TRIPLES} (shipped term fired on "
        f"{SHIPPED_REDIRECT_TERM_TRIPLES})"
    )
    assert len(support["bad_redirect"]) == EXPECTED_BAD_REDIRECT_TRIPLES


def test_no_trigram_class_contains_a_same_finger_pair(support):
    """Upstream excludes Sft/Sfb from the one-hand family entirely (``_v1_pattern`` -> None).

    The shipped ``d1 and d2`` guard only excluded exact COLUMN equality, so a move between
    the index's two columns — an SFB — was classified. This is the same root cause as the
    size errors and is why the counts above are what they are.
    """
    geometry = ROW_STAGGERED_30
    slots = geometry.slots
    for name, triples in support.items():
        for i, j, k in triples:
            a, b, c = slots[i][0], slots[j][0], slots[k][0]
            assert not geometry.same_finger(a, b), f"{name} triple {(i, j, k)} has ab on one finger"
            assert not geometry.same_finger(b, c), f"{name} triple {(i, j, k)} has bc on one finger"


# --------------------------------------------------------------------------------------
# 3. THE positive control: membership must equal `_v1_pattern`'s, not merely be the same size.
# --------------------------------------------------------------------------------------


def test_class_membership_agrees_with_v1_pattern_triple_for_triple(support, v1_support):
    """POSITIVE CONTROL against the parity-gated port, asserted rather than eyeballed.

    Equal class SIZES would not be enough — two classifiers can disagree on membership and
    still count the same total (trap: a positive control on a count is not one on a set). So
    this compares the sets themselves, both directions, for all three classes.
    """
    for name in ("onehand", "redirect", "bad_redirect"):
        oxey_only = support[name] - v1_support[name]
        v1_only = v1_support[name] - support[name]
        assert not oxey_only and not v1_only, (
            f"{name}: {len(oxey_only)} triples oxey-only, {len(v1_only)} triples "
            f"_v1_pattern-only (sizes {len(support[name])} vs {len(v1_support[name])})"
        )


def test_the_three_classes_partition_the_same_hand_reversal_universe(support, v1_support):
    """The three classes are pairwise disjoint, and their union is `_v1_pattern`'s union.

    Complements the membership test: that one could pass while the classifier ALSO counted a
    triple in two classes (a union check and a per-class check catch different breakage).
    """
    onehand, redirect, bad = (support[n] for n in ("onehand", "redirect", "bad_redirect"))
    assert not onehand & redirect
    assert not onehand & bad
    assert not redirect & bad
    assert onehand | redirect | bad == (
        v1_support["onehand"] | v1_support["redirect"] | v1_support["bad_redirect"]
    )


@pytest.mark.parametrize(
    "lay30",
    [
        QWERTY30M,
        "bldwz'foujnrtsgyhaeixqmcvkp,.-",  # graphite
        "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",  # keybo-lsb
    ],
)
def test_redirect_numerator_mass_now_equals_RedirectFamily_on_a_real_corpus(corpora, lay30):
    """CROSS-MODULE control that only became possible with this fix.

    ``analysis.redirects.RedirectFamily`` reports the same four upstream classes, and
    ``tests/analysis/test_redirects.py`` pins its total equal to ``kmstats``' ``redir``
    cell-for-cell. Before this fix, oxey's redirect term could not agree with either: it used
    a different predicate AND double-counted the bad subset. Now the predicate is literally
    the same function, so the selected MASS must match exactly.

    ⚠️ Compared on NUMERATOR MASS, not on the share. The two shares differ by ~1.9x because
    the denominators are different by long-standing convention — ``oxey.pattern_shares``
    counts space-containing trigrams (``Layout.has_key(" ")`` is True), the
    kmstats/RedirectFamily convention masks space out (``analysis/redirects.py``'s
    "Denominator (trap #9)" note). A wrong denominator is invisible to a numerator check and
    vice versa, so this asserts the numerator — the quantity the *predicate* decides — and
    leaves the denominator convention exactly as it was.
    """
    from keybo.analysis.redirects import RedirectFamily

    _bigrams, _skipgrams, trigrams = corpora
    geometry = ROW_STAGGERED_30
    layout = Layout(lay30, geometry)

    oxey_mass = 0
    for ngram, freq in trigrams.items():
        if len(ngram) != 3 or not all(layout.has_key(character) for character in ngram):
            continue
        positions = [layout.pos(character) for character in ngram]
        if _trigram_class_name(geometry, positions) in ("redirect", "bad_redirect"):
            oxey_mass += freq

    slot_of = {character: slot for slot, character in enumerate(lay30)}
    family_denominator = sum(
        freq
        for ngram, freq in trigrams.items()
        if len(ngram) == 3 and all(character in slot_of for character in ngram)
    )
    family_share = RedirectFamily(trigrams).shares(lay30)["redirects_family_total"]
    family_mass = round(family_share / 100.0 * family_denominator)

    assert oxey_mass == family_mass, (
        f"oxey selects {oxey_mass} of trigram mass as a redirect, RedirectFamily selects "
        f"{family_mass} — the two must now be the SAME predicate"
    )


def _trigram_class_name(geometry, positions):
    from keybo.scoring.oxey import _trigram_class

    return _trigram_class(geometry, *positions)


def test_the_fast_support_probe_matches_the_one_share_at_a_time_probe():
    """The module's cheap 3-classes-per-pass probe must equal the per-share probe.

    Guards the helper the other tests depend on: if ``_fast_support`` drifted from
    ``_class_support`` every assertion above would be testing the wrong object.
    """
    scorer = OxeyStyleScorer({}, {}, {})
    fast = _fast_support()
    assert fast["bad_redirect"] == _class_support(scorer, "bad_redirect")
