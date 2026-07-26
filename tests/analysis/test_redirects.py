"""The oxeylyzer redirect family, and its EXACT relationship to kmstats `redir`.

The relationship is re-derived here by exhaustive enumeration rather than asserted from
the docstring, because the whole point of the claim is that it is stronger than the
plausible guess (subset). If a future edit to either predicate breaks the equality, this
is the test that says so.
"""

from __future__ import annotations

import pytest

from keybo.analysis.community import FINGERS, SLOT2DOF, _v1_pattern
from keybo.analysis.kmstats import _KEYS, KmStats, _is_redirect
from keybo.analysis.redirects import REDIRECT_CLASSES, RedirectFamily

KEYBO_LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
GRAPHITE = "bldwz'foujnrtsgyhaeixqmcvkp,.-"


def test_the_two_finger_maps_are_identical():
    """kmstats and community must agree on which finger presses each slot.

    The redirect equality below is only meaningful if both predicates see the same board.
    """
    kmstats_fingers = [key.finger for key in _KEYS]
    community_fingers = [FINGERS[SLOT2DOF[slot]] for slot in range(30)]
    assert kmstats_fingers == community_fingers


def test_redirect_family_equals_kmstats_redir_over_every_slot_triple():
    """EXHAUSTIVE: all 30**3 triples. Not a subset — an equality."""
    fingers = [key.finger for key in _KEYS]
    family = set(REDIRECT_CLASSES)
    both = km_only = v1_only = 0
    for i in range(30):
        for j in range(30):
            for k in range(30):
                km = _is_redirect(_KEYS[i], _KEYS[j], _KEYS[k])
                v1 = _v1_pattern(fingers[i], fingers[j], fingers[k]) in family
                if km and v1:
                    both += 1
                elif km:
                    km_only += 1
                elif v1:
                    v1_only += 1
    assert km_only == 0, f"{km_only} triples are kmstats-redirects but not oxeylyzer-redirects"
    assert v1_only == 0, f"{v1_only} triples are oxeylyzer-redirects but not kmstats-redirects"
    assert both == 2808, f"redirect support changed size: {both}"


@pytest.mark.parametrize("layout", [KEYBO_LSB, QWERTY30M, GRAPHITE])
def test_family_total_equals_kmstats_redir_on_a_real_corpus(corpora, layout):
    """The set equality becomes a MASS equality on a shared denominator."""
    bigrams, skipgrams, trigrams = corpora
    stats = KmStats(bigrams, skipgrams, trigrams).stats(layout)
    shares = RedirectFamily(trigrams).shares(layout)
    assert shares["redirects_family_total"] == pytest.approx(stats["redir"], rel=0, abs=1e-9)


@pytest.mark.parametrize("layout", [KEYBO_LSB, QWERTY30M, GRAPHITE])
def test_the_four_classes_partition_the_family_total(corpora, layout):
    """Every class is bounded by the total, and the four sum to it exactly."""
    _bigrams, _skipgrams, trigrams = corpora
    shares = RedirectFamily(trigrams).shares(layout)
    for name in REDIRECT_CLASSES:
        assert 0.0 <= shares[name] <= shares["redirects_family_total"], name
    assert shares["redirects_family_total"] == pytest.approx(
        sum(shares[name] for name in REDIRECT_CLASSES), rel=0, abs=1e-12
    )
    assert shares["bad_redirects_total"] == pytest.approx(
        shares["bad_redirects"] + shares["bad_redirects_sfs"], rel=0, abs=1e-12
    )


def test_bad_redirects_sfs_is_a_SIBLING_of_bad_redirects_not_a_subset(corpora):
    """`_v1_pattern` returns ONE label, so the `_sfs` rows are excluded from the plain class.

    Pinned with a case where it actually bites: on qwerty the `_sfs` share is LARGER than
    the plain one, so a nesting assumption would not merely be imprecise, it would be
    false. This is why the report rolls the two up as `bad_redirects_total`.
    """
    _bigrams, _skipgrams, trigrams = corpora
    shares = RedirectFamily(trigrams).shares(QWERTY30M)
    assert shares["bad_redirects_sfs"] > shares["bad_redirects"]


def test_bad_redirects_involves_no_index_finger_by_construction():
    """The `_BAD` set is 'not an index finger'; spot-check the classification honours it."""
    fingers = [key.finger for key in _KEYS]
    index_fingers = {3, 6}
    for i in range(30):
        for j in range(30):
            for k in range(30):
                pattern = _v1_pattern(fingers[i], fingers[j], fingers[k])
                if pattern in ("bad_redirects", "bad_redirects_sfs"):
                    involved = {fingers[i], fingers[j], fingers[k]}
                    assert not (involved & index_fingers), (
                        f"bad redirect on slots {i},{j},{k} involves an index finger"
                    )


def test_sfs_variants_have_first_and_third_on_one_finger():
    fingers = [key.finger for key in _KEYS]
    for i in range(30):
        for j in range(30):
            for k in range(30):
                pattern = _v1_pattern(fingers[i], fingers[j], fingers[k])
                if pattern in ("redirects_sfs", "bad_redirects_sfs"):
                    assert fingers[i] == fingers[k]


def test_empty_corpus_yields_zeros_not_a_crash():
    shares = RedirectFamily({}).shares(KEYBO_LSB)
    assert shares["redirects_family_total"] == 0.0
    assert all(shares[name] == 0.0 for name in REDIRECT_CLASSES)


def test_layout_missing_corpus_characters_still_normalizes():
    """A layout that covers nothing scores 0, not a ZeroDivisionError."""
    shares = RedirectFamily({"xyz": 10}).shares(KEYBO_LSB.replace("x", "X"))
    assert shares["redirects_family_total"] == 0.0
