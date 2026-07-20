"""KAN-PRIME-1: exact decomposition + primed (strain-residual) gauges.

The primes drop (T) same-finger time-proxy terms — superseded by the measured
speed surfaces — and (S) the oxeylyzer-1 flow-preference trigram table, keeping
each tool's mechanical-strain terms at native weights. ``score()`` is the sum
of ``components()``, so the golden parity tests in test_kan1_parity.py gate the
split's exactness against the real binaries' frozen outputs; here we pin the
subset arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from keybo.analysis.community import APOS_DOF, _dof_arrays, community_suite

GOLD = json.loads((Path(__file__).parent / "golden_kan1.json").read_text())
LAYOUTS = sorted(GOLD["layouts"])


@pytest.fixture(scope="module")
def suite():
    return community_suite(";")


@pytest.mark.parametrize("name", LAYOUTS)
def test_oxey2_decomposition_and_prime(name, suite):
    _, _, o2 = suite
    lay = GOLD["layouts"][name]
    c = o2.components(lay)
    assert set(c) == {"wfd", "stretch"}
    assert o2.score(lay) == c["wfd"] + c["stretch"]
    assert o2.wfd(lay) == c["wfd"]
    assert o2.score_primed(lay) == c["stretch"] == o2.score(lay) - o2.wfd(lay)


@pytest.mark.parametrize("name", LAYOUTS)
def test_oxey1_decomposition_and_prime(name, suite):
    _, v1, _ = suite
    lay = GOLD["layouts"][name]
    c = v1.components(lay)
    assert v1.score(lay) == c["fspeed"] + c["stretch"] + c["pinky_ring"] + c["trigrams"]
    assert v1.score_primed(lay) == c["stretch"] + c["pinky_ring"]
    # +R sensitivity keeps exactly the redirect part of the trigram table
    assert (
        v1.score_primed(lay, keep_redirects=True) - v1.score_primed(lay)
        == c["trigrams_redirect_part"]
    )
    # redirect part is a genuine (negative-weight) subset of the trigram term
    assert c["trigrams_redirect_part"] <= 0


@pytest.mark.parametrize("name", LAYOUTS)
def test_genkey_decomposition_and_prime(name, suite):
    gk, _, _ = suite
    lay = GOLD["layouts"][name]
    c = gk.components(lay)
    assert gk.score(lay) == pytest.approx(
        gk.FSPEED_W * c["fspeed"] + gk.LSB_W * c["lsb_pct"] + gk.IDX_W * c["index_imbalance_pct"],
        abs=1e-12,
    )
    assert gk.score_primed(lay) == pytest.approx(
        gk.LSB_W * c["lsb_pct"] + gk.IDX_W * c["index_imbalance_pct"], abs=1e-12
    )


def test_prime_moves_with_strain_not_speed(suite):
    """Swapping two same-column keys (a pure fspeed/SFB change on genkey's
    column fingering) leaves genkey' unchanged while genkey moves."""
    gk, _, _ = suite
    base = "qwertyuiopasdfghjkl'zxcvbnm,.-"
    # swap q (r0,c0) and a (r1,c0): same column -> same finger, and column 0
    # enters neither the LSB pairs nor the index balance; the asymmetric row
    # move re-prices the {a,z} vs {q,z} same-finger distances (fspeed only)
    swapped = "awertyuiopqsdfghjkl'zxcvbnm,.-"
    assert gk.score(base) != gk.score(swapped)
    assert gk.score_primed(base) == pytest.approx(gk.score_primed(swapped), abs=1e-12)


def test_qwerty_components_and_primes_are_value_pinned(suite):
    gk, v1, o2 = suite
    qwerty = "qwertyuiopasdfghjkl'zxcvbnm,.-"

    assert gk.components(qwerty) == pytest.approx(
        {
            "fspeed": 34.86316873726387,
            "lsb_pct": 5.491317042653564,
            "index_imbalance_pct": 2.8717234107936287,
        },
        abs=1e-12,
    )
    assert gk.score_primed(qwerty) == pytest.approx(6.352834065891653, abs=1e-12)
    assert v1.components(qwerty) == {
        "fspeed": -32219249071,
        "stretch": -1452649710,
        "pinky_ring": -169609840,
        "trigrams": 12993325250,
        "trigrams_redirect_part": -13349962470,
    }
    assert v1.score_primed(qwerty) == -1622259550
    assert v1.score_primed(qwerty, keep_redirects=True) == -14972222020
    assert o2.components(qwerty) == {
        "wfd": -65746277057400,
        "stretch": -10185000883200,
    }
    assert o2.score_primed(qwerty) == -10185000883200


def test_classic_quote_slot_dof_mapping_and_scores_are_value_pinned():
    classic = "qwertyuiopasdfghjkl;zxcvbnm,./"
    gk, v1, o2 = community_suite("'")
    assert v1.chars == [*classic, "'"]
    assert o2.chars == [*classic, "'"]

    char_at_dof, dof_of_char = _dof_arrays(classic, v1.chars)
    idx = {char: i for i, char in enumerate(v1.chars)}
    assert dof_of_char[idx["q"]] == 0
    assert dof_of_char[idx["p"]] == 9
    assert dof_of_char[idx["a"]] == 10
    assert dof_of_char[idx[";"]] == 19
    assert dof_of_char[idx["'"]] == APOS_DOF == 20
    assert dof_of_char[idx["z"]] == 21
    assert dof_of_char[idx["/"]] == 30
    assert sorted(dof_of_char.tolist()) == list(range(31))
    for char_index, dof in enumerate(dof_of_char):
        assert char_at_dof[dof] == char_index

    assert gk.score(classic) == pytest.approx(110.80736026590948, abs=1e-12)
    assert v1.score(classic) == -20773973077
    assert o2.score(classic) == -75508310571300
    assert o2.wfd(classic) == -65299150616100


def test_dof_mapping_rejects_duplicate_or_mixed_layouts():
    _, v1, _ = community_suite("'")
    with pytest.raises(ValueError, match="permutation"):
        _dof_arrays("a" * 30, v1.chars)
    with pytest.raises(ValueError, match="permutation"):
        _dof_arrays("qwertyuiopasdfghjkl;zxcvbnm,.#", v1.chars)
