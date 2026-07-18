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

from keybo.analysis.community import community_suite

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
