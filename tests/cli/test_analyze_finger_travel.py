"""`keybo analyze`'s finger-travel and off-home blocks — the partition, and the CAVEATS.

The module-level tests in ``tests/analysis/test_finger_travel.py`` prove the metrics; these
prove the **CLI cannot emit them without the qualifiers that make them readable correctly.**
That is not decoration. Three of this round's findings are ones a consumer would get backwards
from the bare number:

* ``travel_total`` is |r| = 0.970 with ``sfb-dist`` — a near-restatement, not a 16th gauge;
* ~88–97% of travel is a MODELLED from-home term, not observed motion;
* **``off_home``'s sign is not "lower is better"** — in the 18-layout field more off-home pinky
  mass goes with FASTER predicted typing, so a naive minimizer would recommend colemak over
  keybo-lsb, a 3.6 ms/char regression.

A number whose correct reading is the opposite of its intuitive one is exactly the kind that gets
quoted bare, so the text and JSON paths are both pinned. This is the same stance
``test_analyze_allgauge.py`` takes on the ``bad_scissor`` support-boundary artifact.

Run on the **production default corpus** deliberately (unlike the frozen-board controls, which
inject ``--corpus iweb``): these assert structural properties — a partition summing to its own
total, a caveat being present — that must hold on whatever corpus a user actually runs.
"""

from __future__ import annotations

import json

import pytest

from keybo.analysis.finger_travel import FINGER_ORDER
from keybo.cli.__main__ import main


def _run(capsys, argv: list[str]) -> dict:
    rc = main(argv)
    assert rc == 0, f"analyze {argv} returned {rc}"
    return json.loads(capsys.readouterr().out)


def _text(capsys, argv: list[str]) -> str:
    rc = main(argv)
    assert rc == 0, f"analyze {argv} returned {rc}"
    return capsys.readouterr().out


@pytest.mark.slow
def test_travel_shares_are_an_exact_partition_of_100_in_the_cli(capsys):
    """The parts sum to the whole — the single best check the attribution rule survived wiring."""
    out = _run(
        capsys, ["analyze", "keybo-lsb", "graphite", "--no-time", "--no-model-scores", "--json"]
    )
    for name, row in out["rows"].items():
        shares = row["finger_travel"]["shares"]
        assert set(shares) == set(FINGER_ORDER), f"{name}: finger set {sorted(shares)}"
        assert sum(shares.values()) == pytest.approx(100.0, abs=1e-9), name


@pytest.mark.slow
def test_the_cli_emits_the_absolute_total_and_the_modelled_fraction_with_the_shares(capsys):
    """Shares without the LEVEL are the ``saved_vs_ref_pct`` artifact; both must travel together."""
    out = _run(capsys, ["analyze", "keybo-lsb", "--no-time", "--no-model-scores", "--json"])
    block = out["rows"]["keybo-lsb"]["finger_travel"]
    assert block["total"] > 0.0
    assert 0.0 < block["observed_fraction_pct"] < 100.0
    assert "sfb-dist" in block["use"], "the near-restatement must be disclosed in the JSON"
    assert "Do NOT optimize" in block["use"]


@pytest.mark.slow
def test_off_home_columns_are_an_exact_partition_and_carry_their_convention(capsys):
    """``off+on == usage`` per finger, and the denominator convention is NAMED in the output.

    The convention matters: the two shipped conventions differ by up to ~0.9 pp on a real board,
    so a consumer that cannot tell which one produced a number cannot reconcile it against
    ``DislocationScorer`` or against the other gauges.
    """
    out = _run(capsys, ["analyze", "keybo-lsb", "--no-time", "--no-model-scores", "--json"])
    row = out["rows"]["keybo-lsb"]
    for key in ("off_home", "off_home_restricted_convention"):
        block = row[key]
        assert block["convention"] in ("letter-freqs", "restricted")
        assert block["denominator"]
        for label in FINGER_ORDER:
            assert block["off_home"][label] + block["on_home"][label] == pytest.approx(
                block["usage"][label], abs=1e-9
            ), f"{key}/{label}"
        assert sum(block["usage"].values()) == pytest.approx(block["coverage_pct"], abs=1e-9)
    # and the two conventions must actually DIFFER, or shipping both is pointless ceremony
    assert row["off_home"]["pinky"]["off_home"] != pytest.approx(
        row["off_home_restricted_convention"]["pinky"]["off_home"], abs=1e-6
    )


@pytest.mark.slow
def test_the_cli_discloses_that_off_home_is_NOT_lower_is_better(capsys):
    """The caveat that stops a 3.6 ms/char regression being recommended by a naive minimizer."""
    out = _run(capsys, ["analyze", "keybo-lsb", "--no-time", "--no-model-scores", "--json"])
    use = out["rows"]["keybo-lsb"]["off_home"]["use"]
    assert "NOT 'lower is better'" in use, "the inverted sign MUST be stated where the number is"
    assert "UNSUPPORTED" in use, "the user's cost claim's status must travel with the metric"
    assert "pinky is NOT special" in use


@pytest.mark.slow
def test_the_text_report_prints_both_blocks_with_their_warnings(capsys):
    """A reader of the text table gets the same caveats as a reader of the JSON."""
    text = _text(capsys, ["analyze", "keybo-lsb", "colemak", "--no-time", "--no-model-scores"])
    assert "== finger travel" in text
    assert "== off-home usage" in text
    # the level is printed beside the shares
    assert "TOTAL" in text and "obs%" in text
    # and every claim a bare number would invert
    assert "NOT 'LOWER IS BETTER'" in text
    assert "ASSUMPTION" in text
    assert "near-restatement" in text
    assert "off%own is a per-finger RATIO — never sum it." in text


@pytest.mark.slow
def test_off_fraction_is_never_presented_as_something_summable(capsys):
    """It has a per-finger denominator; summing it yields a plausible-looking wrong percentage."""
    out = _run(capsys, ["analyze", "keybo-lsb", "--no-time", "--no-model-scores", "--json"])
    block = out["rows"]["keybo-lsb"]["off_home"]
    assert "NOT a partition" in block["off_fraction_note"]
    assert sum(block["off_fraction"].values()) > 150.0, (
        "if this ever sums near 100 the metric changed meaning and the note is now wrong"
    )
