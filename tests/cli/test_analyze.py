"""`keybo analyze` CLI: resolution, report shape, JSON mode (KAN-1)."""

from __future__ import annotations

import json

import pytest

from keybo.cli.__main__ import main


@pytest.mark.slow
def test_analyze_json_end_to_end(capsys):
    rc = main(["analyze", "keybo-c30m", "semimak", "--ref", "qwerty30m", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    rows = out["rows"]
    assert set(rows) == {"qwerty30m", "keybo-c30m", "semimak"}
    # the reference row saves 0% by construction; others report a number
    assert rows["qwerty30m"]["time"]["saved_vs_ref_pct"] == pytest.approx(0.0)
    assert rows["keybo-c30m"]["time"]["saved_vs_ref_pct"] > 3.0
    # community + kmstats present on every row
    for r in rows.values():
        assert {"genkey", "oxeylyzer1", "oxeylyzer2", "wfd"} <= set(r["community"])
        assert "sfb" in r["kmstats"] and "redir" in r["kmstats"]


def test_analyze_rejects_unknown_name():
    with pytest.raises(SystemExit, match="unknown layout"):
        main(["analyze", "not-a-layout"])


def test_analyze_accepts_raw_string_length_check():
    with pytest.raises(SystemExit, match="unknown layout"):
        main(["analyze", "abcdef"])  # neither a name nor 30 chars


@pytest.mark.slow
def test_kmstats_key_is_the_11_keymeow_stats_and_agrees_with_the_gauge_frame(capsys):
    """`row["kmstats"]` is the historical key and must keep working (ALLGAUGE-1).

    ALLGAUGE-1 added `row["gauges"]` — the campaign's 15-gauge frame, which is these 11 plus
    scissor/imbalance/oxey-style/comfort. Both are emitted because they name two different
    things: `kmstats` is a specific external convention (the keymeow metric set), `gauges` is
    the campaign frame. They must agree exactly on the shared 11.
    """
    from keybo.analysis.kmstats import STAT_NAMES

    rc = main(["analyze", "keybo-lsb", "--no-time", "--no-model-scores", "--json"])
    assert rc == 0
    row = json.loads(capsys.readouterr().out)["rows"]["keybo-lsb"]
    assert set(row["kmstats"]) == set(STAT_NAMES)
    for name in STAT_NAMES:
        assert row["kmstats"][name] == row["gauges"][name], name
    # the four gauges ALLGAUGE-1 added are in `gauges` only
    assert {"scissor", "imbalance", "oxey-style", "comfort"} <= set(row["gauges"])
    assert not {"scissor", "imbalance", "oxey-style", "comfort"} & set(row["kmstats"])
