"""`keybo inspect` — the structural diagnostics table (gaps-audit Phase A).

The tool that makes objective blind spots VISIBLE: per-finger corpus load, per-finger
same-finger-bigram share, row/hand/motion-class shares — for any layout string, side by
side with the named layouts. Born from the finger-utilization gap: the objective prices
finger reuse only at lag 1, so utilization skew must at least be measurable at a glance.
"""

import json

import pytest

from keybo.cli.__main__ import main
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.inspect import layout_diagnostics


@pytest.fixture
def freqs():
    # A tiny corpus with known structure: 'th' same-hand-ish, 'e ' space bigram, etc.
    return {"th": 100, "he": 90, "e ": 80, "an": 60, "ju": 10}


def test_diagnostics_shares_sum_to_one(freqs):
    d = layout_diagnostics(Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30), freqs)
    assert sum(d["finger_load"].values()) == pytest.approx(1.0)
    assert sum(d["row_share"].values()) == pytest.approx(1.0)
    assert sum(d["motion_share"].values()) == pytest.approx(1.0)


def test_finger_load_attributes_to_the_right_finger(freqs):
    # On qwerty, 'j' and 'u' are right-index keys; a corpus of only 'ju' loads right index.
    d = layout_diagnostics(Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30), {"ju": 10})
    assert d["finger_load"]["R-index"] == pytest.approx(1.0)


def test_sfb_share_detects_same_finger_bigrams():
    # 'ju' is a same-finger bigram on qwerty (both right index); 'th' is not.
    d = layout_diagnostics(Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30), {"ju": 10, "th": 10})
    assert d["sfb_share"] == pytest.approx(0.5)
    assert d["sfb_by_finger"]["R-index"] == pytest.approx(1.0)  # all SFBs on right index


def test_space_and_offboard_bigrams_handled(freqs):
    # 'e ' involves the thumb/space: counted in loads (thumb), never as SFB; a bigram
    # with a char not on the board is excluded entirely.
    d = layout_diagnostics(Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30), {"e ": 80, "é!": 5})
    assert d["finger_load"]["thumb"] > 0
    assert d["sfb_share"] == 0.0
    assert d["excluded_weight_share"] == pytest.approx(5 / 85)


def test_cli_prints_table_and_json(tmp_path, capsys):
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t100\nhe\t90\nan\t60\n")
    out = tmp_path / "diag.json"
    rc = main(
        [
            "inspect",
            "--layout",
            NAMED_LAYOUTS["qwerty"],
            "--bigram-freqs",
            str(corpus),
            "--out",
            str(out),
        ]
    )
    printed = capsys.readouterr().out
    assert rc == 0
    assert "finger" in printed.lower()
    assert "qwerty" in printed  # named layouts printed for comparison
    report = json.loads(out.read_text())
    assert "layout" in report and "named" in report
    assert "finger_load" in report["layout"]


def test_cli_accepts_named_layout(tmp_path, capsys):
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t100\n")
    rc = main(["inspect", "--layout", "dvorak", "--bigram-freqs", str(corpus)])
    assert rc == 0
    assert "dvorak" in capsys.readouterr().out
