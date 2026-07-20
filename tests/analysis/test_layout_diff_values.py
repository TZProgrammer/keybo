from __future__ import annotations

import numpy as np
import pytest

import keybo.analysis.layout_diff as layout_diff
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout

QWERTY = "qwertyuiopasdfghjkl;zxcvbnm,./"


def test_layout_diff_pins_frequency_weighting_coverage_and_percentages(monkeypatch):
    table = np.fromfunction(lambda i, j: i * 100.0 + j, (31, 31))
    monkeypatch.setattr(layout_diff, "_bigram_table", lambda *_args, **_kwargs: table)
    before = Layout(QWERTY, ROW_STAGGERED_30)
    after = Layout(QWERTY, ROW_STAGGERED_30)
    after.swap("q", "w")

    diff = layout_diff.diff_layouts(
        before,
        after,
        [object()],
        {"qa": 3, "wa": 1, "q#": 6},
    )

    assert diff.total_a == pytest.approx(140.0)
    assert diff.total_b == pytest.approx(340.0)
    assert diff.total_delta == pytest.approx(200.0)
    assert diff.corpus_coverage == pytest.approx(0.4)
    assert [(item.ngram, item.impact, item.moved_chars) for item in diff.top()] == [
        ("qa", 300.0, "q"),
        ("wa", -100.0, "w"),
    ]
    report = diff.to_dict()
    assert report["total_delta_pct_of_a"] == pytest.approx(142.85714285714286)
    assert report["top_impacts"][0]["delta_pct"] == pytest.approx(1000.0)
    assert report["top_impacts"][0]["impact_pct_of_total_a"] == pytest.approx(214.28571428571428)
    assert report["top_impacts"][0]["share_of_total_delta_pct"] == pytest.approx(150.0)
