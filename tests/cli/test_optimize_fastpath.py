"""optimize CLI: TableBigramScorer fast path (6.2) + auto-E5 structural postflight (4.5).

The fast path must be an internal speedup, not a semantic change: same seed => same
layout as the slow model-scorer path (the table is exact-parity-tested elsewhere; here we
assert the CLI wiring preserves determinism and equivalence). The postflight prints the
structural profile (home-row share, SFB share, max finger load) so every search ends with
the Goodhart gate's numbers in view.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from keybo.cli.__main__ import main
from keybo.data.strokes import StrokeRow
from keybo.training.train import train_bigram_model


@pytest.fixture
def model_path(tmp_path):
    rng = np.random.default_rng(0)
    bigrams = ["th", "he", "an", "in", "er", "re", "on", "at", "e ", " t"]
    rows = []
    for i in range(100):
        bg = bigrams[i % len(bigrams)]
        samples = [(90, 100 + int(rng.integers(0, 60)), i % 9 + 1, 50)]
        rows.append(
            StrokeRow(
                layout="qwerty",
                positions=((-1, 3), (1, 2)),
                ngram=bg,
                frequency=5,
                samples=samples,
            )
        )
    m = train_bigram_model(rows, target_wpm=90, n_estimators=8, max_depth=2)
    p = tmp_path / "bg.json"
    m.save(str(p))
    return str(p)


@pytest.fixture
def corpus_path(tmp_path):
    p = tmp_path / "bg.txt"
    p.write_text("th\t100\nhe\t90\nan\t80\ne \t70\n t\t60\n")
    return str(p)


def _run(args_list, capsys):
    rc = main(args_list)
    out = capsys.readouterr().out
    assert rc == 0
    layout_line = [ln for ln in out.splitlines() if ln.startswith("Best fitness")][0]
    return out, layout_line


def test_fast_path_matches_slow_path_layout(model_path, corpus_path, capsys):
    common = [
        "optimize",
        "--model",
        model_path,
        "--bigram-freqs",
        corpus_path,
        "--seed",
        "3",
        "--alpha",
        "0.9",
        "--max-outer",
        "25",
        "--no-progress",
    ]
    out_fast, fit_fast = _run(common, capsys)  # fast path is the default
    out_slow, fit_slow = _run([*common, "--no-table"], capsys)
    # Same seed, same objective => identical best fitness line and rendered layout.
    assert fit_fast == fit_slow
    assert out_fast.splitlines()[-4:] == out_slow.splitlines()[-4:]


def test_postflight_structural_report_printed(model_path, corpus_path, capsys):
    out, _ = _run(
        [
            "optimize",
            "--model",
            model_path,
            "--bigram-freqs",
            corpus_path,
            "--seed",
            "1",
            "--alpha",
            "0.9",
            "--max-outer",
            "10",
            "--no-progress",
        ],
        capsys,
    )
    # The E5 postflight numbers every run must end with:
    assert "home-row share" in out
    assert "sfb share" in out
    assert "max finger load" in out


def test_comfort_weight_changes_the_objective(model_path, corpus_path, capsys):
    common = [
        "optimize",
        "--model",
        model_path,
        "--bigram-freqs",
        corpus_path,
        "--seed",
        "2",
        "--alpha",
        "0.9",
        "--max-outer",
        "10",
        "--no-local-search",
        "--no-progress",
    ]
    _, fit_speed = _run(common, capsys)
    _, fit_comfort = _run([*common, "--comfort-weight", "5.0"], capsys)
    # A nonzero comfort weight adds penalty mass -> reported fitness must differ.
    assert fit_speed != fit_comfort


def test_comfort_weight_loads_the_adjacent_skipgram_corpus(
    tmp_path,
    monkeypatch,
):
    from keybo.cli import optimize
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.layout import Layout
    from keybo.layouts import NAMED_LAYOUTS
    from keybo.scoring import comfort as comfort_module

    corpus_path = tmp_path / "bigrams.txt"
    corpus_path.write_text("")
    Path(corpus_path).with_name("1-skip.txt").write_text("de\t7\n")
    captured = {}
    real_scorer = comfort_module.ComfortBigramScorer

    class CapturingComfortScorer(real_scorer):
        def __init__(self, *args, skipgram_freqs=None, **kwargs):
            captured.update(skipgram_freqs or {})
            super().__init__(*args, skipgram_freqs=skipgram_freqs, **kwargs)

    class ZeroScorer:
        @staticmethod
        def fitness(_layout):
            return 0.0

    monkeypatch.setattr(comfort_module, "ComfortBigramScorer", CapturingComfortScorer)
    monkeypatch.setattr(optimize, "build_scorer", lambda _args: ZeroScorer())
    monkeypatch.setattr(optimize, "load_freqs", lambda _args: {})
    monkeypatch.setattr(
        optimize,
        "_one_attempt",
        lambda _args, _scorer, seed: Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30),
    )
    monkeypatch.setattr(
        optimize,
        "layout_diagnostics",
        lambda _layout, _freqs: {
            "row_share": {"home": 0.0},
            "sfb_share": 0.0,
            "finger_load": {},
        },
    )
    args = SimpleNamespace(
        attempts=1,
        out=None,
        comfort_weight=1.0,
        finger_load_weight=0.0,
        oxey_weight=0.0,
        ngram="bigram",
        comfort_config=None,
        bigram_freqs=str(corpus_path),
        no_table=False,
        seed=0,
        target_wpm=90.0,
        model="unused",
    )

    assert optimize.run(args) == 0

    assert captured == {"de": 7}


def test_finger_load_weight_changes_the_objective(model_path, corpus_path, capsys):
    common = [
        "optimize",
        "--model",
        model_path,
        "--bigram-freqs",
        corpus_path,
        "--seed",
        "4",
        "--alpha",
        "0.9",
        "--max-outer",
        "10",
        "--no-local-search",
        "--no-progress",
    ]
    _, fit_speed = _run(common, capsys)
    _, fit_load = _run([*common, "--finger-load-weight", "50.0"], capsys)
    assert fit_speed != fit_load
