"""`--target-wpm` range validation (fable-audit finding 5; design-audit C.2).

The model's trees clamp WPM at the boundary of the range they were trained on, so a
`--target-wpm` far outside that range silently extrapolates (a measured ~23% fitness shift)
with no signal to the user. `build_scorer` now warns on stderr when the requested WPM falls
outside `model.metadata.wpm_range`. It only warns — power users may want extrapolation.
"""

import argparse

from keybo.cli._scorer import build_scorer
from tests.cli.test_cli import _train_tiny_model


def _scorer_args(model_path, target_wpm):
    """The minimal build_scorer args a bigram score/optimize run supplies."""
    return argparse.Namespace(
        model=model_path,
        ngram="bigram",
        bigram_freqs="data/corpus/bigrams.txt",
        trigram_freqs="data/corpus/trigrams.txt",
        target_wpm=target_wpm,
    )


def test_out_of_range_target_wpm_warns_on_stderr(tmp_path, capsys):
    """A tiny model is trained for wpm_range (60, 120); 200 is well outside it."""
    model_path = _train_tiny_model(tmp_path / "bg.json")

    build_scorer(_scorer_args(model_path, target_wpm=200.0))

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "200" in err
    assert "(60, 120)" in err  # the trained range is named
    assert "extrapolation" in err


def test_in_range_target_wpm_does_not_warn(tmp_path, capsys):
    """90 WPM is inside (60, 120) -> no warning."""
    model_path = _train_tiny_model(tmp_path / "bg.json")

    build_scorer(_scorer_args(model_path, target_wpm=90.0))

    err = capsys.readouterr().err
    assert "WARNING" not in err


def test_range_boundaries_are_inclusive(tmp_path, capsys):
    """The exact endpoints (60 and 120) are in-range and must not warn."""
    model_path = _train_tiny_model(tmp_path / "bg.json")

    build_scorer(_scorer_args(model_path, target_wpm=60.0))
    build_scorer(_scorer_args(model_path, target_wpm=120.0))

    assert "WARNING" not in capsys.readouterr().err
