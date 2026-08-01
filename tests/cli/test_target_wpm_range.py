"""`--target-wpm` range validation (fable-audit finding 5; design-audit C.2).

A `--target-wpm` far outside the model's stamped range silently extrapolates (a measured ~23%
fitness shift) with no signal to the user, so `build_scorer` warns on stderr when the requested
WPM falls outside `model.metadata.wpm_range`. It only warns — power users may want extrapolation.

⚠ This docstring, `build_scorer`'s, and the warning text itself all used to say "the trees clamp
WPM at the boundary of the range they were trained on". That is FALSE, and the correction is what
`test_the_stamped_wpm_range_does_not_bound_the_trees` below pins BEHAVIOURALLY rather than as
prose. `wpm_range` is a cosmetic metadata literal: it reaches `ModelMetadata` and nothing else —
it never filters `X` or `y`, and `cli/train.py` exposes no `--wpm-range`. WPM is an ordinary
continuous feature, so the trees clamp at the largest split threshold their DATA produced, not at
the stamp. On the shipped k31 models that is 213 (50 distinct wpm split thresholds sit above the
stamped 120). The warning's ADVICE survives the correction with a different reason: out-of-range
predictions are unvalidated because training support THINS (~3.8-3.9% of raw k31 samples are
>=120 wpm), not because the output stops moving.
"""

import argparse

import numpy as np

from keybo.cli._scorer import build_scorer
from tests.cli.test_cli import _train_tiny_model

#: wpm values the fixture model's data spans. Deliberately reaches past the stamped 120 so the
#: stamp and the data's true upper bound are DIFFERENT numbers and the test can tell them apart.
DATA_WPMS = (60, 90, 120, 150, 180)


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


def test_the_stamped_wpm_range_does_not_bound_the_trees():
    """The corrected claim, pinned as BEHAVIOUR: crossing the stamp costs evidence, not motion.

    Prose is brittle to pin, so this asserts the fact the prose describes instead. A model is
    trained on data spanning 60..180 wpm but STAMPED (60, 120); if the stamp bounded the trees,
    predictions would be constant above 120. They are not — they keep changing to 150, and freeze
    only past the DATA's maximum, which is the real (and much higher) clamp.

    This is the load-bearing test for the docstring correction: were someone to make `wpm_range`
    actually filter training data, the claim "the trees clamp at the boundary" would become true
    and THIS test would fail, which is exactly the signal wanted.
    """
    from keybo.data.strokes import StrokeRow
    from keybo.features import bigram_features_from_positions
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.training.train import train_bigram_model

    # ms = 12000/w * (1 + w/200) makes the LOGRAT target rise monotonically with wpm, so there is
    # a real wpm signal for the trees to split on (a target that is flat in wpm would pass this
    # test vacuously). practice_term=False keeps the per-bigram backfit out of the comparison.
    rows = [
        StrokeRow(
            layout="qwerty",
            positions=((-1, 3), (1, 2)),
            ngram=bigram,
            frequency=5,
            samples=[(wpm, int(12000 / wpm * (1 + wpm / 200.0)), i, 50) for wpm in DATA_WPMS],
        )
        for i in range(160)
        for bigram in [("th", "he", "an", "in", "er", "re", "on", "at")[i % 8]]
    ]
    model = train_bigram_model(
        rows,
        target_wpm=90,
        wpm_range=(60, 120),  # deliberately NARROWER than the data
        n_estimators=25,
        max_depth=3,
        practice_term=False,
    )
    assert model.metadata.wpm_range == (60, 120), "the stamp under test"

    def predict(wpm: float) -> float:
        features = bigram_features_from_positions(
            ROW_STAGGERED_30, (ROW_STAGGERED_30.slots[0], ROW_STAGGERED_30.slots[3]), wpm=wpm
        )
        return float(model.predict(np.array([features]))[0])

    assert predict(150.0) != predict(120.0), (
        "predictions MOVED past the stamped upper bound -- the stamp does not clamp the trees"
    )
    assert predict(90.0) != predict(120.0), "and wpm is a live feature inside the range too"
    assert predict(max(DATA_WPMS)) == predict(10_000.0), (
        "the real clamp is the largest split threshold the DATA produced, not the stamp"
    )
