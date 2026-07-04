"""CLI-level tests for `keybo validate` (the OQ-5 harness front-end)."""

import json

import numpy as np
import pytest

from keybo.cli.__main__ import main

# Two tiny "layouts" with the same ngrams at different distances; durations follow
# distance so there is real (if noisy) transfer for the harness to exercise.
_LAYOUT_POSITIONS = {
    "qwerty": {"ab": ((-1, 2), (1, 2)), "cd": ((-3, 2), (3, 2)), "ef": ((-2, 2), (4, 2))},
    "dvorak": {"ab": ((-3, 2), (3, 2)), "cd": ((-2, 2), (4, 2)), "ef": ((-1, 2), (1, 2))},
}


def _write_strokes(path):
    rng = np.random.default_rng(0)
    lines = []
    for layout, ngrams in _LAYOUT_POSITIONS.items():
        for ngram, ((x1, y1), (x2, y2)) in ngrams.items():
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            samples = []
            for pid in range(1, 9):
                for _ in range(4):
                    wpm = int(rng.integers(60, 100))
                    dur = int(70 + 22 * dist + rng.normal(0, 5))
                    samples.append(f"({wpm}, {dur}, {pid}, 50)")
            pos = f"(({x1}, {y1}), ({x2}, {y2}))"
            lines.append(f"{layout}\t{pos}\t{ngram}\t50\t" + "\t".join(samples))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def strokes_tsv(tmp_path):
    return _write_strokes(tmp_path / "bi.tsv")


def test_validate_prints_matrix_and_writes_report(tmp_path, strokes_tsv, capsys):
    out = tmp_path / "report.json"
    rc = main(
        [
            "validate",
            "--strokes",
            strokes_tsv,
            "--seeds",
            "0",
            "--min-cell-samples",
            "3",
            "--bucket-width",
            "40",
            "--n-boot",
            "10",
            "--out",
            str(out),
            "--no-progress",
        ]
    )
    printed = capsys.readouterr().out
    assert rc == 0
    assert "qwerty" in printed and "dvorak" in printed
    assert "ceiling" in printed
    assert "pooled held-out layout ranking tau" in printed
    report = json.loads(out.read_text())
    assert set(report["folds"]) == {"qwerty", "dvorak"}
    assert report["config"]["seeds"] == [0]
    for fold in report["folds"].values():
        assert len(fold["seeds"]) == 1
        assert "rho" in fold["seeds"][0]


def test_validate_single_holdout_runs_one_fold(strokes_tsv, capsys):
    rc = main(
        [
            "validate",
            "--strokes",
            strokes_tsv,
            "--holdout",
            "dvorak",
            "--seeds",
            "0",
            "--min-cell-samples",
            "3",
            "--bucket-width",
            "40",
            "--n-boot",
            "5",
            "--no-progress",
        ]
    )
    printed = capsys.readouterr().out
    assert rc == 0
    # One fold: dvorak appears as a fold row; qwerty must not.
    fold_lines = [ln for ln in printed.splitlines() if ln.startswith(("qwerty", "dvorak"))]
    assert len(fold_lines) == 1 and fold_lines[0].startswith("dvorak")


def test_validate_old_format_file_rejected(tmp_path):
    p = tmp_path / "old.tsv"
    p.write_text("((-1, 2), (1, 2))\tab\t5\t(90, 120)\n")
    with pytest.raises(ValueError, match="process-data"):
        main(["validate", "--strokes", str(p), "--seeds", "0", "--no-progress"])


def test_validate_unknown_holdout_fails_loudly(strokes_tsv):
    with pytest.raises(ValueError, match="no rows"):
        main(
            [
                "validate",
                "--strokes",
                strokes_tsv,
                "--holdout",
                "colemak",
                "--seeds",
                "0",
                "--min-cell-samples",
                "3",
                "--no-progress",
            ]
        )
