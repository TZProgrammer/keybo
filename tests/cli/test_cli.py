"""End-to-end CLI tests: each workflow is invocable via `python -m keybo <cmd>`.

These drive the thin CLI layer on small fixtures/models, verifying argument parsing and that
each subcommand runs the library end-to-end and produces its expected artifact/output.
"""

import numpy as np
import pytest

from keybo.cli.__main__ import main
from keybo.data.strokes import StrokeRow
from keybo.training.train import train_bigram_model


def _train_tiny_model(path):
    """Train and save a tiny bigram model for the score/optimize commands to load."""
    rng = np.random.default_rng(0)
    bigrams = ["th", "he", "an", "in", "er", "re", "on", "at", "en", "nd", "'a", "-e"]
    rows = []
    for i in range(120):
        bg = bigrams[i % len(bigrams)]
        samples = [(90, 100 + int(rng.integers(0, 40))), (85, 110 + int(rng.integers(0, 40)))]
        rows.append(StrokeRow(positions=((-1, 3), (1, 2)), ngram=bg, frequency=5, samples=samples))
    model = train_bigram_model(rows, target_wpm=90, n_estimators=10, max_depth=3)
    model.save(str(path))
    return str(path)


def test_no_command_prints_help_and_exits_nonzero(capsys):
    rc = main([])
    assert rc != 0


def test_unknown_command_errors():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_optimize_runs_and_is_reproducible(tmp_path, capsys):
    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t100\nhe\t90\nan\t80\n'a\t20\n-e\t10\n")

    rc1 = main(
        [
            "optimize",
            "--model",
            model_path,
            "--bigram-freqs",
            str(corpus),
            "--seed",
            "7",
            "--alpha",
            "0.9",
            "--max-outer",
            "20",
        ]
    )
    out1 = capsys.readouterr().out
    rc2 = main(
        [
            "optimize",
            "--model",
            model_path,
            "--bigram-freqs",
            str(corpus),
            "--seed",
            "7",
            "--alpha",
            "0.9",
            "--max-outer",
            "20",
        ]
    )
    out2 = capsys.readouterr().out

    assert rc1 == 0 and rc2 == 0
    # Same seed -> identical printed best layout (bug #11 reproducibility, end to end).
    assert out1 == out2


def test_score_prints_a_table_with_baseline(tmp_path, capsys):
    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t100\nhe\t90\nan\t80\n")

    rc = main(["score", "--model", model_path, "--bigram-freqs", str(corpus)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "qwerty" in out  # the baseline layout is listed
    assert "dvorak" in out


def test_process_data_writes_tsv(tmp_path):
    # Build a tiny dataset (participant typed "the").
    files = tmp_path / "files"
    files.mkdir()
    meta = files / "metadata_participants.txt"
    meta.write_text(
        "PARTICIPANT_ID\tFINGERS\tAVG_WPM_15\tKEYBOARD_TYPE\tLAYOUT\n111\t9-10\t90\tfull\tqwerty\n"
    )
    lines = ["PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tPRESS_TIME\tRELEASE_TIME\tLETTER"]
    t = 1000
    for ch in "the":
        lines.append(f"111\ts1\tthe\t{t}\t{t + 50}\t{ch}")
        t += 100
    (files / "111_keystrokes.txt").write_text("\n".join(lines) + "\n")

    out_tsv = tmp_path / "bistrokes.tsv"
    rc = main(
        [
            "process-data",
            "--files-dir",
            str(files),
            "--metadata",
            str(meta),
            "--ngram",
            "bigram",
            "--output",
            str(out_tsv),
        ]
    )
    assert rc == 0
    assert out_tsv.exists()
    assert "th" in out_tsv.read_text()


def test_train_writes_model_and_sidecar(tmp_path):
    # A tiny bistrokes tsv the train command will consume.
    tsv = tmp_path / "bistrokes.tsv"
    lines = []
    for bg in ["th", "he", "an", "in", "er", "re", "on", "at"]:
        lines.append(f"((-1, 3), (1, 2))\t{bg}\t5\t(90, 120)\t(85, 130)\t(92, 118)")
    tsv.write_text("\n".join(lines) + "\n")

    out_model = tmp_path / "bg.json"
    rc = main(
        [
            "train",
            "--strokes",
            str(tsv),
            "--ngram",
            "bigram",
            "--output",
            str(out_model),
            "--target-wpm",
            "90",
            "--n-estimators",
            "10",
            "--min-samples",
            "1",
        ]
    )
    assert rc == 0
    assert out_model.exists()
    assert out_model.with_suffix(".meta.json").exists()
