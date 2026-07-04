"""End-to-end CLI tests: each workflow is invocable via `python -m keybo <cmd>`.

These drive the thin CLI layer on small fixtures/models, verifying argument parsing and that
each subcommand runs the library end-to-end and produces its expected artifact/output.
"""

import numpy as np
import pytest

from keybo.cli.__main__ import main
from keybo.data.strokes import StrokeRow
from keybo.training.train import train_bigram_model, train_trigram_model


def _train_tiny_model(path):
    """Train and save a tiny bigram model for the score/optimize commands to load."""
    rng = np.random.default_rng(0)
    bigrams = ["th", "he", "an", "in", "er", "re", "on", "at", "en", "nd", "'a", "-e"]
    rows = []
    for i in range(120):
        bg = bigrams[i % len(bigrams)]
        samples = [
            (90, 100 + int(rng.integers(0, 40)), i, 50),
            (85, 110 + int(rng.integers(0, 40)), i, 55),
        ]
        rows.append(
            StrokeRow(
                layout="qwerty",
                positions=((-1, 3), (1, 2)),
                ngram=bg,
                frequency=5,
                samples=samples,
            )
        )
    model = train_bigram_model(rows, target_wpm=90, n_estimators=10, max_depth=3)
    model.save(str(path))
    return str(path)


def _train_tiny_trigram_model(path):
    """Train and save a tiny trigram model for the score/optimize commands to load."""
    rng = np.random.default_rng(0)
    trigrams = ["the", "and", "ing", "her", "ere", "ent", "tha", "nth", "was", "eth"]
    rows = []
    for i in range(120):
        tg = trigrams[i % len(trigrams)]
        samples = [
            (90, 200 + int(rng.integers(0, 60)), i, 50),
            (85, 210 + int(rng.integers(0, 60)), i, 55),
        ]
        rows.append(
            StrokeRow(
                layout="qwerty",
                positions=((-1, 3), (1, 2), (-3, 3)),
                ngram=tg,
                frequency=5,
                samples=samples,
            )
        )
    model = train_trigram_model(rows, target_wpm=90, n_estimators=10, max_depth=3)
    model.save(str(path))
    return str(path)


def test_no_command_prints_help_and_exits_nonzero(capsys):
    rc = main([])
    assert rc != 0


def test_fetch_data_downloads_and_extracts(tmp_path, capsys):
    # Reuse the download test's local fixture server so this never hits the network.
    import socketserver
    import threading

    from tests.data.test_download import _make_keystrokes_zip, _RangeHandler

    handler = type("H", (_RangeHandler,), {"payload": _make_keystrokes_zip()})
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    try:
        rc = main(
            [
                "fetch-data",
                "--out-dir",
                str(tmp_path / "dataset"),
                "--url",
                f"http://{host}:{port}/Keystrokes.zip",
                "--no-progress",
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
    assert rc == 0
    out = capsys.readouterr().out
    assert "dataset ready" in out
    assert (tmp_path / "dataset").exists()


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


def test_score_with_trigram_model(tmp_path, capsys):
    model_path = _train_tiny_trigram_model(tmp_path / "tg.json")
    tg = tmp_path / "tg.txt"
    tg.write_text("the\t100\nand\t90\ning\t80\n")

    rc = main(["score", "--ngram", "trigram", "--model", model_path, "--trigram-freqs", str(tg)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "qwerty" in out
    assert "dvorak" in out


def test_optimize_with_trigram_model_runs(tmp_path, capsys):
    model_path = _train_tiny_trigram_model(tmp_path / "tg.json")
    tg = tmp_path / "tg.txt"
    tg.write_text("the\t100\nand\t90\ning\t80\nher\t70\n")

    rc = main(
        [
            "optimize",
            "--ngram",
            "trigram",
            "--model",
            model_path,
            "--trigram-freqs",
            str(tg),
            "--seed",
            "3",
            "--alpha",
            "0.9",
            "--max-outer",
            "15",
            "--no-local-search",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Best fitness" in out


def test_regression_trigram_cli_works_from_any_cwd(tmp_path, monkeypatch, capsys):
    """Delta-audit finding A: the trigram objective must not require (or even touch) the
    bigram/skipgram frequency files -- TrigramModelScorer discards them for train/serve
    parity. The old build_scorer eagerly loaded their repo-relative defaults, so any trigram
    run from a cwd without data/corpus/ crashed on a file it was about to throw away."""
    model_path = _train_tiny_trigram_model(tmp_path / "tg.json")
    tg = tmp_path / "tg.txt"
    tg.write_text("the\t100\nand\t90\n")
    monkeypatch.chdir(tmp_path)  # a cwd with no data/corpus/
    rc = main(
        ["score", "--ngram", "trigram", "--model", str(model_path), "--trigram-freqs", str(tg)]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "qwerty" in out


def test_ngram_mismatch_is_rejected(tmp_path):
    # A bigram model requested as a trigram objective must fail loudly, not silently mis-score.
    bg_model = _train_tiny_model(tmp_path / "bg.json")
    tg = tmp_path / "tg.txt"
    tg.write_text("the\t100\n")
    with pytest.raises(SystemExit):
        main(["score", "--ngram", "trigram", "--model", bg_model, "--trigram-freqs", str(tg)])


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
        lines.append(
            f"qwerty\t((-1, 3), (1, 2))\t{bg}\t5"
            "\t(90, 120, 1, 50)\t(85, 130, 2, 55)\t(92, 118, 3, 52)"
        )
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
