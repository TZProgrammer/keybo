"""CLI tests for `keybo optimize --attempts N --out <file>`.

Restores the legacy best-of-N search (the rewrite ran a single seed and presented one random
local minimum as "the answer") and adds a durable JSON artifact of the best result.
"""

import json

from keybo.cli.__main__ import main

# Reuse the tiny-model helper the existing CLI tests use (keeps runs fast + consistent).
from tests.cli.test_cli import _train_tiny_model


def _corpus(tmp_path):
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t100\nhe\t90\nan\t80\n'a\t20\n-e\t10\n")
    return str(corpus)


def _base_argv(model_path, corpus, *extra):
    return [
        "optimize",
        "--model",
        model_path,
        "--bigram-freqs",
        corpus,
        "--seed",
        "7",
        "--alpha",
        "0.9",
        "--max-outer",
        "15",
        "--no-local-search",
        *extra,
    ]


def _fitness_from_out(path):
    return json.loads(path.read_text())["fitness"]


def test_optimize_attempts_is_reproducible(tmp_path, capsys):
    """Same --seed + --attempts twice -> byte-identical stdout (determinism across attempts)."""
    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = _corpus(tmp_path)

    main(_base_argv(model_path, corpus, "--attempts", "3"))
    out1 = capsys.readouterr().out
    main(_base_argv(model_path, corpus, "--attempts", "3"))
    out2 = capsys.readouterr().out
    assert out1 == out2


def test_optimize_prints_per_attempt_progress(tmp_path, capsys):
    """Each attempt logs a one-liner and the final best is still printed."""
    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = _corpus(tmp_path)

    rc = main(_base_argv(model_path, corpus, "--attempts", "3"))
    out = capsys.readouterr().out
    assert rc == 0
    # Per-attempt progress lines, numbered "i/N".
    assert "attempt 1/3" in out
    assert "attempt 3/3" in out
    # The final best summary (existing output format) is retained.
    assert "Best fitness" in out


def test_optimize_best_of_n_no_worse_than_single(tmp_path, capsys):
    """best-of-3 fitness <= best-of-1 fitness for the same base seed (more tries can't hurt)."""
    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = _corpus(tmp_path)

    out_single = tmp_path / "single.json"
    main(_base_argv(model_path, corpus, "--attempts", "1", "--out", str(out_single)))
    capsys.readouterr()
    out_many = tmp_path / "many.json"
    main(_base_argv(model_path, corpus, "--attempts", "3", "--out", str(out_many)))
    capsys.readouterr()

    assert _fitness_from_out(out_many) <= _fitness_from_out(out_single)


def test_optimize_out_writes_expected_json(tmp_path, capsys):
    """--out writes exactly the agreed keys; layout is a 30-char permutation of the start."""
    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = _corpus(tmp_path)
    out_file = tmp_path / "result.json"

    rc = main(_base_argv(model_path, corpus, "--attempts", "2", "--out", str(out_file)))
    capsys.readouterr()
    assert rc == 0
    assert out_file.exists()

    result = json.loads(out_file.read_text())
    assert set(result) == {
        "layout",
        "fitness",
        "ngram",
        "target_wpm",
        "seed",
        "attempts",
        "model",
    }
    from keybo.layouts import NAMED_LAYOUTS

    start = NAMED_LAYOUTS["qwerty"]
    assert len(result["layout"]) == 30
    # A permutation of the starting layout's characters (optimize only rearranges keys).
    assert sorted(result["layout"]) == sorted(start)
    assert result["ngram"] == "bigram"
    assert result["seed"] == 7
    assert result["attempts"] == 2
    assert result["model"] == model_path
    assert isinstance(result["fitness"], float)
