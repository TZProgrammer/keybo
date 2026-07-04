"""Output-path robustness: never discover a bad --output AFTER hours of work.

Real incident (laptop, 2026-07-04): `keybo train --output models/bigram.json` — the
README's own example — trained to completion, then died in XGBoost's C++ layer
(`LocalFileSystem::Open "models/bigram.json": No such file or directory`) because nothing
creates parent directories and `models/` is gitignored, so a fresh clone lacks it.

Two-layer fix under test here:
1. fail-fast: every CLI validates/creates its output path(s) BEFORE the expensive stage;
2. auto-mkdir: library-level save/write creates parent dirs (belt and suspenders).
"""

import numpy as np
import pytest

from keybo.cli.__main__ import main
from keybo.data.strokes import StrokeRow
from keybo.training.train import train_bigram_model


def _tiny_model(path):
    rng = np.random.default_rng(0)
    rows = [
        StrokeRow(
            positions=((-1, 3), (1, 2)),
            ngram=["th", "he", "an"][i % 3],
            frequency=5,
            samples=[(90, 100 + int(rng.integers(0, 40)))],
        )
        for i in range(30)
    ]
    model = train_bigram_model(rows, target_wpm=90, n_estimators=3, max_depth=2)
    model.save(str(path))
    return str(path)


def _tiny_tsv(tmp_path):
    tsv = tmp_path / "bi.tsv"
    lines = [f"((-1, 3), (1, 2))\t{bg}\t5\t(90, 120)\t(85, 130)" for bg in ["th", "he", "an"]]
    tsv.write_text("\n".join(lines) + "\n")
    return str(tsv)


# --- library level: saves create parent dirs -------------------------------------------


def test_model_save_creates_parent_dirs(tmp_path):
    from keybo.models.base import ModelMetadata
    from keybo.models.xgboost_model import XGBoostTypingModel

    rng = np.random.default_rng(0)
    X, y = rng.random((30, 5)), rng.random(30)
    meta = ModelMetadata(
        feature_version="v",
        feature_names=[f"f{i}" for i in range(5)],
        wpm_range=(60, 120),
        ngram="bigram",
    )
    m = XGBoostTypingModel(meta, n_estimators=3)
    m.fit(X, y)
    dest = tmp_path / "models" / "deep" / "bg.json"  # parents do not exist
    m.save(str(dest))
    assert dest.exists()
    assert dest.with_suffix(".meta.json").exists()


def test_write_ngram_tsv_creates_parent_dirs(tmp_path):
    from keybo.data.keystrokes import write_ngram_tsv

    dest = tmp_path / "out" / "nested" / "bi.tsv"
    aggregated = {(((-1, 3), (1, 2)), "th"): {"frequency": 1, "occurrences": [(90, 100)]}}
    write_ngram_tsv(aggregated, str(dest))
    assert dest.exists()


# --- CLI level: the README example shape works on a fresh clone -------------------------


def test_train_creates_missing_output_dir(tmp_path):
    tsv = _tiny_tsv(tmp_path)
    out = tmp_path / "models" / "bigram.json"  # models/ does not exist (the incident shape)
    rc = main(
        [
            "train",
            "--strokes",
            tsv,
            "--ngram",
            "bigram",
            "--output",
            str(out),
            "--min-samples",
            "1",
            "--n-estimators",
            "3",
            "--no-progress",
        ]
    )
    assert rc == 0
    assert out.exists()


def test_optimize_out_creates_missing_dir(tmp_path):
    model = _tiny_model(tmp_path / "bg.json")
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t10\nhe\t5\n")
    out = tmp_path / "runs" / "best.json"
    rc = main(
        [
            "optimize",
            "--model",
            model,
            "--bigram-freqs",
            str(corpus),
            "--alpha",
            "0.9",
            "--max-outer",
            "5",
            "--no-local-search",
            "--no-progress",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_process_data_creates_missing_output_dir(tmp_path):
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
    out = tmp_path / "tables" / "bi.tsv"
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
            str(out),
            "--no-progress",
        ]
    )
    assert rc == 0
    assert out.exists()


# --- CLI level: fail FAST on an unusable output, BEFORE the expensive stage -------------


def test_train_fails_fast_on_unusable_output_before_loading_strokes(tmp_path):
    # Parent "dir" is a regular FILE -> can never be created. The strokes path is ALSO
    # bogus: if validation correctly runs first, we get the output-path SystemExit, not
    # load_strokes' FileNotFoundError.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    bad_out = blocker / "m.json"
    with pytest.raises(SystemExit, match="output"):
        main(
            [
                "train",
                "--strokes",
                str(tmp_path / "missing.tsv"),
                "--ngram",
                "bigram",
                "--output",
                str(bad_out),
                "--no-progress",
            ]
        )


def test_optimize_fails_fast_on_unusable_out_before_search(tmp_path):
    model = _tiny_model(tmp_path / "bg.json")
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t10\n")
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    with pytest.raises(SystemExit, match="output"):
        main(
            [
                "optimize",
                "--model",
                model,
                "--bigram-freqs",
                str(corpus),
                "--out",
                str(blocker / "best.json"),
                "--no-progress",
            ]
        )


def test_process_data_fails_fast_on_unusable_output(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    with pytest.raises(SystemExit, match="output"):
        main(
            [
                "process-data",
                "--files-dir",
                str(tmp_path),
                "--metadata",
                str(tmp_path / "missing_meta.txt"),
                "--ngram",
                "bigram",
                "--output",
                str(blocker / "bi.tsv"),
                "--no-progress",
            ]
        )


def test_tune_fails_fast_on_unusable_output(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    with pytest.raises(SystemExit, match="output"):
        main(
            [
                "tune",
                "--strokes",
                str(tmp_path / "missing.tsv"),
                "--ngram",
                "bigram",
                "--output",
                str(blocker / "hp.json"),
            ]
        )


def test_train_fails_fast_on_missing_hyperparams_file(tmp_path):
    # --hyperparams pointing nowhere must fail before the (potentially long) strokes load:
    # the strokes path is also bogus, so the error must be about hyperparams.
    with pytest.raises(SystemExit, match="hyperparams"):
        main(
            [
                "train",
                "--strokes",
                str(tmp_path / "missing.tsv"),
                "--ngram",
                "bigram",
                "--output",
                str(tmp_path / "m.json"),
                "--hyperparams",
                str(tmp_path / "nope.json"),
                "--no-progress",
            ]
        )
