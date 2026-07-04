"""Progress visibility: every long-running workflow shows progress by default and accepts
--no-progress. Bars write to stderr (tqdm), so parseable stdout is unaffected — the
determinism tests elsewhere compare stdout and must keep passing.

Library-level rule: progress is opt-in (default False) so library callers stay clean; the
CLIs default it ON and expose --no-progress (mirroring fetch-data).
"""

import numpy as np

from keybo.cli.__main__ import main
from keybo.data.strokes import StrokeRow
from keybo.training.train import train_bigram_model


def _tiny_model(path):
    rng = np.random.default_rng(0)
    bigrams = ["th", "he", "an", "in", "er", "re"]
    rows = [
        StrokeRow(
            layout="qwerty",
            positions=((-1, 3), (1, 2)),
            ngram=bigrams[i % len(bigrams)],
            frequency=5,
            samples=[(90, 100 + int(rng.integers(0, 40)), 1, 50)],
        )
        for i in range(60)
    ]
    model = train_bigram_model(rows, target_wpm=90, n_estimators=5, max_depth=2)
    model.save(str(path))
    return str(path)


def _tiny_dataset(tmp_path):
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
    return files, meta


def test_process_data_accepts_no_progress(tmp_path):
    files, meta = _tiny_dataset(tmp_path)
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
            str(tmp_path / "bi.tsv"),
            "--no-progress",
        ]
    )
    assert rc == 0


def test_optimize_accepts_no_progress(tmp_path):
    model = _tiny_model(tmp_path / "bg.json")
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t10\nhe\t5\n")
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
        ]
    )
    assert rc == 0


def test_train_accepts_no_progress(tmp_path):
    tsv = tmp_path / "bi.tsv"
    lines = [
        f"qwerty\t((-1, 3), (1, 2))\t{bg}\t5\t(90, 120, 1, 50)\t(85, 130, 2, 55)"
        for bg in ["th", "he", "an"]
    ]
    tsv.write_text("\n".join(lines) + "\n")
    rc = main(
        [
            "train",
            "--strokes",
            str(tsv),
            "--ngram",
            "bigram",
            "--output",
            str(tmp_path / "m.json"),
            "--min-samples",
            "1",
            "--n-estimators",
            "3",
            "--no-progress",
        ]
    )
    assert rc == 0


# --- library-level: progress=True must not change results ------------------------------


def test_process_dataset_progress_matches_no_progress(tmp_path):
    from keybo.data.keystrokes import process_dataset

    files, meta = _tiny_dataset(tmp_path)
    quiet = process_dataset(str(files), str(meta), ngram="bigram", progress=False)
    loud = process_dataset(str(files), str(meta), ngram="bigram", progress=True)
    assert quiet == loud


def test_sa_progress_matches_no_progress():
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.layout import Layout
    from keybo.optimize.annealing import SimulatedAnnealing
    from tests.optimize.conftest import CharPlacementScorer

    chars = "qwertyuiopasdfghjkl'zxcvbnm,.-"
    scorer = CharPlacementScorer()
    a = SimulatedAnnealing(seed=5, alpha=0.9, progress=False).optimize(
        Layout(chars, ROW_STAGGERED_30), scorer
    )
    b = SimulatedAnnealing(seed=5, alpha=0.9, progress=True).optimize(
        Layout(chars, ROW_STAGGERED_30), scorer
    )
    assert "".join(a.chars) == "".join(b.chars)


def test_extract_progress_smoke(tmp_path):
    from keybo.data.download import extract_keystrokes
    from tests.data.test_download import _make_keystrokes_zip

    zip_path = tmp_path / "Keystrokes.zip"
    zip_path.write_bytes(_make_keystrokes_zip())
    files_dir = extract_keystrokes(str(zip_path), str(tmp_path / "out"), progress=True)
    assert files_dir.endswith("files")
