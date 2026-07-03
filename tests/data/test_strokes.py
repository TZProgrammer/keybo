"""Tests for loading the bistroke/tristroke TSV tables produced by the data pipeline."""

import pytest

from keybo.data.strokes import StrokeRow, iqr_average, load_strokes


def test_iqr_average_of_uniform_list():
    assert iqr_average([10, 10, 10, 10]) == pytest.approx(10.0)


def test_iqr_average_discards_outliers():
    data = [10, 11, 9, 10, 1000]  # 1000 is an outlier
    avg = iqr_average(data)
    assert avg < 50  # outlier excluded


def test_iqr_average_empty_is_zero():
    assert iqr_average([]) == 0.0


def write_tsv(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_load_strokes_parses_positions_ngram_and_samples(tmp_path):
    # positions<tab>ngram<tab>freq<tab>(wpm, dur)<tab>(wpm, dur)...
    line = "((-1, 3), (1, 2))\tth\t3\t(90, 120)\t(85, 130)\t(95, 110)"
    p = write_tsv(tmp_path / "bi.tsv", [line])
    rows = load_strokes(p, ngram_len=2, wpm_threshold=60, min_samples=1)
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, StrokeRow)
    assert r.ngram == "th"
    assert r.positions == ((-1, 3), (1, 2))
    assert (90, 120) in r.samples


def test_load_strokes_filters_below_wpm_threshold(tmp_path):
    line = "((-1, 3), (1, 2))\tth\t3\t(50, 120)\t(90, 130)"
    p = write_tsv(tmp_path / "bi.tsv", [line])
    rows = load_strokes(p, ngram_len=2, wpm_threshold=80, min_samples=1)
    # Only the (90, ...) sample clears wpm>=80.
    assert len(rows) == 1
    assert rows[0].samples == [(90, 130)]


def test_load_strokes_drops_rows_below_min_samples(tmp_path):
    line = "((-1, 3), (1, 2))\tth\t1\t(90, 120)"
    p = write_tsv(tmp_path / "bi.tsv", [line])
    rows = load_strokes(p, ngram_len=2, wpm_threshold=60, min_samples=5)
    assert rows == []


def test_load_strokes_skips_wrong_length_ngrams(tmp_path):
    lines = [
        "((-1, 3), (1, 2))\tth\t3\t(90, 120)",
        "((-1, 3), (1, 2), (3, 2))\tthe\t3\t(90, 120)",  # trigram in a bigram file
    ]
    p = write_tsv(tmp_path / "bi.tsv", lines)
    rows = load_strokes(p, ngram_len=2, wpm_threshold=60, min_samples=1)
    assert [r.ngram for r in rows] == ["th"]


def test_load_strokes_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_strokes(str(tmp_path / "nope.tsv"), ngram_len=2, wpm_threshold=60, min_samples=1)
