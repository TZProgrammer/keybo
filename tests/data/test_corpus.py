"""Tests for corpus n-gram frequency loading.

Regression for bug #4: the old trigram scorer passed the literal string ``"bigrams_file"``
(not a path) into the loader, so bigram frequencies silently loaded as empty. The typed
loader here takes real paths and a missing/bogus path is an explicit error, not a silent
empty dict.
"""

import pytest

from keybo.data.corpus import load_corpus, load_frequencies


def write(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_loads_ngram_tab_count(tmp_path):
    p = write(tmp_path / "bg.txt", ["th\t100", "he\t80", "an\t60"])
    freqs = load_frequencies(p)
    assert freqs["th"] == 100
    assert freqs["he"] == 80
    assert freqs["an"] == 60


def test_preserves_space_containing_ngrams(tmp_path):
    # "e " (e then space) and " t" (space then t) must NOT be stripped to "e"/"t".
    p = write(tmp_path / "bg.txt", ["e \t15", " t\t12", "th\t9"])
    freqs = load_frequencies(p)
    assert freqs["e "] == 15
    assert freqs[" t"] == 12
    assert "e" not in freqs  # stripping would have produced this


def test_missing_ngram_defaults_to_zero_via_get(tmp_path):
    p = write(tmp_path / "bg.txt", ["th\t100"])
    freqs = load_frequencies(p)
    assert freqs.get("zz", 0) == 0


def test_regression_bug4_bogus_path_raises_not_silent_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frequencies(str(tmp_path / "does_not_exist.txt"))


def test_skips_malformed_lines(tmp_path):
    p = write(tmp_path / "bg.txt", ["th\t100", "garbage-no-tab", "he\t80", "\t"])
    freqs = load_frequencies(p)
    assert freqs == {"th": 100, "he": 80}


def test_load_corpus_reads_all_three_files(tmp_path):
    tg = write(tmp_path / "tg.txt", ["the\t100", "and\t50"])
    bg = write(tmp_path / "bg.txt", ["th\t200", "he\t150"])
    sg = write(tmp_path / "sk.txt", ["te\t30"])
    corpus = load_corpus(trigrams=tg, bigrams=bg, skipgrams=sg)
    assert corpus.trigrams["the"] == 100
    assert corpus.bigrams["th"] == 200
    assert corpus.skipgrams["te"] == 30


def test_load_corpus_real_files(corpus_dir):
    # Smoke test against the committed corpus.
    corpus = load_corpus(
        trigrams=str(corpus_dir / "trigrams.txt"),
        bigrams=str(corpus_dir / "bigrams.txt"),
        skipgrams=str(corpus_dir / "1-skip.txt"),
    )
    assert corpus.bigrams["th"] > 0
    assert len(corpus.trigrams) > 1000
