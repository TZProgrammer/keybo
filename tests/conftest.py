"""Shared pytest fixtures and path helpers for the keybo test suite."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@pytest.fixture
def corpus_dir() -> Path:
    """Directory holding the committed n-gram frequency files."""
    return CORPUS_DIR


#: The skipgram table the campaign's frozen gauge boards were computed on.
#: ``1-skip31.txt`` IS the trigram marginalization ``skip(a,c) = sum_b tri(a,b,c)``
#: (``data/build_corpus.py``, verified byte-exact there); ``1-skip.txt`` is a different,
#: unreproducible pass. Gauges that must reproduce a frozen board use this one.
PRODUCTION_SKIPGRAMS = "1-skip31.txt"


@pytest.fixture(scope="session")
def corpora() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """``(bigrams, skipgrams, trigrams)`` from the committed production corpus.

    Session-scoped: the trigram table is ~100k rows and several tests want it.
    """
    from keybo.data.corpus import load_frequencies

    return (
        load_frequencies(str(CORPUS_DIR / "bigrams.txt")),
        load_frequencies(str(CORPUS_DIR / PRODUCTION_SKIPGRAMS)),
        load_frequencies(str(CORPUS_DIR / "trigrams.txt")),
    )
