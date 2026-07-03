"""Shared pytest fixtures and path helpers for the keybo test suite."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@pytest.fixture
def corpus_dir() -> Path:
    """Directory holding the committed n-gram frequency files."""
    return CORPUS_DIR
