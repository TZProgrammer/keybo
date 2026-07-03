"""Load n-gram frequency tables (the English corpus counts used to weight the objective).

File format is one ``ngram<TAB>count`` per line. The n-gram may contain a space (the space
key), so the key is taken verbatim from before the tab and is NOT stripped of surrounding
whitespace — ``"e "`` (e then space) is a different bigram from ``"e"``.

Fixes bug #4: loaders take explicit file paths and a missing path raises, rather than the
old code's silent-empty behavior when a literal placeholder string was passed by mistake.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def load_frequencies(path: str) -> dict[str, int]:
    """Load one ``ngram<TAB>count`` frequency file into a dict.

    Raises ``FileNotFoundError`` if the path does not exist (so a wrong path fails loudly
    instead of yielding an empty table).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"frequency file not found: {path}")

    freqs: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            ngram, _, count = line.partition("\t")
            count = count.strip()
            if not ngram or not count:
                continue
            try:
                freqs[ngram] = int(count)
            except ValueError:
                continue
    return freqs


@dataclass(frozen=True)
class Corpus:
    """The three frequency tables the objective and models draw on."""

    trigrams: dict[str, int]
    bigrams: dict[str, int]
    skipgrams: dict[str, int]


def load_corpus(trigrams: str, bigrams: str, skipgrams: str) -> Corpus:
    """Load all three frequency tables from their explicit paths."""
    return Corpus(
        trigrams=load_frequencies(trigrams),
        bigrams=load_frequencies(bigrams),
        skipgrams=load_frequencies(skipgrams),
    )
