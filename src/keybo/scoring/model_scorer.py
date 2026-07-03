"""Model-backed scorers: sum a TypingModel's predicted times over the corpus.

The fitness of a layout is the total predicted time to type the corpus:

    fitness(layout) = sum over n-grams of  predict(features(layout, n-gram)) * frequency

Two things worth noting versus the original:

- Every n-gram in the supplied corpus is scored. There is no hardcoded character subset, so
  no key is invisible to the objective (bug #2).
- Feature vectors are built with the shared pipeline and predicted in a single batch, which
  is both correct (identical to training features) and fast.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from keybo.features import bigram_features, trigram_features
from keybo.layout import Layout
from keybo.scoring.base import IScorer


class _ModelScorerBase(IScorer):
    def __init__(self, model, target_wpm: float = 0.0) -> None:
        self.model = model
        self.target_wpm = target_wpm


class BigramModelScorer(_ModelScorerBase):
    """Scores a layout using a bigram typing-time model."""

    def __init__(self, model, bigram_freqs: Mapping[str, int], target_wpm: float = 0.0) -> None:
        super().__init__(model, target_wpm)
        # Freeze the corpus into parallel lists so feature order is stable across calls.
        self._bigrams = list(bigram_freqs.keys())
        self._freqs = np.array([bigram_freqs[b] for b in self._bigrams], dtype=np.float64)

    def fitness(self, layout: Layout) -> float:
        # Only score bigrams whose characters are all on this layout. A corpus n-gram using
        # a character the layout doesn't contain (e.g. ';' when the layout has '-') is simply
        # not typable here and is skipped -- rather than silently mapping it to a phantom
        # position, as the original code did.
        chars = set(layout.chars)
        vectors = []
        freqs = []
        for bg, freq in zip(self._bigrams, self._freqs, strict=True):
            if all(c in chars for c in bg):
                vectors.append(bigram_features(layout, bg, freq=freq, wpm=self.target_wpm))
                freqs.append(freq)
        if not vectors:
            return 0.0
        predicted = self.model.predict(np.vstack(vectors))
        return float(np.sum(predicted * np.array(freqs)))


class TrigramModelScorer(_ModelScorerBase):
    """Scores a layout using a trigram typing-time model."""

    def __init__(
        self,
        model,
        trigram_freqs: Mapping[str, int],
        bigram_freqs: Mapping[str, int] | None = None,
        skipgram_freqs: Mapping[str, int] | None = None,
        target_wpm: float = 0.0,
    ) -> None:
        super().__init__(model, target_wpm)
        self._trigrams = list(trigram_freqs.keys())
        self._freqs = np.array([trigram_freqs[t] for t in self._trigrams], dtype=np.float64)
        self._bg = dict(bigram_freqs or {})
        self._sg = dict(skipgram_freqs or {})

    def fitness(self, layout: Layout) -> float:
        # As with bigrams, skip any trigram using a character not on this layout.
        chars = set(layout.chars)
        rows = []
        freqs = []
        for tg, freq in zip(self._trigrams, self._freqs, strict=True):
            if not all(c in chars for c in tg):
                continue
            rows.append(
                trigram_features(
                    layout,
                    tg,
                    tg_freq=freq,
                    bg1_freq=self._bg.get(tg[:2], 1),
                    bg2_freq=self._bg.get(tg[1:], 1),
                    sg_freq=self._sg.get(tg[0] + tg[2], 1),
                    wpm=self.target_wpm,
                )
            )
            freqs.append(freq)
        if not rows:
            return 0.0
        predicted = self.model.predict(np.vstack(rows))
        return float(np.sum(predicted * np.array(freqs)))
