"""Model-backed scorers: sum a TypingModel's predicted times over the corpus.

The fitness of a layout is the total predicted time to type the corpus:

    fitness(layout) = sum over n-grams of  predict_ms(features(layout, n-gram)) * frequency

Two things worth noting versus the original:

- Every n-gram in the supplied corpus is scored. There is no hardcoded character subset, so
  no key is invisible to the objective (bug #2).
- Frequency is ONLY the weight in that sum, never a feature input (OQ-1): features are pure
  geometry + wpm, so predictions cannot memorize practiced positions via frequency.

Predictions are summed in MILLISECONDS, not raw model output: a LOGRAT-space model's raw
predict() is log(ms*wpm/12000) (T-REL, 2026-07-10), and a sum of log-ratios is not a time.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from keybo.features import bigram_features, trigram_features
from keybo.layout import Layout
from keybo.scoring.base import IScorer


def predict_ms(model, X: np.ndarray) -> np.ndarray:
    """The scorer-side prediction: always milliseconds.

    Routes through ``TypingModel.predict_ms`` (target-space aware) when the model provides
    it; a plain object with only ``predict`` (test stubs) is ms-space by construction.
    """
    fn = getattr(model, "predict_ms", None)
    return fn(X) if fn is not None else model.predict(X)


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
        # Score every bigram whose characters are all typable on this board. Space counts:
        # it is a fixed key (layout.has_key(" ") is True), and the training pipeline emits
        # space bigrams, so the scorer must include them for train/serve parity. A char the
        # board genuinely lacks (e.g. ';' when the layout carries '-') is skipped, rather
        # than mapped to a phantom position as the original code did.
        vectors = []
        freqs = []
        positions = []
        for bg, freq in zip(self._bigrams, self._freqs, strict=True):
            if all(layout.has_key(c) for c in bg):
                vectors.append(bigram_features(layout, bg, wpm=self.target_wpm))
                freqs.append(freq)
                positions.append((layout.pos(bg[0]), layout.pos(bg[1])))
        if not vectors:
            return 0.0
        X = np.vstack(vectors)
        metadata = getattr(self.model, "metadata", None)
        training = (getattr(metadata, "extra", None) or {}).get("training") or {}
        calibration = training.get("calibration")
        if calibration and calibration.get("deltas_ms"):
            predict_at = getattr(self.model, "predict_ms_at", None)
            if predict_at is None:
                raise TypeError("calibrated bigram model must provide predict_ms_at")
            predicted = np.array(
                [
                    predict_at(row.reshape(1, -1), pair)[0]
                    for row, pair in zip(X, positions, strict=True)
                ]
            )
        else:
            predicted = predict_ms(self.model, X)
        return float(np.sum(predicted * np.array(freqs)))


class TrigramModelScorer(_ModelScorerBase):
    """Scores a layout using a trigram typing-time model."""

    def __init__(
        self,
        model,
        trigram_freqs: Mapping[str, int],
        target_wpm: float = 0.0,
    ) -> None:
        from keybo.models.base import reject_calibrated_trigram_model

        reject_calibrated_trigram_model(model, "TrigramModelScorer")
        super().__init__(model, target_wpm)
        self._trigrams = list(trigram_freqs.keys())
        self._freqs = np.array([trigram_freqs[t] for t in self._trigrams], dtype=np.float64)

    def fitness(self, layout: Layout) -> float:
        # As with bigrams: score trigrams typable on this board (space included), skip those
        # using a character the board genuinely lacks.
        rows = []
        freqs = []
        for tg, freq in zip(self._trigrams, self._freqs, strict=True):
            if not all(layout.has_key(c) for c in tg):
                continue
            rows.append(trigram_features(layout, tg, wpm=self.target_wpm))
            freqs.append(freq)
        if not rows:
            return 0.0
        predicted = predict_ms(self.model, np.vstack(rows))
        return float(np.sum(predicted * np.array(freqs)))
