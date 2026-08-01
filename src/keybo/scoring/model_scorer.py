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
    """Scores a layout using a bigram typing-time model.

    ``direction`` MUST match the frame the model was trained on: a widened model
    (``FEATURE_VERSION_DIRECTION``) is featurized on the 22-column frame, a served model on the
    20-column one. The default is the served frame, byte for byte. It exists so a
    narrow-vs-widened ranking A/B runs BOTH arms through this one reviewed scoring path,
    differing only in the frame — a driver re-implementing the has_key/space handling could
    drift from serving.
    """

    def __init__(
        self,
        model,
        bigram_freqs: Mapping[str, int],
        target_wpm: float = 0.0,
        direction: bool = False,
        kitchensink: bool = False,
    ) -> None:
        super().__init__(model, target_wpm)
        # Freeze the corpus into parallel lists so feature order is stable across calls.
        self._bigrams = list(bigram_freqs.keys())
        self._freqs = np.array([bigram_freqs[b] for b in self._bigrams], dtype=np.float64)
        self._direction = direction
        self._kitchensink = kitchensink

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
                vectors.append(
                    bigram_features(
                        layout,
                        bg,
                        wpm=self.target_wpm,
                        direction=self._direction,
                        kitchensink=self._kitchensink,
                    )
                )
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
    """Scores a layout using a trigram typing-time model.

    ``direction``/``kitchensink``/``abspos`` MUST match the frame the model was trained on — the
    same contract :class:`BigramModelScorer` documents. A mismatch is caught at CONSTRUCTION
    against the model's own ``feature_version`` stamp, because otherwise the failure surfaces as an
    XGBoost ``Feature shape mismatch, expected: 54, got 46`` raised deep inside ``fitness()`` after
    the whole corpus has been featurized (ABSPOS-1 hit exactly that).
    """

    def __init__(
        self,
        model,
        trigram_freqs: Mapping[str, int],
        target_wpm: float = 0.0,
        direction: bool = False,
        kitchensink: bool = False,
        abspos: bool = False,
    ) -> None:
        from keybo.models.base import reject_calibrated_trigram_model

        reject_calibrated_trigram_model(model, "TrigramModelScorer")
        super().__init__(model, target_wpm)
        self._trigrams = list(trigram_freqs.keys())
        self._freqs = np.array([trigram_freqs[t] for t in self._trigrams], dtype=np.float64)
        self._direction = direction
        self._kitchensink = kitchensink
        self._abspos = abspos
        self._check_frame_matches_stamp()

    def _check_frame_matches_stamp(self) -> None:
        """Fail fast when the requested frame contradicts the model's version stamp.

        The stamp is the model's own record of which frame it was trained on, so it is the one
        authority available here. Models with no metadata (test stubs) are left alone, as is an
        unrecognised stamp — neither is this class's to adjudicate.
        """
        from keybo.features.schema import (
            FEATURE_VERSION,
            FEATURE_VERSION_ABSPOS,
            FEATURE_VERSION_DIRECTION,
            FEATURE_VERSION_KITCHENSINK,
        )

        stamp = getattr(getattr(self.model, "metadata", None), "feature_version", None)
        expected = {
            FEATURE_VERSION_ABSPOS: (False, False, True),
            FEATURE_VERSION_KITCHENSINK: (False, True, False),
            FEATURE_VERSION_DIRECTION: (True, False, False),
            FEATURE_VERSION: (False, False, False),
        }.get(stamp)
        if expected is None:
            return
        got = (bool(self._direction), bool(self._kitchensink), bool(self._abspos))
        if got != expected:
            raise ValueError(
                f"frame flags (direction, kitchensink, abspos) = {got} do not match the model's "
                f"feature_version {stamp!r}, which was trained on {expected}; scoring would "
                f"featurize the wrong number of columns"
            )

    def fitness(self, layout: Layout) -> float:
        # As with bigrams: score trigrams typable on this board (space included), skip those
        # using a character the board genuinely lacks.
        rows = []
        freqs = []
        for tg, freq in zip(self._trigrams, self._freqs, strict=True):
            if not all(layout.has_key(c) for c in tg):
                continue
            rows.append(
                trigram_features(
                    layout,
                    tg,
                    wpm=self.target_wpm,
                    direction=self._direction,
                    kitchensink=self._kitchensink,
                    abspos=self._abspos,
                )
            )
            freqs.append(freq)
        if not rows:
            return 0.0
        predicted = predict_ms(self.model, np.vstack(rows))
        return float(np.sum(predicted * np.array(freqs)))
