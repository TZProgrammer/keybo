"""The TypingModel seam: features -> predicted per-n-gram typing time.

A model is deliberately shallow — ``predict`` on a feature matrix — so the concrete
implementation (XGBoost today) is plug-and-play. What every model MUST carry is
:class:`ModelMetadata`, in particular the ``feature_version`` it was trained under. Saving
writes the model artifact plus a ``.meta.json`` sidecar; loading validates the stored
feature version against the caller's expectation and raises
:class:`FeatureVersionMismatch` on a mismatch. That check is the guard against scoring a
layout with a model whose features no longer match the pipeline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


class FeatureVersionMismatch(RuntimeError):
    """Raised when a saved model's feature version differs from what the caller expects."""


@dataclass(frozen=True)
class ModelMetadata:
    """Everything needed to know a saved model is safe to use, and how it was built."""

    feature_version: str
    feature_names: list[str]
    wpm_range: tuple[int, int]
    ngram: str  # "bigram" | "trigram"
    trained_at: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wpm_range"] = list(self.wpm_range)  # JSON has no tuples
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelMetadata:
        return cls(
            feature_version=d["feature_version"],
            feature_names=list(d["feature_names"]),
            wpm_range=tuple(d["wpm_range"]),  # type: ignore[arg-type]
            ngram=d["ngram"],
            trained_at=d.get("trained_at"),
            extra=d.get("extra", {}),
        )


def _sidecar_path(artifact_path: str) -> Path:
    return Path(artifact_path).with_suffix(".meta.json")


class TypingModel(ABC):
    """Abstract typing-time model. Subclasses implement predict + artifact (de)serialization."""

    def __init__(self, metadata: ModelMetadata) -> None:
        self.metadata = metadata

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict per-row typing time for a 2-D feature matrix."""

    # --- persistence: shared sidecar logic, artifact handled by the subclass -----------

    @abstractmethod
    def _save_artifact(self, artifact_path: str) -> None:
        """Write the model's own artifact (weights) to ``artifact_path``."""

    @classmethod
    @abstractmethod
    def _load_artifact(cls, artifact_path: str, metadata: ModelMetadata) -> TypingModel:
        """Reconstruct a model from its artifact and already-loaded metadata."""

    def save(self, path: str) -> None:
        """Save the model artifact to ``path`` and its metadata to ``<path>.meta.json``.

        Parent directories are created if missing: XGBoost's C++ writer otherwise dies with
        an opaque ``LocalFileSystem::Open ... No such file or directory`` — after the
        (potentially hours-long) training already ran.
        """
        parent = Path(path).parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self._save_artifact(path)
        _sidecar_path(path).write_text(json.dumps(self.metadata.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str, expected_feature_version: str | None = None) -> TypingModel:
        """Load a model, validating its feature version.

        ``expected_feature_version`` defaults to the current pipeline's version; pass an
        explicit value in tests or when intentionally loading an older model.
        """
        if expected_feature_version is None:
            from keybo.features.schema import FEATURE_VERSION

            expected_feature_version = FEATURE_VERSION

        sidecar = _sidecar_path(path)
        if not sidecar.exists():
            raise FileNotFoundError(f"missing metadata sidecar: {sidecar}")
        metadata = ModelMetadata.from_dict(json.loads(sidecar.read_text()))

        if metadata.feature_version != expected_feature_version:
            raise FeatureVersionMismatch(
                f"model was trained with feature_version={metadata.feature_version!r} "
                f"but the current pipeline is {expected_feature_version!r}; retrain the model"
            )
        return cls._load_artifact(path, metadata)
