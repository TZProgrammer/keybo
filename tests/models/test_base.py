"""Tests for the TypingModel metadata contract and the feature-version guard.

The guard is what prevents the train/serve-skew failure mode: a model trained on one
feature layout must refuse to load into code using a different one, loudly, instead of
silently producing garbage scores. (Contrast the old pickle approach, which loaded only by
accident of __main__ containing the right class.)
"""

import json

import numpy as np
import pytest

from keybo.models.base import (
    FeatureVersionMismatch,
    ModelMetadata,
    TypingModel,
)


class ConstantModel(TypingModel):
    """A minimal concrete model used to exercise the base-class contract."""

    def __init__(self, metadata: ModelMetadata, value: float = 1.5) -> None:
        super().__init__(metadata)
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)

    def _save_artifact(self, artifact_path: str) -> None:
        with open(artifact_path, "w") as f:
            json.dump({"value": self.value}, f)

    @classmethod
    def _load_artifact(cls, artifact_path: str, metadata: ModelMetadata) -> "ConstantModel":
        with open(artifact_path) as f:
            value = json.load(f)["value"]
        return cls(metadata, value=value)


def make_meta(feature_version="v1", ngram="bigram"):
    return ModelMetadata(
        feature_version=feature_version,
        feature_names=["freq", "wpm"],
        wpm_range=(60, 100),
        ngram=ngram,
    )


def test_predict_works():
    m = ConstantModel(make_meta(), value=2.0)
    out = m.predict(np.zeros((3, 2)))
    assert list(out) == [2.0, 2.0, 2.0]


def test_save_writes_artifact_and_sidecar(tmp_path):
    m = ConstantModel(make_meta(), value=3.0)
    path = tmp_path / "model.json"
    m.save(str(path))
    assert path.exists()
    assert path.with_suffix(".meta.json").exists()


def test_save_load_roundtrip_recovers_predictions(tmp_path):
    m = ConstantModel(make_meta(), value=4.25)
    path = tmp_path / "model.json"
    m.save(str(path))
    loaded = ConstantModel.load(str(path), expected_feature_version="v1")
    assert loaded.predict(np.zeros((2, 2))).tolist() == [4.25, 4.25]
    assert loaded.metadata.feature_version == "v1"
    assert loaded.metadata.ngram == "bigram"
    assert loaded.metadata.wpm_range == (60, 100)


def test_load_rejects_feature_version_mismatch(tmp_path):
    m = ConstantModel(make_meta(feature_version="OLD"), value=1.0)
    path = tmp_path / "model.json"
    m.save(str(path))
    # Corrupt the sidecar to a version the loader will not accept.
    with pytest.raises(FeatureVersionMismatch):
        ConstantModel.load(str(path), expected_feature_version="NEW")


def test_load_accepts_matching_feature_version(tmp_path):
    m = ConstantModel(make_meta(feature_version="MATCH"), value=1.0)
    path = tmp_path / "model.json"
    m.save(str(path))
    loaded = ConstantModel.load(str(path), expected_feature_version="MATCH")
    assert loaded.value == 1.0


def test_metadata_serializes_trained_at_when_provided():
    meta = ModelMetadata(
        feature_version="v1",
        feature_names=["a"],
        wpm_range=(60, 100),
        ngram="bigram",
        trained_at="2026-07-03T00:00:00Z",
    )
    d = meta.to_dict()
    assert d["trained_at"] == "2026-07-03T00:00:00Z"
    assert ModelMetadata.from_dict(d) == meta
