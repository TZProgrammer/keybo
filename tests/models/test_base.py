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


# --- target space: MS (incumbent) vs LOGRAT (log(ms*wpm/12000), adopted 2026-07-10) -------


def make_space_meta(target_space=None, feature_names=("distance", "wpm")):
    extra = {}
    if target_space is not None:
        extra = {"training": {"target_space": target_space}}
    return ModelMetadata(
        feature_version="v1",
        feature_names=list(feature_names),
        wpm_range=(60, 100),
        ngram="bigram",
        extra=extra,
    )


def test_target_space_defaults_to_ms():
    assert ConstantModel(make_meta()).target_space == "MS"
    assert ConstantModel(make_space_meta()).target_space == "MS"


def test_target_space_reads_sidecar_case_insensitively():
    assert ConstantModel(make_space_meta("LOGRAT")).target_space == "LOGRAT"
    assert ConstantModel(make_space_meta("lograt")).target_space == "LOGRAT"


def test_target_space_rejects_unknown_space():
    m = ConstantModel(make_space_meta("BOGUS"))
    with pytest.raises(ValueError, match="BOGUS"):
        _ = m.target_space


def test_to_ms_is_identity_for_ms_models():
    m = ConstantModel(make_space_meta(), value=123.0)
    X = np.array([[1.0, 90.0], [2.0, 60.0]])
    pred = m.predict(X)
    assert m.to_ms(pred, X).tolist() == pred.tolist()


def test_to_ms_converts_lograt_predictions_via_the_wpm_column():
    # predict() returns log(ms*wpm/12000); ms = exp(pred) * 12000 / wpm.
    m = ConstantModel(make_space_meta("LOGRAT"), value=0.0)
    X = np.array([[1.0, 90.0], [2.0, 60.0]])
    out = m.to_ms(m.predict(X), X)
    assert out == pytest.approx([12000 / 90, 12000 / 60])


def test_predict_ms_is_predict_plus_conversion():
    m = ConstantModel(make_space_meta("LOGRAT"), value=1.0)
    X = np.array([[0.5, 120.0]])
    assert m.predict_ms(X) == pytest.approx([np.exp(1.0) * 12000 / 120])


def test_to_ms_rejects_nonpositive_wpm_for_lograt():
    # A LOGRAT prediction at wpm<=0 is meaningless (division by zero pace); fail loudly
    # rather than emit inf into a QAP table (scorers default target_wpm=0.0).
    m = ConstantModel(make_space_meta("LOGRAT"), value=0.0)
    X = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="wpm"):
        m.to_ms(m.predict(X), X)


def test_to_ms_rejects_lograt_model_without_wpm_feature():
    m = ConstantModel(make_space_meta("LOGRAT", feature_names=("distance", "angle")))
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="wpm"):
        m.to_ms(m.predict(X), X)
