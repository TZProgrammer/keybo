"""OQ-1 probe: does the frequency FEATURE change how a trained model ranks layouts?

Method (synthetic, so directional — the definitive version needs real data, see the OQ-1
artifact): build bistroke training rows whose durations depend ONLY on geometry (distance +
same-finger penalty), i.e. a world where frequency has no causal effect on time. Train two
XGBoost models on identical rows — one WITH the freq feature populated from a corpus-like
distribution, one with freq pinned to a constant — and compare how they rank the named
layouts. In this frequency-neutral world, any ranking difference is pure freq-feature
artifact: the "disguised geometry bonus" OQ-1 worries about.

Run:  /tmp/keybo_venv/bin/python agent-artifacts/experiments/oq1_freq_feature_probe.py
"""

import numpy as np

from keybo.data.corpus import load_frequencies
from keybo.features import BIGRAM_FEATURE_NAMES, bigram_features
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.model_scorer import BigramModelScorer

rng = np.random.default_rng(0)
QWERTY = Layout(NAMED_LAYOUTS["qwerty"], G)
FREQS = load_frequencies("data/corpus/bigrams.txt")

FREQ_IDX = BIGRAM_FEATURE_NAMES.index("freq")
DIST_IDX = BIGRAM_FEATURE_NAMES.index("distance")
SF_IDX = BIGRAM_FEATURE_NAMES.index("same_finger")


def geometry_time(vec: np.ndarray) -> float:
    """Ground-truth typing time in this synthetic world: geometry only, NO freq effect."""
    return 100.0 + 30.0 * vec[DIST_IDX] + 80.0 * vec[SF_IDX] + rng.normal(0, 5)


def training_matrix(with_freq: bool):
    """Rows: every scoreable qwerty bigram, duration from geometry_time."""
    X, y = [], []
    for bg, freq in FREQS.items():
        if not all(QWERTY.has_key(c) for c in bg):
            continue
        vec = bigram_features(QWERTY, bg, freq=freq if with_freq else 1.0, wpm=90)
        X.append(vec)
        y.append(geometry_time(vec))
    return np.vstack(X), np.array(y)


def train(with_freq: bool) -> XGBoostTypingModel:
    X, y = training_matrix(with_freq)
    meta = ModelMetadata(
        feature_version="probe",
        feature_names=BIGRAM_FEATURE_NAMES,
        wpm_range=(60, 120),
        ngram="bigram",
    )
    m = XGBoostTypingModel(meta, n_estimators=150, max_depth=5)
    m.fit(X, y)
    return m


def rank_layouts(model, with_freq: bool):
    # The WEIGHT stays the real corpus frequency in both arms -- only the FEATURE differs.
    scores = {}
    for name, chars in NAMED_LAYOUTS.items():
        lay = Layout(chars, G)
        if with_freq:
            scores[name] = BigramModelScorer(model, bigram_freqs=FREQS, target_wpm=90).fitness(lay)
        else:
            # Pin the freq FEATURE to 1.0 while keeping the corpus weight: build vectors manually.
            keep = [
                (bigram_features(lay, bg, freq=1.0, wpm=90), f)
                for bg, f in FREQS.items()
                if all(lay.has_key(c) for c in bg)
            ]
            X = np.vstack([v for v, _ in keep])
            w = np.array([f for _, f in keep], dtype=np.float64)
            scores[name] = float(np.sum(model.predict(X) * w))
    return sorted(scores, key=scores.get), scores


def main():
    m_with = train(with_freq=True)
    m_without = train(with_freq=False)

    rank_with, s_with = rank_layouts(m_with, with_freq=True)
    rank_without, s_without = rank_layouts(m_without, with_freq=False)

    print("Ground truth in this synthetic world: time depends ONLY on geometry.")
    print("So the two models SHOULD produce the same ranking; any difference is a")
    print("frequency-feature artifact.\n")
    print(f"ranking WITH freq feature   : {rank_with}")
    print(f"ranking WITHOUT freq feature: {rank_without}")
    agree = rank_with == rank_without
    print(f"\nrankings agree: {agree}")

    # Also quantify: correlation between each model's per-bigram predictions on a NOVEL layout
    novel = Layout(NAMED_LAYOUTS["graphite"], G)
    vec_pairs = []
    for bg, f in FREQS.items():
        if all(novel.has_key(c) for c in bg):
            v_with = bigram_features(novel, bg, freq=f, wpm=90)
            v_without = bigram_features(novel, bg, freq=1.0, wpm=90)
            vec_pairs.append((v_with, v_without))
    Xw = np.vstack([a for a, _ in vec_pairs])
    Xo = np.vstack([b for _, b in vec_pairs])
    pw, po = m_with.predict(Xw), m_without.predict(Xo)
    r = np.corrcoef(pw, po)[0, 1]
    print(f"per-bigram prediction correlation on a novel layout (graphite): r={r:.4f}")
    print(f"mean |prediction difference|: {np.mean(np.abs(pw - po)):.2f} ms")


if __name__ == "__main__":
    main()
