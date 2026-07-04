"""OQ-1 probe v2: does the frequency FEATURE change how a trained model ranks layouts?

v1 post-mortem (fable-audit finding 1, 2026-07-04 — kept so nobody repeats it): the original
probe built ONE synthetic world in which frequency had zero causal effect, ran ONE seed, and
reported "rankings agree". That was (a) near-tautological — a world with no freq effect has
no freq-correlated variance for a tree to absorb, so it cannot exhibit the feared
confound-absorption failure mode; and (b) seed-fragile — across 20 seeds, ~4 produced ranking
disagreements on the nearly-tied graphite/semimak pair.

v2 therefore runs BOTH worlds across a SEED SWEEP and reports distributions:

  world "neutral"   : time = f(geometry only)             (v1's world, kept for contrast)
  world "confounded": time = f(geometry) + practice(freq)  (the world OQ-1 actually fears:
                       a real frequency effect that exists on the trained layout and
                       transfers additively — the model may entangle it with geometry)

For each world x seed: train WITH-freq and WITHOUT-freq models on qwerty-derived rows,
then compare (1) layout-ranking agreement between arms, and (2) per-bigram Spearman rho of
each arm against the WORLD'S GEOMETRY-ONLY component on a novel layout — the ranking-relevant
signal a layout optimizer needs.

This probe is ILLUSTRATIVE, not decisive: the citable evidence for OQ-1's lean remains the
serve-scale saturation measurement; the decisive experiment is the real-data LOLO protocol in
agent-artifacts/OQ1-frequency-feature.md.

Run:  /tmp/keybo_venv/bin/python agent-artifacts/experiments/oq1_freq_feature_probe.py
"""

import numpy as np
from scipy.stats import spearmanr

from keybo.data.corpus import load_frequencies
from keybo.features import BIGRAM_FEATURE_NAMES, bigram_features
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel

QWERTY = Layout(NAMED_LAYOUTS["qwerty"], G)
FREQS = load_frequencies("data/corpus/bigrams.txt")
DIST = BIGRAM_FEATURE_NAMES.index("distance")
SF = BIGRAM_FEATURE_NAMES.index("same_finger")
SEEDS = range(12)
NOVEL = "graphite"  # evaluation layout not used for training rows

META = ModelMetadata(
    feature_version="probe",
    feature_names=BIGRAM_FEATURE_NAMES,
    wpm_range=(60, 120),
    ngram="bigram",
)


def geometry_component(vec: np.ndarray) -> float:
    return 100.0 + 30.0 * vec[DIST] + 80.0 * vec[SF]


def practice_component(freq: float) -> float:
    # A real, additive practice effect: frequent patterns are faster everywhere.
    return -12.0 * np.log1p(freq) / np.log(10)


def make_training(world: str, with_freq: bool, rng) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for bg, f in FREQS.items():
        if not all(QWERTY.has_key(c) for c in bg):
            continue
        vec = bigram_features(QWERTY, bg, freq=f if with_freq else 1.0, wpm=90)
        t = geometry_component(vec)
        if world == "confounded":
            t += practice_component(f)
        X.append(vec)
        y.append(t + rng.normal(0, 5))
    return np.vstack(X), np.array(y)


def layout_ranking(model, with_freq: bool):
    scores = {}
    for name, chars in NAMED_LAYOUTS.items():
        lay = Layout(chars, G)
        keep = [
            (bigram_features(lay, bg, freq=(f if with_freq else 1.0), wpm=90), f)
            for bg, f in FREQS.items()
            if all(lay.has_key(c) for c in bg)
        ]
        X = np.vstack([v for v, _ in keep])
        w = np.array([f for _, f in keep], dtype=np.float64)
        scores[name] = float(np.sum(model.predict(X) * w))
    return tuple(sorted(scores, key=scores.get))


def novel_geometry_rho(model, with_freq: bool) -> float:
    """Spearman rho of predictions vs the GEOMETRY-ONLY truth on the novel layout."""
    lay = Layout(NAMED_LAYOUTS[NOVEL], G)
    vecs, truths = [], []
    for bg, f in FREQS.items():
        if not all(lay.has_key(c) for c in bg):
            continue
        v = bigram_features(lay, bg, freq=(f if with_freq else 1.0), wpm=90)
        vecs.append(v)
        truths.append(geometry_component(v))
    preds = model.predict(np.vstack(vecs))
    return float(spearmanr(preds, truths).statistic)


def main():
    for world in ("neutral", "confounded"):
        agree = 0
        rho_with, rho_without = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            models = {}
            for wf in (True, False):
                X, y = make_training(world, wf, rng)
                m = XGBoostTypingModel(META, n_estimators=150, max_depth=5)
                m.fit(X, y)
                models[wf] = m
            agree += layout_ranking(models[True], True) == layout_ranking(models[False], False)
            rho_with.append(novel_geometry_rho(models[True], True))
            rho_without.append(novel_geometry_rho(models[False], False))

        n = len(list(SEEDS))
        print(f"world={world}:")
        print(f"  ranking agreement across seeds: {agree}/{n}")
        print(
            f"  novel-layout rho vs geometry truth: "
            f"WITH-freq {np.mean(rho_with):.4f}±{np.std(rho_with):.4f}  "
            f"WITHOUT-freq {np.mean(rho_without):.4f}±{np.std(rho_without):.4f}"
        )
        better = sum(w < wo for w, wo in zip(rho_with, rho_without, strict=True))
        print(f"  seeds where WITHOUT-freq tracks geometry better: {better}/{n}")
        print()

    print("Interpretation: the 'neutral' world cannot exhibit the feared confound (kept for")
    print("contrast with v1); the 'confounded' world can. If WITHOUT-freq tracks the")
    print("geometry-only signal better there, the freq feature is absorbing practice/")
    print("geometry-entangled variance that does not help ranking novel layouts.")


if __name__ == "__main__":
    main()
