"""Pattern-class effect curves: how the model prices SFB / rolls / alternation /
scissors / LSB as WPM changes.

Two complementary estimators over the served surface (the 31x31 position-pair table at
each WPM — exactly what the optimizer consumes):

1. **Class contrast** (model-agnostic): mean predicted ms of the class's position pairs
   minus the mean of the reference class (alternation), at each WPM. This is the
   effective price the optimizer sees, all features included.
2. **SHAP attribution** (model-internal): mean SHAP contribution of the class's
   *defining* feature column(s) over the class's pairs, at each WPM. This shows how much
   of the price the model attributes to the named feature vs correlated geometry.

The two answer different questions — the contrast is "what does an SFB cost at 120 wpm?";
the SHAP curve is "does the model know it's the SFB-ness that costs?". Divergence between
them (a big contrast carried by non-SFB features) is itself a finding.

Both are corpus-weighted by default: each position pair is weighted by the corpus mass a
representative layout puts on it — pass ``layout`` to weight by that layout's assignment,
or leave None for uniform pair weighting (pure geometry view).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb

from keybo.features import bigram_features_from_positions
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30, Geometry, Position
from keybo.layout import Layout
from keybo.models.xgboost_model import XGBoostTypingModel

#: pattern class -> (predicate over (geometry, a, b), defining SHAP feature names)
#: Feature names must exist in BIGRAM_FEATURE_NAMES; missing ones are skipped with a note.
PATTERN_CLASSES: dict[str, tuple] = {
    "sfb": (
        lambda g, a, b: C.same_finger(g, a, b) and a != b,
        ["same_finger"],
    ),
    "inroll": (C.is_inwards, ["inwards"]),
    "outroll": (C.is_outwards, ["outwards"]),
    "alternate": (
        lambda g, a, b: C.classify_positions(g, a, b) is C.BigramClass.ALTERNATE,
        ["same_hand"],
    ),
    "scissor": (C.is_scissor, ["scissor"]),
    "lsb": (C.is_lsb, ["lsb"]),
    "same_hand_other": (
        lambda g, a, b: (
            C.classify_positions(g, a, b) is C.BigramClass.SAME_HAND
            and not C.is_inwards(g, a, b)
            and not C.is_outwards(g, a, b)
        ),
        ["same_hand"],
    ),
}

REFERENCE_CLASS = "alternate"


@dataclass
class EffectCurves:
    """Per-class curves over the WPM axis. All times in ms."""

    wpms: list[float]
    #: class -> mean predicted ms per wpm (corpus- or uniform-weighted over its pairs)
    class_mean_ms: dict[str, list[float]]
    #: class -> contrast vs REFERENCE_CLASS per wpm (class mean - reference mean)
    contrast_ms: dict[str, list[float]]
    #: class -> mean SHAP of its defining feature column(s) over its pairs, per wpm
    shap_ms: dict[str, list[float]]
    #: class -> number of position pairs in the class (wpm-invariant)
    n_pairs: dict[str, int]
    weighted_by: str = "uniform"
    notes: list[str] = field(default_factory=list)

    def contrast_pct(self) -> dict[str, list[float]]:
        """Contrast as a percentage of the reference class's mean time per WPM."""
        ref = self.class_mean_ms[REFERENCE_CLASS]
        return {
            cls: [100.0 * c / r if r else float("nan") for c, r in zip(ys, ref, strict=True)]
            for cls, ys in self.contrast_ms.items()
        }

    def to_dict(self) -> dict:
        return {
            "wpms": self.wpms,
            "class_mean_ms": self.class_mean_ms,
            "contrast_vs_alternate_ms": self.contrast_ms,
            "contrast_vs_alternate_pct": self.contrast_pct(),
            "shap_of_defining_feature_ms": self.shap_ms,
            "n_pairs": self.n_pairs,
            "weighted_by": self.weighted_by,
            "notes": self.notes,
        }


def _pair_weights(
    positions: list[Position],
    layout: Layout | None,
    bigram_freqs: dict[str, int] | None,
) -> np.ndarray:
    """Weight per ordered position pair: corpus mass if a layout is given, else uniform."""
    n = len(positions)
    if layout is None or not bigram_freqs:
        return np.ones(n * n)
    pos_of = {}
    for ch in layout.chars:
        pos_of[ch] = layout.pos(ch)
    pos_of[" "] = layout.geometry.space_position
    mass: dict[tuple[Position, Position], float] = defaultdict(float)
    for bg, f in bigram_freqs.items():
        if len(bg) == 2 and all(c in pos_of for c in bg):
            mass[(pos_of[bg[0]], pos_of[bg[1]])] += float(f)
    idx = {p: i for i, p in enumerate(positions)}
    w = np.zeros(n * n)
    for (a, b), m in mass.items():
        w[idx[a] * n + idx[b]] = m
    return w


def compute_effect_curves(
    models: list[XGBoostTypingModel],
    wpms: list[float] | None = None,
    geometry: Geometry = ROW_STAGGERED_30,
    layout: Layout | None = None,
    bigram_freqs: dict[str, int] | None = None,
) -> EffectCurves:
    """Class contrast + SHAP curves over ``wpms`` for an ensemble of bigram models.

    Models are averaged (the production 3-seed convention). ``layout`` +
    ``bigram_freqs`` switch pair weighting from uniform to that layout's corpus mass.
    """
    if wpms is None:
        wpms = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    if not models:
        raise ValueError("need at least one model")
    for m in models:
        if m.metadata.ngram != "bigram":
            raise ValueError("effect curves are defined on bigram models")

    positions = [*geometry.slots, geometry.space_position]
    n = len(positions)
    names = list(models[0].metadata.feature_names)

    # class membership masks over the ordered-pair grid (wpm-invariant)
    masks: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for cls, (pred, _feats) in PATTERN_CLASSES.items():
        mask = np.zeros(n * n, dtype=bool)
        for i, a in enumerate(positions):
            for j, b in enumerate(positions):
                if i != j and a[0] != 0 and b[0] != 0 and pred(geometry, a, b):
                    mask[i * n + j] = True
        masks[cls] = mask

    weights = _pair_weights(positions, layout, bigram_freqs)
    weighted_by = "uniform" if layout is None or not bigram_freqs else "layout-corpus"
    for cls, mask in masks.items():
        if weights[mask].sum() == 0 and mask.any():
            notes.append(f"{cls}: zero corpus mass under this layout; uniform fallback")

    class_mean: dict[str, list[float]] = {c: [] for c in masks}
    contrast: dict[str, list[float]] = {c: [] for c in masks}
    shap_curves: dict[str, list[float]] = {c: [] for c in masks}

    for wpm in wpms:
        X = np.vstack(
            [
                bigram_features_from_positions(geometry, (a, b), wpm=wpm)
                for a in positions
                for b in positions
            ]
        )
        pred_ms = np.mean([m.predict_ms(X) for m in models], axis=0)

        # mean SHAP of each class's defining columns, averaged over models
        shap_by_model = []
        for m in models:
            booster = m._regressor.get_booster()
            contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
            shap_by_model.append(contribs)
        shap_mean = np.mean(shap_by_model, axis=0)  # (n*n, F) in model target space

        # convert per-row SHAP from target space to ms: for LOGRAT, d(ms) ~ ms * d(log);
        # a first-order local conversion at each row's predicted ms.
        space = getattr(models[0], "target_space", "MS")
        if str(space).upper() == "LOGRAT":
            shap_ms_matrix = shap_mean * pred_ms[:, None]
        else:
            shap_ms_matrix = shap_mean

        for cls, mask in masks.items():
            if not mask.any():
                class_mean[cls].append(float("nan"))
                shap_curves[cls].append(float("nan"))
                continue
            w = weights[mask]
            if w.sum() == 0:
                w = np.ones(mask.sum())
            class_mean[cls].append(float(np.average(pred_ms[mask], weights=w)))
            feat_cols = [names.index(f) for f in PATTERN_CLASSES[cls][1] if f in names]
            if feat_cols:
                shap_curves[cls].append(
                    float(np.average(shap_ms_matrix[mask][:, feat_cols].sum(axis=1), weights=w))
                )
            else:
                shap_curves[cls].append(float("nan"))

        ref = class_mean[REFERENCE_CLASS][-1]
        for cls in masks:
            contrast[cls].append(class_mean[cls][-1] - ref)

    missing = [f for cls, (_p, feats) in PATTERN_CLASSES.items() for f in feats if f not in names]
    if missing:
        notes.append(f"features absent from schema (SHAP curves skipped): {sorted(set(missing))}")

    return EffectCurves(
        wpms=list(wpms),
        class_mean_ms=class_mean,
        contrast_ms=contrast,
        shap_ms=shap_curves,
        n_pairs={c: int(m.sum()) for c, m in masks.items()},
        weighted_by=weighted_by,
        notes=notes,
    )


def render_effect_curves(curves: EffectCurves, out_prefix: str) -> list[str]:
    """Three figures: contrast-vs-wpm (ms), contrast-vs-wpm (%), SHAP-attribution-vs-wpm."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {
        "sfb": "#e34948",
        "inroll": "#3987e5",
        "outroll": "#7db4ef",
        "alternate": "#8a8988",
        "scissor": "#c95fd0",
        "lsb": "#e6a23c",
        "same_hand_other": "#55b06b",
    }
    written = []

    def contrast_fig(values: dict[str, list[float]], ylabel: str, tag: str) -> str:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for cls, ys in values.items():
            if cls == REFERENCE_CLASS:
                continue
            ax.plot(
                curves.wpms,
                ys,
                marker="o",
                markersize=4,
                linewidth=2,
                label=f"{cls} (n={curves.n_pairs[cls]})",
                color=palette.get(cls, "#40403e"),
            )
        ax.axhline(0, color="#8a8988", linewidth=1)
        ax.set_xlabel("WPM")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Pattern-class price vs WPM (contrast against {REFERENCE_CLASS}; "
            f"{curves.weighted_by} pair weights)"
        )
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e7e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        path = f"{out_prefix}_{tag}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    written.append(
        contrast_fig(curves.contrast_ms, f"predicted ms vs {REFERENCE_CLASS}", "contrast")
    )
    written.append(
        contrast_fig(
            curves.contrast_pct(),
            f"% of {REFERENCE_CLASS} mean time",
            "contrast_pct",
        )
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cls, ys in curves.shap_ms.items():
        if all(np.isnan(ys)):
            continue
        ax.plot(
            curves.wpms,
            ys,
            marker="o",
            markersize=4,
            linewidth=2,
            label=cls,
            color=palette.get(cls, "#40403e"),
        )
    ax.axhline(0, color="#8a8988", linewidth=1)
    ax.set_xlabel("WPM")
    ax.set_ylabel("mean SHAP of defining feature (ms)")
    ax.set_title("SHAP attribution to the class's own feature vs WPM")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e7e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = f"{out_prefix}_shap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    return written
