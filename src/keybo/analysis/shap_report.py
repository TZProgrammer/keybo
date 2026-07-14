"""SHAP explanation report for a trained XGBoost typing model.

Uses XGBoost's native TreeSHAP (``Booster.predict(pred_contribs=True)``) rather than the
``shap`` package's wrapper: it is exact for tree ensembles, has no extra dependency, and
its additivity (per-row contributions sum to the prediction) is asserted here on every
run. Interaction values come from the same API (``pred_interactions=True``); they are
O(features^2) per row, so the interaction pass runs on a subsample.

What the report answers, in order:

1. **Global ranking** — mean |SHAP| per feature: what the model actually uses. The bar
   chart pairs it with the feature's signed mean, so "important" and "which direction"
   are read together.
2. **Distribution** — a beeswarm-style panel: per-feature SHAP value scatter colored by
   the (normalized) feature value, exposing nonlinearity and asymmetric effects that a
   bar hides.
3. **Dependence** — for the top-K features, SHAP value vs feature value, exposing the
   learned response curve (e.g. is `distance`'s penalty linear? does `same_finger` cost
   depend on `wpm`?).
4. **Interactions** — top off-diagonal mean |interaction| pairs, the "the model prices X
   differently depending on Y" channels.

Explanations are computed on the feature matrix the caller provides. Two natural choices,
both supported by the CLI wrapper (``keybo shap-report``):

- the **serve grid** (all 31x31 position pairs / a sample of 31^3 triples at the scoring
  WPM) — explains the table the optimizer actually consumes ("what drives the objective");
- a **training-style matrix** (stroke rows -> per-WPM-group examples) — explains the model
  over the data distribution ("what drives predictions where the data lives").

The serve grid weights every geometry cell equally regardless of corpus frequency; the
training matrix reflects where samples concentrate. Report both when in doubt — a feature
important on the grid but not the data (or vice versa) is itself a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb

from keybo.models.xgboost_model import XGBoostTypingModel

# Categorical accents (dataviz palette): blue for magnitude bars, red for the
# negative-direction overlay; the beeswarm/dependence color ramp runs blue->red through a
# neutral gray midpoint (diverging: low->high feature value).
_BLUE = "#3987e5"
_RED = "#e34948"
_GRAY = "#8a8988"
_DIVERGING = [_BLUE, "#f0efec", _RED]


@dataclass
class ShapReport:
    """All arrays a report consumer needs, JSON-serializable via :meth:`to_dict`."""

    feature_names: list[str]
    base_value: float  # expected model output (bias term)
    mean_abs: np.ndarray  # (F,) global importance
    mean_signed: np.ndarray  # (F,) direction: + pushes predicted time up
    shap_values: np.ndarray  # (N, F) per-row contributions (bias column stripped)
    X: np.ndarray  # (N, F) the matrix explained
    interaction_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def ranking(self) -> list[tuple[str, float, float]]:
        """(name, mean|SHAP|, signed mean), sorted by importance descending."""
        order = np.argsort(-self.mean_abs)
        return [
            (self.feature_names[i], float(self.mean_abs[i]), float(self.mean_signed[i]))
            for i in order
        ]

    def importance_share(self) -> dict[str, float]:
        """Each feature's share of total mean |SHAP|, in percent (sums to 100)."""
        total = float(self.mean_abs.sum())
        if total <= 0:
            return {n: 0.0 for n in self.feature_names}
        return {
            n: 100.0 * float(self.mean_abs[self.feature_names.index(n)]) / total
            for n in self.feature_names
        }

    def to_dict(self) -> dict:
        share = self.importance_share()
        return {
            "base_value_ms": self.base_value,
            "ranking": [
                {
                    "feature": n,
                    "mean_abs_shap_ms": a,
                    "mean_signed_shap_ms": s,
                    "importance_share_pct": share[n],
                    "mean_abs_shap_pct_of_base": (
                        100.0 * a / self.base_value if self.base_value else None
                    ),
                }
                for n, a, s in self.ranking()
            ],
            "interaction_pairs": [
                {"a": a, "b": b, "mean_abs_interaction_ms": v} for a, b, v in self.interaction_pairs
            ],
        }


def compute_shap(
    model: XGBoostTypingModel,
    X: np.ndarray,
    interactions_max_rows: int = 2000,
    rng_seed: int = 0,
) -> ShapReport:
    """Exact TreeSHAP contributions for ``X``, with additivity asserted.

    ``interactions_max_rows`` bounds the O(F^2)-per-row interaction pass; rows are
    subsampled uniformly (seeded) above that.
    """
    booster = model._regressor.get_booster()
    names = list(model.metadata.feature_names)
    dmat = xgb.DMatrix(X)

    contribs = booster.predict(dmat, pred_contribs=True)
    pred = model.predict(X)
    # Additivity is the correctness invariant of TreeSHAP; if it fails, the artifact and
    # the wrapper disagree about the model and nothing downstream is trustworthy.
    if not np.allclose(contribs.sum(axis=1), pred, atol=1e-2):
        raise AssertionError("TreeSHAP contributions do not sum to model predictions")

    shap_values = contribs[:, :-1]
    base_value = float(contribs[:, -1].mean())

    n = X.shape[0]
    if n > interactions_max_rows:
        idx = np.random.default_rng(rng_seed).choice(n, interactions_max_rows, replace=False)
        inter_X = X[idx]
    else:
        inter_X = X
    inter = booster.predict(xgb.DMatrix(inter_X), pred_interactions=True)
    inter_abs = np.abs(inter[:, :-1, :-1]).mean(axis=0)  # strip bias row/col
    pairs = []
    f = len(names)
    for i in range(f):
        for j in range(i + 1, f):
            # off-diagonal entries are symmetric halves; report their combined magnitude
            pairs.append((names[i], names[j], float(inter_abs[i, j] + inter_abs[j, i])))
    pairs.sort(key=lambda p: -p[2])

    return ShapReport(
        feature_names=names,
        base_value=base_value,
        mean_abs=np.abs(shap_values).mean(axis=0),
        mean_signed=shap_values.mean(axis=0),
        shap_values=shap_values,
        X=X,
        interaction_pairs=pairs[:20],
    )


def _diverging_colors(values: np.ndarray) -> np.ndarray:
    """Map a 1-D array to the blue->gray->red diverging ramp by rank (robust to outliers)."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("keybo_div", _DIVERGING)
    order = values.argsort().argsort()  # ranks
    denom = max(len(values) - 1, 1)
    return cmap(order / denom)


def render_report(report: ShapReport, out_prefix: str, top_k: int = 12) -> list[str]:
    """Write the four report figures as PNGs; returns the written paths.

    ``out_prefix`` is a path prefix: ``<prefix>_ranking.png`` etc.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranking = report.ranking()
    written: list[str] = []

    # --- 1. global ranking bar (mean |SHAP|, annotated with signed direction + share) ---
    names = [r[0] for r in ranking]
    abs_vals = [r[1] for r in ranking]
    signed = [r[2] for r in ranking]
    share = report.importance_share()
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(names) + 1.2))
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, abs_vals, height=0.62, color=_BLUE, edgecolor="none")
    for y, name, a, s in zip(ypos, names, abs_vals, signed, strict=True):
        ax.text(
            a,
            y,
            f"  {a:.2f} ({share[name]:.1f}%, mean {s:+.2f})",
            va="center",
            fontsize=8,
            color="#40403e",
        )
    ax.set_yticks(ypos, names, fontsize=9)
    ax.set_xlabel("mean |SHAP| (ms of predicted time; % = share of total importance)")
    ax.set_title(f"Global feature importance — base value {report.base_value:.1f} ms")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e7e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = f"{out_prefix}_ranking.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # --- 2. beeswarm-style distribution panel (top-K) ----------------------------------
    k = min(top_k, len(names))
    fig, ax = plt.subplots(figsize=(8, 0.5 * k + 1.4))
    rng = np.random.default_rng(0)
    n = report.shap_values.shape[0]
    plot_idx = rng.choice(n, min(n, 4000), replace=False)
    for row, name in enumerate(names[:k]):
        col = report.feature_names.index(name)
        sv = report.shap_values[plot_idx, col]
        fv = report.X[plot_idx, col]
        y = (k - 1 - row) + (rng.random(len(sv)) - 0.5) * 0.55
        ax.scatter(sv, y, s=5, c=_diverging_colors(fv), linewidths=0, alpha=0.8, rasterized=True)
    ax.axvline(0, color=_GRAY, linewidth=1)
    ax.set_yticks(np.arange(k)[::-1], names[:k], fontsize=9)
    ax.set_xlabel("SHAP value (ms; color = feature value, blue low → red high)")
    ax.set_title("SHAP distribution per feature")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    path = f"{out_prefix}_beeswarm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # --- 3. dependence curves (top-K, small multiples) ----------------------------------
    ncols = 3
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.9 * nrows), squeeze=False)
    for i, name in enumerate(names[:k]):
        ax = axes[i // ncols][i % ncols]
        col = report.feature_names.index(name)
        fv = report.X[plot_idx, col]
        sv = report.shap_values[plot_idx, col]
        ax.scatter(fv, sv, s=5, c=_BLUE, linewidths=0, alpha=0.35, rasterized=True)
        # binned median overlay: the learned response curve without scatter noise
        uniq = np.unique(fv)
        if len(uniq) > 12:
            bins = np.quantile(fv, np.linspace(0, 1, 13))
            bins = np.unique(bins)
            centers, medians = [], []
            for lo, hi in zip(bins[:-1], bins[1:], strict=True):
                m = (fv >= lo) & (fv <= hi)
                if m.sum() >= 5:
                    centers.append(float(fv[m].mean()))
                    medians.append(float(np.median(sv[m])))
            ax.plot(centers, medians, color=_RED, linewidth=2)
        else:
            med = [(u, float(np.median(sv[fv == u]))) for u in uniq]
            ax.plot(
                [m[0] for m in med],
                [m[1] for m in med],
                color=_RED,
                linewidth=2,
                marker="o",
                markersize=4,
            )
        ax.axhline(0, color=_GRAY, linewidth=0.8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(k, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("SHAP dependence (red = binned median response)", y=1.0)
    fig.supylabel("SHAP value (ms)")
    fig.tight_layout()
    path = f"{out_prefix}_dependence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # --- 4. interaction pairs ------------------------------------------------------------
    if report.interaction_pairs:
        pairs = report.interaction_pairs[:15]
        labels = [f"{a} × {b}" for a, b, _ in pairs]
        vals = [v for _, _, v in pairs]
        fig, ax = plt.subplots(figsize=(8, 0.34 * len(pairs) + 1.2))
        ypos = np.arange(len(pairs))[::-1]
        ax.barh(ypos, vals, height=0.62, color=_BLUE, edgecolor="none")
        ax.set_yticks(ypos, labels, fontsize=9)
        ax.set_xlabel("mean |SHAP interaction| (ms)")
        ax.set_title("Top feature interactions")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e7e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        path = f"{out_prefix}_interactions.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written
