"""Frequency-weighted n-gram impact diff between two layouts.

For each corpus n-gram, the impact of switching layout A -> layout B is::

    impact_ms_mass = freq * (t_B(ngram) - t_A(ngram))

where t_X is the model-predicted time for the n-gram's position sequence under layout X.
The frequency weighting is the point: "the" getting 2ms slower can outweigh a rare
trigram getting 40ms slower — exactly the quantity the optimizer's objective sums.
The report ranks n-grams by |impact| and decomposes the total objective delta into its
top movers, so "why is B faster than A?" gets a concrete, per-n-gram answer.

Times come from the production tables: T2 (bigram position-pair table, calibration
applied from the model sidecar when present) for bigrams. Trigram time depends on what
the supplied trigram model predicts — its FRAME — and nothing in the model artifact
records that, so the caller must say (``trigram_frame``):

* ``"conditioned"`` — the model predicts the trigram INCREMENT t3 - t2 (the campaign's
  ``trigram_cond*`` models, trained on conditioned tristroke tables): t = T2 + Tcond,
  the production-objective construction.
* ``"absolute"`` — the model predicts the full trigram time (what ``keybo train
  --ngram trigram`` fits on raw tristroke TSVs): t = T3 alone. Adding T2 on top of an
  absolute model double-counts the first transition (measured 1.48x on the shipped
  models — audit finding F3), which is why the frame is explicit and unguessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.layout import Layout
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer


@dataclass
class NgramImpact:
    ngram: str
    freq: int
    t_a_ms: float
    t_b_ms: float
    #: freq * (t_b - t_a), in ms of corpus mass (negative = B faster on this n-gram)
    impact: float
    #: which keys moved: the n-gram's chars whose position differs between layouts
    moved_chars: str

    @property
    def delta_pct(self) -> float:
        """(t_B - t_A) / t_A, in percent — the n-gram's own relative change."""
        return 100.0 * (self.t_b_ms - self.t_a_ms) / self.t_a_ms if self.t_a_ms else 0.0


@dataclass
class LayoutDiff:
    layout_a: str
    layout_b: str
    ngram_len: int
    total_a: float
    total_b: float
    #: total_b - total_a in corpus-mass ms (negative = B faster overall)
    total_delta: float
    impacts: list[NgramImpact]
    #: share of total corpus mass covered by n-grams typeable on BOTH layouts.
    #: 1.0 when the charsets match; less when they differ (common-subset diff).
    corpus_coverage: float = 1.0

    def top(self, k: int = 20) -> list[NgramImpact]:
        return sorted(self.impacts, key=lambda x: -abs(x.impact))[:k]

    def to_dict(self, k: int = 20) -> dict:
        return {
            "layout_a": self.layout_a,
            "layout_b": self.layout_b,
            "ngram_len": self.ngram_len,
            "corpus_coverage": self.corpus_coverage,
            "total_a": self.total_a,
            "total_b": self.total_b,
            "total_delta": self.total_delta,
            "total_delta_pct_of_a": 100.0 * self.total_delta / self.total_a
            if self.total_a
            else None,
            "top_impacts": [
                {
                    "ngram": i.ngram,
                    "freq": i.freq,
                    "t_a_ms": i.t_a_ms,
                    "t_b_ms": i.t_b_ms,
                    "delta_ms": i.t_b_ms - i.t_a_ms,
                    "delta_pct": i.delta_pct,
                    "impact": i.impact,
                    "impact_pct_of_total_a": (
                        100.0 * i.impact / self.total_a if self.total_a else None
                    ),
                    "share_of_total_delta_pct": (
                        100.0 * i.impact / self.total_delta if self.total_delta else None
                    ),
                    "moved_chars": i.moved_chars,
                }
                for i in self.top(k)
            ],
        }


def _bigram_table(
    bigram_models: list[XGBoostTypingModel],
    bigram_freqs: dict[str, int],
    target_wpm: float,
    chars: str,
    geometry: Geometry,
) -> np.ndarray:
    """Mean calibrated T2 over the model ensemble (TableBigramScorer construction)."""
    tables = [
        TableBigramScorer(m, bigram_freqs, target_wpm=target_wpm, chars=chars, geometry=geometry)._T
        for m in bigram_models
    ]
    return np.mean(tables, axis=0)


def diff_layouts(
    layout_a: Layout,
    layout_b: Layout,
    bigram_models: list[XGBoostTypingModel],
    freqs: dict[str, int],
    trigram_models: list[XGBoostTypingModel] | None = None,
    bigram_freqs: dict[str, int] | None = None,
    target_wpm: float = 90.0,
    geometry: Geometry = ROW_STAGGERED_30,
    trigram_frame: str | None = None,
) -> LayoutDiff:
    """Per-n-gram impact of switching layout A -> B.

    ``freqs`` is the corpus table of the n-grams to diff (bigrams or trigrams — the
    length of its keys decides). Trigram diffs additionally need ``trigram_models``,
    ``bigram_freqs`` (to build T2) and an explicit ``trigram_frame`` — see the module
    docstring; a wrong frame mis-times every trigram, so there is no default.

    Charsets may differ (e.g. semimak carries an apostrophe where qwerty has a
    semicolon): the diff runs on the COMMON-subset corpus — n-grams typeable on both
    layouts — exactly the convention the campaign's cross-layout scoreboards use.
    ``corpus_coverage`` reports what share of corpus mass that subset keeps; totals
    are comparable to each other but not to a full-corpus objective when it is < 1.
    """
    chars = "".join(layout_a.chars)

    sample = next(iter(freqs))
    n = len(sample)
    if any(len(k) != n for k in freqs):
        raise ValueError("freqs table mixes n-gram lengths")

    positions = [*geometry.slots, geometry.space_position]
    pidx = {p: i for i, p in enumerate(positions)}

    T2 = _bigram_table(bigram_models, bigram_freqs or freqs, target_wpm, chars, geometry)

    if n == 3:
        if not trigram_models:
            raise ValueError("trigram diff needs trigram_models")
        if trigram_frame not in ("conditioned", "absolute"):
            raise ValueError(
                f"trigram_frame must be 'conditioned' or 'absolute', got {trigram_frame!r} "
                "(conditioned: model predicts t3-t2, time = T2 + model; absolute: model "
                "predicts full t3, time = model alone)"
            )
        vec = np.vstack(
            [
                trigram_features_from_positions(geometry, (a, b, c), wpm=target_wpm)
                for a in positions
                for b in positions
                for c in positions
            ]
        )
        n31 = len(positions)
        Tcond = np.mean([m.predict_ms(vec).reshape(n31, n31, n31) for m in trigram_models], axis=0)

    def pos_of(layout: Layout) -> dict[str, tuple]:
        d = {c: layout.pos(c) for c in layout.chars}
        d[" "] = geometry.space_position
        return d

    pa, pb = pos_of(layout_a), pos_of(layout_b)

    def time_of(ngram: str, pos_map: dict) -> float | None:
        if any(c not in pos_map for c in ngram):
            return None
        ps = [pos_map[c] for c in ngram]
        if n == 2:
            return float(T2[pidx[ps[0]], pidx[ps[1]]])
        t3 = float(Tcond[pidx[ps[0]], pidx[ps[1]], pidx[ps[2]]])
        if trigram_frame == "conditioned":
            t3 += float(T2[pidx[ps[0]], pidx[ps[1]]])
        return t3

    impacts: list[NgramImpact] = []
    total_a = total_b = 0.0
    mass_all = mass_common = 0.0
    for ngram, f in freqs.items():
        mass_all += f
        ta = time_of(ngram, pa)
        tb = time_of(ngram, pb)
        if ta is None or tb is None:
            continue
        mass_common += f
        total_a += f * ta
        total_b += f * tb
        moved = "".join(c for c in dict.fromkeys(ngram) if c != " " and pa.get(c) != pb.get(c))
        impacts.append(
            NgramImpact(
                ngram=ngram,
                freq=f,
                t_a_ms=ta,
                t_b_ms=tb,
                impact=f * (tb - ta),
                moved_chars=moved,
            )
        )

    return LayoutDiff(
        layout_a=chars if isinstance(chars, str) else str(chars),
        layout_b="".join(layout_b.chars),
        ngram_len=n,
        total_a=total_a,
        total_b=total_b,
        total_delta=total_b - total_a,
        impacts=impacts,
        corpus_coverage=mass_common / mass_all if mass_all else 1.0,
    )


def render_diff(diff: LayoutDiff, out_path: str, k: int = 20) -> str:
    """Tornado chart of the top-k n-gram impacts (negative = B faster)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = diff.top(k)
    labels = [
        f"{i.ngram.replace(' ', '␣')}  (f={i.freq:,}, {i.t_a_ms:.0f}→{i.t_b_ms:.0f}ms "
        f"{i.delta_pct:+.0f}%" + (f", moved: {i.moved_chars}" if i.moved_chars else "") + ")"
        for i in top
    ]
    vals = [100.0 * i.impact / diff.total_a if diff.total_a else i.impact for i in top]
    colors = ["#3987e5" if v < 0 else "#e34948" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(top) + 1.6))
    ypos = np.arange(len(top))[::-1]
    ax.barh(ypos, vals, height=0.62, color=colors, edgecolor="none")
    ax.axvline(0, color="#8a8988", linewidth=1)
    ax.set_yticks(ypos, labels, fontsize=8)
    ax.set_xlabel(
        "impact = freq × Δms, as % of A's total objective  (blue = B faster, red = B slower)"
    )
    pct = 100.0 * diff.total_delta / diff.total_a if diff.total_a else 0.0
    ax.set_title(
        f"Top {len(top)} {['', '', 'bigram', 'trigram'][diff.ngram_len]} impacts, "
        f"A→B total Δ {diff.total_delta:+.3g} ({pct:+.2f}%)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e7e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
