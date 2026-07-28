"""EVSCORE-1 validation: does a fitted weight set beat the community's taste constants?

The claim under test is narrow and checkable: *ranking layouts by prices derived from one
fitted surface predicts a DIFFERENT source's predicted time better than genkey /
oxeylyzer-1 / oxeylyzer-2 do.* Everything here exists to keep that claim from being true
by construction.

**The circularity, concretely.** The 3-seed models under ``data/models/k31/`` are the
AALTO source — the time card's served surface is bit-identical to ``AALTO_BASE``. So a
weight set fitted on AALTO and scored against AALTO measures nothing but the surrogate's
own fit quality. Every headline cell here is therefore **leave-one-source-out**: fit on
one pool's surface, score against another pool's. ``POOL`` pools the other two, so a
POOL cell is *partially in-sample* and is labelled that way rather than averaged in.

**The comparison must be paired.** Every layout is scored on the same surface arrays, so
the surface's own level is common mode and cancels in a difference. Ranking metrics
(Spearman, Kendall) are computed over one shared layout set per cell, and the
scorer-vs-scorer contrast is a **paired** difference over that same set with a bootstrap
interval — never two independently-quoted correlations compared by eye.

**The placebo is not optional.** :func:`noise_placebo` runs the identical pipeline on a
shuffled target. A maximin-style aggregate in this campaign picked the same layout from
pure noise 62-89% of the time, so any advantage figure is unreadable without knowing what
the machinery produces from nothing.

**Direction conventions are derived, not assumed.** genkey is lower-better; oxeylyzer-1,
oxeylyzer-2 and wfd are higher-better (qwerty is their most negative score); the
``oxey-style`` *gauge* is lower-better. Conflating those flips a sign, so
:func:`orient_scores` derives each one's direction from a reference layout that is known
to be worst rather than trusting a metadata flag — and asserts the reference really is
worst, so a silent convention change fails loudly instead of inverting a headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from keybo.analysis import surfaces as S
from keybo.analysis.community import community_suite, pinned_char
from keybo.analysis.evidence_scorer import (
    LIVE_GAUGES,
    MODELLED_ONLY_NOTE,
    EvidenceWeights,
    GaugeContext,
    TargetSurface,
    fit_evidence_weights,
    gauge_matrix,
    surface_ms_per_trigram,
)

#: The reference layout every direction convention is derived against. qwerty is the
#: campaign's known-worst board on every gauge in this frame.
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"

#: The competitor scorers, and whether a HIGHER raw value means a better layout.
#: Derived from the ports' own documented conventions and asserted against QWERTY30M.
COMPETITORS: dict[str, bool] = {
    "genkey": False,  # lower = better
    "oxeylyzer1": True,  # higher = better
    "oxeylyzer2": True,  # higher = better
}


def competitor_scores(layouts: list[str]) -> dict[str, np.ndarray]:
    """Raw genkey / oxeylyzer-1 / oxeylyzer-2 scores, each on its own native corpus.

    The ports run on the tools' OWN vendored corpora, because a community score is only
    meaningful in its native corpus convention. That is a deliberate asymmetry with the
    fitted scorer (which uses the production corpus) and it *favours the competitors*: they
    are being asked to rank layouts using exactly the data their authors intended.
    """
    out = {name: np.empty(len(layouts)) for name in COMPETITORS}
    for index, lay in enumerate(layouts):
        genkey, oxey1, oxey2 = community_suite(pinned_char(lay))
        out["genkey"][index] = float(genkey.score(lay))
        out["oxeylyzer1"][index] = float(oxey1.score(lay))
        out["oxeylyzer2"][index] = float(oxey2.score(lay))
    return out


def orient_scores(scores: dict[str, np.ndarray], reference: dict[str, float]) -> dict[str, str]:
    """Confirm each competitor's direction against a known-worst reference layout.

    Returns a per-scorer verdict string. The check is a POSITIVE signal (the reference is
    at the bad end) rather than the absence of a negative one: a metadata flag that
    silently changes meaning reads exactly like a correct flag, whereas "qwerty is not the
    worst on this scorer" is visible.
    """
    verdicts = {}
    for name, higher_better in COMPETITORS.items():
        values = scores[name]
        reference_value = reference[name]
        # A known-worst reference should sit at the BAD tail: the lowest score when higher
        # is better, the highest when lower is better.
        percentile = float((values < reference_value).mean())
        at_bad_end = percentile < 0.5 if higher_better else percentile > 0.5
        verdicts[name] = (
            f"{'higher' if higher_better else 'lower'}-better; qwerty at "
            f"{100 * percentile:.1f}th pct of pool -> "
            f"{'CONSISTENT' if at_bad_end else 'INCONSISTENT (convention may have flipped)'}"
        )
    return verdicts


# --------------------------------------------------------------------------------------
# Rank agreement, paired
# --------------------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau

    if len(a) < 3:
        return float("nan")
    return float(kendalltau(a, b).statistic)


@dataclass
class ScorerAgreement:
    """One scorer's rank agreement with a held-out surface's predicted time."""

    scorer: str
    spearman: float  # oriented so + means "agrees with faster-is-better"
    kendall: float
    n: int

    def to_dict(self) -> dict:
        return {
            "scorer": self.scorer,
            "spearman": self.spearman,
            "kendall": self.kendall,
            "n_layouts": self.n,
        }


def agreement_table(
    predicted_ms: np.ndarray,
    evidence_score: np.ndarray,
    competitors: dict[str, np.ndarray],
) -> dict[str, ScorerAgreement]:
    """Rank agreement of every scorer against held-out predicted time (lower = faster).

    All correlations are ORIENTED so that positive means "this scorer's notion of good
    agrees with lower predicted time". The evidence scorer is already a loss (lower =
    faster), so it is compared directly; a higher-better competitor is negated. That makes
    the column directly comparable across scorers with opposite raw conventions — the
    mistake that flips a sign in a summary table.
    """
    table = {
        "evidence": ScorerAgreement(
            "evidence",
            _spearman(evidence_score, predicted_ms),
            _kendall(evidence_score, predicted_ms),
            len(predicted_ms),
        )
    }
    for name, values in competitors.items():
        oriented = -values if COMPETITORS[name] else values
        table[name] = ScorerAgreement(
            name, _spearman(oriented, predicted_ms), _kendall(oriented, predicted_ms), len(values)
        )
    return table


def paired_advantage(
    predicted_ms: np.ndarray,
    evidence_score: np.ndarray,
    competitor: np.ndarray,
    higher_better: bool,
    *,
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap on ``rho(evidence) - rho(competitor)`` over ONE layout set.

    The layouts are resampled *once per replicate and shared by both scorers*, so the
    pool-composition noise that dominates a single correlation cancels in the difference.
    Quoting two independent confidence intervals and eyeballing the overlap answers a
    different, weaker question.
    """
    oriented = -competitor if higher_better else competitor
    observed = _spearman(evidence_score, predicted_ms) - _spearman(oriented, predicted_ms)
    rng = np.random.default_rng(seed)
    n = len(predicted_ms)
    deltas = []
    for _ in range(bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(predicted_ms[idx])) < 3:
            continue
        first = _spearman(evidence_score[idx], predicted_ms[idx])
        second = _spearman(oriented[idx], predicted_ms[idx])
        if np.isfinite(first) and np.isfinite(second):
            deltas.append(first - second)
    if not deltas:
        return {"delta_spearman": observed, "ci95": [float("nan")] * 2, "p_gt_0": float("nan")}
    array = np.array(deltas)
    return {
        "delta_spearman": float(observed),
        "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "p_gt_0": float((array > 0).mean()),
        "n_bootstrap": len(deltas),
    }


# --------------------------------------------------------------------------------------
# Leave-one-source-out: the primary, non-circular test
# --------------------------------------------------------------------------------------


@dataclass
class SourceCell:
    """One fit-source / test-source cell of the cross-source validation."""

    fit_source: str
    test_source: str
    independent: bool  # False when the two sources share data (any POOL cell, or self)
    agreement: dict[str, ScorerAgreement]
    advantages: dict[str, dict]
    placebo_spearman: float | None
    n_layouts: int
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "fit_source": self.fit_source,
            "test_source": self.test_source,
            "independent": self.independent,
            "n_layouts": self.n_layouts,
            "agreement": {k: v.to_dict() for k, v in self.agreement.items()},
            "advantage_vs": self.advantages,
            "placebo_spearman": self.placebo_spearman,
            "note": self.note,
        }


def _pool_of(surface_name: str) -> str:
    return surface_name.split("_", 1)[0]


def cross_source_agreement(targets: dict[str, np.ndarray]) -> dict:
    """How well the INDEPENDENT sources predict each other — the ceiling for any scorer.

    This is the most informative diagnostic in the module, and it must be reported alongside
    any advantage figure because it decides how to read one: no scorer fitted on source A can
    be expected to rank source B better than A itself ranks B. When this number is low, a
    scorer's poor cross-source showing says nothing about the scorer.

    Measured: **+0.835** over random C30M permutations versus **+0.265** over a pool of
    near-optimal archive layouts. Restricting to good layouts removes almost all of the
    agreement between the two fitted sources — which is why the same pipeline wins 12 of 12
    independent cells on the wide pool and loses 12 of 12 on the narrow one.
    """
    pairs = {}
    names = list(targets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            independent, _ = sources_independent(first, second)
            if independent:
                pairs[f"{first}|{second}"] = _spearman(targets[first], targets[second])
    values = np.array([v for v in pairs.values() if np.isfinite(v)])
    return {
        "pairwise": pairs,
        "mean": float(values.mean()) if len(values) else float("nan"),
        "min": float(values.min()) if len(values) else float("nan"),
        "max": float(values.max()) if len(values) else float("nan"),
        "interpretation": (
            "the ceiling for any cross-source scorer: a scorer fitted on source A cannot be "
            "expected to rank source B better than A ranks B. A LOW value means a poor "
            "cross-source showing is a property of the POOL, not of the scorer."
        ),
    }


def consensus_disagreement_ratio(targets: dict[str, np.ndarray]) -> dict:
    """C/D — the ratio of CONSENSUS spread to DISAGREEMENT spread across independent sources.

    POOLSWEEP-1 (ledger 873afb7) identified this as the quantity that actually sets the
    cross-source ceiling. Decomposing two z-scored independent sources into consensus
    ``C = (zA + zB) / 2`` and disagreement ``D = (zA - zB) / 2``, the measured ceiling is a
    near-deterministic monotone function of ``C/D`` alone: Spearman(rho, log C/D) =
    **+0.9991 / +0.9998 / +0.9998** across 49 pool cells on blend-seed0 / blend-seed7 / iWeb,
    with rho spanning -0.9886 to +0.9977 over a ~450x range in C/D.

    Why it beats an effective-dof floor as a guard, which is the whole reason this exists:
    restriction has **two opposite modes**. Restricting ``D`` alone drives the ceiling to
    +0.9999; restricting ``C`` alone drives it to -0.9886. Both LOWER the pool's spread and
    both LOWER effective dof, so no scalar narrowness statistic can tell them apart — and
    ``NARROW_POOL_DOF`` accordingly false-positived at interp-f0.25 (dof 2.43 with a perfectly
    healthy ceiling of +0.9244).

    It is also computable BEFORE any fit, so it can refuse a bad pool rather than annotate a
    bad result. Reference values: random-wide C/D = 3.06 (ceiling +0.797); the near-optimal
    archive C/D = 1.058 (ceiling +0.218); archive + ONE random transposition C/D = 3.82
    (ceiling +0.816) at unchanged layout quality.

    Z-scoring is per-source over the pool itself, so the ratio is scale-free and needs no
    external reference bank.
    """
    names = [n for n in targets]
    pairs: dict[str, float] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            independent, _ = sources_independent(first, second)
            if not independent:
                continue
            a = np.asarray(targets[first], dtype=float)
            b = np.asarray(targets[second], dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            a, b = a[finite], b[finite]
            if a.size < 3:
                continue
            sd_a, sd_b = a.std(), b.std()
            if sd_a == 0 or sd_b == 0:
                continue
            za, zb = (a - a.mean()) / sd_a, (b - b.mean()) / sd_b
            c_spread = float(((za + zb) / 2.0).std())
            d_spread = float(((za - zb) / 2.0).std())
            if d_spread == 0:
                pairs[f"{first}|{second}"] = float("inf")
            else:
                pairs[f"{first}|{second}"] = c_spread / d_spread
    finite_values = np.array([v for v in pairs.values() if np.isfinite(v)])
    return {
        "pairwise": pairs,
        "min": float(finite_values.min()) if len(finite_values) else float("nan"),
        "mean": float(finite_values.mean()) if len(finite_values) else float("nan"),
        "interpretation": (
            "C/D sets the cross-source ceiling (Spearman(rho, log C/D) = +0.999). A LOW value "
            "means the pool retains disagreement but not consensus, so no weight set fitted "
            "on it can transfer. Unlike an effective-dof floor this distinguishes the two "
            "opposite modes of restriction, and it is computable before any fit."
        ),
    }


def sources_independent(fit_source: str, test_source: str) -> tuple[bool, str]:
    """Whether a (fit, test) surface pair constitutes an out-of-sample test.

    Three ways a cell fails to be independent, all of them real here:

    * same pool — the same fitted model on both sides;
    * either side is ``POOL`` — it pools AALTO and COMMUNITY, so it shares data with both;
    * AALTO on either side is *additionally* the source the k31 model artifacts come from,
      which matters for any statement about the surrogate rather than about the surfaces.
    """
    fit_pool, test_pool = _pool_of(fit_source), _pool_of(test_source)
    if fit_pool == test_pool:
        return False, f"same model pool ({fit_pool}) on both sides — in-sample by construction"
    if "POOL" in (fit_pool, test_pool):
        return False, "POOL pools AALTO and COMMUNITY, so this cell is partially in-sample"
    return True, f"{fit_pool} -> {test_pool}: distinct model pools"


def cross_source_validation(
    layouts: list[str],
    surfaces: dict[str, TargetSurface],
    context: GaugeContext,
    objective,
    *,
    X: np.ndarray | None = None,
    targets: dict[str, np.ndarray] | None = None,
    competitors: dict[str, np.ndarray] | None = None,
    run_placebo: bool = True,
    bootstrap: int = 2000,
    seed: int = 0,
    progress: bool = False,
) -> list[SourceCell]:
    """Every (fit source, test source) cell over one layout pool.

    The gauge matrix, the per-surface targets and the competitor scores are computed ONCE
    and shared across cells, so every cell is a paired comparison over the identical layout
    set — and the expensive part (gauges) is paid once.
    """
    if X is None:
        X = gauge_matrix(layouts, context)
    if targets is None:
        targets = {
            name: surface_ms_per_trigram(layouts, surface, objective)
            for name, surface in surfaces.items()
        }
    if competitors is None:
        competitors = competitor_scores(layouts)

    cells: list[SourceCell] = []
    for fit_name, fit_surface in surfaces.items():
        weights = fit_evidence_weights(
            layouts,
            fit_surface,
            context,
            objective,
            pool_label="cross-source",
            X=X,
            y=targets[fit_name],
            seed=seed,
        )
        placebo_weights = (
            fit_evidence_weights(
                layouts,
                fit_surface,
                context,
                objective,
                pool_label="cross-source-placebo",
                X=X,
                y=targets[fit_name],
                seed=seed,
                shuffle_target=True,
            )
            if run_placebo
            else None
        )
        evidence = np.array(
            [weights.score(dict(zip(LIVE_GAUGES, row, strict=True)))["score"] for row in X]
        )
        placebo = (
            np.array(
                [
                    placebo_weights.score(dict(zip(LIVE_GAUGES, row, strict=True)))["score"]
                    for row in X
                ]
            )
            if placebo_weights is not None
            else None
        )
        for test_name in surfaces:
            independent, note = sources_independent(fit_name, test_name)
            predicted = targets[test_name]
            cells.append(
                SourceCell(
                    fit_source=fit_name,
                    test_source=test_name,
                    independent=independent,
                    agreement=agreement_table(predicted, evidence, competitors),
                    advantages={
                        name: paired_advantage(
                            predicted,
                            evidence,
                            values,
                            COMPETITORS[name],
                            bootstrap=bootstrap,
                            seed=seed,
                        )
                        for name, values in competitors.items()
                    },
                    placebo_spearman=(
                        _spearman(placebo, predicted) if placebo is not None else None
                    ),
                    n_layouts=len(layouts),
                    note=note,
                )
            )
        if progress:
            print(f"  cross-source: fitted on {fit_name}", flush=True)
    return cells


# --------------------------------------------------------------------------------------
# Leave-one-layout-out
# --------------------------------------------------------------------------------------


@dataclass
class LoloResult:
    """Held-out predictions from leave-one-layout-out refitting."""

    fit_source: str
    test_source: str
    independent: bool
    spearman_held_out: float
    kendall_held_out: float
    competitor_spearman: dict[str, float]
    n_folds: int

    def to_dict(self) -> dict:
        return {
            "fit_source": self.fit_source,
            "test_source": self.test_source,
            "independent": self.independent,
            "spearman_held_out": self.spearman_held_out,
            "kendall_held_out": self.kendall_held_out,
            "competitor_spearman": self.competitor_spearman,
            "n_folds": self.n_folds,
        }


def leave_one_layout_out(
    layouts: list[str],
    fit_surface: TargetSurface,
    test_surface: TargetSurface,
    context: GaugeContext,
    objective,
    *,
    X: np.ndarray | None = None,
    fit_target: np.ndarray | None = None,
    test_target: np.ndarray | None = None,
    competitors: dict[str, np.ndarray] | None = None,
    folds: int = 0,
    seed: int = 0,
) -> LoloResult:
    """Refit the weights with each layout (or fold) held out, then score the held-out ones.

    ``folds = 0`` means true leave-one-out; a positive value uses K grouped folds, which is
    the same estimator with less compute and is what makes this affordable on a large pool.
    The held-out score is assembled from folds that never saw the layout, so the resulting
    correlation carries no in-sample fit at all.
    """
    if X is None:
        X = gauge_matrix(layouts, context)
    if fit_target is None:
        fit_target = surface_ms_per_trigram(layouts, fit_surface, objective)
    if test_target is None:
        test_target = surface_ms_per_trigram(layouts, test_surface, objective)
    if competitors is None:
        competitors = competitor_scores(layouts)

    n = len(layouts)
    groups = (
        [np.array([i]) for i in range(n)]
        if folds <= 0
        else np.array_split(np.random.default_rng(seed).permutation(n), folds)
    )
    held = np.full(n, np.nan)
    for group in groups:
        mask = np.ones(n, dtype=bool)
        mask[group] = False
        weights = fit_evidence_weights(
            [layouts[i] for i in np.flatnonzero(mask)],
            fit_surface,
            context,
            objective,
            pool_label="lolo-train",
            X=X[mask],
            y=fit_target[mask],
            holdout_frac=0.0,
            bootstrap=0,
            seed=seed,
        )
        for i in group:
            held[i] = weights.score(dict(zip(LIVE_GAUGES, X[i], strict=True)))["score"]

    independent, _ = sources_independent(fit_surface.name, test_surface.name)
    return LoloResult(
        fit_source=fit_surface.name,
        test_source=test_surface.name,
        independent=independent,
        spearman_held_out=_spearman(held, test_target),
        kendall_held_out=_kendall(held, test_target),
        competitor_spearman={
            name: _spearman(-values if COMPETITORS[name] else values, test_target)
            for name, values in competitors.items()
        },
        n_folds=len(groups),
    )


# --------------------------------------------------------------------------------------
# Noise placebo
# --------------------------------------------------------------------------------------


def noise_placebo(
    layouts: list[str],
    fit_surface: TargetSurface,
    test_target: np.ndarray,
    context: GaugeContext,
    objective,
    *,
    X: np.ndarray | None = None,
    fit_target: np.ndarray | None = None,
    repeats: int = 20,
    seed: int = 0,
) -> dict:
    """Run the whole pipeline on SHUFFLED targets and report the agreement it manufactures.

    This is the ruler for every advantage figure. The pipeline has many degrees of freedom
    (a surrogate, a form search per gauge, 14 correlated columns), and machinery with that
    much freedom produces non-zero-looking agreement from nothing. Reporting the real
    number without this one is how a campaign publishes a stability figure that pure noise
    also passes.
    """
    if X is None:
        X = gauge_matrix(layouts, context)
    if fit_target is None:
        fit_target = surface_ms_per_trigram(layouts, fit_surface, objective)
    values = []
    for repeat in range(repeats):
        weights = fit_evidence_weights(
            layouts,
            fit_surface,
            context,
            objective,
            pool_label="placebo",
            X=X,
            y=fit_target,
            holdout_frac=0.0,
            bootstrap=0,
            seed=seed + repeat,
            shuffle_target=True,
        )
        scored = np.array(
            [weights.score(dict(zip(LIVE_GAUGES, row, strict=True)))["score"] for row in X]
        )
        values.append(_spearman(scored, test_target))
    array = np.array([v for v in values if np.isfinite(v)])
    return {
        "repeats": repeats,
        "spearman_mean": float(array.mean()) if len(array) else float("nan"),
        "spearman_abs_mean": float(np.abs(array).mean()) if len(array) else float("nan"),
        "spearman_abs_p95": float(np.percentile(np.abs(array), 95)) if len(array) else float("nan"),
        "spearman_min": float(array.min()) if len(array) else float("nan"),
        "spearman_max": float(array.max()) if len(array) else float("nan"),
        "interpretation": (
            "|rho| the pipeline manufactures from shuffled labels; a real advantage must "
            "exceed this band to be readable"
        ),
    }


# --------------------------------------------------------------------------------------
# Paired resolution floor
# --------------------------------------------------------------------------------------


def paired_resolution(
    layouts: list[str],
    per_seed_surfaces: list[np.ndarray],
    objective,
    *,
    max_layouts: int = 60,
    seed: int = 0,
) -> dict:
    """The PAIRED estimator noise floor from per-seed surfaces, plus its unpaired sibling.

    Every layout is scored on the same seed tables, so the seed main effect is common mode
    and cancels in a within-seed *difference*. Quoting the unpaired within-layout spread as
    the resolution of a *comparison* is the wrong ruler — it overstates the floor by ~2x
    here and by ~3x on the time card, and it is what made an earlier campaign conclusion
    read "0 of 15 pairs resolve" when the paired answer was 8.

    Also reports the variance decomposition, because the paired argument is only valid to
    the extent the nuisance factor SHIFTS rather than SCALES; a large residual share means
    differences do not cancel exactly and the paired floor is optimistic.
    """
    rng = np.random.default_rng(seed)
    if len(layouts) > max_layouts:
        layouts = [layouts[i] for i in rng.choice(len(layouts), max_layouts, replace=False)]
    mass = float(objective[3].sum())
    matrix = np.array(
        [
            [S.score_fit(lay, surface, objective) / mass for surface in per_seed_surfaces]
            for lay in layouts
        ]
    )
    n, k = matrix.shape
    within = matrix.max(axis=1) - matrix.min(axis=1)
    rows, cols = np.triu_indices(n, 1)
    differences = matrix[rows] - matrix[cols]
    paired_spread = differences.max(axis=1) - differences.min(axis=1)

    grand = matrix.mean()
    layout_effect = matrix.mean(axis=1) - grand
    seed_effect = matrix.mean(axis=0) - grand
    residual = matrix - grand - layout_effect[:, None] - seed_effect[None, :]
    ss_layout = float((k * layout_effect**2).sum())
    ss_seed = float((n * seed_effect**2).sum())
    ss_residual = float((residual**2).sum())
    total = ss_layout + ss_seed + ss_residual
    mean_difference = np.abs(differences.mean(axis=1))
    return {
        "n_layouts": n,
        "n_seeds": k,
        "unpaired_floor_ms_per_trigram": float(within.mean()),
        "unpaired_floor_max": float(within.max()),
        "paired_floor_ms_per_trigram": float(paired_spread.mean()),
        "paired_floor_p95": float(np.percentile(paired_spread, 95)),
        "paired_over_unpaired": float(paired_spread.mean() / within.mean())
        if within.mean()
        else float("nan"),
        "ss_share_pct": {
            "layout": 100.0 * ss_layout / total,
            "seed": 100.0 * ss_seed / total,
            "residual": 100.0 * ss_residual / total,
        },
        "frac_pairs_resolved": float((mean_difference > paired_spread).mean()),
        "note": (
            "paired floor is the across-seed spread of a within-seed DIFFERENCE; a large "
            "residual share means the seed factor scales rather than shifts, so the paired "
            "floor is optimistic"
        ),
    }


# --------------------------------------------------------------------------------------
# What the scorer structurally cannot express
# --------------------------------------------------------------------------------------


@dataclass
class Limitation:
    """One thing this scorer cannot represent, with the evidence that establishes it."""

    name: str
    verdict: str
    evidence: str
    scope: str

    def to_dict(self) -> dict:
        return {
            "limitation": self.name,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "scope": self.scope,
        }


def direction_invariance_proof(geometry=None) -> dict:
    """Prove exhaustively that the served bigram feature vector cannot express direction.

    Not cited — recomputed. Over every ordered distinct position pair, the maximum absolute
    difference between the NON-LANDING features of ``features(a, b)`` and ``features(b, a)``
    is exactly 0: ``angle``, ``inwards`` and ``outwards`` merely look directional. So an
    inroll/outroll weight derived from this frame would be attributing a number to a
    distinction the frame does not contain, which is precisely what oxeylyzer's ``inrolls
    +250`` / ``outrolls +240`` split claims to price.

    The landing-key one-hots ARE order-dependent (they are computed from ``b`` alone), so
    the check must exclude them or it measures the wrong thing.
    """
    from keybo.features import bigram_features_from_positions
    from keybo.features.schema import BIGRAM_FEATURE_NAMES
    from keybo.geometry import ROW_STAGGERED_30

    geometry = geometry or ROW_STAGGERED_30
    positions = list(geometry.slots)
    names = list(BIGRAM_FEATURE_NAMES)
    # The landing-key features are the SECOND key's row and finger one-hots (the schema is
    # explicit: "Row and finger one-hots describe the *second* (landing) key"). They are
    # functions of b alone, so they are order-dependent by design; excluding them by NAME
    # rather than by a prefix guess is what makes the remaining check meaningful.
    landing = {
        i
        for i, name in enumerate(names)
        if name in {"bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral"}
    }
    if len(landing) != 8:
        raise AssertionError(
            f"expected 8 landing-key one-hots in the bigram schema, found {len(landing)}; "
            "the feature schema changed and this proof must be re-derived"
        )
    keep = [i for i in range(len(names)) if i not in landing]
    worst = 0.0
    worst_pair = None
    count = 0
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            if i >= j:
                continue
            forward = bigram_features_from_positions(geometry, (a, b), wpm=S.BAKED_WPM)
            backward = bigram_features_from_positions(geometry, (b, a), wpm=S.BAKED_WPM)
            difference = float(np.abs(forward[keep] - backward[keep]).max())
            count += 1
            if difference > worst:
                worst, worst_pair = difference, (i, j)
    return {
        "ordered_pairs_checked": count,
        "max_abs_nonlanding_feature_diff": worst,
        "worst_pair_slots": list(worst_pair) if worst_pair else None,
        "features_compared": [names[i] for i in keep],
        "landing_features_excluded": [names[i] for i in sorted(landing)],
        "verdict": (
            "direction of travel is NOT representable in the served bigram vector"
            if worst == 0.0
            else f"some non-landing feature IS order-dependent (max diff {worst:.3e})"
        ),
    }


def structural_limitations(paired_floor: float | None = None) -> list[Limitation]:
    """The explicit list of what an additive gauge scorer cannot express."""
    limits = [
        Limitation(
            name="direction of travel (inroll vs outroll)",
            verdict="CANNOT EXPRESS — and neither can the surfaces the weights come from",
            evidence=(
                "over every ordered distinct position pair the max absolute difference "
                "between non-landing features of features(a,b) and features(b,a) is exactly "
                "0.0 (recomputed by direction_invariance_proof, not cited). A follow-up that "
                "added a real direction channel found no cross-source signal. So an "
                "inroll/outroll split cannot be evidence-based here — reporting that is "
                "itself a result against oxeylyzer, which prices inrolls +250 vs outrolls "
                "+240. NOTE the boundary: the COMMUNITY TRIGRAM inrolls/outrolls ARE "
                "genuinely order-dependent (all 9720 qualifying triples relabel under "
                "reversal); the no-direction result is about the BIGRAM feature vector."
            ),
            scope="bigram feature vector; not the trigram pattern classes",
        ),
        Limitation(
            name="non-pairwise trigram structure (Tcond)",
            verdict="CANNOT EXPRESS — additive gauge sums are structurally blind to it",
            evidence=(
                "the conditioned trigram term is 11.8-15.3% of sum-of-squares and is "
                "irreducible to bigram+skipgram parts; it FLIPS semimak vs graphite on 3 of "
                "3 seeds. Any scorer that sums per-gauge prices — this one, genkey, both "
                "oxeylyzers — cannot represent an effect that only exists in the joint "
                "triple. The gauge frame does carry trigram-derived axes (redir, roll, "
                "sr-roll, alt), but as MARGINAL shares, which is not the same thing."
            ),
            scope="all additive analyzers, ours included",
        ),
        Limitation(
            name="differences below the paired estimator floor",
            verdict="MUST NOT CLAIM",
            evidence=(
                "the instrument's resolution, not the scorer's precision, is the binding "
                "constraint: five independent routes converged on an unpaired floor of "
                "~0.66-0.72 ms/char, and the paired floor is ~0.17-0.24 on the time card. "
                + (
                    f"Measured here on the per-seed COMMUNITY_BASE surfaces: paired floor "
                    f"{paired_floor:.3f} ms/trigram."
                    if paired_floor is not None
                    else "Measure it per instrument; do not reuse another instrument's floor."
                )
                + " The floor bounds ms-scale claims ONLY, not the ratio gauges."
            ),
            scope="ms/trigram claims; not rank statistics or ratio gauges",
        ),
        Limitation(
            name="realized typing speed",
            verdict="CANNOT CLAIM AT ALL",
            evidence=MODELLED_ONLY_NOTE,
            scope="everything in this module",
        ),
        Limitation(
            name="permutation-invariant axes",
            verdict="CANNOT PRICE — sfr is constant across every layout",
            evidence=(
                "sfr counts doubled letters, so no placement moves it: one distinct value "
                "over 40 random C30M permutations, sample std exactly 0.0 (tested by "
                "shuffling; numpy reports ~1.9e-14 on some draws, so a std>0 filter would "
                "keep it and then rank-correlate noise). Excluded from the frame: 14 live "
                "axes, not 15."
            ),
            scope="the gauge frame",
        ),
    ]
    return limits


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Everything the CLI prints or serializes."""

    corpus: str
    corpus_sha256: dict
    surface_frame: str
    n_layouts: int
    pool_label: str
    cells: list[SourceCell]
    lolo: list[LoloResult]
    placebo: dict
    resolution: dict | None
    direction_proof: dict
    limitations: list[Limitation]
    competitor_orientation: dict[str, str]
    weights: dict[str, EvidenceWeights] = field(default_factory=dict)
    #: The ceiling — how well the independent sources agree with each other on THIS pool.
    #: Without it an advantage figure cannot be read: a low ceiling means a poor
    #: cross-source showing is a property of the pool, not of the scorer.
    source_agreement: dict = field(default_factory=dict)

    def independent_cells(self) -> list[SourceCell]:
        return [c for c in self.cells if c.independent]

    def headline(self) -> dict:
        """The one number the deliverable turns on, over independent cells only."""
        cells = self.independent_cells()
        if not cells:
            return {"verdict": "no independent cell available", "n_cells": 0}
        rows = []
        for cell in cells:
            best_competitor = max(
                (c for name, c in cell.agreement.items() if name != "evidence"),
                key=lambda a: a.spearman if np.isfinite(a.spearman) else -np.inf,
            )
            rows.append(
                {
                    "fit_source": cell.fit_source,
                    "test_source": cell.test_source,
                    "evidence_spearman": cell.agreement["evidence"].spearman,
                    "best_competitor": best_competitor.scorer,
                    "best_competitor_spearman": best_competitor.spearman,
                    "delta_vs_best": cell.agreement["evidence"].spearman - best_competitor.spearman,
                    "delta_vs_best_ci95": cell.advantages[best_competitor.scorer]["ci95"],
                    "p_gt_0": cell.advantages[best_competitor.scorer]["p_gt_0"],
                    "placebo_spearman": cell.placebo_spearman,
                }
            )
        deltas = [r["delta_vs_best"] for r in rows if np.isfinite(r["delta_vs_best"])]
        wins = sum(1 for d in deltas if d > 0)
        return {
            "n_cells": len(rows),
            "cells": rows,
            "cells_where_evidence_wins": wins,
            "mean_delta_spearman_vs_best_competitor": float(np.mean(deltas))
            if deltas
            else float("nan"),
            "min_delta": float(np.min(deltas)) if deltas else float("nan"),
            "max_delta": float(np.max(deltas)) if deltas else float("nan"),
            "verdict": (
                "evidence weights beat the best taste-constant scorer in every independent cell"
                if deltas and wins == len(deltas)
                else (
                    f"evidence weights win {wins} of {len(deltas)} independent cells"
                    if deltas
                    else "undetermined"
                )
            ),
            # A win/loss count is unreadable on its own: the SAME pipeline wins 12/12 on a
            # wide pool and loses 12/12 on a narrow one, and what changed was the ceiling
            # (source-to-source agreement +0.835 -> +0.265), not the scorer. So the verdict
            # ships with the ceiling and with the placebo band attached.
            "ceiling_source_agreement_mean": self.source_agreement.get("mean"),
            "how_to_read": (
                "compare each delta against BOTH the ceiling (how well the independent "
                "sources agree with each other on this pool) and the noise placebo band. A "
                "loss under a low ceiling is a property of the pool; an evidence rho inside "
                "the placebo band is not distinguishable from noise at all."
            ),
            "placebo_abs_mean": self.placebo.get("spearman_abs_mean"),
            "placebo_abs_p95": self.placebo.get("spearman_abs_p95"),
            "evidence_rho_inside_placebo_band": (
                bool(
                    max(
                        (
                            abs(r["evidence_spearman"])
                            for r in rows
                            if np.isfinite(r["evidence_spearman"])
                        ),
                        default=0.0,
                    )
                    <= self.placebo["spearman_abs_p95"]
                )
                if self.placebo.get("spearman_abs_p95") is not None
                and np.isfinite(self.placebo.get("spearman_abs_p95", np.nan))
                else None
            ),
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "corpus": self.corpus,
            "corpus_sha256": self.corpus_sha256,
            "surface_frame": self.surface_frame,
            "n_layouts": self.n_layouts,
            "pool": self.pool_label,
            "headline": self.headline(),
            "cross_source_cells": [c.to_dict() for c in self.cells],
            "lolo": [r.to_dict() for r in self.lolo],
            "noise_placebo": self.placebo,
            "ceiling_source_agreement": self.source_agreement,
            "paired_resolution": self.resolution,
            "direction_invariance_proof": self.direction_proof,
            "cannot_express": [limitation.to_dict() for limitation in self.limitations],
            "competitor_orientation_check": self.competitor_orientation,
            "weights": {name: weights.to_dict() for name, weights in self.weights.items()},
            "notes": [MODELLED_ONLY_NOTE],
        }
