"""EVSCORE-1: a layout scorer whose weights and loss curves are FITTED, not chosen.

genkey and oxeylyzer score a layout by summing hand-chosen constants over pattern
counts — oxeylyzer-1 pays ``inrolls +250``, ``onehands +90``, ``redirects -340`` and so
on (:class:`keybo.analysis.community.Oxeylyzer1`, ``WT``). Those numbers are taste. This
module derives the same *shape* of scorer — a price per gauge — from the campaign's
fitted timing surfaces, by SHAP-attributing a surface's layout-level behaviour to the
gauge frame and reading each gauge's price off its own attribution.

The pipeline, in one line per stage:

1. **Gauge matrix.** Each layout in a pool becomes a vector over the 14 *live* axes of
   the campaign's 15-gauge frame (:data:`LIVE_GAUGES`). ``sfr`` is dropped because it is
   a permutation invariant — one distinct value over 40 random C30M permutations, sample
   standard deviation exactly 0.0 (tested by shuffling, never by a variance threshold:
   numpy reports its std as ~1.9e-14 on some draws, so a ``std > 0`` filter keeps it and
   then rank-correlates pure noise).
2. **Target.** The layout's corpus-weighted fit on one fitted surface, in ms per
   trigram — the campaign's QAP objective on the served **geometry-only** ``g`` frame
   (:mod:`keybo.analysis.surfaces`). Not ``g + b``.
3. **Surrogate + TreeSHAP.** An XGBoost surrogate maps the gauge vector to the target;
   :func:`keybo.analysis.shap_report.compute_shap` (exact TreeSHAP, additivity asserted)
   attributes each layout's predicted time across the gauges. Shapley is the right tool
   *because* the gauges are heavily correlated: it distributes credit among correlated
   features by construction, where a marginal regression coefficient would let two
   restatements of one fact split or steal each other's price.
4. **Loss curve.** Per gauge, the SHAP value as a function of the gauge's own level —
   fitted as a function (linear / quadratic / hinge), model-selected, with a bootstrap
   band and a **stated valid domain**. The scalar "weight" a community analyzer would
   publish is this curve's linearization; both are reported, because a weight that
   pretends a saturating cost is linear is wrong outside the band it was fitted in.

Four things this module refuses to do, each because doing them has already produced a
wrong number in this campaign:

**It will not validate a weight against the surface that produced it.** The 3-seed
models under ``data/models/k31/`` *are* the AALTO source: the time card's served surface
``T2[a,b] + Tcond[a,b,c]`` is bit-identical to ``AALTO_BASE`` (max absolute difference
over all 31^3 cells exactly 0.0). So "fit on the models, score against AALTO" is not a
test. Every headline number here is **cross-source**: weights fitted on one source's
surface, scored against a different source's.

**It uses the NATIVE surfaces, not the standardized ones.** The vendored
``data/surfaces/`` holds ``<NAME>.standardized.npy.gz``, and "standardized" means *the
production AALTO seed-mean bigram tensor is substituted into all eight surfaces* so that
only the trigram model differs. Verified mechanically: ``standardized - native`` has
variance ~3e-27 along the third axis (i.e. it is a pure bigram-tensor swap), and for
``COMMUNITY_BASE`` — the one surface with per-seed parts —
``max|standardized - (T2_aalto + conditional_own)| = 1.1e-13``. A cross-source claim
built on standardized surfaces therefore shares the AALTO bigram tensor with the source
it is being tested against, which is the same circularity one layer down. The
standardized frame is still reported, labelled, as a sensitivity: it is not
rank-equivalent (Spearman between native and standardized cells is 0.93, not 1.0).

**It states the domain each curve is valid over, and flags a score outside it.** The
layouts a curve is fitted on occupy a narrow band of every gauge; qwerty sits outside
several of them. A price read off a curve outside its fitted band is extrapolation, and
:meth:`EvidenceWeights.score` marks it rather than quietly returning a number.

**It reports per-CLUSTER attribution beside per-gauge.** Effective degrees of freedom
over this frame is ~4-5, not 14: ``lsb`` and ``lsb-dist`` correlate at rho 1.00,
``sr-roll`` is a strict subset of ``roll``, and ``oxey-style`` is R^2 = 0.9937 on
{sfb, lsb, scissor, imbalance, redir, alt} — a restatement, not a corroboration. Summing
14 per-gauge prices as if they were 14 independent facts over-counts; the cluster view is
what a reader should use to compare against a competitor's axis count.

MODELLED ONLY. Every number here is a prediction of a fitted surface, not a measurement
of realized typing. The held-layout transfer statistic is saturated and the human
validation phase was cancelled, so nothing in this module is a claim about how fast
anyone actually types. :data:`MODELLED_ONLY_NOTE` says so in the tool's own output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from keybo.analysis import surfaces as S
from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.comfort import ComfortBigramScorer
from keybo.scoring.oxey import OxeyStyleScorer

# --------------------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------------------

#: The full 15-gauge corpus-sensitive frame, in the frozen board's order.
ALL_GAUGES: tuple[str, ...] = (*STAT_NAMES, "scissor", "imbalance", "oxey-style", "comfort")

#: ``sfr`` counts doubled letters, so no placement can move it: it is a PERMUTATION
#: INVARIANT and carries exactly one value across every layout. Excluded — a constant
#: column would be handed a share of the attribution it cannot have earned.
INVARIANT_GAUGES: tuple[str, ...] = ("sfr",)

#: The 14 axes a layout can actually move.
LIVE_GAUGES: tuple[str, ...] = tuple(g for g in ALL_GAUGES if g not in INVARIANT_GAUGES)

#: The frame the target lives on, spelled out wherever a number is printed.
FRAME_NOTE = S.FRAME_NOTE

#: Stated in the tool's own output, not only in the write-up.
MODELLED_ONLY_NOTE = (
    "MODELLED ONLY: every price here is attribution of a FITTED timing surface, not a "
    "measurement of realized typing speed. Held-layout transfer is saturated and the "
    "human-validation phase was cancelled, so no number here predicts how fast anyone types."
)

#: The sign each gauge's price SHOULD carry if the mechanism is what the community says it
#: is: same-finger reuse, lateral stretch, scissors, redirects and hand imbalance cost time;
#: alternation and rolls save it. Used only to AUDIT a fitted weight (:meth:`sign_audit`) —
#: never to constrain the fit, because overturning one of these priors is a legitimate
#: result and this campaign has already measured two real community mispricings.
#: ``comfort`` and ``oxey-style`` are composite penalty scores, so higher = worse for both.
EXPECTED_SIGN: dict[str, float] = {
    "sfb": +1.0,
    "sfs": +1.0,
    "sfb-dist": +1.0,
    "sfs-dist": +1.0,
    "lsb": +1.0,
    "lsb-dist": +1.0,
    "scissor": +1.0,
    "redir": +1.0,
    "imbalance": +1.0,
    "alt": -1.0,
    "roll": -1.0,
    "sr-roll": -1.0,
    "comfort": +1.0,
    "oxey-style": +1.0,
}

#: ⚠ DEPRECATED AS A GUARD — retained for diagnostics only, never to gate a verdict.
#: It was calibrated on the SAME archive-vs-random contrast it was then used to detect
#: (archive 3.99 / random 5.03), which is circular, and POOLSWEEP-1 (ledger 873afb7) measured
#: it FALSE-POSITIVING at interp-f0.25: effective dof 2.43 with a perfectly healthy
#: cross-source ceiling of +0.9244. Root cause: restriction has TWO OPPOSITE MODES (removing
#: consensus vs removing disagreement) and both lower effective dof, so no scalar narrowness
#: statistic can tell a fatal pool from a fine one. Use NARROW_POOL_CD instead.
NARROW_POOL_DOF = 4.5

#: C/D floor — the PRIMARY pool guard, replacing NARROW_POOL_DOF. POOLSWEEP-1 identified the
#: consensus/disagreement ratio as the quantity that actually sets the cross-source ceiling
#: (Spearman(rho, log C/D) = +0.999 over 49 cells x 3 corpora). Measured anchors: the
#: near-optimal archive sits at C/D 1.058 with ceiling +0.218 and loses 12/12 cross-source
#: cells; random-wide sits at 3.06 with ceiling +0.797 and wins 12/12; archive + ONE random
#: transposition sits at 3.82 with ceiling +0.816 at unchanged layout quality. The floor is
#: placed at 2.0 — above every failing pool measured and below every passing one — and is
#: deliberately closer to the failing end. Unlike the dof floor this is computable BEFORE any
#: fit, so it can refuse a bad pool rather than annotate a bad result.
NARROW_POOL_CD = 2.0

#: Cross-source agreement below which a weight set must not be used for ranking at all.
#: Measured: +0.835 on the wide pool (weights transfer, 12/12 wins) vs +0.265 on the narrow
#: one (weights do not, 0/12). The midpoint is deliberately closer to the failing end.
NARROW_POOL_SOURCE_AGREEMENT = 0.5

#: Why each surface family/pool may or may not be used to validate the other.
SOURCE_INDEPENDENCE_NOTE = (
    "AALTO is NOT independent of data/models/k31: the time card's served surface "
    "T2[a,b]+Tcond[a,b,c] is bit-identical to AALTO_BASE (max abs diff 0.0 over 31^3 "
    "cells). POOL is not independent of AALTO+COMMUNITY either (it pools them). So the "
    "strongest available test is fit-on-COMMUNITY / score-on-AALTO and its reverse, and "
    "any POOL cell must be read as partially in-sample."
)


# --------------------------------------------------------------------------------------
# Gauge extraction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GaugeContext:
    """Everything needed to turn a layout string into a gauge vector, on ONE corpus.

    Built once and reused: the scorers cache per-corpus tables, and rebuilding them per
    layout dominates the runtime.
    """

    corpus: str | None
    kmstats: KmStats
    oxey: OxeyStyleScorer
    comfort: ComfortBigramScorer
    bigram_mass: float
    identity: dict

    @property
    def corpus_name(self) -> str:
        return str(self.identity.get("corpus", self.corpus or "unknown"))

    def vector(self, lay30: str) -> dict[str, float]:
        """The 14 live gauges for one layout, on this context's corpus.

        Conventions are the analyzer's, not re-derived: the keymeow-class statistics come
        from :class:`KmStats` (space-excluded denominator), ``scissor``/``imbalance`` from
        ``oxey.pattern_shares`` (layout-restricted bigram mass), and ``comfort`` is the
        absolute ms-equivalent sum over the FULL corpus bigram mass — a different
        denominator from every other gauge here, which is stated rather than hidden.
        """
        layout = Layout(lay30, ROW_STAGGERED_30)
        gauges = dict(self.kmstats.stats(lay30))
        shares = self.oxey.pattern_shares(layout)
        gauges["scissor"] = shares["scissor"]
        gauges["imbalance"] = shares["imbalance"]
        gauges["oxey-style"] = self.oxey.fitness(layout)
        gauges["comfort"] = self.comfort.fitness(layout) / self.bigram_mass
        return {name: float(gauges[name]) for name in LIVE_GAUGES}


@lru_cache(maxsize=4)
def gauge_context(corpus: str | None = None) -> GaugeContext:
    """A :class:`GaugeContext` over one corpus (cached; table loading is the slow part)."""
    from keybo.data.corpus import corpus_identity, load_frequencies, production_corpus_dir

    directory = production_corpus_dir(corpus)
    bigrams = load_frequencies(str(directory / "bigrams.txt"))
    # 1-skip31.txt IS the trigram marginalization and is the table every frozen campaign
    # board was computed on; 1-skip.txt is a different, unreproducible pass.
    skipgrams = load_frequencies(str(directory / "1-skip31.txt"))
    trigrams = load_frequencies(str(directory / "trigrams.txt"))
    return GaugeContext(
        corpus=corpus,
        kmstats=KmStats(bigrams, skipgrams, trigrams),
        oxey=OxeyStyleScorer(bigrams, skipgrams, trigrams),
        comfort=ComfortBigramScorer(bigrams, skipgram_freqs=skipgrams),
        bigram_mass=float(sum(bigrams.values())),
        identity=corpus_identity(directory),
    )


def gauge_matrix(layouts: list[str], context: GaugeContext, progress_every: int = 0) -> np.ndarray:
    """``(n_layouts, 14)`` gauge matrix in :data:`LIVE_GAUGES` order.

    One :meth:`GaugeContext.vector` call per layout, hoisted out of the gauge loop. The
    obvious comprehension ``[[vector(lay)[g] for g in LIVE_GAUGES] for lay in layouts]``
    recomputes the whole vector once per gauge — 14x the work, and none of the underlying
    scorers memoize (verified: repeating ``pattern_shares`` on the same ``Layout`` object
    costs the same 0.055 s every time), so it turned 0.15 s per layout into 2.0 s.
    """
    rows = []
    for index, lay in enumerate(layouts):
        vector = context.vector(lay)
        rows.append([vector[g] for g in LIVE_GAUGES])
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  gauges: {index + 1}/{len(layouts)} layouts", flush=True)
    return np.array(rows)


# --------------------------------------------------------------------------------------
# Targets: fitted-surface time, on the honest (native) frame
# --------------------------------------------------------------------------------------

#: The two frames a surface array can be in. ``native`` keeps each source's OWN bigram
#: tensor; ``standardized`` substitutes the production AALTO one into all of them, which
#: shares that tensor across sources and so weakens any cross-source claim.
SURFACE_FRAMES = ("native", "standardized")


@dataclass(frozen=True)
class TargetSurface:
    """One fitted surface as a scoring target, with its identity attached."""

    name: str
    frame: str
    array: np.ndarray
    sha256: str

    @property
    def pool(self) -> str:
        """``AALTO`` / ``COMMUNITY`` / ``POOL`` — the model pool this surface came from."""
        return self.name.split("_", 1)[0]


def load_target_surface(name: str, surface_dir: str, frame: str = "native") -> TargetSurface:
    """Load ``<name>.<frame>.npy`` from ``surface_dir`` with its digest.

    Unlike :func:`keybo.analysis.surfaces.load_surface` (which resolves only the
    ``.standardized`` arrays, because those are what the repo vendors) this reads either
    frame, so a cross-source claim can be made on the frame that does not share a bigram
    tensor between the sources being compared.
    """
    if frame not in SURFACE_FRAMES:
        raise ValueError(f"unknown surface frame {frame!r}; expected one of {SURFACE_FRAMES}")
    path = Path(surface_dir) / f"{name}.{frame}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"surface {name!r} ({frame} frame) not found at {path}")
    array = np.load(path)
    if array.shape != (31, 31, 31):
        raise ValueError(f"surface {name!r} has shape {array.shape}, expected (31, 31, 31)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"surface {name!r} holds non-finite values")
    return TargetSurface(
        name=name,
        frame=frame,
        array=array,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def surface_ms_per_trigram(layouts: list[str], surface: TargetSurface, objective) -> np.ndarray:
    """Corpus-weighted fit per layout, normalized to ms per scored trigram.

    Normalizing by the objective's own frequency mass makes the number comparable across
    corpora and readable as a rate; the ranking is identical to the raw sum (the mass is
    one positive constant for a fixed corpus).
    """
    mass = float(objective[3].sum())
    return np.array([S.score_fit(lay, surface.array, objective) / mass for lay in layouts])


# --------------------------------------------------------------------------------------
# Correlation clusters (failure mode 1)
# --------------------------------------------------------------------------------------


def correlation_clusters(X: np.ndarray, threshold: float = 0.9) -> dict[str, list[str]]:
    """Group :data:`LIVE_GAUGES` by |Spearman rho| >= ``threshold`` (single linkage).

    Single linkage on the absolute rank correlation, because the hazard being guarded
    against is *any* chain of near-restatement: ``lsb``->``lsb-dist`` at rho 1.00 means
    dropping one leaves the information in place, so leave-one-GAUGE-out is
    anti-conservative and only leave-one-GROUP-out bites.

    Cluster keys are the member names joined by ``+``, so a cluster is self-describing in
    a table with no legend.
    """
    from scipy.stats import spearmanr

    rho = np.atleast_2d(spearmanr(X).statistic)
    n = len(LIVE_GAUGES)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(rho[i, j]) and abs(rho[i, j]) >= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(LIVE_GAUGES[i])
    return {"+".join(members): members for members in groups.values()}


def effective_dof(X: np.ndarray) -> float:
    """Participation-ratio effective rank of the gauge correlation matrix.

    ``(sum lambda)^2 / sum lambda^2`` over the eigenvalues of the correlation matrix — 14
    if the gauges were independent, ~1 if they all restate one fact. Estimate it inside a
    HOMOGENEOUS pool: pooling optimized and random layouts gives a number BELOW both
    sub-pools (a Simpson artifact), which is also the most flattering number available.
    """
    if X.shape[0] < 3:
        return float("nan")
    keep = X.std(axis=0) > 0
    corr = np.corrcoef(X[:, keep], rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    return float(total**2 / (eigenvalues**2).sum()) if total > 0 else float("nan")


# --------------------------------------------------------------------------------------
# Loss curves (failure mode 2)
# --------------------------------------------------------------------------------------

#: Candidate functional forms for a gauge's price, with their parameter counts. A hinge is
#: included because a saturating or threshold cost is the shape a linear weight most
#: badly misprices, and it is exactly what a community constant cannot express.
CURVE_FORMS = ("linear", "quadratic", "hinge")


def _design(form: str, x: np.ndarray, knot: float | None = None) -> np.ndarray:
    centered = x - (knot if knot is not None else 0.0)
    if form == "linear":
        return np.column_stack([np.ones_like(x), x])
    if form == "quadratic":
        return np.column_stack([np.ones_like(x), x, x**2])
    if form == "hinge":
        return np.column_stack([np.ones_like(x), x, np.clip(centered, 0.0, None)])
    raise ValueError(f"unknown curve form {form!r}; expected one of {CURVE_FORMS}")


@dataclass
class LossCurve:
    """One gauge's fitted price: SHAP contribution (ms/trigram) vs the gauge's own level.

    ``form``/``coeffs``/``knot`` define the function; ``weight`` is its linearization (the
    ordinary-least-squares slope over the fitted domain), which is the scalar a community
    analyzer would publish. ``domain`` is the range the fit is supported over — outside it
    the curve is extrapolation and :meth:`price` says so via
    :meth:`EvidenceWeights.score`.
    """

    metric: str
    form: str
    coeffs: list[float]
    knot: float | None
    domain: tuple[float, float]  # 1st..99th percentile of the fitted pool
    observed_range: tuple[float, float]  # full min..max of the fitted pool
    weight: float  # linearized slope, ms/trigram per gauge unit
    weight_ci: tuple[float, float]  # bootstrap 95% interval on the slope
    r2: float  # of the chosen form against the SHAP values
    r2_linear: float  # the linear form's own R^2, so "curved" is quantified
    mean_abs_shap: float  # attribution magnitude, ms/trigram
    shap_share_pct: float  # share of total mean |SHAP| across gauges

    def price(self, level: float) -> float:
        """Attributed ms/trigram at a gauge level (no domain check — see ``score``)."""
        design = _design(self.form, np.array([float(level)]), self.knot)
        return float((design @ np.array(self.coeffs))[0])

    def in_domain(self, level: float) -> bool:
        return self.domain[0] <= float(level) <= self.domain[1]

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "form": self.form,
            "coeffs": list(self.coeffs),
            "knot": self.knot,
            "valid_domain": list(self.domain),
            "observed_range": list(self.observed_range),
            "weight_ms_per_unit": self.weight,
            "weight_ci95": list(self.weight_ci),
            "r2": self.r2,
            "r2_linear": self.r2_linear,
            "mean_abs_shap_ms": self.mean_abs_shap,
            "shap_share_pct": self.shap_share_pct,
        }


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    residual = float(((y - fitted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else 0.0


#: Interior quantiles the hinge knot is searched over. A one-dimensional grid rather than a
#: jointly-fitted knot: every candidate is scored by the same criterion, and no nonlinear
#: optimizer can land the knot outside the data.
_KNOT_QUANTILES = (0.25, 0.4, 0.5, 0.6, 0.75)

#: Folds for the form-selection cross-validation.
_CURVE_CV_FOLDS = 5

#: Extra share of the price's TOTAL variance a curved form must explain, out of fold, to be
#: adopted. Measured against total variance and NOT against the linear form's leftover
#: residual: a relative-to-residual threshold is unstable exactly where the linear fit is
#: already good, because 5% of an almost-zero residual is almost zero and a hinge can shave
#: that much off pure noise. (Caught by test_linear_price_selects_the_linear_form, which
#: selected "hinge" for y = 0.4x + N(0, 0.02).)
_CURVE_MIN_CV_GAIN = 0.01


def fit_loss_curve(
    metric: str,
    levels: np.ndarray,
    shap_values: np.ndarray,
    *,
    mean_abs_shap: float,
    shap_share_pct: float,
    bootstrap: int = 200,
    rng_seed: int = 0,
) -> LossCurve:
    """Fit a gauge's price curve, choosing the form by K-fold cross-validated error.

    Form selection is by out-of-fold error, **with the hinge's knot searched inside each
    training fold**, and that detail is the whole point. Selecting on R^2 always picks the
    most flexible form (R^2 cannot decrease when a parameter is added). Selecting on AICc
    is better but still wrong here: the knot is chosen by looking at the data, so a
    best-of-five grid search buys fit that a per-parameter penalty does not charge for. In
    the first run of this pipeline that mispricing made the hinge win **all 14** gauges,
    including ones whose price is flat noise (``lsb`` at R^2 = 0.02). Cross-validation
    charges for the search directly, because a knot picked on the training fold has to earn
    its keep on the held-out one.

    A curved form is adopted only if it beats linear by :data:`_CURVE_MIN_CV_GAIN` in
    relative out-of-fold error — a tie-break toward the simpler form, so "this cost
    saturates" is a claim the data had to pay for.
    """
    finite = np.isfinite(levels) & np.isfinite(shap_values)
    x, y = levels[finite], shap_values[finite]
    n = len(x)
    if n < 8 or x.std() == 0:
        # Not enough support (or a constant column): report an explicit flat curve rather
        # than a fitted-looking one.
        return LossCurve(
            metric=metric,
            form="linear",
            coeffs=[float(y.mean()) if n else 0.0, 0.0],
            knot=None,
            domain=(float(x.min()), float(x.max())) if n else (0.0, 0.0),
            observed_range=(float(x.min()), float(x.max())) if n else (0.0, 0.0),
            weight=0.0,
            weight_ci=(0.0, 0.0),
            r2=0.0,
            r2_linear=0.0,
            mean_abs_shap=mean_abs_shap,
            shap_share_pct=shap_share_pct,
        )

    def _lstsq(form: str, knot: float | None, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
        coeffs, *_ = np.linalg.lstsq(_design(form, xi, knot), yi, rcond=None)
        return coeffs

    def _best_knot(xi: np.ndarray, yi: np.ndarray) -> float | None:
        """The knot minimizing in-fold error — the search a CV fold has to pay for."""
        best: tuple[float, float | None] = (np.inf, None)
        for quantile in _KNOT_QUANTILES:
            knot = float(np.quantile(xi, quantile))
            if knot <= xi.min() or knot >= xi.max():
                continue
            residual = yi - _design("hinge", xi, knot) @ _lstsq("hinge", knot, xi, yi)
            best = min(best, (float((residual**2).sum()), knot))
        return best[1]

    # --- form selection by out-of-fold error, knot searched INSIDE each training fold ---
    folds = np.array_split(np.random.default_rng(rng_seed).permutation(n), _CURVE_CV_FOLDS)
    cv_error: dict[str, float] = {}
    for form in CURVE_FORMS:
        errors: list[float] = []
        for fold in folds:
            test = np.zeros(n, dtype=bool)
            test[fold] = True
            xtr, ytr, xte, yte = x[~test], y[~test], x[test], y[test]
            if len(xte) == 0 or xtr.std() == 0:
                continue
            knot = _best_knot(xtr, ytr) if form == "hinge" else None
            if form == "hinge" and knot is None:
                errors = []
                break
            predicted = _design(form, xte, knot) @ _lstsq(form, knot, xtr, ytr)
            errors.append(float(((yte - predicted) ** 2).sum()))
        if errors:
            cv_error[form] = float(np.sum(errors))
    if not cv_error:
        cv_error = {"linear": 0.0}

    linear_error = cv_error.get("linear", np.inf)
    form = min(cv_error, key=lambda f: cv_error[f])
    total_variance = float(((y - y.mean()) ** 2).sum())
    # Tie-break toward simplicity: the curve must explain this much MORE of the price's
    # total variance, out of fold, than the straight line does.
    if (
        form != "linear"
        and np.isfinite(linear_error)
        and total_variance > 0
        and (linear_error - cv_error[form]) / total_variance < _CURVE_MIN_CV_GAIN
    ):
        form = "linear"

    knot = _best_knot(x, y) if form == "hinge" else None
    if form == "hinge" and knot is None:
        form = "linear"
    coeffs = _lstsq(form, knot, x, y)
    r2 = _r2(y, _design(form, x, knot) @ coeffs)
    r2_linear = _r2(y, _design("linear", x, None) @ _lstsq("linear", None, x, y))

    # The linearization: the OLS slope of price on level. This is the number that is
    # directly comparable to a community taste constant.
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(rng_seed)
    slopes = []
    for _ in range(bootstrap):
        idx = rng.integers(0, n, n)
        if x[idx].std() == 0:
            continue
        slopes.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
    ci = (
        (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))
        if slopes
        else (slope, slope)
    )
    return LossCurve(
        metric=metric,
        form=form,
        coeffs=[float(c) for c in coeffs],
        knot=knot,
        domain=(float(np.percentile(x, 1)), float(np.percentile(x, 99))),
        observed_range=(float(x.min()), float(x.max())),
        weight=slope,
        weight_ci=ci,
        r2=r2,
        r2_linear=r2_linear,
        mean_abs_shap=mean_abs_shap,
        shap_share_pct=shap_share_pct,
    )


# --------------------------------------------------------------------------------------
# The surrogate and the fitted weight set
# --------------------------------------------------------------------------------------

#: Marks the surrogate's metadata so it can never be mistaken for a trained typing model:
#: it maps LAYOUT-LEVEL gauges to a surface fit, not n-gram features to a keystroke time.
SURROGATE_FEATURE_VERSION = "evidence-scorer-surrogate/1"

#: Surrogate hyperparameters. Deliberately NOT the production typing-model defaults: those
#: carry gamma=0.957, a min-split-loss tuned for the stroke-level target whose scale is
#: ~100x this one, and on a layout-level target it prunes the tree to a stump. Shallow
#: trees + many rounds keep the SHAP dependence curves smooth enough to fit a form to.
SURROGATE_PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    # Single-threaded ON PURPOSE. The matrix here is a few hundred rows x 14 columns, and
    # on a 192-core box XGBoost's thread fan-out costs far more than the work: the same fit
    # took 96 s at the default thread count and ~1 s at n_jobs=1. This also makes the fit
    # bit-reproducible across machines with different core counts, which matters because
    # the leave-one-source-out cells are compared against each other.
    "n_jobs": 1,
}


def _fit_surrogate(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """An XGBoost surrogate mapping the gauge vector to the surface's ms/trigram."""
    from keybo.models.base import ModelMetadata
    from keybo.models.xgboost_model import XGBoostTypingModel

    metadata = ModelMetadata(
        feature_version=SURROGATE_FEATURE_VERSION,
        feature_names=list(LIVE_GAUGES),
        wpm_range=(int(S.BAKED_WPM), int(S.BAKED_WPM)),
        ngram="bigram",
        extra={"training": {"target_space": "MS"}, "role": "evidence-scorer gauge surrogate"},
    )
    model = XGBoostTypingModel(metadata, random_state=seed, **SURROGATE_PARAMS)
    return model.fit(X, y)


@dataclass
class EvidenceWeights:
    """A fitted price per gauge, plus the cluster view and the provenance to read it by."""

    source: str  # surface name the weights were fitted on
    frame: str  # native | standardized
    corpus: str
    corpus_sha256: dict
    surface_sha256: str
    n_layouts: int
    pool_label: str
    curves: dict[str, LossCurve]
    clusters: dict[str, list[str]]
    cluster_shap_share_pct: dict[str, float]
    cluster_weight: dict[str, float]
    effective_dof: float
    surrogate_r2_in_sample: float
    surrogate_r2_holdout: float | None
    base_value: float
    #: Pool guards, attached AFTER construction by `attach_pool_guards` because they are
    #: properties of the fitting POOL rather than of the fitted weights. Left None they simply
    #: produce no verdict — which is why `attach_pool_guards` exists and the CLI always calls it.
    cd_ratio: float | None = None
    source_agreement: float | None = None
    notes: list[str] = field(default_factory=list)

    # -- scoring ----------------------------------------------------------------------

    def score(self, gauges: dict[str, float]) -> dict:
        """Score one layout's gauge vector. Lower = faster (it is a predicted-time loss).

        Returns the total, the per-gauge price, and — load-bearing — every gauge whose
        level falls outside the curve's fitted domain. An out-of-domain gauge is still
        priced (refusing outright would make the tool useless on qwerty, which is the most
        interesting comparison), but the total carries the flag so a caller cannot quote it
        as an in-domain number by accident.
        """
        per_gauge: dict[str, float] = {}
        out_of_domain: dict[str, dict] = {}
        for name, curve in self.curves.items():
            level = float(gauges[name])
            per_gauge[name] = curve.price(level)
            if not curve.in_domain(level):
                out_of_domain[name] = {
                    "level": level,
                    "valid_domain": list(curve.domain),
                    "distance_outside": float(
                        max(curve.domain[0] - level, level - curve.domain[1], 0.0)
                    ),
                }
        clusters = {
            key: float(sum(per_gauge[m] for m in members)) for key, members in self.clusters.items()
        }
        return {
            "score": float(sum(per_gauge.values())),
            "unit": "ms/trigram (attributed); lower = faster",
            "frame": FRAME_NOTE,
            "per_gauge": per_gauge,
            "per_cluster": clusters,
            "out_of_domain": out_of_domain,
            "extrapolating": bool(out_of_domain),
            "source": self.source,
            "surface_frame": self.frame,
            "corpus": self.corpus,
            "modelled_only": MODELLED_ONLY_NOTE,
        }

    def score_layout(self, lay30: str, context: GaugeContext) -> dict:
        return self.score(context.vector(lay30))

    def transfer_warning(
        self, source_agreement: float | None = None, cd_ratio: float | None = None
    ) -> str | None:
        """The caveat a caller must see when the fitting pool is too narrow to transfer.

        Measured, not asserted: the same pipeline that beats genkey / oxeylyzer-1 /
        oxeylyzer-2 in **12 of 12** independent cross-source cells on a WIDE pool of random
        C30M permutations (mean Spearman advantage +0.346) **loses all 12** on a NARROW pool
        of near-optimal archive layouts (mean advantage -0.308). The cause is not the scorer:
        the two independent sources' agreement with *each other* — the ceiling for any
        cross-source scorer — falls from +0.835 on the wide pool to +0.265 on the narrow one.
        Where there is shared signal the fit recovers 86-89% of it; where there is almost
        none it learns the fitting source's idiosyncrasy instead, and the community's
        constants (which track the source-robust component: +0.502 against a two-source
        consensus versus +0.313 against one source alone) win.

        So a weight set fitted on a narrow pool must not be quoted as a general scorer, and
        this string is how the artifact says so.
        """
        if cd_ratio is not None and cd_ratio < NARROW_POOL_CD:
            return (
                f"DO NOT TRUST FOR RANKING: the fitting pool's consensus/disagreement ratio "
                f"is {cd_ratio:.3f}, below {NARROW_POOL_CD}. The pool retains the sources' "
                f"DISAGREEMENT but not their CONSENSUS, so there is no shared signal to learn "
                f"and no weight set fitted here can transfer (measured: C/D 1.058 -> ceiling "
                f"+0.218, 0 of 12 cross-source cells). This is computable before any fit — "
                f"widen the pool, or use the community scorers."
            )
        if source_agreement is not None and source_agreement < NARROW_POOL_SOURCE_AGREEMENT:
            return (
                f"DO NOT TRUST FOR RANKING: the fitting pool's cross-source agreement is "
                f"{source_agreement:.3f}, below {NARROW_POOL_SOURCE_AGREEMENT}. On a pool this "
                f"narrow these weights lost to genkey/oxeylyzer in 12 of 12 independent cells "
                f"(mean Spearman -0.308) because there is too little shared signal to learn. "
                f"Fit on a wider pool, or use the community scorers."
            )
        # ⚠ NO effective-dof branch. It used to fire here on `self.effective_dof <
        # NARROW_POOL_DOF`, but that floor was calibrated on the very contrast it was used to
        # detect and POOLSWEEP-1 measured it false-positiving at interp-f0.25 (dof 2.43,
        # ceiling +0.9244). A pool can be narrow in the HARMLESS direction — restricted
        # disagreement rather than restricted consensus — which raises the ceiling to +0.9999
        # while lowering dof. effective_dof stays on the artifact as a diagnostic; it must not
        # gate a verdict. See NARROW_POOL_DOF's own docstring.
        return None

    def sign_audit(self) -> dict:
        """Which fitted weights have a mechanistically IMPLAUSIBLE sign, and how many.

        Not a correctness gate — the whole point of a fitted scorer is that it may overturn a
        prior, and this campaign has already measured two real community mispricings. But a
        fitted sign that contradicts the mechanism is far more likely to be the correlated
        frame speaking than a discovery, so the count travels with the weights.

        It is load-bearing: on the wide pool **5 of 14** signs come out implausible, including
        ``sfb`` at -0.112 — "more same-finger bigrams is faster", which is false and is the
        community's most agreed-upon penalty. With effective dof ~5 over 14 axes, Shapley
        distributes credit among restatements, so individual signs are not identified even
        when the ensemble ranks well. A user reading a per-gauge weight as "the evidence-based
        price of sfb" would be actively misled, so the tool says so itself.
        """
        implausible, plausible = [], []
        for name, curve in self.curves.items():
            expected = EXPECTED_SIGN.get(name)
            sign = float(np.sign(curve.weight))
            if expected is None or sign == 0.0:
                continue
            (plausible if sign == expected else implausible).append(
                {"metric": name, "weight": curve.weight, "expected_sign": expected}
            )
        total = len(plausible) + len(implausible)
        return {
            "n_checked": total,
            "n_plausible": len(plausible),
            "n_implausible": len(implausible),
            "implausible": sorted(implausible, key=lambda r: -abs(r["weight"])),
            "interpretation": (
                "a sign that contradicts the mechanism is more likely the correlated frame "
                "(effective dof ~5 over 14 axes) than a discovery. Read the CLUSTER column "
                "and the ensemble ranking; do NOT quote a single gauge's weight as its "
                "evidence-based price."
            ),
        }

    # -- serialization ----------------------------------------------------------------

    def weight_table(self) -> list[dict]:
        """Per-gauge rows sorted by attribution share, descending.

        Each row carries ``sign_plausible``: whether the fitted sign agrees with the
        mechanism. See :meth:`sign_audit` for why that flag is necessary.
        """
        rows = []
        for curve in self.curves.values():
            row = curve.to_dict()
            expected = EXPECTED_SIGN.get(curve.metric)
            sign = float(np.sign(curve.weight))
            row["expected_sign"] = expected
            row["sign_plausible"] = None if expected is None or sign == 0.0 else sign == expected
            rows.append(row)
        rows.sort(key=lambda r: -r["shap_share_pct"])
        return rows

    def attach_pool_guards(
        self, cd_ratio: float | None, source_agreement: float | None
    ) -> EvidenceWeights:
        """Record the fitting pool's guard values so they survive into `to_dict`.

        Separate from the constructor because both quantities describe the POOL the weights
        were fitted on, not the weights: they are computed from the multi-source targets, which
        the fitting routine does not own. Returns self so it can be chained at the call site.
        """
        self.cd_ratio = cd_ratio
        self.source_agreement = source_agreement
        return self

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "surface_frame": self.frame,
            "surface_sha256": self.surface_sha256,
            "corpus": self.corpus,
            "corpus_sha256": self.corpus_sha256,
            "pool": self.pool_label,
            "n_layouts": self.n_layouts,
            "frame": FRAME_NOTE,
            "gauges": list(LIVE_GAUGES),
            "excluded_invariant_gauges": list(INVARIANT_GAUGES),
            "effective_dof": self.effective_dof,
            "surrogate_r2_in_sample": self.surrogate_r2_in_sample,
            "surrogate_r2_holdout": self.surrogate_r2_holdout,
            "base_value_ms_per_trigram": self.base_value,
            "weights": self.weight_table(),
            "clusters": {
                key: {
                    "members": members,
                    "shap_share_pct": self.cluster_shap_share_pct[key],
                    "weight_ms_per_unit_sum": self.cluster_weight[key],
                }
                for key, members in self.clusters.items()
            },
            # The guards must SURVIVE SERIALIZATION. Before GUARD-CD-1 this called
            # transfer_warning() with no arguments, which was harmless only while the retired
            # effective-dof branch could fire from `self` alone. Now that the verdict depends
            # on C/D — a property of the POOL, not of this object — the values have to be
            # attached (see `attach_pool_guards`) or the artifact silently carries no verdict.
            "transfer_warning": self.transfer_warning(
                source_agreement=self.source_agreement, cd_ratio=self.cd_ratio
            ),
            "pool_guards": {
                "cd_ratio": self.cd_ratio,
                "cd_floor": NARROW_POOL_CD,
                "source_agreement": self.source_agreement,
                "source_agreement_floor": NARROW_POOL_SOURCE_AGREEMENT,
                "effective_dof": self.effective_dof,
                "effective_dof_note": (
                    "diagnostic ONLY — never gates a verdict; the old NARROW_POOL_DOF floor was "
                    "circular and false-positived at dof 2.43 with a healthy +0.9244 ceiling"
                ),
            },
            "sign_audit": self.sign_audit(),
            "notes": [*self.notes, MODELLED_ONLY_NOTE, SOURCE_INDEPENDENCE_NOTE],
        }


def fit_evidence_weights(
    layouts: list[str],
    surface: TargetSurface,
    context: GaugeContext,
    objective,
    *,
    pool_label: str = "unspecified",
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    holdout_frac: float = 0.25,
    cluster_threshold: float = 0.9,
    bootstrap: int = 200,
    seed: int = 0,
    shuffle_target: bool = False,
) -> EvidenceWeights:
    """Derive a price per gauge from one surface, over one layout pool.

    ``X``/``y`` may be supplied to skip recomputation (the gauge matrix is the expensive
    part). ``shuffle_target`` permutes the target — the NOISE PLACEBO: it runs the entire
    pipeline on labels that carry no signal, so the real pipeline's advantage can be read
    against what the machinery manufactures from noise. A stability or advantage figure
    without that comparison is uninterpretable.
    """
    from keybo.analysis.shap_report import compute_shap

    if X is None:
        X = gauge_matrix(layouts, context)
    if y is None:
        y = surface_ms_per_trigram(layouts, surface, objective)
    y = np.asarray(y, dtype=np.float64)
    if shuffle_target:
        y = np.random.default_rng(seed + 9_973).permutation(y)

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    holdout_r2: float | None = None
    if 0.0 < holdout_frac < 1.0 and n >= 40:
        order = rng.permutation(n)
        cut = int(round(n * (1.0 - holdout_frac)))
        train_idx, test_idx = order[:cut], order[cut:]
        probe = _fit_surrogate(X[train_idx], y[train_idx], seed=seed)
        holdout_r2 = _r2(y[test_idx], probe.predict(X[test_idx]))

    model = _fit_surrogate(X, y, seed=seed)
    in_sample_r2 = _r2(y, model.predict(X))
    report = compute_shap(model, X, interactions_max_rows=min(n, 400), rng_seed=seed)
    share = report.importance_share()

    curves: dict[str, LossCurve] = {}
    for column, name in enumerate(LIVE_GAUGES):
        curves[name] = fit_loss_curve(
            name,
            X[:, column],
            report.shap_values[:, column],
            mean_abs_shap=float(report.mean_abs[column]),
            shap_share_pct=float(share[name]),
            bootstrap=bootstrap,
            rng_seed=seed + column,
        )

    clusters = correlation_clusters(X, threshold=cluster_threshold)
    return EvidenceWeights(
        source=surface.name,
        frame=surface.frame,
        corpus=context.corpus_name,
        corpus_sha256=dict(context.identity.get("sha256", {})),
        surface_sha256=surface.sha256,
        n_layouts=n,
        pool_label=pool_label,
        curves=curves,
        clusters=clusters,
        cluster_shap_share_pct={
            key: float(sum(curves[m].shap_share_pct for m in members))
            for key, members in clusters.items()
        },
        cluster_weight={
            key: float(sum(curves[m].weight for m in members)) for key, members in clusters.items()
        },
        effective_dof=effective_dof(X),
        surrogate_r2_in_sample=in_sample_r2,
        surrogate_r2_holdout=holdout_r2,
        base_value=float(report.base_value),
        notes=(
            ["NOISE PLACEBO: the target was SHUFFLED; these weights are fitted to noise."]
            if shuffle_target
            else []
        ),
    )


def weights_to_json(weights: EvidenceWeights, path: str | Path) -> None:
    Path(path).write_text(json.dumps(weights.to_dict(), indent=2))
