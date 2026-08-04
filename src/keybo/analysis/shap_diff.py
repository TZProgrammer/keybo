"""Per-feature attribution of the ms/char GAP between two layouts (SHAPDIFF-1).

``shap-report`` answers "what does the model use?" over a matrix of rows. This module
answers the different, harder question: **why is layout B slower than layout A?**, as a
signed per-feature budget in the units the analyzer publishes (ms/char) that SUMS BACK to
the measured gap.

The construction, and the three places a plausible-looking version goes wrong:

**1. The channel.** ``analyze``'s gauge is ``ms/char = sum_tri f*(T2[a,b] + Tcond[a,b,c]) /
covered``. A per-BIGRAM feature attribution can only speak for the ``T2`` term; the
conditioned-trigram increment is a separate model on a 46-column trigram frame and is not
bigram-decomposable. So the gap is split into ``gap_t2 + gap_tcond`` FIRST, both reported,
and only ``gap_t2`` is decomposed. Attributing the whole gap to bigram features would be
wrong, and the split is what makes the over-claim impossible rather than merely discouraged.

**2. The weights.** The bigram weight is NOT ``bigrams.txt``. ``TimeSurface.card`` iterates
the TRIGRAM table and accumulates ``T2[a,b]*f`` per trigram, so the effective weight of a
character bigram is the trigram table's first-two-character marginal ``w2(x,y) = sum_z
tri(x,y,z)`` restricted to on-board trigrams (``triple_ms_table``'s docstring records that
using ``bigrams.txt`` here is ~1.5e-2 wrong). ``weighting="bigram-table"`` exists to
reproduce that error on demand as a NEGATIVE CONTROL — it must fail the reconciliation the
default passes.

**3. The space.** The boosters predict ``p = log(ms*wpm/12000)``; TreeSHAP is exact and
additive in ``p``, but log contributions do not sum to a ms difference. Rather than
linearize, this module uses the **log-mean Divisia (LMDI)** weight, which is an ALGEBRAIC
IDENTITY, not an approximation::

    L = (ms_B - ms_A) / (p_B - p_A)          [L := (ms_A + ms_B)/2 when p_B == p_A]
    ms_B - ms_A  ==  L * (p_B - p_A)  ==  L * sum_i (shap_i^B - shap_i^A)     EXACTLY

so there is no linearization residual to report — the measured reconciliation residual is
pure float error (~1e-16 relative). The alternatives are not close: on the same cells,
weighting by ``ms_A`` (a first-order expansion at A) is ~4000% wrong and weighting by the
midpoint ``(ms_A+ms_B)/2`` is ~1.65% wrong. ⚠ The honest price of exactness is that ``L``
is **pair-specific** — it depends on both boards' predictions at that cell — so an LMDI ms
attribution is a property of the A->B COMPARISON and not of either board alone. A per-board
budget is only well defined in log space, which is why :attr:`FeatureContribution` carries
both and labels them.

**Seed averaging is exact, not approximate.** The production ``T2`` is the mean over three
boosters *in milliseconds*, i.e. ``T2 = mean_s exp(p_s)*K``. A mean of exponentials is not
the exponential of a mean, so a single log-space attribution against ``log(T2*wpm/12000)``
would not sum correctly. Attributing per seed and averaging the resulting MS attributions
does, because both steps are exact: LMDI closes per seed, and the seed mean is linear in the
space the attributions live in.

**Common support.** Trigrams are scored only when every character is on the board, so two
layouts with different CHARSETS cover different corpus subsets and their per-board ms/char
have different denominators — a difference of two averages over two different populations is
not decomposable per bigram. This module therefore decomposes the COMMON-SUPPORT gap
(trigrams typeable on both boards, the convention :mod:`keybo.analysis.layout_diff` uses)
and reports the own-support gap beside it, so the coverage cost is a number rather than an
assumption. When the two charsets are permutations of one another — as any two boards over a
fixed key set are — the two are the same by construction and the restriction is a no-op,
which :attr:`ShapDiff.common_support_is_noop` states rather than implies.

What the decomposition CANNOT see, by construction of the 20-column served frame: no
hand-identity channel and no direction-of-travel channel (``inwards``/``outwards`` are
swap-invariant — see :mod:`keybo.features.schema`), and the row/finger one-hots describe the
LANDING key only. And a SHAP attribution is an attribution ON THIS MODEL: it says what the
fitted surface prices, not what a hand does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb

from keybo.analysis.timecard import TimeSurface, default_surface
from keybo.features import bigram_features_from_positions
from keybo.models.xgboost_model import XGBoostTypingModel

#: Weighting conventions. ``trigram-marginal`` is the gauge's own weight (the default and the
#: only correct one); ``bigram-table`` reproduces the ~1.5e-2 error of weighting by
#: ``bigrams.txt`` and exists to be a failing negative control.
WEIGHTINGS = ("trigram-marginal", "bigram-table")


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed share of the A->B gap.

    ``ms_per_char`` is the LMDI attribution and is the headline: it is denominated in the
    same ms/char the analyzer publishes, and the contributions SUM to
    :attr:`ShapDiff.gap_t2`. Its sign follows the gap's — POSITIVE means this feature makes
    ``layout_b`` slower than ``layout_a`` (i.e. it favours A).

    ``log_a`` / ``log_b`` are the frequency-weighted mean SHAP value of this feature on each
    board in the model's own log space. Unlike the ms column they are per-BOARD quantities
    (no pair-specific weight enters), so they are the honest answer to "what does this
    feature cost on board X"; ``log_delta`` is their difference. The two columns are
    monotonically related but NOT proportional, because LMDI reweights each bigram cell by
    its own local ms-per-log slope.
    """

    feature: str
    ms_per_char: float
    log_a: float
    log_b: float
    log_delta: float

    @property
    def favours(self) -> str:
        """Which layout this feature's contribution favours (``"a"``, ``"b"``, or ``"tie"``)."""
        if self.ms_per_char > 0:
            return "a"
        return "b" if self.ms_per_char < 0 else "tie"


@dataclass
class ShapDiff:
    """The full decomposition, with every reconciliation residual exposed as a number.

    A consumer should check :meth:`reconciles` before reading any feature: a table that does
    not sum to the gap it claims to decompose is worse than no table.
    """

    name_a: str
    name_b: str
    layout_a: str
    layout_b: str
    corpus: str
    target_wpm: float
    weighting: str
    feature_names: list[str]

    #: Each board's ms/char over its OWN typable corpus subset, computed on THIS module's
    #: tables. Matches ``TimeSurface.card().ms_per_char`` (what ``keybo analyze`` prints) to
    #: the float32 booster noise measured in :attr:`resid_vs_card_gap`.
    ms_per_char_own_a: float
    ms_per_char_own_b: float
    #: The shipped ``TimeSurface.card()`` numbers verbatim, for the R5 comparison.
    card_ms_per_char_a: float
    card_ms_per_char_b: float
    #: Each board's ms/char over the COMMON subset (equal to the own-support number whenever
    #: the charsets are permutations of one another).
    ms_per_char_a: float
    ms_per_char_b: float

    #: ``b - a``: POSITIVE means layout_a is faster. Split exactly into the two channels.
    gap_total: float
    gap_t2: float
    gap_tcond: float

    contributions: list[FeatureContribution]

    covered_mass_a: int
    covered_mass_b: int
    covered_mass_common: int
    corpus_total_mass: int

    #: Residuals. Every one is a float-error quantity; see :meth:`reconciles` for the bars.
    resid_additivity: float  # R1: base + sum_i shap_i vs booster.predict, in log space
    resid_cell_lmdi: float  # R2: per-cell  sum_i attrib_i  vs  T2[b] - T2[a]  (rel)
    resid_feature_sum: float  # R3: sum_i contribution_i  vs  gap_t2            (rel)
    resid_channel_split: float  # R4: gap_t2 + gap_tcond    vs  gap_total       (rel)
    #: R5: this module's own ms/char vs ``TimeSurface.card`` — the tie to what ``analyze``
    #: prints, in ABSOLUTE ms/char. Not a float64 identity: the SHAP-anchored T2 differs from
    #: the shipped ``predict()``-anchored one by the booster's float32 noise (see
    #: :attr:`resid_vs_shipped_t2`), so this is a SMALLNESS check, not an exactness one.
    resid_vs_card_gap: float
    #: Max absolute ms deviation between the SHAP-anchored T2 table and the shipped
    #: ``TimeSurface._T2``. The price of anchoring the identity on the TreeSHAP walk.
    resid_vs_shipped_t2: float
    #: The weighted log-space identity checked against the INDEPENDENT ``predict()`` path.
    #: (Checking ``base + sum(shap)`` against a ``p`` DEFINED as ``base + sum(shap)`` would be
    #: a tautology that can never fail — a degenerate control. This compares the TreeSHAP walk
    #: to the ordinary prediction, so it can.)
    resid_log_vs_predict: float

    #: ``(n_char, n_char, n_feature)`` weighted ms/char attribution per character bigram,
    #: retained so :meth:`top_bigrams` can name WHICH bigrams drive a feature.
    _weighted: np.ndarray = field(repr=False)
    _chars: str = field(repr=False)

    # --- the reconciliation gate -------------------------------------------------------

    def reconciles(
        self, rel_tol: float = 1e-9, add_tol: float = 1e-5, gauge_tol: float = 1e-3
    ) -> bool:
        """True iff every identity holds at the SHAPDIFF-1 registered bars.

        Three tolerances because three DIFFERENT kinds of quantity are being checked, and
        collapsing them would either hide a real bug or fail on irreducible artifact noise:

        * ``rel_tol`` — the ms-space identities (R2, R3, R4). These are exact algebra and land
          at float64 rounding; they are the bars that catch an attribution bug.
        * ``add_tol`` — the log-space cross-checks (R1 and the ``predict()`` comparison). The
          boosters are float32, so two independent xgboost code paths agree to ~4e-7 and never
          better. Holding these to ``rel_tol`` would fail on the artifact, not on this code.
        * ``gauge_tol`` — the tie to the shipped gauge (R5), in ABSOLUTE ms/char. Same float32
          origin, but it rides on a ~255 ms/char level, so it is expressed absolutely.
        """
        # bool(), not the bare `and` chain: several residuals are numpy scalars, so the chain
        # returns np.bool_ — which is falsy-correct but NOT JSON-serializable, and `to_dict`
        # embeds this value. Caught by the CLI's own JSON round-trip test.
        return bool(
            self.resid_additivity <= add_tol
            and self.resid_log_vs_predict <= add_tol
            and self.resid_cell_lmdi <= rel_tol
            and self.resid_feature_sum <= rel_tol
            and self.resid_channel_split <= rel_tol
            and self.resid_vs_card_gap <= gauge_tol
        )

    @property
    def common_support_is_noop(self) -> bool:
        """True iff restricting to common support changed nothing (equal charsets).

        Stated rather than assumed: when it is True the own-support and common-support gaps
        are the same number, so the coverage-asymmetry confound cannot be present. When it
        is False, :attr:`coverage_cost` prices what the restriction moved.
        """
        return self.covered_mass_a == self.covered_mass_b == self.covered_mass_common

    @property
    def coverage_cost(self) -> float:
        """Own-support gap minus common-support gap: what the coverage restriction moved.

        Exactly zero when the charsets are permutations of one another — and it is zero *by
        arithmetic*, not by tolerance, because both sides are computed from THIS module's
        tables. (Differencing against the shipped ``card()`` instead would mix two anchors and
        make this quantity report the float32 gauge noise of :attr:`resid_vs_card_gap`, which
        has nothing to do with coverage.) Non-zero means the two boards' published ms/char are
        averages over DIFFERENT corpus subsets, and only the common-support gap is decomposable
        per bigram.
        """
        return (self.ms_per_char_own_b - self.ms_per_char_own_a) - self.gap_total

    @property
    def coverage_pct_a(self) -> float:
        return 100.0 * self.covered_mass_a / max(self.corpus_total_mass, 1)

    @property
    def coverage_pct_b(self) -> float:
        return 100.0 * self.covered_mass_b / max(self.corpus_total_mass, 1)

    @property
    def decomposed_share_pct(self) -> float:
        """Share of the total gap the bigram channel accounts for, in percent.

        The number that keeps the headline honest. It can exceed 100% or be negative — that
        happens when the two channels DISAGREE in sign, which is a finding about the pair,
        not an error.
        """
        return 100.0 * self.gap_t2 / self.gap_total if self.gap_total else float("nan")

    # --- views ------------------------------------------------------------------------

    def ranked(self) -> list[FeatureContribution]:
        """Contributions sorted by |ms/char| descending — the report's primary table."""
        return sorted(self.contributions, key=lambda c: -abs(c.ms_per_char))

    def top_bigrams(self, feature: str, k: int = 8) -> list[tuple[str, float]]:
        """``(bigram, ms/char)`` pairs driving ``feature``, largest |contribution| first.

        Turns "``distance`` explains 1.2 ms/char" into a statement about the bigrams that
        produced it. The bigram is written in the CHARACTERS of the corpus (space rendered
        as ``␣``), and its value already carries the corpus weight, so the values for one
        feature sum to that feature's :attr:`FeatureContribution.ms_per_char`.
        """
        col = self.feature_names.index(feature)
        block = self._weighted[:, :, col]
        flat = np.argsort(-np.abs(block), axis=None)[:k]
        out = []
        for pos in flat:
            i, j = np.unravel_index(pos, block.shape)
            value = float(block[i, j])
            if value == 0.0:
                continue
            out.append((f"{self._chars[i]}{self._chars[j]}".replace(" ", "␣"), value))
        return out

    def to_dict(self, top_bigrams_k: int = 8) -> dict:
        return {
            "layout_a": {"name": self.name_a, "layout": self.layout_a},
            "layout_b": {"name": self.name_b, "layout": self.layout_b},
            "corpus": self.corpus,
            "target_wpm": self.target_wpm,
            "weighting": self.weighting,
            "ms_per_char": {
                "own_support": {"a": self.ms_per_char_own_a, "b": self.ms_per_char_own_b},
                "common_support": {"a": self.ms_per_char_a, "b": self.ms_per_char_b},
                "shipped_card": {"a": self.card_ms_per_char_a, "b": self.card_ms_per_char_b},
            },
            "gap": {
                "total": self.gap_total,
                "t2_bigram_channel": self.gap_t2,
                "tcond_trigram_channel": self.gap_tcond,
                "decomposed_share_pct": self.decomposed_share_pct,
                "sign_convention": "positive = layout_a is faster",
            },
            "coverage": {
                "covered_mass_a": self.covered_mass_a,
                "covered_mass_b": self.covered_mass_b,
                "covered_mass_common": self.covered_mass_common,
                "corpus_total_mass": self.corpus_total_mass,
                "pct_a": self.coverage_pct_a,
                "pct_b": self.coverage_pct_b,
                "common_support_is_noop": self.common_support_is_noop,
                "coverage_cost_ms_per_char": self.coverage_cost,
            },
            "residuals": {
                "additivity_log_abs": self.resid_additivity,
                "log_walk_vs_predict_abs": self.resid_log_vs_predict,
                "cell_lmdi_rel": self.resid_cell_lmdi,
                "feature_sum_vs_gap_t2_rel": self.resid_feature_sum,
                "channel_split_rel": self.resid_channel_split,
                "gap_vs_shipped_card_abs_ms_per_char": self.resid_vs_card_gap,
                "t2_table_vs_shipped_abs_ms": self.resid_vs_shipped_t2,
                "reconciles": self.reconciles(),
            },
            "contributions": [
                {
                    "feature": c.feature,
                    "ms_per_char": c.ms_per_char,
                    "favours": c.favours,
                    "share_of_gap_t2_pct": (
                        100.0 * c.ms_per_char / self.gap_t2 if self.gap_t2 else None
                    ),
                    "log_a": c.log_a,
                    "log_b": c.log_b,
                    "log_delta": c.log_delta,
                    "top_bigrams": [
                        {"bigram": bg, "ms_per_char": v}
                        for bg, v in self.top_bigrams(c.feature, top_bigrams_k)
                    ],
                }
                for c in self.ranked()
            ],
        }


def _bigram_shap_tables(
    models: Sequence[XGBoostTypingModel],
    geometry,
    target_wpm: float,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], float, list[str]
]:
    """Per-seed ``(shap, p, p_predict, ms)`` position-pair tables from the exact TreeSHAP path.

    Returns ``(shap_tables, p_tables, p_predict_tables, ms_tables, worst_additivity,
    feature_names)`` with shapes ``(n_pos, n_pos, n_feat)`` / ``(n_pos, n_pos)`` x 3.

    ``p`` is the TreeSHAP walk's own total (``base + sum_i shap_i``) and is the ANCHOR for the
    ms conversion; ``p_predict`` is the ordinary prediction. Both are returned because they are
    two INDEPENDENT xgboost code paths, and comparing them is the only non-tautological
    additivity check available — a check of ``base + sum(shap)`` against a ``p`` *defined* as
    ``base + sum(shap)`` can never fail and would be a degenerate control.

    Contributions are cast to **float64 on arrival**. XGBoost returns them as float32, and a
    float32 sum over 20 columns carries ~1e-7 relative error — which is a hundred times the
    reconciliation bar and would be indistinguishable from a real attribution bug. The cast
    is lossless (every float32 is a float64) and moves every subsequent sum into float64.

    Every model is CHECKED, not assumed, for the three properties the identity needs: the
    served 20-column frame, LOGRAT output, and NO first-finger calibration. The last one
    matters most and is invisible: ``TableBigramScorer`` applies calibration deltas as a
    per-POSITION multiplicative factor OUTSIDE the feature path, so a calibrated model's
    table would not equal ``exp(prediction)`` and the ms attribution would silently stop
    summing to the gauge. The three shipped ``bigram_reg31`` artifacts carry none.
    """
    positions = [*geometry.slots, geometry.space_position]
    n = len(positions)
    X = np.vstack(
        [
            bigram_features_from_positions(geometry, (a, b), wpm=target_wpm)
            for a in positions
            for b in positions
        ]
    )
    names = list(models[0].metadata.feature_names)
    dmat = xgb.DMatrix(X)

    shap_tables, p_tables, p_predict_tables, ms_tables = [], [], [], []
    worst = 0.0
    for model in models:
        if list(model.metadata.feature_names) != names:
            raise ValueError("bigram models disagree on their feature frame")
        if model.target_space != "LOGRAT":
            raise ValueError(
                f"shap_diff needs LOGRAT bigram models (got {model.target_space}); the LMDI "
                "ms conversion is derived from ms = exp(p)*12000/wpm"
            )
        training = (model.metadata.extra.get("training") or {}) if model.metadata.extra else {}
        cal = training.get("calibration")
        if cal and cal.get("deltas_ms"):
            raise NotImplementedError(
                "shap_diff cannot attribute a model carrying first-finger calibration deltas: "
                "the deltas are a per-POSITION offset outside the 20-column feature path, so "
                "the SHAP contributions would not sum to the served T2 table"
            )
        contribs = np.asarray(
            model._regressor.get_booster().predict(dmat, pred_contribs=True), dtype=np.float64
        )
        shap, base = contribs[:, :-1], contribs[:, -1]
        # R1 measures the two INDEPENDENT xgboost code paths against each other: the TreeSHAP
        # walk and the ordinary prediction. They agree to float32 booster precision (~4e-7),
        # never to float64 — the disagreement is the artifact's own noise, not ours.
        p_predict = model.predict(X)
        p = shap.sum(axis=1) + base
        worst = max(worst, float(np.abs(p - p_predict).max()))
        # The SHAP-implied prediction is the ANCHOR for everything downstream, deliberately:
        # the LMDI weight must divide by exactly the quantity it later multiplies back, or the
        # identity inherits that ~4e-7 as a floor. Anchoring on predict() instead measured
        # 9.5e-07; anchoring here measures ~1e-16. The two differ by the booster's own float32
        # noise (median 7.5e-07 on the weight, max 1.6e-03), which is why the tables carry the
        # `base` column: a caller wanting the predict()-anchored number can reconstruct it.
        ms = np.exp(p) * 12000.0 / target_wpm
        shap_tables.append(shap.reshape(n, n, len(names)))
        p_tables.append(p.reshape(n, n))
        p_predict_tables.append(p_predict.reshape(n, n))
        ms_tables.append(ms.reshape(n, n))
    return shap_tables, p_tables, p_predict_tables, ms_tables, worst, names


def _char_weight_tables(surface: TimeSurface, chars: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Corpus weights in CHARACTER space over trigrams typeable on ``chars`` + space.

    Returns ``(w3, w2, covered)`` where ``w3[i,j,k]`` is the trigram frequency and ``w2`` is
    its first-two-character marginal — derived from ``w3`` rather than loaded separately, so
    the bigram weight cannot drift from the trigram weight that produced it (the ~1.5e-2
    ``bigrams.txt`` trap). ``covered`` is the summed mass, i.e. the ms/char denominator.
    """
    index = {c: i for i, c in enumerate(chars)}
    index[" "] = len(chars)
    n = len(chars) + 1
    w3 = np.zeros((n, n, n))
    for ngram, freq in surface.tri.items():
        try:
            i, j, k = index[ngram[0]], index[ngram[1]], index[ngram[2]]
        except KeyError:
            continue
        w3[i, j, k] += freq
    return w3, w3.sum(axis=2), int(round(w3.sum()))


def shap_diff(
    layout_a: str,
    layout_b: str,
    *,
    name_a: str | None = None,
    name_b: str | None = None,
    surface: TimeSurface | None = None,
    bigram_models: Sequence[XGBoostTypingModel] | None = None,
    target_wpm: float = 90.0,
    corpus: str | None = None,
    weighting: str = "trigram-marginal",
    control_bigram_freqs: Mapping[str, int] | None = None,
    shuffle_seed: int | None = None,
) -> ShapDiff:
    """Decompose ``ms/char(layout_b) - ms/char(layout_a)`` into per-feature contributions.

    ``layout_a`` / ``layout_b`` are 30-character row-major layout strings (the caller
    resolves registry names). The sign convention is fixed and stated in the output:
    POSITIVE means ``layout_a`` is faster.

    ``surface`` may be supplied to reuse a loaded :class:`TimeSurface` (model load dominates
    the runtime); ``bigram_models`` likewise, defaulting to the same three seeded
    ``bigram_reg31`` artifacts the surface itself uses.

    The two control knobs exist to be FAILING controls, and both are recorded in the result:

    * ``weighting="bigram-table"`` with ``control_bigram_freqs`` weights the bigram channel
      by a standalone bigram table instead of the trigram marginal. It must FAIL
      :meth:`ShapDiff.reconciles`; a version that passes means the gauge is not what this
      module thinks it is.
    * ``shuffle_seed`` permutes the per-cell SHAP-delta vectors across cells. It must also
      FAIL, which is what shows the reconciliation is testing the ATTRIBUTION and not merely
      re-adding the same numbers in a different order.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    if weighting == "bigram-table" and control_bigram_freqs is None:
        raise ValueError(
            "weighting='bigram-table' is a negative control and needs control_bigram_freqs"
        )

    if surface is None:
        surface = default_surface(target_wpm, corpus)
    geometry = surface.geometry
    if bigram_models is None:
        from keybo.analysis.timecard import _SEEDS, _load_gz_model

        bigram_models = [_load_gz_model(f"bigram_reg31_seed{s}") for s in _SEEDS]

    # Reuse the reviewed guard: it REFUSES a short or repeating layout, which would
    # otherwise be scored over a fraction of the corpus and still look plausible.
    slot_a = surface._slot_of(layout_a)
    slot_b = surface._slot_of(layout_b)
    if set(layout_a) != set(layout_b):
        # Different charsets cover different trigram subsets; the decomposition then runs on
        # the intersection and `coverage_cost` prices what that moved.
        common = "".join(sorted(set(layout_a) & set(layout_b)))
        if not common:
            raise ValueError("layouts share no characters; there is no common support to diff")
    chars = layout_a  # character-index order; both boards are indexed through it

    shap_tables, p_tables, p_predict_tables, ms_tables, resid_add, names = _bigram_shap_tables(
        bigram_models, geometry, target_wpm
    )
    n_feat = len(names)

    # --- corpus weights on the COMMON support (see module docstring) --------------------
    common_chars = "".join(c for c in chars if c in set(layout_b))
    w3, w2, covered_common = _char_weight_tables(surface, common_chars)
    # re-express on the layout_a character index (common_chars is a subsequence of it)
    if common_chars != chars:
        full = np.zeros((len(chars) + 1,) * 3)
        keep = [chars.index(c) for c in common_chars] + [len(chars)]
        full[np.ix_(keep, keep, keep)] = w3
        w3 = full
        w2 = w3.sum(axis=2)
    _, _, covered_a = _char_weight_tables(surface, layout_a)
    _, _, covered_b = _char_weight_tables(surface, layout_b)

    if weighting == "bigram-table":
        # THE NEGATIVE CONTROL: weight the bigram channel by a standalone bigram table.
        index = {c: i for i, c in enumerate(chars)}
        index[" "] = len(chars)
        w2 = np.zeros_like(w2)
        for bigram, freq in control_bigram_freqs.items():
            if len(bigram) != 2:
                continue
            try:
                w2[index[bigram[0]], index[bigram[1]]] += freq
            except KeyError:
                continue

    # --- position indices for each board, as (n_char, n_char) grids ---------------------
    perm_a = np.array([slot_a[c] for c in chars] + [slot_a[" "]], dtype=np.intp)
    perm_b = np.array([slot_b[c] for c in chars] + [slot_b[" "]], dtype=np.intp)
    rows = np.arange(len(perm_a))[:, None]
    cols = np.arange(len(perm_a))[None, :]
    ia, ja = perm_a[rows], perm_a[cols]
    ib, jb = perm_b[rows], perm_b[cols]

    # --- per-seed LMDI attribution, then the seed mean --------------------------------
    # Both steps are EXACT: LMDI closes per seed, and the production T2 is the seed mean
    # IN MILLISECONDS, which is the space the attributions live in.
    attrib = np.zeros(w2.shape + (n_feat,))
    d_ms_cells = np.zeros(w2.shape)
    log_a = np.zeros(n_feat)
    log_b = np.zeros(n_feat)
    resid_log = 0.0
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    weight_norm = w2 / max(covered_common, 1)
    for shap, p, p_predict, ms in zip(
        shap_tables, p_tables, p_predict_tables, ms_tables, strict=True
    ):
        ms_a, ms_b = ms[ia, ja], ms[ib, jb]
        d_ms = ms_b - ms_a
        # Exact division whenever the denominator is non-zero — NOT a small-|dp| fallback,
        # which would break the identity precisely where it is delicate. dp == 0 means the
        # cell did not move, so d_ms == 0 too and the chosen L multiplies zero.
        d_shap = shap[ib, jb, :] - shap[ia, ja, :]
        if rng is not None:
            # THE SHUFFLE CONTROL: permute which cell's SHAP-delta vector lands where. Applied
            # BEFORE the LMDI weight is derived, so the control breaks the ATTRIBUTION rather
            # than being silently absorbed into a rescaled weight.
            flat = d_shap.reshape(-1, n_feat)
            d_shap = flat[rng.permutation(flat.shape[0])].reshape(d_shap.shape)
        # The denominator is the SHAP-IMPLIED log delta, not predict()'s: `d_p` is defined as
        # `sum_i d_shap_i` here, so LMDI divides by exactly the quantity it multiplies back and
        # the identity closes at float64 rather than inheriting the booster's float32 noise.
        d_p = d_shap.sum(axis=2)
        lmdi = np.where(d_p != 0.0, d_ms / np.where(d_p != 0.0, d_p, 1.0), 0.5 * (ms_a + ms_b))
        attrib += lmdi[:, :, None] * d_shap / len(shap_tables)
        d_ms_cells += d_ms / len(shap_tables)
        log_a += (weight_norm[:, :, None] * shap[ia, ja, :]).sum(axis=(0, 1)) / len(shap_tables)
        log_b += (weight_norm[:, :, None] * shap[ib, jb, :]).sum(axis=(0, 1)) / len(shap_tables)
        # The log-space control, per board, comparing the TreeSHAP WALK (`p`) against the
        # INDEPENDENT ordinary prediction (`p_predict`) under the same corpus weighting. Two
        # different implementations, so this can actually fail; checking the walk against
        # itself would be a tautology.
        for idx_i, idx_j in ((ia, ja), (ib, jb)):
            walked = (weight_norm * p[idx_i, idx_j]).sum()
            predicted = (weight_norm * p_predict[idx_i, idx_j]).sum()
            resid_log = max(resid_log, abs(walked - predicted))

    weighted = weight_norm[:, :, None] * attrib
    contributions_ms = weighted.sum(axis=(0, 1))

    # --- the two channels, and the ms/char numbers ------------------------------------
    t2 = np.mean(ms_tables, axis=0)
    tc = surface._Tc
    t2_a = float((w2 * t2[ia, ja]).sum() / max(covered_common, 1))
    t2_b = float((w2 * t2[ib, jb]).sum() / max(covered_common, 1))
    tc_a = float(
        (w3 * tc[ia[:, :, None], ja[:, :, None], perm_a[None, None, :]]).sum()
        / max(covered_common, 1)
    )
    tc_b = float(
        (w3 * tc[ib[:, :, None], jb[:, :, None], perm_b[None, None, :]]).sum()
        / max(covered_common, 1)
    )

    card_a = surface.card(layout_a)
    card_b = surface.card(layout_b)
    gap_t2, gap_tcond = t2_b - t2_a, tc_b - tc_a
    gap_total = (t2_b + tc_b) - (t2_a + tc_a)

    # Own-support ms/char on THIS module's tables (so `coverage_cost` isolates the coverage
    # restriction rather than the SHAP-vs-predict anchor difference — see `coverage_cost`).
    def _own(w3_own: np.ndarray, perm: np.ndarray, covered: int) -> float:
        i, j = perm[rows], perm[cols]
        t2_part = (w3_own.sum(axis=2) * t2[i, j]).sum()
        tc_part = (w3_own * tc[i[:, :, None], j[:, :, None], perm[None, None, :]]).sum()
        return float((t2_part + tc_part) / max(covered, 1))

    w3_a, _, _ = _char_weight_tables(surface, layout_a)
    own_a = _own(w3_a, perm_a, covered_a)
    if set(layout_a) == set(layout_b):
        # Same charset: re-index the SAME trigram weights through layout_b's permutation.
        own_b = _own(w3_a, perm_b, covered_b)
    else:
        b_chars = layout_b
        w3_b_native, _, _ = _char_weight_tables(surface, b_chars)
        keep = [b_chars.index(c) for c in b_chars] + [len(b_chars)]
        perm_b_native = np.array([slot_b[c] for c in b_chars] + [slot_b[" "]], dtype=np.intp)
        own_b = _own(w3_b_native[np.ix_(keep, keep, keep)], perm_b_native, covered_b)

    def _rel(lhs: float, rhs: float) -> float:
        scale = max(abs(rhs), 1e-300)
        return abs(lhs - rhs) / scale

    resid_cell = float(
        np.abs(attrib.sum(axis=2) - d_ms_cells).max() / max(np.abs(d_ms_cells).max(), 1e-300)
    )
    # R5: the tie to what `analyze` prints. `card()` is the shipped path (predict()-anchored,
    # own-support); this module is SHAP-anchored. Comparing the GAPS rather than the levels is
    # deliberate — a gap is the quantity being decomposed, and a common-mode level offset that
    # cancels in the gap cannot invalidate the decomposition.
    resid_card = abs(gap_total - (card_b.ms_per_char - card_a.ms_per_char))
    resid_shipped_t2 = float(np.abs(t2 - surface._T2).max())
    return ShapDiff(
        name_a=name_a or layout_a,
        name_b=name_b or layout_b,
        layout_a=layout_a,
        layout_b=layout_b,
        corpus=corpus or "default",
        target_wpm=target_wpm,
        weighting=weighting,
        feature_names=names,
        ms_per_char_own_a=own_a,
        ms_per_char_own_b=own_b,
        card_ms_per_char_a=card_a.ms_per_char,
        card_ms_per_char_b=card_b.ms_per_char,
        ms_per_char_a=t2_a + tc_a,
        ms_per_char_b=t2_b + tc_b,
        gap_total=gap_total,
        gap_t2=gap_t2,
        gap_tcond=gap_tcond,
        contributions=[
            FeatureContribution(
                feature=name,
                ms_per_char=float(contributions_ms[i]),
                log_a=float(log_a[i]),
                log_b=float(log_b[i]),
                log_delta=float(log_b[i] - log_a[i]),
            )
            for i, name in enumerate(names)
        ],
        covered_mass_a=covered_a,
        covered_mass_b=covered_b,
        covered_mass_common=covered_common,
        corpus_total_mass=int(surface.total_mass),
        # float() on every residual: numpy scalars are not JSON-serializable, and `to_dict`
        # embeds these directly.
        resid_additivity=float(resid_add),
        resid_cell_lmdi=float(resid_cell),
        resid_feature_sum=float(_rel(float(contributions_ms.sum()), gap_t2)),
        resid_channel_split=float(_rel(gap_t2 + gap_tcond, gap_total)),
        resid_vs_card_gap=float(resid_card),
        resid_vs_shipped_t2=float(resid_shipped_t2),
        resid_log_vs_predict=float(resid_log),
        _weighted=weighted,
        _chars=chars + " ",
    )


def format_report(diff: ShapDiff, top_bigrams_k: int = 5) -> str:
    """The human-readable report: reconciliation FIRST, then the feature table.

    The order is deliberate — a reader who sees the ranked features before the residuals can
    form a story about a table that does not sum to anything.
    """
    lines: list[str] = []
    a, b = diff.name_a, diff.name_b
    lines.append(f"SHAP-DIFF  {a} -> {b}   corpus={diff.corpus}  wpm={diff.target_wpm:g}")
    lines.append(f"  weighting: {diff.weighting}")
    lines.append("")
    lines.append("RECONCILIATION (checked before any interpretation)")
    lines.append(
        f"  ms/char shipped card()  {a}: {diff.card_ms_per_char_a:.4f}   "
        f"{b}: {diff.card_ms_per_char_b:.4f}   <- what `keybo analyze` prints"
    )
    lines.append(
        f"  ms/char own-support     {a}: {diff.ms_per_char_own_a:.4f}   {b}: {diff.ms_per_char_own_b:.4f}"
    )
    lines.append(
        f"  ms/char common-support  {a}: {diff.ms_per_char_a:.4f}   {b}: {diff.ms_per_char_b:.4f}"
    )
    lines.append(f"  gap (b-a; +ve = {a} faster) : {diff.gap_total:+.4f} ms/char")
    lines.append(
        f"    T2 bigram channel   : {diff.gap_t2:+.4f}  ({diff.decomposed_share_pct:.1f}% of gap)  <- decomposed"
    )
    lines.append(f"    Tcond trigram channel: {diff.gap_tcond:+.4f}  <- NOT bigram-decomposable")
    lines.append(f"  coverage {a}: {diff.coverage_pct_a:.3f}%   {b}: {diff.coverage_pct_b:.3f}%")
    lines.append(
        f"  common support is a no-op: {diff.common_support_is_noop}   coverage cost: "
        f"{diff.coverage_cost:+.3e} ms/char"
    )
    lines.append("  residuals:")
    lines.append(f"    R1 TreeSHAP walk vs predict (log, abs) {diff.resid_additivity:.3e}")
    lines.append(f"    R2 per-cell LMDI identity (rel)        {diff.resid_cell_lmdi:.3e}")
    lines.append(f"    R3 sum(features) vs gap_T2 (rel)       {diff.resid_feature_sum:.3e}")
    lines.append(f"    R4 channel split vs total (rel)        {diff.resid_channel_split:.3e}")
    lines.append(f"    R5 gap vs shipped card (abs ms/char)   {diff.resid_vs_card_gap:.3e}")
    lines.append(f"    log walk-vs-predict, weighted (abs)    {diff.resid_log_vs_predict:.3e}")
    lines.append(f"    T2 table vs shipped _T2 (abs ms)       {diff.resid_vs_shipped_t2:.3e}")
    lines.append(f"  RECONCILES: {diff.reconciles()}")
    lines.append("")
    if not diff.reconciles():
        lines.append("!! RECONCILIATION FAILED — the feature table below explains NOTHING. !!")
        lines.append("")
    lines.append(f"PER-FEATURE CONTRIBUTIONS to the T2 gap ({diff.gap_t2:+.4f} ms/char)")
    lines.append(
        f"  {'feature':<16} {'ms/char':>10} {'share':>8} {'favours':>8}   {'log_a':>9} {'log_b':>9} {'d_log':>10}"
    )
    for c in diff.ranked():
        share = 100.0 * c.ms_per_char / diff.gap_t2 if diff.gap_t2 else float("nan")
        favour = {"a": a, "b": b, "tie": "-"}[c.favours]
        lines.append(
            f"  {c.feature:<16} {c.ms_per_char:>+10.4f} {share:>7.1f}% {favour:>8}   "
            f"{c.log_a:>+9.5f} {c.log_b:>+9.5f} {c.log_delta:>+10.5f}"
        )
    lines.append(f"  {'SUM':<16} {sum(c.ms_per_char for c in diff.contributions):>+10.4f}")
    if top_bigrams_k:
        lines.append("")
        lines.append("TOP BIGRAMS per leading feature (ms/char, corpus-weighted)")
        for c in diff.ranked()[:6]:
            top = diff.top_bigrams(c.feature, top_bigrams_k)
            rendered = "  ".join(f"{bg!r}{v:+.4f}" for bg, v in top)
            lines.append(f"  {c.feature:<16} {rendered}")
    return "\n".join(lines)
