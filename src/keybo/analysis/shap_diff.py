"""Per-feature attribution of the ms/char GAP between two layouts (SHAPDIFF-1, -TCOND).

``shap-report`` answers "what does the model use?" over a matrix of rows. This module
answers the different, harder question: **why is layout B slower than layout A?**, as a
signed per-feature budget in the units the analyzer publishes (ms/char) that SUMS BACK to
the measured gap.

The construction, and the places a plausible-looking version goes wrong:

**1. The channels.** ``analyze``'s gauge is ``ms/char = sum_tri f*(T2[a,b] + Tcond[a,b,c]) /
covered``, i.e. TWO models on two different frames. The gap is therefore split into
``gap_t2 + gap_tcond`` FIRST, both reported, and each channel is decomposed **on its own
frame**: the 20-column bigram frame over ``n**2`` position pairs, and the 46-column trigram
frame over ``n**3`` position triples. Attributing the whole gap to bigram features would be
wrong, and the split is what makes the over-claim impossible rather than merely discouraged.
``channel="both"`` (the default) decomposes both, which is the only setting whose decomposed
share can reach 100%.

**2. The weights, and they are NOT the same in the two channels.** ``TimeSurface.card``
iterates the TRIGRAM table. The ``Tcond`` term is indexed by all three looped characters, so
its weight is the trigram frequency ``w3`` **directly**. The ``T2`` term is indexed by only
the first two, so a character bigram's effective weight is the trigram table's
first-two-character marginal ``w2(x,y) = sum_z tri(x,y,z)`` restricted to on-board trigrams
(``triple_ms_table``'s docstring records that using ``bigrams.txt`` here is ~1.5e-2 wrong).
⚠ Reasoning by symmetry between the channels is the trap: "T2 needed a marginal, so Tcond
needs one too" is FALSE, and :data:`TCOND_WEIGHTINGS` carries ``"tcond-marginal"`` — that
marginal broadcast over the third character — as a NEGATIVE CONTROL which must fail.

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
both and labels them. This is identical in both channels, so the LMDI core is shared code
(:func:`_lmdi_channel`) parameterized by n-gram order rather than reimplemented per channel.

**4. The unit of interpretation is a BLOCK, not a column.** TreeSHAP's split of credit
across CORRELATED columns is not unique, and SHAPDIFF-1 measured the symptom: ``wpm`` is a
CONSTANT column at a fixed scoring WPM yet carried -0.0922 ms/char, which is the tree
crediting its node on interaction paths. The trigram frame makes this structurally worse —
``bg1_*`` and ``bg2_*`` are the SAME 19 placement features on two overlapping key pairs — so
:meth:`ChannelAttribution.blocks` is the PRIMARY table and the per-column list is subordinate
detail. A block sum is invariant to how credit is redistributed WITHIN the block. ⚠ It is
NOT invariant to leakage BETWEEN blocks (``bg1_``<->``bg2_`` in particular), so blocks reduce
the non-uniqueness rather than eliminating it.

**Seed averaging is exact, not approximate.** The production tables are the mean over three
boosters *in milliseconds*, i.e. ``T = mean_s exp(p_s)*K``. A mean of exponentials is not
the exponential of a mean, so a single log-space attribution against ``log(T*wpm/12000)``
would not sum correctly. Attributing per seed and averaging the resulting MS attributions
does, because both steps are exact: LMDI closes per seed, and the seed mean is linear in the
space the attributions live in.

**Common support.** Trigrams are scored only when every character is on the board, so two
layouts with different CHARSETS cover different corpus subsets and their per-board ms/char
have different denominators — a difference of two averages over two different populations is
not decomposable per cell. This module therefore decomposes the COMMON-SUPPORT gap
(trigrams typeable on both boards, the convention :mod:`keybo.analysis.layout_diff` uses)
and reports the own-support gap beside it, so the coverage cost is a number rather than an
assumption. When the two charsets are permutations of one another — as any two boards over a
fixed key set are — the two are the same by construction and the restriction is a no-op,
which :attr:`ShapDiff.common_support_is_noop` states rather than implies.

What the decomposition CANNOT see: neither frame carries a hand-identity channel, so "board B
overloads one hand" is unaskable. The bigram frame additionally carries NO direction-of-travel
channel (``inwards``/``outwards`` are swap-invariant — see :mod:`keybo.features.schema`) and
its row/finger one-hots describe the LANDING key only. The TRIGRAM frame is the exception
worth naming: ``redirect``/``bad_redirect`` compare successive keys' column magnitudes and so
ARE order-aware, which makes them the one direction-of-travel signal in the served frames.
And a SHAP attribution is an attribution ON THIS MODEL: it says what the fitted surface
prices, not what a hand does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import xgboost as xgb

from keybo.analysis.timecard import TimeSurface, default_surface
from keybo.features import (
    bigram_features_from_positions,
    interp_features_from_positions,
    interp_wpm_features_from_positions,
    trigram_features_from_positions,
)
from keybo.features.schema import (
    BIGRAM_FEATURE_NAMES,
    BIGRAM_INTERP_FEATURE_NAMES,
    BIGRAM_INTERP_WPM_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.models.xgboost_model import XGBoostTypingModel

#: T2-channel weighting conventions. ``trigram-marginal`` is the gauge's own weight (the
#: default and the only correct one); ``bigram-table`` reproduces the ~1.5e-2 error of
#: weighting by ``bigrams.txt`` and exists to be a failing negative control.
WEIGHTINGS = ("trigram-marginal", "bigram-table")

#: Tcond-channel weighting conventions, kept SEPARATE from :data:`WEIGHTINGS` because the two
#: channels have different correct weights and therefore different wrong ones.
#: ``trigram-direct`` is the gauge's own weight (``card`` indexes all three looped characters,
#: so no marginalization occurs). ``tcond-marginal`` commits the symmetry error — the
#: first-two-character marginal broadcast uniformly over the third character, i.e. what you get
#: by assuming SHAPDIFF-1's T2 marginal correction transfers — and must FAIL the external bars
#: while still passing the internal one.
TCOND_WEIGHTINGS = ("trigram-direct", "tcond-marginal")

#: Which channels can be decomposed. ``both`` is the default: a single-channel report is a
#: partial answer by construction, and SHAPDIFF-1 measured that the T2-only answer was 31.3%.
CHANNELS = ("t2", "tcond", "both")

#: Which FEATURE FRAME the T2 channel is decomposed on. ``"served"`` is the 20-column frame all
#: three shipped ``bigram_reg31`` artifacts carry — the default, and the only frame with a
#: production model behind it. ``"interp"`` is INTERPFRAME-1's 10-column interpretability basis,
#: which exists to be COMPARED against ``"served"`` on the SAME layout pair; it requires
#: ``bigram_models=`` explicitly, because no shipped artifact carries it.
#:
#: ⚠ The frame changes what the ATTRIBUTION is expressed in, NOT what the gauge is. A run at
#: ``frame="interp"`` decomposes the interp model's OWN T2 gap and its shipped-table ties are
#: therefore to THAT model's table — not to ``TimeSurface._T2``. Comparing the two frames'
#: attributions is comparing two explanations of two (nearly identical) surfaces, and the
#: ``resid_gap_vs_shipped`` / ``resid_table_vs_shipped`` residuals are what price "nearly".
#: ``"interp-wpm"`` is the 11-column pace-adapting variant: the same ten mechanistic columns with
#: ``wpm`` restored. It CANNOT reach CONSTFRAC == 0 (``wpm`` is constant on any fixed-WPM serve
#: grid, so TreeSHAP can credit it again) and exists to price that trade against the high-wpm gate.
FRAMES = ("served", "interp", "interp-wpm")

#: Column -> (block, sub-block) for the served BIGRAM frame. Blocks are the primary reporting
#: unit; the bigram frame's blocks are atomic (no second level) because they are already only
#: 1-6 columns wide.
_T2_BLOCKS: dict[str, tuple[str, str]] = {
    **{n: ("ROW", "") for n in ("bottom", "home", "top")},
    **{n: ("FINGER", "") for n in ("pinky", "ring", "middle", "index", "lateral")},
    **{n: ("RELATIONAL", "") for n in ("same_hand", "same_finger", "adjacent", "scissor", "lsb")},
    **{n: ("GEOMETRY", "") for n in ("dx", "dy", "distance", "angle", "inwards", "outwards")},
    "wpm": ("WPM", ""),
}

#: Sub-block partition applied INSIDE each of ``BG1``/``BG2``, so the two 19-column blocks are
#: directly comparable term by term.
_PLACEMENT_SUBBLOCK = {
    **{n: "row" for n in ("bottom", "home", "top")},
    **{n: "finger" for n in ("pinky", "ring", "middle", "index", "lateral")},
    **{n: "relational" for n in ("same_hand", "same_finger", "adjacent", "scissor", "lsb")},
    **{n: "geometry" for n in ("dx", "dy", "distance", "angle", "inwards", "outwards")},
}

#: Column -> (block, sub-block) for the served TRIGRAM frame, exactly as registered in the
#: SHAPDIFF-TCOND prereg §1 BEFORE any decomposition number existed.
_TCOND_BLOCKS: dict[str, tuple[str, str]] = {
    **{n: ("TRI_LEVEL", "") for n in ("same_hand_trigram", "redirect", "bad_redirect")},
    **{n: ("SKIPGRAM", "") for n in ("sg_same_finger", "sg_dx", "sg_dy", "sg_distance")},
    **{f"bg1_{n}": ("BG1", sub) for n, sub in _PLACEMENT_SUBBLOCK.items()},
    **{f"bg2_{n}": ("BG2", sub) for n, sub in _PLACEMENT_SUBBLOCK.items()},
    "wpm": ("WPM", ""),
}

#: Column -> (block, sub-block) for the INTERPRETABILITY frame (INTERPFRAME-1), registered here
#: because :func:`block_map` REFUSES an unregistered frame — this dict is the integration point.
#:
#: ⚠ The blocks are named for the MECHANISM, not for the served frame's blocks, and every one is
#: 1-3 columns wide. That is the point of the frame: the served frame needed blocks because credit
#: could not be trusted at column level, and the wider a block is the more it hides. A 10-column
#: frame in five blocks of 1-3 makes the BLOCK table and the COLUMN table nearly the same claim,
#: which is what "the per-feature number means what it says" cashes out to.
#:
#: There is NO ``WPM`` block, because there is no ``wpm`` column — the constant-column artifact
#: this frame exists to remove (see :data:`keybo.features.schema.BIGRAM_INTERP_FEATURE_NAMES`).
_INTERP_BLOCKS: dict[str, tuple[str, str]] = {
    **{n: ("CONTACT", "") for n in ("hand_conflict", "finger_load", "off_home_column")},
    **{n: ("SPAN", "") for n in ("row_span", "lateral_span", "same_hand_travel")},
    **{n: ("ROWCOST", "") for n in ("row_load", "row_arrival", "bottom_bias")},
    "roll_inward": ("DIRECTION", ""),
}

#: The pace-adapting variant's partition: the same four blocks plus ``WPM``, which is kept as its
#: OWN one-column block precisely so its (artifactual) credit is never mixed into a mechanism block.
_INTERP_WPM_BLOCKS: dict[str, tuple[str, str]] = {**_INTERP_BLOCKS, "wpm": ("WPM", "")}

#: Tie-break order for equal-magnitude blocks, so a report reads the same way every run.
_BLOCK_ORDER = (
    "TRI_LEVEL",
    "SKIPGRAM",
    "BG1",
    "BG2",
    "ROW",
    "FINGER",
    "RELATIONAL",
    "GEOMETRY",
    # the INTERPFRAME-1 blocks
    "CONTACT",
    "SPAN",
    "ROWCOST",
    "DIRECTION",
    "WPM",
)


def block_map(feature_names: Sequence[str]) -> dict[str, tuple[str, str]]:
    """``column -> (block, sub-block)`` for a served frame, REFUSING an unknown frame.

    Refusing rather than bucketing the remainder into an ``OTHER`` block is deliberate: an
    unrecognised frame would otherwise be reported with a silently incomplete primary table
    while every identity still closed (the block sums are taken over whatever columns the map
    knows). A widened frame must be taught to this map on purpose.
    """
    names = list(feature_names)
    for spec in (_T2_BLOCKS, _TCOND_BLOCKS, _INTERP_BLOCKS, _INTERP_WPM_BLOCKS):
        if set(names) == set(spec):
            return {n: spec[n] for n in names}
    raise ValueError(
        f"no block partition registered for this {len(names)}-column frame; add one to "
        "keybo.analysis.shap_diff rather than reporting per-column only (SHAP's credit split "
        "across correlated columns is not unique, so blocks are the reportable unit)"
    )


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed share of the A->B gap in ONE channel.

    ``ms_per_char`` is the LMDI attribution and is the headline: it is denominated in the
    same ms/char the analyzer publishes, and the contributions SUM to the channel's gap. Its
    sign follows the gap's — POSITIVE means this feature makes ``layout_b`` slower than
    ``layout_a`` (i.e. it favours A).

    ``log_a`` / ``log_b`` are the frequency-weighted mean SHAP value of this feature on each
    board in the model's own log space. Unlike the ms column they are per-BOARD quantities
    (no pair-specific weight enters), so they are the honest answer to "what does this
    feature cost on board X"; ``log_delta`` is their difference. The two columns are
    monotonically related but NOT proportional, because LMDI reweights each cell by its own
    local ms-per-log slope.
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


@dataclass(frozen=True)
class BlockContribution:
    """A block of columns and its signed share — the PRIMARY unit of interpretation.

    ``parts`` is the within-block breakdown: the sub-blocks for ``BG1``/``BG2``, and the bare
    columns for blocks narrow enough to read directly. It is a partition of the block, so its
    values sum to :attr:`ms_per_char` — it lets a reader see what a block is made of without
    falling back to a per-column table whose split is not unique.
    """

    block: str
    ms_per_char: float
    columns: tuple[str, ...]
    parts: tuple[tuple[str, float], ...]

    @property
    def favours(self) -> str:
        if self.ms_per_char > 0:
            return "a"
        return "b" if self.ms_per_char < 0 else "tie"


@dataclass
class ChannelAttribution:
    """One channel's decomposition, with every within-channel residual as a number.

    Two residual FAMILIES, and the distinction is the whole methodological point SHAPDIFF-1
    established: :attr:`resid_cell_lmdi` / :attr:`resid_feature_sum` are INTERNAL
    self-consistency (they validate the arithmetic), while :attr:`resid_gap_vs_shipped` and
    :attr:`resid_table_vs_shipped` are EXTERNAL ties to the independently-shipped table (they
    validate the CHOICE OF QUANTITY). A wrong corpus weighting passes the first family at
    ~1e-16 and fails the second; a shuffled attribution does the reverse. Neither family alone
    is sufficient, which is why :meth:`reconciles` requires both.
    """

    channel: str
    weighting: str
    feature_names: list[str]
    #: What the contributions sum to: this channel's gap on the SHAP-anchored table, under the
    #: weighting IN USE. Compared against :attr:`shipped_gap` (which is always computed at the
    #: CORRECT weighting) by :attr:`resid_gap_vs_shipped`.
    gap: float
    #: The same channel gap from the SHIPPED table (``predict``-anchored) under the CORRECT
    #: weighting — an immovable external reference a control cannot move with it.
    shipped_gap: float

    contributions: list[FeatureContribution]

    resid_cell_lmdi: float  # INTERNAL: per-cell  sum_i attrib_i  vs  ms_B - ms_A   (rel)
    resid_feature_sum: float  # INTERNAL: sum_i contribution_i  vs  gap             (rel)
    resid_gap_vs_shipped: float  # EXTERNAL: gap vs shipped_gap          (abs ms/char)
    resid_table_vs_shipped: float  # EXTERNAL: max abs ms deviation of the whole table
    resid_additivity: float  # TreeSHAP walk vs predict(), per row, log space  (abs)
    resid_log_vs_predict: float  # the same comparison under the corpus weighting (abs)

    #: ``(n_char,)*order + (n_feature,)`` weighted ms/char attribution per character n-gram,
    #: retained so :meth:`top_ngrams` can name WHICH n-grams drive a feature.
    _weighted: np.ndarray = field(repr=False)
    _chars: str = field(repr=False)

    @property
    def order(self) -> int:
        """2 for the bigram channel, 3 for the conditioned-trigram channel."""
        return self._weighted.ndim - 1

    def reconciles(
        self, rel_tol: float = 1e-9, add_tol: float = 1e-5, gauge_tol: float = 1e-3
    ) -> bool:
        """True iff this channel's bars hold — BOTH the internal and the external family.

        See :meth:`ShapDiff.reconciles` for why there are three different tolerances.
        """
        return bool(
            self.resid_cell_lmdi <= rel_tol
            and self.resid_feature_sum <= rel_tol
            and self.resid_gap_vs_shipped <= gauge_tol
            and self.resid_additivity <= add_tol
            and self.resid_log_vs_predict <= add_tol
        )

    def ranked(self) -> list[FeatureContribution]:
        """Contributions sorted by |ms/char| descending — the SUBORDINATE table."""
        return sorted(self.contributions, key=lambda c: -abs(c.ms_per_char))

    def blocks(self) -> list[BlockContribution]:
        """Block contributions, largest |ms/char| first — the PRIMARY table.

        Sums over a registered partition of the frame, so every column lands in exactly one
        block and the block sums add to the same channel gap the columns do.
        """
        spec = block_map(self.feature_names)
        by_feature = {c.feature: c.ms_per_char for c in self.contributions}
        grouped: dict[str, list[str]] = {}
        for name in self.feature_names:
            grouped.setdefault(spec[name][0], []).append(name)
        out = []
        for block, columns in grouped.items():
            # sub-blocks where the partition defines them, else the bare columns: either way
            # `parts` is a partition of the block, so its values sum to the block's total.
            subs: dict[str, float] = {}
            for name in columns:
                key = spec[name][1] or name
                subs[key] = subs.get(key, 0.0) + by_feature[name]
            out.append(
                BlockContribution(
                    block=block,
                    ms_per_char=sum(by_feature[n] for n in columns),
                    columns=tuple(columns),
                    parts=tuple(sorted(subs.items(), key=lambda kv: -abs(kv[1]))),
                )
            )
        out.sort(key=lambda b: (-abs(b.ms_per_char), _BLOCK_ORDER.index(b.block)))
        return out

    def top_ngrams(self, feature: str, k: int = 8) -> list[tuple[str, float]]:
        """``(ngram, ms/char)`` pairs driving ``feature``, largest |contribution| first.

        Turns "``bg2_bottom`` explains 1.2 ms/char" into a statement about the n-grams that
        produced it. The n-gram is written in the CHARACTERS of the corpus (space rendered as
        ``␣``), and its value already carries the corpus weight, so the values for one feature
        sum to that feature's :attr:`FeatureContribution.ms_per_char`.
        """
        col = self.feature_names.index(feature)
        block = self._weighted[..., col]
        flat = np.argsort(-np.abs(block), axis=None)[:k]
        out = []
        for pos in flat:
            idx = np.unravel_index(pos, block.shape)
            value = float(block[idx])
            if value == 0.0:
                continue
            out.append(("".join(self._chars[i] for i in idx).replace(" ", "␣"), value))
        return out

    def to_dict(self, top_ngrams_k: int = 8) -> dict:
        spec = block_map(self.feature_names)
        return {
            "channel": self.channel,
            "weighting": self.weighting,
            "gap_decomposed": self.gap,
            "shipped_gap": self.shipped_gap,
            "reconciles": self.reconciles(),
            "residuals": {
                "cell_lmdi_rel": self.resid_cell_lmdi,
                "feature_sum_vs_gap_rel": self.resid_feature_sum,
                "gap_vs_shipped_abs_ms_per_char": self.resid_gap_vs_shipped,
                "table_vs_shipped_abs_ms": self.resid_table_vs_shipped,
                "additivity_log_abs": self.resid_additivity,
                "log_walk_vs_predict_abs": self.resid_log_vs_predict,
            },
            "blocks": [
                {
                    "block": b.block,
                    "ms_per_char": b.ms_per_char,
                    "share_of_channel_pct": (
                        100.0 * b.ms_per_char / self.gap if self.gap else None
                    ),
                    "favours": b.favours,
                    "columns": list(b.columns),
                    "parts": [{"part": p, "ms_per_char": v} for p, v in b.parts],
                }
                for b in self.blocks()
            ],
            "contributions": [
                {
                    "feature": c.feature,
                    "block": spec[c.feature][0],
                    "ms_per_char": c.ms_per_char,
                    "favours": c.favours,
                    "share_of_channel_pct": (
                        100.0 * c.ms_per_char / self.gap if self.gap else None
                    ),
                    "log_a": c.log_a,
                    "log_b": c.log_b,
                    "log_delta": c.log_delta,
                    "top_ngrams": [
                        {"ngram": ng, "ms_per_char": v}
                        for ng, v in self.top_ngrams(c.feature, top_ngrams_k)
                    ],
                }
                for c in self.ranked()
            ],
        }


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
    channel: str

    #: Each board's ms/char over its OWN typable corpus subset, computed on THIS module's
    #: tables. Matches ``TimeSurface.card().ms_per_char`` (what ``keybo analyze`` prints) to
    #: the float32 booster noise measured in :attr:`resid_vs_card_gap`.
    ms_per_char_own_a: float
    ms_per_char_own_b: float
    #: The shipped ``TimeSurface.card()`` numbers verbatim, for the external comparison.
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

    #: Per-channel decompositions. ``None`` for a channel this run did not decompose.
    t2: ChannelAttribution | None
    tcond: ChannelAttribution | None

    covered_mass_a: int
    covered_mass_b: int
    covered_mass_common: int
    corpus_total_mass: int

    resid_channel_split: float  # gap_t2 + gap_tcond  vs  gap_total                 (rel)
    #: This module's own ms/char gap vs ``TimeSurface.card``'s — the tie to what ``analyze``
    #: prints, in ABSOLUTE ms/char. Not a float64 identity: the SHAP-anchored tables differ
    #: from the shipped ``predict``-anchored ones by the booster's float32 noise, so this is a
    #: SMALLNESS check, not an exactness one.
    resid_vs_card_gap: float

    #: Which feature frame the T2 channel was decomposed on (:data:`FRAMES`). ``"served"`` on
    #: every production run. Defaulted so no existing caller's constructor call changes.
    frame: str = "served"

    # --- the reconciliation gate -------------------------------------------------------

    @property
    def card_tie_applies(self) -> bool:
        """Whether :attr:`resid_vs_card_gap` is a VALID bar for this run, stated not assumed.

        ``card()`` is the SHIPPED surface. A non-``"served"`` frame is attributed on a model that
        is not the shipped one, so its T2 table differs from production's by a MODEL difference —
        millisecond-scale, not the float32 noise ``gauge_tol`` is sized for. Gating on the card
        tie there would fail the run for a reason that has nothing to do with the attribution,
        and silently dropping the bar would hide that a bar was dropped. So this names the
        condition, :meth:`reconciles` reads it, and :attr:`resid_vs_card_gap` is reported EITHER
        WAY — on an ``"interp"`` run it is a legitimate quantity in its own right: how far this
        POC model's surface sits from the production gauge.
        """
        return self.frame == "served"

    def reconciles(
        self, rel_tol: float = 1e-9, add_tol: float = 1e-5, gauge_tol: float = 1e-3
    ) -> bool:
        """True iff every identity holds at the registered bars.

        Three tolerances because three DIFFERENT kinds of quantity are being checked, and
        collapsing them would either hide a real bug or fail on irreducible artifact noise:

        * ``rel_tol`` — the ms-space identities. These are exact algebra and land at float64
          rounding; they are the bars that catch an attribution bug.
        * ``add_tol`` — the log-space cross-checks. The boosters are float32, so two
          independent xgboost code paths agree to ~1e-6 and never better. Holding these to
          ``rel_tol`` would fail on the artifact, not on this code.
        * ``gauge_tol`` — the ties to the shipped gauge, in ABSOLUTE ms/char. Same float32
          origin, but they ride on a ~255 ms/char level, so they are expressed absolutely.

        The per-channel EXTERNAL bar (``resid_gap_vs_shipped``) is checked on every frame, so a
        non-served run is NOT unbarred: it still has to reproduce its own model's
        ``predict``-side gap through an independent code path. Only the tie to ``card()``, which
        is a claim about the SHIPPED surface, is scoped by :attr:`card_tie_applies`.
        """
        # bool(), not the bare `and` chain: several residuals are numpy scalars, so the chain
        # returns np.bool_ — which is falsy-correct but NOT JSON-serializable, and `to_dict`
        # embeds this value. Caught by the CLI's own JSON round-trip test.
        return bool(
            self.resid_channel_split <= rel_tol
            and (self.resid_vs_card_gap <= gauge_tol or not self.card_tie_applies)
            and all(
                ch.reconciles(rel_tol, add_tol, gauge_tol)
                for ch in (self.t2, self.tcond)
                if ch is not None
            )
        )

    # --- SHAPDIFF-1 compatibility: the T2 channel read directly off the result ----------
    # These predate the Tcond channel and remain the T2 channel's, rather than being
    # redefined to mean "whichever channel is present": a caller written against the bigram
    # tool must not silently start reading trigram numbers.

    def _t2(self) -> ChannelAttribution:
        if self.t2 is None:
            raise ValueError(
                f"this run decomposed channel={self.channel!r}, so there is no T2 attribution "
                "to read; use .tcond, or re-run with channel='t2' or 'both'"
            )
        return self.t2

    @property
    def feature_names(self) -> list[str]:
        return self._t2().feature_names

    @property
    def weighting(self) -> str:
        return self._t2().weighting

    @property
    def contributions(self) -> list[FeatureContribution]:
        return self._t2().contributions

    @property
    def resid_cell_lmdi(self) -> float:
        return self._t2().resid_cell_lmdi

    @property
    def resid_feature_sum(self) -> float:
        return self._t2().resid_feature_sum

    @property
    def resid_additivity(self) -> float:
        return self._t2().resid_additivity

    @property
    def resid_log_vs_predict(self) -> float:
        return self._t2().resid_log_vs_predict

    @property
    def resid_vs_shipped_t2(self) -> float:
        return self._t2().resid_table_vs_shipped

    def ranked(self) -> list[FeatureContribution]:
        return self._t2().ranked()

    def top_bigrams(self, feature: str, k: int = 8) -> list[tuple[str, float]]:
        return self._t2().top_ngrams(feature, k)

    # --- shared views -----------------------------------------------------------------

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
        per cell.
        """
        return (self.ms_per_char_own_b - self.ms_per_char_own_a) - self.gap_total

    @property
    def coverage_pct_a(self) -> float:
        return 100.0 * self.covered_mass_a / max(self.corpus_total_mass, 1)

    @property
    def coverage_pct_b(self) -> float:
        return 100.0 * self.covered_mass_b / max(self.corpus_total_mass, 1)

    @property
    def decomposed_gap(self) -> float:
        """The part of ``gap_total`` this run's channels actually decompose."""
        return sum(ch.gap for ch in (self.t2, self.tcond) if ch is not None)

    @property
    def decomposed_share_pct(self) -> float:
        """Share of the total gap this run's channels account for, in percent.

        The number that keeps the headline honest. It can exceed 100% or be negative — that
        happens when the two channels DISAGREE in sign, which is a finding about the pair,
        not an error.
        """
        return 100.0 * self.decomposed_gap / self.gap_total if self.gap_total else float("nan")

    @property
    def undecomposed_ms_per_char(self) -> float:
        """The part of ``gap_total`` no channel in this run accounts for.

        Named as a first-class quantity rather than left to the reader's subtraction: a
        channel-restricted run is a PARTIAL answer, and SHAPDIFF-1 measured that the T2-only
        answer on this pair covered 31.3% of the gap.
        """
        return self.gap_total - self.decomposed_gap

    def to_dict(self, top_ngrams_k: int = 8) -> dict:
        payload: dict = {
            "layout_a": {"name": self.name_a, "layout": self.layout_a},
            "layout_b": {"name": self.name_b, "layout": self.layout_b},
            "corpus": self.corpus,
            "target_wpm": self.target_wpm,
            "channel": self.channel,
            "frame": self.frame,
            "weighting": self.t2.weighting if self.t2 is not None else None,
            "ms_per_char": {
                "own_support": {"a": self.ms_per_char_own_a, "b": self.ms_per_char_own_b},
                "common_support": {"a": self.ms_per_char_a, "b": self.ms_per_char_b},
                "shipped_card": {"a": self.card_ms_per_char_a, "b": self.card_ms_per_char_b},
            },
            "gap": {
                "total": self.gap_total,
                "t2_bigram_channel": self.gap_t2,
                "tcond_trigram_channel": self.gap_tcond,
                "decomposed_ms_per_char": self.decomposed_gap,
                "decomposed_share_pct": self.decomposed_share_pct,
                "undecomposed_ms_per_char": self.undecomposed_ms_per_char,
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
                "channel_split_rel": self.resid_channel_split,
                "gap_vs_shipped_card_abs_ms_per_char": self.resid_vs_card_gap,
                # Named so a reader of the JSON can see the bar was SCOPED rather than silently
                # dropped: on a non-served frame the card() number above is a model DIFFERENCE,
                # not an attribution residual (see `card_tie_applies`).
                "card_tie_applies": self.card_tie_applies,
                "reconciles": self.reconciles(),
            },
            "channels": {
                name: ch.to_dict(top_ngrams_k)
                for name, ch in (("t2", self.t2), ("tcond", self.tcond))
                if ch is not None
            },
        }
        if self.t2 is not None:
            # SHAPDIFF-1's artifact shape, kept so its consumers keep working: the top-level
            # residual and contribution keys are the T2 channel's, as they always were.
            t2 = self.t2
            payload["residuals"].update(
                {
                    "additivity_log_abs": t2.resid_additivity,
                    "log_walk_vs_predict_abs": t2.resid_log_vs_predict,
                    "cell_lmdi_rel": t2.resid_cell_lmdi,
                    "feature_sum_vs_gap_t2_rel": t2.resid_feature_sum,
                    "t2_table_vs_shipped_abs_ms": t2.resid_table_vs_shipped,
                }
            )
            payload["contributions"] = [
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
                        for bg, v in t2.top_ngrams(c.feature, top_ngrams_k)
                    ],
                }
                for c in t2.ranked()
            ]
        return payload


# --- the TreeSHAP tables ----------------------------------------------------------------


@lru_cache(maxsize=4)
def default_models(kind: str) -> tuple[XGBoostTypingModel, ...]:
    """The three seeded production artifacts for one channel (``"bigram"``/``"trigram"``).

    Cached for the process because the TreeSHAP table cache below is keyed on these objects'
    identity, and because a trigram ``pred_contribs`` walk over 29,791 rows x 427 trees is
    ~16 s per seed — reloading the models per call would recompute all of it.
    """
    from keybo.analysis.timecard import _SEEDS, _load_gz_model

    stem = {"bigram": "bigram_reg31", "trigram": "trigram_cond31"}[kind]
    return tuple(_load_gz_model(f"{stem}_seed{s}") for s in _SEEDS)


#: TreeSHAP table cache. Keyed on the geometry's VALUES (``Geometry`` holds a dict and is
#: unhashable), the scoring WPM, the n-gram order, and the identity of the model objects — with
#: the models tuple stored in the VALUE, so the ids the key uses cannot be recycled by GC while
#: the entry is still live.
_TABLE_CACHE: dict[tuple, tuple] = {}


def _geometry_key(geometry) -> tuple:
    return (
        tuple(geometry.slots),
        geometry.space_position,
        tuple(sorted(geometry.row_offsets.items())),
    )


def _shap_tables(
    models: Sequence[XGBoostTypingModel],
    geometry,
    target_wpm: float,
    order: int,
    frame: str = "served",
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], float, list[str]
]:
    """Per-seed ``(shap, p, p_predict, ms)`` position-tuple tables from the exact TreeSHAP path.

    Returns ``(shap_tables, p_tables, p_predict_tables, ms_tables, worst_additivity,
    feature_names)`` with shapes ``(n_pos,)*order + (n_feat,)`` and ``(n_pos,)*order`` x 3.

    ``frame`` selects which FEATURIZER builds the serve grid and, with it, which schema list the
    models' ``feature_names`` are asserted against — see :data:`FRAMES`. It defaults to
    ``"served"`` so every existing caller is byte-unaffected, and it is part of the table cache
    key: two frames produce different matrices from the same ``(geometry, wpm, order)``, so
    sharing a cache entry between them would silently serve one frame's SHAP table for the other.

    ``p`` is the TreeSHAP walk's own total (``base + sum_i shap_i``) and is the ANCHOR for the
    ms conversion; ``p_predict`` is the ordinary prediction. Both are returned because they are
    two INDEPENDENT xgboost code paths, and comparing them is the only non-tautological
    additivity check available — a check of ``base + sum(shap)`` against a ``p`` *defined* as
    ``base + sum(shap)`` can never fail and would be a degenerate control (SHAPDIFF-1's first
    draft did exactly that, printed EXACTLY 0.000e+00, and only its residual LIST caught it).

    Contributions are cast to **float64 on arrival**. XGBoost returns them as float32, and a
    float32 sum over 20 columns carries ~1e-7 — a hundred times the reconciliation bar and
    indistinguishable from a real attribution bug; over the trigram frame's 46 columns it is
    worse. The cast is lossless (every float32 is a float64) and moves every subsequent sum
    into float64.

    Every model is CHECKED, not assumed, for the three properties the identity needs: the
    expected frame FOR ITS ORDER AND ``frame`` (checked against the schema, not merely across the
    models), LOGRAT output, and NO first-finger calibration. The last matters most and is invisible:
    ``TableBigramScorer`` applies calibration deltas as a per-POSITION multiplicative factor
    OUTSIDE the feature path, so a calibrated model's table would not equal ``exp(prediction)``
    and the ms attribution would silently stop summing to the gauge. None of the six shipped
    k31 artifacts carries any.
    """
    key = (order, target_wpm, _geometry_key(geometry), frame, tuple(id(m) for m in models))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key][1]

    positions = [*geometry.slots, geometry.space_position]
    n = len(positions)
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    if frame in ("interp", "interp-wpm"):
        # INTERPFRAME-1's 10-column basis, or its 11-column pace-adapting variant. Order 2 only:
        # both are re-expressions of the BIGRAM columns with no trigram counterpart, so an order-3
        # request would otherwise featurize with `interp` and assert against the served trigram
        # list — i.e. fail confusingly instead of stating what is missing.
        if order != 2:
            raise ValueError(
                f"frame={frame!r} is a bigram-only frame (INTERPFRAME-1 is a POC on the T2 "
                f"channel); got order={order}"
            )
        builder = (
            interp_wpm_features_from_positions
            if frame == "interp-wpm"
            else interp_features_from_positions
        )
        rows = [
            builder(geometry, (a, b), wpm=target_wpm) for a in positions for b in positions
        ]
        expected = list(
            BIGRAM_INTERP_WPM_FEATURE_NAMES
            if frame == "interp-wpm"
            else BIGRAM_INTERP_FEATURE_NAMES
        )
    elif order == 2:
        rows = [
            bigram_features_from_positions(geometry, (a, b), wpm=target_wpm)
            for a in positions
            for b in positions
        ]
        expected = list(BIGRAM_FEATURE_NAMES)
    elif order == 3:
        rows = [
            trigram_features_from_positions(geometry, (a, b, c), wpm=target_wpm)
            for a in positions
            for b in positions
            for c in positions
        ]
        expected = list(TRIGRAM_FEATURE_NAMES)
    else:
        raise ValueError(f"order must be 2 or 3, got {order}")
    X = np.vstack(rows)
    names = list(models[0].metadata.feature_names)
    # Asserted against the SCHEMA, not merely across the models: three models agreeing with
    # each other on the WRONG frame would pass a mutual check, and every downstream lookup is
    # by column NAME — so the report would attribute to the wrong feature while reconciling.
    # This is also the guard that makes a FRAME-SWAP loud: an interp model handed frame="served"
    # (or vice versa) fails HERE rather than being scored on a matrix it was never fitted for.
    if names != expected:
        raise ValueError(
            f"order-{order} models do not carry the {frame!r} frame: expected {len(expected)} "
            f"columns {expected[:3]}..., got {len(names)} {names[:3]}..."
        )
    dmat = xgb.DMatrix(X)

    shap_tables, p_tables, p_predict_tables, ms_tables = [], [], [], []
    worst = 0.0
    for model in models:
        if list(model.metadata.feature_names) != names:
            raise ValueError(f"order-{order} models disagree on their feature frame")
        if model.target_space != "LOGRAT":
            raise ValueError(
                f"shap_diff needs LOGRAT models (got {model.target_space}); the LMDI ms "
                "conversion is derived from ms = exp(p)*12000/wpm"
            )
        if order == 3:
            # The shipped guard, reused so the trigram path has ONE source of truth for this
            # refusal rather than a second hand-rolled copy of the same predicate.
            from keybo.models.base import reject_calibrated_trigram_model

            reject_calibrated_trigram_model(model, "shap_diff")
        training = (model.metadata.extra.get("training") or {}) if model.metadata.extra else {}
        cal = training.get("calibration")
        if cal and cal.get("deltas_ms"):
            raise NotImplementedError(
                "shap_diff cannot attribute a model carrying first-finger calibration deltas: "
                "the deltas are a per-POSITION offset outside the feature path, so the SHAP "
                "contributions would not sum to the served table"
            )
        contribs = np.asarray(
            model._regressor.get_booster().predict(dmat, pred_contribs=True), dtype=np.float64
        )
        shap, base = contribs[:, :-1], contribs[:, -1]
        # This measures the two INDEPENDENT xgboost code paths against each other: the TreeSHAP
        # walk and the ordinary prediction. They agree to float32 booster precision (~1e-6),
        # never to float64 — the disagreement is the artifact's own noise, not ours.
        p_predict = model.predict(X)
        p = shap.sum(axis=1) + base
        worst = max(worst, float(np.abs(p - p_predict).max()))
        # The SHAP-implied prediction is the ANCHOR for everything downstream, deliberately:
        # the LMDI weight must divide by exactly the quantity it later multiplies back, or the
        # identity inherits that ~1e-6 as a FLOOR. Anchoring on predict() instead measured
        # 9.5e-07 on the bigram channel; anchoring here measures ~1e-16.
        ms = np.exp(p) * 12000.0 / target_wpm
        cell = (n,) * order
        shap_tables.append(shap.reshape(*cell, len(names)))
        p_tables.append(p.reshape(cell))
        p_predict_tables.append(p_predict.reshape(cell))
        ms_tables.append(ms.reshape(cell))
    out = (shap_tables, p_tables, p_predict_tables, ms_tables, worst, names)
    _TABLE_CACHE[key] = (tuple(models), out)
    return out


def _bigram_shap_tables(models, geometry, target_wpm: float):
    """SHAPDIFF-1's entry point, retained: :func:`_shap_tables` at ``order=2``."""
    return _shap_tables(models, geometry, target_wpm, 2)


def _char_weight_tables(surface: TimeSurface, chars: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Corpus weights in CHARACTER space over trigrams typeable on ``chars`` + space.

    Returns ``(w3, w2, covered)`` where ``w3[i,j,k]`` is the trigram frequency — the weight the
    ``Tcond`` channel uses DIRECTLY — and ``w2`` is its first-two-character marginal, which is
    the weight the ``T2`` channel uses. ``w2`` is derived from ``w3`` rather than loaded
    separately, so the bigram weight cannot drift from the trigram weight that produced it (the
    ~1.5e-2 ``bigrams.txt`` trap). ``covered`` is the summed mass, i.e. the ms/char denominator
    for BOTH channels — the gauge divides one total by one ``covered``.
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


# --- the shared LMDI core ---------------------------------------------------------------


def _lmdi_channel(
    tables: tuple,
    perm_a: np.ndarray,
    perm_b: np.ndarray,
    weight: np.ndarray,
    covered: int,
    order: int,
    rng: np.random.Generator | None,
) -> tuple:
    """The LMDI attribution for ONE channel — the code both channels share.

    Returns ``(weighted, ms_mean_a, ms_mean_b, log_a, log_b, resid_log, attrib, d_ms_cells)``.
    ``weighted`` is the corpus-weighted ms/char attribution per ``(cell, feature)``; ``attrib``
    and ``d_ms_cells`` are the UNWEIGHTED per-cell quantities the caller reconciles against.

    Written once and parameterized by ``order`` rather than duplicated per channel: the
    identity, the exact-division rule, the shuffle control and the anchoring discipline are the
    delicate parts, and two copies of them would drift apart under maintenance.
    """
    shap_tables, p_tables, p_predict_tables, ms_tables = tables[:4]
    n_feat = shap_tables[0].shape[-1]
    idx_a = np.ix_(*(perm_a,) * order)
    idx_b = np.ix_(*(perm_b,) * order)

    attrib = np.zeros(weight.shape + (n_feat,))
    d_ms_cells = np.zeros(weight.shape)
    ms_mean_a = np.zeros(weight.shape)
    ms_mean_b = np.zeros(weight.shape)
    log_a = np.zeros(n_feat)
    log_b = np.zeros(n_feat)
    resid_log = 0.0
    weight_norm = weight / max(covered, 1)
    n_seed = len(shap_tables)

    for shap, p, p_predict, ms in zip(
        shap_tables, p_tables, p_predict_tables, ms_tables, strict=True
    ):
        ms_a, ms_b = ms[idx_a], ms[idx_b]
        d_ms = ms_b - ms_a
        d_shap = shap[idx_b] - shap[idx_a]
        if rng is not None:
            # THE SHUFFLE CONTROL: permute which cell's SHAP-delta vector lands where. Applied
            # BEFORE the LMDI weight is derived, so the control breaks the ATTRIBUTION rather
            # than being silently absorbed into a rescaled weight.
            flat = d_shap.reshape(-1, n_feat)
            d_shap = flat[rng.permutation(flat.shape[0])].reshape(d_shap.shape)
        # The denominator is the SHAP-IMPLIED log delta, not predict()'s: `d_p` is defined as
        # `sum_i d_shap_i` here, so LMDI divides by exactly the quantity it multiplies back and
        # the identity closes at float64 instead of inheriting the booster's float32 noise.
        # Exact division whenever the denominator is non-zero — NOT a small-|dp| fallback,
        # which would break the identity precisely where it is delicate. dp == 0 means the cell
        # did not move, so d_ms == 0 too and the chosen L multiplies zero.
        d_p = d_shap.sum(axis=-1)
        lmdi = np.where(d_p != 0.0, d_ms / np.where(d_p != 0.0, d_p, 1.0), 0.5 * (ms_a + ms_b))
        attrib += lmdi[..., None] * d_shap / n_seed
        d_ms_cells += d_ms / n_seed
        ms_mean_a += ms_a / n_seed
        ms_mean_b += ms_b / n_seed
        log_a += np.tensordot(weight_norm, shap[idx_a], axes=order) / n_seed
        log_b += np.tensordot(weight_norm, shap[idx_b], axes=order) / n_seed
        # The log-space control, per board, comparing the TreeSHAP WALK (`p`) against the
        # INDEPENDENT ordinary prediction (`p_predict`) under the same corpus weighting. Two
        # different implementations, so this can actually fail.
        for idx in (idx_a, idx_b):
            walked = (weight_norm * p[idx]).sum()
            predicted = (weight_norm * p_predict[idx]).sum()
            resid_log = max(resid_log, abs(walked - predicted))

    weighted = weight_norm[..., None] * attrib
    return weighted, ms_mean_a, ms_mean_b, log_a, log_b, resid_log, attrib, d_ms_cells


def _rel(lhs: float, rhs: float) -> float:
    scale = max(abs(rhs), 1e-300)
    return abs(lhs - rhs) / scale


def _build_channel(
    channel: str,
    weighting: str,
    tables: tuple,
    perm_a: np.ndarray,
    perm_b: np.ndarray,
    weight_used: np.ndarray,
    covered_used: int,
    shipped_gap: float,
    shipped_table: np.ndarray,
    order: int,
    chars: str,
    rng: np.random.Generator | None,
) -> tuple[ChannelAttribution, np.ndarray, np.ndarray]:
    """Assemble one :class:`ChannelAttribution` plus its per-board seed-mean ms tables."""
    from keybo.verdicts import require_finite

    names = list(tables[5])
    weighted, ms_a, ms_b, log_a, log_b, resid_log, attrib, d_ms_cells = _lmdi_channel(
        tables, perm_a, perm_b, weight_used, covered_used, order, rng
    )
    contributions_ms = weighted.sum(axis=tuple(range(order)))
    weight_norm = weight_used / max(covered_used, 1)
    level_a = float((weight_norm * ms_a).sum())
    level_b = float((weight_norm * ms_b).sum())
    gap = level_b - level_a

    # An empty character intersection or an all-zero weight table is the silent path into a nan
    # cascade (0/0 in the normalization), and a nan gap would still "sum back" to a nan without
    # tripping any relative bar. Refused here rather than reported as a residual.
    require_finite(
        [gap, level_a, level_b, *contributions_ms.tolist()], f"{channel} channel attribution"
    )

    resid_cell = float(
        np.abs(attrib.sum(axis=-1) - d_ms_cells).max() / max(np.abs(d_ms_cells).max(), 1e-300)
    )
    ms_mean_table = np.mean(tables[3], axis=0)
    return (
        ChannelAttribution(
            channel=channel,
            weighting=weighting,
            feature_names=names,
            gap=gap,
            shipped_gap=shipped_gap,
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
            resid_cell_lmdi=resid_cell,
            resid_feature_sum=float(_rel(float(contributions_ms.sum()), gap)),
            resid_gap_vs_shipped=float(abs(gap - shipped_gap)),
            resid_table_vs_shipped=float(np.abs(ms_mean_table - shipped_table).max()),
            resid_additivity=float(tables[4]),
            resid_log_vs_predict=float(resid_log),
            _weighted=weighted,
            _chars=chars + " ",
        ),
        ms_a,
        ms_b,
    )


def shap_diff(
    layout_a: str,
    layout_b: str,
    *,
    name_a: str | None = None,
    name_b: str | None = None,
    surface: TimeSurface | None = None,
    bigram_models: Sequence[XGBoostTypingModel] | None = None,
    trigram_models: Sequence[XGBoostTypingModel] | None = None,
    target_wpm: float = 90.0,
    corpus: str | None = None,
    channel: str = "both",
    weighting: str = "trigram-marginal",
    tcond_weighting: str = "trigram-direct",
    control_bigram_freqs: Mapping[str, int] | None = None,
    shuffle_seed: int | None = None,
    frame: str = "served",
) -> ShapDiff:
    """Decompose ``ms/char(layout_b) - ms/char(layout_a)`` into per-feature contributions.

    ``layout_a`` / ``layout_b`` are 30-character row-major layout strings (the caller resolves
    registry names). The sign convention is fixed and stated in the output: POSITIVE means
    ``layout_a`` is faster.

    ``channel`` selects which of the gauge's two terms to decompose — ``"t2"`` (the bigram
    table), ``"tcond"`` (the conditioned-trigram increment), or ``"both"`` (the default, and
    the only setting whose decomposed share can reach 100%).

    ``frame`` selects the FEATURE FRAME the T2 channel is decomposed on (:data:`FRAMES`).
    ``"interp"`` requires ``bigram_models=`` and ``channel="t2"``: no shipped artifact carries
    that frame, and it has no trigram counterpart. Its shipped-table anchors are taken from the
    SUPPLIED models' own table rather than from ``TimeSurface._T2`` — a non-production model's
    surface is not the production surface, and tying it to one would report a model DIFFERENCE as
    an attribution residual.

    ``surface`` may be supplied to reuse a loaded :class:`TimeSurface` (model load dominates a
    short run); ``bigram_models`` / ``trigram_models`` likewise, defaulting to the same seeded
    k31 artifacts the surface itself uses.

    The control knobs exist to be FAILING controls, and each is recorded in the result:

    * ``weighting="bigram-table"`` with ``control_bigram_freqs`` weights the T2 channel by a
      standalone bigram table instead of the trigram marginal.
    * ``tcond_weighting="tcond-marginal"`` weights the Tcond channel by that same
      first-two-character marginal broadcast over the third character — the error of assuming
      the T2 correction transfers to a term indexed by all three characters.
    * ``shuffle_seed`` permutes the per-cell SHAP-delta vectors across cells, in every channel.

    The two weighting controls must FAIL the EXTERNAL bars while PASSING the internal ones; the
    shuffle does the reverse. That pairing is what shows the two bar families are not
    redundant — a self-consistency identity validates the arithmetic, never the choice of
    quantity.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}")
    if tcond_weighting not in TCOND_WEIGHTINGS:
        raise ValueError(
            f"tcond_weighting must be one of {TCOND_WEIGHTINGS}, got {tcond_weighting!r}"
        )
    if weighting == "bigram-table" and control_bigram_freqs is None:
        raise ValueError(
            "weighting='bigram-table' is a negative control and needs control_bigram_freqs"
        )
    if weighting == "bigram-table" and channel == "tcond":
        raise ValueError(
            "weighting='bigram-table' is a T2-channel control but channel='tcond' does not "
            "decompose T2; use tcond_weighting='tcond-marginal' for the Tcond control"
        )
    if tcond_weighting == "tcond-marginal" and channel == "t2":
        raise ValueError(
            "tcond_weighting='tcond-marginal' is a Tcond-channel control but channel='t2' does "
            "not decompose Tcond; use weighting='bigram-table' for the T2 control"
        )
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    if frame in ("interp", "interp-wpm"):
        # Both refusals rather than defaults, because both silent alternatives are wrong in a way
        # that still RECONCILES: defaulting to the shipped bigram models would attribute the
        # SERVED frame while reporting a non-served frame, and decomposing Tcond would mix a
        # 10-column T2 explanation with a 46-column trigram one under one headline.
        if bigram_models is None:
            raise ValueError(
                f"frame={frame!r} has no shipped artifact (no data/models/k31 model carries "
                f"an INTERPFRAME stamp), so bigram_models= must be supplied explicitly"
            )
        if channel != "t2":
            raise ValueError(
                f"frame={frame!r} is bigram-only (INTERPFRAME-1 is a T2-channel POC); "
                f"got channel={channel!r} -- use channel='t2'"
            )
    want_t2 = channel in ("t2", "both")
    want_tcond = channel in ("tcond", "both")

    if surface is None:
        surface = default_surface(target_wpm, corpus)
    geometry = surface.geometry
    if bigram_models is None:
        bigram_models = default_models("bigram")
    if trigram_models is None:
        trigram_models = default_models("trigram")

    # Reuse the reviewed guard: it REFUSES a short or repeating layout, which would otherwise
    # be scored over a fraction of the corpus and still look plausible.
    slot_a = surface._slot_of(layout_a)
    slot_b = surface._slot_of(layout_b)
    if set(layout_a) != set(layout_b) and not set(layout_a) & set(layout_b):
        # Different charsets cover different trigram subsets; the decomposition then runs on
        # the intersection and `coverage_cost` prices what that moved. An EMPTY intersection
        # has nothing to decompose and is refused rather than producing a nan cascade.
        raise ValueError("layouts share no characters; there is no common support to diff")
    chars = layout_a  # character-index order; both boards are indexed through it

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
    # The CORRECT weights, kept aside: the shipped-table anchors below are ALWAYS computed on
    # these, so a control cannot move the reference it is supposed to be caught by. (Comparing
    # a control's gap against a reference recomputed under the SAME wrong weight is exactly the
    # self-consistency tautology that makes `resid_feature_sum` insufficient on its own.)
    w2_true, w3_true, covered_true = w2, w3, covered_common

    w2_used, covered_w2 = w2, covered_common
    if weighting == "bigram-table":
        # T2 NEGATIVE CONTROL: weight the bigram channel by a standalone bigram table.
        index = {c: i for i, c in enumerate(chars)}
        index[" "] = len(chars)
        w2_used = np.zeros_like(w2)
        for bigram, freq in control_bigram_freqs.items():
            if len(bigram) != 2:
                continue
            try:
                w2_used[index[bigram[0]], index[bigram[1]]] += freq
            except KeyError:
                continue
        covered_w2 = int(round(w2_used.sum()))

    w3_used, covered_w3 = w3, covered_common
    if tcond_weighting == "tcond-marginal":
        # Tcond NEGATIVE CONTROL: the first-two-character marginal broadcast UNIFORMLY over the
        # third character — i.e. assuming SHAPDIFF-1's T2 marginal correction transfers to a
        # term that is indexed by all three characters. Its own mass is used as the
        # denominator, so this is a wrong weight DISTRIBUTION rather than a mere rescaling (a
        # rescaling would be a trivially-caught arithmetic error; a redistribution is the
        # interesting failure, and it is the one a symmetry-seeking implementer would commit).
        w3_used = np.repeat(w2[:, :, None], w3.shape[2], axis=2)
        covered_w3 = int(round(w3_used.sum()))

    # --- position indices for each board -------------------------------------------------
    perm_a = np.array([slot_a[c] for c in chars] + [slot_a[" "]], dtype=np.intp)
    perm_b = np.array([slot_b[c] for c in chars] + [slot_b[" "]], dtype=np.intp)
    idx_a2, idx_b2 = np.ix_(perm_a, perm_a), np.ix_(perm_b, perm_b)
    idx_a3 = np.ix_(perm_a, perm_a, perm_a)
    idx_b3 = np.ix_(perm_b, perm_b, perm_b)

    # --- the SHIPPED anchors, on the CORRECT weights ------------------------------------
    # These are the external references. They come from `predict`/`predict_ms` — a DIFFERENT
    # xgboost code path from `pred_contribs` — under the gauge's own weights, so comparing a
    # TreeSHAP-anchored gap against them is a genuine external tie rather than a restatement.
    shipped_t2, shipped_tc = surface._T2, surface._Tc
    if frame in ("interp", "interp-wpm"):
        # ⚠ The T2 anchor must be THE ATTRIBUTED MODEL'S OWN predict()-side table, not
        # `surface._T2`. An interp model is a different fit, so its table differs from the
        # production one by a MODEL difference (~ms), not by float32 noise (~1e-5) — anchoring to
        # the production table would book that model difference as an attribution residual and
        # fail the external bar for a reason that has nothing to do with the attribution. Built
        # from `predict_ms` here, which is the same independent code path the shipped anchor uses,
        # so the external tie stays a genuine cross-check of the TreeSHAP walk.
        positions = [*geometry.slots, geometry.space_position]
        n_pos = len(positions)
        vecs = np.vstack(
            [
                interp_features_from_positions(geometry, (a, b), wpm=target_wpm)
                for a in positions
                for b in positions
            ]
        )
        # `wpm=target_wpm` is REQUIRED here, not optional: the interp frame has no wpm column, so
        # `to_ms` cannot recover the pace from the matrix and refuses to guess. Passing the SAME
        # `target_wpm` the features were built at is what keeps this anchor comparable to the
        # TreeSHAP side, whose ms conversion divides by that same constant.
        shipped_t2 = np.mean(
            [m.predict_ms(vecs, wpm=target_wpm).reshape(n_pos, n_pos) for m in bigram_models],
            axis=0,
        )
    norm_true = 1.0 / max(covered_true, 1)
    shipped_gap_t2 = float(
        ((w2_true * shipped_t2[idx_b2]).sum() - (w2_true * shipped_t2[idx_a2]).sum()) * norm_true
    )
    shipped_gap_tc = float(
        ((w3_true * shipped_tc[idx_b3]).sum() - (w3_true * shipped_tc[idx_a3]).sum()) * norm_true
    )

    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    t2_channel = tcond_channel = None
    t2_ms_a = t2_ms_b = tc_ms_a = tc_ms_b = None
    if want_t2:
        t2_channel, t2_ms_a, t2_ms_b = _build_channel(
            "t2",
            weighting,
            _shap_tables(bigram_models, geometry, target_wpm, 2, frame),
            perm_a,
            perm_b,
            w2_used,
            covered_w2,
            shipped_gap_t2,
            shipped_t2,
            2,
            chars,
            rng,
        )
    if want_tcond:
        tcond_channel, tc_ms_a, tc_ms_b = _build_channel(
            "tcond",
            tcond_weighting,
            _shap_tables(trigram_models, geometry, target_wpm, 3),
            perm_a,
            perm_b,
            w3_used,
            covered_w3,
            shipped_gap_tc,
            shipped_tc,
            3,
            chars,
            rng,
        )

    # --- the ms/char numbers -------------------------------------------------------------
    # A DECOMPOSED channel is levelled on its own SHAP-anchored table under the weights in use
    # (which is what lets a wrong-weighting control move the `card()` tie, as it must); an
    # UNDECOMPOSED one falls back to the shipped table on the correct weights, so `gap_total`
    # and the channel split stay meaningful under `--channel` without pretending to a
    # decomposition this run did not do.
    t2_a = float(
        (w2_used * (t2_ms_a if want_t2 else shipped_t2[idx_a2])).sum() / max(covered_w2, 1)
    )
    t2_b = float(
        (w2_used * (t2_ms_b if want_t2 else shipped_t2[idx_b2])).sum() / max(covered_w2, 1)
    )
    tc_a = float(
        (w3_used * (tc_ms_a if want_tcond else shipped_tc[idx_a3])).sum() / max(covered_w3, 1)
    )
    tc_b = float(
        (w3_used * (tc_ms_b if want_tcond else shipped_tc[idx_b3])).sum() / max(covered_w3, 1)
    )

    card_a = surface.card(layout_a)
    card_b = surface.card(layout_b)
    gap_t2, gap_tcond = t2_b - t2_a, tc_b - tc_a
    gap_total = (t2_b + tc_b) - (t2_a + tc_a)

    # Own-support ms/char on THIS module's tables where available (so `coverage_cost` isolates
    # the coverage restriction rather than the SHAP-vs-predict anchor difference — see
    # `coverage_cost`), using the shipped table for any channel this run did not decompose so
    # the two sides of that difference are built the same way.
    t2_table = (
        np.mean(_shap_tables(bigram_models, geometry, target_wpm, 2, frame)[3], axis=0)
        if want_t2
        else shipped_t2
    )
    tc_table = (
        np.mean(_shap_tables(trigram_models, geometry, target_wpm, 3)[3], axis=0)
        if want_tcond
        else shipped_tc
    )

    def _own(w3_own: np.ndarray, perm: np.ndarray, covered: int) -> float:
        i2 = np.ix_(perm, perm)
        i3 = np.ix_(perm, perm, perm)
        t2_part = (w3_own.sum(axis=2) * t2_table[i2]).sum()
        tc_part = (w3_own * tc_table[i3]).sum()
        return float((t2_part + tc_part) / max(covered, 1))

    w3_a, _, _ = _char_weight_tables(surface, layout_a)
    own_a = _own(w3_a, perm_a, covered_a)
    if set(layout_a) == set(layout_b):
        # Same charset: re-index the SAME trigram weights through layout_b's permutation.
        own_b = _own(w3_a, perm_b, covered_b)
    else:
        w3_b_native, _, _ = _char_weight_tables(surface, layout_b)
        perm_b_native = np.array([slot_b[c] for c in layout_b] + [slot_b[" "]], dtype=np.intp)
        own_b = _own(w3_b_native, perm_b_native, covered_b)

    return ShapDiff(
        name_a=name_a or layout_a,
        name_b=name_b or layout_b,
        layout_a=layout_a,
        layout_b=layout_b,
        corpus=corpus or "default",
        target_wpm=target_wpm,
        channel=channel,
        ms_per_char_own_a=own_a,
        ms_per_char_own_b=own_b,
        card_ms_per_char_a=card_a.ms_per_char,
        card_ms_per_char_b=card_b.ms_per_char,
        ms_per_char_a=t2_a + tc_a,
        ms_per_char_b=t2_b + tc_b,
        gap_total=gap_total,
        gap_t2=gap_t2,
        gap_tcond=gap_tcond,
        t2=t2_channel,
        tcond=tcond_channel,
        covered_mass_a=covered_a,
        covered_mass_b=covered_b,
        covered_mass_common=covered_common,
        corpus_total_mass=int(surface.total_mass),
        # float() on every residual: numpy scalars are not JSON-serializable, and `to_dict`
        # embeds these directly.
        resid_channel_split=float(_rel(gap_t2 + gap_tcond, gap_total)),
        resid_vs_card_gap=float(abs(gap_total - (card_b.ms_per_char - card_a.ms_per_char))),
        frame=frame,
    )


# --- reporting ---------------------------------------------------------------------------


def _format_channel(
    diff: ShapDiff, ch: ChannelAttribution, top_ngrams_k: int, columns: bool
) -> list[str]:
    a, b = diff.name_a, diff.name_b
    label = {"t2": "T2 (bigram table)", "tcond": "Tcond (conditioned trigram)"}[ch.channel]
    share = 100.0 * ch.gap / diff.gap_total if diff.gap_total else float("nan")
    lines = [
        "",
        f"=== CHANNEL {ch.channel.upper()} — {label} ===",
        f"  weighting: {ch.weighting}    gap {ch.gap:+.4f} ms/char ({share:.1f}% of the total gap)",
        "  residuals:",
        f"    INTERNAL per-cell LMDI identity (rel)     {ch.resid_cell_lmdi:.3e}",
        f"    INTERNAL sum(features) vs channel gap     {ch.resid_feature_sum:.3e}",
        f"    EXTERNAL gap vs shipped table (abs ms/ch) {ch.resid_gap_vs_shipped:.3e}",
        f"    EXTERNAL table vs shipped (abs ms)        {ch.resid_table_vs_shipped:.3e}",
        f"    TreeSHAP walk vs predict, per row (abs)   {ch.resid_additivity:.3e}",
        f"    TreeSHAP walk vs predict, weighted (abs)  {ch.resid_log_vs_predict:.3e}",
        f"  RECONCILES: {ch.reconciles()}",
    ]
    if not ch.reconciles():
        lines.append(f"  !! CHANNEL {ch.channel} FAILED — its tables below explain NOTHING. !!")
    lines.append("")
    # BLOCKS FIRST: a block sum is invariant to how SHAP split credit inside it, so it is the
    # claim that survives the correlated-credit non-uniqueness. Columns are subordinate.
    lines.append(f"  BLOCK CONTRIBUTIONS (primary) to gap_{ch.channel} ({ch.gap:+.4f} ms/char)")
    lines.append(f"    {'block':<12} {'ms/char':>10} {'share':>8} {'favours':>12}   parts")
    for blk in ch.blocks():
        pct = 100.0 * blk.ms_per_char / ch.gap if ch.gap else float("nan")
        favour = {"a": a, "b": b, "tie": "-"}[blk.favours]
        parts = "  ".join(f"{p}{v:+.4f}" for p, v in blk.parts[:6])
        lines.append(
            f"    {blk.block:<12} {blk.ms_per_char:>+10.4f} {pct:>7.1f}% {favour:>12}   {parts}"
        )
    lines.append(f"    {'SUM':<12} {sum(x.ms_per_char for x in ch.blocks()):>+10.4f}")
    if columns:
        lines.append("")
        # Header keeps SHAPDIFF-1's "PER-FEATURE CONTRIBUTIONS" wording (its report-ordering
        # test pins that substring) while naming these as per-COLUMN and subordinate, which is
        # the whole point of putting the block table above them.
        lines.append(
            "  PER-FEATURE CONTRIBUTIONS, per COLUMN (subordinate — SHAP's split across "
            "correlated columns is NOT unique; read the blocks above)"
        )
        lines.append(
            f"    {'feature':<20} {'block':<11} {'ms/char':>10} {'share':>8} {'favours':>12}"
            f"   {'log_a':>9} {'log_b':>9} {'d_log':>10}"
        )
        spec = block_map(ch.feature_names)
        for c in ch.ranked():
            pct = 100.0 * c.ms_per_char / ch.gap if ch.gap else float("nan")
            favour = {"a": a, "b": b, "tie": "-"}[c.favours]
            lines.append(
                f"    {c.feature:<20} {spec[c.feature][0]:<11} {c.ms_per_char:>+10.4f} "
                f"{pct:>7.1f}% {favour:>12}   {c.log_a:>+9.5f} {c.log_b:>+9.5f} "
                f"{c.log_delta:>+10.5f}"
            )
        lines.append(
            f"    {'SUM':<20} {'':<11} {sum(c.ms_per_char for c in ch.contributions):>+10.4f}"
        )
    if top_ngrams_k:
        lines.append("")
        lines.append("  TOP N-GRAMS per leading column (ms/char, corpus-weighted)")
        for c in ch.ranked()[:6]:
            top = ch.top_ngrams(c.feature, top_ngrams_k)
            rendered = "  ".join(f"{ng!r}{v:+.4f}" for ng, v in top)
            lines.append(f"    {c.feature:<20} {rendered}")
    return lines


def format_report(diff: ShapDiff, top_bigrams_k: int = 5, columns: bool = True) -> str:
    """The human-readable report: reconciliation FIRST, then blocks, then columns.

    The order is deliberate twice over — a reader who meets the ranked features before the
    residuals can form a story about a table that does not sum to anything, and a reader who
    meets the 46-column table before the blocks can form a story about a credit split that is
    not unique.
    """
    lines: list[str] = []
    a, b = diff.name_a, diff.name_b
    lines.append(f"SHAP-DIFF  {a} -> {b}   corpus={diff.corpus}  wpm={diff.target_wpm:g}")
    lines.append(f"  channel: {diff.channel}    frame: {diff.frame}")
    if not diff.card_tie_applies:
        # Printed at the TOP, before any number: a reader must not meet an attribution table
        # believing it decomposes the shipped gauge when it decomposes a POC model's.
        lines.append(
            f"  ⚠ frame={diff.frame!r} is NOT the served frame: the T2 table below comes from "
            f"SUPPLIED models, not from data/models/k31, so the card() tie is a MODEL difference "
            f"and is not gated (see `card_tie_applies`)."
        )
    lines.append("")
    lines.append("RECONCILIATION (checked before any interpretation)")
    lines.append(
        f"  ms/char shipped card()  {a}: {diff.card_ms_per_char_a:.4f}   "
        f"{b}: {diff.card_ms_per_char_b:.4f}   <- what `keybo analyze` prints"
    )
    lines.append(
        f"  ms/char own-support     {a}: {diff.ms_per_char_own_a:.4f}   "
        f"{b}: {diff.ms_per_char_own_b:.4f}"
    )
    lines.append(
        f"  ms/char common-support  {a}: {diff.ms_per_char_a:.4f}   {b}: {diff.ms_per_char_b:.4f}"
    )
    lines.append(f"  gap (b-a; +ve = {a} faster) : {diff.gap_total:+.4f} ms/char")
    t2_pct = 100.0 * diff.gap_t2 / diff.gap_total if diff.gap_total else float("nan")
    tc_pct = 100.0 * diff.gap_tcond / diff.gap_total if diff.gap_total else float("nan")
    lines.append(
        f"    T2 bigram channel    : {diff.gap_t2:+.4f}  ({t2_pct:.1f}%)  "
        f"<- {'decomposed' if diff.t2 is not None else 'NOT decomposed in this run'}"
    )
    lines.append(
        f"    Tcond trigram channel: {diff.gap_tcond:+.4f}  ({tc_pct:.1f}%)  "
        f"<- {'decomposed' if diff.tcond is not None else 'NOT decomposed in this run'}"
    )
    lines.append(
        f"  DECOMPOSED SHARE: {diff.decomposed_share_pct:.1f}% of the gap    "
        f"undecomposed: {diff.undecomposed_ms_per_char:+.4f} ms/char"
    )
    lines.append(f"  coverage {a}: {diff.coverage_pct_a:.3f}%   {b}: {diff.coverage_pct_b:.3f}%")
    lines.append(
        f"  common support is a no-op: {diff.common_support_is_noop}   coverage cost: "
        f"{diff.coverage_cost:+.3e} ms/char"
    )
    lines.append(f"  channel split vs total (rel)           {diff.resid_channel_split:.3e}")
    tie = "" if diff.card_tie_applies else "   <- NOT GATED (non-served frame; a model difference)"
    lines.append(f"  gap vs shipped card (abs ms/char)      {diff.resid_vs_card_gap:.3e}{tie}")
    lines.append(f"  RECONCILES: {diff.reconciles()}")
    if not diff.reconciles():
        lines.append("")
        lines.append("!! RECONCILIATION FAILED — the tables below explain NOTHING. !!")
    for ch in (diff.t2, diff.tcond):
        if ch is not None:
            lines.extend(_format_channel(diff, ch, top_bigrams_k, columns))
    return "\n".join(lines)
