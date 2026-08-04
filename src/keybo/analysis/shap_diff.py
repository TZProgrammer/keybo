"""Per-feature attribution of the ms/char GAP between two layouts (SHAPDIFF-1, -TCOND, COMPARE-1).

The engine behind ``keybo compare``. (The module keeps its ``shap_diff`` name after COMPARE-1
renamed the *command*: the SHAPDIFF-1/-TCOND ledger entries, committed artifacts and tests all
name it, and renaming it would cost that audit trail for no user-facing gain.)

``shap-report`` answers "what does the model use?" over a matrix of rows. This module
answers the different, harder question: **why is layout B slower than layout A?**, as a
signed per-feature budget in the units the analyzer publishes (ms/char) that SUMS BACK to
the measured gap.

**What a reader gets, and the four things the output refuses to let them conclude.** Every row
carries not just its signed contribution but the FEATURE'S OWN VALUE on each board
(:attr:`FeatureContribution.mean_a` / ``mean_b``), under the same corpus weight the gap uses —
because a contribution alone cannot distinguish "board B does less of this" from "board B does
more of it and the model prices it cheaply". The refusals are :meth:`ChannelAttribution.leakage`
(rows whose credit is not their own), :meth:`ShapDiff.total_for_property` (which raises rather
than adding a property across channels), :meth:`ShapDiff.gauge_tie_ok` (which suppresses the
tables outright when the external tie fails), and the :data:`CAVEATS` block printed with the
magnitudes. Each corresponds to a way an attribution was MEASURED to mislead, not imagined.

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
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import xgboost as xgb

from keybo.analysis.timecard import TimeSurface, default_surface
from keybo.features import bigram_features_from_positions, trigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES, TRIGRAM_FEATURE_NAMES
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

#: |ms/char| below which a column is not flagged for leakage. A flag on numerical dust would
#: mark rows nobody reads and train a reader to ignore the flag on the rows that matter; the
#: measured instances the flag exists for are 0.0273-0.7382, two orders of magnitude above this.
LEAKAGE_MS_FLOOR = 0.01

#: The tie to the shipped ``card()`` gauge, in ABSOLUTE ms/char, beyond which a report REFUSES
#: to print an interpretable table. Same value as :meth:`ShapDiff.reconciles`'s ``gauge_tol``,
#: named separately because this one is a PRODUCT decision (suppress the tables) rather than a
#: test bar. SHAPDIFF-1 measured why it cannot be dropped: the internal sums-back identity
#: passes at ~1e-16 under a weighting that is wrong by 5.6e-2 here, and SHAPDIFF-TCOND measured
#: that the analogous Tcond error INVERTS which block leads.
GAUGE_REFUSAL_MS = 1e-3

#: One line naming what the numbers ARE, carried into every report and every JSON artifact.
ESTIMAND = (
    "frequency-weighted LMDI attribution of the ms/char gap between two layouts, on the "
    "FITTED time surface, split into the gauge's own T2 (bigram-table) and Tcond "
    "(conditioned-trigram) terms and exact by algebraic identity within each"
)

#: The four ways this attribution can mislead a reader, each MEASURED in SHAPDIFF-1 or
#: SHAPDIFF-TCOND rather than imagined. Printed where magnitudes are printed, on the principle
#: that a caveat filed in a report nobody reopens is not a caveat. Kept as data (not prose baked
#: into the formatter) so the JSON artifact carries the identical text.
CAVEATS = (
    "MODEL, NOT BIOMECHANICS: every number is a contribution to THIS fitted surface's "
    "prediction, not a measured property of a hand. Some features have no clean standalone "
    "mechanism -- this surface prices LONG travel as CHEAPER, so a positive 'dx' is a fact "
    "about the model's pricing, not about physical distance being good.",
    "MAGNITUDES CARRY THE MODEL'S CALIBRATION ERROR: the per-fold calibration slope reaches "
    "1.407 on the bigram surface (qwerty fold) and 0.7304 on the trigram one (dvorak fold), so "
    "an ms/char magnitude can be off by tens of percent. ORDERINGS are affine-invariant and "
    "therefore safe; read the ranking, and treat the ms figures as scaled.",
    "PER-COLUMN CREDIT IS NOT UNIQUE: TreeSHAP's split across CORRELATED columns is one of "
    "many valid splits, so BLOCK sums are the primary result and per-column rows are "
    "subordinate detail. Rows whose credit is provably not their own are FLAGGED.",
    "CHANNELS DO NOT ADD PER FEATURE: a property present in both frames (T2 'bottom' and Tcond "
    "'bg1_bottom'/'bg2_bottom') carries its own channel's full share in each, so the two must "
    "NOT be summed. Only gap_total adds across channels.",
)

#: What the tool structurally CANNOT do, stated in the product rather than left to be inferred.
CANNOT = (
    "It cannot attribute any part of the gap that is not in the model's features at all: "
    "neither frame carries a hand-identity channel (so 'board B overloads one hand' is "
    "unaskable), and the bigram frame carries no direction-of-travel channel. Such a "
    "difference is priced inside the features that ARE served, not reported as a remainder."
)

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
    for spec in (_T2_BLOCKS, _TCOND_BLOCKS):
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

    ``mean_a`` / ``mean_b`` are the FEATURE VALUE itself on each board — the corpus-frequency
    weighted mean of this column, under the SAME weight the gap is computed with (``w2`` for
    the T2 channel, ``w3`` for Tcond; see :func:`_char_weight_tables`). They answer the
    question the attribution alone cannot: a contribution says a feature moved the gap, and
    these say WHICH DIRECTION the boards differ in. ``bottom +0.7453 favours flagship-c3`` is
    compatible both with flagship doing LESS bottom-row work and with it doing more and being
    priced cheaper; ``mean_a 0.0770`` vs ``mean_b 0.1190`` settles it.

    ⚠ Also a per-BOARD quantity, and NOT an attribution — it is a mean of the design matrix,
    with no SHAP value and no LMDI weight in it. ``sign(mean_delta)`` therefore need NOT match
    ``sign(ms_per_char)``, and a mismatch is not a bug: TreeSHAP's credit on correlated columns
    need not align with the marginal feature delta, and this surface genuinely prices some
    features counter-intuitively (it prices LONG travel cheaper). What the pair licenses is
    reading the DIRECTION, never inferring a mechanism.
    """

    feature: str
    ms_per_char: float
    log_a: float
    log_b: float
    log_delta: float
    mean_a: float
    mean_b: float
    #: Why this row's per-column number may not be trustworthy on its own, or ``""``. See
    #: :meth:`ChannelAttribution.leakage` for the two kinds and the registered decision rule.
    flag: str = ""

    @property
    def mean_delta(self) -> float:
        """``mean_b - mean_a``: how much MORE of this feature layout B does."""
        return self.mean_b - self.mean_a

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

    ``leading`` names the block's biggest single column and carries ITS feature values, so the
    primary table can show the two-column view without a reader dropping to the subordinate
    per-column table. There is deliberately NO block-level mean: a block spans a one-hot
    (``bottom``, in [0,1]) and a distance in key units (``dx``, ~4.3), so a summed or averaged
    "block feature value" would be a number in no unit at all. The honest block-level statement
    is its leading column's value, named as such.
    """

    block: str
    ms_per_char: float
    columns: tuple[str, ...]
    parts: tuple[tuple[str, float], ...]
    #: ``(column, mean_a, mean_b)`` for the block's largest-|ms| column, or ``None`` if empty.
    leading: tuple[str, float, float] | None = None
    #: Non-empty when any column in this block is flagged — see :meth:`ChannelAttribution.leakage`.
    flag: str = ""

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

    def leakage(self) -> dict[str, str]:
        """``column -> reason`` for every column whose per-column number is not trustworthy alone.

        Computed, never hand-listed, from the two failure modes SHAPDIFF-1/-TCOND MEASURED.
        Both need a magnitude floor (:data:`LEAKAGE_MS_FLOOR`) so numerical dust in a column
        nobody would read does not raise a flag that then cries wolf on the ones that matter:

        ``COUPLED`` — the property appears as both ``bg1_X`` and ``bg2_X`` and the two carry
        OPPOSITE signs. The same physical property is then being credited in both directions
        across two blocks, so neither column's number stands alone and only their JOINT sum
        (:meth:`joint`) is meaningful. The measured instance: ``bg1_bottom`` -0.2337 against
        ``bg2_bottom`` +0.7382.

        ``NO-DIFF`` — the two boards do not differ in the feature at all (``mean_a == mean_b``)
        yet the column still carries credit. The credit is then necessarily a coupled-column
        interaction artifact rather than a difference between the boards. The measured instance:
        ``wpm``, constant at the scoring WPM on both boards, carrying -0.0922 on the bigram
        frame and -0.0273 on the trigram one. ⚠ This flag exists ONLY because the ``mean_a`` /
        ``mean_b`` columns exist — without them the report cannot tell "B does less of it" from
        "B does exactly as much of it and the tree split on an interaction path".
        """
        by_name = {c.feature: c for c in self.contributions}
        flags: dict[str, str] = {}
        for name, contribution in by_name.items():
            if abs(contribution.ms_per_char) < LEAKAGE_MS_FLOOR:
                continue
            if _rel(contribution.mean_b, contribution.mean_a) <= 1e-12:
                flags[name] = "NO-DIFF"
        for name, contribution in by_name.items():
            if not name.startswith("bg1_"):
                continue
            mate = by_name.get("bg2_" + name[4:])
            if mate is None:
                continue
            pair = (contribution, mate)
            if min(abs(c.ms_per_char) for c in pair) < LEAKAGE_MS_FLOOR:
                continue
            if contribution.ms_per_char * mate.ms_per_char < 0.0:
                for c in pair:
                    # A COUPLED pair is the stronger statement, so it wins over NO-DIFF: the
                    # reader's action differs (read the joint, vs. discount the row entirely).
                    flags[c.feature] = "COUPLED"
        return flags

    def joint(self, property_name: str) -> float:
        """``bg1_X + bg2_X`` — the trustworthy total for a property split across the two blocks.

        The joint is what survives ``bg1_``<->``bg2_`` leakage: the split between the two
        overlapping key pairs is not unique, but their sum does not depend on how TreeSHAP
        divided the credit. Raises for a property this frame does not carry twice, rather than
        returning a single column's number under a name that promises two.
        """
        by_name = {c.feature: c.ms_per_char for c in self.contributions}
        try:
            return by_name[f"bg1_{property_name}"] + by_name[f"bg2_{property_name}"]
        except KeyError:
            raise ValueError(
                f"{property_name!r} is not a property this {len(self.feature_names)}-column "
                "frame carries as both bg1_* and bg2_*, so there is no joint to take"
            ) from None

    def blocks(self) -> list[BlockContribution]:
        """Block contributions, largest |ms/char| first — the PRIMARY table.

        Sums over a registered partition of the frame, so every column lands in exactly one
        block and the block sums add to the same channel gap the columns do.
        """
        spec = block_map(self.feature_names)
        by_feature = {c.feature: c.ms_per_char for c in self.contributions}
        by_name = {c.feature: c for c in self.contributions}
        flags = self.leakage()
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
            lead = max(columns, key=lambda n: abs(by_feature[n]))
            flagged = sorted({flags[n] for n in columns if n in flags})
            out.append(
                BlockContribution(
                    block=block,
                    ms_per_char=sum(by_feature[n] for n in columns),
                    columns=tuple(columns),
                    parts=tuple(sorted(subs.items(), key=lambda kv: -abs(kv[1]))),
                    leading=(lead, by_name[lead].mean_a, by_name[lead].mean_b),
                    flag=",".join(flagged),
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
            "leakage_flags": self.leakage(),
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
                    "leading_column": b.leading[0] if b.leading else None,
                    "leading_mean_a": b.leading[1] if b.leading else None,
                    "leading_mean_b": b.leading[2] if b.leading else None,
                    "flag": b.flag,
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
                    "mean_a": c.mean_a,
                    "mean_b": c.mean_b,
                    "mean_delta": c.mean_delta,
                    "flag": c.flag,
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

    # --- the reconciliation gate -------------------------------------------------------

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
        """
        # bool(), not the bare `and` chain: several residuals are numpy scalars, so the chain
        # returns np.bool_ — which is falsy-correct but NOT JSON-serializable, and `to_dict`
        # embeds this value. Caught by the CLI's own JSON round-trip test.
        return bool(
            self.resid_channel_split <= rel_tol
            and self.resid_vs_card_gap <= gauge_tol
            and all(
                ch.reconciles(rel_tol, add_tol, gauge_tol)
                for ch in (self.t2, self.tcond)
                if ch is not None
            )
        )

    def gauge_tie_ok(self, gauge_tol: float = GAUGE_REFUSAL_MS) -> bool:
        """True iff the EXTERNAL ties hold — the bar that licenses printing a table at all.

        Separated from :meth:`reconciles` on purpose. ``reconciles`` is the full gate and
        includes the internal identities; THIS is specifically the family a wrong CHOICE OF
        QUANTITY breaks, and it is the one the report refuses on. SHAPDIFF-1 measured why the
        distinction has to be drawn in code rather than in a comment: under a wrong corpus
        weighting the internal bars still read ~1e-16 — a self-consistent decomposition of the
        WRONG quantity — while this tie moves by 5.6e-2 ms/char. A run that passed only the
        internal family would be an unfalsifiable table.
        """
        return bool(
            self.resid_vs_card_gap <= gauge_tol
            and all(
                ch.resid_gap_vs_shipped <= gauge_tol
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

    # --- the cross-channel non-additivity guard (H3) -------------------------------------

    def cross_channel_properties(self) -> list[str]:
        """Properties carried by BOTH channels, whose contributions MUST NOT be summed.

        The T2 frame's ``bottom`` and the Tcond frame's ``bg1_bottom``/``bg2_bottom`` are the
        same physical property measured on two DIFFERENT frames over two different populations
        of cells (``n**2`` position pairs vs ``n**3`` triples), each already carrying its own
        channel's full share of the gap. Adding them double-counts: on the registered pair
        ``bottom`` is 23.3% of the total gap and ``bg2_bottom`` 23.1%, and the reader who adds
        them gets 46.4% for a property that moved the gap by neither figure.

        Returned as a LIST OF NAMES, computed from the two frames, so the report can name what
        it is refusing to add rather than printing a general disclaimer nobody reads.
        """
        if self.t2 is None or self.tcond is None:
            return []
        tcond_props = {n[4:] for n in self.tcond.feature_names if n.startswith(("bg1_", "bg2_"))}
        return [n for n in self.t2.feature_names if n in tcond_props]

    def total_for_property(self, property_name: str) -> float:
        """REFUSES to total a property across channels — H3, as executable code.

        There is no correct number to return here, so this raises rather than picking one. A
        convenience that returned ``t2 + tcond`` would be the double-count the honesty layer
        exists to prevent, and a convenience that returned one channel's value would silently
        answer a different question than the one asked. Read the per-channel numbers, or use
        :attr:`gap_total`, which is the only cross-channel total that means anything.
        """
        if property_name not in self.cross_channel_properties():
            raise ValueError(
                f"{property_name!r} is not carried by both channels in this run; there is "
                "nothing to refuse and nothing to add"
            )
        t2_value = next(c.ms_per_char for c in self.t2.contributions if c.feature == property_name)
        joint = self.tcond.joint(property_name)
        raise ValueError(
            f"REFUSED: {property_name!r} appears in BOTH channels (T2 {t2_value:+.4f} ms/char, "
            f"Tcond bg1+bg2 {joint:+.4f} ms/char) and the two MUST NOT be summed — they are the "
            "same physical property attributed on two different frames, each already carrying "
            "its own channel's full share of the gap. Report them separately, or use "
            "gap_total, which is the only cross-channel total that is well defined"
        )

    def to_dict(self, top_ngrams_k: int = 8) -> dict:
        payload: dict = {
            "layout_a": {"name": self.name_a, "layout": self.layout_a},
            "layout_b": {"name": self.name_b, "layout": self.layout_b},
            "corpus": self.corpus,
            "target_wpm": self.target_wpm,
            "channel": self.channel,
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
                "reconciles": self.reconciles(),
            },
            "channels": {
                name: ch.to_dict(top_ngrams_k)
                for name, ch in (("t2", self.t2), ("tcond", self.tcond))
                if ch is not None
            },
            # The honesty layer, in the MACHINE artifact and not only in stdout: a consumer that
            # reads the JSON and never sees the printed report must still receive the caveats.
            "honesty": {
                "estimand": ESTIMAND,
                "caveats": list(CAVEATS),
                "cross_channel_properties_not_summable": self.cross_channel_properties(),
                "gauge_refusal_threshold_ms_per_char": GAUGE_REFUSAL_MS,
                "gauge_tie_ok": self.gauge_tie_ok(),
                "leakage_ms_floor": LEAKAGE_MS_FLOOR,
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
                    "mean_a": c.mean_a,
                    "mean_b": c.mean_b,
                    "mean_delta": c.mean_delta,
                    "flag": c.flag,
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
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    float,
    list[str],
    np.ndarray,
]:
    """Per-seed ``(shap, p, p_predict, ms)`` position-tuple tables from the exact TreeSHAP path.

    Returns ``(shap_tables, p_tables, p_predict_tables, ms_tables, worst_additivity,
    feature_names, features)`` with shapes ``(n_pos,)*order + (n_feat,)`` and
    ``(n_pos,)*order`` x 3.

    ``features`` is the served design matrix itself, reshaped to ``(n_pos,)*order + (n_feat,)``:
    the VALUES the boosters were shown. It is returned so the per-board feature means
    (:attr:`FeatureContribution.mean_a`) contract over exactly the rows the SHAP walk ran on.
    Re-featurizing for the means would open a second path that could silently disagree with the
    attribution's — the same drift the single ``_char_weight_tables`` weight source prevents on
    the corpus side.

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
    served frame FOR ITS ORDER (checked against the schema, not merely across the models),
    LOGRAT output, and NO first-finger calibration. The last matters most and is invisible:
    ``TableBigramScorer`` applies calibration deltas as a per-POSITION multiplicative factor
    OUTSIDE the feature path, so a calibrated model's table would not equal ``exp(prediction)``
    and the ms attribution would silently stop summing to the gauge. None of the six shipped
    k31 artifacts carries any.
    """
    key = (order, target_wpm, _geometry_key(geometry), tuple(id(m) for m in models))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key][1]

    positions = [*geometry.slots, geometry.space_position]
    n = len(positions)
    if order == 2:
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
    if names != expected:
        raise ValueError(
            f"order-{order} models do not carry the served frame: expected {len(expected)} "
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
    out = (
        shap_tables,
        p_tables,
        p_predict_tables,
        ms_tables,
        worst,
        names,
        X.reshape(*(n,) * order, len(names)),
    )
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

    Returns ``(weighted, ms_mean_a, ms_mean_b, log_a, log_b, resid_log, attrib, d_ms_cells,
    feat_a, feat_b)``. ``weighted`` is the corpus-weighted ms/char attribution per
    ``(cell, feature)``; ``attrib`` and ``d_ms_cells`` are the UNWEIGHTED per-cell quantities the
    caller reconciles against; ``feat_a``/``feat_b`` are the per-board FEATURE-VALUE means.

    Written once and parameterized by ``order`` rather than duplicated per channel: the
    identity, the exact-division rule, the shuffle control and the anchoring discipline are the
    delicate parts, and two copies of them would drift apart under maintenance.

    The feature means are computed HERE, in the same function and against the same
    ``weight_norm``, precisely so they cannot be weighted differently from the attribution they
    sit beside. Two properties follow from that and are load-bearing:

    * the ``w2``/``w3`` choice is made once by the caller and applies to both;
    * the means are contracted from ``tables[6]`` — the design matrix the boosters were shown —
      so no second featurization exists to drift.

    They are also deliberately OUTSIDE the seed loop: the design matrix does not depend on the
    seed, so averaging it over three identical copies would be arithmetic theatre.
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

    # THE FEATURE-VALUE MEANS. The same `weight_norm` and the same `axes=order` contraction the
    # log-space SHAP means above use, applied to the DESIGN MATRIX instead of the SHAP matrix —
    # so "the weight the gap uses" is true by construction rather than by convention. Outside
    # the seed loop because the design matrix is seed-independent.
    features = tables[6]
    feat_a = np.tensordot(weight_norm, features[idx_a], axes=order)
    feat_b = np.tensordot(weight_norm, features[idx_b], axes=order)

    weighted = weight_norm[..., None] * attrib
    return (
        weighted,
        ms_mean_a,
        ms_mean_b,
        log_a,
        log_b,
        resid_log,
        attrib,
        d_ms_cells,
        feat_a,
        feat_b,
    )


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
    (
        weighted,
        ms_a,
        ms_b,
        log_a,
        log_b,
        resid_log,
        attrib,
        d_ms_cells,
        feat_a,
        feat_b,
    ) = _lmdi_channel(tables, perm_a, perm_b, weight_used, covered_used, order, rng)
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
    channel_attribution = ChannelAttribution(
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
                mean_a=float(feat_a[i]),
                mean_b=float(feat_b[i]),
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
    )
    # `flag` is a per-row copy of the channel-level leakage verdict, stamped once the whole
    # contribution list exists (the COUPLED rule is a statement about a PAIR of rows, so it
    # cannot be decided while building a row). Kept on the row so a consumer reading a single
    # FeatureContribution — a JSON record, a spreadsheet paste — carries its own caveat instead
    # of needing to re-derive it from the channel.
    flags = channel_attribution.leakage()
    channel_attribution.contributions = [
        replace(c, flag=flags.get(c.feature, "")) for c in channel_attribution.contributions
    ]
    return channel_attribution, ms_a, ms_b


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
) -> ShapDiff:
    """Decompose ``ms/char(layout_b) - ms/char(layout_a)`` into per-feature contributions.

    ``layout_a`` / ``layout_b`` are 30-character row-major layout strings (the caller resolves
    registry names). The sign convention is fixed and stated in the output: POSITIVE means
    ``layout_a`` is faster.

    ``channel`` selects which of the gauge's two terms to decompose — ``"t2"`` (the bigram
    table), ``"tcond"`` (the conditioned-trigram increment), or ``"both"`` (the default, and
    the only setting whose decomposed share can reach 100%).

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
            _shap_tables(bigram_models, geometry, target_wpm, 2),
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
        np.mean(_shap_tables(bigram_models, geometry, target_wpm, 2)[3], axis=0)
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
    )


# --- reporting ---------------------------------------------------------------------------


def _format_channel(
    diff: ShapDiff, ch: ChannelAttribution, top_ngrams_k: int, columns: bool, top: int = 0
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
    flags = ch.leakage()
    # BLOCKS FIRST: a block sum is invariant to how SHAP split credit inside it, so it is the
    # claim that survives the correlated-credit non-uniqueness. Columns are subordinate.
    lines.append(f"  BLOCK CONTRIBUTIONS (primary) to gap_{ch.channel} ({ch.gap:+.4f} ms/char)")
    lines.append(
        f"    {'block':<12} {'ms/char':>10} {'share':>8} {'favours':>12}"
        f"   {'top column':<14} {_short(a):>9} {_short(b):>9}"
    )
    for blk in ch.blocks():
        pct = 100.0 * blk.ms_per_char / ch.gap if ch.gap else float("nan")
        favour = {"a": a, "b": b, "tie": "-"}[blk.favours]
        lead, mean_a, mean_b = blk.leading if blk.leading else ("-", float("nan"), float("nan"))
        mark = f" [{blk.flag}]" if blk.flag else ""
        lines.append(
            f"    {blk.block:<12} {blk.ms_per_char:>+10.4f} {pct:>7.1f}% {favour:>12}"
            f"   {lead:<14} {mean_a:>9.4f} {mean_b:>9.4f}{mark}"
        )
    lines.append(f"    {'SUM':<12} {sum(x.ms_per_char for x in ch.blocks()):>+10.4f}")
    lines.append("")
    lines.append("    within-block parts (a partition of each block, so these sum to it):")
    for blk in ch.blocks():
        parts = "  ".join(f"{p}{v:+.4f}" for p, v in blk.parts[:6])
        lines.append(f"      {blk.block:<12} {parts}")
    # THE JOINTS. A COUPLED property's two columns are individually untrustworthy, so the
    # trustworthy number is printed for the reader rather than left as an exercise.
    coupled = sorted({n[4:] for n, kind in flags.items() if kind == "COUPLED"})
    if coupled:
        lines.append("")
        lines.append(
            "  ⚠ COUPLED PROPERTIES — bg1_X and bg2_X carry OPPOSITE-signed credit for the SAME"
        )
        lines.append(
            "    physical property, so NEITHER column stands alone. The JOINT is what survives:"
        )
        lines.append(
            f"      {'property':<14} {'bg1':>10} {'bg2':>10} {'JOINT':>10}   {'favours':>12}"
        )
        by_name = {c.feature: c.ms_per_char for c in ch.contributions}
        for prop in sorted(coupled, key=lambda p: -abs(ch.joint(p))):
            joint = ch.joint(prop)
            favour = a if joint > 0 else (b if joint < 0 else "-")
            lines.append(
                f"      {prop:<14} {by_name['bg1_' + prop]:>+10.4f} "
                f"{by_name['bg2_' + prop]:>+10.4f} {joint:>+10.4f}   {favour:>12}"
            )
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
            f"   {_short(a):>9} {_short(b):>9} {'Δvalue':>10}  flag"
        )
        spec = block_map(ch.feature_names)
        ranked = ch.ranked()
        shown = ranked[:top] if top and top < len(ranked) else ranked
        for c in shown:
            pct = 100.0 * c.ms_per_char / ch.gap if ch.gap else float("nan")
            favour = {"a": a, "b": b, "tie": "-"}[c.favours]
            lines.append(
                f"    {c.feature:<20} {spec[c.feature][0]:<11} {c.ms_per_char:>+10.4f} "
                f"{pct:>7.1f}% {favour:>12}   {c.mean_a:>9.4f} {c.mean_b:>9.4f} "
                f"{c.mean_delta:>+10.4f}  {c.flag}"
            )
        if len(shown) < len(ranked):
            # A truncation must NAME what it withheld and price it, or a reader takes the visible
            # rows for the whole decomposition — the same over-claim the undecomposed-remainder
            # line exists to prevent one level up. The SUM below is still over ALL columns.
            hidden = ranked[len(shown) :]
            lines.append(
                f"    ... and {len(hidden)} more columns, totalling "
                f"{sum(c.ms_per_char for c in hidden):+.4f} ms/char "
                f"(largest withheld: {hidden[0].feature} {hidden[0].ms_per_char:+.4f}). "
                "Use --top 0 for all; --json is never truncated."
            )
        lines.append(
            f"    {'SUM':<20} {'':<11} {sum(c.ms_per_char for c in ch.contributions):>+10.4f}"
            "   <- over ALL columns, truncated or not"
        )
        lines.append(
            f"    ({_short(a)}/{_short(b)} are the FEATURE VALUE on each board — the "
            "corpus-frequency-weighted mean of that column under this channel's own weight "
            f"({'w2, the trigram marginal' if ch.order == 2 else 'w3, the trigram frequency'})."
        )
        lines.append(
            "     They are per-BOARD and NOT attributions: sign(Δvalue) need not match "
            "sign(ms/char), because this surface prices some features counter-intuitively.)"
        )
    if top_ngrams_k:
        lines.append("")
        lines.append("  TOP N-GRAMS per leading column (ms/char, corpus-weighted)")
        for c in ch.ranked()[:6]:
            top = ch.top_ngrams(c.feature, top_ngrams_k)
            rendered = "  ".join(f"{ng!r}{v:+.4f}" for ng, v in top)
            lines.append(f"    {c.feature:<20} {rendered}")
    return lines


def _short(name: str, width: int = 9) -> str:
    """Shorten a layout name for a table header (a raw 30-char layout is not a column label)."""
    return name if len(name) <= width else name[: width - 1] + "…"


def format_report(
    diff: ShapDiff, top_bigrams_k: int = 5, columns: bool = False, top: int = 0
) -> str:
    """The human-readable report: reconciliation FIRST, then blocks, then columns on request.

    The order is deliberate twice over — a reader who meets the ranked features before the
    residuals can form a story about a table that does not sum to anything, and a reader who
    meets the 46-column table before the blocks can form a story about a credit split that is
    not unique.

    ``columns`` defaults to **False**: the per-column table is OPT-IN. It was opt-out through
    SHAPDIFF-1/-TCOND, when the audience was the author. The block table is the one whose
    numbers are invariant to how TreeSHAP redistributed credit among correlated columns, so the
    misleading table is the one a reader now has to ask for by name.
    """
    lines: list[str] = []
    a, b = diff.name_a, diff.name_b
    lines.append(f"COMPARE  {a} -> {b}   corpus={diff.corpus}  wpm={diff.target_wpm:g}")
    lines.append(f"  channel: {diff.channel}")
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
    lines.append(f"  gap vs shipped card (abs ms/char)      {diff.resid_vs_card_gap:.3e}")
    lines.append(f"  RECONCILES: {diff.reconciles()}")
    lines.append(
        f"  EXTERNAL GAUGE TIE: {'OK' if diff.gauge_tie_ok() else 'FAILED'}   "
        f"(bar {GAUGE_REFUSAL_MS:g} ms/char — the bar a WRONG WEIGHTING breaks and the "
        "internal sums-back identity cannot see)"
    )

    # THE REFUSAL. Not a warning: the tables are SUPPRESSED. A run whose external tie has moved
    # is decomposing some quantity other than the gauge's own, and it would do so
    # self-consistently — SHAPDIFF-1 measured the internal bars at ~1e-16 under a weighting
    # wrong by 5.6e-2 ms/char, and SHAPDIFF-TCOND measured that the analogous Tcond error
    # INVERTS which block leads. Printing an interpretable table under those conditions is the
    # one failure mode this tool exists to make impossible, so the tables do not print at all.
    if not diff.gauge_tie_ok():
        lines.append("")
        lines.append("=" * 88)
        lines.append("!! REFUSED: THE EXTERNAL GAUGE TIE FAILED — NO ATTRIBUTION TABLE IS SHOWN !!")
        lines.append("=" * 88)
        lines.append(
            f"  This run's gap is {diff.resid_vs_card_gap:.3e} ms/char away from the shipped "
            f"card() gauge (bar {GAUGE_REFUSAL_MS:g})."
        )
        for ch in (diff.t2, diff.tcond):
            if ch is not None and ch.resid_gap_vs_shipped > GAUGE_REFUSAL_MS:
                lines.append(
                    f"  channel {ch.channel}: gap {ch.gap:+.4f} vs shipped "
                    f"{ch.shipped_gap:+.4f}  (off by {ch.resid_gap_vs_shipped:.3e})"
                )
        lines.append(
            "  A decomposition can be internally self-consistent to ~1e-16 while decomposing "
            "the WRONG QUANTITY: both sides of the sums-back identity share the weight table, "
            "so a wrong weighting cancels out of it. Only this external tie catches that, and "
            "on the registered pair the analogous error INVERTED which block leads. The tables "
            "are therefore suppressed rather than annotated — there is nothing here to read."
        )
        if diff.t2 is not None and diff.t2.weighting != "trigram-marginal":
            lines.append(f"  NOTE: this run used weighting={diff.t2.weighting!r} (a control).")
        if diff.tcond is not None and diff.tcond.weighting != "trigram-direct":
            lines.append(
                f"  NOTE: this run used tcond_weighting={diff.tcond.weighting!r} (a control)."
            )
        return "\n".join(lines)

    if not diff.reconciles():
        lines.append("")
        lines.append("!! RECONCILIATION FAILED — the tables below explain NOTHING. !!")
    for ch in (diff.t2, diff.tcond):
        if ch is not None:
            lines.extend(_format_channel(diff, ch, top_bigrams_k, columns, top))
    lines.extend(_format_honesty(diff))
    return "\n".join(lines)


def _format_honesty(diff: ShapDiff) -> list[str]:
    """The caveat block — printed WITH the magnitudes, not filed somewhere a reader won't look.

    Two parts, both computed from this run rather than boilerplate: the cross-channel properties
    this pair actually carries twice (so the non-additivity refusal names names), and the four
    measured caveats.
    """
    lines = ["", "=" * 88, "HOW TO READ THIS (and how not to)", "=" * 88]
    lines.append(f"  ESTIMAND: {ESTIMAND}.")
    shared = diff.cross_channel_properties()
    if shared:
        lines.append("")
        lines.append(
            f"  ⚠ DO NOT ADD ACROSS CHANNELS. {len(shared)} properties appear in BOTH channels "
            "and each already"
        )
        lines.append(
            "    carries its own channel's full share, so summing them double-counts. Named:"
        )
        for i in range(0, len(shared), 8):
            lines.append("      " + "  ".join(shared[i : i + 8]))
        if diff.t2 is not None and diff.tcond is not None:
            lead = max(
                shared,
                key=lambda p: abs(
                    next(c.ms_per_char for c in diff.t2.contributions if c.feature == p)
                ),
            )
            t2_value = next(c.ms_per_char for c in diff.t2.contributions if c.feature == lead)
            lines.append(
                f"    e.g. {lead!r}: T2 {t2_value:+.4f} ms/char and Tcond bg1+bg2 "
                f"{diff.tcond.joint(lead):+.4f} ms/char are the same property on two frames — "
                f"NOT {t2_value + diff.tcond.joint(lead):+.4f}."
            )
    lines.append("")
    for caveat in CAVEATS:
        wrapped = _wrap(caveat, 84)
        lines.append(f"  * {wrapped[0]}")
        lines.extend(f"    {line}" for line in wrapped[1:])
    lines.append("")
    for line in _wrap(f"CANNOT: {CANNOT}", 84):
        lines.append(f"  {line}")
    return lines


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap. Local rather than ``textwrap`` so the report has no import surprise."""
    out: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        out.append(current)
    return out
