"""Decomposed strain objective for the keybo true-best-layout campaign.

The measured speed surfaces and the comfort prior are deliberately separate:

* Aalto T2 + conditioned T3 is the primary speed axis.
* Community and pool surfaces are evaluation gauges, not search objectives.
* The seven submetrics below are transparent comfort/strain mechanisms. Their
  outer aggregation is linear in corpus exposure; convexity exists only inside
  each event-level loss curve.

All default comfort coefficients marked OPEN are sensitivity defaults, not
measured milliseconds and not promotion evidence. Searchers should preserve the
registered Aalto 0.10 percentage-point plateau rule and report the raw axes.
The weighted ``combined`` value is a sensitivity scalar; ``rank_key`` enforces
the speed plateau before comparing comfort.

This module has no import-time model work. ``speed_axes_from_modules`` adapts
already-loaded ``p16_coopt`` / ``comm_opt1`` modules instead of importing them:
``comm_opt1`` trains three community surfaces as an import side effect.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

REPO = Path("/local/home/zegertho/repos/keybo")
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from keybo.analysis.community import community_suite, pinned_char  # noqa: E402
from keybo.analysis.select import usage_stats  # noqa: E402
from keybo.data.corpus import load_frequencies  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import (  # noqa: E402
    Finger,
    Geometry,
    Position,
    ROW_STAGGERED_30,
)
from keybo.layout import Layout  # noqa: E402
from keybo.scoring.inspect import layout_diagnostics  # noqa: E402

# Canonical C30M character order. It is also qwerty30m, so the identity
# permutation is the speed surfaces' qwerty reference.
C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
KEYBO_LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
LSB_SIB = "fyou,vgdnlheaikcstrmzj'.-pwbxq"
GRAPHITE = "bldwz'foujnrtsgyhaeixqmcvkp,.-"
SEMIMAK = "flhvz'wuoysrntkcdeaixjbmqpg,.-"
QWERTY = C30M
REFERENCE_LAYOUTS: Mapping[str, str] = MappingProxyType(
    {
        "keybo-lsb": KEYBO_LSB,
        "lsb-sib": LSB_SIB,
        "graphite": GRAPHITE,
        "semimak": SEMIMAK,
        "qwerty": QWERTY,
    }
)

# Canonical values from the checked-in C30M bigram/skipgram corpora. These are
# drift guards for the formulas and matrices, not fitted targets or evidence
# that the OPEN coefficients are calibrated.
SANITY_GOLDEN_VALUES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "keybo-lsb": MappingProxyType(
            {
                "sfb": 0.7853502262,
                "sfs": 0.5199944046,
                "lsb": 0.1855555029,
                "scissor": 0.2171293394,
                "pinky_load": 0.0,
                "row_jump": 2.1438453590,
                "redirect": 0.0,
            }
        ),
        "lsb-sib": MappingProxyType(
            {
                "sfb": 1.1311603258,
                "sfs": 0.4820653086,
                "lsb": 0.1961260540,
                "scissor": 0.2018346712,
                "pinky_load": 0.0,
                "row_jump": 2.1923330007,
                "redirect": 0.0,
            }
        ),
        "graphite": MappingProxyType(
            {
                "sfb": 0.8667634806,
                "sfs": 0.5124860525,
                "lsb": 0.1885449792,
                "scissor": 0.1948673472,
                "pinky_load": 0.0,
                "row_jump": 2.6094395976,
                "redirect": 0.0,
            }
        ),
        "semimak": MappingProxyType(
            {
                "sfb": 0.6290119100,
                "sfs": 0.4196189903,
                "lsb": 0.2386538305,
                "scissor": 0.4531281221,
                "pinky_load": 0.0004853032,
                "row_jump": 3.4788437705,
                "redirect": 0.0,
            }
        ),
        "qwerty": MappingProxyType(
            {
                "sfb": 7.2071204507,
                "sfs": 1.1319438506,
                "lsb": 1.3104465261,
                "scissor": 1.2165799148,
                "pinky_load": 0.0,
                "row_jump": 9.0323709958,
                "redirect": 0.0,
            }
        ),
    }
)

BIGRAM_PATH = REPO / "data" / "corpus" / "bigrams.txt"
SKIPGRAM_PATH = REPO / "data" / "corpus" / "1-skip.txt"

# OPEN sensitivity constants. None is presented as a fitted comfort parameter.
SFS_DECAY = 0.10
LSB_COMFORT_SPAN = 1.50
PINKY_CAPACITY = 0.09
NONADJACENT_SCISSOR_FACTOR = 0.60
PREFERRED_ORIENTATION_FACTOR = 0.20
AALTO_PLATEAU_WIDTH_PP = 0.10
WEIGHT_STATUS = (
    "OPEN sensitivity defaults in heterogeneous comfort units; "
    "not fitted milliseconds or promotion evidence"
)


def aalto_plateau_loss(
    candidate_loss_pp: float,
    reference_loss_pp: float,
    width_pp: float = AALTO_PLATEAU_WIDTH_PP,
) -> float:
    """Loss outside a fixed reference plateau; lower raw speed loss is better."""

    candidate = float(candidate_loss_pp)
    reference = float(reference_loss_pp)
    width = float(width_pp)
    if not all(math.isfinite(value) for value in (candidate, reference, width)):
        raise ValueError("Aalto plateau inputs must be finite")
    if width < 0.0:
        raise ValueError("Aalto plateau width must be non-negative")
    boundary = reference + width
    return 0.0 if candidate <= boundary else candidate - boundary


@dataclass(frozen=True)
class MetricSpec:
    """Public metadata for one decomposed comfort mechanism."""

    name: str
    weight: float
    loss_curve: str
    justification: str


SUBMETRIC_SPECS: Mapping[str, MetricSpec] = MappingProxyType(
    {
        "sfb": MetricSpec(
            name="sfb",
            weight=1.0,
            loss_curve=(
                "linear direct-bigram exposure; per event "
                "1 + [staggered_distance-1]_+^2 + 0.5[|dy|-1]_+^2"
            ),
            justification=(
                "Direct SFB has a measured +27/+32/+38 ms low/mid/high-WPM effect; "
                "weight 1.0 is an OPEN comfort-axis coefficient because speed already carries T2."
            ),
        ),
        "sfs": MetricSpec(
            name="sfs",
            weight=0.25,
            loss_curve=(
                "linear one-skip exposure times rho=0.10, then the SFB convex movement curve; "
                "deeper lag terms are absent"
            ),
            justification=(
                "Lag-2 timing is measured null, so only an OPEN weak comfort sensitivity remains; "
                "rho=0.10 and weight 0.25 keep it secondary without Oxey's 1/7 doctrine."
            ),
        ),
        "lsb": MetricSpec(
            name="lsb",
            weight=1.0,
            loss_curve=(
                "linear direct exposure; pair event [stagger-adjusted dx - 1.50]_+^2"
            ),
            justification=(
                "A convex comfort-envelope hinge avoids charging ordinary index-middle motion; "
                "the 1.50u threshold and weight are OPEN defaults."
            ),
        ),
        "scissor": MetricSpec(
            name="scissor",
            weight=1.0,
            loss_curve=(
                "linear direct exposure; convex two-row posture excess times measured "
                "finger-pair/row-direction plus OPEN A/B/C biomechanical-tier, "
                "adverse-orientation, and adjacency factors"
            ),
            justification=(
                "This closes the non-adjacent gap continuously; K31 supplies pair/direction "
                "ratios, while tier factors 1.15/1.00/0.85, 0.60 non-adjacent, and "
                "0.20 preferred are OPEN."
            ),
        ),
        "pinky_load": MetricSpec(
            name="pinky_load",
            weight=1.0,
            loss_curve="sum over pinkies [load_f / 0.09 - 1]_+^2",
            justification=(
                "Fatigue should accelerate only beyond capacity, not around equal load; "
                "9% per pinky and weight 1.0 are OPEN thresholds, not measured fatigue constants."
            ),
        ),
        "row_jump": MetricSpec(
            name="row_jump",
            weight=0.10,
            loss_curve=(
                "linear same-hand distinct-finger exposure; per event (|dy|/2)^2"
            ),
            justification=(
                "This is the intentional low base beneath the scissor interaction residual; "
                "event overlap is expected, and 0.10 is an OPEN coefficient."
            ),
        ),
        "redirect": MetricSpec(
            name="redirect",
            weight=0.0,
            loss_curve="constant zero after constituent T2/T3 transitions",
            justification=(
                "Measured redirects track roll continuation at every WPM band, so the categorical "
                "Oxey redirect residual is dropped."
            ),
        ),
    }
)

_FINGER_KIND = {
    Finger.LP: "pinky",
    Finger.LR: "ring",
    Finger.LM: "middle",
    Finger.LI: "index",
    Finger.RI: "index",
    Finger.RM: "middle",
    Finger.RR: "ring",
    Finger.RP: "pinky",
}
_KIND_ORDER = {"index": 0, "middle": 1, "ring": 2, "pinky": 3}

# K31 T2 means at WPM 90 from the audited exhaustive two-row table. These are
# descriptive fitted-surface ratios, not causal comfort coefficients.
PAIR_FITTED_MS: Mapping[tuple[str, str], float] = MappingProxyType(
    {
        ("index", "middle"): 163.5,
        ("index", "ring"): 147.6,
        ("index", "pinky"): 140.4,
        ("middle", "ring"): 175.0,
        ("middle", "pinky"): 147.5,
        ("ring", "pinky"): 185.1,
    }
)
PAIR_DIRECTION_MS: Mapping[tuple[str, str], Mapping[str, float]] = MappingProxyType(
    {
        ("index", "middle"): MappingProxyType(
            {"bottom_to_top": 150.8, "top_to_bottom": 176.2}
        ),
        ("index", "ring"): MappingProxyType(
            {"bottom_to_top": 140.7, "top_to_bottom": 154.4}
        ),
        ("index", "pinky"): MappingProxyType(
            {"bottom_to_top": 137.2, "top_to_bottom": 143.6}
        ),
        ("middle", "ring"): MappingProxyType(
            {"bottom_to_top": 157.4, "top_to_bottom": 192.5}
        ),
        ("middle", "pinky"): MappingProxyType(
            {"bottom_to_top": 141.3, "top_to_bottom": 153.6}
        ),
        ("ring", "pinky"): MappingProxyType(
            {"bottom_to_top": 162.6, "top_to_bottom": 207.7}
        ),
    }
)

# C's broad pair tiers are ordinal, not measured effect sizes. OPEN factors use
# mild +/-15% steps so the prior separates tiers without overwhelming K31 T2.
PAIR_BIOMECHANICAL_FACTORS: Mapping[tuple[str, str], float] = MappingProxyType(
    {
        ("index", "middle"): 1.00,
        ("index", "ring"): 0.85,
        ("index", "pinky"): 0.85,
        ("middle", "ring"): 1.15,
        ("middle", "pinky"): 1.00,
        ("ring", "pinky"): 1.15,
    }
)

# Community vertical-order prior: middle prefers highest, then ring, pinky,
# index. It is used only to identify the adverse orientation.
_PREFERRED_HEIGHT = {"index": 0, "pinky": 1, "ring": 2, "middle": 3}


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b), key=_KIND_ORDER.__getitem__))  # type: ignore[return-value]


def scissor_event_cost(
    source: Position,
    target: Position,
    geometry: Geometry = ROW_STAGGERED_30,
) -> float:
    """Continuous first-class residual for one ordered two-row motion.

    Unlike :func:`keybo.features.classify.is_scissor`, this includes
    non-adjacent distinct-finger reaches such as ``bl``. Preferred orientations
    retain only a small sensitivity residual; the generic row-jump axis carries
    their ordinary vertical motion.
    """

    if not C.same_hand(geometry, source, target):
        return 0.0
    if C.same_finger(geometry, source, target):
        return 0.0
    dy = abs(source[1] - target[1])
    if dy != 2:
        return 0.0

    source_kind = _FINGER_KIND[geometry.finger(source[0])]
    target_kind = _FINGER_KIND[geometry.finger(target[0])]
    pair = _pair_key(source_kind, target_kind)
    pair_mean = PAIR_FITTED_MS[pair]
    pair_factor = (
        pair_mean
        / PAIR_FITTED_MS[("index", "middle")]
        * PAIR_BIOMECHANICAL_FACTORS[pair]
    )

    direction = "bottom_to_top" if source[1] < target[1] else "top_to_bottom"
    direction_factor = PAIR_DIRECTION_MS[pair][direction] / pair_mean

    upper_kind = source_kind if source[1] > target[1] else target_kind
    lower_kind = target_kind if source[1] > target[1] else source_kind
    adverse = _PREFERRED_HEIGHT[upper_kind] < _PREFERRED_HEIGHT[lower_kind]
    orientation_factor = 1.0 if adverse else PREFERRED_ORIENTATION_FACTOR

    adjacency_factor = (
        1.0 if C.is_adjacent(geometry, source, target) else NONADJACENT_SCISSOR_FACTOR
    )
    posture_excess = max(0.0, float(dy) - 1.0)
    return (
        pair_factor
        * direction_factor
        * orientation_factor
        * adjacency_factor
        * posture_excess**2
    )


def _layout_string(layout: str | Layout) -> str:
    value = "".join(layout.chars) if isinstance(layout, Layout) else str(layout)
    if len(value) != len(C30M) or set(value) != set(C30M):
        raise ValueError(
            "layout must be a permutation of the 30-character C30M charset"
        )
    return value


def layout_to_permutation(layout: str | Layout) -> np.ndarray:
    """Return the p16-compatible char->slot permutation, with fixed space last."""

    value = _layout_string(layout)
    slot_of = {char: slot for slot, char in enumerate(value)}
    return np.asarray([slot_of[char] for char in C30M] + [len(C30M)], dtype=np.intp)


def _validated_frequency(ngram: str, frequency: int | float) -> float:
    try:
        value = float(frequency)
    except (TypeError, ValueError) as error:
        raise ValueError(f"frequency for {ngram!r} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"frequency for {ngram!r} must be finite")
    if value < 0.0:
        raise ValueError(f"negative frequency for {ngram!r}")
    return value


def _validate_permutation(
    permutation: np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    raw = np.asarray(permutation)
    size = len(C30M) + 1
    if raw.shape != (size,):
        raise ValueError(f"permutation must have shape ({size},)")
    if np.issubdtype(raw.dtype, np.floating):
        if not bool(np.isfinite(raw).all()):
            raise ValueError("permutation values must be finite")
        if not bool(np.equal(raw, np.floor(raw)).all()):
            raise ValueError("permutation values must be exact integers")
    elif not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("permutation values must be numeric integers")
    if bool(np.any(raw < 0)) or bool(np.any(raw >= size)):
        raise ValueError("permutation values must be valid slot indices")
    value = raw.astype(np.intp, copy=False)
    if value[-1] != size - 1 or set(value[:-1].tolist()) != set(range(size - 1)):
        raise ValueError(
            "permutation must map C30M onto slots 0..29 with fixed space at 30"
        )
    return value


def _frequency_probability(
    frequencies: Mapping[str, int | float],
) -> tuple[np.ndarray, float, float]:
    chars = C30M + " "
    index = {char: i for i, char in enumerate(chars)}
    matrix = np.zeros((len(chars), len(chars)), dtype=float)
    raw_total = 0.0
    for ngram, frequency in frequencies.items():
        value = _validated_frequency(ngram, frequency)
        if len(ngram) == 2:
            raw_total += value
        if len(ngram) != 2 or ngram[0] not in index or ngram[1] not in index:
            continue
        matrix[index[ngram[0]], index[ngram[1]]] += value
    covered_total = float(matrix.sum())
    if not math.isfinite(raw_total) or not math.isfinite(covered_total):
        raise ValueError("frequency totals must be finite")
    if not covered_total:
        raise ValueError("frequency table has no C30M-covered bigrams")
    return matrix / covered_total, covered_total, raw_total


def _select_press_probability(
    frequencies: Mapping[str, int | float],
) -> np.ndarray:
    """Match SELECT-1 first-character mass, irrespective of the second char."""

    index = {char: i for i, char in enumerate(C30M)}
    probability = np.zeros(len(C30M) + 1, dtype=float)
    for ngram, frequency in frequencies.items():
        value = _validated_frequency(ngram, frequency)
        if len(ngram) == 2 and ngram[0] in index:
            probability[index[ngram[0]]] += value
    total = float(probability.sum())
    if not math.isfinite(total):
        raise ValueError("SELECT-1 first-character frequency total must be finite")
    if not total:
        raise ValueError("bigram table has no SELECT-1 first-character C30M mass")
    return probability / total


def _pair_value(
    probability: np.ndarray, cost: np.ndarray, permutation: np.ndarray
) -> float:
    placed_cost = cost[permutation[:, None], permutation[None, :]]
    return 100.0 * float(np.sum(probability * placed_cost))


class PairSubmetric:
    """Corpus-weighted positional event loss."""

    def __init__(self, spec: MetricSpec, probability: np.ndarray, cost: np.ndarray):
        self.name = spec.name
        self.WEIGHT = float(spec.weight)
        self.LOSS_CURVE = spec.loss_curve
        self.JUSTIFICATION = spec.justification
        self._probability = probability
        self._cost = cost

    def value(self, layout: str | Layout) -> float:
        return self.value_permutation(layout_to_permutation(layout))

    def value_permutation(self, permutation: np.ndarray) -> float:
        return _pair_value(
            self._probability, self._cost, _validate_permutation(permutation)
        )


class PinkyLoadSubmetric:
    """Per-pinky SELECT-1 press load above an explicit capacity threshold."""

    def __init__(
        self,
        spec: MetricSpec,
        press_probability: np.ndarray,
        slot_fingers: np.ndarray,
    ):
        self.name = spec.name
        self.WEIGHT = float(spec.weight)
        self.LOSS_CURVE = spec.loss_curve
        self.JUSTIFICATION = spec.justification
        self.BASIS = "SELECT-1 first-character C30M mass, excluding leading space"
        self._press_probability = press_probability
        self._slot_fingers = slot_fingers

    def loads_permutation(self, permutation: np.ndarray) -> dict[Finger, float]:
        permutation = _validate_permutation(permutation)
        char_fingers = self._slot_fingers[permutation]
        return {
            finger: float(self._press_probability[char_fingers == finger].sum())
            for finger in (Finger.LP, Finger.RP)
        }

    def value(self, layout: str | Layout) -> float:
        return self.value_permutation(layout_to_permutation(layout))

    def value_permutation(self, permutation: np.ndarray) -> float:
        loads = self.loads_permutation(permutation)
        return sum(
            max(0.0, load / PINKY_CAPACITY - 1.0) ** 2 for load in loads.values()
        )


class ZeroSubmetric:
    """An explicitly dropped residual retained in the public decomposition."""

    def __init__(self, spec: MetricSpec):
        self.name = spec.name
        self.WEIGHT = float(spec.weight)
        self.LOSS_CURVE = spec.loss_curve
        self.JUSTIFICATION = spec.justification

    def value(self, layout: str | Layout) -> float:
        _layout_string(layout)
        return 0.0

    def value_permutation(self, permutation: np.ndarray) -> float:
        _validate_permutation(permutation)
        return 0.0


def _positional_matrices(
    geometry: Geometry,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    positions = (*geometry.slots, geometry.space_position)
    size = len(positions)
    costs = {
        name: np.zeros((size, size), dtype=float)
        for name in ("sfb", "sfs", "lsb", "scissor", "row_jump")
    }
    masks = {
        name: np.zeros((size, size), dtype=float)
        for name in (
            "sfb",
            "lsb",
            "row_jump",
            "adjacent_scissor",
            "broad_scissor",
        )
    }

    for i, source in enumerate(positions):
        for j, target in enumerate(positions):
            if source == target:
                continue
            dy = abs(source[1] - target[1])

            if C.same_finger(geometry, source, target):
                dx = geometry.stagger_adjusted_dx(source, target)
                distance = math.hypot(dx, float(dy))
                travel_excess = max(0.0, distance - 1.0)
                posture_excess = max(0.0, float(dy) - 1.0)
                event_cost = 1.0 + travel_excess**2 + 0.5 * posture_excess**2
                costs["sfb"][i, j] = event_cost
                costs["sfs"][i, j] = SFS_DECAY * event_cost
                masks["sfb"][i, j] = 1.0
                continue

            if not C.same_hand(geometry, source, target):
                continue

            if C.is_lsb(geometry, source, target):
                horizontal_excess = max(
                    0.0,
                    geometry.stagger_adjusted_dx(source, target) - LSB_COMFORT_SPAN,
                )
                costs["lsb"][i, j] = horizontal_excess**2
                masks["lsb"][i, j] = 1.0

            if dy:
                costs["row_jump"][i, j] = (float(dy) / 2.0) ** 2
                masks["row_jump"][i, j] = 1.0

            scissor_cost = scissor_event_cost(source, target, geometry)
            if scissor_cost:
                costs["scissor"][i, j] = scissor_cost
                masks["broad_scissor"][i, j] = 1.0
            if C.is_scissor(geometry, source, target):
                masks["adjacent_scissor"][i, j] = 1.0

    return costs, masks


class ComfortObjective:
    """Reusable decomposed objective with layout and permutation entry points."""

    def __init__(
        self,
        bigrams: Mapping[str, int | float] | None = None,
        skipgrams: Mapping[str, int | float] | None = None,
        geometry: Geometry = ROW_STAGGERED_30,
    ):
        if len(geometry.slots) != len(C30M):
            raise ValueError(
                f"C30M objective requires exactly 30 movable slots; "
                f"geometry has {len(geometry.slots)}"
            )
        bigrams = bigrams if bigrams is not None else load_frequencies(str(BIGRAM_PATH))
        skipgrams = (
            skipgrams if skipgrams is not None else load_frequencies(str(SKIPGRAM_PATH))
        )
        self.geometry = geometry
        (
            self._bigram_probability,
            self.bigram_total,
            self.bigram_raw_total,
        ) = _frequency_probability(bigrams)
        (
            self._skip_probability,
            self.skipgram_total,
            self.skipgram_raw_total,
        ) = _frequency_probability(skipgrams)
        self.corpus_coverage: Mapping[str, float] = MappingProxyType(
            {
                "bigram": self.bigram_total / self.bigram_raw_total,
                "skipgram": self.skipgram_total / self.skipgram_raw_total,
            }
        )
        self._endpoint_probability = (
            self._bigram_probability.sum(axis=0) + self._bigram_probability.sum(axis=1)
        ) / 2.0
        self._capacity_probability = _select_press_probability(bigrams)

        positions = (*geometry.slots, geometry.space_position)
        self._slot_fingers = np.asarray(
            [geometry.finger(position[0]) for position in positions],
            dtype=object,
        )
        costs, self._masks = _positional_matrices(geometry)
        pinky = PinkyLoadSubmetric(
            SUBMETRIC_SPECS["pinky_load"],
            self._capacity_probability,
            self._slot_fingers,
        )
        self.submetrics: Mapping[str, Any] = MappingProxyType(
            {
                "sfb": PairSubmetric(
                    SUBMETRIC_SPECS["sfb"], self._bigram_probability, costs["sfb"]
                ),
                "sfs": PairSubmetric(
                    SUBMETRIC_SPECS["sfs"], self._skip_probability, costs["sfs"]
                ),
                "lsb": PairSubmetric(
                    SUBMETRIC_SPECS["lsb"], self._bigram_probability, costs["lsb"]
                ),
                "scissor": PairSubmetric(
                    SUBMETRIC_SPECS["scissor"],
                    self._bigram_probability,
                    costs["scissor"],
                ),
                "pinky_load": pinky,
                "row_jump": PairSubmetric(
                    SUBMETRIC_SPECS["row_jump"],
                    self._bigram_probability,
                    costs["row_jump"],
                ),
                "redirect": ZeroSubmetric(SUBMETRIC_SPECS["redirect"]),
            }
        )

    def values(self, layout: str | Layout) -> dict[str, float]:
        return self.values_permutation(layout_to_permutation(layout))

    def values_permutation(self, permutation: np.ndarray) -> dict[str, float]:
        permutation = _validate_permutation(permutation)
        return {
            name: metric.value_permutation(permutation)
            for name, metric in self.submetrics.items()
        }

    def event_shares(self, layout: str | Layout) -> dict[str, float]:
        """Raw event shares used to check overlap with existing diagnostics."""

        permutation = layout_to_permutation(layout)

        def direct_share(mask: np.ndarray) -> float:
            return _pair_value(self._bigram_probability, mask, permutation)

        pinky_metric = self.submetrics["pinky_load"]
        capacity_loads = pinky_metric.loads_permutation(permutation)
        char_fingers = self._slot_fingers[permutation]
        endpoint_loads = {
            finger: float(self._endpoint_probability[char_fingers == finger].sum())
            for finger in (Finger.LP, Finger.RP)
        }
        broad_scissor = self._masks["broad_scissor"]
        return {
            "sfb_pct": direct_share(self._masks["sfb"]),
            "lsb_pct": direct_share(self._masks["lsb"]),
            "adjacent_scissor_pct": direct_share(self._masks["adjacent_scissor"]),
            "broad_scissor_pct": direct_share(broad_scissor),
            "scissor_row_jump_overlap_pct": direct_share(
                broad_scissor * self._masks["row_jump"]
            ),
            "scissor_lsb_overlap_pct": direct_share(broad_scissor * self._masks["lsb"]),
            "left_pinky_capacity_pct": 100.0 * capacity_loads[Finger.LP],
            "right_pinky_capacity_pct": 100.0 * capacity_loads[Finger.RP],
            "pinky_capacity_pct": 100.0
            * (capacity_loads[Finger.LP] + capacity_loads[Finger.RP]),
            "pinky_pct": 100.0
            * (capacity_loads[Finger.LP] + capacity_loads[Finger.RP]),
            "left_pinky_endpoint_pct": 100.0 * endpoint_loads[Finger.LP],
            "right_pinky_endpoint_pct": 100.0 * endpoint_loads[Finger.RP],
            "pinky_endpoint_pct": 100.0
            * (endpoint_loads[Finger.LP] + endpoint_loads[Finger.RP]),
        }

    @staticmethod
    def _require_close(label: str, actual: float, expected: float) -> None:
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")

    def assert_component_overlap(self, layout: str | Layout) -> dict[str, Any]:
        """Assert existing component identities and shared event-mask semantics.

        Native tool component values are not expected to equal the recurred
        submetrics: they use different corpora, geometry, units, and integer
        quantization. The exact overlap is (1) their documented additive
        component/prime identities and (2) our SFB/LSB/adjacent-scissor/pinky
        event masks, which must equal ``layout_diagnostics`` on the same corpus.
        """

        layout_string = _layout_string(layout)
        layout_object = Layout(layout_string, self.geometry)
        shares = self.event_shares(layout_string)
        diagnostics = layout_diagnostics(layout_object, self._raw_bigram_counts())
        self._require_close(
            "sfb mask", shares["sfb_pct"], 100.0 * diagnostics["sfb_share"]
        )
        self._require_close(
            "lsb mask", shares["lsb_pct"], 100.0 * diagnostics["lsb_share"]
        )
        self._require_close(
            "adjacent scissor mask",
            shares["adjacent_scissor_pct"],
            100.0 * diagnostics["scissor_share"],
        )
        self._require_close(
            "endpoint pinky load",
            shares["pinky_endpoint_pct"],
            100.0
            * (
                diagnostics["finger_load"]["L-pinky"]
                + diagnostics["finger_load"]["R-pinky"]
            ),
        )
        select_usage = usage_stats(layout_string, self._select_first_character_mass())
        self._require_close(
            "SELECT pinky capacity load",
            shares["pinky_capacity_pct"],
            select_usage["pinky_pct"],
        )

        genkey, oxey1, oxey2 = community_suite(pinned_char(layout_string))
        gk = genkey.components(layout_string)
        o1 = oxey1.components(layout_string)
        o2 = oxey2.components(layout_string)
        self._require_close(
            "genkey components",
            genkey.score(layout_string),
            3.0 * gk["fspeed"] + gk["lsb_pct"] + 0.3 * gk["index_imbalance_pct"],
        )
        self._require_close(
            "genkey prime",
            genkey.score_primed(layout_string),
            gk["lsb_pct"] + 0.3 * gk["index_imbalance_pct"],
        )
        if oxey1.score(layout_string) != (
            o1["fspeed"] + o1["stretch"] + o1["pinky_ring"] + o1["trigrams"]
        ):
            raise AssertionError("oxey1 components do not reconstruct score")
        if oxey1.score_primed(layout_string) != o1["stretch"] + o1["pinky_ring"]:
            raise AssertionError("oxey1 prime does not match strain components")
        if oxey2.score(layout_string) != o2["wfd"] + o2["stretch"]:
            raise AssertionError("oxey2 components do not reconstruct score")
        if oxey2.score_primed(layout_string) != o2["stretch"]:
            raise AssertionError("oxey2 prime does not match stretch component")
        return {
            "genkey": gk,
            "oxey1": o1,
            "oxey2": o2,
            "diagnostics": {
                "objective": shares,
                "existing": diagnostics,
                "select_usage": select_usage,
            },
        }

    def _raw_bigram_counts(self) -> dict[str, float]:
        """Recreate a mapping for the existing diagnostics from normalized data."""

        chars = C30M + " "
        scale = self.bigram_total
        return {
            a + b: float(self._bigram_probability[i, j] * scale)
            for i, a in enumerate(chars)
            for j, b in enumerate(chars)
            if self._bigram_probability[i, j]
        }

    def _select_first_character_mass(self) -> dict[str, float]:
        """SELECT-1's first-character C30M mass; leading space is excluded."""

        return {
            char: float(self._capacity_probability[i]) for i, char in enumerate(C30M)
        }

    def score(
        self,
        layout: str | Layout,
        weights: Mapping[str, float] | None = None,
        speed_axes: SpeedAxes | None = None,
        include_gauges: bool = False,
    ) -> dict[str, Any]:
        return self.score_permutation(
            layout_to_permutation(layout),
            weights=weights,
            speed_axes=speed_axes,
            include_gauges=include_gauges,
        )

    def score_permutation(
        self,
        permutation: np.ndarray,
        weights: Mapping[str, float] | None = None,
        speed_axes: SpeedAxes | None = None,
        include_gauges: bool = False,
    ) -> dict[str, Any]:
        """Return raw values, policy-aware losses, speed axes, and rank key."""

        permutation = _validate_permutation(permutation)
        effective_weights = {
            name: metric.WEIGHT for name, metric in self.submetrics.items()
        }
        if speed_axes:
            effective_weights.update(
                {name: axis.WEIGHT for name, axis in speed_axes.axes.items()}
            )
        if weights:
            unknown = set(weights) - set(effective_weights)
            if unknown:
                raise ValueError(f"unknown objective weight(s): {sorted(unknown)}")
            effective_weights.update(
                {name: float(value) for name, value in weights.items()}
            )
        if not all(math.isfinite(value) for value in effective_weights.values()):
            raise ValueError("objective weights must be finite")
        if any(value < 0.0 for value in effective_weights.values()):
            raise ValueError("objective weights must be non-negative")
        if speed_axes:
            for name, axis in speed_axes.axes.items():
                if axis.role == "gauge" and effective_weights[name] != 0.0:
                    raise ValueError(
                        f"{name} is an evaluation gauge and cannot be weighted"
                    )

        comfort_values = self.values_permutation(permutation)
        speed_values = (
            speed_axes.values_permutation(
                permutation,
                include_gauges=include_gauges,
            )
            if speed_axes
            else {}
        )
        values = {**comfort_values, **speed_values}
        losses = {
            name: float(value) * effective_weights[name]
            for name, value in comfort_values.items()
        }
        speed_policy = {}
        if speed_axes:
            for name, value in speed_values.items():
                axis = speed_axes.axes[name]
                policy = axis.policy(value)
                speed_policy[name] = policy
                losses[name] = policy["loss"] * effective_weights[name]
        comfort = sum(losses[name] for name in comfort_values)
        speed = sum(losses[name] for name in speed_values)
        aalto_policy = speed_policy.get("speed_aalto")
        plateau_excess = float(aalto_policy["loss"]) if aalto_policy else 0.0
        plateau_pass = bool(aalto_policy["pass"]) if aalto_policy else True
        return {
            "values": values,
            "weights": effective_weights,
            "losses": losses,
            "comfort": comfort,
            "speed_loss": speed,
            "speed": speed_values,
            "speed_policy": speed_policy,
            "plateau_pass": plateau_pass,
            "rank_key": (0 if plateau_pass else 1, plateau_excess, comfort),
            "combined": comfort + speed,
            "weight_status": WEIGHT_STATUS,
        }


@dataclass(frozen=True)
class SpeedAxis:
    """One qwerty-relative measured speed surface (lower is better)."""

    name: str
    scorer: Callable[[np.ndarray], float]
    qwerty_value: float
    WEIGHT: float
    role: str
    JUSTIFICATION: str
    reference_value: float | None = None
    plateau_width_pp: float = 0.0

    def value_permutation(self, permutation: np.ndarray) -> float:
        raw = float(self.scorer(_validate_permutation(permutation)))
        if not math.isfinite(raw):
            raise ValueError(f"{self.name} returned non-finite value")
        return 100.0 * (raw / self.qwerty_value - 1.0)

    def policy(self, value: float) -> dict[str, Any]:
        """Apply role and fixed-reference plateau policy to a raw axis value."""

        raw = float(value)
        if self.role == "gauge":
            return {"role": self.role, "loss": 0.0, "pass": True}
        if self.reference_value is None:
            raise ValueError(f"{self.name} primary axis lacks a fixed reference")
        loss = aalto_plateau_loss(
            raw,
            self.reference_value,
            self.plateau_width_pp,
        )
        return {
            "role": self.role,
            "loss": loss,
            "pass": loss == 0.0,
            "candidate_loss_pp": raw,
            "reference_loss_pp": self.reference_value,
            "width_pp": self.plateau_width_pp,
            "maximum_loss_pp": self.reference_value + self.plateau_width_pp,
        }


class SpeedAxes:
    """Collection of measured speed axes kept separate from comfort."""

    def __init__(self, axes: Mapping[str, SpeedAxis]):
        copied = dict(axes)
        collisions = set(copied).intersection(SUBMETRIC_SPECS)
        if collisions:
            raise ValueError(
                "speed axis names must not collide with comfort axes: "
                f"{sorted(collisions)}"
            )

        def finite_real(name: str, label: str, value: Any) -> float:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} {label} must be a finite real number")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} {label} must be finite")
            return converted

        for name, axis in copied.items():
            if axis.name != name:
                raise ValueError(
                    f"speed axis key {name!r} does not match {axis.name!r}"
                )
            if not callable(axis.scorer):
                raise TypeError(f"{name} scorer must be callable")
            if axis.role not in {"primary", "gauge"}:
                raise ValueError(f"{name} has unknown role {axis.role!r}")
            qwerty_value = finite_real(name, "qwerty baseline", axis.qwerty_value)
            if qwerty_value <= 0.0:
                raise ValueError(f"{name} qwerty baseline must be positive")
            weight = finite_real(name, "weight", axis.WEIGHT)
            if weight < 0.0:
                raise ValueError(f"{name} weight must be non-negative")
            plateau_width = finite_real(
                name,
                "plateau width",
                axis.plateau_width_pp,
            )
            if plateau_width < 0.0:
                raise ValueError(f"{name} plateau width must be non-negative")
            if axis.role == "gauge" and weight != 0.0:
                raise ValueError(f"{name} gauge weight must be zero")
            if axis.role == "primary" and axis.reference_value is None:
                raise ValueError(f"{name} primary axis requires a fixed reference")
            if axis.reference_value is not None:
                finite_real(name, "reference value", axis.reference_value)
        primaries = [name for name, axis in copied.items() if axis.role == "primary"]
        if primaries != ["speed_aalto"]:
            raise ValueError(
                "speed axes require exactly one primary named 'speed_aalto'; "
                "all remaining axes must be gauges"
            )
        self.axes: Mapping[str, SpeedAxis] = MappingProxyType(copied)

    def values_permutation(
        self,
        permutation: np.ndarray,
        *,
        include_gauges: bool = False,
    ) -> dict[str, float]:
        return {
            name: axis.value_permutation(permutation)
            for name, axis in self.axes.items()
            if include_gauges or axis.role != "gauge"
        }


def speed_axes_from_callables(
    *,
    aalto: Callable[[np.ndarray], float],
    comm: Callable[[np.ndarray], float] | None = None,
    pool: Callable[[np.ndarray], float] | None = None,
    qwerty_permutation: np.ndarray | None = None,
    aalto_reference_permutation: np.ndarray | None = None,
) -> SpeedAxes:
    """Build qwerty-relative speed axes without importing model drivers."""

    qwerty = _validate_permutation(
        np.arange(len(C30M) + 1, dtype=np.intp)
        if qwerty_permutation is None
        else qwerty_permutation
    )
    reference = _validate_permutation(
        layout_to_permutation(KEYBO_LSB)
        if aalto_reference_permutation is None
        else aalto_reference_permutation
    )

    def make(
        name: str,
        function: Callable[[np.ndarray], float],
        weight: float,
        role: str,
        justification: str,
        reference_permutation: np.ndarray | None = None,
        plateau_width_pp: float = 0.0,
    ) -> SpeedAxis:
        baseline = float(function(qwerty))
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError(f"{name} qwerty baseline must be finite and positive")
        reference_value = None
        if reference_permutation is not None:
            reference_raw = float(function(reference_permutation))
            if not math.isfinite(reference_raw) or reference_raw <= 0.0:
                raise ValueError(f"{name} reference must be finite and positive")
            reference_value = 100.0 * (reference_raw / baseline - 1.0)
        return SpeedAxis(
            name,
            function,
            baseline,
            weight,
            role,
            justification,
            reference_value,
            plateau_width_pp,
        )

    axes = {
        "speed_aalto": make(
            "speed_aalto",
            aalto,
            1.0,
            "primary",
            "K31 Aalto T2 + conditioned T3, three-seed mean; 0.10pp differences are noise.",
            reference,
            AALTO_PLATEAU_WIDTH_PP,
        )
    }
    if comm is not None:
        axes["speed_comm"] = make(
            "speed_comm",
            comm,
            0.0,
            "gauge",
            "Four-community-typist style-fit gauge; qwerty level is extrapolated and n=4.",
        )
    if pool is not None:
        axes["speed_pool"] = make(
            "speed_pool",
            pool,
            0.0,
            "gauge",
            "All-data evaluation gauge; campaign governance forbids it as a search objective.",
        )
    return SpeedAxes(axes)


def speed_axes_from_modules(
    p16_module: Any,
    comm_module: Any | None = None,
    pool: Callable[[np.ndarray], float] | Any | None = None,
) -> SpeedAxes:
    """Adapt the existing p16/comm/pool machinery after callers load it."""

    qwerty = _validate_permutation(p16_module.perm_of(C30M))
    reference = _validate_permutation(p16_module.perm_of(KEYBO_LSB))
    comm_fn = None
    if comm_module is not None:
        comm_fn = getattr(comm_module, "fit_comm", None)
        if not callable(comm_fn):
            raise TypeError("comm module must expose fit_comm(permutation)")
    if pool is None or callable(pool):
        pool_fn = pool
    else:
        pool_fn = getattr(pool, "fit_pool", None)
        if pool_fn is None:
            raise TypeError("pool module must expose fit_pool(permutation)")
    return speed_axes_from_callables(
        aalto=p16_module.fit_speed,
        comm=comm_fn,
        pool=pool_fn,
        qwerty_permutation=qwerty,
        aalto_reference_permutation=reference,
    )


@lru_cache(maxsize=1)
def default_objective() -> ComfortObjective:
    return ComfortObjective()


def scorer(
    layout: str | Layout,
    weights: Mapping[str, float] | None = None,
    *,
    objective: ComfortObjective | None = None,
    speed_axes: SpeedAxes | None = None,
    include_gauges: bool = False,
) -> dict[str, Any]:
    """Convenience combined scorer returning per-axis losses plus ``combined``."""

    objective = objective or default_objective()
    return objective.score(
        layout,
        weights=weights,
        speed_axes=speed_axes,
        include_gauges=include_gauges,
    )


def sanity_rows(
    objective: ComfortObjective | None = None,
) -> dict[str, dict[str, float]]:
    objective = objective or default_objective()
    return {
        name: objective.values(layout) for name, layout in REFERENCE_LAYOUTS.items()
    }


def print_sanity_table(objective: ComfortObjective | None = None) -> None:
    objective = objective or default_objective()
    rows = sanity_rows(objective)
    names = list(objective.submetrics)
    print("layout       " + " ".join(f"{name:>12}" for name in names))
    for layout_name, values in rows.items():
        print(
            f"{layout_name:<12}" + " ".join(f"{values[name]:12.6f}" for name in names)
        )


def run_self_check(
    objective: ComfortObjective | None = None,
) -> dict[str, dict[str, float]]:
    """Run canonical goldens, component overlap, and scissor-bench assertions."""

    objective = objective or default_objective()
    rows = sanity_rows(objective)
    for name, layout in REFERENCE_LAYOUTS.items():
        values = rows[name]
        if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
            raise AssertionError(f"{name} has invalid submetric value")
        expected = SANITY_GOLDEN_VALUES[name]
        if set(values) != set(expected):
            raise AssertionError(f"{name} golden axes do not match objective axes")
        for axis, golden in expected.items():
            objective._require_close(
                f"{name} golden {axis}",
                values[axis],
                golden,
            )
        objective.assert_component_overlap(layout)
        result = objective.score(layout)
        objective._require_close(
            f"{name} combined decomposition",
            result["combined"],
            sum(result["losses"].values()),
        )

    bl = scissor_event_cost((3, 1), (5, 3))
    ring_pinky = scissor_event_cost((4, 1), (5, 3))
    if not math.isfinite(bl) or bl <= 0.0:
        raise AssertionError("bl scissor residual must be finite and non-zero")
    if ring_pinky <= bl:
        raise AssertionError(
            "ring-pinky must exceed middle-pinky in the cheap direction"
        )
    print_sanity_table(objective)
    print(f"scissor bench: bl={bl:.6f} ring-pinky={ring_pinky:.6f}")
    return rows


__all__ = [
    "AALTO_PLATEAU_WIDTH_PP",
    "BIGRAM_PATH",
    "C30M",
    "ComfortObjective",
    "GRAPHITE",
    "KEYBO_LSB",
    "LSB_COMFORT_SPAN",
    "LSB_SIB",
    "MetricSpec",
    "NONADJACENT_SCISSOR_FACTOR",
    "PAIR_BIOMECHANICAL_FACTORS",
    "PAIR_DIRECTION_MS",
    "PAIR_FITTED_MS",
    "PINKY_CAPACITY",
    "PREFERRED_ORIENTATION_FACTOR",
    "QWERTY",
    "REFERENCE_LAYOUTS",
    "SANITY_GOLDEN_VALUES",
    "SEMIMAK",
    "SFS_DECAY",
    "SKIPGRAM_PATH",
    "SUBMETRIC_SPECS",
    "SpeedAxis",
    "SpeedAxes",
    "WEIGHT_STATUS",
    "aalto_plateau_loss",
    "default_objective",
    "layout_to_permutation",
    "print_sanity_table",
    "run_self_check",
    "sanity_rows",
    "scissor_event_cost",
    "scorer",
    "speed_axes_from_callables",
    "speed_axes_from_modules",
]


if __name__ == "__main__":
    run_self_check()
