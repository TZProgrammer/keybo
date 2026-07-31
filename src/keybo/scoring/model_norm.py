"""Per-model normalized gauges (`aalto-n` / `comm-n` / `pool-n`) and their weighted blend.

WHAT THIS IS
------------
The three fitted model surfaces (AALTO / COMMUNITY / POOL) predict typing time in
milliseconds, but **on three different scales** — so "weight the three models" was
uninterpretable before this module: a weight multiplied a quantity whose range differed per
model. Each gauge here maps one surface's fit onto a common 0-1 scale::

    norm_m(L) = (zero_m - fit_m(L)) / (zero_m - one_m)

* ``zero_m`` — the **mean** fit over a fixed pool of uniformly random layouts.
* ``one_m`` — the best fit a per-model search found (that model's own optimum).

``fit`` is predicted TIME, so lower is faster and the numerator is **inverted on purpose**:
``fit == zero_m`` maps to 0, ``fit == one_m`` maps to 1, and **higher normalized is better**.
The blend then negates it, because the optimizer minimizes (see :class:`ModelBlendScorer`).

THE DIRECTION GUARD IS THE OPTIMUM, NOT QWERTY
----------------------------------------------
The tempting sanity check "qwerty should score ~0" is **FALSE** and would invert the sign if
"fixed". Qwerty normalizes to roughly 0.42-0.56 because it sits at the 0.00-0.20 percentile of
a random pool while this scale's zero is the pool **MEAN**, not its floor. The correct guard —
the one :meth:`Anchors.assert_direction` enforces — is that **each model's own optimum
normalizes to exactly 1.0** and the pool mean to exactly 0.0.

WHAT THE SCALE COSTS, STATED RATHER THAN BURIED
----------------------------------------------
A random-layout zero **wastes most of the scale**: every layout anyone would actually consider
is crammed into the top tenth of the range. The gauge is well-posed but low-resolution exactly
where it is used, and :func:`interpretation_note` says so on every report.

FRAME AND UNITS (both stated because both have cost this campaign a retraction)
------------------------------------------------------------------------------
* The surfaces are the **shipped** ``.standardized`` arrays resolved by
  :mod:`keybo.analysis.surfaces` — geometry-only ``g``, ``b(ngram)`` excluded.
* They are **BAKED at 90 WPM** and cannot be re-evaluated at another WPM.
* ⚠ On the standardized frame ``standardized - native`` is *exactly* independent of the third
  slot, i.e. standardization substitutes AALTO's bigram tensor into all three sources, which
  leaves them differing only in their conditional trigram increment. **The three sources are
  therefore LESS independent here than on the native arrays.** This is a property of the
  shipped surfaces, not of this module, and :func:`frame_caveat` reports it.

MODELLED ONLY: nothing here is a claim about realized typing speed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from keybo.analysis import surfaces as S
from keybo.layout import Layout
from keybo.scoring.base import IScorer

#: Gauge name per surface pool — the reportable column names.
GAUGE_OF_POOL: dict[str, str] = {"AALTO": "aalto-n", "COMMUNITY": "comm-n", "POOL": "pool-n"}
#: Reverse of :data:`GAUGE_OF_POOL`.
POOL_OF_GAUGE: dict[str, str] = {v: k for k, v in GAUGE_OF_POOL.items()}
#: Gauge names in report order.
GAUGE_NAMES: tuple[str, ...] = tuple(GAUGE_OF_POOL[p] for p in S.POOLS)

#: Anchor-artifact schema version. Bump when a field's MEANING changes.
ANCHOR_SCHEMA = "normgauge-anchors-1"

#: Batch tile for the histogram matmul. LOAD-BEARING, not a tuning knob: BLAS selects its
#: kernel from the operand shape, so an unpadded ``(B, 29791) @ (29791, m)`` makes one
#: layout's fit depend on how many OTHER layouts share its batch (~1e-14 rel). Every matmul
#: is issued at exactly this tile with the final partial tile zero-padded, which makes the
#: result bit-identical regardless of batch length. Pinning the tile cannot make the answer
#: independent of the tile ITSELF (the tile size *is* the operand shape), so the value is
#: frozen here and recorded in the anchor provenance alongside the numpy version.
TILE = 16

#: Tolerance for the direction guard. The identities are exact up to float rounding of an
#: affine map, so this is a rounding budget, not a fit tolerance.
DIRECTION_TOL = 1e-9


def frame_caveat() -> str:
    """The independence caveat that belongs on every report this gauge appears in."""
    return (
        "the shipped .standardized surfaces share AALTO's bigram tensor (standardization "
        "substitutes it), so the three sources differ only in their conditional trigram "
        "increment and are LESS independent than the .native arrays; POOL is additionally "
        "fitted on the union of AALTO's and COMMUNITY's sources"
    )


def interpretation_note() -> str:
    """What a normalized value means, including the resolution cost."""
    return (
        "0 = the mean of a random-layout pool, 1 = that model's own searched optimum; higher "
        "is better. A searched optimum bounds the true optimum from one side only, so every "
        "value is an UPPER bound on the true normalized score. Because the zero is the random "
        "pool's MEAN, realistic layouts occupy only the top fraction of the range -- the scale "
        "is well-posed but low-resolution where it is actually used, and qwerty sits near 0.5, "
        "NOT near 0"
    )


# ---------------------------------------------------------------------------
# the batched, shape-stable evaluator
# ---------------------------------------------------------------------------
class SurfaceFits:
    """Corpus-weighted fits of many layouts against many surfaces, bit-stable in batch size.

    ``fits(layouts)`` returns an ``(n_layouts, n_pools)`` array of predicted ms on the served
    geometry frame. Parity with the shipped :func:`keybo.analysis.surfaces.score_fit` is
    asserted by the test suite, and bit-exactness across batch lengths is asserted by
    :meth:`assert_batch_invariant` (the guard that catches the BLAS shape-dependence class).
    """

    def __init__(
        self,
        pools: Sequence[str] = S.POOLS,
        *,
        family: str = S.DEFAULT_FAMILY,
        corpus: str | None = None,
        surface_dir: str | None = None,
        trigram_path: str | None = None,
    ) -> None:
        self.pools = tuple(pools)
        self.family = family
        self.trigram_path = trigram_path or S.default_trigram_path(corpus)
        first, second, third, freq = S.trigram_objective(self.trigram_path)
        # int64 for the flat index arithmetic: 31^3 fits easily, but int32 promotion rules
        # differ across numpy versions and a silent overflow here would mis-address cells.
        self._i = first.astype(np.int64)
        self._j = second.astype(np.int64)
        self._k = third.astype(np.int64)
        self._freq = np.ascontiguousarray(freq, dtype=np.float64)
        self._flat = np.ascontiguousarray(
            np.stack([S.load_surface(f"{p}_{family}", surface_dir).ravel() for p in self.pools]).T
        )

    # -- fits ---------------------------------------------------------------
    def _histogram(self, permutation: np.ndarray) -> np.ndarray:
        """Corpus mass per flattened surface cell for one layout."""
        index = permutation[self._i] * 961 + permutation[self._j] * 31 + permutation[self._k]
        return np.bincount(index, weights=self._freq, minlength=29791)

    def fits_from_permutations(self, permutations: Sequence[np.ndarray]) -> np.ndarray:
        """``(n, n_pools)`` fits, evaluated in fixed-size tiles so BLAS sees ONE shape.

        The tiling is what makes a layout's fit independent of its batch: every matmul is
        issued at ``(TILE, 29791) @ (29791, n_pools)``, with the last partial tile
        zero-padded. Zero rows contribute exactly 0.0 to their own outputs and are discarded.
        """
        count = len(permutations)
        if count == 0:
            return np.zeros((0, len(self.pools)), dtype=np.float64)
        out = np.empty((count, len(self.pools)), dtype=np.float64)
        block = np.zeros((TILE, 29791), dtype=np.float64)
        for start in range(0, count, TILE):
            chunk = permutations[start : start + TILE]
            block[:] = 0.0
            for row, permutation in enumerate(chunk):
                block[row] = self._histogram(permutation)
            out[start : start + len(chunk)] = (block @ self._flat)[: len(chunk)]
        return out

    def fits(self, layouts: Sequence[str]) -> np.ndarray:
        """``(n, n_pools)`` fits for 30-char C30M layout strings."""
        return self.fits_from_permutations([S.layout_permutation(lay) for lay in layouts])

    def fit_of(self, layout: str) -> dict[str, float]:
        """``{pool: fit}`` for one layout."""
        values = self.fits([layout])[0]
        return {pool: float(values[n]) for n, pool in enumerate(self.pools)}

    # -- guards -------------------------------------------------------------
    def assert_batch_invariant(
        self, layout: str, batch_lengths: Sequence[int] = (1, 3, 16, 17, 64)
    ) -> None:
        """Assert one layout's fit is BIT-identical however many others share its batch.

        A tolerance-based check cannot detect shape-dependence, which is exactly what breaks
        checkpoint-resume and cross-artifact diffs, so this compares with ``==``.
        """
        target = S.layout_permutation(layout)
        rng = np.random.default_rng(0)
        reference = None
        for length in batch_lengths:
            filler = [
                np.concatenate([rng.permutation(30), [30]]) for _ in range(max(0, length - 1))
            ]
            values = self.fits_from_permutations([target, *filler])[0]
            if reference is None:
                reference = values
            elif not np.array_equal(values, reference):
                raise AssertionError(
                    f"fit of {layout!r} changed with batch length {length}: {values} != "
                    f"{reference} — the matmul is seeing more than one operand shape, so "
                    f"resume and cross-artifact diffs are not reproducible"
                )

    def unpadded_fits(self, permutations: Sequence[np.ndarray]) -> np.ndarray:
        """The UNPADDED evaluator, kept solely as the mutation control for the guard above.

        If a future BLAS makes this batch-invariant too, the control test fails loudly rather
        than the guard silently stopping to test anything.
        """
        histograms = np.stack([self._histogram(p) for p in permutations])
        return histograms @ self._flat


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Anchors:
    """The persisted 0/1 anchors per model, with the provenance that makes them reproducible.

    ``zero`` is the random-pool MEAN fit; ``one`` is the searched optimum's fit. Both are in
    predicted ms on the frame named by ``frame`` — a gauge whose anchors are not reproducible
    is not a gauge, so every field needed to rebuild them is carried here and refused if
    missing.
    """

    zero: Mapping[str, float]
    one: Mapping[str, float]
    provenance: Mapping[str, object] = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    def __post_init__(self) -> None:
        missing = [p for p in self.zero if p not in self.one]
        if missing:
            raise ValueError(f"anchors have a zero but no one for {missing}")
        for pool, zero in self.zero.items():
            one = self.one[pool]
            if not (math.isfinite(zero) and math.isfinite(one)):
                raise ValueError(f"anchors for {pool} are not finite: zero={zero} one={one}")
            if one >= zero:
                # `one` is the FASTEST fit, so it must be strictly below the pool mean. An
                # inverted pair would silently flip the gauge's sign.
                raise ValueError(
                    f"anchors for {pool} are inverted or degenerate: one={one} must be "
                    f"strictly less than zero={zero} (fit is TIME, so the optimum is lower)"
                )

    @property
    def pools(self) -> tuple[str, ...]:
        return tuple(self.zero)

    def span(self, pool: str) -> float:
        """``zero - one``: the ms width of this model's 0-1 range."""
        return float(self.zero[pool]) - float(self.one[pool])

    # -- the gauge ----------------------------------------------------------
    def normalize(self, pool: str, fit: float) -> float:
        """One fit (ms) -> its normalized 0-1 value. Higher is better."""
        return (float(self.zero[pool]) - float(fit)) / self.span(pool)

    def normalize_many(self, fits: Mapping[str, float]) -> dict[str, float]:
        """``{pool: fit}`` -> ``{pool: normalized}``."""
        return {pool: self.normalize(pool, value) for pool, value in fits.items()}

    def normalize_array(self, values: np.ndarray, pools: Sequence[str]) -> np.ndarray:
        """Vectorized :meth:`normalize` over an ``(n, len(pools))`` fit array."""
        zero = np.array([float(self.zero[p]) for p in pools], dtype=np.float64)
        one = np.array([float(self.one[p]) for p in pools], dtype=np.float64)
        return (zero - np.asarray(values, dtype=np.float64)) / (zero - one)

    # -- guards -------------------------------------------------------------
    def assert_direction(self, tol: float = DIRECTION_TOL) -> None:
        """THE direction guard: each model's own optimum is 1.0 and its pool mean is 0.0.

        Deliberately NOT a "qwerty scores ~0" check — qwerty normalizes to roughly 0.5 here
        (the zero is the pool MEAN, not its floor), so such a check would fail a correct
        implementation and "fixing" it would invert every preference weight.
        """
        for pool in self.pools:
            at_one = self.normalize(pool, self.one[pool])
            at_zero = self.normalize(pool, self.zero[pool])
            if abs(at_one - 1.0) > tol:
                raise AssertionError(
                    f"{pool}: its own optimum normalizes to {at_one!r}, not 1.0 — the gauge's "
                    f"direction or scale is wrong"
                )
            if abs(at_zero) > tol:
                raise AssertionError(
                    f"{pool}: the random-pool mean normalizes to {at_zero!r}, not 0.0"
                )
            # A faster-than-optimum layout must exceed 1, a slower-than-pool one must go
            # negative: the sign of the SLOPE, checked independently of the two fixed points.
            if not self.normalize(pool, self.one[pool] - self.span(pool)) > 1.0:
                raise AssertionError(f"{pool}: a faster fit does not score higher — sign error")

    def assert_matches_surfaces(
        self, fits: SurfaceFits, probe: str, *, rel_tol: float = 1e-9
    ) -> None:
        """Refuse anchors that were built against DIFFERENT surfaces or a different corpus.

        Planted-drift refusal: the artifact records the fit of a probe layout under the
        surfaces the anchors were built on. If today's surfaces or corpus give a different
        fit for that probe, the anchors do not describe these surfaces and the gauge must
        refuse rather than silently rescale.
        """
        recorded = self.provenance.get("probe_fits")
        if not recorded:
            raise ValueError(
                "anchors carry no probe_fits, so drift against the surfaces cannot be "
                "checked; rebuild them with build_anchors()"
            )
        if self.provenance.get("probe_layout") != probe:
            raise ValueError(
                f"anchors recorded probe {self.provenance.get('probe_layout')!r} but were "
                f"checked against {probe!r}"
            )
        actual = fits.fit_of(probe)
        for pool, expected in recorded.items():
            if pool not in actual:
                continue
            got, want = float(actual[pool]), float(expected)
            if want == 0.0 or abs(got - want) / abs(want) > rel_tol:
                raise ValueError(
                    f"ANCHOR DRIFT for {pool}: probe {probe!r} fits {got!r} now but "
                    f"{want!r} when the anchors were built (rel "
                    f"{abs(got - want) / abs(want) if want else float('inf'):.3e} > "
                    f"{rel_tol:.0e}). The surfaces, corpus, or evaluator changed — these "
                    f"anchors do not describe these surfaces. Rebuild them; do NOT rescale."
                )

    # -- persistence --------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": ANCHOR_SCHEMA,
                "zero": dict(self.zero),
                "one": dict(self.one),
                "provenance": dict(self.provenance),
            },
            indent=1,
            sort_keys=True,
        )

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Anchors:
        schema = payload.get("schema")
        if schema != ANCHOR_SCHEMA:
            raise ValueError(
                f"anchor artifact has schema {schema!r}, expected {ANCHOR_SCHEMA!r} — a "
                f"different schema may mean different field SEMANTICS, so it is refused "
                f"rather than read optimistically"
            )
        zero, one = payload.get("zero"), payload.get("one")
        if not isinstance(zero, Mapping) or not isinstance(one, Mapping):
            raise ValueError("anchor artifact is missing its zero/one mappings")
        provenance = payload.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            raise ValueError("anchor artifact's provenance is not a mapping")
        return cls(
            zero={str(k): float(v) for k, v in zero.items()},
            one={str(k): float(v) for k, v in one.items()},
            provenance=dict(provenance),
        )

    @classmethod
    def read(cls, path: str | Path) -> Anchors:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def random_pool(n: int, seed: int) -> list[np.ndarray]:
    """``n`` uniformly random C30M permutations (space pinned at slot 30), reproducibly.

    One constructor, so the pool a zero anchor was built on can be rebuilt exactly from
    ``(n, seed)`` alone.
    """
    rng = np.random.default_rng(seed)
    return [np.concatenate([rng.permutation(30), [30]]) for _ in range(n)]


# ---------------------------------------------------------------------------
# the weighted blend
# ---------------------------------------------------------------------------
def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Non-negative weights summing to 1, refusing the degenerate all-zero case."""
    cleaned = {}
    for pool, weight in weights.items():
        value = float(weight)
        if not math.isfinite(value):
            raise ValueError(f"weight for {pool} is not finite: {weight!r}")
        if value < 0:
            raise ValueError(
                f"weight for {pool} is negative ({value!r}); a negative weight would ask the "
                f"optimizer to make that model WORSE, which is never the intent here"
            )
        cleaned[pool] = value
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("weights sum to 0, so the blended objective would be constant")
    return {pool: value / total for pool, value in cleaned.items()}


@dataclass(frozen=True)
class BlendSpec:
    """A weighting of the normalized gauges, plus the provenance of WHY those weights.

    ``rule`` names the evidence the weights came from; ``evidence`` carries the measured
    quantities behind it. A weight chosen after seeing which layout wins is not evidence, so
    the artifact records the rule alongside the numbers.
    """

    weights: Mapping[str, float]
    rule: str = "unspecified"
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", normalize_weights(self.weights))

    @property
    def pools(self) -> tuple[str, ...]:
        return tuple(p for p in S.POOLS if p in self.weights)

    def blend(self, normalized: Mapping[str, float]) -> float:
        """Weighted mean of the normalized gauges. Higher is better."""
        missing = [p for p in self.weights if p not in normalized]
        if missing:
            raise ValueError(f"blend needs normalized values for {missing}")
        return sum(float(self.weights[p]) * float(normalized[p]) for p in self.weights)

    def blend_array(self, normalized: np.ndarray, pools: Sequence[str]) -> np.ndarray:
        """Vectorized :meth:`blend` over an ``(n, len(pools))`` normalized array."""
        vector = np.array([float(self.weights.get(p, 0.0)) for p in pools], dtype=np.float64)
        if vector.sum() <= 0:
            raise ValueError(f"none of {tuple(self.weights)} appear in {tuple(pools)}")
        return np.asarray(normalized, dtype=np.float64) @ vector

    def describe(self) -> str:
        parts = ", ".join(f"{GAUGE_OF_POOL.get(p, p)}={self.weights[p]:.4f}" for p in self.pools)
        return f"{parts} [rule: {self.rule}]"


#: Weightings the campaign reports as its sensitivity band. Filled by the driver from measured
#: evidence; the two degenerate references are here because they are definitional.
def equal_weights(pools: Sequence[str] = S.POOLS) -> BlendSpec:
    """``(1/3, 1/3, 1/3)`` — reported as a REFERENCE, not as a neutral default.

    Equal weights are not neutral: POOL is fitted on the union of the other two sources and
    is measurably a near-symmetric blend of them, so an equal vote over-counts whatever the
    correlated pair agrees on.
    """
    return BlendSpec(
        weights=dict.fromkeys(pools, 1.0),
        rule="equal (reference only — NOT neutral: POOL is a union of the other two)",
    )


def solo_weights(pool: str) -> BlendSpec:
    """All weight on one model — the per-model solo objective the ``one`` anchors come from."""
    return BlendSpec(weights={pool: 1.0}, rule=f"solo {pool}")


class ModelBlendScorer(IScorer):
    """The combined objective as an :class:`~keybo.scoring.base.IScorer`.

    ``fitness`` is what the optimizer MINIMIZES, and the normalized gauges are
    higher-is-better, so this returns the **negated** weighted blend. That single sign flip
    is the whole reason :meth:`Anchors.assert_direction` exists.

    Layouts whose charset the surfaces cannot index score ``+inf`` (the worst possible
    fitness) rather than raising: the optimizer explores permutations of its starting board,
    so a non-C30M start is a user error worth reporting once, not a traceback per evaluation.
    """

    def __init__(self, anchors: Anchors, spec: BlendSpec, fits: SurfaceFits) -> None:
        self.anchors = anchors
        self.spec = spec
        self.fits = fits
        for pool in spec.pools:
            if pool not in anchors.zero:
                raise ValueError(f"blend weights {pool} but the anchors have no anchor for it")
            if pool not in fits.pools:
                raise ValueError(f"blend weights {pool} but the evaluator has no surface for it")
        anchors.assert_direction()

    def normalized(self, layout: Layout | str) -> dict[str, float]:
        """The three normalized gauges for one layout (higher is better)."""
        text = layout if isinstance(layout, str) else "".join(layout.chars)
        return self.anchors.normalize_many(self.fits.fit_of(text))

    def blend(self, layout: Layout | str) -> float:
        """The weighted blend for one layout (higher is better)."""
        return self.spec.blend(self.normalized(layout))

    def fitness(self, layout: Layout) -> float:
        text = "".join(layout.chars)
        if not S.is_c30m(text):
            return float("inf")
        return -self.spec.blend(self.anchors.normalize_many(self.fits.fit_of(text)))
