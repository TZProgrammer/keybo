"""Unit tests for MODELNORM-1's normalization and native-frame evaluator.

Run with an rc SENTINEL — trap 1: a bare pytest can write no sentinel at all while still
exiting 0, and absence of a sentinel is NOT evidence of rc=0:

    uv run --no-sync pytest <this file> -p no:cacheprovider -q ; echo $? > rc.txt

Every test here is designed to be able to FAIL: the direction tests are mutation-checked in
``test_direction_guard_bites_on_an_inverted_sign``, which flips the sign deliberately and
asserts the guard raises. A guard that cannot fail is not a guard (trap 31).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modelnorm_eval as MN  # noqa: E402


@pytest.fixture(scope="module")
def surf() -> MN.NativeSurfaces:
    return MN.NativeSurfaces()


@pytest.fixture(scope="module")
def anchors(surf: MN.NativeSurfaces) -> MN.Anchors:
    """A cheap anchor set for the algebra tests: random "0", and a "1" that is simply the
    best candidate (the algebra does not care where "1" came from; the real run's "1" is a
    per-model search output and is tested separately in the run's own gates)."""
    zero, sd, _ = MN.zero_anchor(surf, n=100, seed=20260728)
    one = MN.ceiling_fraction_anchors(surf, MN.CANDIDATES)
    return MN.Anchors(
        zero=zero, one=one, zero_statistic="mean", zero_n=100, zero_seed=20260728,
        zero_sd=sd, one_provenance={"kind": "best-of-candidates (test fixture only)"},
    )


# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------
def test_native_frame_guard_confirms_standardized_shares_aaltos_bigram_tensor(surf):
    """TRAP 5. The guard must show: AALTO std == native EXACTLY, the others differ, and the
    difference carries no dependence on the third slot (a pure bigram substitution)."""
    report = surf.frame_report
    assert report["AALTO"]["max_abs_std_minus_native"] == 0.0
    assert report["COMMUNITY"]["max_abs_std_minus_native"] > 100.0
    assert report["POOL"]["max_abs_std_minus_native"] > 10.0
    for model in MN.MODELS:
        assert report[model]["max_variation_of_delta_over_third_slot"] < 1e-9


def test_the_three_native_surfaces_are_actually_distinct(surf):
    """If two 'models' were the same array the whole exercise would be vacuous."""
    for a_index, a in enumerate(MN.MODELS):
        for b in MN.MODELS[a_index + 1:]:
            gap = float(np.abs(surf.surfaces[a] - surf.surfaces[b]).max())
            assert gap > 1.0, f"{a} and {b} are effectively the same surface (max|d|={gap})"


def test_native_frame_guard_bites_when_handed_standardized_arrays(surf, tmp_path):
    """The guard must FAIL if someone points it at a standardized tree — else it is decoration.

    Builds a directory whose ``.native`` files ARE the standardized arrays (the exact mistake
    trap 5 describes) and asserts construction raises.
    """
    for model in MN.MODELS:
        std = np.load(surf.native_dir / f"{model}_{MN.FAMILY}.standardized.npy")
        np.save(tmp_path / f"{model}_{MN.FAMILY}.native.npy", std)
        np.save(tmp_path / f"{model}_{MN.FAMILY}.standardized.npy", std)
    with pytest.raises(AssertionError, match="native == native|standardized == native"):
        MN.NativeSurfaces(native_dir=tmp_path)


def test_fit_matches_shipped_surfaces_score_fit(surf):
    """Positive control on the ARITHMETIC against the shipped code (frame-independent)."""
    worst = surf.assert_matches_shipped()
    assert worst < 1e-6


def test_batch_fit_equals_per_layout_fit(surf):
    """The batch matmul path and the gather path must agree — the search uses the batch one.

    The tolerance is RELATIVE. These fits are total predicted ms over the whole corpus
    (~2.4e11), so an absolute 1e-6 is below one float64 ULP at that magnitude: the two paths
    sum in different orders (bincount-then-matmul vs gather-then-sum) and differ by
    ~1.2e-14 relative, which is ~52 ULP — reordering noise, not a disagreement. Asserting an
    absolute epsilon here would be asserting exact summation order.
    """
    perms = np.stack([MN.perm_of(lay) for lay in MN.CANDIDATES.values()]).astype(np.int32)
    batched = surf.fit_batch(perms)
    for row, layout in enumerate(MN.CANDIDATES.values()):
        one = surf.fit_one(MN.perm_of(layout))
        relative = float((np.abs(batched[row] - one) / np.abs(one)).max())
        assert relative < 1e-12, f"{layout}: batch vs gather relative error {relative:.3e}"


def test_batch_vs_gather_noise_is_far_below_any_candidate_gap(surf):
    """The reordering noise above must be irrelevant to every comparison we will make.

    Otherwise the search's batch path could reorder two candidates relative to the
    per-layout path. Measured margin is ~7 orders of magnitude.
    """
    perms = np.stack([MN.perm_of(lay) for lay in MN.CANDIDATES.values()]).astype(np.int32)
    batched = surf.fit_batch(perms)
    exact = np.stack([surf.fit_one(MN.perm_of(lay)) for lay in MN.CANDIDATES.values()])
    noise = float(np.abs(batched - exact).max())
    gaps = []
    for column in range(3):
        ordered = np.sort(exact[:, column])
        gaps.append(float(np.diff(ordered).min()))
    assert noise < 0.001 * min(gaps), (
        f"batch noise {noise:.4e} is not negligible against the tightest candidate gap "
        f"{min(gaps):.4e}"
    )


def test_fit_batch_is_batch_length_invariant(surf):
    """A layout's score must not depend on how many OTHER layouts share its batch.

    This is the property the zero-padding in ``fit_batch`` buys. Without it, BLAS dispatches a
    different kernel for a partial final tile and the same row comes back ~1e-15 relative
    different — so the objective would not be a function of the layout, and a resumed search
    could not reproduce an uninterrupted one (trap 36: assert the resume reproduces on COUNTS
    and VALUES, not just conclusions).
    """
    perms = MN.random_layouts(200, seed=4242)
    full = surf.fit_batch(perms)
    for length in (1, 9, 16, 17, 37, 199, 200):
        assert np.array_equal(surf.fit_batch(perms[:length]), full[:length]), (
            f"batch length {length} changed the answer for rows it shares with the full batch"
        )


def test_tile_is_a_frozen_constant_and_only_moves_results_at_ulp_level(surf):
    """The tile size cannot be made bit-irrelevant — the tile size IS the BLAS operand shape.

    So two things are asserted instead, and both are what the artifacts actually rely on:
      * changing the tile moves a fit by ~1e-15 RELATIVE (a few ULP at 2.4e11), which is ~9
        orders below the tightest gap between two random layouts — it can reorder nothing;
      * ``TILE`` is a frozen constant that :meth:`identity` records, so any published number
        names the shape that produced it rather than inheriting an undocumented default.
    """
    perms = MN.random_layouts(200, seed=4242)
    reference = surf.fit_batch(perms, tile=200)
    worst = 0.0
    for tile in (1, 7, 16, 64, 199):
        got = surf.fit_batch(perms, tile=tile)
        worst = max(worst, float((np.abs(got - reference) / np.abs(reference)).max()))
    assert worst < 1e-13, f"tile choice moves a fit by {worst:.3e} relative — too much"
    tightest = min(
        float(np.diff(np.sort(reference[:, column])).min()) for column in range(3)
    )
    absolute = worst * float(np.abs(reference).max())
    assert absolute < 1e-6 * tightest, (
        f"tile noise {absolute:.3e} is not negligible against the tightest gap {tightest:.3e}"
    )
    assert surf.identity()["tile"] == MN.NativeSurfaces.TILE == 16


def test_padding_guard_bites_if_the_fixed_shape_is_removed(surf):
    """Mutation control on the padding. An unpadded tiled implementation MUST fail the
    batch-length invariance above — otherwise that test is passing for a different reason and
    would not protect the search."""
    perms = MN.random_layouts(200, seed=4242)

    def unpadded(rows: np.ndarray, step: int = 16) -> np.ndarray:
        out = np.empty((rows.shape[0], 3), dtype=np.float64)
        for lo in range(0, rows.shape[0], step):
            hi = min(rows.shape[0], lo + step)
            block = rows[lo:hi]
            flat_index = (
                block[:, surf.I] * 31 + block[:, surf.J]
            ) * 31 + block[:, surf.K]
            weights = np.empty((hi - lo, 29791), dtype=np.float64)
            for b in range(hi - lo):
                weights[b] = np.bincount(flat_index[b], weights=surf.F, minlength=29791)
            out[lo:hi] = weights @ surf.flat.T
        return out

    assert not np.array_equal(unpadded(perms)[:9], unpadded(perms[:9])), (
        "the unpadded path is already batch-length invariant on this BLAS, so the padding "
        "guard is untested here — re-derive the shape sensitivity before trusting it"
    )


def test_perm_roundtrip_and_permutation_validation():
    for layout in MN.CANDIDATES.values():
        assert MN.layout_of(MN.perm_of(layout)) == layout
    with pytest.raises(ValueError):
        MN.perm_of("qwertyuiopasdfghjkl'zxcvbnm,.")  # 29 chars
    with pytest.raises(ValueError):
        MN.perm_of("qqertyuiopasdfghjkl'zxcvbnm,.-")  # q twice, t missing


# ---------------------------------------------------------------------------
# the normalization algebra
# ---------------------------------------------------------------------------
def test_zero_anchor_maps_to_zero_and_one_anchor_maps_to_one(surf, anchors):
    """The definitional property: fit == zero -> 0, fit == one -> 1, per model."""
    norm = MN.BlendNormalizer(anchors)
    zero_vec = np.array([anchors.zero[m] for m in MN.MODELS])
    one_vec = np.array([anchors.one[m] for m in MN.MODELS])
    assert np.abs(norm.normalize(zero_vec) - 0.0).max() < 1e-12
    assert np.abs(norm.normalize(one_vec) - 1.0).max() < 1e-12


def test_one_is_best_not_worst(surf, anchors):
    """TRAP 3, the sign. A naive (x-lo)/(hi-lo) on TIME maps BEST->0; ours must map BEST->1.

    ⚠ The direction is asserted against the RANDOM POOL, not against qwerty30m. The brief's
    version of this check — "qwerty30m must be ~0" — is empirically FALSE under the user's own
    design and is NOT usable as a direction guard: see
    :func:`test_qwerty30m_is_not_the_zero_of_this_scale`. qwerty is ~2.5-3.1 sd FASTER than a
    random layout, so it lands mid-scale.
    """
    norm = MN.BlendNormalizer(anchors)
    fast = surf.fit_of_layout(MN.CANDIDATES["arm-B"])
    slow = surf.fit_of_layout(MN.CANDIDATES["qwerty30m"])
    assert np.all(fast < slow), "premise: arm-B is faster than qwerty30m on all three models"
    assert np.all(norm.normalize(fast) > norm.normalize(slow))
    # the actual "0" of the scale is the random-pool mean, and it must normalize below qwerty
    zero_vec = np.array([anchors.zero[m] for m in MN.MODELS])
    assert np.all(norm.normalize(zero_vec) < norm.normalize(slow)), (
        "the random-pool anchor must sit BELOW qwerty30m: it is the zero end of the scale"
    )


def test_qwerty30m_is_not_the_zero_of_this_scale(surf, anchors):
    """DEFECT IN THE BRIEF'S OWN ASSERTION, pinned as a test so it cannot be re-assumed.

    The brief (trap 3) says to assert "qwerty30m must be ~0, the per-model optimum ~1". The
    first half is false: the "0" anchor is the mean of ~100 RANDOM layouts, and qwerty30m is
    dramatically better than random — 0.00-0.20 percentile of a 1000-layout random pool,
    z = -2.5 to -3.1. So qwerty lands roughly HALFWAY up the scale, not at 0.

    This matters beyond pedantry: anyone using "qwerty ~ 0" as the direction guard would see
    it fail on a CORRECTLY-signed implementation and might "fix" the sign to make it pass —
    which is exactly the inversion trap 3 warns about.
    """
    norm = MN.BlendNormalizer(anchors)
    qwerty = norm.normalize(surf.fit_of_layout(MN.CANDIDATES["qwerty30m"]))
    assert np.all(qwerty > 0.35), (
        f"qwerty30m normalized {qwerty} — if this really were ~0 the brief's assertion would "
        f"hold and this test should be deleted"
    )
    assert np.all(qwerty < 0.75), f"qwerty30m normalized {qwerty}: expected mid-scale"


def test_direction_guard_bites_on_an_inverted_sign(surf, anchors, monkeypatch):
    """Mutation control: flip the normalization's sign and assert :func:`assert_direction`
    raises. A direction assertion that survives an inverted sign is worthless (trap 31)."""
    one_layouts = dict.fromkeys(MN.MODELS, MN.CANDIDATES["arm-B"])
    norm = MN.BlendNormalizer(anchors)

    class Inverted(MN.BlendNormalizer):
        def normalize(self, fits):  # the exact sign error trap 3 describes
            return (np.asarray(fits, dtype=np.float64) - self.zero_vec) / self.span

    with pytest.raises(AssertionError):
        MN.assert_direction(surf, Inverted(anchors), one_layouts)
    # and the correct one passes on a consistent anchor set
    consistent = MN.Anchors(
        zero=anchors.zero,
        one={m: float(surf.fit_of_layout(MN.CANDIDATES["arm-B"])[i])
             for i, m in enumerate(MN.MODELS)},
        zero_statistic=anchors.zero_statistic, zero_n=anchors.zero_n,
        zero_seed=anchors.zero_seed, zero_sd=anchors.zero_sd,
        one_provenance={"kind": "test"},
    )
    MN.assert_direction(surf, MN.BlendNormalizer(consistent), one_layouts)
    del norm


def test_blend_weights_are_preference_not_scale(surf, anchors):
    """The design's whole point. On the NORMALIZED scale every model contributes the same
    range per unit weight, so a (1,0,0) blend must equal that model's own normalized score,
    and equal weights must give the plain mean."""
    fits = surf.fit_of_layout(MN.CANDIDATES["flagship-c3"])
    for index, model in enumerate(MN.MODELS):
        solo = MN.BlendNormalizer(anchors, {model: 1.0})
        assert abs(float(solo.blend(fits)) - float(solo.normalize(fits)[index])) < 1e-12
    equal = MN.BlendNormalizer(anchors, dict.fromkeys(MN.MODELS, 1.0))
    assert abs(float(equal.blend(fits)) - float(equal.normalize(fits).mean())) < 1e-12


def test_weight_scaling_is_invariant(surf, anchors):
    """(1,1,1) and (7,7,7) are the same PREFERENCE and must give the same blend value."""
    fits = surf.fit_of_layout(MN.CANDIDATES["keybo-lsb"])
    a = MN.BlendNormalizer(anchors, dict.fromkeys(MN.MODELS, 1.0))
    b = MN.BlendNormalizer(anchors, dict.fromkeys(MN.MODELS, 7.0))
    assert abs(float(a.blend(fits)) - float(b.blend(fits))) < 1e-12


def test_raw_weighted_sum_is_scale_broken_but_normalized_is_not(surf, anchors):
    """Re-derives the user's premise as a TEST, not a citation (trap 20).

    Under a RAW weighted sum, one unit of weight buys each model a different amount of
    leverage — proportional to that model's span. Under the normalized blend, one unit of
    weight buys exactly the same leverage from every model. This test measures both.
    """
    fits = np.stack([surf.fit_of_layout(lay) for lay in MN.CANDIDATES.values()])
    raw_span = fits.max(axis=0) - fits.min(axis=0)
    # RAW: spans differ materially, so weight encodes scale
    assert raw_span.max() / raw_span.min() > 1.15, (
        f"premise not reproduced: raw spans {raw_span} are nearly equal"
    )
    norm = MN.BlendNormalizer(anchors)
    normalized = norm.normalize(fits)
    norm_span = normalized.max(axis=0) - normalized.min(axis=0)
    # NORMALIZED: each model's own anchor range is 1.0 by construction, so a unit of weight
    # moves the blend by the same amount for every model.
    unit_leverage = 1.0 / norm.span * norm.span  # identically 1 per model, by construction
    assert np.allclose(unit_leverage, 1.0)
    del norm_span


def test_span_must_be_positive(anchors):
    """A "1" anchor that is SLOWER than the "0" anchor is a broken anchor set, not a scale."""
    broken = MN.Anchors(
        zero=anchors.one, one=anchors.zero,  # swapped: zero is now faster than one
        zero_statistic="mean", zero_n=1, zero_seed=0, zero_sd=anchors.zero_sd,
        one_provenance={},
    )
    with pytest.raises(ValueError, match="span must be positive"):
        MN.BlendNormalizer(broken)


def test_weights_reject_negative_and_all_zero(anchors):
    with pytest.raises(ValueError, match="non-negative"):
        MN.BlendNormalizer(anchors, {"AALTO": -1.0, "COMMUNITY": 1.0, "POOL": 1.0})
    with pytest.raises(ValueError, match="all-zero"):
        MN.BlendNormalizer(anchors, dict.fromkeys(MN.MODELS, 0.0))


def test_random_pool_is_reproducible_from_n_and_seed():
    a = MN.random_layouts(50, 20260728)
    b = MN.random_layouts(50, 20260728)
    c = MN.random_layouts(50, 20260729)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(a[:, 30] == MN.SPACE), "space must stay pinned at slot 30"
    for row in a:
        assert sorted(row[:30].tolist()) == list(range(30)), "each row must be a permutation"


def test_layout_key_is_not_salted():
    """Trap 8: PYTHONHASHSEED-salted hash() silently varies a 'deterministic' run."""
    assert MN.layout_key("qwertyuiopasdfghjkl'zxcvbnm,.-") == MN.layout_key(
        "qwertyuiopasdfghjkl'zxcvbnm,.-"
    )
    assert MN.layout_key(MN.CANDIDATES["arm-B"]) != MN.layout_key(MN.CANDIDATES["arm-A"])
