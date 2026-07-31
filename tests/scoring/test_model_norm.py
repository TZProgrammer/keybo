"""Tests for the per-model normalized gauges and their weighted blend.

Positive-controlled first: :func:`test_harness_is_trustworthy_before_any_pass_is_believed`
plants a fatal mutant and requires this module's own suite to go red, so a PASS below means
something. Three project failures came from an instrument that could not report a problem.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from keybo.analysis import surfaces as S
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring import model_norm as MN
from keybo.testkit import assert_harness_detects_a_fatal_mutant, assert_module_under

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "src" / "keybo" / "scoring" / "model_norm.py"

#: A layout whose anchors we can state exactly, for arithmetic tests that must not depend on
#: the (slow) real surfaces.
_FAKE_ZERO = {"AALTO": 200.0, "COMMUNITY": 400.0, "POOL": 300.0}
_FAKE_ONE = {"AALTO": 100.0, "COMMUNITY": 200.0, "POOL": 150.0}


def _fake_anchors(**kwargs) -> MN.Anchors:
    provenance = {"probe_layout": S.C30M, "probe_fits": {"AALTO": 150.0}}
    provenance.update(kwargs.pop("provenance", {}))
    return MN.Anchors(zero=dict(_FAKE_ZERO), one=dict(_FAKE_ONE), provenance=provenance, **kwargs)


# ---------------------------------------------------------------------------
# 0. the harness must be able to fail before any pass is believed
# ---------------------------------------------------------------------------
def test_module_under_this_worktree():
    """An editable .pth can shadow a worktree checkout while every printed path looks right."""
    assert_module_under("keybo.scoring.model_norm", REPO)


def _write_module(text: str) -> None:
    """Write the module AND invalidate its bytecode cache.

    ⚠ LOAD-BEARING, and it cost a real debugging round to find: CPython validates a ``.pyc``
    by ``(source mtime truncated to the second, source size)``. This control's mutant is
    **size-preserving** (it reorders ``a - b`` into ``b - a``), so a mutate-or-restore that
    lands inside the same mtime second leaves a cache CPython considers valid while it holds
    the OTHER version's bytecode. Observed directly: the ``.pyc`` recorded mtime 1785288965 /
    size 24429 exactly matching the restored source, and the subprocess kept executing the
    mutant.

    Both failure directions are dangerous and neither is visible in any printed path: a stale
    cache can make the mutant appear CAUGHT when it never ran, or appear SURVIVED when the
    original ran. Deleting the cache is what makes the verdict mean anything.
    """
    MODULE.write_text(text, encoding="utf-8")
    for cache in (MODULE.parent / "__pycache__").glob("model_norm.*.pyc"):
        cache.unlink()


def test_harness_is_trustworthy_before_any_pass_is_believed():
    """Plant a fatal mutant in the gauge and require THIS FILE's suite to report failure.

    The mutant inverts the normalization's sign — the one error that would silently invert
    every preference weight. If the suite still passes, every other PASS here is
    uninformative. Asserts on the EXIT CODE, not on pytest's prose: the original harness bug
    in this project was a case-sensitive grep on human-readable output that made all 24
    mutants read as SURVIVED.
    """
    original = MODULE.read_text(encoding="utf-8")
    target = "return (float(self.zero[pool]) - float(fit)) / self.span(pool)"
    assert target in original, "the mutation target moved; update this control"
    mutant = original.replace(
        target, "return (float(fit) - float(self.zero[pool])) / self.span(pool)"
    )
    assert mutant != original, "the mutation did not change the source"
    assert len(mutant) == len(original), (
        "this control relies on the mutant being size-preserving to also exercise the "
        "bytecode-cache hazard documented in _write_module; if the mutant changes size, the "
        "hazard is no longer covered"
    )

    def run_suite() -> int:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-x",
                "-q",
                str(Path(__file__).resolve()),
                "-k",
                "not trustworthy",
            ],
            cwd=REPO,
            capture_output=True,
        ).returncode

    assert_harness_detects_a_fatal_mutant(
        run_suite,
        lambda: _write_module(mutant),
        lambda: _write_module(original),
    )


# ---------------------------------------------------------------------------
# 1. the weighting arithmetic
# ---------------------------------------------------------------------------
def test_normalization_maps_the_two_anchors_to_zero_and_one():
    anchors = _fake_anchors()
    for pool in anchors.pools:
        assert anchors.normalize(pool, _FAKE_ONE[pool]) == pytest.approx(1.0, abs=1e-12)
        assert anchors.normalize(pool, _FAKE_ZERO[pool]) == pytest.approx(0.0, abs=1e-12)


def test_normalization_is_higher_is_better_and_unbounded_above():
    """A fit FASTER than the anchor optimum must exceed 1; slower than the pool mean, negative."""
    anchors = _fake_anchors()
    assert anchors.normalize("AALTO", 50.0) > 1.0
    assert anchors.normalize("AALTO", 250.0) < 0.0
    # strictly decreasing in fit
    values = [anchors.normalize("AALTO", f) for f in (100.0, 120.0, 150.0, 200.0)]
    assert values == sorted(values, reverse=True)


def test_assert_direction_accepts_correct_anchors_and_rejects_inverted_ones():
    _fake_anchors().assert_direction()
    with pytest.raises(ValueError, match="inverted or degenerate"):
        MN.Anchors(zero={"AALTO": 100.0}, one={"AALTO": 200.0})


def test_the_direction_guard_is_not_a_qwerty_check():
    """Regression guard for a documented trap.

    "qwerty should score ~0" is FALSE: the zero is the pool MEAN, not its floor, so a
    correctly-signed gauge puts qwerty near the middle. Anyone "fixing" that would invert
    every preference weight. This test pins that a MID-RANGE qwerty is ACCEPTED.
    """
    anchors = _fake_anchors()
    mid = 0.5 * (_FAKE_ZERO["AALTO"] + _FAKE_ONE["AALTO"])
    assert anchors.normalize("AALTO", mid) == pytest.approx(0.5)
    anchors.assert_direction()  # must NOT complain about a 0.5-scoring layout


def test_weights_are_normalized_and_degenerate_cases_refused():
    assert MN.normalize_weights({"A": 1, "B": 3}) == {"A": 0.25, "B": 0.75}
    with pytest.raises(ValueError, match="sum to 0"):
        MN.normalize_weights({"A": 0.0})
    with pytest.raises(ValueError, match="negative"):
        MN.normalize_weights({"A": 1.0, "B": -0.5})
    with pytest.raises(ValueError, match="not finite"):
        MN.normalize_weights({"A": float("nan")})


def test_blend_is_the_weighted_mean_and_matches_the_vectorized_form():
    spec = MN.BlendSpec(weights={"AALTO": 2.0, "COMMUNITY": 1.0, "POOL": 1.0}, rule="test")
    normalized = {"AALTO": 1.0, "COMMUNITY": 0.0, "POOL": 0.5}
    assert spec.blend(normalized) == pytest.approx(0.5 * 1.0 + 0.25 * 0.0 + 0.25 * 0.5)
    array = np.array([[1.0, 0.0, 0.5], [0.2, 0.4, 0.6]])
    expected = [spec.blend(dict(zip(S.POOLS, row, strict=True))) for row in array]
    assert spec.blend_array(array, S.POOLS) == pytest.approx(expected)


def test_blend_refuses_a_missing_gauge_instead_of_scoring_a_partial_blend():
    spec = MN.BlendSpec(weights=dict.fromkeys(S.POOLS, 1.0))
    with pytest.raises(ValueError, match="needs normalized values"):
        spec.blend({"AALTO": 1.0, "COMMUNITY": 1.0})


def test_solo_weights_reduce_the_blend_to_one_gauge():
    spec = MN.solo_weights("COMMUNITY")
    assert spec.blend({"AALTO": 0.0, "COMMUNITY": 0.7, "POOL": 0.0}) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 2. anchor persistence and the planted-drift refusal
# ---------------------------------------------------------------------------
def test_anchors_round_trip_through_json_with_provenance(tmp_path):
    anchors = _fake_anchors(provenance={"zero_n": 100, "zero_seed": 20260728})
    path = anchors.write(tmp_path / "anchors.json")
    back = MN.Anchors.read(path)
    assert back.zero == anchors.zero and back.one == anchors.one
    assert back.provenance["zero_seed"] == 20260728
    for pool in back.pools:
        assert back.normalize(pool, _FAKE_ONE[pool]) == pytest.approx(1.0, abs=1e-12)


def test_a_foreign_schema_is_refused_rather_than_read_optimistically(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps({"schema": "something-else", "zero": {"AALTO": 2.0}, "one": {"AALTO": 1.0}})
    )
    with pytest.raises(ValueError, match="schema"):
        MN.Anchors.read(path)


def test_planted_drift_is_refused_not_silently_rescaled():
    """The load-bearing safety property: anchors that do not describe today's surfaces refuse.

    Simulated by planting a probe fit that disagrees with what the evaluator computes — the
    same signature a changed surface, corpus, or evaluator would produce.
    """
    fits = _StubFits({"AALTO": 150.0, "COMMUNITY": 300.0, "POOL": 225.0})
    ok = _fake_anchors(
        provenance={"probe_fits": {"AALTO": 150.0, "COMMUNITY": 300.0, "POOL": 225.0}}
    )
    ok.assert_matches_surfaces(fits, S.C30M)  # in agreement: passes

    drifted = _fake_anchors(
        provenance={"probe_fits": {"AALTO": 150.0, "COMMUNITY": 300.0, "POOL": 225.9}}
    )
    with pytest.raises(ValueError, match="ANCHOR DRIFT"):
        drifted.assert_matches_surfaces(fits, S.C30M)


def test_anchors_without_a_probe_cannot_claim_to_be_drift_checked():
    bare = MN.Anchors(zero=dict(_FAKE_ZERO), one=dict(_FAKE_ONE))
    with pytest.raises(ValueError, match="no probe_fits"):
        bare.assert_matches_surfaces(_StubFits({"AALTO": 1.0}), S.C30M)


def test_drift_check_refuses_a_probe_it_did_not_record():
    anchors = _fake_anchors()
    with pytest.raises(ValueError, match="recorded probe"):
        anchors.assert_matches_surfaces(_StubFits({"AALTO": 150.0}), "a-different-layout")


class _StubFits:
    """Minimal stand-in for SurfaceFits, so the refusal logic is tested without real surfaces."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.pools = tuple(mapping)

    def fit_of(self, _layout):
        return dict(self._mapping)


# ---------------------------------------------------------------------------
# 3. the random pool is reproducible from (n, seed) alone
# ---------------------------------------------------------------------------
def test_random_pool_is_reproducible_and_well_formed():
    first = MN.random_pool(5, 20260728)
    again = MN.random_pool(5, 20260728)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not np.array_equal(first[0], MN.random_pool(5, 20260729)[0])
    for permutation in first:
        assert permutation.shape == (31,)
        assert permutation[30] == 30  # space is pinned
        assert sorted(permutation[:30].tolist()) == list(range(30))


# ---------------------------------------------------------------------------
# 4. the real surfaces: parity, batch-shape stability, and its mutation control
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_fits():
    if not S.available_surfaces():
        pytest.skip("model surfaces are not vendored in this checkout")
    return MN.SurfaceFits()


def test_evaluator_matches_the_shipped_score_fit(real_fits):
    """Parity against `keybo.analysis.surfaces.score_fit` — a different code path, same answer."""
    layout = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
    objective = S.trigram_objective(real_fits.trigram_path)
    mine = real_fits.fit_of(layout)
    for pool in real_fits.pools:
        shipped = S.score_fit(layout, S.load_surface(f"{pool}_{S.DEFAULT_FAMILY}"), objective)
        assert mine[pool] == pytest.approx(shipped, rel=1e-12)


def test_fit_is_bit_identical_however_many_layouts_share_the_batch(real_fits):
    """The BLAS shape-dependence guard. Compared with ==, because a tolerance cannot see it."""
    real_fits.assert_batch_invariant("pyuo,vgdnlhiea.cstrmkj-z'fwbxq")


def test_the_unpadded_path_really_is_batch_dependent(real_fits):
    """MUTATION CONTROL for the test above.

    If a future BLAS/numpy makes the unpadded path batch-invariant, the guard would still pass
    while testing nothing. This fails loudly in that case, so the guard cannot silently retire.
    """
    target = S.layout_permutation("pyuo,vgdnlhiea.cstrmkj-z'fwbxq")
    rng = np.random.default_rng(0)
    seen = []
    for length in (1, 3, 16, 17, 64):
        filler = [np.concatenate([rng.permutation(30), [30]]) for _ in range(length - 1)]
        seen.append(real_fits.unpadded_fits([target, *filler])[0])
    stacked = np.array(seen)
    assert not np.all(stacked == stacked[0]), (
        "the UNPADDED evaluator is batch-invariant on this BLAS, so "
        "test_fit_is_bit_identical_however_many_layouts_share_the_batch is no longer testing "
        "anything — re-derive the guard before trusting it"
    )
    # ...and the difference must be tiny: this is a summation-order effect, not a real bug.
    spread = (stacked.max(axis=0) - stacked.min(axis=0)) / np.abs(stacked[0])
    assert spread.max() < 1e-12


def test_qwerty_lands_mid_range_not_near_zero(real_fits, tmp_path):
    """The trap, pinned against the REAL surfaces and REAL anchors when they are available.

    Skips (rather than guesses) if the shipped anchor artifact is absent, because the claim is
    about the anchors' scale, not about the surfaces alone.
    """
    artifact = REPO / "drivers-normgauge" / "anchors.json"
    if not artifact.exists():
        pytest.skip("anchors.json has not been built in this checkout")
    anchors = MN.Anchors.read(artifact)
    anchors.assert_direction()
    normalized = anchors.normalize_many(real_fits.fit_of("qwertyuiopasdfghjkl'zxcvbnm,.-"))
    for pool, value in normalized.items():
        assert 0.2 < value < 0.8, (
            f"qwerty30m normalizes to {value:.4f} on {pool}. Near 0 would mean the gauge's "
            f"zero has become the pool FLOOR (or the sign flipped); near 1 would mean the "
            f"optimum anchor collapsed."
        )


def test_each_models_own_optimum_normalizes_to_exactly_one(real_fits):
    """Deliverable 3's direction guard, against the SHIPPED artifact and the REAL surfaces."""
    artifact = REPO / "drivers-normgauge" / "anchors.json"
    if not artifact.exists():
        pytest.skip("anchors.json has not been built in this checkout")
    anchors = MN.Anchors.read(artifact)
    champions = anchors.provenance["one_provenance"]["layout_of_record"]
    for pool, layout in champions.items():
        fit = real_fits.fit_of(layout)[pool]
        assert anchors.normalize(pool, fit) == pytest.approx(1.0, abs=1e-9), (
            f"{pool}'s own champion {layout!r} does not normalize to 1.0"
        )


def test_anchor_artifact_reproduces_its_recorded_zero_from_n_and_seed(real_fits):
    """Anchor reproducibility: rebuild the zero from (n, seed) and require the recorded value."""
    artifact = REPO / "drivers-normgauge" / "anchors.json"
    if not artifact.exists():
        pytest.skip("anchors.json has not been built in this checkout")
    anchors = MN.Anchors.read(artifact)
    pool = MN.random_pool(anchors.provenance["zero_n"], anchors.provenance["zero_seed"])
    rebuilt = real_fits.fits_from_permutations(pool)
    for index, name in enumerate(real_fits.pools):
        assert float(rebuilt[:, index].mean()) == pytest.approx(anchors.zero[name], rel=1e-12), (
            f"{name}'s zero anchor does not rebuild from (n={anchors.provenance['zero_n']}, "
            f"seed={anchors.provenance['zero_seed']}) — the anchor is not reproducible"
        )


# ---------------------------------------------------------------------------
# 5. the scorer seam the optimizer consumes
# ---------------------------------------------------------------------------
def test_scorer_negates_the_blend_because_the_optimizer_minimizes(real_fits):
    anchors = _fake_anchors()
    # the stub anchors are on a made-up scale, so pair them with a stub evaluator
    fits = _StubFits({"AALTO": 150.0, "COMMUNITY": 300.0, "POOL": 225.0})
    scorer = MN.ModelBlendScorer(anchors, MN.equal_weights(), fits)
    layout = Layout(S.C30M, ROW_STAGGERED_30)
    blend = scorer.blend(layout)
    assert blend == pytest.approx(0.5)  # each stub fit is mid-range
    assert scorer.fitness(layout) == pytest.approx(-blend)


def test_scorer_prefers_the_faster_layout(real_fits):
    """End-to-end sign check on the REAL surfaces: lower fitness must mean a better blend."""
    artifact = REPO / "drivers-normgauge" / "anchors.json"
    if not artifact.exists():
        pytest.skip("anchors.json has not been built in this checkout")
    anchors = MN.Anchors.read(artifact)
    scorer = MN.ModelBlendScorer(anchors, MN.equal_weights(), real_fits)
    good = Layout("pyuo,vgdnlhiea.cstrmkj-z'fwbxq", ROW_STAGGERED_30)
    bad = Layout("qwertyuiopasdfghjkl'zxcvbnm,.-", ROW_STAGGERED_30)
    assert scorer.fitness(good) < scorer.fitness(bad)
    assert scorer.blend(good) > scorer.blend(bad)


def test_scorer_refuses_a_blend_over_a_gauge_it_has_no_anchor_for():
    fits = _StubFits({"AALTO": 150.0})
    with pytest.raises(ValueError, match="no anchor"):
        MN.ModelBlendScorer(
            MN.Anchors(zero={"AALTO": 200.0}, one={"AALTO": 100.0}),
            MN.BlendSpec(weights={"COMMUNITY": 1.0}),
            fits,
        )


def test_non_c30m_layout_scores_worst_rather_than_raising():
    """The optimizer explores permutations of its start; a bad charset is a user error, once."""
    fits = _StubFits({"AALTO": 150.0})
    scorer = MN.ModelBlendScorer(
        MN.Anchors(
            zero={"AALTO": 200.0},
            one={"AALTO": 100.0},
            provenance={"probe_layout": S.C30M, "probe_fits": {"AALTO": 150.0}},
        ),
        MN.solo_weights("AALTO"),
        fits,
    )
    assert scorer.fitness(Layout("qwertyuiopasdfghjkl;zxcvbnm,./", ROW_STAGGERED_30)) == float(
        "inf"
    )


def test_equal_weights_are_labelled_as_a_reference_not_a_neutral_default():
    """The label is load-bearing: POOL is a union of the other two, so equal is NOT neutral."""
    assert "NOT neutral" in MN.equal_weights().rule
    assert "union" in MN.frame_caveat()
