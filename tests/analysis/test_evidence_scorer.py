"""EVSCORE-1 unit tests.

Two classes of test here, and the second is the load-bearing one:

* mechanics — the frame excludes the invariant gauge, curves fit, domains are honoured;
* **guards that must be able to FAIL** — a parity test that regenerates its own
  expectation tests nothing, so each guard is paired with a case that trips it. The
  form-selection tests in particular exist because the first implementation selected by
  AICc and, because the hinge knot is searched over a grid AICc does not charge for, chose
  a hinge for all 14 gauges including ones whose price is flat noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.analysis import evidence_scorer as E
from keybo.analysis import evidence_validation as V

# --------------------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------------------


def test_invariant_gauge_is_excluded_from_the_live_frame():
    assert "sfr" in E.ALL_GAUGES
    assert "sfr" in E.INVARIANT_GAUGES
    assert "sfr" not in E.LIVE_GAUGES
    assert len(E.LIVE_GAUGES) == len(E.ALL_GAUGES) - 1 == 14


def test_sfr_really_is_permutation_invariant(corpus_dir):
    """The exclusion must rest on measured invariance, not on a variance threshold.

    numpy reports sfr's sample std as ~1e-14 on some draws, so a ``std > 0`` filter KEEPS
    it and then rank-correlates pure noise. Shuffle the layout and count distinct values.
    """
    from keybo.analysis import surfaces as S

    context = E.gauge_context(None)
    rng = np.random.default_rng(20260727)
    values = set()
    for _ in range(12):
        layout = "".join(rng.permutation(list(S.C30M)))
        values.add(round(dict(context.kmstats.stats(layout))["sfr"], 12))
    assert len(values) == 1, f"sfr moved across permutations: {sorted(values)}"


def test_gauge_matrix_matches_per_layout_vectors():
    """The hoisted matrix builder must agree with the obvious per-layout call.

    Guards the optimization that fixed a 14x slowdown: ``gauge_matrix`` used to recompute
    the whole vector once per gauge.
    """
    from keybo.analysis import surfaces as S

    context = E.gauge_context(None)
    rng = np.random.default_rng(3)
    layouts = ["".join(rng.permutation(list(S.C30M))) for _ in range(3)]
    matrix = E.gauge_matrix(layouts, context)
    assert matrix.shape == (3, 14)
    for row, layout in zip(matrix, layouts, strict=True):
        vector = context.vector(layout)
        np.testing.assert_allclose(row, [vector[g] for g in E.LIVE_GAUGES])


# --------------------------------------------------------------------------------------
# Loss-curve form selection — the guard that must be able to fail
# --------------------------------------------------------------------------------------


def _levels(n: int = 240, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(5.0, 2.0, n)


def test_pure_noise_price_selects_the_linear_form():
    """A gauge whose SHAP is noise must NOT be handed a curved form.

    This is the case AICc got wrong: with the knot searched over five quantiles, a hinge
    always won, so a flat-noise gauge was reported as "saturating".
    """
    x = _levels()
    y = np.random.default_rng(1).normal(size=len(x))
    curve = E.fit_loss_curve("noise", x, y, mean_abs_shap=1.0, shap_share_pct=1.0, bootstrap=0)
    assert curve.form == "linear"
    assert abs(curve.r2) < 0.1


def test_linear_price_selects_the_linear_form():
    x = _levels(seed=2)
    y = 0.4 * x + np.random.default_rng(2).normal(scale=0.02, size=len(x))
    curve = E.fit_loss_curve("linear", x, y, mean_abs_shap=1.0, shap_share_pct=1.0, bootstrap=0)
    assert curve.form == "linear"
    assert curve.weight == pytest.approx(0.4, abs=0.02)
    assert curve.r2 > 0.95


def test_a_genuinely_curved_price_is_detected():
    """The selector must still be ABLE to choose a curve, or the test above is vacuous."""
    x = _levels(seed=4)
    y = 0.6 * x**2 + np.random.default_rng(4).normal(scale=0.05, size=len(x))
    curve = E.fit_loss_curve("curved", x, y, mean_abs_shap=1.0, shap_share_pct=1.0, bootstrap=0)
    assert curve.form in ("quadratic", "hinge")
    assert curve.r2 > curve.r2_linear


def test_hinge_is_recovered_from_hinge_shaped_data():
    x = np.linspace(0.0, 10.0, 300)
    y = 0.1 * x + 1.5 * np.clip(x - 6.0, 0.0, None)
    curve = E.fit_loss_curve("hinge", x, y, mean_abs_shap=1.0, shap_share_pct=1.0, bootstrap=0)
    assert curve.form == "hinge"
    assert curve.knot == pytest.approx(6.0, abs=1.5)
    assert curve.r2 > 0.99


def test_curve_weight_ci_brackets_the_slope():
    x = _levels(seed=5)
    y = -0.3 * x + np.random.default_rng(5).normal(scale=0.3, size=len(x))
    curve = E.fit_loss_curve("ci", x, y, mean_abs_shap=1.0, shap_share_pct=1.0, bootstrap=200)
    low, high = curve.weight_ci
    assert low <= curve.weight <= high
    assert low < 0.0


def test_degenerate_input_yields_an_explicit_flat_curve():
    """A constant column must produce a declared-flat curve, not a fitted-looking one."""
    x = np.full(50, 3.0)
    y = np.random.default_rng(6).normal(size=50)
    curve = E.fit_loss_curve("flat", x, y, mean_abs_shap=0.0, shap_share_pct=0.0, bootstrap=0)
    assert curve.weight == 0.0
    assert curve.r2 == 0.0


def test_price_evaluates_the_selected_form():
    curve = E.LossCurve(
        metric="m",
        form="hinge",
        coeffs=[1.0, 0.5, 2.0],
        knot=4.0,
        domain=(0.0, 10.0),
        observed_range=(0.0, 10.0),
        weight=0.5,
        weight_ci=(0.4, 0.6),
        r2=0.9,
        r2_linear=0.5,
        mean_abs_shap=1.0,
        shap_share_pct=10.0,
    )
    assert curve.price(2.0) == pytest.approx(1.0 + 0.5 * 2.0)
    assert curve.price(6.0) == pytest.approx(1.0 + 0.5 * 6.0 + 2.0 * 2.0)


def test_in_domain_boundaries_are_inclusive():
    curve = E.LossCurve(
        metric="m",
        form="linear",
        coeffs=[0.0, 1.0],
        knot=None,
        domain=(1.0, 5.0),
        observed_range=(0.5, 6.0),
        weight=1.0,
        weight_ci=(1.0, 1.0),
        r2=1.0,
        r2_linear=1.0,
        mean_abs_shap=1.0,
        shap_share_pct=1.0,
    )
    assert curve.in_domain(1.0) and curve.in_domain(5.0) and curve.in_domain(3.0)
    assert not curve.in_domain(0.9)
    assert not curve.in_domain(5.1)


# --------------------------------------------------------------------------------------
# Correlation clusters and effective dof
# --------------------------------------------------------------------------------------


def test_perfectly_correlated_columns_land_in_one_cluster():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(120, 1))
    matrix = np.hstack([base, 2.0 * base + 1.0, rng.normal(size=(120, 12))])
    clusters = E.correlation_clusters(matrix, threshold=0.9)
    first, second = E.LIVE_GAUGES[0], E.LIVE_GAUGES[1]
    together = [key for key, members in clusters.items() if first in members]
    assert second in clusters[together[0]]


def test_effective_dof_collapses_when_every_column_restates_one_fact():
    rng = np.random.default_rng(8)
    base = rng.normal(size=(200, 1))
    restated = np.hstack([base + rng.normal(scale=1e-6, size=(200, 1)) for _ in range(14)])
    independent = rng.normal(size=(200, 14))
    assert E.effective_dof(restated) < 1.5
    assert E.effective_dof(independent) > 10.0


# --------------------------------------------------------------------------------------
# Surface frames — the second circularity layer
# --------------------------------------------------------------------------------------


def test_unknown_surface_frame_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown surface frame"):
        E.load_target_surface("AALTO_BASE", str(tmp_path), frame="bogus")


def test_missing_surface_raises_with_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="native"):
        E.load_target_surface("AALTO_BASE", str(tmp_path), frame="native")


def test_wrong_shape_surface_is_rejected(tmp_path):
    np.save(tmp_path / "AALTO_BASE.native.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="expected \\(31, 31, 31\\)"):
        E.load_target_surface("AALTO_BASE", str(tmp_path), frame="native")


def test_non_finite_surface_is_rejected(tmp_path):
    array = np.zeros((31, 31, 31))
    array[0, 0, 0] = np.nan
    np.save(tmp_path / "AALTO_BASE.native.npy", array)
    with pytest.raises(ValueError, match="non-finite"):
        E.load_target_surface("AALTO_BASE", str(tmp_path), frame="native")


def test_surface_digest_and_pool_are_carried(tmp_path):
    rng = np.random.default_rng(9)
    np.save(tmp_path / "COMMUNITY_BASE.native.npy", rng.normal(250, 10, (31, 31, 31)))
    surface = E.load_target_surface("COMMUNITY_BASE", str(tmp_path), frame="native")
    assert surface.pool == "COMMUNITY"
    assert len(surface.sha256) == 64
    assert surface.frame == "native"


# --------------------------------------------------------------------------------------
# End-to-end fit on a synthetic surface with a KNOWN answer
# --------------------------------------------------------------------------------------


def _synthetic_surface(tmp_path, seed: int = 0):
    """A surface whose cost is dominated by same-finger reuse, so sfb MUST price high.

    Built from the geometry rather than at random: cell ``[a, b, c]`` is penalized when a
    and b share a finger. A pipeline that cannot recover "sfb is expensive" from this has a
    wiring bug, and a random surface could not detect that.
    """
    from keybo.geometry import ROW_STAGGERED_30

    geometry = ROW_STAGGERED_30
    positions = [*geometry.slots, geometry.space_position]
    array = np.full((31, 31, 31), 250.0)
    for i, first in enumerate(positions):
        for j, second in enumerate(positions):
            if i == 30 or j == 30:
                continue
            if geometry.finger(first[0]) is geometry.finger(second[0]) and i != j:
                array[i, j, :] += 120.0
    path = tmp_path / "COMMUNITY_BASE.native.npy"
    np.save(path, array)
    return E.load_target_surface("COMMUNITY_BASE", str(tmp_path), frame="native")


def test_pipeline_recovers_a_planted_same_finger_penalty(tmp_path):
    from keybo.analysis import surfaces as S

    surface = _synthetic_surface(tmp_path)
    context = E.gauge_context(None)
    objective = S.trigram_objective(S.default_trigram_path(None))
    rng = np.random.default_rng(11)
    pool = ["".join(rng.permutation(list(S.C30M))) for _ in range(60)]
    weights = E.fit_evidence_weights(
        pool, surface, context, objective, pool_label="synthetic", bootstrap=0, seed=0
    )
    # sfb must be priced POSITIVE (more same-finger bigrams => more predicted time) and must
    # carry a large share of the attribution.
    assert weights.curves["sfb"].weight > 0.0
    ranked = [row["metric"] for row in weights.weight_table()]
    assert "sfb" in ranked[:4], f"planted sfb penalty not recovered; ranking was {ranked}"


def test_score_flags_out_of_domain_gauges(tmp_path):
    from keybo.analysis import surfaces as S

    surface = _synthetic_surface(tmp_path)
    context = E.gauge_context(None)
    objective = S.trigram_objective(S.default_trigram_path(None))
    rng = np.random.default_rng(12)
    pool = ["".join(rng.permutation(list(S.C30M))) for _ in range(60)]
    weights = E.fit_evidence_weights(
        pool, surface, context, objective, pool_label="synthetic", bootstrap=0, seed=0
    )
    # Force an out-of-domain level: push one gauge far past its fitted band.
    gauges = context.vector(pool[0])
    gauges["sfb"] = weights.curves["sfb"].domain[1] + 100.0
    result = weights.score(gauges)
    assert result["extrapolating"] is True
    assert "sfb" in result["out_of_domain"]
    assert result["out_of_domain"]["sfb"]["distance_outside"] > 0.0
    # An in-domain vector must NOT be flagged, or the flag is meaningless.
    assert weights.score(context.vector(pool[0]))["out_of_domain"].keys() <= set(E.LIVE_GAUGES)


def test_cluster_prices_sum_to_the_total(tmp_path):
    """Per-cluster prices must partition the total exactly — they are a regrouping."""
    from keybo.analysis import surfaces as S

    surface = _synthetic_surface(tmp_path)
    context = E.gauge_context(None)
    objective = S.trigram_objective(S.default_trigram_path(None))
    rng = np.random.default_rng(13)
    pool = ["".join(rng.permutation(list(S.C30M))) for _ in range(60)]
    weights = E.fit_evidence_weights(
        pool, surface, context, objective, pool_label="synthetic", bootstrap=0, seed=0
    )
    result = weights.score(context.vector(pool[0]))
    assert sum(result["per_cluster"].values()) == pytest.approx(result["score"])
    assert sum(result["per_gauge"].values()) == pytest.approx(result["score"])


def test_noise_placebo_flag_marks_the_weights(tmp_path):
    from keybo.analysis import surfaces as S

    surface = _synthetic_surface(tmp_path)
    context = E.gauge_context(None)
    objective = S.trigram_objective(S.default_trigram_path(None))
    rng = np.random.default_rng(14)
    pool = ["".join(rng.permutation(list(S.C30M))) for _ in range(60)]
    weights = E.fit_evidence_weights(
        pool,
        surface,
        context,
        objective,
        pool_label="synthetic",
        bootstrap=0,
        seed=0,
        shuffle_target=True,
    )
    assert any("NOISE PLACEBO" in note for note in weights.notes)
    assert any("NOISE PLACEBO" in note for note in weights.to_dict()["notes"])


def test_serialized_weights_carry_provenance_and_caveats(tmp_path):
    from keybo.analysis import surfaces as S

    surface = _synthetic_surface(tmp_path)
    context = E.gauge_context(None)
    objective = S.trigram_objective(S.default_trigram_path(None))
    rng = np.random.default_rng(15)
    pool = ["".join(rng.permutation(list(S.C30M))) for _ in range(60)]
    weights = E.fit_evidence_weights(
        pool, surface, context, objective, pool_label="synthetic", bootstrap=0, seed=0
    )
    payload = weights.to_dict()
    assert payload["surface_frame"] == "native"
    assert len(payload["surface_sha256"]) == 64
    assert payload["corpus_sha256"]  # the per-table digests analyze emits
    assert payload["excluded_invariant_gauges"] == ["sfr"]
    assert len(payload["weights"]) == 14
    # The modelled-only caveat must travel with the artifact, not just the terminal output.
    assert any("MODELLED ONLY" in note for note in payload["notes"])
    assert any("bit-identical to AALTO_BASE" in note for note in payload["notes"])


# --------------------------------------------------------------------------------------
# Validation harness
# --------------------------------------------------------------------------------------


def test_source_independence_rejects_same_pool_and_any_pool_cell():
    independent, note = V.sources_independent("COMMUNITY_BASE", "AALTO_BASE")
    assert independent is True

    independent, note = V.sources_independent("COMMUNITY_BASE", "COMMUNITY_TRI_PS_FREQ_PRIOR")
    assert independent is False
    assert "same model pool" in note

    independent, note = V.sources_independent("COMMUNITY_BASE", "POOL_BASE")
    assert independent is False
    assert "partially in-sample" in note

    independent, note = V.sources_independent("POOL_BASE", "AALTO_BASE")
    assert independent is False


def test_direction_invariance_proof_is_exhaustive_and_exact():
    """The served bigram vector cannot express direction — recomputed, not cited."""
    proof = V.direction_invariance_proof()
    assert proof["ordered_pairs_checked"] == 435  # C(30, 2)
    assert proof["max_abs_nonlanding_feature_diff"] == 0.0
    assert "NOT representable" in proof["verdict"]
    # The landing one-hots must be the excluded set; including them would trivially find
    # order-dependence and make the proof meaningless.
    assert set(proof["landing_features_excluded"]) == {
        "bottom",
        "home",
        "top",
        "pinky",
        "ring",
        "middle",
        "index",
        "lateral",
    }
    for name in ("angle", "inwards", "outwards"):
        assert name in proof["features_compared"]


def test_agreement_table_orients_every_scorer_the_same_way():
    """A higher-better rival must be negated, or its column carries the wrong sign."""
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    evidence = predicted.copy()  # a perfect loss-shaped scorer
    competitors = {
        "genkey": predicted.copy(),  # lower-better, perfectly aligned
        "oxeylyzer1": -predicted,  # higher-better, perfectly aligned
        "oxeylyzer2": predicted.copy(),  # higher-better, perfectly ANTI-aligned
    }
    table = V.agreement_table(predicted, evidence, competitors)
    assert table["evidence"].spearman == pytest.approx(1.0)
    assert table["genkey"].spearman == pytest.approx(1.0)
    assert table["oxeylyzer1"].spearman == pytest.approx(1.0)
    assert table["oxeylyzer2"].spearman == pytest.approx(-1.0)


def test_paired_advantage_detects_a_real_gap_and_a_real_tie():
    predicted = np.linspace(0.0, 1.0, 80)
    rng = np.random.default_rng(21)
    evidence = predicted + rng.normal(scale=0.02, size=80)  # nearly perfect
    rival = rng.normal(size=80)  # pure noise, lower-better
    result = V.paired_advantage(predicted, evidence, rival, higher_better=False, bootstrap=400)
    assert result["delta_spearman"] > 0.5
    assert result["p_gt_0"] > 0.95
    assert result["ci95"][0] > 0.0

    # And it must be able to report NO advantage: identical scorers tie exactly.
    tie = V.paired_advantage(predicted, evidence, evidence, higher_better=False, bootstrap=400)
    assert tie["delta_spearman"] == pytest.approx(0.0, abs=1e-9)


def test_orientation_check_flags_a_flipped_convention():
    """The check must be able to FAIL, or it is decoration."""
    pool = {
        "genkey": np.array([1.0, 2.0, 3.0, 4.0]),
        "oxeylyzer1": np.array([1.0, 2.0, 3.0, 4.0]),
        "oxeylyzer2": np.array([1.0, 2.0, 3.0, 4.0]),
    }
    # genkey is lower-better, so a WORST reference should be HIGH. Give it a low one.
    verdicts = V.orient_scores(pool, {"genkey": 0.5, "oxeylyzer1": 0.5, "oxeylyzer2": 0.5})
    assert "INCONSISTENT" in verdicts["genkey"]
    assert "CONSISTENT" in verdicts["oxeylyzer1"]

    verdicts = V.orient_scores(pool, {"genkey": 5.0, "oxeylyzer1": 5.0, "oxeylyzer2": 5.0})
    assert "CONSISTENT" in verdicts["genkey"]
    assert "INCONSISTENT" in verdicts["oxeylyzer1"]


def test_paired_resolution_is_tighter_than_unpaired_when_seeds_shift():
    """A common-mode seed offset must cancel in a difference; the ruler must show it."""
    from keybo.analysis import surfaces as S

    objective = S.trigram_objective(S.default_trigram_path(None))
    rng = np.random.default_rng(22)
    base = rng.normal(250.0, 20.0, (31, 31, 31))
    # Three "seeds" that differ by a pure additive offset: the paired floor must be ~0.
    per_seed = [base + offset for offset in (0.0, 5.0, -5.0)]
    pool = ["".join(rng.permutation(list(S.C30M))) for _ in range(12)]
    result = V.paired_resolution(pool, per_seed, objective)
    assert result["unpaired_floor_ms_per_trigram"] > 1.0
    assert result["paired_floor_ms_per_trigram"] < 1e-6
    assert result["ss_share_pct"]["residual"] < 1e-6
    assert result["frac_pairs_resolved"] == pytest.approx(1.0)


def test_paired_resolution_reports_a_real_residual_when_seeds_scale():
    """When the nuisance SCALES rather than shifts, the paired floor is NOT ~0."""
    from keybo.analysis import surfaces as S

    objective = S.trigram_objective(S.default_trigram_path(None))
    rng = np.random.default_rng(23)
    base = rng.normal(250.0, 20.0, (31, 31, 31))
    per_seed = [base * scale for scale in (0.9, 1.0, 1.1)]
    pool = ["".join(rng.permutation(list(S.C30M))) for _ in range(12)]
    result = V.paired_resolution(pool, per_seed, objective)
    assert result["paired_floor_ms_per_trigram"] > 1e-3
    assert result["ss_share_pct"]["residual"] > 0.0


def test_structural_limitations_name_the_four_required_classes():
    limits = {limitation.name: limitation for limitation in V.structural_limitations(0.5)}
    assert any("direction" in name for name in limits)
    assert any("Tcond" in name or "non-pairwise" in name for name in limits)
    assert any("paired" in name for name in limits)
    assert any("realized" in name for name in limits)
    # The paired floor, when supplied, must appear in the evidence string rather than being
    # silently dropped.
    paired = next(v for k, v in limits.items() if "paired" in k)
    assert "0.500" in paired.evidence
    # And the direction limitation must keep the bigram/trigram boundary exact.
    direction = next(v for k, v in limits.items() if "direction" in k)
    assert "9720" in direction.evidence
    assert "BIGRAM" in direction.evidence


def test_competitor_scores_run_on_real_layouts():
    scores = V.competitor_scores([V.QWERTY30M, "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"])
    assert set(scores) == {"genkey", "oxeylyzer1", "oxeylyzer2"}
    for values in scores.values():
        assert len(values) == 2
        assert np.all(np.isfinite(values))
    # genkey is lower-better and qwerty is the worst board, so it must score HIGHER.
    assert scores["genkey"][0] > scores["genkey"][1]
    # oxeylyzer-1 is higher-better, so qwerty must score LOWER.
    assert scores["oxeylyzer1"][0] < scores["oxeylyzer1"][1]
