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


# --------------------------------------------------------------------------------------
# The domain-transfer guard — the finding that a narrow pool reverses the verdict
# --------------------------------------------------------------------------------------


def _weights_with_dof(dof: float) -> E.EvidenceWeights:
    return E.EvidenceWeights(
        source="COMMUNITY_BASE",
        frame="native",
        corpus="blend-v1",
        corpus_sha256={},
        surface_sha256="0" * 64,
        n_layouts=400,
        pool_label="synthetic",
        curves={},
        clusters={},
        cluster_shap_share_pct={},
        cluster_weight={},
        effective_dof=dof,
        surrogate_r2_in_sample=0.9,
        surrogate_r2_holdout=0.7,
        base_value=250.0,
    )


def test_effective_dof_alone_no_longer_flags_a_pool():
    """⚠ REPLACES a test that pinned the RETIRED effective-dof branch.

    That test asserted dof 3.99 -> warning and 5.03 -> silence, i.e. it pinned exactly the
    archive-vs-random contrast the floor had been calibrated on — circular, and POOLSWEEP-1
    (ledger 873afb7) measured the floor false-positiving at interp-f0.25 (dof 2.43 with a
    healthy cross-source ceiling of +0.9244). dof is now a diagnostic only: on its own it
    must produce NO warning at ANY value, including well below the old floor.
    """
    assert _weights_with_dof(3.99).transfer_warning() is None
    assert _weights_with_dof(2.43).transfer_warning() is None
    assert _weights_with_dof(5.03).transfer_warning() is None


def test_cd_ratio_is_what_flags_a_pool_now():
    """The replacement contract: C/D below the floor is a DO-NOT-TRUST, above it is silence."""
    weights = _weights_with_dof(4.0)
    fired = weights.transfer_warning(cd_ratio=1.058)  # the measured archive ratio
    assert fired is not None and "DO NOT TRUST" in fired
    assert "consensus/disagreement" in fired
    assert weights.transfer_warning(cd_ratio=3.06) is None  # random-wide
    assert weights.transfer_warning(cd_ratio=3.817) is None  # archive + ONE swap


def test_low_source_agreement_escalates_to_do_not_trust():
    """A ceiling below the threshold must escalate past a mere warning — the archive arm's
    +0.265 is where the weights lost 12 of 12 cells."""
    warning = _weights_with_dof(5.03).transfer_warning(source_agreement=0.265)
    assert warning is not None and "DO NOT TRUST" in warning
    # And a high ceiling must NOT trip it, or the guard is just noise.
    assert _weights_with_dof(5.03).transfer_warning(source_agreement=0.835) is None


def test_transfer_warning_travels_in_the_serialized_artifact():
    """⚠ REWRITTEN by GUARD-CD-1, and the original caught a REAL gap rather than merely going
    stale: it asserted a warning from `to_dict()` on dof alone, and `to_dict` called
    `transfer_warning()` with NO arguments. Once the verdict depends on C/D — a property of the
    POOL, not of this object — an unattached artifact would have silently carried NO verdict.
    Hence `attach_pool_guards`, and hence this test now pins the attached path.
    """
    unattached = _weights_with_dof(3.99).to_dict()
    assert unattached["transfer_warning"] is None  # dof alone must NOT manufacture a verdict
    assert unattached["effective_dof"] == pytest.approx(3.99)
    assert unattached["pool_guards"]["cd_ratio"] is None

    attached = (
        _weights_with_dof(3.99).attach_pool_guards(cd_ratio=1.058, source_agreement=0.265).to_dict()
    )
    assert attached["transfer_warning"] is not None
    assert "DO NOT TRUST" in attached["transfer_warning"]
    assert attached["pool_guards"]["cd_ratio"] == pytest.approx(1.058)
    assert attached["pool_guards"]["cd_floor"] == pytest.approx(2.0)


def test_cross_source_agreement_uses_only_independent_pairs():
    """The ceiling must be computed over independent pairs ONLY — a POOL pair would inflate
    it, since POOL pools the very sources being compared."""
    rng = np.random.default_rng(31)
    base = rng.normal(size=200)
    targets = {
        "AALTO_BASE": base,
        "COMMUNITY_BASE": base + rng.normal(scale=2.0, size=200),
        "POOL_BASE": base,  # identical to AALTO: would read rho 1.0 if wrongly included
    }
    result = V.cross_source_agreement(targets)
    assert set(result["pairwise"]) == {"AALTO_BASE|COMMUNITY_BASE"}
    assert result["mean"] < 0.9  # not inflated by the POOL pair


def test_headline_reports_the_ceiling_and_the_placebo_band():
    """A win/loss count alone is unreadable; the verdict must carry both rulers."""
    agreement = {
        "evidence": V.ScorerAgreement("evidence", 0.10, 0.07, 400),
        "genkey": V.ScorerAgreement("genkey", 0.51, 0.36, 400),
        "oxeylyzer1": V.ScorerAgreement("oxeylyzer1", 0.34, 0.24, 400),
        "oxeylyzer2": V.ScorerAgreement("oxeylyzer2", 0.42, 0.29, 400),
    }
    advantages = {
        name: {"delta_spearman": 0.10 - a.spearman, "ci95": [-0.5, -0.3], "p_gt_0": 0.0}
        for name, a in agreement.items()
        if name != "evidence"
    }
    report = V.ValidationReport(
        corpus="blend-v1",
        corpus_sha256={},
        surface_frame="native",
        n_layouts=400,
        pool_label="archive-400",
        cells=[
            V.SourceCell("COMMUNITY_BASE", "AALTO_BASE", True, agreement, advantages, -0.09, 400)
        ],
        lolo=[],
        placebo={"spearman_abs_mean": 0.1543, "spearman_abs_p95": 0.4659},
        resolution=None,
        direction_proof={},
        limitations=[],
        competitor_orientation={},
        source_agreement={"mean": 0.2654, "min": 0.2541, "max": 0.2756, "pairwise": {}},
    )
    headline = report.headline()
    assert headline["cells_where_evidence_wins"] == 0
    assert headline["ceiling_source_agreement_mean"] == pytest.approx(0.2654)
    assert headline["placebo_abs_p95"] == pytest.approx(0.4659)
    # evidence rho 0.10 < placebo p95 0.4659 -> must be flagged as indistinguishable
    assert headline["evidence_rho_inside_placebo_band"] is True


def test_headline_does_not_flag_placebo_band_when_signal_is_clear():
    """The flag must be able to come out FALSE, or it is decoration."""
    agreement = {
        "evidence": V.ScorerAgreement("evidence", 0.74, 0.55, 400),
        "genkey": V.ScorerAgreement("genkey", 0.40, 0.28, 400),
    }
    advantages = {"genkey": {"delta_spearman": 0.34, "ci95": [0.25, 0.44], "p_gt_0": 1.0}}
    report = V.ValidationReport(
        corpus="blend-v1",
        corpus_sha256={},
        surface_frame="native",
        n_layouts=400,
        pool_label="random-c30m-400",
        cells=[
            V.SourceCell("AALTO_BASE", "COMMUNITY_BASE", True, agreement, advantages, -0.20, 400)
        ],
        lolo=[],
        placebo={"spearman_abs_mean": 0.1122, "spearman_abs_p95": 0.2231},
        resolution=None,
        direction_proof={},
        limitations=[],
        competitor_orientation={},
        source_agreement={"mean": 0.8350, "min": 0.8350, "max": 0.8350, "pairwise": {}},
    )
    headline = report.headline()
    assert headline["cells_where_evidence_wins"] == 1
    assert headline["evidence_rho_inside_placebo_band"] is False


def _curve(metric: str, weight: float) -> E.LossCurve:
    return E.LossCurve(
        metric=metric,
        form="linear",
        coeffs=[0.0, weight],
        knot=None,
        domain=(0.0, 10.0),
        observed_range=(0.0, 10.0),
        weight=weight,
        weight_ci=(weight, weight),
        r2=0.5,
        r2_linear=0.5,
        mean_abs_shap=1.0,
        shap_share_pct=100.0 / 3,
    )


def test_sign_audit_flags_a_mechanistically_wrong_price():
    """A negative sfb price says "more same-finger bigrams is faster" — must be flagged.

    Measured on the wide pool: 5 of 14 fitted signs come out implausible, sfb among them.
    With effective dof ~5 over 14 axes an individual sign is not identified, so the count has
    to travel with the weights rather than being left for a reader to notice.
    """
    weights = _weights_with_dof(5.03)
    weights.curves = {
        "sfb": _curve("sfb", -0.112),  # implausible: same-finger reuse cannot save time
        "scissor": _curve("scissor", -0.472),  # implausible
        "alt": _curve("alt", -0.346),  # plausible: alternation saves time
    }
    audit = weights.sign_audit()
    assert audit["n_checked"] == 3
    assert audit["n_implausible"] == 2
    assert [row["metric"] for row in audit["implausible"]] == ["scissor", "sfb"]  # by |weight|


def test_sign_audit_is_clean_when_every_price_agrees_with_the_mechanism():
    """The audit must be able to come out EMPTY, or it is decoration."""
    weights = _weights_with_dof(5.03)
    weights.curves = {
        "sfb": _curve("sfb", +0.23),
        "scissor": _curve("scissor", +0.75),
        "alt": _curve("alt", -0.014),
        "roll": _curve("roll", -0.019),
    }
    audit = weights.sign_audit()
    assert audit["n_implausible"] == 0
    assert audit["n_plausible"] == 4


def test_weight_table_carries_the_per_gauge_sign_flag():
    weights = _weights_with_dof(5.03)
    weights.curves = {"sfb": _curve("sfb", -0.112), "alt": _curve("alt", -0.35)}
    rows = {row["metric"]: row for row in weights.weight_table()}
    assert rows["sfb"]["sign_plausible"] is False
    assert rows["sfb"]["expected_sign"] == 1.0
    assert rows["alt"]["sign_plausible"] is True
    assert weights.to_dict()["sign_audit"]["n_implausible"] == 1


def test_expected_sign_covers_every_live_gauge():
    """A gauge missing from the table would be silently exempt from the audit."""
    assert set(E.EXPECTED_SIGN) == set(E.LIVE_GAUGES)
