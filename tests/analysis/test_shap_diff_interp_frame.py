"""The interp frame's INTEGRATION with shap_diff, and the refusals that keep it safe.

Every test here is about a way the integration could go wrong SILENTLY — producing a table that
reconciles while explaining the wrong thing. That is the failure class shap_diff's own docstring
names as "worse than no table", so each path gets a test that the wrong thing RAISES.
"""

import numpy as np
import pytest

from keybo.analysis.shap_diff import FRAMES, _shap_tables, block_map, shap_diff
from keybo.features import (
    BIGRAM_FEATURE_NAMES,
    BIGRAM_INTERP_FEATURE_NAMES,
    FEATURE_VERSION_INTERP,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_31
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel

G = ROW_STAGGERED_31


def _toy_interp_models(n_models=2, seed=0):
    """Cheap LOGRAT models on the interp frame — enough to exercise the plumbing.

    Fitted on synthetic targets: this suite tests the ATTRIBUTION MACHINERY, not what a model
    learned, and a real training run would make it a slow test for no extra coverage.

    ⚠ ``gamma=0`` and a LARGE synthetic signal are both load-bearing, and finding out why cost a
    debugging round worth recording. ``_DEFAULT_PARAMS`` carries ``gamma=0.957`` (REG-LOLO's
    adopted split-pruning penalty), which is calibrated to the real LOGRAT target's variance. A
    small synthetic signal never clears it, so xgboost makes ZERO splits and the model predicts a
    CONSTANT — at which point every cell's gap is exactly 0.0, every residual is exactly 0.0, and
    the shuffle control "passes" because there is no attribution left to shuffle. That is a
    VACUOUS control, the precise failure shap_diff's own docstring warns about, and it looks
    identical to a healthy run in every residual. :func:`_assert_not_degenerate` is therefore
    called before the models are handed out.
    """
    rng = np.random.default_rng(seed)
    positions = [*G.slots, G.space_position]
    from keybo.features import interp_features_from_positions

    X = np.vstack(
        [interp_features_from_positions(G, (a, b), wpm=90.0) for a in positions for b in positions]
    )
    out = []
    for i in range(n_models):
        meta = ModelMetadata(
            feature_version=FEATURE_VERSION_INTERP,
            feature_names=list(BIGRAM_INTERP_FEATURE_NAMES),
            wpm_range=(60, 120),
            ngram="bigram",
        )
        meta.extra["training"] = {"target_space": "LOGRAT", "calibration": None}
        m = XGBoostTypingModel(
            meta, n_estimators=24, max_depth=3, gamma=0.0, random_state=i
        )
        y = 0.3 + 0.15 * X[:, 0] + 0.10 * X[:, 1] + 0.05 * X[:, 4] + rng.normal(0, 0.01, len(X))
        m.fit(X, y)
        out.append(m)
    _assert_not_degenerate(out, X)
    return out


def _assert_not_degenerate(models, X):
    """Refuse a fixture whose models predict a CONSTANT — see :func:`_toy_interp_models`."""
    for m in models:
        spread = float(np.ptp(m.predict(X)))
        assert spread > 1e-3, (
            f"degenerate toy model (prediction spread {spread:.3e}): every gap and residual "
            f"would be exactly 0 and the shuffle control would pass VACUOUSLY"
        )


# --- block_map: the registration that IS the integration point ---------------------------


def test_block_map_accepts_the_interp_frame_and_partitions_every_column():
    spec = block_map(BIGRAM_INTERP_FEATURE_NAMES)
    assert set(spec) == set(BIGRAM_INTERP_FEATURE_NAMES)
    assert all(blk for blk, _sub in spec.values())


def test_block_map_still_refuses_an_unregistered_frame():
    """The guard the interp registration must not have weakened."""
    with pytest.raises(ValueError, match="no block partition registered"):
        block_map(["made", "up", "columns"])


def test_the_interp_blocks_are_narrow_which_is_the_whole_claim():
    """The served frame needs blocks because column credit cannot be trusted; a frame whose
    widest block is 3 columns makes the block and column tables nearly the same claim."""
    spec = block_map(BIGRAM_INTERP_FEATURE_NAMES)
    sizes = {}
    for blk, _sub in spec.values():
        sizes[blk] = sizes.get(blk, 0) + 1
    assert max(sizes.values()) <= 3, sizes


def test_no_interp_block_is_named_WPM():
    """There is no wpm column, so a WPM block would be an empty claim in the primary table."""
    assert "WPM" not in {blk for blk, _ in block_map(BIGRAM_INTERP_FEATURE_NAMES).values()}


def test_the_served_frames_still_map_to_their_own_blocks():
    assert {b for b, _ in block_map(BIGRAM_FEATURE_NAMES).values()} == {
        "ROW",
        "FINGER",
        "RELATIONAL",
        "GEOMETRY",
        "WPM",
    }
    assert "BG1" in {b for b, _ in block_map(TRIGRAM_FEATURE_NAMES).values()}


# --- the FRAME-SWAP refusals (prereg §7 NC2) ----------------------------------------------


def test_an_interp_model_handed_the_SERVED_frame_RAISES():
    """The registered NC2 control. A silent success here would attribute 10 interp columns'
    predictions to 20 served column NAMES — reconciling perfectly while explaining nothing."""
    models = _toy_interp_models(1)
    with pytest.raises(ValueError, match="do not carry the 'served' frame"):
        _shap_tables(models, G, 90.0, 2, "served")


def test_a_SERVED_model_handed_the_interp_frame_RAISES():
    from keybo.analysis.shap_diff import default_models

    with pytest.raises(ValueError, match="do not carry the 'interp' frame"):
        _shap_tables(default_models("bigram")[:1], G, 90.0, 2, "interp")


def test_the_interp_frame_refuses_order_three():
    models = _toy_interp_models(1)
    with pytest.raises(ValueError, match="bigram-only frame"):
        _shap_tables(models, G, 90.0, 3, "interp")


def test_an_unknown_frame_name_raises():
    models = _toy_interp_models(1)
    with pytest.raises(ValueError, match="frame must be one of"):
        _shap_tables(models, G, 90.0, 2, "nonsense")


def test_the_table_cache_does_not_confuse_two_frames():
    """`frame` is part of the cache key. Without it, the SECOND call would be served the FIRST
    frame's SHAP table for the same (geometry, wpm, order) — and it would reconcile."""
    from keybo.analysis.shap_diff import default_models

    served = _shap_tables(default_models("bigram"), G, 90.0, 2, "served")
    with pytest.raises(ValueError):
        _shap_tables(default_models("bigram"), G, 90.0, 2, "interp")
    assert len(served[5]) == 20


# --- shap_diff()'s own refusals -----------------------------------------------------------


def test_shap_diff_refuses_interp_without_explicit_models():
    """Defaulting to the shipped models would attribute the SERVED frame under a report that
    says frame='interp'."""
    from keybo.cli.analyze import _resolve

    _, a = _resolve("flagship-c3")
    _, b = _resolve("graphite")
    with pytest.raises(ValueError, match="bigram_models= must be supplied"):
        shap_diff(a, b, frame="interp", channel="t2")


def test_shap_diff_refuses_interp_on_the_tcond_channel():
    from keybo.cli.analyze import _resolve

    _, a = _resolve("flagship-c3")
    _, b = _resolve("graphite")
    with pytest.raises(ValueError, match="bigram-only"):
        shap_diff(a, b, frame="interp", channel="both", bigram_models=_toy_interp_models(1))


def test_frames_lists_exactly_the_supported_frames():
    """Pinned so a frame cannot be added without a deliberate edit here: every frame needs a
    block partition, a stamp and a featurizer, and FRAMES is the list a caller reads.

    ``hybridb`` (HYBRIDB-1) added deliberately: it has a registered block partition
    (``_HYBRIDB_BLOCKS``), a stamp (``FEATURE_VERSION_HYBRIDB``) and a featurizer resolved through
    ``keybo.features.ngram.replacement_frame``. This tripwire fired on that edit, which is it
    working."""
    assert FRAMES == ("served", "interp", "interp-wpm", "hybridb")


# --- the card() bar is SCOPED, not dropped ------------------------------------------------


def test_card_tie_applies_is_true_for_served_and_false_for_interp():
    """A bar that silently vanishes is worse than one that fails: `card_tie_applies` names the
    condition in the result and in the JSON, so a reader can see the bar was scoped."""
    from keybo.cli.analyze import _resolve

    _, a = _resolve("flagship-c3")
    _, b = _resolve("graphite")
    served = shap_diff(a, b, channel="t2")
    assert served.card_tie_applies is True
    assert served.frame == "served"
    assert served.to_dict()["residuals"]["card_tie_applies"] is True

    interp = shap_diff(
        a, b, channel="t2", frame="interp", bigram_models=_toy_interp_models(2)
    )
    assert interp.card_tie_applies is False
    assert interp.frame == "interp"
    assert interp.to_dict()["residuals"]["card_tie_applies"] is False


def test_an_interp_run_still_has_to_pass_its_own_EXTERNAL_bar():
    """Scoping the card() tie must not leave the run unbarred: the per-channel
    `resid_gap_vs_shipped` compares the TreeSHAP walk against the SAME models' predict()-side
    table through an independent code path, and it is checked on every frame."""
    from keybo.cli.analyze import _resolve

    _, a = _resolve("flagship-c3")
    _, b = _resolve("graphite")
    d = shap_diff(a, b, channel="t2", frame="interp", bigram_models=_toy_interp_models(2))
    assert d.t2 is not None
    assert d.t2.resid_gap_vs_shipped <= 1e-3, d.t2.resid_gap_vs_shipped
    assert d.t2.resid_cell_lmdi <= 1e-9
    assert d.t2.resid_feature_sum <= 1e-9
    assert d.reconciles()


def test_the_interp_attribution_SUMS_to_its_own_channel_gap():
    from keybo.cli.analyze import _resolve

    _, a = _resolve("flagship-c3")
    _, b = _resolve("graphite")
    d = shap_diff(a, b, channel="t2", frame="interp", bigram_models=_toy_interp_models(2))
    total = sum(c.ms_per_char for c in d.t2.contributions)
    assert total == pytest.approx(d.t2.gap, rel=1e-9, abs=1e-12)
    blocks = sum(x.ms_per_char for x in d.t2.blocks())
    assert blocks == pytest.approx(d.t2.gap, rel=1e-9, abs=1e-12)


def test_the_shuffle_control_BREAKS_an_interp_run():
    """Prereg §7 NC3: a control that RECONCILES means the identity is vacuous."""
    from keybo.cli.analyze import _resolve

    _, a = _resolve("flagship-c3")
    _, b = _resolve("graphite")
    models = _toy_interp_models(2)
    good = shap_diff(a, b, channel="t2", frame="interp", bigram_models=models)
    bad = shap_diff(
        a, b, channel="t2", frame="interp", bigram_models=models, shuffle_seed=0
    )
    assert good.reconciles()
    assert not bad.reconciles()
    # It must break the INTERNAL bars specifically (that is what a shuffled attribution is).
    assert bad.t2.resid_cell_lmdi > good.t2.resid_cell_lmdi


def test_the_report_declares_the_frame_and_flags_the_unGATED_tie():
    from keybo.analysis.shap_diff import format_report
    from keybo.cli.analyze import _resolve

    _, a = _resolve("flagship-c3")
    _, b = _resolve("graphite")
    text = format_report(
        shap_diff(a, b, channel="t2", frame="interp", bigram_models=_toy_interp_models(2))
    )
    assert "frame: interp" in text
    assert "NOT the served frame" in text
    assert "NOT GATED" in text
