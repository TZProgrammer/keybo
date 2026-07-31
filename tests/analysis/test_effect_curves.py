from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from keybo.analysis.effect_curves import EffectCurves, compute_effect_curves
from keybo.features.schema import BIGRAM_FEATURE_NAMES
from keybo.geometry import Geometry
from keybo.models.base import TypingModel


class _ZeroShapBooster:
    def predict(self, matrix, pred_contribs=False):
        assert pred_contribs is True
        return np.zeros((matrix.num_row(), len(BIGRAM_FEATURE_NAMES) + 1))


class _FakeModel:
    metadata = SimpleNamespace(ngram="bigram", feature_names=list(BIGRAM_FEATURE_NAMES))
    target_space = "MS"
    _regressor = SimpleNamespace(get_booster=lambda: _ZeroShapBooster())

    def predict_ms(self, features):
        same_hand = BIGRAM_FEATURE_NAMES.index("same_hand")
        return 100.0 + 20.0 * features[:, same_hand]


class _CalibratedLogratModel:
    metadata = SimpleNamespace(
        ngram="bigram",
        feature_names=list(BIGRAM_FEATURE_NAMES),
        extra={
            "training": {
                "target_space": "LOGRAT",
                "calibration": {"deltas_ms": {"pinky_first": 62.0}},
            }
        },
    )
    target_space = "LOGRAT"
    _regressor = SimpleNamespace(get_booster=lambda: _ZeroShapBooster())
    to_ms = TypingModel.to_ms

    def predict(self, features):
        wpm = features[:, BIGRAM_FEATURE_NAMES.index("wpm")]
        return np.log(138.0 * wpm / 12000.0)

    def predict_ms(self, features):
        return self.to_ms(self.predict(features), features)


def test_effect_curves_pin_class_means_contrasts_and_percent_units():
    geometry = Geometry(slots=((-5, 2), (-4, 2), (1, 2)))

    curves = compute_effect_curves([_FakeModel()], wpms=[90.0], geometry=geometry)

    assert curves.class_mean_ms["alternate"] == pytest.approx([100.0])
    assert curves.class_mean_ms["same_hand_other"] == pytest.approx([120.0])
    assert curves.contrast_ms["alternate"] == pytest.approx([0.0])
    assert curves.contrast_ms["same_hand_other"] == pytest.approx([20.0])
    assert curves.contrast_pct()["same_hand_other"] == pytest.approx([20.0])
    assert curves.shap_ms["same_hand_other"] == pytest.approx([0.0])
    assert curves.n_pairs["alternate"] == 4
    assert curves.n_pairs["same_hand_other"] == 2
    assert curves.weighted_by == "uniform"
    assert curves.to_dict()["contrast_vs_alternate_pct"]["same_hand_other"] == pytest.approx([20.0])


def test_contrast_pct_preserves_sign_and_uses_each_wpm_reference():
    curves = EffectCurves(
        wpms=[60.0, 120.0],
        class_mean_ms={"alternate": [100.0, 200.0]},
        contrast_ms={"sfb": [25.0, -50.0]},
        shap_ms={},
        n_pairs={},
    )
    assert curves.contrast_pct() == {"sfb": [25.0, -25.0]}


def test_effect_curves_apply_position_calibration_to_the_served_surface():
    geometry = Geometry(slots=((-5, 2), (-4, 2), (1, 2)))

    curves = compute_effect_curves(
        [_CalibratedLogratModel()],
        wpms=[100.0],
        geometry=geometry,
    )

    # The same-hand class has q->w (pinky-first, 200 ms) and w->q
    # (uncalibrated, 138 ms). Alternate pairs remain at the 138 ms baseline.
    assert curves.class_mean_ms["alternate"] == pytest.approx([138.0])
    assert curves.class_mean_ms["same_hand_other"] == pytest.approx([169.0])
    assert curves.contrast_ms["same_hand_other"] == pytest.approx([31.0])


def test_the_roll_classes_are_named_for_what_the_gauge_can_actually_represent():
    """The two same-hand roll classes must NOT be named for a direction of travel.

    The served bigram gauge has no direction-of-travel channel: every relational and
    geometric feature is a function of the UNORDERED pair, and direction enters only via
    the landing-key one-hots (computed from the second key alone). A class named
    "inroll"/"outroll" therefore asserts an effect the gauge structurally cannot express,
    which is why they are `outer_high`/`outer_low`.
    """
    from keybo.analysis.effect_curves import PATTERN_CLASSES

    assert "outer_high" in PATTERN_CLASSES and "outer_low" in PATTERN_CLASSES
    assert "inroll" not in PATTERN_CLASSES, "misnamed: the predicate is order-invariant"
    assert "outroll" not in PATTERN_CLASSES, "misnamed: the predicate is order-invariant"


#: Classes whose predicate is deliberately ORDER-DEPENDENT (direction of travel). Every other
#: class in ``PATTERN_CLASSES`` must be a function of the unordered pair. Listed explicitly so
#: adding an order-dependent class is a decision someone makes here, not a silent side effect.
ORDER_DEPENDENT_CLASSES = {"roll_inward_ordered", "roll_outward_ordered"}


def test_every_pattern_class_predicate_is_order_invariant_unless_declared():
    """Exhaustive, over all 900 ordered pairs — the proof behind the rename.

    Originally this asserted order-invariance for the WHOLE table, with the docstring caveat
    that "if a future class IS genuinely directional it must not go in this table without a
    landing-key channel to carry it". The ordered roll classes are that future class, and the
    caveat's condition is satisfied (see the next test, which measures the channel rather than
    assuming it), so the blanket assertion is now an allowlist: undeclared classes must still
    be order-invariant, and declared ones must ACTUALLY be order-dependent — a class that
    lands in the allowlist while being invariant is the failure this second half catches.
    """
    import itertools

    from keybo.analysis.effect_curves import PATTERN_CLASSES
    from keybo.geometry import ROW_STAGGERED_30 as geometry

    pairs = list(itertools.product(geometry.slots, repeat=2))
    assert len(pairs) == 900
    assert set(PATTERN_CLASSES) >= ORDER_DEPENDENT_CLASSES, "allowlist names a missing class"
    for name, (predicate, _features) in PATTERN_CLASSES.items():
        violations = [
            (a, b) for a, b in pairs if predicate(geometry, a, b) != predicate(geometry, b, a)
        ]
        if name in ORDER_DEPENDENT_CLASSES:
            assert len(violations) == 324, (
                f"{name} is declared directional but moves on {len(violations)}"
            )
        else:
            assert violations == [], f"{name} is order-DEPENDENT on {len(violations)} pairs"


def test_the_ordered_roll_classes_are_separable_in_the_served_frame():
    """The condition the original guard's docstring set for admitting a directional class.

    A class contrast is only meaningful if the model can tell the class's pairs apart from the
    reference's. The ordered roll classes carry no served feature COLUMN of their own —
    ``inwards_ordered`` lives behind the ``direction=True`` opt-in — so this measures whether
    the served frame separates them at all, rather than assuming it. It does, through the
    landing-key one-hots: an inward stroke lands nearer the index by definition, so ``index``
    and ``lateral`` rise and ``pinky``/``ring`` fall. Exactly those four columns move and no
    non-landing column does, which is the docstring's "landing-key channel to carry it",
    measured.
    """
    import itertools

    import numpy as np

    from keybo.analysis.effect_curves import PATTERN_CLASSES
    from keybo.features import bigram_features_from_positions
    from keybo.geometry import ROW_STAGGERED_30 as geometry

    pairs = [(a, b) for a, b in itertools.product(geometry.slots, repeat=2) if a != b]
    features = np.vstack([bigram_features_from_positions(geometry, p, wpm=90.0) for p in pairs])
    inward = np.array([PATTERN_CLASSES["roll_inward_ordered"][0](geometry, *p) for p in pairs])
    outward = np.array([PATTERN_CLASSES["roll_outward_ordered"][0](geometry, *p) for p in pairs])
    assert inward.sum() == outward.sum() == 162

    separating = {
        name
        for i, name in enumerate(BIGRAM_FEATURE_NAMES)
        if abs(features[inward, i].mean() - features[outward, i].mean()) > 1e-12
    }
    assert separating == {"pinky", "ring", "index", "lateral"}
    # and the SHAP feature list is empty, because none of those four is the class's own column
    assert PATTERN_CLASSES["roll_inward_ordered"][1] == []
    assert PATTERN_CLASSES["roll_outward_ordered"][1] == []


def test_the_two_roll_classes_span_the_same_unordered_pairs_as_their_ordered_count_implies():
    """108 ordered pairs over 54 unordered — every pair's reverse is in its OWN class."""
    import itertools

    from keybo.analysis.effect_curves import PATTERN_CLASSES
    from keybo.geometry import ROW_STAGGERED_30 as geometry

    for name in ("outer_high", "outer_low"):
        predicate = PATTERN_CLASSES[name][0]
        ordered = {
            (a, b)
            for a, b in itertools.product(geometry.slots, repeat=2)
            if predicate(geometry, a, b)
        }
        assert len(ordered) == 108, name
        assert len({frozenset((a, b)) for a, b in ordered}) == 54, name
        assert all((b, a) in ordered for a, b in ordered), name
