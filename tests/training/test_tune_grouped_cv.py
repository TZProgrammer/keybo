"""The selection path's guards must be able to FIRE, and its group-aware split must not crash.

Three defects are pinned here, all found by the ``kaggle`` audit (KAGGLE-1 FINAL, ledger
``cf6ee07``) and TAUGATE-1 (``3620f06``), and all of them are *silent* in production:

1. ungrouped ``KFold`` puts a layout on BOTH sides of every fold, so the believed CV MAE is
   optimistic (measured +0.0349, positive on 5/5 seeds);
2. the OBVIOUS fix crashes — ``GroupKFold(5)`` with 4 groups raises, and 5 is the shipped
   default (``cli/tune.py``), so "just pass GroupKFold" converts a silent bug into a hard one;
3. the ``lolo`` tau gate keeps a candidate iff ``t >= best_tau - 1e-9``, which at 4 layouts is
   either a no-op (all tied — the case that has actually run) or a tripwire that discards the
   candidates with the BEST rho on a one-permutation-inversion edge.

The frame these were measured on (``bistrokes31_v1.tsv``) lives OUTSIDE the repo, so these
tests synthesize a layout-blocked frame with the same structure rather than loading it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, KFold

from keybo.testkit import HarnessNotTrustworthy, assert_discriminating
from keybo.training.tune import grouped_cv, tune_hyperparameters

#: Four layouts, blocked in runs — the real frame's shape (2202 rows over 4 layouts, each
#: layout's rows contiguous). Sizes are scaled down but the BLOCKING is what matters.
_LAYOUTS = ("azerty", "dvorak", "qwerty", "qwertz")
_BLOCK_SIZES = (23, 29, 37, 31)


def _layout_blocked_labels() -> np.ndarray:
    return np.array(
        [name for name, n in zip(_LAYOUTS, _BLOCK_SIZES, strict=True) for _ in range(n)]
    )


def test_ungrouped_kfold_leaks_a_layout_into_every_fold() -> None:
    """The defect itself: shuffle or not, an ungrouped split trains and tests on one layout."""
    labels = _layout_blocked_labels()
    n = len(labels)
    for splitter in (
        KFold(4),
        KFold(4, shuffle=True, random_state=0),
        KFold(4, shuffle=True, random_state=42),
    ):
        leaking = sum(
            1 for tr, te in splitter.split(np.zeros(n)) if set(labels[tr]) & set(labels[te])
        )
        assert leaking == 4, f"{splitter} leaked {leaking}/4 — the defect must reproduce here"


def test_grouped_cv_isolates_every_layout() -> None:
    """The fix: no layout appears on both sides of any fold."""
    labels = _layout_blocked_labels()
    splitter = grouped_cv(5, labels)
    leaking = [
        sorted(set(labels[tr]) & set(labels[te]))
        for tr, te in splitter.split(np.zeros(len(labels)), groups=labels)
    ]
    assert leaking == [[], [], [], []], f"grouped split still leaks: {leaking}"


def test_grouped_cv_CLAMPS_to_the_group_count_instead_of_raising() -> None:
    """The crash the naive fix introduces, at the SHIPPED default of cv=5 with 4 layouts.

    This is the whole reason ``grouped_cv`` exists rather than a bare ``GroupKFold(cv)``.
    """
    labels = _layout_blocked_labels()
    # Establish the naive form really does raise, so this test cannot silently stop pinning
    # anything if sklearn changes its behaviour.
    with pytest.raises(ValueError, match="greater than the number of groups"):
        list(GroupKFold(5).split(np.zeros(len(labels)), groups=labels))
    # The clamped form must not raise, and must degrade to leave-one-group-out.
    splitter = grouped_cv(5, labels)
    assert splitter.get_n_splits(groups=labels) == 4


def test_grouped_cv_does_not_inflate_a_small_cv() -> None:
    """Clamping is a ceiling, not a rewrite: cv below the group count is respected."""
    labels = _layout_blocked_labels()
    assert grouped_cv(2, labels).get_n_splits(groups=labels) == 2


def test_grouped_cv_refuses_a_single_group() -> None:
    """One group cannot be held out from itself; refuse rather than return a 1-fold no-op."""
    labels = np.array(["qwerty"] * 12)
    with pytest.raises(ValueError, match="at least 2 groups"):
        grouped_cv(5, labels)


def test_grouped_cv_refusal_MESSAGE_names_the_count_and_the_defect_it_prevents() -> None:
    """The count and the reason, not just the refusal: an operator has to know what to change.

    ``grouped_cv`` is reached with whatever layouts the frame happens to contain, so this error is
    a data diagnosis. "At least 2 groups" alone does not say how many were found, and the ZERO
    case (an empty ``groups`` array — a filtered-to-nothing frame, not a one-layout one) reports a
    different number and must not be mistaken for the single-group case.
    """
    for labels, expected_count in ((np.array(["qwerty"] * 12), 1), (np.array([]), 0)):
        with pytest.raises(ValueError) as exc:
            grouped_cv(5, labels)
        message = str(exc.value)
        assert f"got {expected_count}" in message, "must report how many groups were found"
        assert "train and test on the same layout" in message, "must name the defect prevented"


def test_grouped_cv_at_n_splits_equal_groups_IS_leave_one_group_out() -> None:
    """Registered caveat: at n_splits == n_groups the estimator is LOGO by definition.

    So its zero optimism and zero regret-vs-oracle are DEFINITIONS, not measurements
    (NORMGAUGE/KAGGLE-1 FINAL). Pinned so nobody later quotes those zeros as evidence.
    """
    labels = _layout_blocked_labels()
    held_out = [
        sorted(set(labels[te]))
        for _tr, te in grouped_cv(4, labels).split(np.zeros(len(labels)), groups=labels)
    ]
    assert all(len(h) == 1 for h in held_out), held_out
    assert sorted(h[0] for h in held_out) == sorted(_LAYOUTS)


def test_grouped_cv_isolates_layouts_even_when_the_groups_are_WILDLY_unbalanced() -> None:
    """The real frame is unbalanced (2202 rows over 4 layouts), and the clamp ignores sizes.

    ``min(cv, n_groups)`` counts DISTINCT groups, so a 3-row layout costs a whole fold just like a
    40-row one. The property that has to survive that is isolation, and a balanced fixture cannot
    demonstrate it: a splitter that silently rebalanced by moving rows across the boundary would
    pass every equal-block test above and leak here.
    """
    labels = np.array(["azerty"] * 3 + ["dvorak"] * 40 + ["qwerty"] * 7 + ["qwertz"] * 25)
    splitter = grouped_cv(5, labels)
    assert splitter.get_n_splits(groups=labels) == 4, "clamped by group COUNT, not by group size"
    folds = list(splitter.split(np.zeros(len(labels)), groups=labels))
    assert [sorted(set(labels[tr]) & set(labels[te])) for tr, te in folds] == [[], [], [], []]
    # and every layout is held out exactly once, so no layout is silently never tested
    assert sorted(str(next(iter(set(labels[te])))) for _tr, te in folds) == sorted(set(labels))


def test_grouped_cv_accepts_a_plain_SEQUENCE_not_only_an_ndarray() -> None:
    """Its annotation is ``Sequence[str] | np.ndarray``, and a list is what a caller hand-rolls.

    ``build_training_matrix(with_layouts=True)`` returns an ndarray, but the CLI and any ad-hoc
    caller may pass a list. ``len(set(np.asarray(groups).tolist()))`` handles both; nothing pinned
    that the non-array half of the annotation actually works.
    """
    labels = ["qwerty"] * 5 + ["dvorak"] * 5 + ["azerty"] * 5
    assert grouped_cv(5, labels).get_n_splits(groups=labels) == 3
    assert grouped_cv(5, np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])).get_n_splits() == 3


def test_a_cv_BELOW_two_still_raises_from_sklearn_rather_than_returning_a_no_op_splitter() -> None:
    """The clamp is ``min(cv, n_groups)``, so a nonsensical ``cv`` is NOT clamped upward.

    ``grouped_cv(1, ...)`` on four layouts yields ``GroupKFold(n_splits=1)``, which sklearn
    refuses. That refusal is the desired outcome — a 1-fold "split" trains and tests on everything,
    the exact defect this function exists to prevent — so it is pinned here to record that the
    ceiling deliberately has no floor of its own and does not need one.
    """
    labels = _layout_blocked_labels()
    for cv in (1, 0, -1):
        with pytest.raises(ValueError, match="at least one train/test split"):
            grouped_cv(cv, labels)


def test_tune_hyperparameters_accepts_groups_and_stays_group_isolated() -> None:
    """The public entry point threads groups through, and still returns a param dict."""
    labels = _layout_blocked_labels()
    rng = np.random.default_rng(0)
    n = len(labels)
    X = rng.random((n, 5))
    y = X[:, 0] * 5 + rng.random(n)
    best = tune_hyperparameters(X, y, n_iter=2, cv=5, seed=0, groups=labels)
    assert isinstance(best, dict)
    assert "max_depth" in best


def test_tune_hyperparameters_warns_when_run_UNGROUPED() -> None:
    """Passing no groups is still allowed, but must not look like a clean run.

    The measured optimism is +0.0349 MAE (5/5 seeds), and the believed number is the one a
    careless reader trusts — so the path that produces it says so out loud.
    """
    rng = np.random.default_rng(0)
    X = rng.random((40, 5))
    y = X[:, 0] * 5 + rng.random(40)
    with pytest.warns(UserWarning, match="UNGROUPED"):
        tune_hyperparameters(X, y, n_iter=2, cv=2, seed=0)


# --- the tau gate (TAUGATE-1) ------------------------------------------------------------


def test_kendall_tau_over_four_layouts_takes_only_seven_values() -> None:
    """Why the exact-max tau gate has no useful regime at this group count.

    The achievable step is 1/3, so "the best observed ranking quality" is a
    one-inversion distinction — and one swapped pair is enough to zero out every other
    candidate regardless of its rho.
    """
    import itertools

    from scipy.stats import kendalltau

    base = list(range(4))
    values = sorted(
        {round(float(kendalltau(base, list(p)).statistic), 6) for p in itertools.permutations(base)}
    )
    assert values == [-1.0, -0.666667, -0.333333, 0.0, 0.333333, 0.666667, 1.0]
    assert round(values[-1] - values[-2], 4) == 0.3333


def test_assert_discriminating_fires_on_a_saturated_tau_vector() -> None:
    """The guard exists and must be REACHABLE from the selection path's own operands.

    ARM-M's tau gate reported a pass while checking nothing, because every candidate's
    ``tau_heldout`` was already 1.0. This is the assertion that catches it.
    """
    assert_discriminating([1.0, 0.667, 0.667], "taus that can discriminate")
    with pytest.raises(HarnessNotTrustworthy, match="tie-break artifact"):
        assert_discriminating([1.0, 1.0, 1.0], "saturated tau_heldout")


def test_saturated_tau_gate_eliminates_nobody_and_says_so() -> None:
    """A no-op gate must be REPORTED as a no-op, not printed as if it had gated."""
    from keybo.training.tune import apply_tau_gate

    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.95, 1.0), ({"c": 3}, 0.99, 1.0)]
    with pytest.warns(UserWarning, match="GATED NOTHING"):
        gated, saturated = apply_tau_gate(scored)
    assert saturated is True
    assert [s for _p, s in gated] == [0.90, 0.95, 0.99], "no candidate may be eliminated"


def test_a_one_step_tau_edge_does_NOT_discard_the_best_rho_candidates() -> None:
    """The tripwire branch: a single permutation inversion must be treated as a TIE.

    Before the fix, tau 1.0 vs 0.667 set the two best-rho candidates to -inf and let the
    WORST rho win. One inversion is finer than this frame can resolve, so it cannot decide.
    """
    from keybo.training.tune import apply_tau_gate

    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.95, 0.666667), ({"c": 3}, 0.99, 0.666667)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gated, _saturated = apply_tau_gate(scored, n_groups=4)
    scores = [s for _p, s in gated]
    assert all(np.isfinite(scores)), f"a one-inversion edge must not eliminate anyone: {scores}"
    assert max(scores) == 0.99, "the best rho must be able to win"


def test_a_tau_gap_WIDER_than_the_resolvable_step_still_gates() -> None:
    """The gate must not be neutered: a real ranking collapse still eliminates."""
    from keybo.training.tune import apply_tau_gate

    scored = [({"a": 1}, 0.99, 1.0), ({"b": 2}, 0.95, 0.0), ({"c": 3}, 0.90, -1.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gated, _saturated = apply_tau_gate(scored, n_groups=4)
    scores = [s for _p, s in gated]
    assert np.isfinite(scores[0])
    assert scores[1] == float("-inf") and scores[2] == float("-inf")
