"""The high-wpm gate must REFUSE, not merely report — and the tau gate's saturation must be readable.

The user's requirement was that "validation on high WPM buckets does not regress". `_per_bucket_rho`
computed those numbers all along and `validate()` gained a `high_wpm_gate` block, but NOTHING raised on
it: a widened model that gave up rho above 80 wpm was reported and ranked exactly like one that did not,
and every arm was judged by a human reading the block. That is the "detector exists, nothing calls it"
class this campaign has now caught four times (`all_distinct` with zero callers, the saturated tau gate,
this gate living only on an unmerged branch, and a test file that never used `assert_module_under`).

STRUCTURAL vs NOISE is load-bearing rather than decorative: it decided the last two arms. A bucket that
regresses on EVERY seed of a fold is structural (RETRAIN-DIRECTION-1's dvorak b120 at 3/3 seeds, deltas
-0.0326/-0.0306/-0.0316); one that regresses on some seeds is wobble (the bigram arm's 4/12 scattered
cells). Only structural refuses, so seed noise cannot veto an arm.
"""

from __future__ import annotations

import pytest

from keybo.training.tune import apply_tau_gate
from keybo.training.validate import require_no_high_wpm_regression_in_report
from keybo.verdicts import HighWpmRegression


def _block(regressing: list[int], *, gated: bool = True) -> dict:
    return {"gated": gated, "regressing_high_buckets": list(regressing)}


def _report(folds: dict[str, list[list[int]]], *, gated: bool = True) -> dict:
    return {
        "folds": {
            name: {"seeds": [{"high_wpm_gate": _block(r, gated=gated)} for r in seeds]}
            for name, seeds in folds.items()
        }
    }


def test_a_STRUCTURAL_regression_is_REFUSED_using_the_measured_case() -> None:
    """RETRAIN-DIRECTION-1's dvorak b120: the same bucket on all three seeds."""
    with pytest.raises(HighWpmRegression) as exc:
        require_no_high_wpm_regression_in_report(
            _report({"dvorak": [[120], [120], [120]]}), "trigram"
        )
    message = str(exc.value)
    assert "STRUCTURAL" in message
    assert "dvorak" in message, "the message must name the fold"
    assert "120" in message, "and the bucket"


def test_SEED_NOISE_does_NOT_veto_an_arm() -> None:
    """The bigram arm's scattered failures: one seed of three. Refusing these would veto on wobble."""
    verdict = require_no_high_wpm_regression_in_report(
        _report({"azerty": [[120], [], []]}), "bigram"
    )
    assert verdict["passed"] is True
    assert verdict["per_fold"]["azerty"]["noise_buckets"] == [120]
    assert verdict["per_fold"]["azerty"]["structural_buckets"] == []


def test_a_clean_arm_PASSES_and_the_verdict_is_SERIALIZABLE() -> None:
    """A passing verdict must be recorded as explicitly as a failure (TAUGATE-1's ambiguity)."""
    verdict = require_no_high_wpm_regression_in_report(
        _report({"qwerty": [[], [], []], "qwertz": [[], [], []]}), "clean"
    )
    # Exact equality on purpose: a verdict quietly gaining a field is how a reader ends up
    # trusting a dict whose shape they no longer know. `support` is {} here because `_report`
    # builds blocks without it — "not supplied" stays distinguishable from "supplied and thin".
    assert verdict == {
        "passed": True,
        "gated": True,
        "per_fold": {
            "qwerty": {
                "n_seeds": 3,
                "regressing_bucket_seed_counts": {},
                "structural_buckets": [],
                "noise_buckets": [],
                "support": {},
            },
            "qwertz": {
                "n_seeds": 3,
                "regressing_bucket_seed_counts": {},
                "structural_buckets": [],
                "noise_buckets": [],
                "support": {},
            },
        },
    }


def test_an_UNGATED_fold_is_REFUSED_because_not_measured_is_not_did_not_regress() -> None:
    with pytest.raises(HighWpmRegression, match="not measured|could not run"):
        require_no_high_wpm_regression_in_report(
            _report({"qwerty": [[], [], []]}, gated=False), "ungated"
        )


def test_a_MISSING_gate_block_is_REFUSED_not_silently_skipped() -> None:
    """The failure mode that matters most: a caller who forgot baseline_buckets."""
    report = {"folds": {"qwerty": {"seeds": [{"rho": 0.5}, {"rho": 0.6}]}}}
    with pytest.raises(HighWpmRegression):
        require_no_high_wpm_regression_in_report(report, "no block at all")


def test_structural_is_per_fold_so_one_bad_fold_cannot_hide_behind_three_good_ones() -> None:
    with pytest.raises(HighWpmRegression) as exc:
        require_no_high_wpm_regression_in_report(
            _report(
                {
                    "azerty": [[], [], []],
                    "qwerty": [[], [], []],
                    "qwertz": [[], [], []],
                    "dvorak": [[100, 120], [100, 120], [100, 120]],
                }
            ),
            "one bad fold",
        )
    assert "dvorak" in str(exc.value)
    assert "100" in str(exc.value) and "120" in str(exc.value)


def test_a_bucket_structural_in_ONE_fold_and_absent_elsewhere_still_refuses() -> None:
    """Folds are independent populations; averaging them would hide a real regression."""
    with pytest.raises(HighWpmRegression):
        require_no_high_wpm_regression_in_report(
            _report({"a": [[120], [120]], "b": [[], []]}), "mixed"
        )


# --- the tau gate's saturation must be READABLE, not discarded ----------------------------


def test_apply_tau_gate_reports_saturation_and_the_caller_can_no_longer_discard_it() -> None:
    """It was assigned to `_saturated` at the call site — a gate that eliminated NOBODY produced a
    leaderboard indistinguishable from a tau-filtered one."""
    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.95, 1.0), ({"c": 3}, 0.99, 1.0)]
    with pytest.warns(UserWarning, match="GATED NOTHING"):
        gated, saturated = apply_tau_gate(scored)
    assert saturated is True
    assert [s for _p, s in gated] == [0.90, 0.95, 0.99], "a no-op gate must eliminate nobody"


def test_a_discriminating_tau_vector_is_NOT_flagged_saturated() -> None:
    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.95, 0.0)]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _gated, saturated = apply_tau_gate(scored, n_groups=4)
    assert saturated is False
