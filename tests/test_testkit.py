"""The pre-flight checks must themselves fire — an unfireable guard is the original defect."""

from __future__ import annotations

from pathlib import Path

import pytest

from keybo.testkit import (
    HarnessNotTrustworthy,
    assert_branch_only_symbol,
    assert_discriminating,
    assert_harness_detects_a_fatal_mutant,
    assert_module_under,
    assert_operands_computed,
)


def _src_root() -> Path:
    import keybo

    return Path(keybo.__file__).resolve().parent.parent


def test_assert_module_under_accepts_the_real_tree_and_names_the_path() -> None:
    where = assert_module_under("keybo.optimize.qap_bound", _src_root())
    assert where.name == "qap_bound.py"
    assert where.is_file()


def test_assert_module_under_REJECTS_a_wrong_root() -> None:
    with pytest.raises(HarnessNotTrustworthy) as exc:
        assert_module_under("keybo.optimize.qap_bound", "/definitely/not/here")
    msg = str(exc.value)
    assert "NOT under" in msg
    assert ".pth" in msg, "the message must name the usual cause to be actionable"


def test_assert_branch_only_symbol_passes_and_fails_appropriately() -> None:
    assert_branch_only_symbol("keybo.optimize.qap_bound", "certificate")
    with pytest.raises(HarnessNotTrustworthy) as exc:
        assert_branch_only_symbol("keybo.optimize.qap_bound", "not_a_real_symbol")
    assert "only on the branch under test" in str(exc.value)


# --- the mutation pre-flight --------------------------------------------------------------


class _FakeHarness:
    """A suite whose exit code we control, standing in for a real pytest run."""

    def __init__(self, *, detects: bool) -> None:
        self.detects = detects
        self.mutated = False
        self.restored = False

    def run(self) -> int:
        if self.mutated and self.detects:
            return 1
        return 0

    def mutate(self) -> None:
        self.mutated = True

    def restore(self) -> None:
        self.mutated = False
        self.restored = True


def test_the_preflight_passes_a_harness_that_detects_breakage() -> None:
    h = _FakeHarness(detects=True)
    assert_harness_detects_a_fatal_mutant(h.run, h.mutate, h.restore)
    assert h.restored, "restore must run"


def test_the_preflight_CATCHES_the_harness_bug_that_actually_happened() -> None:
    """A harness that always reports success — the case that reported 24/24 SURVIVED."""
    h = _FakeHarness(detects=False)
    with pytest.raises(HarnessNotTrustworthy) as exc:
        assert_harness_detects_a_fatal_mutant(h.run, h.mutate, h.restore)
    assert "every SURVIVED verdict" in str(exc.value)
    assert h.restored, "restore must run even when the check fails"


def test_the_preflight_refuses_to_start_from_a_red_suite() -> None:
    """Otherwise 'caught' and 'already broken' are indistinguishable."""

    def red() -> int:
        return 1

    with pytest.raises(HarnessNotTrustworthy) as exc:
        assert_harness_detects_a_fatal_mutant(red, lambda: None, lambda: None)
    assert "does not pass before mutation" in str(exc.value)


def test_the_preflight_notices_a_failed_restore() -> None:
    calls = {"n": 0}

    def run() -> int:
        calls["n"] += 1
        return 0 if calls["n"] == 1 else 1  # never returns green again

    with pytest.raises(HarnessNotTrustworthy) as exc:
        assert_harness_detects_a_fatal_mutant(run, lambda: None, lambda: None)
    assert "restore" in str(exc.value)


# --- operand and discrimination guards ---------------------------------------------------


def test_assert_operands_computed_catches_the_all_inf_AB() -> None:
    assert assert_operands_computed([1.0, 2.0], "arms") == [1.0, 2.0]
    with pytest.raises(HarnessNotTrustworthy) as exc:
        assert_operands_computed([float("-inf")] * 3, "candidate scores")
    assert "no difference" in str(exc.value)
    with pytest.raises(HarnessNotTrustworthy):
        assert_operands_computed([], "empty")


def test_assert_discriminating_catches_a_saturated_metric() -> None:
    """The shipped-test defect: both candidates at exactly 1.0, ranked by stable sort."""
    assert_discriminating([0.9, 0.5], "leaderboard")
    with pytest.raises(HarnessNotTrustworthy) as exc:
        assert_discriminating([1.0, 1.0], "rho_frac_ceiling over two depths")
    assert "tie-break artifact" in str(exc.value)
    # a single value has nothing to discriminate and is not an error
    assert_discriminating([1.0], "single")
