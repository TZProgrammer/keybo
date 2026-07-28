"""Pre-flight checks a test harness must pass before its own results mean anything.

Three failures in this project came from an instrument that could not report a problem:

* a mutation harness judged mutants by a case-sensitive grep on pytest prose, so ``FAILED``
  never matched and it reported **all 24 mutants SURVIVED** — it would have published "the
  gate catches nothing" instead of 18-of-24.
* worktree probes were run with an interpreter whose ``.venv`` carries an editable ``.pth``
  into the SHARED clone's ``src``, so the probe silently tested the wrong tree while every
  printed path looked right. It was caught only by accident, via an ``ImportError`` on a
  symbol that happened to be new on the branch.
* a paired A/B printed ``ARGMAX MOVES: False`` over two arms in which every candidate scored
  ``-inf``, because the objective was never evaluable.

The lesson was registered as prose and kept recurring, so it lives here as code the harnesses
actually call. See :mod:`keybo.verdicts` for the runtime half (finite operands, margins).
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable, Sequence
from pathlib import Path


class HarnessNotTrustworthy(RuntimeError):
    """A harness failed a pre-flight check, so its PASS/SURVIVED verdicts are meaningless."""


def assert_module_under(module_name: str, root: str | Path) -> Path:
    """Assert ``module_name`` resolves to a file under ``root``, and return that path.

    Guards the wrong-tree failure: an editable install can shadow a worktree checkout, and
    every printed path still looks plausible because the module NAME is right. Ask the module
    where it actually lives.
    """
    module = importlib.import_module(module_name)
    location = getattr(module, "__file__", None)
    if location is None:
        raise HarnessNotTrustworthy(f"{module_name!r} has no __file__ to check")
    resolved = Path(location).resolve()
    root_resolved = Path(root).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise HarnessNotTrustworthy(
            f"{module_name!r} resolved to {resolved} which is NOT under {root_resolved} — the "
            f"harness is testing a different tree than it thinks. An editable .pth in the "
            f"interpreter's site-packages is the usual cause; put the intended src on "
            f"PYTHONPATH ahead of it."
        )
    return resolved


def assert_branch_only_symbol(module_name: str, symbol: str) -> None:
    """Assert a symbol that exists ONLY on the branch under test is importable.

    ``assert_module_under`` catches a wrong-tree import only when the paths differ visibly.
    This catches the harder case positively: if the branch adds ``spearman_brown`` and the
    import lacks it, the harness is on the wrong code no matter what any path says.
    """
    module = importlib.import_module(module_name)
    if not hasattr(module, symbol):
        raise HarnessNotTrustworthy(
            f"{module_name!r} has no {symbol!r} — this symbol exists only on the branch under "
            f"test, so the harness has imported a different version of the module"
        )


def assert_harness_detects_a_fatal_mutant(run_suite, apply_fatal_mutant, restore) -> None:
    """Assert the harness reports FAILURE for a mutant that MUST fail, before trusting a pass.

    ``run_suite`` returns a process exit code (0 = pass). ``apply_fatal_mutant`` breaks
    something the suite provably covers; ``restore`` undoes it. Gate on the EXIT CODE — the
    original harness bug was parsing human-readable output.

    ``restore`` runs even if the check raises, because leaving a mutant in place is worse than
    the failure being reported.
    """
    baseline = run_suite()
    if baseline != 0:
        raise HarnessNotTrustworthy(
            f"the suite does not pass before mutation (rc={baseline}); fix that first — a "
            f"harness cannot distinguish 'the mutant was caught' from 'it was already red'"
        )
    try:
        apply_fatal_mutant()
        mutated = run_suite()
    finally:
        restore()
    if mutated == 0:
        raise HarnessNotTrustworthy(
            "the suite PASSED with a fatal mutant applied, so it cannot detect breakage and "
            "every SURVIVED verdict from this harness is uninformative"
        )
    after = run_suite()
    if after != 0:
        raise HarnessNotTrustworthy(
            f"restore() did not return the suite to green (rc={after}) — later results would "
            f"be measured against a mutated tree"
        )


def assert_operands_computed(values: Iterable[float], what: str) -> list[float]:
    """Assert every operand of a forthcoming comparison is finite.

    The lightweight local form of :func:`keybo.verdicts.require_finite`, kept here so a test
    harness can use it without importing runtime code. A comparison over uncomputed operands
    returns the answer meaning "no difference".
    """
    out = [float(v) for v in values]
    if not out:
        raise HarnessNotTrustworthy(f"{what}: no operands, so no comparison is possible")
    bad = [i for i, v in enumerate(out) if not math.isfinite(v)]
    if bad:
        raise HarnessNotTrustworthy(
            f"{what}: operands {bad} are not finite, so any verdict comparing them would "
            f"report 'no difference' regardless of the truth"
        )
    return out


def assert_discriminating(values: Sequence[float], what: str) -> None:
    """Assert a comparison's operands are not all equal — i.e. it can distinguish anything.

    A saturated metric (every candidate at exactly 1.0) makes a leaderboard's ordering a
    stable-sort artifact. That produced a shipped test claiming a preference it never checked.
    """
    vals = assert_operands_computed(values, what)
    if len(set(vals)) <= 1 and len(vals) > 1:
        raise HarnessNotTrustworthy(
            f"{what}: all {len(vals)} operands are identical ({vals[0]!r}), so any ranking "
            f"over them is a tie-break artifact and not a measurement"
        )
