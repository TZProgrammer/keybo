"""Positive control for the finger-travel test harness (FT round, 2026-07-28).

A green suite is evidence only if the suite can go red. This drives
``keybo.testkit.assert_harness_detects_a_fatal_mutant`` over
``tests/analysis/test_finger_travel.py`` with a mutant that MUST be caught: charge the
DEPARTING finger instead of the landing one.

That mutant is chosen deliberately because it is **share-preserving in aggregate** — it still
produces eight cells summing to 100.0 — so it is exactly the class of error the exactness tests
are blind to. If the suite passes with it applied, every PASS this harness reports is
uninformative.

Run: ``PYTHONPATH=src python agent-artifacts/ft_harness_control.py``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "keybo" / "analysis" / "finger_travel.py"
SUITE = ROOT / "tests" / "analysis" / "test_finger_travel.py"

#: The landing-finger attribution, and the mutant that charges the departure instead.
ORIGINAL = """                label = finger_label(geometry, second[0])
                self._charge(
                    charged,
                    label,
                    freq * geometry.distance(HOME_POSITION[label], second),
                    "from-home travel",
                )"""
MUTANT = """                label = finger_label(geometry, first[0])
                self._charge(
                    charged,
                    label,
                    freq * geometry.distance(HOME_POSITION[label], second),
                    "from-home travel",
                )"""


def run_suite() -> int:
    """Exit code of the suite — NEVER parse prose; the original harness bug was a grep."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(SUITE), "-q", "-x"],
        cwd=ROOT,
        capture_output=True,
    ).returncode


def apply_fatal_mutant() -> None:
    source = TARGET.read_text()
    if ORIGINAL not in source:
        raise SystemExit(
            "the mutation target text is not in finger_travel.py — the control cannot run, and "
            "a control that silently no-ops is worse than none"
        )
    TARGET.write_text(source.replace(ORIGINAL, MUTANT, 1))


def restore() -> None:
    source = TARGET.read_text()
    if MUTANT in source:
        TARGET.write_text(source.replace(MUTANT, ORIGINAL, 1))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from keybo.testkit import assert_harness_detects_a_fatal_mutant, assert_module_under

    assert_module_under("keybo.analysis.finger_travel", ROOT / "src")
    assert_harness_detects_a_fatal_mutant(run_suite, apply_fatal_mutant, restore)
    print("HARNESS CONTROL PASSED: the suite is green, goes RED on a share-preserving")
    print("wrong-finger mutant, and returns to green after restore.")
