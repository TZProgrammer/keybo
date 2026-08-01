"""The suite's BLAS/OpenMP thread caps must be in force, and must be OVERRIDABLE.

XGBoost defaults to one thread per core. On a 192-core shared box that means every model fit in
this suite spawns 192 threads which contend rather than work, and under fleet load the suite looks
HUNG rather than slow — four separate agents independently rediscovered that in one session, which
is why the caps are now applied in ``tests/conftest.py`` instead of being re-derived per person.

Measured on this box, same test, same tree:
    tests/training/test_tune.py::test_tune_returns_param_dict    107.09s unpinned -> 2.19s pinned

Two properties are worth pinning, and neither is about speed (a timing assertion would be flaky):

1. **The caps are applied at conftest MODULE scope, not from a fixture.** OpenMP samples
   ``OMP_NUM_THREADS`` when its runtime initializes, so a value set after ``import xgboost`` is
   largely ignored — measured 38.99s for a fit whose env was set post-import against 0.06s with
   the env set first. A session-scoped autouse fixture runs long after collection imported the
   test modules, so it would report the right value from ``os.environ`` and buy almost nothing.
   That is ``present != effective``, and it is the failure mode this file exists to prevent
   someone from "tidying" the pins into.
2. **An explicitly-set value wins.** Someone bisecting a threading bug, profiling across all
   cores, or running under a scheduler that already pinned the process must keep their choice.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import _THREAD_VARS

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_all_four_thread_vars_are_capped_for_the_session(thread_caps):
    """All four, because they gate different libraries and one uncapped var un-pins its own.

    Asserted through the fixture rather than ``os.environ`` directly, so the fixture stays the one
    place a test asks what the effective caps are.
    """
    assert set(thread_caps) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    for name, value in thread_caps.items():
        assert int(value) >= 1, name
        assert os.environ[name] == value, f"{name} drifted from what the fixture reports"


def test_an_EXPLICIT_value_from_the_caller_is_not_clobbered():
    """The overridability requirement, tested where it actually happens: a subprocess.

    The caps are applied at import time, so no in-process manipulation can exercise the decision —
    by the time this test runs, conftest has long since chosen. Only a fresh interpreter with the
    var pre-set can prove ``setdefault`` semantics, so this spawns one and imports the real
    conftest in it.
    """
    # Strip the caps this process inherited before pre-setting two of them: the point is to give
    # the child exactly two explicit values and NO value for the other two, so both halves of
    # setdefault get exercised. Passing os.environ through unmodified would hand the child all
    # four already-set (this suite's own pins), and the "still gets capped" half would be vacuous.
    env = {name: value for name, value in os.environ.items() if name not in _THREAD_VARS}
    env |= {"OMP_NUM_THREADS": "7", "MKL_NUM_THREADS": "5"}
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import tests.conftest as c; "
            "import os; "
            "print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'], "
            "os.environ['OPENBLAS_NUM_THREADS'], sorted(c.THREAD_VARS_PINNED_BY_CONFTEST))",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    omp, mkl, openblas, pinned = result.stdout.strip().split(" ", 3)

    assert omp == "7", "an explicitly-set OMP_NUM_THREADS must survive the test harness"
    assert mkl == "5", "...and so must every other var the caller chose"
    assert openblas == "2", "while the ones the caller did NOT set still get capped"
    assert "OMP_NUM_THREADS" not in pinned and "MKL_NUM_THREADS" not in pinned
    assert "OPENBLAS_NUM_THREADS" in pinned


def test_the_caps_are_applied_at_IMPORT_and_not_deferred_to_a_fixture():
    """The load-bearing property: importing conftest is ENOUGH, no pytest run required.

    This is what distinguishes a working pin from an inert one. If someone moves the assignment
    into the ``thread_caps`` fixture body, importing the module will no longer set anything and
    this fails — which is the only cheap way to catch a change that leaves every other test in
    this file passing while the suite silently goes back to 107s.
    """
    env = {name: value for name, value in os.environ.items() if name not in _THREAD_VARS}
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import tests.conftest; import os; "
            "print(','.join(f'{k}={os.environ.get(k, \"UNSET\")}' for k in "
            "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')))",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    settings = dict(pair.split("=") for pair in result.stdout.strip().split(","))

    assert "UNSET" not in settings.values(), (
        f"importing tests/conftest.py must be sufficient to apply the caps, got {settings}. "
        "If these moved into a fixture they are applied AFTER xgboost imports, where OpenMP has "
        "already read its thread count and the pin no longer bites."
    )
    assert set(settings.values()) == {"2"}


def test_the_caps_do_not_change_a_computed_RESULT():
    """A thread count must be a performance knob only, never an answer knob.

    Fits the same tiny model at two different OMP_NUM_THREADS values in the SAME process. XGBoost's
    histogram building is thread-count dependent in general, so this pins that for the shapes this
    suite actually trains the predictions are identical — the claim the fixture makes when it says
    it changes no result.
    """
    import numpy as np

    from keybo.data.strokes import StrokeRow
    from keybo.features import bigram_features_from_positions
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.training.train import train_bigram_model

    rows = [
        StrokeRow(
            layout="qwerty",
            positions=((-1, 3), (1, 2)),
            ngram=("th", "he", "an", "in")[i % 4],
            frequency=5,
            samples=[(wpm, int(12000 / wpm * (1 + wpm / 200.0)), i, 50) for wpm in (60, 90, 120)],
        )
        for i in range(80)
    ]
    features = np.array(
        [
            bigram_features_from_positions(
                ROW_STAGGERED_30,
                (ROW_STAGGERED_30.slots[0], ROW_STAGGERED_30.slots[3]),
                wpm=90,
            )
        ]
    )

    def fit_and_predict(n_threads: int) -> float:
        model = train_bigram_model(
            rows, target_wpm=90, n_estimators=15, max_depth=3, nthread=n_threads
        )
        return float(model.predict(features)[0])

    assert fit_and_predict(1) == pytest.approx(fit_and_predict(2), abs=0.0), (
        "the thread cap must be a performance knob, not an answer knob"
    )
