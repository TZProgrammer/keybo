"""Shared pytest fixtures and path helpers for the keybo test suite."""

import os
from pathlib import Path

import pytest

#: BLAS/OpenMP thread caps applied to the whole test session. XGBoost otherwise defaults to one
#: thread per core, and on a 192-core shared box every model fit in the suite spawns 192 threads
#: that spend their time contending rather than working. Measured on this box:
#: ``tests/training/test_tune.py::test_tune_returns_param_dict`` takes **107.09s unpinned vs 2.19s
#: pinned — 48.9x**. Under fleet load the suite looks HUNG rather than slow, and four separate
#: agents rediscovered that independently in one session before this fixture existed.
#:
#: ⚠ THIS MUST STAY AT MODULE SCOPE, NOT IN THE FIXTURE BELOW. OpenMP reads OMP_NUM_THREADS once,
#: when its runtime initializes, so a value set after ``import xgboost`` is largely ignored:
#: measured 38.99s for a fit whose env was set post-import vs 0.06s for the same fit with the env
#: set first. pytest imports the root conftest before collecting any test module, which is what
#: makes this the last point where the setting still bites. A session-scoped autouse fixture runs
#: far too late — it would look correct, report the right values via os.environ, and buy almost
#: nothing. (`present != effective`.)
#:
#: 2 rather than 1: these tests fit real (tiny) XGBoost models, and 1 would serialize them for no
#: benefit while 2 keeps the fits parallel enough without oversubscribing a shared box.
_THREAD_VARS = {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
}

#: Which vars this module actually set, versus which the caller had already chosen. Recorded at
#: import so the fixture below can report it without re-deriving (and so the override is
#: assertable — see ``tests/test_thread_pinning.py``).
THREAD_VARS_PINNED_BY_CONFTEST = {
    name: value for name, value in _THREAD_VARS.items() if name not in os.environ
}

# setdefault, never assignment: an explicit value from the caller WINS. Someone bisecting a
# threading bug, profiling with all cores, or running under a scheduler that already pinned the
# process must not have their choice silently overwritten by the test harness.
for _name, _value in _THREAD_VARS.items():
    os.environ.setdefault(_name, _value)

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@pytest.fixture(scope="session", autouse=True)
def thread_caps() -> dict[str, str]:
    """The thread caps in force for this session, as ``{var: value}``.

    The values are applied at MODULE scope above, because by the time any fixture runs OpenMP has
    already initialized (see the note on ``_THREAD_VARS``). This fixture therefore does not set
    them — it asserts they are still in force and makes them injectable, so a test can read the
    effective caps instead of hardcoding "2" and a mid-session ``monkeypatch.delenv`` that would
    quietly un-pin the rest of the run fails here rather than as a mystery slowdown.
    """
    missing = [name for name in _THREAD_VARS if name not in os.environ]
    assert not missing, f"thread caps went missing mid-session: {missing}"
    return {name: os.environ[name] for name in _THREAD_VARS}


@pytest.fixture
def corpus_dir() -> Path:
    """Directory holding the committed n-gram frequency files."""
    return CORPUS_DIR


#: The skipgram table the campaign's frozen gauge boards were computed on.
#: ``1-skip31.txt`` IS the trigram marginalization ``skip(a,c) = sum_b tri(a,b,c)``
#: (``data/build_corpus.py``, verified byte-exact there); ``1-skip.txt`` is a different,
#: unreproducible pass. Gauges that must reproduce a frozen board use this one.
PRODUCTION_SKIPGRAMS = "1-skip31.txt"


@pytest.fixture(scope="session")
def corpora() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """``(bigrams, skipgrams, trigrams)`` from the committed production corpus.

    Session-scoped: the trigram table is ~100k rows and several tests want it.
    """
    from keybo.data.corpus import load_frequencies

    return (
        load_frequencies(str(CORPUS_DIR / "bigrams.txt")),
        load_frequencies(str(CORPUS_DIR / PRODUCTION_SKIPGRAMS)),
        load_frequencies(str(CORPUS_DIR / "trigrams.txt")),
    )
