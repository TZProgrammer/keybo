"""Write the REAL pytest exit code from inside the pytest process itself.

Not a parsed summary line: pytest's own `pytest_sessionfinish` receives the true exitstatus, so a
crash, a collection error or an internal error all land here with their real code. Parsing "N
passed" from stdout would report success for a session that errored after the last test — and the
project sets ``addopts = "-q"``, so a second ``-q`` suppresses that line entirely.

⚠ THIS HOOK IS ONLY ACTIVE IF THIS FILE IS ACTUALLY LOADED, AND FOR A FULL-REPO RUN IT IS NOT.
`pyproject.toml` sets ``testpaths = ["tests"]``, so a bare ``pytest`` collects only ``tests/`` and
never reaches this directory — the hook never runs and **no sentinel is written at all**. That
happened here: a full-repo run printed "576 passed" with `FULL_SHELL_RC=0` while this sentinel file
was absent, which would have meant reporting a *parsed* rc as if it were the verified one. Absence
of the sentinel is NOT evidence of rc=0.

So invoke a full-repo run one of these two ways, both of which load this file as a **plugin**
independently of collection:

    PYTHONPATH=$PWD/keybo-e2e KEYBO_RC_SENTINEL=<path> \
        uv run --no-sync pytest tests keybo-e2e -p conftest

    # or, for the harness suite alone, name the files (this directory is then collected normally):
    KEYBO_RC_SENTINEL=<path> uv run --no-sync pytest keybo-e2e/test_layout_specialize.py

And check two things before believing a green result: that the sentinel file **exists** (a missing
file is a failed gate, not a pass), and — at least once per harness change — that it still **bites**
on a deliberate failure. `test_layout_specialize.py` asserts the first; the second is a one-off
probe (a temp test with `assert False` must produce `rc=1 failed=1`).
"""

import os
from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    target = os.environ.get("KEYBO_RC_SENTINEL")
    if target:
        Path(target).write_text(
            f"rc={int(exitstatus)}\n"
            f"collected={getattr(session, 'testscollected', 'NA')}\n"
            f"failed={getattr(session, 'testsfailed', 'NA')}\n"
        )
