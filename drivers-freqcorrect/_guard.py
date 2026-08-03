"""Shared preamble: pin threads BEFORE xgboost, force MY worktree, assert D5.

Import FIRST in every driver. Thread pins are inert once xgboost has loaded, and the sys.path
insert is load-bearing: the venv's editable install resolves `import keybo` to the SHARED checkout
/local/home/zegertho/repos/keybo, which other agents move between branches (measured trap D5).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "48")

import hashlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

MY_WT = "/local/home/zegertho/repos/keybo-wt-freqcorrect"
SHARED = "/local/home/zegertho/repos/keybo"
E2E = "/local/home/zegertho/keybo-e2e"
BI = f"{E2E}/bistrokes31_v1.tsv"
SHIPPED = f"{SHARED}/data/models/k31"          # READ-ONLY. Never written.
ART = "/local/home/zegertho/agent/state/freqcorrect/artifacts"
CACHE = "/tmp/freqcorrect-drv/cache"

sys.path.insert(0, MY_WT + "/src")
os.makedirs(ART, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

WPM = 90.0            # production scoring wpm
BOOT_SEED = 20260803  # registered in FREQCORRECT-1 PREREG §9
# CALIB-1's cell construction, reproduced EXACTLY so my numbers are comparable to k03's.
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
SEEDS = [0, 1, 2]
HOLDOUTS = ["azerty", "dvorak", "qwerty", "qwertz"]


def _branch(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:                                    # noqa: BLE001
        return f"<unknown: {e}>"


def assert_d5():
    """G-D5: refuse to measure another agent's branch. Prints BOTH checkouts' branches."""
    import keybo
    resolved = keybo.__file__
    mine, shared = _branch(MY_WT), _branch(SHARED)
    print(f"  keybo.__file__ = {resolved}")
    print(f"  my worktree {MY_WT} @ branch {mine!r}")
    print(f"  SHARED checkout {SHARED} @ branch {shared!r}  (not mine; must not be measured)")
    assert resolved.startswith(MY_WT + "/"), (
        f"D5 FAIL: keybo resolved to {resolved}, NOT my worktree {MY_WT}. The shared checkout is "
        f"live for other agents (currently on {shared!r}); refusing to measure another branch."
    )
    assert mine == "freqcorrect", f"D5 FAIL: my worktree is on branch {mine!r}, expected 'freqcorrect'"
    return resolved


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_rows(cache=True):
    """The 2202 bistroke rows. Asserts the frame has not drifted.

    Parsing the 609 MB TSV costs ~230 s and every driver needs the same rows, so the parsed
    objects are pickled to /tmp on first use. The cache key carries the source file's SIZE and
    MTIME, so an edited or replaced TSV can never be served from a stale pickle. /tmp is tmpfs
    here (a reboot wipes it), which is fine: this is pure derived data and a miss only costs the
    230 s parse again.
    """
    import pickle

    from keybo.data.strokes import load_strokes

    st = os.stat(BI)
    path = os.path.join(CACHE, f"rows_bi2_{st.st_size}_{int(st.st_mtime)}.pkl")
    if cache and os.path.exists(path):
        with open(path, "rb") as fh:
            rows = pickle.load(fh)
        print(f"  rows from cache {path}")
    else:
        rows = load_strokes(BI, ngram_len=2, wpm_threshold=0, min_samples=1)
        if cache:
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)   # atomic: a killed writer cannot leave a half-written pickle
            print(f"  rows cached to {path}")
    assert len(rows) == 2202, f"frame drift: {len(rows)} != 2202"
    return rows
