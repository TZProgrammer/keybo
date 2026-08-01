"""The campaign field, the published figures, and the one gauge construction every driver uses.

Shared so the reconciliation step, the comparison table and the fresh search cannot drift onto
three slightly different rulers — which is the exact defect this arm exists to measure. A
second copy of the gauge wiring is a second place for the objective under test and the
objective being reconciled to diverge.
"""

from __future__ import annotations

import os
from pathlib import Path

# Pin the thread vars at import, before any consumer pulls in xgboost through
# `keybo.analysis.timecard`. `setdefault` so an explicit outer value still wins.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

#: This worktree's `src`, which every driver must be running out of. The shared venv resolves
#: `keybo` to `repos/keybo/src` — a checkout that follows whatever branch it happens to be on —
#: so a driver that forgets PYTHONPATH silently measures OTHER code. Four agents hit this.
WORKTREE = Path(__file__).resolve().parents[1]

#: The 13 boards named in the campaign field, all permutations of C30M
#: (`qwertyuiopasdfghjkl'zxcvbnm,.-`), so all comparable on one corpus-restricted mean.
#: BALL-1 / arm B / arm A are given literally as the brief states them; the rest come from
#: `keybo.cli.analyze._EXTRA_NAMED` and `keybo.layouts.NAMED_LAYOUTS`, which is where the
#: campaign's own `analyze` invocations resolve them from, so a transcription error here would
#: show up as a reconciliation failure rather than as a quiet ranking change.
CAMPAIGN_FIELD: dict[str, str] = {
    "BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-c30m": "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "p16-balance": "frlwg'uyoksntdc.ieahvxmpb,-jqz",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "arm-A": "udy.,fgpmliheaocsntr-k'qjwzbvx",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
}

#: Published ms/char figures, for the pre-flight reconciliation. Cited to the ledger lines they
#: are read from so a successor can re-check the citation, not just the arithmetic.
#: PREREGISTRATIONS.md:9426 (arm B) and :9423 (BALL-1), the ARMG-1 board comparison.
LEDGER_FIGURES: dict[str, float] = {
    "arm-B": 253.900579,
    "BALL-1": 253.966426,
}

#: The MODEL-SEED floor (ms/char): the estimator spread of the gauge itself across the three
#: K31 model seeds. This is the right floor for THIS comparison because every board here is a
#: FIXED INPUT — the only noise between two of these numbers is the estimator's, not a
#: search's. The campaign's other floor, 0.883, is the spread of stochastic SEARCH OUTCOMES and
#: would be the right one only for "could a rerun of the search have produced this board".
#: Using the search spread on fixed boards would be a ~6.5x too-loose bar.
SEED_FLOOR = 0.135

#: The search spread, recorded only so the report can name the floor it did NOT use and why.
SEARCH_SPREAD_FLOOR = 0.883


def assert_own_keybo() -> None:
    """Fail loudly if `keybo` resolved to the shared checkout instead of this worktree.

    Called by every driver's `main`. The failure this guards is silent and plausible: the
    shared `repos/keybo/src` is a real, working keybo whose branch moved repeatedly during this
    campaign, so a run against it produces believable numbers for the wrong code.
    """
    import keybo

    got = Path(keybo.__file__).resolve()
    if not str(got).startswith(str(WORKTREE)):
        raise SystemExit(
            f"keybo resolved to {got}, not this worktree ({WORKTREE}). Re-run with "
            f"PYTHONPATH={WORKTREE / 'src'} — the shared checkout follows another branch, so "
            f"these numbers would describe code this driver is not the source of."
        )


def gauge(corpus: str | None = None, target_wpm: float = 90.0):
    """``(scorer, surface)`` for the REPORTED gauge over ``corpus``.

    The single construction point for the ruler. Returns the surface too, because the
    reconciliation step needs `card` (the slow path it reconciles against) and the report needs
    the coverage denominator.
    """
    from keybo.analysis import surfaces as SF
    from keybo.analysis.timecard import default_surface, gauge_scorer_from_surface

    surface = default_surface(target_wpm, corpus)
    return gauge_scorer_from_surface(surface, SF.C30M), surface
