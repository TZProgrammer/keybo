"""Step 0: reconcile the gauge objective to PUBLISHED ledger figures, BEFORE any new numbers.

The two established facts this campaign arm combines were each verified elsewhere, and both
carry the same failure mode: a *plausible* re-implementation of the reported gauge is ~1.5e-2
off (≈11 resolution floors) while still ranking boards in nearly the right order, so it looks
right in every comparison. The defence is not "the code reviews well" — it is that a board
whose value is ALREADY PUBLISHED must come back out of this objective at the published value.

So this driver refuses to produce any comparison number until two ledger figures reproduce:

    arm B    flmpg-yuo,sntdcireahkxbwv'.jzq    253.900579  (PREREGISTRATIONS.md:9426)
    BALL-1   flmpg-yuo,sntcdireahkxbwv'.jzq    253.966426  (PREREGISTRATIONS.md:9423)

If they do not reproduce, the run is not measuring what it thinks it is measuring, and the
correct output is that fact rather than a table of new numbers.

Two independent things are checked, because they can fail independently:

1. STRUCTURAL PARITY — the fast table objective vs ``TimeSurface.card``'s own ~50 ms loop, on
   the same board in the same process. Catches a mis-built table. Tolerance 1e-12 (the two
   upstream implementations measured 1.2e-14).
2. LEDGER PARITY — the objective's ms/char vs the number in the ledger. Catches the case where
   the objective is internally consistent but is not the ruler the campaign reported on (wrong
   corpus, wrong WPM, wrong seed set). This is the one that cannot be faked by careful code.

Usage:  reconcile.py [--corpus NAME] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Thread pinning MUST precede the xgboost import (OpenMP samples the env at runtime init;
# setting these afterwards is inert — measured 0.08 s vs 17.62 s on this box).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boards import CAMPAIGN_FIELD, LEDGER_FIGURES, assert_own_keybo, gauge  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402

#: Structural parity: the fast table vs ``card``'s own loop. The two reviewed upstream builds
#: measured 1.2e-14, so this is ~2 orders of magnitude of headroom over float64 accumulation
#: order and still ~10 orders tighter than the 1.5e-2 naive re-implementation it must reject.
STRUCTURAL_TOL = 1e-12

#: Ledger parity: |mine - published| in ms/char. The ledger prints 6 decimals, so anything
#: under 1e-5 is agreement to the precision the figure was published at. This is deliberately
#: NOT a relative tolerance: the quantity is a mean in ms/char and the floor it will later be
#: compared against (0.135) is absolute, so the reconciliation should be too.
LEDGER_TOL = 1e-5


def check(corpus: str | None) -> dict:
    """Run both parity checks on every board that has a published figure."""
    scorer, surface = gauge(corpus)
    rows = []
    for name, published in sorted(LEDGER_FIGURES.items()):
        lay = CAMPAIGN_FIELD[name]
        layout = Layout(lay, ROW_STAGGERED_30)
        # `float(...)` on the way out: these come back as numpy scalars, whose `bool` is
        # `np.bool_` and is NOT JSON-serializable. Casting at the boundary rather than with a
        # custom encoder keeps the artifact a plain-JSON file any reader can load.
        mine = float(scorer.ms_per_char(layout))
        structural = float(scorer.parity_rel_dev(layout))
        rows.append(
            {
                "board": name,
                "layout": lay,
                "published_ms_per_char": published,
                "measured_ms_per_char": mine,
                "ledger_abs_diff": abs(mine - published),
                "structural_rel_dev": structural,
                "structural_ok": bool(structural <= STRUCTURAL_TOL),
                "ledger_ok": bool(abs(mine - published) <= LEDGER_TOL),
            }
        )
    return {
        "corpus": corpus,
        "charset": SF.C30M,
        "coverage_pct": float(100.0 * scorer._covered / max(surface.total_mass, 1)),
        "structural_tol": STRUCTURAL_TOL,
        "ledger_tol": LEDGER_TOL,
        "rows": rows,
        "all_structural_ok": all(r["structural_ok"] for r in rows),
        "all_ledger_ok": all(r["ledger_ok"] for r in rows),
    }


def report(result: dict) -> None:
    print(f"corpus={result['corpus']!r}  coverage={result['coverage_pct']:.4f}%")
    print(
        f"{'board':10s} {'published':>14s} {'measured':>14s} {'abs diff':>11s} "
        f"{'struct rel':>11s}  verdict"
    )
    for r in result["rows"]:
        verdict = "OK" if (r["structural_ok"] and r["ledger_ok"]) else "MISMATCH"
        print(
            f"{r['board']:10s} {r['published_ms_per_char']:14.6f} "
            f"{r['measured_ms_per_char']:14.6f} {r['ledger_abs_diff']:11.3e} "
            f"{r['structural_rel_dev']:11.3e}  {verdict}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=None, help="corpus name (default: production default)")
    ap.add_argument("--json", default=None, help="write the result JSON here")
    args = ap.parse_args(argv)

    assert_own_keybo()
    result = check(args.corpus)
    report(result)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")

    if not (result["all_structural_ok"] and result["all_ledger_ok"]):
        print(
            "\nRECONCILIATION FAILED — refusing to produce comparison numbers. The objective in "
            "this process is not the ruler the ledger figures were published on, so any table "
            "built from it would be a different measurement wearing the campaign's labels."
        )
        return 1
    print("\nreconciled: the objective reproduces the published figures; comparisons may proceed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
