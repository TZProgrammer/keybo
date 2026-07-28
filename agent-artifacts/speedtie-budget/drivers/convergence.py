"""Convergence diagnostic from each run's own epoch trace.

ARME-1 registered the rule this implements: **diagnose convergence by whether best-fitness has
STOPPED IMPROVING, not by budget fraction.** That matters here because the whole question is
whether the 1M runs were under-converged — and "1M is only 10% of 10M" is exactly the
budget-fraction reasoning ARME-1 rejected. What settles it is where the champion last moved.

Reads only the `search.py` epoch log lines, which are written per epoch and are therefore
independent of anything my analysis code computes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUNS = Path("/local/home/zegertho/agent/state/speedtie/artifacts/runs")
LINE = re.compile(r"epoch (\d+)/(\d+): unique=([\d,]+) \(calls this epoch [\d,]+\) "
                  r"best=([\d.]+) \[(\S+)\]")


def trace(log: Path) -> list[dict]:
    rows = []
    for m in LINE.finditer(log.read_text()):
        rows.append({"epoch": int(m.group(1)), "epochs_total": int(m.group(2)),
                     "unique": int(m.group(3).replace(",", "")),
                     "best": float(m.group(4)), "layout": m.group(5)})
    return rows


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    out = {}
    for log in sorted(RUNS.glob(f"b{budget}-r*.log")):
        r = int(log.stem.split("-r")[1])
        rows = trace(log)
        if not rows:
            out[str(r)] = {"error": "no epoch lines parsed"}
            continue
        final = rows[-1]["best"]
        # last epoch at which the best IMPROVED (strictly)
        last_imp = rows[0]["epoch"]
        for a, b in zip(rows, rows[1:], strict=False):
            if b["best"] < a["best"] - 1e-12:
                last_imp = b["epoch"]
        at_last_imp = next(x for x in rows if x["epoch"] == last_imp)
        half = rows[len(rows) // 2]
        # first epoch whose unique count passes 1M — the point matched to the 1M placebo
        at_1m = next((x for x in rows if x["unique"] >= 1_000_000), None)
        out[str(r)] = {
            "seed": 900_000 + 7919 * r,
            "epochs_run": rows[-1]["epoch"],
            "unique_final": rows[-1]["unique"],
            "best_final": final,
            "champion_final": rows[-1]["layout"],
            "last_improvement_epoch": last_imp,
            "unique_at_last_improvement": at_last_imp["unique"],
            "frac_of_budget_at_last_improvement": at_last_imp["unique"] / rows[-1]["unique"],
            "epochs_flat_after_last_improvement": rows[-1]["epoch"] - last_imp,
            "unique_spent_after_last_improvement": rows[-1]["unique"] - at_last_imp["unique"],
            "improvement_over_final_half": half["best"] - final,
            "best_at_half": half["best"],
            "at_first_epoch_past_1M_unique": (
                {"epoch": at_1m["epoch"], "unique": at_1m["unique"], "best": at_1m["best"],
                 "layout": at_1m["layout"]} if at_1m else None),
        }
    path = RUNS.parent / f"convergence-{budget}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"WROTE {path}\n")
    print(f"{'r':>2} {'seed':>7} {'uniq_final':>11} {'best_final':>12} {'lastImpEp':>9} "
          f"{'uniqAtLastImp':>13} {'epochsFlat':>10} {'improve_2ndHalf':>15}")
    for r, v in sorted(out.items(), key=lambda kv: int(kv[0])):
        if "error" in v:
            print(f"{r:>2} PARSE ERROR")
            continue
        print(f"{r:>2} {v['seed']:>7} {v['unique_final']:>11,} {v['best_final']:>12.6f} "
              f"{v['last_improvement_epoch']:>9} {v['unique_at_last_improvement']:>13,} "
              f"{v['epochs_flat_after_last_improvement']:>10} "
              f"{v['improvement_over_final_half']:>15.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
