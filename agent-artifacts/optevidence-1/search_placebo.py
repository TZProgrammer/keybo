"""SEARCH-NOISE PLACEBO — how much does a champion's ms/char move between independent
runs of the SAME arm at a SMALLER budget?

Every headline in this campaign is a difference between two searches, and a search is a
stochastic optimizer: two runs of the same arm with different seeds land on different
champions with different ms/char. Without this band, "arm A is 1.2 ms/char worse than arm B"
cannot be distinguished from run-to-run variation (trap 34: a count/point statistic needs a
placebo before a small delta is readable).

Runs R independent seeds of each arm at a reduced budget, and reports the spread of the
champion's ms/char and trained-objective score. The band is a CONSERVATIVE overestimate for
the full-budget arms (less budget = more variance), which is the right direction: if the
observed arm A -> arm B gap exceeds even this inflated band, the gap is real.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import evobj as EV  # noqa: E402

STATE = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARM_JSON = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["evidence", "baseline"]

    fe = EV.FastEval(corpus=None, weights_json=ARM_JSON, with_surface=True)
    results: dict[str, list[dict]] = {a: [] for a in arms}
    tmpdir = Path(tempfile.mkdtemp(prefix="placebo-"))

    for arm in arms:
        for r in range(repeats):
            seed = 900_000 + 7919 * r
            out = tmpdir / f"{arm}-{r}.json"
            cmd = ["uv", "run", "--no-sync", "python", str(Path(__file__).parent / "search.py"),
                   "--arm", arm, "--budget", str(budget), "--islands", "20", "--epochs", "12",
                   "--overshoot", "1.95", "--ga-share", "0.6", "--polish-sweeps", "40",
                   "--seed", str(seed), "--out", str(out)]
            proc = subprocess.run(cmd, cwd="/tmp/optev", capture_output=True, text=True,
                                  timeout=3600)
            if proc.returncode != 0:
                print(f"  {arm} seed {seed}: FAILED rc={proc.returncode}\n{proc.stderr[-800:]}",
                      flush=True)
                continue
            blob = json.load(open(out))
            lay = blob["champion"]["layout"]
            p = EV.perm_of(lay)[None]
            g = fe.gauges(p)
            ev = float(fe.evidence_score(g)[0])
            ms = float(g["_ms_per_char"][0])
            results[arm].append({"seed": seed, "layout": lay, "trained_fitness":
                                 blob["champion"]["fitness"], "evidence_score": ev,
                                 "ms_per_char": ms, "unique_evals": blob["unique_evals"]})
            print(f"  {arm} seed {seed}: champ {lay} trained={blob['champion']['fitness']:.4f} "
                  f"evidence={ev:.4f} ms/char={ms:.4f} ({blob['unique_evals']:,} uniq)", flush=True)

    summary = {"budget_per_run": budget, "repeats": repeats, "runs": results, "bands": {}}
    for arm, rows in results.items():
        if len(rows) < 2:
            continue
        ms = np.array([r["ms_per_char"] for r in rows])
        ev = np.array([r["evidence_score"] for r in rows])
        summary["bands"][arm] = {
            "n": len(rows),
            "ms_per_char": {"mean": float(ms.mean()), "sd": float(ms.std(ddof=1)),
                            "min": float(ms.min()), "max": float(ms.max()),
                            "range": float(ms.max() - ms.min())},
            "evidence_score": {"mean": float(ev.mean()), "sd": float(ev.std(ddof=1)),
                               "min": float(ev.min()), "max": float(ev.max()),
                               "range": float(ev.max() - ev.min())},
            "n_distinct_champions": len({r["layout"] for r in rows}),
        }
    summary["note"] = (
        f"CONSERVATIVE band: each run used {budget:,} unique evals vs the arms' 10M, so this "
        "OVERSTATES the full-budget run-to-run spread. A gap that exceeds this band is real.")
    path = STATE / "search-noise-placebo.json"
    json.dump(summary, open(path, "w"), indent=1)
    print(f"\nWROTE {path}")
    for arm, b in summary["bands"].items():
        print(f"  {arm:<12s} ms/char sd={b['ms_per_char']['sd']:.4f} "
              f"range={b['ms_per_char']['range']:.4f} over n={b['n']} "
              f"({b['n_distinct_champions']} distinct champions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
