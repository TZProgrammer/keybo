"""REHUNT verification — every REPORTED dominator, re-scored through the ZERO-REUSE slow path.

The hunt's inner loop uses fast bilinear forms. A reported dominator is only a result if the
SLOW reference machinery agrees: ``KmStats.stats`` (Python triple loop), ``ComfortObjective.values``,
``ScissorSeverity.share`` on a real ``Layout``, and a fresh-scorer explicit-loop wfd that shares no
cached array with the fast path. Prior campaign rounds achieved max relative error EXACTLY 0.0.

Also asserted, per reported layout:
  * it is a valid C30M permutation (30 distinct movable chars) — the board-level guard;
  * its 31-key dof mapping is a permutation — the guard whose absence IS the wfd bug;
  * the dominance verdict RE-DERIVED from the slow axes matches the hunt's verdict, using the
    STRICT predicate (>= on every axis AND > on at least one). A self-tie is NOT a dominator.

And per dominator, the axes it actually WINS on, plus its wide-scissor share movement — prior
rounds found dominators win on lsb/sfb/sfs while their wscissor share moved -0.34% to -44.57%
with no relation to achieving dominance, so that column is reported, never used as evidence.

MODELED/gauge only. Held-layout tau saturated at 1.0; Phase-D cancelled. Nothing promoted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus_eval as CE  # noqa: E402
import wscissor_eval as WE  # noqa: E402
from rehunt_hunt import FRAMES  # noqa: E402
from wfd_fix import assert_c30m_permutation  # noqa: E402


def rel_err(fast: float, slow: float) -> float:
    if fast == slow:
        return 0.0
    denom = max(abs(fast), abs(slow))
    return abs(fast - slow) / denom if denom else 0.0


def strict_dominates(cand: dict, targ: dict, frame: list[str], atol: float = 1e-9):
    """(is_dominator, n_ge, n_gt, per-axis win/loss). Oriented so higher is better."""
    cv = np.array([WE.SIGN12[a] * cand[a] for a in frame])
    tv = np.array([WE.SIGN12[a] * targ[a] for a in frame])
    n_ge = int(np.sum(cv >= tv - atol))
    n_gt = int(np.sum(cv > tv + atol))
    per_axis = {
        a: ("win" if cv[i] > tv[i] + atol else "tie" if cv[i] >= tv[i] - atol else "LOSS")
        for i, a in enumerate(frame)
    }
    return (n_ge == len(frame) and n_gt >= 1), n_ge, n_gt, per_axis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="rehunt-*.json outputs to verify")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    boards: dict[tuple, WE.WScissorBoard] = {}
    rows = []
    for run_path in args.runs:
        run = json.loads(Path(run_path).read_text())
        corpus, arm, frame_name = run["corpus"], run["arm"], run["frame"]
        frame = FRAMES[frame_name]
        if run["wfd_mode"] != "corrected":
            raise SystemExit(f"{run_path}: wfd_mode={run['wfd_mode']!r}, refusing to verify")
        key = (corpus, arm)
        if key not in boards:
            boards[key] = WE.WScissorBoard(
                corpus=corpus,
                arm=arm,
                ceilings=CE.SixSurface(corpus).ceiling_map,
                objective="wide",
                wfd_mode="corrected",
            )
        board = boards[key]
        floor_kind = run["floor_kind"]

        # incumbents on the SLOW path too — a dominance test needs both sides scored alike
        inc_slow = {
            name: board.axes12_slow(lay, floor_kind) for name, lay in CE.INCUMBENTS.items()
        }
        inc_fast = {name: run["incumbent_axes"][name] for name in CE.INCUMBENTS}
        inc_err = {
            f"{name}.{a}": rel_err(inc_fast[name][a], inc_slow[name][a])
            for name in CE.INCUMBENTS
            for a in frame
        }

        for target, best in run["per_target_best"].items():
            lay = best["best_layout"]
            assert_c30m_permutation(lay)
            board.gauges["o2"].dof_of_char(lay)  # 31-key dof permutation guard
            fast = best["best_axes"]
            slow = board.axes12_slow(lay, floor_kind)
            errs = {a: rel_err(fast[a], slow[a]) for a in frame}

            if target.startswith("IDEAL"):
                targ_slow = {
                    a: (
                        max(inc_slow[n][a] for n in inc_slow)
                        if WE.SIGN12[a] > 0
                        else min(inc_slow[n][a] for n in inc_slow)
                    )
                    for a in frame
                }
            else:
                targ_slow = inc_slow[target]

            dom, n_ge, n_gt, per_axis = strict_dominates(slow, targ_slow, frame)
            # zero-reuse wfd cross-check: the explicit-loop reference vs the fast contraction
            wfd_fast = float(board.axes12(lay, floor_kind)["wfd"])
            wfd_slow = float(board.gauges["o2"].wfd_slow_reference(lay))

            rows.append(
                dict(
                    run=Path(run_path).name,
                    corpus=corpus,
                    corpus_label=run["corpus_label"],
                    arm=arm,
                    frame=frame_name,
                    frame_size=len(frame),
                    target=target,
                    layout=lay,
                    is_c30m_permutation=True,
                    dof_map_is_permutation=True,
                    self_tie=bool(
                        not target.startswith("IDEAL") and lay == CE.INCUMBENTS[target]
                    ),
                    hunt_dominates=bool(best["dominates_target"]),
                    slow_dominates=bool(dom),
                    verdict_agrees=bool(dom == best["dominates_target"]),
                    hunt_n_ge=int(best["best_n_ge"]),
                    slow_n_ge=n_ge,
                    slow_n_gt=n_gt,
                    max_rel_err=max(errs.values()),
                    per_axis_rel_err=errs,
                    wfd_fast=wfd_fast,
                    wfd_slow_reference=wfd_slow,
                    wfd_rel_err=rel_err(wfd_fast, wfd_slow),
                    axes_slow=slow,
                    target_axes_slow=targ_slow,
                    per_axis_verdict=per_axis,
                    axes_won=[a for a, v in per_axis.items() if v == "win"],
                    axes_lost=[a for a, v in per_axis.items() if v == "LOSS"],
                    wscissor_cand=slow.get("wscissor"),
                    wscissor_target=targ_slow.get("wscissor"),
                    wscissor_pct_change=(
                        100.0 * (slow["wscissor"] - targ_slow["wscissor"]) / targ_slow["wscissor"]
                        if "wscissor" in slow and targ_slow.get("wscissor")
                        else None
                    ),
                    incumbent_max_rel_err=max(inc_err.values()),
                )
            )

    doms = [r for r in rows if r["slow_dominates"]]
    disagree = [r for r in rows if not r["verdict_agrees"]]
    worst = max((r["max_rel_err"] for r in rows), default=0.0)
    worst_wfd = max((r["wfd_rel_err"] for r in rows), default=0.0)
    worst_inc = max((r["incumbent_max_rel_err"] for r in rows), default=0.0)

    print(f"verified {len(rows)} reported (cell, target) rows from {len(args.runs)} runs\n")
    print(f"  max rel err, candidate axes  fast vs SLOW: {worst:.3e}")
    print(f"  max rel err, incumbent axes  fast vs SLOW: {worst_inc:.3e}")
    print(f"  max rel err, wfd  fast vs ZERO-REUSE loop: {worst_wfd:.3e}")
    print(f"  hunt verdict == slow-path verdict:         {len(rows) - len(disagree)}/{len(rows)}")
    print(f"  CONFIRMED dominators (strict, slow path):  {len(doms)}\n")
    for r in disagree:
        print(
            f"   DISAGREE {r['corpus']:9s} arm{r['arm']} {r['frame']:8s} {r['target']:14s} "
            f"hunt={r['hunt_dominates']} slow={r['slow_dominates']} "
            f"n_ge={r['slow_n_ge']}/{r['frame_size']} n_gt={r['slow_n_gt']}"
        )
    print(
        f"{'corpus':9s} {'arm':4s} {'frame':9s} {'target':14s} {'n_ge':>6s} {'n_gt':>5s} "
        f"{'wscissor%':>10s}  axes_won"
    )
    for r in sorted(rows, key=lambda x: (x["frame"], x["corpus"], x["target"])):
        if not r["slow_dominates"]:
            continue
        pct = f"{r['wscissor_pct_change']:+.2f}" if r["wscissor_pct_change"] is not None else "n/a"
        print(
            f"{r['corpus']:9s} {r['arm']:4s} {r['frame']:9s} {r['target']:14s} "
            f"{r['slow_n_ge']:>3d}/{r['frame_size']:<2d} {r['slow_n_gt']:>5d} {pct:>10s}  "
            f"{','.join(r['axes_won'])}"
        )

    ok = not disagree and worst == 0.0 and worst_wfd == 0.0 and worst_inc == 0.0
    out = dict(
        verdict="PASS" if ok else "CHECK",
        zero_reuse_max_rel_err_candidates=worst,
        zero_reuse_max_rel_err_incumbents=worst_inc,
        zero_reuse_max_rel_err_wfd=worst_wfd,
        n_rows=len(rows),
        n_verdict_disagreements=len(disagree),
        n_confirmed_dominators=len(doms),
        rows=rows,
        note="MODELED/gauge only; tau saturated, Phase-D cancelled; nothing promoted.",
    )
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nverdict={out['verdict']}  wrote {args.out}")


if __name__ == "__main__":
    main()
