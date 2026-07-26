"""GEN-ON-BLEND — targeted 10-axis dominance hunt, re-pointed at an arbitrary corpus.

The island NSGA-II optimizes only 6 of the 10 board axes (it does not search wfd /
genkey / oxey1 / oxey2), so it cannot be relied on to land a FULL 10-axis dominator.
This engine attacks dominance directly: for EACH target, a feasibility-first
simulated-annealing + mixed-op search minimizes the spread-normalized 10-axis
dominance deficit

    margin_i = (cand_oriented_i - target_i) / scale_i
    deficit  = sum_i max(0, -margin_i)      (shortfall BELOW target -> feasibility)
    reward   = sum_i max(0,  margin_i)      (strict-win HEADROOM above target)
    minimize  deficit - EPS_REWARD * reward

`deficit == 0 and n_strict >= 1` IS dominance. EPS_REWARD is tiny so the walk never
trades a feasible point for an infeasible higher-reward one (the failure mode of a
large-penalty scalarization, which missed known dominators); inside the feasible cone the
-EPS*reward term gives a gradient toward a strict win.

Targets: each of the 5 incumbents individually (warm-started from that incumbent — the
iWeb run found the archive-1846 dominator only when pointed AT it), plus the IDEAL POINT
(per-axis best over all five; driving its deficit to 0 with a strict win would be a
layout that dominates ALL FIVE).

The incumbents are re-scored on the SAME arm/corpus board as the candidates, because a
dominance test is only meaningful when both sides are scored the same way.

MODELED/gauge only. Held-layout tau saturated; Phase-D cancelled. Model held FIXED.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus_eval as CE  # noqa: E402

N = 30
AXES = CE.AXES
START = time.time()
EPS_REWARD = 1e-4  # tie-breaker only; matters solely inside the feasible cone

_G: dict = {}


def _init(corpus: str, arm: str, ceilings: dict) -> None:
    if "board" not in _G:
        _G["board"] = CE.ArmBoard(corpus=corpus, arm=arm, ceilings=ceilings)


def score_vs_target(axes: dict, target: np.ndarray, scale: np.ndarray) -> dict:
    """Per-axis margins against a domination target, scale-balanced by the incumbent-set
    spread so the 1e13-magnitude wfd/oxey axes cannot swamp the 0.03-magnitude ones."""
    cv = CE.oriented(axes)
    margin = (cv - target) / scale
    deficit = float(np.maximum(0.0, -margin).sum())
    reward = float(np.maximum(0.0, margin).sum())
    n_strict = int(np.sum(cv > target + 1e-9))
    n_ge = int(np.sum(cv >= target - 1e-9))
    return {
        "deficit": deficit,
        "reward": reward,
        "n_strict": n_strict,
        "n_ge": n_ge,
        "obj": deficit - EPS_REWARD * reward,
        "dominates": deficit <= 1e-9 and n_strict >= 1,
    }


def better(a: dict, b: dict | None) -> bool:
    """Is score `a` a better dominator prospect than `b`? A true dominator always wins;
    among dominators prefer more strict wins; otherwise climb the (minimized) obj."""
    if b is None:
        return True
    if a["dominates"] != b["dominates"]:
        return a["dominates"]
    if a["dominates"]:
        return a["n_strict"] > b["n_strict"]
    return a["obj"] < b["obj"] - 1e-12


def hunt_target(args) -> dict:
    (
        name,
        target,
        warm,
        seed,
        iters,
        restarts,
        floor_kind,
        scale,
        corpus,
        arm,
        ceilings,
    ) = args
    _init(corpus, arm, ceilings)
    board = _G["board"]
    rng = np.random.default_rng(seed)
    target = np.array(target)
    scale = np.array(scale)

    def score(layout: str):
        axes = board.axes(layout, floor_kind)
        return score_vs_target(axes, target, scale), axes

    best = None
    best_n_ge = 0
    for r in range(restarts):
        if warm and r < len(warm):
            cur = warm[r]
        elif warm and r % 3 == 0:
            cur = warm[r % len(warm)]
        else:
            cur = "".join(rng.permutation(list(CE.C30M)))
        cur_sc, cur_ax = score(cur)
        best_n_ge = max(best_n_ge, cur_sc["n_ge"])
        if better(cur_sc, None if best is None else best["sc"]):
            best = {"sc": cur_sc, "layout": cur, "ax": cur_ax}
        cur_obj = cur_sc["obj"]
        chars = list(cur)
        temperature = 0.3  # obj is O(1) after spread-normalization; reheat per restart
        for _it in range(iters):
            move = rng.random()
            trial = chars.copy()
            if move < 0.7:
                i, j = rng.integers(0, N, 2)
                trial[i], trial[j] = trial[j], trial[i]
            elif move < 0.9:
                i, j, k = rng.choice(N, 3, replace=False)
                trial[i], trial[j], trial[k] = trial[k], trial[i], trial[j]
            else:
                length = int(rng.integers(2, 5))
                start = int(rng.integers(0, N - length))
                block = trial[start : start + length]
                del trial[start : start + length]
                ins = int(rng.integers(0, len(trial) + 1))
                trial[ins:ins] = block
            layout = "".join(trial)
            sc, ax = score(layout)
            if sc["obj"] < cur_obj or rng.random() < np.exp(
                -(sc["obj"] - cur_obj) / max(temperature, 1e-9)
            ):
                chars = trial
                cur_obj = sc["obj"]
                best_n_ge = max(best_n_ge, sc["n_ge"])
                if better(sc, best["sc"]):
                    best = {"sc": sc, "layout": layout, "ax": ax}
            temperature *= 0.9997

    # residual per-axis shortfall in interpretable ORIENTED units (not scaled), so
    # "which axis it uniquely cannot clear" is readable — this is the depth-of-negative
    # evidence the prereg promises when the answer is a null.
    cv = CE.oriented(best["ax"])
    residual = {a: float(max(0.0, target[i] - cv[i])) for i, a in enumerate(AXES)}
    return {
        "target": name,
        "best_deficit": best["sc"]["deficit"],
        "best_n_strict_better": best["sc"]["n_strict"],
        "best_n_ge": best["sc"]["n_ge"],
        "max_n_ge_seen": best_n_ge,
        "best_obj": best["sc"]["obj"],
        "best_layout": best["layout"],
        "dominates_target": bool(best["sc"]["dominates"]),
        "best_axes": best["ax"],
        "residual_shortfall": residual,
        "floor_kind": floor_kind,
        "corpus": corpus,
        "arm": arm,
    }


def build_targets(board: CE.ArmBoard, floor_kind: str):
    """{name: (oriented target vector, warm-start layouts)} for the 5 incumbents + ideal."""
    inc = board.incumbent_axes(floor_kind)
    names = list(inc)
    stacked = np.array([CE.oriented(inc[n]) for n in names])
    targets = {n: (stacked[i], [inc[n]["layout"]]) for i, n in enumerate(names)}
    targets["IDEAL(all5)"] = (
        stacked.max(axis=0),
        [inc[n]["layout"] for n in names],
    )
    # per-axis deficit SCALE = incumbent-set spread, floored away from zero, so the
    # deficit is comparable across 14 orders of axis magnitude.
    scale = np.maximum(stacked.max(axis=0) - stacked.min(axis=0), 1e-12)
    return targets, inc, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CE.CORPUS_DIRS), default="blend")
    ap.add_argument("--arm", choices=["A", "B"], default="B")
    ap.add_argument("--floor", choices=["norm", "raw"], default="norm")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=6, help="parallel seeds per target")
    ap.add_argument("--procs", type=int, default=46)
    ap.add_argument("--extra-warm", nargs="*", default=[], help="extra warm-start layouts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    six = CE.SixSurface(args.corpus)
    ceilings = six.ceiling_map
    board = CE.ArmBoard(corpus=args.corpus, arm=args.arm, ceilings=ceilings)
    targets, inc, scale = build_targets(board, args.floor)
    out_path = Path(
        args.out or HERE / "runs" / f"hunt-{args.corpus}-arm{args.arm}-{args.floor}.json"
    )

    jobs = []
    for name, (vec, warm) in targets.items():
        warm_all = list(warm) + list(args.extra_warm)
        for s in range(args.seeds):
            jobs.append(
                (
                    name,
                    vec,
                    warm_all,
                    # Per-target seed offset. Was `abs(hash(name)) % 1000`, but str hashing is
                    # salted per process (PYTHONHASHSEED), so the hunt's seeds — and therefore
                    # its results — silently varied between runs of the SAME command. A stable
                    # digest of the target name keeps the per-target spread while making a
                    # re-run reproducible, which a hunt that reports "not found" needs.
                    770001
                    + (
                        int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") % 1000
                    )
                    + s * 137,
                    args.iters,
                    args.restarts,
                    args.floor,
                    scale.tolist(),
                    args.corpus,
                    args.arm,
                    ceilings,
                )
            )
    print(
        f"[{time.time() - START:.1f}s] hunt corpus={args.corpus} arm={args.arm} "
        f"floor={args.floor}: {len(jobs)} jobs ({len(targets)} targets x {args.seeds} "
        f"seeds), {args.iters} iters x {args.restarts} restarts",
        flush=True,
    )
    with mp.Pool(processes=min(len(jobs), args.procs)) as pool:
        results = pool.map(hunt_target, jobs)

    by_target: dict[str, dict] = {}
    max_n_ge: dict[str, int] = {}
    for r in results:
        key = r["target"]
        max_n_ge[key] = max(max_n_ge.get(key, 0), r["max_n_ge_seen"])
        cur = by_target.get(key)
        if cur is None or better(
            {
                "dominates": r["dominates_target"],
                "n_strict": r["best_n_strict_better"],
                "obj": r["best_obj"],
            },
            {
                "dominates": cur["dominates_target"],
                "n_strict": cur["best_n_strict_better"],
                "obj": cur["best_obj"],
            },
        ):
            by_target[key] = r
    for key in by_target:
        by_target[key]["max_n_ge_across_seeds"] = max_n_ge[key]

    dominated = [r["target"] for r in by_target.values() if r["dominates_target"]]
    out = {
        "corpus": args.corpus,
        "arm": args.arm,
        "floor_kind": args.floor,
        "config": vars(args),
        "ceilings": ceilings,
        "universal_dominator_found": bool(
            by_target.get("IDEAL(all5)", {}).get("dominates_target", False)
        ),
        "dominated_targets": dominated,
        "per_target_best": by_target,
        "incumbent_axes": inc,
        "deficit_scale": {a: float(scale[i]) for i, a in enumerate(AXES)},
        "note": "MODELED/gauge only; tau saturated, Phase-D cancelled; model held FIXED.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"[{time.time() - START:.1f}s] done; dominated={dominated}", flush=True)
    for name, r in by_target.items():
        shortfall = sorted(r["residual_shortfall"].items(), key=lambda kv: -kv[1])[:3]
        pretty = ", ".join(f"{a}:{v:.4g}" for a, v in shortfall if v > 0) or "(none)"
        print(
            f"  {name:14s} deficit={r['best_deficit']:.5f} "
            f"n_ge={r['best_n_ge']}/10 (max seen {r['max_n_ge_across_seeds']}) "
            f"strict={r['best_n_strict_better']} "
            f"{'DOMINATES' if r['dominates_target'] else 'no'} shortfall[{pretty}]",
            flush=True,
        )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
