"""WSCISSOR-GEN — TARGETED 12-axis dominance hunt, per incumbent.

AN ARCHIVE-ONLY NULL IS NOT A NULL. This campaign established that the hard way: the 10/10
dominators appeared ONLY when a deficit hunt was pointed AT a specific incumbent; the NSGA-II
archive alone reported ``dominator_exists=False`` for the same layout. So the archive scan in
``wscissor_score.py`` is not allowed to be the last word, and this driver supplies the real test.

Engine is ``hunt_on_blend.py`` VERBATIM (feasibility-first simulated annealing + mixed ops on the
spread-normalized dominance deficit; warm-started from the target; per-target stable seeds). Two
changes:

  1. the frame is the **12 axes** (the campaign's 10 + ``wscissor`` + ``nscissor``), so a
     dominator must clear the wide gauge AND the narrow gauge AND everything else at once;
  2. ``--frame`` also allows the 10-axis frame and a **wide-only-added** 11-axis frame, so the
     hunt can attribute a failure to a specific added axis rather than to "the frame got harder".

The 12-axis hunt is the strongest form of the question the task asks: is there a layout that
beats a NAMED incumbent on the wide gauge without giving up anything the board already grades?
A null here is a real null (a targeted hunt found nothing), and is reported with its per-axis
residual shortfall so a reader can see WHICH axis blocked it.

MODELED/gauge only. Model held FIXED. Nothing promoted.
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
import wscissor_eval as WE  # noqa: E402

N = 30
START = time.time()
EPS_REWARD = 1e-4  # tie-breaker only; matters solely inside the feasible cone

#: The three frames. "wide11" isolates the wide axis so a 12-axis null can be attributed.
FRAMES = {
    "ten": CE.AXES,
    "wide11": [*CE.AXES, "wscissor"],
    "twelve": WE.AXES12,
}

_G: dict = {}


def _init(corpus: str, arm: str, ceilings: dict) -> None:
    if "board" not in _G:
        _G["board"] = WE.WScissorBoard(corpus=corpus, arm=arm, ceilings=ceilings, objective="wide")


def oriented(axes: dict, frame: list[str]) -> np.ndarray:
    return np.array([WE.SIGN12[a] * axes[a] for a in frame])


def score_vs_target(axes: dict, target: np.ndarray, scale: np.ndarray, frame: list[str]) -> dict:
    """Per-axis margins against a domination target, scale-balanced by the incumbent-set spread
    so the 1e13-magnitude wfd/oxey axes cannot swamp the 0.03-magnitude ones."""
    cv = oriented(axes, frame)
    margin = (cv - target) / scale
    deficit = float(np.maximum(0.0, -margin).sum())
    reward = float(np.maximum(0.0, margin).sum())
    return {
        "deficit": deficit,
        "reward": reward,
        "n_strict": int(np.sum(cv > target + 1e-9)),
        "n_ge": int(np.sum(cv >= target - 1e-9)),
        "obj": deficit - EPS_REWARD * reward,
        "dominates": deficit <= 1e-9 and int(np.sum(cv > target + 1e-9)) >= 1,
    }


def better(a: dict, b: dict | None) -> bool:
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
        frame_name,
    ) = args
    _init(corpus, arm, ceilings)
    board = _G["board"]
    frame = FRAMES[frame_name]
    rng = np.random.default_rng(seed)
    target = np.array(target)
    scale = np.array(scale)

    def score(layout: str):
        axes = board.axes12(layout, floor_kind)
        return score_vs_target(axes, target, scale, frame), axes

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
        temperature = 0.3
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

    # Residual shortfall in interpretable ORIENTED units, so a null names the blocking axis.
    cv = oriented(best["ax"], frame)
    residual = {a: float(max(0.0, target[i] - cv[i])) for i, a in enumerate(frame)}
    return {
        "target": name,
        "frame": frame_name,
        "best_deficit": best["sc"]["deficit"],
        "best_n_strict_better": best["sc"]["n_strict"],
        "best_n_ge": best["sc"]["n_ge"],
        "frame_size": len(frame),
        "max_n_ge_seen": best_n_ge,
        "best_obj": best["sc"]["obj"],
        "best_layout": best["layout"],
        "dominates_target": bool(best["sc"]["dominates"]),
        "best_axes": best["ax"],
        "residual_shortfall": residual,
        "corpus": corpus,
        "arm": arm,
    }


def build_targets(board: WE.WScissorBoard, floor_kind: str, frame: list[str], extra_warm):
    """{name: (oriented target, warm starts)} for the 5 incumbents + the ideal point.

    Warm-starting from the target itself is load-bearing: the campaign's iWeb run found the
    archive-1846 dominator ONLY when pointed at it from its own neighbourhood.
    """
    inc = {name: board.axes12(string, floor_kind) for name, string in CE.INCUMBENTS.items()}
    for name, string in CE.INCUMBENTS.items():
        inc[name]["layout"] = string
    names = list(inc)
    stacked = np.array([oriented(inc[n], frame) for n in names])
    targets = {n: (stacked[i], [inc[n]["layout"], *extra_warm]) for i, n in enumerate(names)}
    targets["IDEAL(all5)"] = (
        stacked.max(axis=0),
        [inc[n]["layout"] for n in names] + list(extra_warm),
    )
    scale = np.maximum(stacked.max(axis=0) - stacked.min(axis=0), 1e-12)
    return targets, inc, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CE.CORPUS_DIRS), default="iweb")
    ap.add_argument("--arm", choices=["A", "B"], default="A")
    ap.add_argument("--frame", choices=sorted(FRAMES), default="twelve")
    ap.add_argument("--floor", choices=["norm", "raw"], default="norm")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--procs", type=int, default=46)
    ap.add_argument(
        "--warm-from-arm",
        default=None,
        help="seed the hunt with the best wide-gauge layouts from a generator arm json",
    )
    ap.add_argument("--warm-count", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ceilings = CE.SixSurface(args.corpus).ceiling_map
    board = WE.WScissorBoard(corpus=args.corpus, arm=args.arm, ceilings=ceilings, objective="wide")
    frame = FRAMES[args.frame]

    # Warm starts from the searched field: giving the hunt the generator's best wide layouts is
    # what makes this a test of "can a WIDE-graded search reach dominance", not just of the SA.
    #
    # Reads `front0` when present (a COMPLETED pass) and otherwise falls back to `archive` (a
    # per-epoch CHECKPOINT, which carries no front0). Without the fallback a checkpoint silently
    # yields zero warm starts and the hunt quietly degrades to a cold SA run — it still reports a
    # result, so the degradation is invisible. `warm_from_searched_field` in the output records
    # what was actually used, and the count is logged.
    extra_warm: list[str] = []
    warm_source = "none"
    if args.warm_from_arm:
        blob = json.loads(Path(args.warm_from_arm).read_text())
        rows = blob.get("front0")
        if rows:
            warm_source = f"front0 of {Path(args.warm_from_arm).name}"
            ranked = sorted(rows, key=lambda r: r.get("wscissor_P", float("inf")))
            extra_warm = [r["layout"] for r in ranked[: args.warm_count]]
        else:
            warm_source = f"archive of {Path(args.warm_from_arm).name} (checkpoint, no front0)"
            # Rank the checkpoint's archive on the WIDE severity share, recomputed here.
            ranked_arch = sorted(
                (row["layout"] for row in blob.get("archive", [])),
                key=lambda layout: board.severity_axes(layout)["wscissor_P"],
            )
            extra_warm = ranked_arch[: args.warm_count]
        if not extra_warm:
            raise SystemExit(
                f"--warm-from-arm {args.warm_from_arm} yielded NO warm starts; refusing to run a "
                "silently-cold hunt (a cold hunt still reports a result, so this must be loud)"
            )

    targets, inc, scale = build_targets(board, args.floor, frame, extra_warm)
    out_path = Path(
        args.out or HERE / "runs" / f"whunt-{args.corpus}-arm{args.arm}-{args.frame}.json"
    )

    jobs = []
    for name, (vec, warm) in targets.items():
        for s in range(args.seeds):
            jobs.append(
                (
                    name,
                    vec,
                    warm,
                    # Stable per-target seed: a plain str hash is salted per process, which
                    # silently varied a "not found" result between runs of the same command.
                    770001
                    + (int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") % 1000)
                    + s * 137,
                    args.iters,
                    args.restarts,
                    args.floor,
                    scale.tolist(),
                    args.corpus,
                    args.arm,
                    ceilings,
                    args.frame,
                )
            )
    print(
        f"[{time.time() - START:.1f}s] hunt corpus={args.corpus} arm={args.arm} "
        f"frame={args.frame}({len(frame)} axes) warm_extra={len(extra_warm)} "
        f"warm_source={warm_source}: "
        f"{len(jobs)} jobs, {args.iters} iters x {args.restarts} restarts",
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
        "corpus_label": CE.CORPUS_LABELS[args.corpus],
        "arm": args.arm,
        "frame": args.frame,
        "frame_axes": frame,
        "config": vars(args),
        "warm_from_searched_field": extra_warm,
        "warm_start_source": warm_source,
        "ceilings": ceilings,
        "universal_dominator_found": bool(
            by_target.get("IDEAL(all5)", {}).get("dominates_target", False)
        ),
        "dominated_targets": dominated,
        "per_target_best": by_target,
        "incumbent_axes": inc,
        "deficit_scale": {a: float(scale[i]) for i, a in enumerate(frame)},
        "null_semantics": (
            "This is a TARGETED per-incumbent hunt, so an empty 'dominated_targets' IS a real "
            "null (not an archive-only null). residual_shortfall names the blocking axis."
        ),
        "note": "MODELED/gauge only; model held FIXED; nothing promoted.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"[{time.time() - START:.1f}s] done; dominated={dominated}", flush=True)
    for name, r in by_target.items():
        shortfall = sorted(r["residual_shortfall"].items(), key=lambda kv: -kv[1])[:3]
        pretty = ", ".join(f"{a}:{v:.4g}" for a, v in shortfall if v > 0) or "(none)"
        print(
            f"  {name:14s} deficit={r['best_deficit']:.5f} "
            f"n_ge={r['best_n_ge']}/{r['frame_size']} "
            f"n_strict={r['best_n_strict_better']} blocked_by=[{pretty}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
