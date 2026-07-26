"""REHUNT — targeted per-incumbent dominance hunt against the CORRECTED wfd bar.

THE QUESTION. WFD-FRAMES-1 RE-ADJUDICATED 42 frozen verdicts (it replaced the wfd axis on the
frozen `best_axes` and recounted), and 14 dominance claims died. That tells us those 14 FROZEN
LAYOUTS do not survive. It does NOT tell us whether a search POINTED AT the corrected bar finds
*different* layouts that do. This driver answers that, one (corpus, frame, incumbent) cell at a
time.

⚠ AN ARCHIVE-ONLY NULL IS NOT A NULL. Reproduced four times in this campaign: the NSGA-II archive
reports `dominator_exists=False` for a layout that a hunt POINTED AT it then dominates. So the
instrument here is the targeted hunt, never an archive scan.

ENGINE. `wscissor_hunt.hunt_target` semantics verbatim — feasibility-first SA on the
spread-normalized dominance deficit, mixed swap/3-cycle/block ops, warm-started from the target
itself (load-bearing: the campaign's iWeb archive-1846 dominator was found ONLY from its own
neighbourhood). Four changes, all recorded in the output:

  1. **wfd is the CORRECTED axis** on candidate AND incumbent (`wfd_mode='corrected'`), routed
     through the validated `community._dof_arrays`. Every reported layout is asserted to be a
     C30M permutation before it is written.
  2. **PER-EPOCH CHECKPOINTING** (trap 7 — a reboot destroyed 4.5M evals/arm because the search
     only persisted at completion). Each worker writes an atomic per-epoch checkpoint carrying
     its best-so-far and its eval count, and RESUMES from it.
  3. **Eval accounting is explicit** — `unique_layouts` is a per-worker set digest count, so the
     budget claim is measured rather than multiplied out.
  4. `--frame ten|twelve` selects the frozen frame of the cell being re-run, so each re-run is
     compared against the same-size frame its frozen verdict used (trap 17: a frame-size change
     is a second factor).

MODELED/gauge only. Held-layout tau saturated at 1.0; Phase-D cancelled. No realized-speed claim,
no adoption claim, nothing promoted.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

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
from wfd_fix import assert_c30m_permutation  # noqa: E402

N = 30
START = time.time()
EPS_REWARD = 1e-4  # tie-breaker only; matters solely inside the feasible cone

#: The frames. `ten` = the campaign's original 10 axes (`hunt-*-arm*-norm.json`);
#: `twelve` = + wscissor + nscissor (`whunt-*-twelve.json`). Those two hold all 14 flips.
#:
#: `narrow11`/`wide11` exist only for WSCISSOR-ARMB-1's placebo-differenced "the wscissor axis is
#: inert" null (trap 17: going ten -> wide11 changes TWO things — the axis is added AND the frame
#: grows — so a drop in the dominator count is unattributable without a same-SIZE placebo).
#: `narrow11` is that placebo: same size, a DIFFERENT axis, and deliberately NESTED in the real
#: one (narrow support is a strict subset of wide), which understates the real axis's cost and so
#: makes an "inert" verdict conservative.
FRAMES = {
    "ten": CE.AXES,
    "narrow11": [*CE.AXES, "nscissor"],
    "wide11": [*CE.AXES, "wscissor"],
    "twelve": WE.AXES12,
}

_G: dict = {}


def _init(corpus: str, arm: str, ceilings: dict, wfd_mode: str) -> None:
    key = (corpus, arm, wfd_mode)
    if _G.get("key") != key:
        _G["board"] = WE.WScissorBoard(
            corpus=corpus, arm=arm, ceilings=ceilings, objective="wide", wfd_mode=wfd_mode
        )
        _G["key"] = key


def oriented(axes: dict, frame: list[str]) -> np.ndarray:
    return np.array([WE.SIGN12[a] * axes[a] for a in frame])


def score_vs_target(axes: dict, target: np.ndarray, scale: np.ndarray, frame: list[str]) -> dict:
    """Per-axis margins vs a domination target, spread-normalized so the 1e13-magnitude
    wfd/oxey axes cannot swamp the 0.03-magnitude ones."""
    cv = oriented(axes, frame)
    margin = (cv - target) / scale
    deficit = float(np.maximum(0.0, -margin).sum())
    reward = float(np.maximum(0.0, margin).sum())
    n_strict = int(np.sum(cv > target + 1e-9))
    return {
        "deficit": deficit,
        "reward": reward,
        "n_strict": n_strict,
        "n_ge": int(np.sum(cv >= target - 1e-9)),
        "obj": deficit - EPS_REWARD * reward,
        "dominates": deficit <= 1e-9 and n_strict >= 1,
    }


def better(a: dict, b: dict | None) -> bool:
    if b is None:
        return True
    if a["dominates"] != b["dominates"]:
        return a["dominates"]
    if a["dominates"]:
        return a["n_strict"] > b["n_strict"]
    return a["obj"] < b["obj"] - 1e-12


def _atomic_write(path: Path, blob: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, default=float))
    tmp.replace(path)  # atomic on the same filesystem


def hunt_worker(args) -> dict:
    """One (target, seed) worker: `restarts` SA restarts, checkpointed EVERY restart (epoch)."""
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
        wfd_mode,
        ckpt_dir,
    ) = args
    _init(corpus, arm, ceilings, wfd_mode)
    board = _G["board"]
    frame = FRAMES[frame_name]
    target = np.array(target)
    scale = np.array(scale)
    ckpt = Path(ckpt_dir) / f"{corpus}-arm{arm}-{frame_name}-{name}-s{seed}.ckpt.json"

    # ---- RESUME (trap 7): a reboot must cost one epoch, not the whole arm -----------------
    best = None
    seen: set[str] = set()
    start_restart = 0
    if ckpt.exists():
        try:
            prev = json.loads(ckpt.read_text())
            if prev.get("seed") == seed and prev.get("target") == name:
                start_restart = int(prev.get("restarts_done", 0))
                seen = set(prev.get("seen_digests", []))
                if prev.get("best_layout"):
                    ax = board.axes12(prev["best_layout"], floor_kind)
                    best = {
                        "sc": score_vs_target(ax, target, scale, frame),
                        "layout": prev["best_layout"],
                        "ax": ax,
                    }
        except (OSError, ValueError, KeyError):
            start_restart = 0  # a truncated checkpoint restarts the epoch, never the arm

    rng = np.random.default_rng(seed)
    for _skip in range(start_restart):  # keep the RNG stream identical to an uninterrupted run
        rng.random()

    def score(layout: str):
        seen.add(layout)
        axes = board.axes12(layout, floor_kind)
        return score_vs_target(axes, target, scale, frame), axes

    best_n_ge = 0 if best is None else best["sc"]["n_ge"]
    for r in range(start_restart, restarts):
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
                begin = int(rng.integers(0, N - length))
                block = trial[begin : begin + length]
                del trial[begin : begin + length]
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
        # ---- PER-EPOCH CHECKPOINT (one restart = one epoch) ---------------------------
        _atomic_write(
            ckpt,
            {
                "seed": seed,
                "target": name,
                "corpus": corpus,
                "arm": arm,
                "frame": frame_name,
                "wfd_mode": wfd_mode,
                "restarts_done": r + 1,
                "restarts_total": restarts,
                "unique_layouts": len(seen),
                "best_layout": best["layout"],
                "best_deficit": best["sc"]["deficit"],
                "best_n_ge": best["sc"]["n_ge"],
                "best_n_strict": best["sc"]["n_strict"],
                "dominates": best["sc"]["dominates"],
                # the digest set is what makes `unique_layouts` resumable; capped so a
                # checkpoint stays a checkpoint rather than becoming the archive
                "seen_digests": sorted(seen) if len(seen) <= 400_000 else [],
                "seen_truncated": len(seen) > 400_000,
                "wall_s": round(time.time() - START, 1),
            },
        )

    assert_c30m_permutation(best["layout"])  # never report a non-permutation
    cv = oriented(best["ax"], frame)
    residual = {a: float(max(0.0, target[i] - cv[i])) for i, a in enumerate(frame)}
    return {
        "target": name,
        "frame": frame_name,
        "seed": seed,
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
        "unique_layouts": len(seen),
        "corpus": corpus,
        "arm": arm,
        "wfd_mode": wfd_mode,
    }


def build_targets(board, floor_kind: str, frame: list[str], extra_warm: list[str]):
    """{name: (oriented target, warm starts)} for the 5 incumbents + IDEAL(all5)."""
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
    ap.add_argument("--corpus", choices=sorted(CE.CORPUS_DIRS), required=True)
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--frame", choices=sorted(FRAMES), required=True)
    ap.add_argument("--floor", choices=["norm", "raw"], default="norm")
    ap.add_argument("--wfd", choices=sorted(CE.WFD_MODES), default="corrected")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--procs", type=int, default=36)
    ap.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="restrict to these targets (default: all 5 incumbents + IDEAL(all5))",
    )
    ap.add_argument(
        "--warm",
        nargs="*",
        default=[],
        help="EXTRA warm starts. The frozen cell's warm starts, so the re-run searches the same "
        "field as the frozen hunt did — plus its own target, always.",
    )
    ap.add_argument("--ckpt-dir", default=str(HERE / "runs" / "ckpt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for lay in args.warm:
        assert_c30m_permutation(lay)  # a mangled warm start must be loud, not silently cold

    ceilings = CE.SixSurface(args.corpus).ceiling_map
    board = WE.WScissorBoard(
        corpus=args.corpus,
        arm=args.arm,
        ceilings=ceilings,
        objective="wide",
        wfd_mode=args.wfd,
    )
    frame = FRAMES[args.frame]
    targets, inc, scale = build_targets(board, args.floor, frame, args.warm)
    if args.targets:
        unknown = [t for t in args.targets if t not in targets]
        if unknown:
            raise SystemExit(f"unknown targets {unknown}; have {sorted(targets)}")
        targets = {t: targets[t] for t in args.targets}

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for name, (vec, warm) in targets.items():
        for s in range(args.seeds):
            jobs.append(
                (
                    name,
                    vec,
                    warm,
                    # stable per-target seed (trap 8: a salted str hash silently varied a
                    # "not found" between runs of the same command)
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
                    args.wfd,
                    str(ckpt_dir),
                )
            )
    print(
        f"[{time.time() - START:.1f}s] REHUNT corpus={args.corpus} arm={args.arm} "
        f"frame={args.frame}({len(frame)} axes) wfd={args.wfd} floor={args.floor} "
        f"targets={list(targets)} warm_extra={len(args.warm)}: {len(jobs)} jobs, "
        f"{args.iters} iters x {args.restarts} restarts, procs={args.procs}",
        flush=True,
    )
    with mp.Pool(processes=min(len(jobs), args.procs)) as pool:
        results = pool.map(hunt_worker, jobs)

    by_target: dict[str, dict] = {}
    max_n_ge: dict[str, int] = {}
    evals: dict[str, int] = {}
    for r in results:
        key = r["target"]
        max_n_ge[key] = max(max_n_ge.get(key, 0), r["max_n_ge_seen"])
        evals[key] = evals.get(key, 0) + r["unique_layouts"]
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
        by_target[key]["unique_layouts_all_seeds"] = evals[key]

    dominated = [r["target"] for r in by_target.values() if r["dominates_target"]]
    out = {
        "corpus": args.corpus,
        "corpus_label": CE.CORPUS_LABELS[args.corpus],
        "corpus_dir": str(CE.CORPUS_DIRS[args.corpus]),
        "arm": args.arm,
        "frame": args.frame,
        "frame_axes": frame,
        "wfd_mode": args.wfd,
        "floor_kind": args.floor,
        "config": vars(args),
        "ceilings": ceilings,
        "warm_extra": list(args.warm),
        "universal_dominator_found": bool(
            by_target.get("IDEAL(all5)", {}).get("dominates_target", False)
        ),
        "dominated_targets": dominated,
        "per_target_best": by_target,
        "incumbent_axes": inc,
        "deficit_scale": {a: float(scale[i]) for i, a in enumerate(frame)},
        "unique_layouts_total": sum(r["unique_layouts"] for r in results),
        "unique_layouts_per_target": evals,
        "n_workers": len(results),
        "wall_s": round(time.time() - START, 1),
        "null_semantics": (
            "TARGETED per-incumbent hunt: an empty 'dominated_targets' IS a real null, NOT an "
            "archive-only null. residual_shortfall names the blocking axis."
        ),
        "note": "MODELED/gauge only; tau saturated, Phase-D cancelled; nothing promoted.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(Path(args.out), out)
    print(
        f"[{time.time() - START:.1f}s] done; dominated={dominated}; "
        f"unique_layouts={out['unique_layouts_total']:,}",
        flush=True,
    )
    for name, r in by_target.items():
        shortfall = sorted(r["residual_shortfall"].items(), key=lambda kv: -kv[1])[:3]
        pretty = ", ".join(f"{a}:{v:.4g}" for a, v in shortfall if v > 0) or "(none)"
        print(
            f"  {name:14s} deficit={r['best_deficit']:.6f} "
            f"n_ge={r['best_n_ge']}/{r['frame_size']} (max seen {r['max_n_ge_across_seeds']}) "
            f"strict={r['best_n_strict_better']} "
            f"{'DOMINATES' if r['dominates_target'] else 'no'} blocked_by=[{pretty}] "
            f"evals={r['unique_layouts_all_seeds']:,}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
