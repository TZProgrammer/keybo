"""MODELNORM-1 island memetic search — ONE engine, used for both anchor steps and the blend.

Two jobs, deliberately the same code so an anchor and a blend champion are never products of
different searches:

  ``--objective solo:<MODEL>``   step 2: maximize ONE model alone. Its optimum defines that
                                model's "1" anchor. Run at IDENTICAL budget/seed for all
                                three models, else a model whose optimum is found less
                                completely gets a compressed scale and is silently
                                DOWN-weighted (the exact failure the scheme exists to
                                prevent — trap 1 of the brief).
  ``--objective blend``         step 4: maximize the weighted normalized blend, weights from
                                ``--weights AALTO,COMMUNITY,POOL``.

Engine (arm D/E's, structurally): independent islands, steady-state populations, mixed
permutation operators (swap / 3-cycle / block-relocate / order-crossover), a
first-improvement 2-opt polish on the best offspring, and a multi-start descent stream that
keeps the UNIQUE-eval rate from collapsing once islands converge. Elites migrate each epoch.
Space is pinned at slot 30; the 30 movable characters permute. Unique evals are counted by
blake2b-8 of the layout string (trap 8: never salted ``hash()``).

PER-EPOCH CHECKPOINTING is mandatory (trap 7: a reboot once destroyed 4.5M evals/arm).
``--resume`` restores islands, the best-so-far, the unique-key set AND the per-epoch
best-so-far curve, so the convergence evidence survives a restart too.

MODELLED ONLY: fitted surfaces on the .native frame at a BAKED 90 WPM. Not a claim about
realized typing speed; no layout here is promoted or adopted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modelnorm_eval as MN  # noqa: E402


def _read_json(path):
    """json.load with the handle closed (ruff SIM115)."""
    with open(path) as handle:
        return json.load(handle)


def _write_json(path, payload):
    """json.dump with the handle closed (ruff SIM115)."""
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)


POP = 64
KIDS = 48
_EVAL: dict = {}


# ---------------------------------------------------------------------------
# worker-side objective
# ---------------------------------------------------------------------------
def _init_worker(objective: str, corpus: str | None, anchors_path: str | None,
                 weights: list[float] | None) -> None:
    surf = MN.NativeSurfaces(corpus=corpus)
    _EVAL["surf"] = surf
    _EVAL["objective"] = objective
    if objective.startswith("solo:"):
        model = objective.split(":", 1)[1]
        if model not in MN.MODELS:
            raise SystemExit(f"unknown model {model!r}; expected one of {MN.MODELS}")
        _EVAL["column"] = MN.MODELS.index(model)
    elif objective == "blend":
        anchors = MN.load_anchors(Path(anchors_path))
        mapping = dict(zip(MN.MODELS, weights, strict=True))
        _EVAL["norm"] = MN.BlendNormalizer(anchors, mapping)
    else:
        raise SystemExit(f"unknown objective {objective!r}")


def _fitness(perms31: np.ndarray) -> np.ndarray:
    """(B,31) -> (B,) value to MINIMIZE.

    ``solo`` minimizes that model's raw predicted ms directly (an affine, strictly decreasing
    function of its normalized score, so it finds the same optimum without needing the anchors
    that step 2 is what produces). ``blend`` minimizes the NEGATED weighted normalized blend.
    """
    surf: MN.NativeSurfaces = _EVAL["surf"]
    fits = surf.fit_batch(perms31)
    if "column" in _EVAL:
        return fits[:, _EVAL["column"]].copy()
    return _EVAL["norm"].objective(fits)


# ---------------------------------------------------------------------------
# permutation operators (all act on the 30 movable slots)
# ---------------------------------------------------------------------------
def _swap(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = p.copy()
    i, j = rng.choice(30, 2, replace=False)
    q[i], q[j] = q[j], q[i]
    return q


def _cycle3(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = p.copy()
    i, j, k = rng.choice(30, 3, replace=False)
    q[i], q[j], q[k] = q[j], q[k], q[i]
    return q


def _block_relocate(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = p.copy()
    length = int(rng.integers(2, 6))
    start = int(rng.integers(0, 30 - length + 1))
    shift = int(rng.integers(1, length + 1))
    q[start:start + length] = np.roll(q[start:start + length], shift)
    return q


def _ox(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order crossover on the char->slot assignment, repaired to a permutation."""
    lo, hi = sorted(rng.choice(31, 2, replace=False).tolist())
    child = -np.ones(30, dtype=np.int32)
    taken = set()
    for i in range(lo, min(hi, 30)):
        child[i] = a[i]
        taken.add(int(a[i]))
    donor = iter([int(s) for s in b[:30] if int(s) not in taken])
    for i in range(30):
        if child[i] < 0:
            child[i] = next(donor)
    return child


def _as31(p30: np.ndarray) -> np.ndarray:
    return np.concatenate([p30.astype(np.int32), np.array([30], dtype=np.int32)])


_ALL_PAIRS = np.array([(i, j) for i in range(30) for j in range(i + 1, 30)], dtype=np.int32)


def _neighbours(p30: np.ndarray) -> np.ndarray:
    """All 435 single-swap neighbours as (435, 31)."""
    count = _ALL_PAIRS.shape[0]
    out = np.repeat(p30[None, :], count, axis=0)
    rows = np.arange(count)
    i, j = _ALL_PAIRS[:, 0], _ALL_PAIRS[:, 1]
    out[rows, i], out[rows, j] = p30[j], p30[i]
    return np.concatenate([out, np.full((count, 1), 30, dtype=np.int32)], axis=1)


# ---------------------------------------------------------------------------
# island epoch
# ---------------------------------------------------------------------------
def _run_epoch(task: tuple) -> dict:
    island, seed, pop30, target, polish_sweeps, immigrants, ga_share = task
    rng = np.random.default_rng([seed, island, len(pop30)])
    seen: set[int] = set()
    n_eval = 0

    def score(batch31: np.ndarray) -> np.ndarray:
        nonlocal n_eval
        value = _fitness(batch31)
        for row in batch31:
            seen.add(MN.layout_key(MN.layout_of(row)))
        n_eval += batch31.shape[0]
        return value

    pop = [np.asarray(p, dtype=np.int32) for p in pop30]
    pop.extend(np.asarray(imm, dtype=np.int32) for imm in immigrants)
    fit = list(score(np.stack([_as31(p) for p in pop])))
    best_index = int(np.argmin(fit))
    best = (float(fit[best_index]), pop[best_index].copy())

    restart_ratio = (1.0 - ga_share) / max(ga_share, 1e-9)
    while n_eval < target:
        generation_start = n_eval
        kids = []
        for _ in range(KIDS):
            roll = rng.random()
            if roll < 0.35:
                kids.append(_swap(pop[int(rng.integers(len(pop)))], rng))
            elif roll < 0.55:
                kids.append(_cycle3(pop[int(rng.integers(len(pop)))], rng))
            elif roll < 0.72:
                kids.append(_block_relocate(pop[int(rng.integers(len(pop)))], rng))
            elif roll < 0.95:
                a, b = rng.choice(len(pop), 2, replace=False)
                kids.append(_ox(pop[a], pop[b], rng))
            else:
                kids.append(rng.permutation(30).astype(np.int32))
        kid_fit = score(np.stack([_as31(k) for k in kids]))

        for index in np.argsort(kid_fit)[:4]:
            current, current_fit = kids[int(index)].copy(), float(kid_fit[int(index)])
            for _ in range(polish_sweeps):
                if n_eval >= target:
                    break
                neighbours = _neighbours(current)
                values = score(neighbours)
                pick = int(np.argmin(values))
                if values[pick] < current_fit - 1e-12:
                    current_fit = float(values[pick])
                    current = neighbours[pick, :30].copy()
                else:
                    break
            kids[int(index)] = current
            kid_fit[int(index)] = current_fit

        restarts, restart_fit = [], []
        restart_budget = min(target, n_eval + int((n_eval - generation_start) * restart_ratio))
        while n_eval < restart_budget:
            current = rng.permutation(30).astype(np.int32)
            current_fit = float(score(_as31(current)[None])[0])
            for _ in range(polish_sweeps):
                if n_eval >= restart_budget:
                    break
                neighbours = _neighbours(current)
                values = score(neighbours)
                pick = int(np.argmin(values))
                if values[pick] < current_fit - 1e-12:
                    current_fit = float(values[pick])
                    current = neighbours[pick, :30].copy()
                else:
                    break
            restarts.append(current)
            restart_fit.append(current_fit)

        merged = pop + kids + restarts
        merged_fit = np.concatenate([
            np.asarray(fit, dtype=np.float64),
            np.asarray(kid_fit, dtype=np.float64),
            np.asarray(restart_fit, dtype=np.float64),
        ])
        keep, seen_layout = [], set()
        for index in np.argsort(merged_fit):
            layout = MN.layout_of(_as31(merged[int(index)]))
            if layout in seen_layout:
                continue
            seen_layout.add(layout)
            keep.append(int(index))
            if len(keep) >= POP:
                break
        pop = [merged[i].copy() for i in keep]
        fit = [float(merged_fit[i]) for i in keep]
        if fit[0] < best[0]:
            best = (fit[0], pop[0].copy())

    return {
        "island": island,
        "pop": [p.tolist() for p in pop],
        "fit": fit,
        "best_fit": best[0],
        "best_layout": MN.layout_of(_as31(best[1])),
        "n_eval": n_eval,
        "keys": list(seen),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True,
                        help="solo:AALTO | solo:COMMUNITY | solo:POOL | blend")
    parser.add_argument("--anchors", default=None, help="anchors JSON (required for blend)")
    parser.add_argument("--weights", default="1,1,1",
                        help="preference weights AALTO,COMMUNITY,POOL (blend only)")
    parser.add_argument("--budget", type=int, default=10_000_000, help="target UNIQUE evals")
    parser.add_argument("--islands", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--polish-sweeps", type=int, default=40)
    parser.add_argument("--ga-share", type=float, default=0.6)
    parser.add_argument("--overshoot", type=float, default=1.95)
    parser.add_argument("--processes", type=int, default=40)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    weights = [float(v) for v in args.weights.split(",")]
    if len(weights) != 3:
        raise SystemExit(f"--weights needs 3 comma-separated numbers, got {args.weights!r}")
    if args.objective == "blend" and not args.anchors:
        raise SystemExit("--objective blend requires --anchors")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out.with_suffix(".ckpt.json")
    keys_path = out.with_suffix(".keys.npy")
    start = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - start:8.1f}s] {message}", flush=True)

    surf = MN.NativeSurfaces(corpus=args.corpus)
    identity = surf.identity()
    log(f"objective={args.objective} corpus={identity['corpus']} frame={identity['frame']} "
        f"seed={args.seed} budget={args.budget:,}")

    # Identical initial populations for every objective at a given seed, so a difference
    # between two runs is the OBJECTIVE and not the draw (the anchors depend on this).
    seed_rng = np.random.default_rng(args.seed)
    initial = [[seed_rng.permutation(30).astype(np.int32) for _ in range(POP)]
               for _ in range(args.islands)]
    log(f"seeded {args.islands} islands x {POP} random C30M permutations from seed {args.seed}")

    # Every knob that changes the SCHEDULE or the OBJECTIVE is part of the run's identity, so
    # a resume that differs in any of them is refused loudly rather than silently continuing a
    # different experiment. `--epochs` matters most and is the least obvious: `per_epoch` is
    # `budget * overshoot / (epochs * islands)`, so resuming with a different `--epochs`
    # re-scales every remaining epoch's spend and the run is no longer the one that was
    # checkpointed. (Found while testing resume: a 2-epoch and a 4-epoch run of the same
    # budget are different searches, not the same search stopped early.)
    RUN_IDENTITY = ("objective", "weights", "budget", "islands", "epochs", "seed",
                    "polish_sweeps", "ga_share", "overshoot", "corpus")
    run_identity = {
        "objective": args.objective, "weights": weights, "budget": args.budget,
        "islands": args.islands, "epochs": args.epochs, "seed": args.seed,
        "polish_sweeps": args.polish_sweeps, "ga_share": args.ga_share,
        "overshoot": args.overshoot, "corpus": identity["corpus"],
    }
    start_epoch, all_keys, state, curve = 0, set(), None, []
    if args.resume and checkpoint.exists():
        blob = _read_json(checkpoint)
        stored = blob.get("run_identity")
        if stored is None:
            raise SystemExit(
                f"{checkpoint} predates run-identity stamping; refusing to resume it blind. "
                "Delete it to start fresh."
            )
        differing = {k: (stored.get(k), run_identity[k]) for k in RUN_IDENTITY
                     if stored.get(k) != run_identity[k]}
        if differing:
            raise SystemExit(
                "refusing to resume: the checkpoint is a DIFFERENT run.\n  "
                + "\n  ".join(f"{k}: checkpoint={old!r} requested={new!r}"
                              for k, (old, new) in differing.items())
                + "\nResuming across any of these silently changes the experiment "
                  "(--epochs rescales every epoch's spend). Match the flags, or delete "
                  f"{checkpoint} to start fresh."
            )
        start_epoch = blob["epoch"]
        state = blob["islands"]
        curve = blob.get("curve", [])
        if keys_path.exists():
            all_keys = set(np.load(keys_path).tolist())
        log(f"RESUMED at epoch {start_epoch}, {len(all_keys):,} unique evals")

    per_epoch = max(1, int(args.budget * args.overshoot) // (args.epochs * args.islands))
    log(f"{args.epochs} epochs x {args.islands} islands x {per_epoch:,} calls/island/epoch "
        f"(overshoot {args.overshoot}x)")

    context = mp.get_context("fork")
    best_overall: tuple[float, str | None] = (float("inf"), None)
    epoch = start_epoch - 1
    with context.Pool(processes=min(args.processes, args.islands), initializer=_init_worker,
                      initargs=(args.objective, args.corpus, args.anchors, weights)) as pool:
        if state:
            for island in state:
                if island["best_fit"] < best_overall[0]:
                    best_overall = (island["best_fit"], island["best_layout"])
        for epoch in range(start_epoch, args.epochs):
            immigrants: list[list] = [[] for _ in range(args.islands)]
            if state:
                top = sorted(state, key=lambda s: s["best_fit"])[:3]
                for index in range(args.islands):
                    immigrants[index] = [
                        MN.perm_of(t["best_layout"])[:30].tolist() for t in top
                    ]
            tasks = []
            for index in range(args.islands):
                population = (state[index]["pop"] if state
                              else [p.tolist() for p in initial[index]])
                tasks.append((index, args.seed + 1009 * epoch, population, per_epoch,
                              args.polish_sweeps, immigrants[index], args.ga_share))
            results = pool.map(_run_epoch, tasks)
            state = sorted(results, key=lambda r: r["island"])
            for island in state:
                all_keys.update(island["keys"])
                island["keys"] = []
                if island["best_fit"] < best_overall[0]:
                    best_overall = (island["best_fit"], island["best_layout"])
            calls = sum(island["n_eval"] for island in state)
            curve.append({"epoch": epoch + 1, "unique": len(all_keys), "calls": calls,
                          "best_fit": best_overall[0], "best_layout": best_overall[1],
                          "island_bests": [island["best_fit"] for island in state]})
            log(f"epoch {epoch + 1}/{args.epochs}: unique={len(all_keys):,} "
                f"(calls {calls:,}) best={best_overall[0]:.9f} [{best_overall[1]}]")
            keys_tmp = keys_path.with_suffix(".tmp.npy")
            np.save(keys_tmp, np.fromiter(all_keys, dtype=np.uint64, count=len(all_keys)))
            os.replace(keys_tmp, keys_path)
            tmp = checkpoint.with_suffix(".tmp")
            with open(tmp, "w") as handle:
                json.dump({"run_identity": run_identity, "objective": args.objective,
                           "weights": weights, "epoch": epoch + 1,
                           "n_unique": len(all_keys), "islands": state, "curve": curve,
                           "best_fit": best_overall[0], "best_layout": best_overall[1],
                           "seed": args.seed, "budget": args.budget,
                           "elapsed_s": time.time() - start}, handle)
            os.replace(tmp, checkpoint)
            if len(all_keys) >= args.budget:
                log(f"budget reached at epoch {epoch + 1}")
                break

    archive: dict[str, float] = {}
    for island in state:
        for population, value in zip(island["pop"], island["fit"], strict=False):
            layout = MN.layout_of(_as31(np.asarray(population, dtype=np.int32)))
            archive[layout] = float(value)
    ordered = sorted(archive.items(), key=lambda kv: kv[1])

    champion_layout = ordered[0][0]
    champion_fits = surf.fit_of_layout(champion_layout)
    blob = {
        "what": f"MODELNORM-1 search, objective={args.objective}",
        "objective": args.objective,
        "objective_unit": (
            "predicted ms over the corpus on ONE native surface (lower = faster); this "
            "model's optimum is its '1' anchor"
            if args.objective.startswith("solo:")
            else "NEGATED weighted normalized blend (lower = better; blend itself is "
                 "higher = better, 1 = per-model optimum)"),
        "weights": weights if args.objective == "blend" else None,
        "anchors_file": args.anchors if args.objective == "blend" else None,
        "identity": identity,
        "budget_requested": args.budget,
        "unique_evals": len(all_keys),
        "islands": args.islands,
        "epochs_run": epoch + 1,
        "seed": args.seed,
        "polish_sweeps": args.polish_sweeps,
        "ga_share": args.ga_share,
        "overshoot": args.overshoot,
        "champion": {
            "layout": champion_layout,
            "fitness": ordered[0][1],
            "fits_per_model": {m: float(v) for m, v in
                               zip(MN.MODELS, champion_fits, strict=True)},
        },
        "top50": [{"layout": layout, "fitness": value} for layout, value in ordered[:50]],
        "per_island_best": [{"island": island["island"], "best_fit": island["best_fit"],
                             "best_layout": island["best_layout"]} for island in state],
        "curve": curve,
        "elapsed_s": time.time() - start,
        "modelled_only": (
            "MODELLED ONLY: fitted surfaces on the .native frame at a BAKED 90 WPM; tau "
            "saturated at 1.0 and Phase-D cancelled. Not a claim about realized typing "
            "speed. No layout here is promoted or adopted."
        ),
    }
    _write_json(out, blob)
    log(f"WROTE {out}: champion {champion_layout} fitness {ordered[0][1]:.9f} "
        f"({len(all_keys):,} unique evals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
