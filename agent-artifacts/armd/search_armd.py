"""OPTEVIDENCE island memetic search — one engine, three arms, identical budget and seeds.

Arms (the objective is ALWAYS minimized):
  A  evidence   minimize the evidence score from the fitted SHAP weights + loss curves
  B  baseline   minimize predicted ms/char on the served K31 surface at 90 WPM
  C  constrained arm A subject to HARD non-regression bounds on the five wrong-signed gauges
                (a candidate may not exceed the incumbent band's max on any of them)
  D  domain     arm A with each loss curve's `valid_domain` enforced as a HARD CONSTRAINT: the
                price is evaluated at the nearest domain edge (`SEARCH_DOMAIN_POLICY` = CLAMP),
                so a gauge pushed past its fitted band pays the edge price and NOTHING more.
                This is the ONLY difference from arm A — same weights JSON, same island seeds,
                same initial populations, same budget, same corpus, same operators. Arm A
                manufactured 96.5% of its apparent win by leaving those bands, so arm A never
                tested the weights; it tested an unbounded objective. Arm D tests the weights.

The three arms share island seeds and initial populations, so a difference between arms is
the OBJECTIVE, not the draw. Space is pinned: space stays on slot 30, the 30 movable
characters permute.

Engine: independent islands, each a steady-state population with mixed permutation
operators (swap / 3-cycle / block-relocation / order-crossover) plus a first-improvement
2-opt polish on offspring. Elites migrate between islands every epoch. Every layout ever
scored is counted once (blake2b-8 of the layout string), so the reported budget is UNIQUE
evaluations, not calls.

PER-EPOCH CHECKPOINTING is mandatory here — a host reboot destroyed 4.5M evals/arm in an
earlier round (trap 7). Each epoch writes the full island state atomically; `--resume` picks
it up.
"""

from __future__ import annotations

import argparse
import hashlib
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

sys.path.append(str(Path(__file__).resolve().parent))
# Arm D's own drivers live beside this file; `evobj` is the FROZEN arm-A evaluator, taken from
# OPTEVIDENCE-1's committed copy so the gauge computation is provably the same code.
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
import armd_obj as AD  # noqa: E402
import evobj as EV  # noqa: E402

ARM_JSON = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
STATE = Path("/local/home/zegertho/agent/state/armd/artifacts")
#: Arm C's bounds come from OPTEVIDENCE-1's frozen reference, not a re-derivation.
OPTEV_STATE = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
WRONG_SIGNED = ("scissor", "sfb", "sfb-dist", "lsb-dist", "sfs")

_EVAL: dict = {}


def _key(layout: str) -> int:
    return int.from_bytes(hashlib.blake2b(layout.encode(), digest_size=8).digest(), "little")


# --------------------------------------------------------------------------------------
# worker-side objective
# --------------------------------------------------------------------------------------
def _init_worker(arm: str, corpus: str | None, bounds: dict | None) -> None:
    fe = EV.FastEval(corpus=corpus, weights_json=ARM_JSON, with_surface=True)
    if arm == "domain":
        # ARM D. The clamp has to live HERE, on the vectorized search path, because
        # `evobj.Curve.price` is a hand-rolled reimplementation that never calls
        # `LossCurve.price` — so adding a `policy` to `LossCurve` (which the parent did) does
        # NOT clamp the search. `ClampedEval` wraps this same `FastEval`, so every gauge value,
        # kernel and denominator is LITERALLY arm A's; only `evidence_score` differs.
        # `verify_policy.py` gate D pins `ClampedCurve` against
        # `LossCurve.price(..., policy=CLAMP)` and asserts in-domain bit-identity with
        # `evobj.Curve`, so arm D perturbs no supported level of arm A's objective.
        from keybo.analysis.evidence_scorer import SEARCH_DOMAIN_POLICY

        _EVAL["fe"] = AD.ClampedEval(fe, policy=SEARCH_DOMAIN_POLICY)
    else:
        _EVAL["fe"] = fe
    _EVAL["arm"] = arm
    _EVAL["bounds"] = bounds or {}


def _objective(perms: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(B,31) perms -> (fitness to MINIMIZE, gauge dict). Constraint violation is a
    quadratic-in-excess penalty added to the arm-A objective, scaled so that any violation
    dominates the whole feasible score range (which spans ~35 units)."""
    fe = _EVAL["fe"]
    arm = _EVAL["arm"]
    g = fe.gauges(perms)
    if arm == "baseline":
        return g["_ms_per_char"].copy(), g
    # For arm D this is the CLAMPED total; for arms A/C the extrapolating one. Same call, and
    # the dispatch happened once in `_init_worker`, so there is no per-eval branch to get wrong.
    fit = fe.evidence_score(g)
    if arm == "constrained":
        excess = np.zeros_like(fit)
        for metric, cap in _EVAL["bounds"].items():
            excess = excess + np.maximum(g[metric] - cap, 0.0) / max(abs(cap), 1e-9)
        fit = fit + 1000.0 * excess + 1e6 * (excess > 0)
    return fit, g


def _feasible(g: dict[str, np.ndarray]) -> np.ndarray:
    bounds = _EVAL["bounds"]
    if not bounds:
        return np.ones(len(next(iter(g.values()))), dtype=bool)
    ok = np.ones(len(g["sfb"]), dtype=bool)
    for metric, cap in bounds.items():
        ok &= g[metric] <= cap + 1e-12
    return ok


# --------------------------------------------------------------------------------------
# permutation operators (all act on the 30 movable slots)
# --------------------------------------------------------------------------------------
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
    """Move a contiguous block of CHARS to a different slot region (a rotation of slots)."""
    q = p.copy()
    length = int(rng.integers(2, 6))
    start = int(rng.integers(0, 30 - length + 1))
    shift = int(rng.integers(1, length + 1))
    block = q[start:start + length]
    q[start:start + length] = np.roll(block, shift)
    return q


def _ox(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order crossover on the char->slot assignment, repaired to a permutation."""
    lo, hi = sorted(rng.choice(31, 2, replace=False).tolist())
    child = -np.ones(30, dtype=np.int32)
    taken = set()
    for i in range(lo, min(hi, 30)):
        child[i] = a[i]
        taken.add(int(a[i]))
    donor = [int(s) for s in b[:30] if int(s) not in taken]
    it = iter(donor)
    for i in range(30):
        if child[i] < 0:
            child[i] = next(it)
    return child


def _as31(p30: np.ndarray) -> np.ndarray:
    return np.concatenate([p30.astype(np.int32), np.array([30], dtype=np.int32)])


_ALL_PAIRS = np.array([(i, j) for i in range(30) for j in range(i + 1, 30)], dtype=np.int32)


def _neighbours(p30: np.ndarray) -> np.ndarray:
    """All 435 single-swap neighbours as (435, 31)."""
    n = _ALL_PAIRS.shape[0]
    out = np.repeat(p30[None, :], n, axis=0)
    rows = np.arange(n)
    i, j = _ALL_PAIRS[:, 0], _ALL_PAIRS[:, 1]
    out[rows, i], out[rows, j] = p30[j], p30[i]
    return np.concatenate([out, np.full((n, 1), 30, dtype=np.int32)], axis=1)


# --------------------------------------------------------------------------------------
# island epoch
# --------------------------------------------------------------------------------------
def _run_epoch(task: tuple) -> dict:
    """One island, one epoch. Returns updated state + this epoch's unique-key set."""
    island, seed, pop30, evals_target, polish_sweeps, immigrants, ga_share = task
    rng = np.random.default_rng([seed, island, len(pop30)])
    fe: EV.FastEval = _EVAL["fe"]
    seen: set[int] = set()
    n_eval = 0

    def score(batch31: np.ndarray) -> tuple[np.ndarray, dict]:
        nonlocal n_eval
        fit, g = _objective(batch31)
        for row in batch31:
            seen.add(_key(EV.layout_of(row)))
        n_eval += batch31.shape[0]
        return fit, g

    pop = [np.asarray(p, dtype=np.int32) for p in pop30]
    for imm in immigrants:
        pop.append(np.asarray(imm, dtype=np.int32))
    fit, gau = score(np.stack([_as31(p) for p in pop]))
    fit = list(fit)

    best_i = int(np.argmin(fit))
    best = (float(fit[best_i]), pop[best_i].copy())

    # Split each generation's spend: the GA explores/exploits the population, the restart
    # stream keeps the UNIQUE-eval rate up once the population converges. Both feed the same
    # steady-state replacement, so a restart's local optimum has to earn its slot on merit.
    restart_ratio = (1.0 - ga_share) / max(ga_share, 1e-9)
    while n_eval < evals_target:
        gen_start = n_eval
        # ---- produce offspring ----
        kids = []
        for _ in range(48):
            r = rng.random()
            if r < 0.35:
                kids.append(_swap(pop[int(rng.integers(len(pop)))], rng))
            elif r < 0.55:
                kids.append(_cycle3(pop[int(rng.integers(len(pop)))], rng))
            elif r < 0.72:
                kids.append(_block_relocate(pop[int(rng.integers(len(pop)))], rng))
            elif r < 0.95:
                a, b = rng.choice(len(pop), 2, replace=False)
                kids.append(_ox(pop[a], pop[b], rng))
            else:
                kids.append(rng.permutation(30).astype(np.int32))
        kfit, _ = score(np.stack([_as31(k) for k in kids]))

        # ---- polish the best few offspring with first-improvement 2-opt ----
        order = np.argsort(kfit)[:4]
        for idx in order:
            cur, cur_fit = kids[int(idx)].copy(), float(kfit[int(idx)])
            for _ in range(polish_sweeps):
                if n_eval >= evals_target:
                    break
                nb = _neighbours(cur)
                nfit, _ = score(nb)
                bi = int(np.argmin(nfit))
                if nfit[bi] < cur_fit - 1e-12:
                    cur_fit = float(nfit[bi])
                    cur = nb[bi, :30].copy()
                else:
                    break
            kids[int(idx)] = cur
            kfit[int(idx)] = cur_fit

        # ---- multi-start descent stream: fresh random restarts polished to a local
        # optimum. This is what keeps the UNIQUE-eval rate up once the islands converge:
        # a converged population's 2-opt polish re-walks basins it has already visited, so
        # unique/calls collapses (measured: 247k unique/epoch at epoch 1 -> 62k at epoch 18
        # on a call budget that never changed). A restart explores virgin space every time,
        # and its local optimum competes for a population slot on merit like any offspring.
        restarts, rfit = [], []
        restart_budget = min(evals_target, n_eval + int((n_eval - gen_start) * restart_ratio))
        while n_eval < restart_budget:
            cur = rng.permutation(30).astype(np.int32)
            cur_fit = float(score(_as31(cur)[None])[0][0])
            for _ in range(polish_sweeps):
                if n_eval >= restart_budget:
                    break
                nb = _neighbours(cur)
                nfit, _ = score(nb)
                bi = int(np.argmin(nfit))
                if nfit[bi] < cur_fit - 1e-12:
                    cur_fit = float(nfit[bi])
                    cur = nb[bi, :30].copy()
                else:
                    break
            restarts.append(cur)
            rfit.append(cur_fit)

        # ---- steady-state replacement: keep the best `POP` distinct ----
        kids = kids + restarts
        kfit = np.concatenate([np.asarray(kfit, dtype=np.float64),
                               np.asarray(rfit, dtype=np.float64)])
        merged = pop + kids
        mfit = np.concatenate([np.asarray(fit, dtype=np.float64), np.asarray(kfit, dtype=np.float64)])
        keep, seen_lay = [], set()
        for idx in np.argsort(mfit):
            lay = EV.layout_of(_as31(merged[int(idx)]))
            if lay in seen_lay:
                continue
            seen_lay.add(lay)
            keep.append(int(idx))
            if len(keep) >= 64:
                break
        pop = [merged[i].copy() for i in keep]
        fit = [float(mfit[i]) for i in keep]
        if fit[0] < best[0]:
            best = (fit[0], pop[0].copy())

    return {
        "island": island,
        "pop": [p.tolist() for p in pop],
        "fit": fit,
        "best_fit": best[0],
        "best_layout": EV.layout_of(_as31(best[1])),
        "n_eval": n_eval,
        "keys": list(seen),
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("evidence", "baseline", "constrained", "domain"))
    ap.add_argument("--budget", type=int, default=10_000_000, help="target UNIQUE evals")
    ap.add_argument("--islands", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--polish-sweeps", type=int, default=40)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ga-share", type=float, default=0.6,
                    help="fraction of each epoch's call budget spent on the GA population; "
                         "the remainder goes to the multi-start descent stream, which is "
                         "what keeps the UNIQUE-eval rate from collapsing on convergence")
    ap.add_argument("--overshoot", type=float, default=1.45,
                    help="calls-per-epoch multiplier covering the duplicate rate; the run "
                         "still stops on UNIQUE >= budget, so this only affects scheduling")
    args = ap.parse_args()

    out = Path(args.out)
    ckpt = out.with_suffix(".ckpt.json")
    keys_path = out.with_suffix(".keys.npy")
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    bounds = None
    if args.arm == "constrained":
        ref = json.load(open(OPTEV_STATE / "incumbent-reference.json"))
        bounds = {m: ref["incumbent_bounds"][m]["inc_max"] for m in WRONG_SIGNED}
        log(f"arm C hard bounds (incumbent max): " +
            ", ".join(f"{m}<={v:.4f}" for m, v in bounds.items()))

    # ---- shared initial populations: identical across arms (same seeds) ----
    seed_rng = np.random.default_rng(args.seed)
    init = [[seed_rng.permutation(30).astype(np.int32) for _ in range(64)]
            for _ in range(args.islands)]
    log(f"seeded {args.islands} islands x 64 random C30M permutations from seed {args.seed} "
        f"(IDENTICAL across arms)")

    start_epoch, all_keys, state = 0, set(), None
    if args.resume and ckpt.exists():
        blob = json.load(open(ckpt))
        if blob.get("arm") == args.arm:
            start_epoch = blob["epoch"]
            state = blob["islands"]
            if keys_path.exists():
                all_keys = set(np.load(keys_path).tolist())
            log(f"RESUMED from {ckpt} at epoch {start_epoch}, {len(all_keys):,} unique evals")

    # Overshoot the per-epoch CALL budget so the UNIQUE target is actually reachable: the
    # polish loop revisits neighbourhoods, so unique/calls runs ~0.7-0.95 and a 1:1 schedule
    # lands short. The stop condition is still UNIQUE >= budget (checked every epoch).
    per_epoch = max(1, int(args.budget * args.overshoot) // (args.epochs * args.islands))
    log(f"budget {args.budget:,} unique | {args.epochs} epochs x {args.islands} islands "
        f"x {per_epoch:,} calls/island/epoch (overshoot {args.overshoot}x)")

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(args.islands, 48), initializer=_init_worker,
                  initargs=(args.arm, args.corpus, bounds)) as pool:
        best_overall = (float("inf"), None)
        if state:
            for isl in state:
                if isl["best_fit"] < best_overall[0]:
                    best_overall = (isl["best_fit"], isl["best_layout"])
        for epoch in range(start_epoch, args.epochs):
            # migration: every island receives the 2 best layouts from the previous epoch
            immigrants: list[list] = [[] for _ in range(args.islands)]
            if state:
                tops = sorted(state, key=lambda s: s["best_fit"])[:3]
                for i in range(args.islands):
                    immigrants[i] = [EV.perm_of(t["best_layout"])[:30].tolist() for t in tops]
            tasks = []
            for i in range(args.islands):
                pop = state[i]["pop"] if state else [p.tolist() for p in init[i]]
                tasks.append((i, args.seed + 1009 * epoch, pop, per_epoch,
                              args.polish_sweeps, immigrants[i], args.ga_share))
            results = pool.map(_run_epoch, tasks)
            state = sorted(results, key=lambda r: r["island"])
            for r in state:
                all_keys.update(r["keys"])
                r["keys"] = []
                if r["best_fit"] < best_overall[0]:
                    best_overall = (r["best_fit"], r["best_layout"])
            calls = sum(r["n_eval"] for r in state)
            log(f"epoch {epoch + 1}/{args.epochs}: unique={len(all_keys):,} "
                f"(calls this epoch {calls:,}) best={best_overall[0]:.6f} "
                f"[{best_overall[1]}]")
            # PER-EPOCH CHECKPOINT (trap 7: a reboot destroyed 4.5M evals/arm once).
            # Keys go to a binary sidecar — 10M ints as JSON text is ~200 MB and minutes.
            ktmp = keys_path.with_suffix(".tmp.npy")
            np.save(ktmp, np.fromiter(all_keys, dtype=np.uint64, count=len(all_keys)))
            os.replace(ktmp, keys_path)
            tmp = ckpt.with_suffix(".tmp")
            json.dump({"arm": args.arm, "epoch": epoch + 1, "keys_file": str(keys_path),
                       "n_unique": len(all_keys),
                       "islands": state, "best_fit": best_overall[0],
                       "best_layout": best_overall[1], "seed": args.seed,
                       "budget": args.budget, "elapsed_s": time.time() - t0},
                      open(tmp, "w"))
            os.replace(tmp, ckpt)
            if len(all_keys) >= args.budget:
                log(f"budget reached at epoch {epoch + 1}")
                break

    # ---- final archive: every island's population, deduped, best first ----
    archive: dict[str, float] = {}
    for r in state:
        for p, f in zip(r["pop"], r["fit"], strict=False):
            archive[EV.layout_of(_as31(np.asarray(p, dtype=np.int32)))] = float(f)
    top = sorted(archive.items(), key=lambda kv: kv[1])
    blob = {
        "arm": args.arm, "corpus": args.corpus or "blend-v1 (production default)",
        "budget_requested": args.budget, "unique_evals": len(all_keys),
        "islands": args.islands, "epochs_run": len(state) and epoch + 1, "seed": args.seed,
        "bounds": bounds, "objective_unit": (
            "ms/char on the served K31 surface at 90 WPM (lower=better)" if args.arm == "baseline"
            else "attributed ms/trigram from the fitted SHAP loss curves, priced at the nearest "
                 "valid_domain edge (CLAMP; lower=better)" if args.arm == "domain"
            else "attributed ms/trigram from the fitted SHAP loss curves (lower=better)"),
        "domain_policy": ("clamp" if args.arm == "domain" else "extrapolate"),
        "champion": {"layout": top[0][0], "fitness": top[0][1]},
        "top50": [{"layout": lay, "fitness": f} for lay, f in top[:50]],
        "per_island_best": [{"island": r["island"], "best_fit": r["best_fit"],
                             "best_layout": r["best_layout"]} for r in state],
        "elapsed_s": time.time() - t0,
        "modelled_only": ("MODELLED ONLY: fitted-surface attribution, not measured typing "
                          "speed. No layout here is promoted or adopted."),
    }
    json.dump(blob, open(out, "w"), indent=1)
    log(f"WROTE {out}: champion {top[0][0]} fitness {top[0][1]:.6f} "
        f"({len(all_keys):,} unique evals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
