"""OPTEVIDENCE island memetic search — one engine, three arms, identical budget and seeds.

Arms (the objective is ALWAYS minimized):
  A  evidence   minimize the evidence score from the fitted SHAP weights + loss curves
  B  baseline   minimize predicted ms/char on the served K31 surface at 90 WPM
  C  constrained arm A subject to HARD non-regression bounds on the five wrong-signed gauges
                (a candidate may not exceed the incumbent band's max on any of them)

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
import evobj as EV  # noqa: E402

ARM_JSON = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
STATE = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
WRONG_SIGNED = ("scissor", "sfb", "sfb-dist", "lsb-dist", "sfs")

_EVAL: dict = {}


def _key(layout: str) -> int:
    return int.from_bytes(hashlib.blake2b(layout.encode(), digest_size=8).digest(), "little")


# --------------------------------------------------------------------------------------
# worker-side objective
# --------------------------------------------------------------------------------------
#: ARM G (see agent-artifacts/armg/PREREGISTRATION.md). Reference layout, the 14 live
#: gauges' directions (DERIVED: EXPECTED_SIGN agrees 13/14 with rank-correlation over 4000
#: random perms and 14/14 with the qwerty-is-worst reference), and the pool-matched scale
#: (range over the six frozen 1M champions -- near-optimal, NOT a random pool, per trap 26).
#: ⚠ ARMG_REF / ARMG_SCALE below are GENERATED from D-prereg-input.json, never hand-typed.
#: An earlier hand-transcribed copy drifted ~1e-5 on all 14 constants and the objective's
#: own positive control caught it. Regenerate, never retype.
ARMG_LAYOUT_REF = "flmpg-yuo,sntdcireahkxbwv'.jzq"   # arm B
ARMG_REF_MS = 253.90057910352604
ARMG_EPS = 0.1234          # = 2 x 0.0617, SPEEDTIE-1's registered 2x-sd band
ARMG_LAMBDA = 1000.0
#: the six frozen SPEEDTIE-1 1M champions -- the pool ARMG_SCALE is the range over.
ARMG_SIX = ("flmpg-yuo,sntdcireahkxbwv'.jzq", "puy.,vdfnlheioamtsrc'jqk-gwbxz",
            "pyou,vdflrghaeictsnmk'j.-wbzxq", "lcfmk.uoyprnstdiaeghzxwbv-,'qj",
            "lnfdg.,yehcrstmaoiupxzbwvk-q'j", "pyu.,gdfnlhieaocstrmkj'-qbwzvx")
ARMG_DIR = {"sfb": 1.0, "sfs": 1.0, "sfb-dist": 1.0, "sfs-dist": 1.0, "lsb": 1.0,
            "lsb-dist": 1.0, "alt": -1.0, "roll": -1.0, "sr-roll": -1.0, "redir": 1.0,
            "scissor": 1.0, "imbalance": 1.0, "oxey-style": 1.0, "comfort": 1.0}
ARMG_REF = {"sfb": 2.539124615505356, "sfs": 6.799508454836403,
            "sfb-dist": 3.0423296718509327, "sfs-dist": 8.005623911481186,
            "lsb": 1.1410913569660872, "lsb-dist": 2.322675604674633,
            "alt": 37.13733008655502, "roll": 45.442097658599806,
            "sr-roll": 17.813072191285166, "redir": 4.420586814037255,
            "scissor": 0.25671106120493087, "imbalance": 4.8753996439391,
            "oxey-style": 8.611045585392063, "comfort": 3.4140465}
ARMG_SCALE = {"sfb": 0.7619453842296489, "sfs": 3.7067685207163077,
              "sfb-dist": 0.8994860182130573, "sfs-dist": 4.408887627329467,
              "lsb": 0.9720269138545692, "lsb-dist": 2.270010737371103,
              "alt": 8.282488925949309, "roll": 7.266205092474905,
              "sr-roll": 7.472619632776423, "redir": 1.0304806510713163,
              "scissor": 0.18851669433861878, "imbalance": 4.599910684123468,
              "oxey-style": 13.153434063343994, "comfort": 0.6334071809999999}


def armg_assert_constants(fe: EV.FastEval, six: list[str]) -> dict:
    """Re-derive ARMG_REF/ARMG_SCALE from the LIVE code and refuse to run if the frozen
    constants have drifted. This exists because a hand-transcribed copy of these 28 numbers
    drifted ~1e-5 on ALL of them and only a positive control caught it -- the same
    "a label is not its referent" class as the campaign's borrowed-ruler failures. A
    constant that is never re-derived is a constant that is silently wrong."""
    ref = fe.gauges(np.stack([EV.perm_of(ARMG_LAYOUT_REF)]))
    g6 = fe.gauges(np.stack([EV.perm_of(x) for x in six]))
    worst_ref = max(abs(float(ref[k][0]) - ARMG_REF[k]) for k in ARMG_DIR)
    worst_scale = max(abs(float(g6[k].max() - g6[k].min()) - ARMG_SCALE[k]) for k in ARMG_DIR)
    worst_ms = abs(float(ref["_ms_per_char"][0]) - ARMG_REF_MS)
    assert worst_ref < 1e-12, f"ARMG_REF drifted by {worst_ref:.3e}"
    assert worst_scale < 1e-12, f"ARMG_SCALE drifted by {worst_scale:.3e}"
    assert worst_ms < 1e-11, f"ARMG_REF_MS drifted by {worst_ms:.3e}"
    d0 = float(armg_deficit(ref)[0])
    assert d0 == 0.0, f"D(reference) must be EXACTLY 0.0, got {d0!r}"
    return {"worst_ref": worst_ref, "worst_scale": worst_scale, "worst_ms": worst_ms,
            "D_reference": d0}


def armg_deficit(g: dict[str, np.ndarray]) -> np.ndarray:
    """D(L) = sum_g max(0, dir_g*(g(L) - g(armB))/s_g). Zero IFF the layout is no worse
    than arm B on all 14 live gauges. `sfr` is excluded: it is a PERMUTATION INVARIANT
    (trap 23) and so is a tie by construction that cannot be earned."""
    d = np.zeros_like(g["sfb"])
    for name, direction in ARMG_DIR.items():
        excess = direction * (g[name] - ARMG_REF[name]) / ARMG_SCALE[name]
        d = d + np.maximum(excess, 0.0)
    return d


# --------------------------------------------------------------------------------------
# ARM H: minimize `oxey-style` ALONE subject to HARD constraints (see
# agent-artifacts/armh/PREREGISTRATION.md, committed 491138b BEFORE any result existed).
#
# ARM G's registered self-diagnosis is the premise: its `D` was an UNWEIGHTED SUM of
# RANGE-NORMALIZED excesses, so the WIDEST axis was the CHEAPEST to trade away -- and
# `oxey-style` is 48.5% of the board's whole gauge range. Its objective was built to
# sacrifice exactly the axis it existed to collect.
#
# So hardness here is LEXICOGRAPHIC BY CONSTRUCTION, not a summed penalty (trap 51: a
# maximizer does not read flags, and A SUMMED PENALTY IS A FLAG). The two branches occupy
# DISJOINT intervals with a 4-order gap, so there is NO exchange rate between the objective
# and a constraint:
#     V == 0  ->  fitness = oxey_style        in [-13, +89] over the whole real board
#     V >  0  ->  fitness = BIG + V           >= 1e6
# `V` is retained inside the infeasible branch so the search still has a gradient TOWARD
# feasibility, which pure rejection (+inf) would not give -- necessary because the feasible
# set is a needle (0 of 200,000 random layouts hold even 6 of the 13 constraints).
# --------------------------------------------------------------------------------------
import armh_constants as AH  # noqa: E402


def armh_violation(g: dict[str, np.ndarray], ms_edge: float, eps: float) -> np.ndarray:
    """V(L): 0 iff L satisfies all 13 hard axis constraints AND the speed band.

    Each axis excess is normalized by |g(armB)| so no axis's scale dominates the gradient,
    and the speed leg by `eps` (= 2*sd_H). These normalizations affect only the SHAPE of the
    infeasible gradient -- they can never trade against the objective, because the feasible
    and infeasible branches are disjoint intervals.
    """
    v = np.zeros_like(g["sfb"])
    for name in AH.ARMH_CONSTRAINED:
        excess = AH.ARMH_DIR[name] * (g[name] - AH.ARMH_REF[name]) / max(abs(AH.ARMH_REF[name]), 1e-9)
        v = v + np.maximum(excess, 0.0)
    v = v + np.maximum(g["_ms_per_char"] - ms_edge, 0.0) / max(eps, 1e-12)
    return v


def armh_fitness(g: dict[str, np.ndarray], ms_edge: float, eps: float) -> np.ndarray:
    v = armh_violation(g, ms_edge, eps)
    feasible = v <= 0.0
    return np.where(feasible, g[AH.ARMH_TARGET], AH.ARMH_BIG + v)


def armh_assert_constants(fe: EV.FastEval, ms_edge: float, eps: float) -> dict:
    """Re-derive every ARM H constant from LIVE code and REFUSE TO RUN on drift.

    Modelled on ARM G's `armg_assert_constants()`, which exists because ARM G hand-typed 28
    constants and ALL of them were wrong by ~1e-5. It also asserts the two properties the
    whole design rests on: arm B is FEASIBLE with V == 0 (it is the constraint reference, so
    this must hold by construction), and the INTERVAL SEPARATION that makes hardness real.
    """
    ref = fe.gauges(np.stack([EV.perm_of(AH.ARMH_LAYOUT_REF)]))
    worst_ref = max(abs(float(ref[k][0]) - AH.ARMH_REF[k]) for k in AH.ARMH_LIVE)
    worst_ms = abs(float(ref["_ms_per_char"][0]) - AH.ARMH_REF_MS)
    assert worst_ref < 1e-12, f"ARMH_REF drifted by {worst_ref:.3e}"
    assert worst_ms < 1e-11, f"ARMH_REF_MS drifted by {worst_ms:.3e}"
    # arm B must be feasible with EXACTLY zero violation -- it IS the reference.
    v0 = float(armh_violation(ref, ms_edge, eps)[0])
    assert v0 == 0.0, f"V(armB) must be EXACTLY 0.0, got {v0!r}"
    f0 = float(armh_fitness(ref, ms_edge, eps)[0])
    assert abs(f0 - AH.ARMH_REF["oxey-style"]) < 1e-12, f"fitness(armB) != oxey(armB): {f0!r}"
    # BALL-1: the prereg's enumerated feasible layout. Re-derive its two published numbers.
    b1 = fe.gauges(np.stack([EV.perm_of(AH.ARMH_BALL1)]))
    d_ms = abs(float(b1["_ms_per_char"][0]) - AH.ARMH_BALL1_MS)
    d_ox = abs(float(b1[AH.ARMH_TARGET][0]) - AH.ARMH_BALL1_OXEY)
    assert d_ms < 1e-9 and d_ox < 1e-9, f"BALL-1 drifted: ms {d_ms:.3e} oxey {d_ox:.3e}"
    b1_v = float(armh_violation(b1, 1e18, eps)[0])   # axes only, speed leg disabled
    assert b1_v == 0.0, f"BALL-1 must satisfy the 13 AXES exactly, got V={b1_v!r}"
    # INTERVAL SEPARATION, asserted numerically rather than argued: a deliberately awful
    # layout (qwerty) must land in the infeasible branch, above BIG, and therefore above
    # every possible feasible score.
    qw = fe.gauges(np.stack([EV.perm_of(EV.C30M)]))
    f_qw = float(armh_fitness(qw, ms_edge, eps)[0])
    assert f_qw >= AH.ARMH_BIG, f"qwerty must be infeasible-branch, got {f_qw!r}"
    return {"worst_ref": worst_ref, "worst_ms": worst_ms, "V_armB": v0,
            "fitness_armB": f0, "ball1_ms_diff": d_ms, "ball1_oxey_diff": d_ox,
            "ball1_V_axes_only": b1_v, "fitness_qwerty": f_qw,
            "ms_edge": ms_edge, "eps": eps}


def _init_worker(arm: str, corpus: str | None, bounds: dict | None,
                 armh_band: tuple[float, float] | None = None) -> None:
    # ARM G needs no fitted weight curves -- it optimizes shipped gauges + the served
    # surface only. Passing weights_json=None avoids loading (and depending on) the
    # SHAP arm JSON, whose curves OPTEVIDENCE-1 showed are unbounded under extrapolation.
    wj = None if arm in ("armg", "armh") else ARM_JSON
    fe = EV.FastEval(corpus=corpus, weights_json=wj, with_surface=True)
    _EVAL["fe"] = fe
    _EVAL["arm"] = arm
    _EVAL["bounds"] = bounds or {}
    _EVAL["armh_band"] = armh_band


def _objective(perms: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(B,31) perms -> (fitness to MINIMIZE, gauge dict). Constraint violation is a
    quadratic-in-excess penalty added to the arm-A objective, scaled so that any violation
    dominates the whole feasible score range (which spans ~35 units)."""
    fe: EV.FastEval = _EVAL["fe"]
    arm = _EVAL["arm"]
    g = fe.gauges(perms)
    if arm == "baseline":
        return g["_ms_per_char"].copy(), g
    if arm == "armg":
        # Lexicographic-by-penalty: minimize the dominance deficit against arm B subject
        # to staying inside arm B's speed band. The speed term is QUADRATIC in the excess
        # and normalized by EPS, and LAMBDA=1000 against a D range of ~14, so one full EPS
        # of violation dominates any achievable gauge gain. Trap 51's lesson applied in
        # advance: a maximizer does not read flags, so the band must be hard IN EFFECT.
        d = armg_deficit(g)
        over = np.maximum(g["_ms_per_char"] - (ARMG_REF_MS + ARMG_EPS), 0.0) / ARMG_EPS
        return d + ARMG_LAMBDA * over * over, g
    if arm == "armh":
        ms_edge, eps = _EVAL["armh_band"]
        return armh_fitness(g, ms_edge, eps), g
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
                    choices=("evidence", "baseline", "constrained", "armg", "armh"))
    ap.add_argument("--budget", type=int, default=10_000_000, help="target UNIQUE evals")
    ap.add_argument("--islands", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--polish-sweeps", type=int, default=40)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--armh-eps", type=float, default=None,
                    help="ARM H speed band half-width = 2*sd_H, MEASURED from this arm's own "
                         "baseline control. The search band and the verdict band are THE SAME "
                         "NUMBER by construction -- ARM G's failure was a search band LOOSER "
                         "than its verdict band by 0.0251.")
    ap.add_argument("--armh-warm", action="store_true",
                    help="inject arm B into every island's initial population. FAIL-LOUD: the "
                         "run exits rc=1 if the injection is absent or evaluates infeasible, "
                         "because an optional warm start that finds nothing degrades to a COLD "
                         "run and still reports (trap 10).")
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
        ref = json.load(open(STATE / "incumbent-reference.json"))
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

    # ARM G: refuse to run on drifted constants, and record the check in the output blob.
    # A frozen constant that is never re-derived is a constant that is silently wrong --
    # this gate exists because a hand-transcribed copy of it was wrong on all 28 numbers.
    armg_check = None
    if args.arm == "armg":
        _fe_chk = EV.FastEval(corpus=args.corpus, weights_json=None, with_surface=True)
        assert str(Path(_fe_chk.corpus_dir).resolve()).startswith("/tmp/armh/"), (
            f"corpus dir escaped the worktree: {_fe_chk.corpus_dir}")
        armg_check = armg_assert_constants(_fe_chk, list(ARMG_SIX))
        log(f"ARM G constants re-derived from live code: {armg_check}")
        log(f"ARM G band: ms <= {ARMG_REF_MS:.10f} + {ARMG_EPS} = "
            f"{ARMG_REF_MS + ARMG_EPS:.10f}  (lambda={ARMG_LAMBDA})")
        del _fe_chk

    # ARM H: same discipline. The band comes from THIS arm's own measured sd_H via
    # --armh-eps, so the SEARCH band and the VERDICT band are the same number (ARM G's
    # failure was a search band LOOSER than its verdict band by 0.0251).
    armh_check = None
    armh_band = None
    if args.arm == "armh":
        if args.armh_eps is None:
            raise SystemExit("ARM H requires --armh-eps (= 2*sd_H from its OWN baseline "
                             "control). Refusing to invent a band.")
        armh_band = (AH.ARMH_REF_MS + args.armh_eps, args.armh_eps)
        _fe_chk = EV.FastEval(corpus=args.corpus, weights_json=None, with_surface=True)
        assert str(Path(_fe_chk.corpus_dir).resolve()).startswith("/tmp/armh/"), (
            f"corpus dir escaped the worktree: {_fe_chk.corpus_dir}")
        armh_check = armh_assert_constants(_fe_chk, armh_band[0], armh_band[1])
        log(f"ARM H constants re-derived from live code: {armh_check}")
        log(f"ARM H band: ms <= {AH.ARMH_REF_MS:.10f} + {args.armh_eps} = "
            f"{armh_band[0]:.10f}   (13 hard axes at g(armB), TOL={AH.ARMH_TOL})")
        # WARM START, fail-loud (trap 10): an optional warm start that finds nothing
        # degrades to a COLD run and still reports a result.
        if args.armh_warm:
            inj = EV.perm_of(AH.ARMH_LAYOUT_REF)[:30].astype(np.int32)
            for i in range(args.islands):
                init[i][0] = inj.copy()
            got = sum(1 for i in range(args.islands)
                      if EV.layout_of(np.concatenate([init[i][0], [30]]).astype(np.int32))
                      == AH.ARMH_LAYOUT_REF)
            assert got == args.islands, (
                f"WARM START FAILED: arm B present in {got}/{args.islands} islands")
            gi = _fe_chk.gauges(np.stack([EV.perm_of(AH.ARMH_LAYOUT_REF)]))
            vi = float(armh_violation(gi, armh_band[0], armh_band[1])[0])
            assert vi == 0.0, f"WARM START FAILED: injected layout is INFEASIBLE (V={vi!r})"
            log(f"WARM START: arm B injected into {got}/{args.islands} islands, V={vi}")
        else:
            log("COLD START: islands x 64 uniform random C30M permutations, no injection")
        del _fe_chk

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(args.islands, 48), initializer=_init_worker,
                  initargs=(args.arm, args.corpus, bounds, armh_band)) as pool:
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
            else ("oxey-style if FEASIBLE else BIG+V (lexicographic; lower=better)")
            if args.arm == "armh"
            else "attributed ms/trigram from the fitted SHAP loss curves (lower=better)"),
        "champion": {"layout": top[0][0], "fitness": top[0][1]},
        "armg_constants_check": armg_check,
        "armh_constants_check": armh_check,
        "armh_band": ({"ref_ms": AH.ARMH_REF_MS, "eps": args.armh_eps,
                       "edge": armh_band[0] if armh_band else None,
                       "tol": AH.ARMH_TOL, "warm": bool(args.armh_warm),
                       "constrained_axes": list(AH.ARMH_CONSTRAINED),
                       "target": AH.ARMH_TARGET} if args.arm == "armh" else None),
        "armg_band": ({"ref_ms": ARMG_REF_MS, "eps": ARMG_EPS, "lambda": ARMG_LAMBDA,
                       "edge": ARMG_REF_MS + ARMG_EPS} if args.arm == "armg" else None),
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
