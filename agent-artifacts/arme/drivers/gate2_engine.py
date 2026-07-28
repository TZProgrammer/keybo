"""ARM E gate 2 — the engine is arm D's engine, and the ONLY thing that changed is the weights.

Gate 1 proved the domain policy is correct on the ARCHIVE curves and that `price_many` and arm D's
`ClampedCurve` are the same function. That is not enough. Arm E's comparability claim is "engine,
seed, islands, budget, corpus, operators and CLAMP are arm D's; only the weights JSON differs", and
that claim is testable — trap 10's lesson is that an "identical except X" assertion is worthless
unless you actually RAN the without-X case and got the frozen answer back.

Six checks:

  1. **POSITIVE CONTROL, ARM A.** Run `search_arme.py --arm evidence` (extrapolating arm A) and
     OPTEVIDENCE-1's frozen `search.py --arm evidence` at the same tiny budget/seed/islands. The
     champion, fitness, unique count and top50 ORDER must match exactly. If they do, my edits
     changed nothing about the engine.
  2. **POSITIVE CONTROL, ARM D.** Same, for `--arm domain` against arm D's frozen `search_armd.py`.
     This is the stronger control: it pins the whole CLAMPED path, so any arm D vs arm E difference
     is provably the weights and not my rewiring.
  3. **ARM E's WORKER HOLDS THE RIGHT OBJECTIVE.** The initializer must produce
     `ValidatedClampedEval` at policy `clamp`, over curves loaded from the ARCHIVE json — and its
     score on a known layout must equal a clamped total computed independently, and NOT the
     extrapolating one.
  4. **ARM E USES DIFFERENT WEIGHTS FROM ARM D, AND NOTHING ELSE.** The two workers' gauge
     dictionaries must be BITWISE identical on the same layouts (same corpus, same kernels, same
     denominators), while their scores differ. That is the experiment isolated to one variable.
  5. **P6, THE ABORT CONDITION, IS MEASURABLE THROUGH THE SEARCH'S OWN OBJECTIVE.** Push each
     gauge far past each edge and confirm the total is unchanged — via `search_arme._objective`,
     not via a curve, so it tests the thing the search actually calls.
  6. **CHECKPOINT/RESUME IS BIT-EXACT** (trap 36: a resume that was not bit-exact gave
     `unique_layouts` 7167 vs 7140 while every verdict matched). Interrupt a real run after epoch 1
     and resume it; champion, fitness AND unique count must match the uninterrupted run.

MODELLED ONLY. Corpus: blend-v1 (production default). Frame .native.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DRIVERS = Path("/local/home/zegertho/agent/state/arme/artifacts/drivers")
ARMD_DRIVERS = Path("/local/home/zegertho/agent/state/armd/artifacts/drivers")
FROZEN = Path("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
REPO = Path("/tmp/arme")
STATE = Path("/local/home/zegertho/agent/state/arme/artifacts")
ARCHIVE_JSON = ("/local/home/zegertho/agent/state/evidence-scorer/artifacts/"
                "arm-archive400-native.json")

FAILURES: list[str] = []
N_CHECKS = 0


def check(ok: bool, label: str) -> None:
    global N_CHECKS
    N_CHECKS += 1
    if ok:
        print(f"  ok    {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL  {label}")


def run_search(script: Path, arm: str, out: Path, budget: int, islands: int, epochs: int,
               seed: int, extra: list[str] | None = None) -> dict:
    cmd = ["uv", "run", "--no-sync", "python", str(script), "--arm", arm,
           "--budget", str(budget), "--islands", str(islands), "--epochs", str(epochs),
           "--seed", str(seed), "--polish-sweeps", "8", "--overshoot", "1.2",
           "--out", str(out)] + (extra or [])
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"{script.name} --arm {arm} rc={proc.returncode}\n"
                           f"{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
    return json.load(open(out))


def run_interrupted_after_epoch1(script: Path, arm: str, out: Path, budget: int, islands: int,
                                 epochs: int, seed: int) -> int:
    """Start a run with the FULL `--epochs`, kill it once epoch 1's checkpoint lands.

    This is what a host reboot looks like (trap 7), and it is the only faithful way to test
    resume: the per-epoch call schedule depends on `--epochs`, so a short run is a *different*
    experiment rather than a prefix of the long one. Returns the checkpointed epoch count.
    """
    import os
    import signal
    import time

    ckpt = out.with_suffix(".ckpt.json")
    for stale in (out, ckpt, out.with_suffix(".keys.npy")):
        stale.unlink(missing_ok=True)
    cmd = ["uv", "run", "--no-sync", "python", str(script), "--arm", arm,
           "--budget", str(budget), "--islands", str(islands), "--epochs", str(epochs),
           "--seed", str(seed), "--polish-sweeps", "8", "--overshoot", "1.2",
           "--out", str(out)]
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.time() + 900
    try:
        while time.time() < deadline:
            if ckpt.exists():
                try:
                    epoch = json.load(open(ckpt))["epoch"]
                except (json.JSONDecodeError, KeyError):
                    time.sleep(0.2)  # mid-atomic-replace; the driver uses os.replace
                    continue
                if epoch >= 1:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=60)
                    return int(epoch)
            if proc.poll() is not None:
                raise RuntimeError("the run finished before epoch 1's checkpoint was seen")
            time.sleep(0.2)
        raise RuntimeError("timed out waiting for epoch 1's checkpoint")
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=60)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arme-gate2-"))
    print(f"scratch {tmp}")
    B, ISL, EP, SEED = 40_000, 4, 2, 20260728
    summary: dict = {"scratch": str(tmp), "budget": B, "islands": ISL, "epochs": EP, "seed": SEED}

    # ---- 1. positive control on arm A: my engine == the frozen engine ---------------------
    print("\n1. POSITIVE CONTROL, ARM A — search_arme.py --arm evidence vs the FROZEN search.py")
    mine = run_search(DRIVERS / "search_arme.py", "evidence", tmp / "mineA.json", B, ISL, EP, SEED)
    frozen = run_search(FROZEN / "search.py", "evidence", tmp / "frozenA.json", B, ISL, EP, SEED)
    check(mine["champion"]["layout"] == frozen["champion"]["layout"],
          f"champion identical: {mine['champion']['layout']} vs {frozen['champion']['layout']}")
    check(mine["champion"]["fitness"] == frozen["champion"]["fitness"],
          f"fitness identical: {mine['champion']['fitness']!r} vs "
          f"{frozen['champion']['fitness']!r}")
    check(mine["unique_evals"] == frozen["unique_evals"],
          f"unique_evals identical: {mine['unique_evals']} vs {frozen['unique_evals']}")
    check([r["layout"] for r in mine["top50"]] == [r["layout"] for r in frozen["top50"]],
          "top50 order identical")
    check(mine.get("domain_policy") == "extrapolate",
          f"arm A labelled extrapolate, got {mine.get('domain_policy')!r}")
    summary["positive_control_armA"] = {
        "champion": mine["champion"], "frozen_champion": frozen["champion"],
        "unique_evals": mine["unique_evals"], "frozen_unique_evals": frozen["unique_evals"]}

    # ---- 2. positive control on arm D: the CLAMPED path is arm D's -----------------------
    print("\n2. POSITIVE CONTROL, ARM D — search_arme.py --arm domain vs FROZEN search_armd.py")
    mineD = run_search(DRIVERS / "search_arme.py", "domain", tmp / "mineD.json", B, ISL, EP, SEED)
    frozenD = run_search(ARMD_DRIVERS / "search_armd.py", "domain", tmp / "frozenD.json",
                         B, ISL, EP, SEED)
    check(mineD["champion"]["layout"] == frozenD["champion"]["layout"],
          f"champion identical: {mineD['champion']['layout']} vs "
          f"{frozenD['champion']['layout']}")
    check(mineD["champion"]["fitness"] == frozenD["champion"]["fitness"],
          f"fitness identical: {mineD['champion']['fitness']!r} vs "
          f"{frozenD['champion']['fitness']!r}")
    check(mineD["unique_evals"] == frozenD["unique_evals"],
          f"unique_evals identical: {mineD['unique_evals']} vs {frozenD['unique_evals']}")
    check([r["layout"] for r in mineD["top50"]] == [r["layout"] for r in frozenD["top50"]],
          "top50 order identical")
    summary["positive_control_armD"] = {
        "champion": mineD["champion"], "frozen_champion": frozenD["champion"],
        "unique_evals": mineD["unique_evals"], "frozen_unique_evals": frozenD["unique_evals"]}

    # ---- 3. arm E's worker holds the right objective --------------------------------------
    print("\n3. ARM E's WORKER OBJECTIVE")
    sys.path.insert(0, str(DRIVERS))
    sys.path.append(str(ARMD_DRIVERS))
    sys.path.append(str(FROZEN))
    import armd_obj as AD  # noqa: E402
    import arme_obj as AE  # noqa: E402
    import evobj as EV  # noqa: E402
    import search_arme as SA  # noqa: E402
    from arme_load import load_curves  # noqa: E402
    from keybo.analysis.evidence_scorer import CLAMP, EXTRAPOLATE, LIVE_GAUGES  # noqa: E402

    SA._init_worker("archive", None, None)
    fe_e = SA._EVAL["fe"]
    check(isinstance(fe_e, AE.ValidatedClampedEval),
          f"worker objective is ValidatedClampedEval: {type(fe_e).__name__}")
    check(getattr(fe_e, "policy", None) == CLAMP,
          f"policy is {getattr(fe_e, 'policy', None)!r}")
    check(SA.weights_for("archive") == ARCHIVE_JSON,
          f"arm E's weights json is {SA.weights_for('archive')}")
    check(fe_e.weights_meta.get("pool") == "archive-400",
          f"loaded weights pool is {fe_e.weights_meta.get('pool')!r}")
    check(fe_e.weights_meta.get("surface_frame") == "native",
          f"loaded weights frame is {fe_e.weights_meta.get('surface_frame')!r}")
    check(str(fe_e.corpus_dir).endswith("blend-v1"), f"corpus is {fe_e.corpus_dir}")

    # its score must be the CLAMPED total, computed independently, and not the extrapolating one
    curves = load_curves(ARCHIVE_JSON)
    inc = json.load(open(FROZEN.parent / "incumbent-reference.json"))
    probe_layouts = list(inc["incumbents"].values())[:4] + [inc["reference"]["qwerty30m"]]
    perms = np.stack([EV.perm_of(lay) for lay in probe_layouts])
    fit_e, _ = SA._objective(perms)
    g_e = fe_e.gauges(perms)
    want_clamp = np.zeros(len(probe_layouts))
    want_ext = np.zeros(len(probe_layouts))
    for name in LIVE_GAUGES:
        want_clamp = want_clamp + curves[name].price_many(g_e[name], policy=CLAMP)
        want_ext = want_ext + curves[name].price_many(g_e[name], policy=EXTRAPOLATE)
    check(np.array_equal(np.asarray(fit_e), want_clamp),
          "arm E's search objective == the independently computed CLAMPED archive total")
    check(float(np.max(np.abs(want_clamp - want_ext))) > 1.0,
          f"...and the clamp is not inert: max|clamp - extrap| = "
          f"{float(np.max(np.abs(want_clamp - want_ext))):.4f}")
    summary["armE_objective"] = {
        "type": type(fe_e).__name__, "policy": fe_e.policy,
        "pool": fe_e.weights_meta.get("pool"),
        "clamp_vs_extrap_max_gap": float(np.max(np.abs(want_clamp - want_ext)))}

    # ---- 4. arm E vs arm D: SAME gauges, DIFFERENT score ----------------------------------
    print("\n4. ARM E vs ARM D — identical gauges (one variable changed), different score")
    SA._init_worker("domain", None, None)
    fe_d = SA._EVAL["fe"]
    check(isinstance(fe_d, AD.ClampedEval), f"arm D worker is ClampedEval: {type(fe_d).__name__}")
    g_d = fe_d.gauges(perms)
    same_gauges = all(np.array_equal(g_d[m], g_e[m]) for m in LIVE_GAUGES)
    check(same_gauges, "every gauge value is BITWISE identical between the two workers "
                       "(same corpus, kernels and denominators — so only the curves differ)")
    fit_d, _ = SA._objective(perms)
    check(float(np.max(np.abs(np.asarray(fit_d) - np.asarray(fit_e)))) > 1.0,
          f"...while the scores differ (max gap "
          f"{float(np.max(np.abs(np.asarray(fit_d) - np.asarray(fit_e)))):.4f})")
    check(fe_d.weights_meta.get("pool") == "random-c30m-400",
          f"arm D's pool is {fe_d.weights_meta.get('pool')!r}")
    summary["one_variable"] = {
        "gauges_bitwise_identical": bool(same_gauges),
        "armD_pool": fe_d.weights_meta.get("pool"), "armE_pool": fe_e.weights_meta.get("pool"),
        "score_max_gap": float(np.max(np.abs(np.asarray(fit_d) - np.asarray(fit_e))))}

    # ---- 5. P6: pushing further out-of-domain buys NOTHING, through the search's objective --
    # ⚠ TWO DIFFERENT CLAIMS, AND ONLY ONE OF THEM IS EXACT. My first version demanded exact
    # equality against `base - edge_before + edge_after`, and it failed at 2.220e-16 — because
    # that expression RE-ASSOCIATES the 14-term sum (subtract one term, add another) and so
    # carries its own rounding. The clamp was never in question; my expectation was.
    #   (a) RE-ASSOCIATION check: the total moves by the edge-price delta and nothing else.
    #       Compared with a tolerance, because the two ways of summing 14 doubles legitimately
    #       differ in the last bit.
    #   (b) THE HEADLINE P6 CLAIM, which IS exact and is the brief's abort condition: evaluating
    #       at the clamped level and at 50 (and 1000) domain-widths beyond gives *bit-identical*
    #       totals — same terms, same order, so `==` is the right operator and 0.000e+00 is the
    #       required answer. This is what "the clamp binds" means for a maximizer.
    print("\n5. P6 (the abort condition) — measured through arm E's own search objective")
    SA._init_worker("archive", None, None)
    fe_e = SA._EVAL["fe"]
    one = perms[:1]
    g1 = fe_e.gauges(one)
    base = float(fe_e.evidence_score(g1)[0])

    # (a) re-association
    worst_assoc, n_pushes = 0.0, 0
    for name in LIVE_GAUGES:
        curve = next(c for c in fe_e.curves if c.metric == name)
        lo, hi = curve.domain
        width = hi - lo
        for level in (hi + width, hi + 50 * width, lo - width, lo - 50 * width):
            if lo <= level <= hi:
                continue
            g2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in g1.items()}
            g2[name] = np.array([level])
            moved = float(fe_e.evidence_score(g2)[0])
            edge = float(curve.price_many(np.array([min(max(g1[name][0], lo), hi)]),
                                          policy=CLAMP)[0])
            edge_at = float(curve.price_many(np.array([min(max(level, lo), hi)]),
                                             policy=CLAMP)[0])
            worst_assoc = max(worst_assoc, abs(moved - (base - edge + edge_at)))
            n_pushes += 1
    check(worst_assoc < 1e-9,
          f"(a) the total moves by the edge-price delta and nothing else: worst re-association "
          f"deviation {worst_assoc:.3e} over {n_pushes} pushes (tolerance 1e-9)")

    # (b) the headline, EXACT: further out buys bit-identically nothing
    worst_reward, n_pairs = 0.0, 0
    for name in LIVE_GAUGES:
        curve = next(c for c in fe_e.curves if c.metric == name)
        lo, hi = curve.domain
        width = hi - lo
        for edge_level, far_level in ((hi, hi + 50 * width), (hi, hi + 1000 * width),
                                      (lo, lo - 50 * width), (lo, lo - 1000 * width)):
            g_edge = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in g1.items()}
            g_far = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in g1.items()}
            g_edge[name] = np.array([edge_level])
            g_far[name] = np.array([far_level])
            v_edge = float(fe_e.evidence_score(g_edge)[0])
            v_far = float(fe_e.evidence_score(g_far)[0])
            worst_reward = max(worst_reward, abs(v_far - v_edge))
            n_pairs += 1
            check(v_far == v_edge,
                  f"P6 {name}: {abs(far_level - edge_level) / width:.0f} domain-widths past the "
                  f"{'ceiling' if far_level > hi else 'floor'} buys EXACTLY nothing")
    check(worst_reward == 0.0,
          f"P6 HEADLINE: worst |reward for going N domain-widths outside| = {worst_reward:.3e} "
          f"over {n_pairs} pairs on all 14 gauges — must be exactly 0.000e+00")
    print(f"     worst |reward outside| = {worst_reward:.3e} over {n_pairs} pairs, all 14 gauges")
    summary["P6_clamp_binds"] = {"worst_reward_outside": worst_reward, "n_pairs": n_pairs,
                                 "worst_reassociation_deviation": worst_assoc,
                                 "n_pushes": n_pushes}

    # ---- 6. checkpoint/resume is bit-exact -----------------------------------------------
    print("\n6. CHECKPOINT/RESUME BIT-EXACTNESS (trap 36)")
    straight = run_search(DRIVERS / "search_arme.py", "archive", tmp / "straight.json",
                          B, ISL, EP, SEED)
    killed_at = run_interrupted_after_epoch1(DRIVERS / "search_arme.py", "archive",
                                             tmp / "resumed.json", B, ISL, EP, SEED)
    print(f"     (killed after epoch {killed_at}; resuming)")
    resumed = run_search(DRIVERS / "search_arme.py", "archive", tmp / "resumed.json",
                         B, ISL, EP, SEED, extra=["--resume"])
    check(straight["champion"]["layout"] == resumed["champion"]["layout"],
          f"champion identical: {straight['champion']['layout']} vs "
          f"{resumed['champion']['layout']}")
    check(straight["champion"]["fitness"] == resumed["champion"]["fitness"],
          f"fitness identical: {straight['champion']['fitness']!r} vs "
          f"{resumed['champion']['fitness']!r}")
    check(straight["unique_evals"] == resumed["unique_evals"],
          f"unique_evals identical: {straight['unique_evals']} vs {resumed['unique_evals']} "
          f"(trap 36: verdicts matching is NOT enough, the COUNT has to match)")
    check([r["layout"] for r in straight["top50"]] == [r["layout"] for r in resumed["top50"]],
          "top50 order identical")
    check(straight.get("domain_policy") == "clamp",
          f"arm E labelled clamp, got {straight.get('domain_policy')!r}")
    check(straight.get("weights_pool") == "archive-400",
          f"arm E labelled archive-400, got {straight.get('weights_pool')!r}")
    summary["resume"] = {"killed_after_epoch": killed_at,
                         "straight": straight["champion"], "resumed": resumed["champion"],
                         "unique_straight": straight["unique_evals"],
                         "unique_resumed": resumed["unique_evals"]}

    ok = not FAILURES
    summary["n_checks"] = N_CHECKS
    summary["n_failures"] = len(FAILURES)
    summary["failures"] = FAILURES
    summary["modelled_only"] = ("MODELLED ONLY: fitted-surface attribution, not measured typing "
                                "speed.")
    STATE.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(STATE / "gate2-engine.json", "w"), indent=1)
    print(f"\nGATE 2: {N_CHECKS} checks, {len(FAILURES)} failures — {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    p = STATE / "gate2-rc.txt"
    t = p.with_suffix(".tmp")
    t.write_text(f"{rc}\n")
    t.replace(p)
    sys.exit(rc)
