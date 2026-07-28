"""ARM D gate 2 — the engine is arm A's engine, and the clamp is actually WIRED INTO IT.

Gate 1 proved the domain policy is correct on the fitted curves. That is not enough: arm A's
pricing lives on a hand-rolled vectorized path (`evobj.Curve`) that never calls
`LossCurve.price`, so "the policy is correct" and "the search uses it" are two different claims.
This gate tests the second, which is the one arm D's validity actually rests on.

Four checks:

  1. **POSITIVE CONTROL — the engine is unchanged.** Run `search_armd.py --arm evidence` (the
     EXTRAPOLATING arm A objective) at a tiny budget, and run OPTEVIDENCE-1's own frozen
     `search.py --arm evidence` at the same budget/seed/islands. The champions and fitnesses must
     match EXACTLY. If they do, my edits changed nothing about arm A, so any arm A vs arm D
     difference is the policy. This is the check trap 10 is about: an "identical except X" claim
     is worthless unless you *ran* the without-X case and got the frozen answer back.

  2. **THE CLAMP IS LIVE IN THE WORKER.** Spawn the arm-D worker initializer and confirm the
     objective it holds is `ClampedEval` with policy `clamp`, and that its score on a known
     layout equals the clamped total (not the extrapolating one).

  3. **P6, THE ABORT CONDITION, IS MEASURABLE.** Under the arm-D objective, take a layout, push a
     gauge far past its ceiling, and confirm the objective's value is UNCHANGED. Done through the
     search's own `_objective`, not through a curve — so it tests the thing the search calls.

  4. **CHECKPOINT/RESUME IS BIT-EXACT** (trap 36: a resume that is not bit-exact gave
     `unique_layouts` 7167 vs 7140 while every verdict matched). Run 2 epochs straight, then run
     1 epoch + resume for 1, and compare champion, fitness AND unique count.

MODELLED ONLY. Corpus: blend-v1 (production default).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DRIVERS = Path("/local/home/zegertho/agent/state/armd/artifacts/drivers")
FROZEN = Path("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
REPO = Path("/tmp/domainfix")
ARM_JSON = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"

FAILURES: list[str] = []


def check(ok: bool, label: str) -> None:
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


def run_interrupted_after_epoch1(script: Path, out: Path, budget: int, islands: int,
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
    cmd = ["uv", "run", "--no-sync", "python", str(script), "--arm", "domain",
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
    tmp = Path(tempfile.mkdtemp(prefix="armd-gate2-"))
    print(f"scratch {tmp}")
    B, ISL, EP, SEED = 40_000, 4, 2, 20260728

    # ---- 1. positive control: my engine == the frozen engine, on arm A -------------------
    print("\n1. POSITIVE CONTROL — search_armd.py --arm evidence vs the FROZEN search.py")
    mine = run_search(DRIVERS / "search_armd.py", "evidence", tmp / "mine.json", B, ISL, EP, SEED)
    frozen = run_search(FROZEN / "search.py", "evidence", tmp / "frozen.json", B, ISL, EP, SEED)
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

    # ---- 2. the clamp is live in the worker ----------------------------------------------
    print("\n2. THE CLAMP IS LIVE IN THE ARM-D WORKER")
    sys.path.insert(0, str(DRIVERS))
    sys.path.append(str(FROZEN))
    import armd_obj as AD  # noqa: E402
    import evobj as EV  # noqa: E402
    import search_armd as SA  # noqa: E402
    from keybo.analysis.evidence_scorer import CLAMP, LIVE_GAUGES  # noqa: E402

    SA._init_worker("domain", None, None)
    fe_d = SA._EVAL["fe"]
    check(isinstance(fe_d, AD.ClampedEval), f"worker objective is ClampedEval: {type(fe_d)}")
    check(getattr(fe_d, "policy", None) == CLAMP, f"policy is {getattr(fe_d, 'policy', None)!r}")

    armA_champ = json.load(open("/local/home/zegertho/agent/state/optevidence/artifacts/"
                                "runs/arm-evidence.json"))["champion"]
    perm = EV.perm_of(armA_champ["layout"])[None]
    fit_d, _ = SA._objective(perm)
    pre = json.load(open("/local/home/zegertho/agent/state/armd/artifacts/pre-run-analysis.json"))
    a_row = next(r for r in pre["board"] if r["label"] == "armA")
    check(abs(float(fit_d[0]) - a_row["ev_clamp"]) < 1e-9,
          f"arm-D objective on arm A's champion = {float(fit_d[0]):.6f}, "
          f"expected clamped {a_row['ev_clamp']:.6f}")
    check(abs(float(fit_d[0]) - a_row["ev_extrapolate"]) > 1.0,
          f"...and is NOT the extrapolating {a_row['ev_extrapolate']:.6f} "
          f"(diff {abs(float(fit_d[0]) - a_row['ev_extrapolate']):.4f})")
    # and the frozen arm-A worker must still give the EXTRAPOLATING value
    SA._init_worker("evidence", None, None)
    fit_a, _ = SA._objective(perm)
    check(abs(float(fit_a[0]) - a_row["ev_extrapolate"]) < 1e-9,
          f"arm-A objective on its own champion = {float(fit_a[0]):.6f} "
          f"(frozen {armA_champ['fitness']:.6f})")
    check(abs(float(fit_a[0]) - armA_champ["fitness"]) < 1e-9,
          "...and reproduces the FROZEN arm A fitness exactly")

    # ---- 3. P6: pushing further out-of-domain buys nothing, THROUGH the search's objective
    print("\n3. P6 (the abort condition) — measured through the search's own objective")
    SA._init_worker("domain", None, None)
    fe_d = SA._EVAL["fe"]
    g = fe_d.gauges(perm)
    base = float(fe_d.evidence_score(g)[0])
    n_moved = 0
    for name in LIVE_GAUGES:
        curve = next(c for c in fe_d.curves if c.metric == name)
        lo, hi = curve.domain
        width = hi - lo
        for level in (hi + width, hi + 50 * width, lo - width, lo - 50 * width):
            if lo <= level <= hi:
                continue
            g2 = {k: v.copy() for k, v in g.items()}
            g2[name] = np.array([level])
            moved = float(fe_d.evidence_score(g2)[0])
            # The gauge's own term must be pinned; other terms are untouched, so the TOTAL
            # changes only by this gauge's clamped price minus its previous clamped price.
            expected = base - curve.price(g[name]) + curve.price(np.array([level]))
            if abs(moved - float(expected[0] if hasattr(expected, "__len__") else expected)) > 1e-9:
                check(False, f"{name} @ {level:.4f}: total moved unexpectedly")
                n_moved += 1
    # the headline form of P6: at the clamped level vs 50 widths beyond, EXACTLY equal
    for name in LIVE_GAUGES:
        curve = next(c for c in fe_d.curves if c.metric == name)
        lo, hi = curve.domain
        width = hi - lo
        g_hi = {k: v.copy() for k, v in g.items()}
        g_far = {k: v.copy() for k, v in g.items()}
        g_hi[name] = np.array([hi])
        g_far[name] = np.array([hi + 50 * width])
        check(float(fe_d.evidence_score(g_hi)[0]) == float(fe_d.evidence_score(g_far)[0]),
              f"P6 {name}: 50 domain-widths past the ceiling buys EXACTLY nothing")
    check(n_moved == 0, "no gauge's clamped term behaved unexpectedly")

    # ---- 4. resume is bit-exact ----------------------------------------------------------
    # ⚠ The partial run MUST be launched with the SAME `--epochs` as the straight one and then
    # KILLED, not launched with `--epochs 1`. `per_epoch = budget*overshoot // (epochs*islands)`
    # is derived FROM `--epochs`, so `--epochs 1` doubles the calls-per-island in epoch 0 and the
    # comparison is then between two different schedules, not between a resume and its
    # uninterrupted twin. My first version of this gate did exactly that and reported a
    # 71,207-vs-46,072 "resume bug" that was entirely my harness (the champion and fitness still
    # matched exactly, which is the tell trap 36 warns you NOT to stop at: matching verdicts do
    # not license skipping the count check, and a mismatching count needs its cause found).
    print("\n4. CHECKPOINT/RESUME BIT-EXACTNESS (trap 36)")
    straight = run_search(DRIVERS / "search_armd.py", "domain", tmp / "straight.json",
                          B, ISL, 2, SEED)
    part = run_interrupted_after_epoch1(DRIVERS / "search_armd.py", tmp / "resumed.json",
                                        B, ISL, 2, SEED)
    resumed = run_search(DRIVERS / "search_armd.py", "domain", tmp / "resumed.json",
                         B, ISL, 2, SEED, extra=["--resume"])
    check(resumed["champion"]["layout"] == straight["champion"]["layout"],
          f"resumed champion == straight: {resumed['champion']['layout']} vs "
          f"{straight['champion']['layout']}")
    check(resumed["champion"]["fitness"] == straight["champion"]["fitness"],
          f"resumed fitness == straight: {resumed['champion']['fitness']!r} vs "
          f"{straight['champion']['fitness']!r}")
    check(resumed["unique_evals"] == straight["unique_evals"],
          f"resumed unique_evals == straight: {resumed['unique_evals']} vs "
          f"{straight['unique_evals']} (trap 36 is about exactly this field)")
    check(part == 1, f"the partial run was killed after exactly 1 checkpointed epoch (got {part})")
    check(resumed["epochs_run"] == 2, f"resumed ran to epoch 2 (got {resumed['epochs_run']})")
    check(straight.get("domain_policy") == "clamp",
          f"arm D labelled clamp, got {straight.get('domain_policy')!r}")

    # ---- and arm D must differ from arm A at the same seed/budget -----------------------
    print("\n5. arm D actually differs from arm A at the same seed/budget")
    check(straight["champion"]["layout"] != mine["champion"]["layout"]
          or straight["champion"]["fitness"] != mine["champion"]["fitness"],
          f"arm D {straight['champion']['layout']} @ {straight['champion']['fitness']:.4f} "
          f"vs arm A {mine['champion']['layout']} @ {mine['champion']['fitness']:.4f}")

    print(f"\n{len(FAILURES)} failures")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("GATE 2 PASS — the engine is arm A's, the clamp is wired in, resume is bit-exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
