"""ARM H CHAMPION GATE — the second, INDEPENDENT hardness layer.

Layer 1 is the objective's interval separation (feasible scores in ~[-13,+89], infeasible
>= 1e6), which makes it impossible for the SEARCH to rank an infeasible layout above a
feasible one. But layer 1 lives inside `FastEval`, i.e. inside the thing under test. So this
gate re-checks feasibility through the **SHIPPED `keybo analyze`** path and exits rc=1 if any
returned champion is infeasible. `pc_fasteval.py` (C3/C4) pins the two paths at 1.233e-14 and
is mutation-proven to bite, which is what makes this an independent check rather than the
same check twice (trap 45 / the SELF-AUDIT SWEEP's "two controls that shared the component
under test").

    --plant-infeasible   inject a KNOWN-INFEASIBLE layout as a champion and require rc=1.
                         If a gate cannot fail, it tests nothing.

usage:  gate_armh.py <eps> [--plant-infeasible] [--layouts a,b,c]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import armh_constants as AH  # noqa: E402

WORKTREE = Path("/tmp/armh")
STATE = Path("/local/home/zegertho/agent/state/armh/artifacts")
#: qwerty is infeasible on many axes by a wide margin -- the planted fatal case.
PLANTED = "qwertyuiopasdfghjkl'zxcvbnm,.-"


def shipped_analyze(layouts: list[str]) -> dict:
    """Score through the SHIPPED CLI. One call, row-key set asserted (trap 38)."""
    uniq = sorted(set(layouts))
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "8"
    p = subprocess.run(["uv", "run", "--no-sync", "keybo", "analyze", "--json", *uniq],
                       cwd=str(WORKTREE), capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    rows = json.loads(p.stdout)["rows"]
    for lay in uniq:
        assert lay in rows, f"analyze DROPPED {lay!r} (trap 38)"
    return rows


def check(lay: str, row: dict, eps: float) -> dict:
    """Recompute (A13) and (Spd) from the SHIPPED numbers."""
    g = row["gauges"]
    ms = row["time"]["ms_per_char"]
    edge = AH.ARMH_REF_MS + eps
    viol = {}
    strict_better, ties = [], []
    for a in AH.ARMH_CONSTRAINED:
        ex = AH.ARMH_DIR[a] * (g[a] - AH.ARMH_REF[a])
        if ex > AH.ARMH_TOL:
            viol[a] = ex
        elif ex < -AH.ARMH_TOL:
            strict_better.append(a)
        else:
            ties.append(a)
    ox = g[AH.ARMH_TARGET]
    ox_delta = ox - AH.ARMH_REF[AH.ARMH_TARGET]
    return {
        "layout": lay, "ms": ms, "ms_minus_armB": ms - AH.ARMH_REF_MS,
        "band_edge": edge, "speed_ok": ms <= edge + AH.ARMH_TOL,
        "speed_excess": max(0.0, ms - edge),
        "axes_violated": viol, "n_axes_violated": len(viol),
        "axes_ok": len(viol) == 0,
        "axes_strictly_better": strict_better, "axes_tied": ties,
        "oxey": ox, "oxey_minus_armB": ox_delta,
        "oxey_strictly_better": ox_delta < -AH.ARMH_TOL,
        "FEASIBLE": len(viol) == 0 and ms <= edge + AH.ARMH_TOL,
        "COLLECTED": (len(viol) == 0 and ms <= edge + AH.ARMH_TOL
                      and ox_delta < -AH.ARMH_TOL),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    eps = float(sys.argv[1])
    plant = "--plant-infeasible" in sys.argv
    # ⚠ layouts are split on WHITESPACE, never on ',' -- a C30M layout CONTAINS ',' and '.'
    # and '-'. My first version split on ',' and shipped `.-` to the CLI as a layout name.
    explicit = None
    if "--layouts" in sys.argv:
        explicit = sys.argv[sys.argv.index("--layouts") + 1].split()
    for a in sys.argv:
        if a.startswith("--layouts="):
            explicit = a.split("=", 1)[1].split()
    if explicit:
        for lay in explicit:
            assert len(lay) == 30, f"not a 30-char layout: {lay!r} (split on WHITESPACE)"

    if explicit:
        champs = {f"explicit{i}": x for i, x in enumerate(explicit)}
    else:
        summary = json.load(open(STATE / "runs" / "armh-summary.json"))
        champs = {}
        for row in summary.get("phase2_armh", []):
            if row.get("ok"):
                champs[row["tag"]] = row["layout"]
        if not champs:
            print("no ARM H champions to gate")
            return 2
    if plant:
        champs["PLANTED-INFEASIBLE"] = PLANTED

    # always include the two reference layouts as positive/negative controls
    champs["_ref-armB"] = AH.ARMH_LAYOUT_REF
    champs["_ref-BALL1"] = AH.ARMH_BALL1

    rows = shipped_analyze(list(champs.values()))
    out = {"eps": eps, "band_edge": AH.ARMH_REF_MS + eps, "planted": plant,
           "tol": AH.ARMH_TOL, "results": {}}
    bad = []
    for tag, lay in champs.items():
        r = check(lay, rows[lay], eps)
        out["results"][tag] = r
        flag = ("FEASIBLE" if r["FEASIBLE"] else "INFEASIBLE")
        mark = "  COLLECTED" if r["COLLECTED"] else ""
        print(f"{tag:22s} ms={r['ms']:.6f} ({r['ms_minus_armB']:+.6f}) "
              f"oxey={r['oxey']:9.6f} ({r['oxey_minus_armB']:+.6f}) "
              f"{flag} viol={r['n_axes_violated']}{mark}")
        if not r["FEASIBLE"] and not tag.startswith("_ref"):
            bad.append(tag)

    # arm B is the reference: it MUST be feasible with 13 ties. A positive control on the gate.
    ab = out["results"]["_ref-armB"]
    assert ab["axes_ok"] and len(ab["axes_tied"]) == 13, (
        f"GATE BROKEN: arm B is not 13-tied-feasible: {ab}")
    out["gate_positive_control_armB_13_ties"] = True

    dest = STATE / ("gate-armh-PLANTED.json" if plant else "gate-armh.json")
    json.dump(out, open(dest, "w"), indent=1)
    rc = 1 if bad else 0
    print(f"\nWROTE {dest}   infeasible-returned={bad}   rc={rc}")
    if plant:
        print("PLANTED test: rc=1 is the REQUIRED outcome "
              f"(planted layout rejected: {'PLANTED-INFEASIBLE' in bad})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
