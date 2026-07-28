"""ARM H — GENERATE every constant the prereg and the objective need, from LIVE code.

Nothing in this file is hand-typed from another agent's artifact. ARM G had all 14 of its
scale constants hand-transcribed wrong by ~1e-5 and an INVENTED layout string that existed
nowhere; it then hypothesised BLAS batch-shape dependence and measured it (refuted by ~10
orders) when the cause was its typing. So: GENERATE OR ASSERT, NEVER RETYPE.

The six frozen 1M champions are read from the ORIGINAL artifact JSON, not from ARM G's
transcription of it, and their `ms/char` values are re-derived here rather than trusted.

Writes prereg-inputs.json. Run BEFORE the prereg is written; the prereg quotes this file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import evobj as EV  # noqa: E402

WORKTREE = Path("/tmp/armh")
#: the ORIGINAL artifact SPEEDTIE-1 read; ARM G's ARMG_SIX is a transcription of it.
PLACEBO = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
               "optevidence-1/search-noise-placebo.json")
ARMB = "flmpg-yuo,sntdcireahkxbwv'.jzq"
#: frozen arm B ms/char (SPEEDTIE-1 / ARM-B). Re-derived below; the assert is the check.
ARMB_MS_FROZEN = 253.90057910352604

#: the 14 live gauges = the shipped frame minus `sfr` (a PERMUTATION INVARIANT, trap 23).
#: ARM H minimises `oxey-style` and hard-constrains the other 13.
TARGET = "oxey-style"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - provenance tag only


def main() -> int:
    out: dict = {}

    # ---- C1 worktree isolation: a POSITIVE control (trap 35), not "no hardcodes found" ----
    import keybo
    fe = EV.FastEval(corpus=None, weights_json=None, with_surface=True)
    iso = {
        "keybo.__file__": keybo.__file__,
        "sys.prefix": sys.prefix,
        "FastEval.corpus_dir": str(Path(fe.corpus_dir).resolve()),
        "trigrams_md5": md5(Path(fe.corpus_dir) / "trigrams.txt"),
    }
    for k in ("keybo.__file__", "sys.prefix", "FastEval.corpus_dir"):
        assert iso[k].startswith("/tmp/armh"), f"{k} ESCAPED the worktree: {iso[k]}"
    # trap-8 reference md5 for blend-v1 trigrams
    assert iso["trigrams_md5"] == "c5066fa7bcc46dea1ecbc987fb465b4a", iso["trigrams_md5"]
    out["C1_isolation"] = iso

    # ---- the 14 live gauges, READ from live code ----
    from keybo.analysis.evidence_scorer import EXPECTED_SIGN, LIVE_GAUGES
    live = list(LIVE_GAUGES)
    assert TARGET in live, live
    assert "sfr" not in live, live
    assert len(live) == 14, live
    out["live_gauges"] = live
    out["constrained_axes"] = [g for g in live if g != TARGET]
    assert len(out["constrained_axes"]) == 13, out["constrained_axes"]

    # ---- directions DERIVED two ways, never assumed (trap 5) ----
    rng = np.random.default_rng(20260728)
    pool = np.stack([np.concatenate([rng.permutation(30), [30]]).astype(np.int32)
                     for _ in range(4000)])
    gp = fe.gauges(pool)
    ms = gp["_ms_per_char"]
    from scipy.stats import spearmanr
    qwerty = fe.gauges(np.stack([EV.perm_of(EV.C30M)]))
    armb_g = fe.gauges(np.stack([EV.perm_of(ARMB)]))
    dirs: dict[str, float] = {}
    derive = {}
    for g in live:
        rho = float(spearmanr(gp[g], ms).statistic)
        # (a) rank-correlation with predicted ms/char: +1 if higher gauge => slower
        d_rho = 1.0 if rho > 0 else -1.0
        # (b) qwerty-is-worst reference point: qwerty should be on the BAD side of arm B
        d_qw = 1.0 if float(qwerty[g][0]) > float(armb_g[g][0]) else -1.0
        d_tbl = float(EXPECTED_SIGN[g])
        dirs[g] = d_tbl
        derive[g] = {"rho_with_ms": rho, "dir_from_rho": d_rho, "dir_from_qwerty": d_qw,
                     "dir_shipped_EXPECTED_SIGN": d_tbl,
                     "rho_agrees": d_rho == d_tbl, "qwerty_agrees": d_qw == d_tbl}
    out["directions"] = dirs
    out["direction_derivation"] = derive
    out["direction_agreement"] = {
        "rho_n_agree": sum(1 for v in derive.values() if v["rho_agrees"]),
        "qwerty_n_agree": sum(1 for v in derive.values() if v["qwerty_agrees"]),
        "n": len(live),
    }

    # ---- the six frozen 1M champions, READ FROM THE ORIGINAL ARTIFACT ----
    blob = json.load(open(PLACEBO))
    six = [r["layout"] for r in blob["runs"]["baseline"]]
    assert len(six) == 6 and len(set(six)) == 6, six
    assert ARMB in six, "arm B must be one of the six frozen champions"
    out["six_source"] = {
        "path": str(PLACEBO), "md5": md5(PLACEBO),
        "borrowed_sd_in_artifact": blob["bands"]["baseline"]["ms_per_char"]["sd"],
        "borrowed_range_in_artifact": blob["bands"]["baseline"]["ms_per_char"]["range"],
        "artifact_seeds": [r["seed"] for r in blob["runs"]["baseline"]],
        "artifact_unique_evals": [r["unique_evals"] for r in blob["runs"]["baseline"]],
    }
    out["six_frozen_champions"] = six

    g6 = fe.gauges(np.stack([EV.perm_of(x) for x in six]))
    out["armB_ms_rederived"] = float(armb_g["_ms_per_char"][0])
    out["armB_ms_frozen"] = ARMB_MS_FROZEN
    out["armB_ms_absdiff"] = abs(out["armB_ms_rederived"] - ARMB_MS_FROZEN)
    assert out["armB_ms_absdiff"] < 1e-9, out["armB_ms_absdiff"]

    out["armB_gauges"] = {g: float(armb_g[g][0]) for g in live}
    out["scale_range_over_six"] = {g: float(g6[g].max() - g6[g].min()) for g in live}
    out["six_ms"] = [float(x) for x in g6["_ms_per_char"]]
    out["six_oxey"] = [float(x) for x in g6[TARGET]]
    out["six_ms_sd_ddof1"] = float(np.std(g6["_ms_per_char"], ddof=1))
    out["six_ms_range"] = float(g6["_ms_per_char"].max() - g6["_ms_per_char"].min())
    out["six_oxey_ratio"] = float(g6[TARGET].max() / g6[TARGET].min())

    # ---- HOW HARD IS THE FEASIBLE SET? per-axis satisfaction of the 13 constraints by the
    # six frozen champions + the incumbents. This is a PREREG INPUT (frozen data only, no
    # ARM H result exists), and it is what makes EMPTY a pre-registered live outcome.
    refs = {"armB": ARMB}
    for i, lay in enumerate(six):
        if lay != ARMB:
            refs[f"six-s{blob['runs']['baseline'][i]['seed']}"] = lay
    # incumbents: layout STRINGS read from a durable artifact, then RE-SCORED here (their
    # gauge values are never transcribed). Plus the shipped registry, read from live code.
    armg_in = json.load(open("/local/home/zegertho/agent/state/armg/artifacts/"
                             "D-prereg-input.json"))
    for name in ("arm-A", "keybo-lsb", "keybo-lsb+lm", "flagship-c3"):
        refs[name] = armg_in["D_of_existing"][name]["layout"]
    from keybo.layouts import NAMED_LAYOUTS
    for name, lay in NAMED_LAYOUTS.items():
        if set(lay) == set(EV.C30M):
            refs[name] = lay
    out["incumbent_source"] = {
        "strings_from": "state/armg/artifacts/D-prereg-input.json (+ live NAMED_LAYOUTS)",
        "note": "layout STRINGS borrowed; every gauge value re-derived here.",
    }
    feas_table = {}
    for name, lay in refs.items():
        gg = fe.gauges(np.stack([EV.perm_of(lay)]))
        sat, viol = [], {}
        for g in out["constrained_axes"]:
            excess = dirs[g] * (float(gg[g][0]) - out["armB_gauges"][g])
            if excess <= 1e-9:
                sat.append(g)
            else:
                viol[g] = excess
        feas_table[name] = {
            "layout": lay, "ms": float(gg["_ms_per_char"][0]),
            "oxey": float(gg[TARGET][0]),
            "n_sat_of_13": len(sat), "violated": viol,
        }
    out["frozen_feasibility_table"] = feas_table
    # which axes are violated by EVERY non-armB reference => candidate binding constraints
    others = [v for k, v in feas_table.items() if k != "armB"]
    all_viol = [set(v["violated"]) for v in others]
    out["axes_violated_by_all_six_siblings"] = sorted(set.intersection(*all_viol)) if all_viol else []
    out["axes_violated_by_any_sibling"] = sorted(set.union(*all_viol)) if all_viol else []
    out["armB_best_of_six_on"] = sorted(
        g for g in live
        if (float(g6[g].min()) == float(armb_g[g][0]) if dirs[g] > 0
            else float(g6[g].max()) == float(armb_g[g][0])))

    # ---- oxey-style vs the six axes it restates (trap 27) — the structural tension ----
    # ARM H minimises oxey while constraining 6 of oxey's own components. Quantify how much
    # of oxey's variance those components explain IN THE POOL ARM H WILL SEARCH.
    comps = ["sfb", "lsb", "scissor", "imbalance", "redir", "alt"]
    band = ms <= float(np.percentile(ms, 2.0))   # the fastest 2% of the random pool
    X = np.stack([gp[c][band] for c in comps], axis=1)
    y = gp[TARGET][band]
    Xd = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    out["oxey_restates_components"] = {
        "components": comps,
        "R2_in_fastest_2pct_of_4000_random": float(1 - resid.var() / y.var()),
        "n": int(band.sum()),
    }

    # ---- seeds: PRE-REGISTERED formula, verified DISJOINT from prior families ----
    seeds = [31337 + 104729 * r for r in range(5)]
    armg_family = {20_260_728 + 7919 * r for r in range(20)}
    placebo_family = {900_000 + 7919 * r for r in range(20)}
    assert not (set(seeds) & armg_family), "seed family collides with ARM G's"
    assert not (set(seeds) & placebo_family), "seed family collides with the placebo's"
    out["seeds"] = {"formula": "31337 + 104729*r, r=0..4", "values": seeds,
                    "disjoint_from_armg_family": True,
                    "disjoint_from_placebo_family": True,
                    "repro_control_seed": 20_260_728,
                    "repro_control_note": (
                        "ONE extra baseline run at ARM G's r=0 seed, as a bit-exact "
                        "reproduction control against ARM G's own baseline-r0 champion. "
                        "It is NOT in the sd_H pool (that would share a draw with ARM G)."),
                    }

    json.dump(out, open(HERE / "prereg-inputs.json", "w"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("direction_derivation", "frozen_feasibility_table")},
                     indent=1, sort_keys=True))
    print("\n--- frozen feasibility table (the bar ARM H must clear) ---")
    for name, v in sorted(feas_table.items(), key=lambda kv: -kv[1]["n_sat_of_13"]):
        print(f"  {name:6s} ms={v['ms']:.4f} oxey={v['oxey']:9.4f} "
              f"sat {v['n_sat_of_13']:2d}/13  violates: "
              + ", ".join(f"{g}+{e:.4g}" for g, e in sorted(v["violated"].items())))
    print(f"\nWROTE {HERE / 'prereg-inputs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
