"""DIRECTION refit: does a real direction-of-travel channel change what we optimize for?

Three arms per surface, and the middle one is the whole reason this is trustworthy:

    v1       the shipped 20-column vector (the anchor)
    PLACEBO  v1 + 9 columns carrying NO new information (TOOLING-TRAPS #17)
    v2       v1 + the 9 real direction columns

Why the placebo. Going v1 -> v2 changes TWO things: the direction information is added AND
the feature frame grows by 9 columns. A wider frame changes XGBoost's colsample_bytree
draws, its split search, and its effective regularization, so ANY v1->v2 delta is
unattributable on its own. The placebo carries the same 9 extra columns built from
quantities the v1 vector ALREADY determines (origin row one-hot, signed_dy, o_lateral, and
copies) — so it is exactly as wide, exactly as redundant, and carries zero new information.
The direction effect is read as PLACEBO -> v2, never as v1 -> v2.

Per TOOLING-TRAPS #17 the placebo's axis is deliberately NESTED in (determined by) the real
frame's information, which is the conservative choice: it shares structure with v2, so it
understates v2's marginal effect. An "inert" verdict therefore survives the bias.

Reported per surface (AALTO / COMMUNITY / POOL — POOL is NOT independent, it contains both
others) and per seed. Metrics are the campaign's registered LOLO gate: layout-ranking tau
(the decisive one), rho/ceiling, umae, wmae.

Outputs: runs/direction_refit.json + a printed table. NOTHING is published.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WT = "/local/home/zegertho/agent/state/direction/wt-direction"
sys.path.insert(0, f"{WT}/src")

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

E2E = Path("/local/home/zegertho/keybo-e2e")
COMM = Path("/local/home/zegertho/repos/keybo/data/community/processed")
OUT = Path("/local/home/zegertho/agent/state/direction/artifacts")

#: The four rowStagger community labels POOL-1 registered (pool_train.py LABELS).
COMM_LABELS = [
    "colemak@rowStagger#alite",
    "mtgap-variant@rowStagger#richarddavison",
    "custom-d42a1f92@rowStagger#ddn",
    "custom-aa426873@rowStagger#vg",
]

#: Production bigram recipe (REG-LOLO, 2026-07-13). Re-derived from reg_lolo.log's winner.
REG_LOLO = dict(
    colsample_bytree=0.7,
    gamma=0.957,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=4,
    n_estimators=300,
    reg_alpha=0.141,
    reg_lambda=0.011,
    subsample=0.7,
    verbosity=0,
)

SEEDS = [0, 1, 2]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def load_surface(name: str):
    """Stroke rows per surface. POOL = AALTO + COMMUNITY (so it is NOT independent)."""
    if name == "AALTO":
        return load_strokes(str(E2E / "bistrokes_v5.tsv"), ngram_len=2, wpm_threshold=0, min_samples=1)
    comm = load_strokes(
        str(COMM / "bistrokes_community.tsv"), ngram_len=2, wpm_threshold=0, min_samples=1
    )
    comm = [r for r in comm if r.layout in COMM_LABELS]
    if name == "COMMUNITY":
        return comm
    if name == "POOL":
        aalto = load_strokes(
            str(E2E / "bistrokes_v5.tsv"), ngram_len=2, wpm_threshold=0, min_samples=1
        )
        return aalto + comm
    raise ValueError(name)


def summarize(report: dict) -> dict:
    """Collapse a LOLO report to the registered gate metrics."""
    fracs, umaes, wmaes, rhos = [], [], [], []
    per_fold = {}
    for layout, fold in report["folds"].items():
        f = [m["rho_frac_ceiling"] for m in fold["seeds"] if m["rho_frac_ceiling"] is not None]
        per_fold[layout] = {
            "rho_frac_ceiling": float(np.mean(f)) if f else None,
            "umae": float(np.mean([m["umae"] for m in fold["seeds"]])),
            "wmae": float(np.mean([m["wmae"] for m in fold["seeds"]])),
            "rho": float(np.mean([m["rho"] for m in fold["seeds"]])),
            "ceiling": fold["seeds"][0]["ceiling"],
            "n_cells": fold["n_cells"],
        }
        fracs += f
        umaes += [m["umae"] for m in fold["seeds"]]
        wmaes += [m["wmae"] for m in fold["seeds"]]
        rhos += [m["rho"] for m in fold["seeds"]]
    taus = [p["tau_heldout"] for p in report["pooled"]]
    return {
        "rho_frac_ceiling": float(np.mean(fracs)) if fracs else None,
        "rho": float(np.mean(rhos)),
        "umae": float(np.mean(umaes)),
        "wmae": float(np.mean(wmaes)),
        "taus": taus,
        "tau_min": float(min(taus)),
        "tau_mean": float(np.mean(taus)),
        "per_fold": per_fold,
    }


def main() -> None:
    surfaces = sys.argv[1].split(",") if len(sys.argv) > 1 else ["AALTO", "COMMUNITY", "POOL"]
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["v1", "placebo", "v2"]
    n_jobs = int(os.environ.get("KEYBO_NJOBS", "8"))
    out_path = OUT / (os.environ.get("KEYBO_OUT") or "direction_refit.json")

    results: dict = {
        "meta": {
            "rule": "DIRECTION refit — v1 vs same-width PLACEBO vs v2 (direction)",
            "recipe": "REG_LOLO (production bigram), LOGRAT, ROW_STAGGERED_30",
            "seeds": SEEDS,
            "params": REG_LOLO,
            "arms": arms,
            "surfaces": surfaces,
            "pool_note": "POOL is NOT independent of AALTO/COMMUNITY — it contains both.",
            "placebo_note": (
                "PLACEBO adds 9 columns determined by the v1 vector (origin row one-hot, "
                "signed_dy, o_lateral + copies): same width, zero new information. Read the "
                "direction effect as PLACEBO->v2, never v1->v2 (TOOLING-TRAPS #17)."
            ),
        },
        "surfaces": {},
    }

    for name in surfaces:
        rows = load_surface(name)
        n_samples = sum(len(r.samples) for r in rows)
        layouts = sorted({r.layout for r in rows})
        log(f"{name}: {len(rows)} rows, {n_samples} samples, {len(layouts)} layouts")
        # Participants per layout: the split-half noise ceiling BISECTS participants, so a
        # layout typed by ONE participant has no ceiling at all (nan, hence rho/ceiling
        # None). That is a structural property of the source, not a missing computation —
        # record it so the n/a in the table is explainable (TOOLING-TRAPS #19).
        pids_per_layout = {
            la: len({s[2] for r in rows if r.layout == la for s in r.samples}) for la in layouts
        }
        results["surfaces"][name] = {
            "n_rows": len(rows),
            "n_samples": n_samples,
            "layouts": layouts,
            "participants_per_layout": pids_per_layout,
            "arms": {},
        }
        log(f"  participants/layout: {pids_per_layout}")
        for arm in arms:
            kw = {}
            if arm == "v2":
                kw["direction"] = True
            elif arm == "placebo":
                kw["placebo"] = True
            t = time.time()
            report = validate(
                rows,
                seeds=SEEDS,
                ngram="bigram",
                n_boot=10,
                geometry=ROW_STAGGERED_30,
                train_params={**REG_LOLO, "n_jobs": n_jobs},
                **kw,
            )
            s = summarize(report)
            s["seconds"] = round(time.time() - t, 1)
            results["surfaces"][name]["arms"][arm] = s
            frac = s["rho_frac_ceiling"]
            frac_s = f"{frac:.4f}" if frac is not None else "n/a(1 participant)"
            log(
                f"  {name}/{arm}: rho/ceiling {frac_s} rho {s['rho']:.4f} "
                f"umae {s['umae']:.3f} wmae {s['wmae']:.3f} "
                f"taus {['%.3f' % t for t in s['taus']]} ({s['seconds']}s)"
            )
            out_path.write_text(json.dumps(results, indent=1, default=float))

    out_path.write_text(json.dumps(results, indent=1, default=float))
    log(f"wrote {out_path}")

    # --- the table -----------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("LOLO per surface — v1 anchor, same-width PLACEBO, v2 direction (3 seeds, REG_LOLO)")
    print("=" * 96)
    hdr = (
        f"{'surface':10s} {'arm':8s} {'rho/ceil':>9s} {'rho':>8s} {'umae':>8s} "
        f"{'wmae':>8s} {'tau_min':>8s} {'tau_mean':>9s}"
    )
    print(hdr)
    for name in surfaces:
        for arm in arms:
            a = results["surfaces"][name]["arms"].get(arm)
            if not a:
                continue
            frac = a["rho_frac_ceiling"]
            frac_s = f"{frac:9.4f}" if frac is not None else f"{'n/a':>9s}"
            print(
                f"{name:10s} {arm:8s} {frac_s} {a['rho']:8.4f} {a['umae']:8.3f} "
                f"{a['wmae']:8.3f} {a['tau_min']:8.3f} {a['tau_mean']:9.3f}"
            )
        arms_d = results["surfaces"][name]["arms"]
        def _delta(hi, lo, label, note):
            d_frac = (
                f"{hi['rho_frac_ceiling'] - lo['rho_frac_ceiling']:+9.4f}"
                if hi["rho_frac_ceiling"] is not None and lo["rho_frac_ceiling"] is not None
                else f"{'n/a':>9s}"
            )
            print(
                f"{'':10s} {label:8s} {d_frac} {hi['rho'] - lo['rho']:+8.4f} "
                f"{hi['umae'] - lo['umae']:+8.3f} {hi['wmae'] - lo['wmae']:+8.3f}   <- {note}"
            )

        if "placebo" in arms_d and "v2" in arms_d:
            _delta(
                arms_d["v2"],
                arms_d["placebo"],
                "DELTA",
                "placebo->v2 (the ATTRIBUTABLE direction effect)",
            )
        if "v1" in arms_d and "placebo" in arms_d:
            _delta(
                arms_d["placebo"],
                arms_d["v1"],
                "(width)",
                "v1->placebo (frame-width artifact ONLY, no information added)",
            )
    print("ALL-DONE")


if __name__ == "__main__":
    main()
