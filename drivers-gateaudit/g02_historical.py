"""INVARIANT 4 + NEGATIVE CONTROL: replay the gate against the SHIPPED HISTORICAL artifact.

`agent-artifacts/results_bigram.json` is a real `validate()` report from the campaign (identical
config to my run: seeds [0,1,2], 4 holdouts, wpm [40,140) x20, cell floor 10, n_boot 50). It
carries `calibration_slope` (pooled) and a full `bucket_matrix` per fold x seed, but NO
`calibration_gate` key -- it predates the branch. So it is exactly the counterfactual INVARIANT 4
asks for: feed the gate the numbers that WERE on disk when the ledger wrote "no compression", and
see what it would have said.

NEGATIVE CONTROL (mandatory, runs first): reproduce a published quantity from this artifact and
compare to the ledger. The target is CALIB-1's registered per-fold POOLED calibration range
"0.914-0.999" (ledger 0.3) -- if my reader disagrees with that, my reader is wrong, not the ledger.

Note this artifact carries POOLED slope + per-bucket slopes but NOT `bucket_centered` (the branch
added that slice), so the bucket_centered column here is reported as unavailable rather than
guessed. That absence is itself a finding: it is the slice CALIB-1's headline 1.4067 is measured on.
"""
import json
import statistics

ART = "/local/home/zegertho/repos/keybo/agent-artifacts/results_bigram.json"
OUT = "/tmp/gateaudit/run/g02_historical.json"
BAND = (0.90, 1.10)

d = json.load(open(ART, encoding="utf-8"))
narrow = d["bigram"]["transfer"]["narrow"]

# ---- NEGATIVE CONTROL -----------------------------------------------------------------------
pooled_all = []
for layout, fold in narrow["folds"].items():
    for m in fold["seeds"]:
        pooled_all.append((layout, m["seed"], m["calibration_slope"]))
lo, hi = min(v for _, _, v in pooled_all), max(v for _, _, v in pooled_all)
negctrl = {
    "target": "CALIB-1 ledger 0.3: per-fold pooled calibration slope range 0.914-0.999",
    "measured_min": lo,
    "measured_max": hi,
    "ledger_min": 0.914,
    "ledger_max": 0.999,
    "abs_err_min": abs(lo - 0.914),
    "abs_err_max": abs(hi - 0.999),
    "n_fold_seed_cells": len(pooled_all),
    "all_pooled": [{"layout": a, "seed": b, "pooled": c} for a, b, c in pooled_all],
}
negctrl["passes"] = negctrl["abs_err_min"] < 0.001 and negctrl["abs_err_max"] < 0.001
print("NEGATIVE CONTROL:", json.dumps({k: v for k, v in negctrl.items() if k != "all_pooled"},
                                      indent=2), flush=True)

# ---- the historical slice table, WITH support --------------------------------------------------
rows = []
for layout, fold in narrow["folds"].items():
    for m in fold["seeds"]:
        slices = {"pooled": m["calibration_slope"]}
        support = {"pooled": {"n_cells": fold["n_cells"], "n_participants": None}}
        for b, bm in (m.get("bucket_matrix") or {}).items():
            slices[f"bucket_{b}"] = bm.get("slope")
            support[f"bucket_{b}"] = {
                "n_cells": bm.get("n"),
                "n_participants": bm.get("n_participants"),
            }
        rows.append({"layout": layout, "seed": m["seed"], "slices": slices, "support": support,
                     "bucket_centered": None})  # not recorded pre-branch


def scope(row, keys):
    vals = {k: v for k, v in row["slices"].items() if k in keys
            and v is not None and v == v}  # finite
    oob = sorted(k for k, v in vals.items() if not BAND[0] <= v <= BAND[1])
    return {"n_slices": len(vals), "out_of_band": oob, "passed": (not oob) if vals else None}


ALLK = lambda r: set(r["slices"])                                        # noqa: E731
POOL = lambda r: {"pooled"}                                              # noqa: E731
BUCK = lambda r: {k for k in r["slices"] if k.startswith("bucket_")}     # noqa: E731


def support_gated(row, min_cells):
    """every-slice, but a slice with n_cells < min_cells is EXCLUDED from the verdict."""
    keys = set()
    for k in row["slices"]:
        n = (row["support"].get(k) or {}).get("n_cells")
        if n is None or n >= min_cells:
            keys.add(k)
    return keys


configs = {}
for row in rows:
    key = f"{row['layout']}/seed{row['seed']}"
    configs[key] = {
        "every_slice": scope(row, ALLK(row)),
        "pooled_only": scope(row, POOL(row)),
        "buckets_only": scope(row, BUCK(row)),
        "support_gated_n100": scope(row, support_gated(row, 100)),
        "support_gated_n200": scope(row, support_gated(row, 200)),
        "slices": row["slices"],
        "support": row["support"],
    }

# fold-level roll-up: a fold FAILS if ANY of its seeds fails (the campaign's own "hold across seeds")
folds = {}
for layout in narrow["folds"]:
    sub = {k: v for k, v in configs.items() if k.startswith(layout + "/")}
    folds[layout] = {}
    for cfg in ("every_slice", "pooled_only", "buckets_only", "support_gated_n100",
                "support_gated_n200"):
        passed = [v[cfg]["passed"] for v in sub.values()]
        oob = sorted({s for v in sub.values() for s in v[cfg]["out_of_band"]})
        folds[layout][cfg] = {
            "passed_all_seeds": all(p is True for p in passed),
            "per_seed_passed": passed,
            "out_of_band_union": oob,
        }
    folds[layout]["pooled_by_seed"] = [v["slices"]["pooled"] for v in sub.values()]
    folds[layout]["pooled_mean"] = statistics.fmean(folds[layout]["pooled_by_seed"])

out = {"artifact": ART, "band": list(BAND), "negative_control": negctrl,
       "config": narrow["config"], "per_fold_seed": configs, "fold_rollup": folds}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print("WROTE", OUT, flush=True)

for layout, r in sorted(folds.items()):
    print(f"{layout:8s} pooled={r['pooled_mean']:.4f} "
          f"every={r['every_slice']['passed_all_seeds']!s:5s} "
          f"pooled_only={r['pooled_only']['passed_all_seeds']!s:5s} "
          f"n200={r['support_gated_n200']['passed_all_seeds']!s:5s} "
          f"oob={r['every_slice']['out_of_band_union']}", flush=True)
