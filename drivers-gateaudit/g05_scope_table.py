"""INVARIANT 2: the scope comparison table, from MY OWN LOLO (g01_scope.json).

Applies every candidate `passed` rule POST-HOC to one expensive run, so no configuration can be
accused of having been measured on its own private run. Reports, per fold:
  * which seeds pass and whether the fold passes on ALL seeds (the campaign's own "hold across
    seeds" convention)
  * the SUPPORT (n_cells, n_participants) behind every failing slice
  * the MEASURED false-flag probability for that slice's n, from g03's floor table, so a failure
    can be read against the noise it has to clear.

Candidate scopes:
  every_slice          -- as written on the branch
  bucket_centered_only -- the parent's hypothesis
  pooled_only          -- ruled out on constructed grounds in g04, included for completeness
  structural_pair      -- {pooled, bucket_centered}  <- my candidate
  support_gated_nXXX   -- every slice, but slices with n_cells < XXX are excluded from `passed`
  buckets_only
"""
import json
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

SRC = "/tmp/gateaudit/run/g01_scope.json"
FLOOR = "/tmp/gateaudit/run/g03_estimand.json"
OUT = "/tmp/gateaudit/run/g05_scope_table.json"
BAND = (0.90, 1.10)
R_REPO = 0.657889   # the repo's own pooled correlation (CALIB-1 ledger:11925)

d = json.load(open(SRC, encoding="utf-8"))
floor_tbl = json.load(open(FLOOR, encoding="utf-8"))["E_false_flag_thin_slice_noise"]


def false_flag_rate(n_cells, r=R_REPO):
    """Interpolate the MEASURED floor at this n (log-linear in n between measured grid points)."""
    if n_cells is None:
        return None
    pts = sorted((int(k.split("n=")[1]), v["false_flag_rate"])
                 for k, v in floor_tbl.items() if k.startswith(f"r={r}/"))
    if n_cells <= pts[0][0]:
        return pts[0][1]
    if n_cells >= pts[-1][0]:
        return pts[-1][1]
    for (n0, p0), (n1, p1) in zip(pts, pts[1:]):
        if n0 <= n_cells <= n1:
            w = (math.log(n_cells) - math.log(n0)) / (math.log(n1) - math.log(n0))
            return p0 + w * (p1 - p0)
    return None


def analytic_sd(n_cells, r=R_REPO):
    if not n_cells or n_cells < 3:
        return None
    return math.sqrt((1 - r * r) / (r * r)) / math.sqrt(n_cells - 2)


# ---------- gather every (fold, seed) gate block ------------------------------------------------
cells = []
for layout, fold in d["folds"].items():
    for s in fold["seeds"]:
        g = s["calibration_gate"]
        slopes = {k: v for k, v in g["slopes"].items()}
        support = g.get("support") or {}
        cells.append({"layout": layout, "seed": s["seed"], "slopes": slopes, "support": support,
                      "gate_passed_as_written": g["passed"],
                      "gate_out_of_band_as_written": g["out_of_band"],
                      "n_cells_fold": fold["n_cells"]})

BUCKETS = sorted({k for c in cells for k in c["slopes"] if k.startswith("bucket_")
                  and k != "bucket_centered"})


def keys_for(scope, cell):
    sl = cell["slopes"]
    buckets = {k for k in sl if k.startswith("bucket_") and k != "bucket_centered"}
    if scope == "every_slice":
        return set(sl)
    if scope == "bucket_centered_only":
        return {"bucket_centered"} & set(sl)
    if scope == "pooled_only":
        return {"pooled"} & set(sl)
    if scope == "structural_pair":
        return {"pooled", "bucket_centered"} & set(sl)
    if scope == "buckets_only":
        return buckets
    if scope.startswith("support_gated_n"):
        floor_n = int(scope.split("_n")[1])
        keep = set()
        for k in sl:
            n = ((cell["support"] or {}).get(k) or {}).get("n_cells")
            if n is None or n >= floor_n:
                keep.add(k)
        return keep
    raise ValueError(scope)


SCOPES = ["every_slice", "bucket_centered_only", "pooled_only", "structural_pair", "buckets_only",
          "support_gated_n100", "support_gated_n200", "support_gated_n400"]

per_cell = {}
for c in cells:
    key = f"{c['layout']}/seed{c['seed']}"
    entry = {"slopes": c["slopes"], "support": c["support"],
             "as_written_passed": c["gate_passed_as_written"]}
    for scope in SCOPES:
        ks = keys_for(scope, c)
        vals = {k: c["slopes"][k] for k in ks
                if c["slopes"].get(k) is not None and c["slopes"][k] == c["slopes"][k]}
        oob = sorted(k for k, v in vals.items() if not BAND[0] <= v <= BAND[1])
        entry[scope] = {"n_slices": len(vals), "out_of_band": oob,
                        "passed": (not oob) if vals else None}
    per_cell[key] = entry

folds = {}
for layout in d["folds"]:
    sub = {k: v for k, v in per_cell.items() if k.startswith(layout + "/")}
    folds[layout] = {"n_cells": d["folds"][layout]["n_cells"],
                     "ceiling": d["folds"][layout]["ceiling"]}
    for scope in SCOPES:
        passed = [v[scope]["passed"] for v in sub.values()]
        oob = sorted({s for v in sub.values() for s in v[scope]["out_of_band"]})
        folds[layout][scope] = {"passed_all_seeds": all(p is True for p in passed),
                                "per_seed": passed, "out_of_band_union": oob}
    for slice_name in ("pooled", "bucket_centered"):
        vals = [v["slopes"].get(slice_name) for v in sub.values()]
        folds[layout][f"{slice_name}_by_seed"] = vals
        folds[layout][f"{slice_name}_mean"] = sum(vals) / len(vals) if vals else None

# ---------- the SUPPORT + noise table for every slice ------------------------------------------
support_tbl = {}
for c in cells:
    for k, v in c["slopes"].items():
        n = ((c["support"] or {}).get(k) or {}).get("n_cells")
        npart = ((c["support"] or {}).get(k) or {}).get("n_participants")
        support_tbl.setdefault(f"{c['layout']}/{k}", []).append(
            {"seed": c["seed"], "slope": v, "n_cells": n, "n_participants": npart,
             "in_band": BAND[0] <= v <= BAND[1] if v == v else None,
             "measured_false_flag_rate_at_this_n": false_flag_rate(n),
             "analytic_slope_sd_at_this_n": analytic_sd(n),
             "deviation_in_sds": (abs(v - 1.0) / analytic_sd(n))
                                 if (analytic_sd(n) and v == v) else None})

summary = {}
for scope in SCOPES:
    summary[scope] = {
        "folds_passing": sorted(L for L in folds if folds[L][scope]["passed_all_seeds"]),
        "folds_failing": sorted(L for L in folds if not folds[L][scope]["passed_all_seeds"]),
        "n_pass": sum(1 for L in folds if folds[L][scope]["passed_all_seeds"]),
    }

out = {"provenance": d["provenance"], "config": d["config"], "band": list(BAND),
       "r_used_for_floor": R_REPO, "per_fold_seed": per_cell, "fold_rollup": folds,
       "scope_summary": summary, "support_and_noise": support_tbl}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)

print(f"{'scope':<24} {'#pass':>5}  passing folds / failing folds")
for scope in SCOPES:
    s = summary[scope]
    print(f"{scope:<24} {s['n_pass']:>5}  {','.join(s['folds_passing']) or '-'}  /  "
          f"{','.join(s['folds_failing']) or '-'}")
print()
for L in sorted(folds):
    f_ = folds[L]
    print(f"{L:8s} pooled={f_['pooled_mean']:.4f} bucket_centered={f_['bucket_centered_mean']:.4f} "
          f"n_cells={f_['n_cells']}")
print("\nWROTE", OUT)
