"""INVARIANT 4: would the gate have CAUGHT the thing it exists for, at each candidate scope?

Three separate questions, kept separate because they have different answers:

  Q1. Would it have flagged the qwerty-fold compression BEFORE a human noticed? Tested against the
      SHIPPED HISTORICAL artifact (`agent-artifacts/results_bigram.json`, dated 2026-07-31), which
      is what was on disk during the campaign. That artifact has NO `bucket_centered` key -- the
      branch added it -- so for the historical counterfactual only `pooled` and the per-bucket
      slopes were available. That is the honest test: could the gate have fired with the numbers
      that EXISTED?

  Q2. Would it have caught the ledger's MISREADINGS? A gate emits a machine-readable verdict; a
      misreading is a prose claim. Test what the artifact would have contained, and whether the
      specific false sentences would have been contradicted BY THE ARTIFACT ITSELF.

  Q3. The false-flag budget over the campaign. If the gate had been live at every scope, how many
      fold-seed FAILURES would it have produced, and what fraction are attributable to noise at the
      measured floor? A gate that cries wolf on 11 of 12 cells trains its readers to ignore it --
      that is a cost, and it is quantifiable.
"""
import json
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

HIST = "/local/home/zegertho/repos/keybo/agent-artifacts/results_bigram.json"
MINE = "/tmp/gateaudit/run/g01_scope.json"
OUT = "/tmp/gateaudit/run/g06_invariant4.json"
BAND = (0.90, 1.10)
R = 0.657889

res = {}

# ---------------- Q1: the historical counterfactual ---------------------------------------------
hist = json.load(open(HIST, encoding="utf-8"))["bigram"]["transfer"]["narrow"]
q1 = {"artifact": HIST, "artifact_has_bucket_centered": False, "folds": {}}
for layout, fold in hist["folds"].items():
    per_seed = []
    for m in fold["seeds"]:
        sl = {"pooled": m["calibration_slope"]}
        sup = {"pooled": {"n_cells": fold["n_cells"]}}
        for b, bm in (m.get("bucket_matrix") or {}).items():
            sl[f"bucket_{b}"] = bm.get("slope")
            sup[f"bucket_{b}"] = {"n_cells": bm.get("n"),
                                  "n_participants": bm.get("n_participants")}
        per_seed.append({"seed": m["seed"], "slopes": sl, "support": sup})
    q1["folds"][layout] = per_seed


def oob(slopes, keys):
    vals = {k: v for k, v in slopes.items() if k in keys and v is not None and v == v}
    return sorted(k for k, v in vals.items() if not BAND[0] <= v <= BAND[1]), len(vals)


HIST_SCOPES = {
    "every_available_slice": lambda sl: set(sl),
    "pooled_only": lambda sl: {"pooled"},
    "buckets_only": lambda sl: {k for k in sl if k.startswith("bucket_")},
    "support_gated_n400": None,  # handled specially, needs support
}
q1_verdicts = {}
for layout, seeds in q1["folds"].items():
    q1_verdicts[layout] = {}
    for name, fn in HIST_SCOPES.items():
        fails = []
        for s in seeds:
            if name == "support_gated_n400":
                keys = {k for k in s["slopes"]
                        if (s["support"].get(k) or {}).get("n_cells", 10**9) >= 400}
            else:
                keys = fn(s["slopes"])
            o, n = oob(s["slopes"], keys)
            fails.append({"seed": s["seed"], "out_of_band": o, "n_slices": n,
                          "passed": (not o) if n else None})
        q1_verdicts[layout][name] = {
            "passed_all_seeds": all(f["passed"] is True for f in fails),
            "per_seed": fails,
        }
res["Q1_historical_counterfactual"] = {
    "note": "the shipped artifact predates the bucket_centered slice, so only pooled + per-bucket "
            "slopes existed. This is what the gate COULD have fired on at the time.",
    "verdicts": q1_verdicts,
    "qwerty_pooled_by_seed": [s["slopes"]["pooled"] for s in q1["folds"]["qwerty"]],
}

# ---------------- Q2: would the artifact have contradicted the prose? ----------------------------
mine = json.load(open(MINE, encoding="utf-8"))
qwerty_bc = [s["calibration_gate"]["slopes"]["bucket_centered"]
             for s in mine["folds"]["qwerty"]["seeds"]]
qwerty_pooled = [s["calibration_gate"]["slopes"]["pooled"]
                 for s in mine["folds"]["qwerty"]["seeds"]]
all_pooled = {L: [s["calibration_gate"]["slopes"]["pooled"] for s in f["seeds"]]
              for L, f in mine["folds"].items()}
res["Q2_would_it_have_caught_the_misreadings"] = {
    "ledger_98": {
        "claim": "calibration slope 1.04 on qwerty (no compression)",
        "is_about": "the TRIGRAM model (runs/lolo_trigram_v1.json), a different artifact/model",
        "bigram_qwerty_pooled_measured": qwerty_pooled,
        "bigram_qwerty_bucket_centered_measured": qwerty_bc,
        "would_artifact_contradict_it": "NOT DIRECTLY -- it is a trigram claim and I did not "
                                        "re-run the trigram LOLO. But the gate emits passed=False "
                                        "per fold-seed, so the WORD 'no compression' could not have "
                                        "been written beside a passed=False block without the "
                                        "contradiction being visible in the same file.",
    },
    "ledger_356": {
        "claim": "both champions hold tau +1.0 with calibration slopes ~1.0 per fold",
        "measured_pooled_per_fold": {k: sum(v) / len(v) for k, v in all_pooled.items()},
        "would_artifact_contradict_it": "YES for the qwerty fold at pooled 1.2283 and "
                                       "bucket_centered 1.4067; '~1.0 per fold' is false of 1 of 4 "
                                       "folds and the artifact would say passed=False on it.",
    },
    "ledger_11919_and_12013": {
        "claim": "per-fold pooled 0.914-0.999 ... the surface does not compress",
        "measured_full_range_over_12_cells": [
            min(v for vs in all_pooled.values() for v in vs),
            max(v for vs in all_pooled.values() for v in vs)],
        "measured_range_excluding_qwerty": [
            min(v for L, vs in all_pooled.items() if L != "qwerty" for v in vs),
            max(v for L, vs in all_pooled.items() if L != "qwerty" for v in vs)],
        "would_artifact_contradict_it": "YES, decisively. An always-emitted per-fold gate block "
                                        "makes fold-omission visible: you cannot quote a range over "
                                        "3 of 4 folds when the 4th fold's block sits in the same "
                                        "artifact with passed=False.",
    },
}

# ---------------- Q3: the false-flag budget over the campaign ------------------------------------
floor = json.load(open("/tmp/gateaudit/run/g03_estimand.json",
                      encoding="utf-8"))["E_false_flag_thin_slice_noise"]
pts = sorted((int(k.split("n=")[1]), v["false_flag_rate"])
             for k, v in floor.items() if k.startswith(f"r={R}/"))


def ff(n):
    if n is None:
        return None
    if n <= pts[0][0]:
        return pts[0][1]
    if n >= pts[-1][0]:
        return pts[-1][1]
    for (n0, p0), (n1, p1) in zip(pts, pts[1:]):
        if n0 <= n <= n1:
            w = (math.log(n) - math.log(n0)) / (math.log(n1) - math.log(n0))
            return p0 + w * (p1 - p0)
    return None


def sd_at(n):
    return math.sqrt((1 - R * R) / (R * R)) / math.sqrt(n - 2) if n and n > 2 else None


budget = {}
for scope, keyfn in (
    ("every_slice", lambda sl, sup: set(sl)),
    ("bucket_centered_only", lambda sl, sup: {"bucket_centered"}),
    ("pooled_only", lambda sl, sup: {"pooled"}),
    ("structural_pair", lambda sl, sup: {"pooled", "bucket_centered"}),
    ("support_gated_n400", lambda sl, sup: {k for k in sl
                                            if (sup.get(k) or {}).get("n_cells", 10**9) >= 400}),
):
    n_cells_failing, expected_noise_flags, flagged = 0, 0.0, []
    for L, f in mine["folds"].items():
        for s in f["seeds"]:
            g = s["calibration_gate"]
            sl, sup = g["slopes"], (g.get("support") or {})
            keys = keyfn(sl, sup)
            o, n = oob(sl, keys)
            if o:
                n_cells_failing += 1
                flagged.append(f"{L}/seed{s['seed']}:{','.join(o)}")
            # expected number of PURELY-NOISE flags in this cell, at the measured floor
            for k in keys:
                nc = (sup.get(k) or {}).get("n_cells")
                r = ff(nc)
                if r is not None:
                    expected_noise_flags += r
    budget[scope] = {
        "fold_seed_cells_failing_of_12": n_cells_failing,
        "expected_noise_only_slice_flags_over_the_12_cells": round(expected_noise_flags, 2),
        "flagged": flagged,
    }
res["Q3_false_flag_budget"] = budget

# ---------------- Q4: how big must the band be for every-slice to be sane? ----------------------
# For each failing thin bucket, what band width would it need to not fire on a perfect surface at
# a 5% false-flag rate? (band half-width = 1.96 * sd)
need = {}
for L, f in mine["folds"].items():
    for s in f["seeds"][:1]:
        g = s["calibration_gate"]
        for k, v in g["slopes"].items():
            nc = ((g.get("support") or {}).get(k) or {}).get("n_cells")
            sd = sd_at(nc)
            if sd:
                need[f"{L}/{k}"] = {
                    "n_cells": nc,
                    "slope": v,
                    "sd": round(sd, 4),
                    "band_halfwidth_for_5pct_false_flag": round(1.96 * sd, 4),
                    "implied_band": [round(1 - 1.96 * sd, 3), round(1 + 1.96 * sd, 3)],
                }
res["Q4_band_width_needed_per_slice_for_5pct_false_flag"] = need

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, sort_keys=True)
print(json.dumps({"Q1": q1_verdicts, "Q3": budget}, indent=2, sort_keys=True))
print("WROTE", OUT)
