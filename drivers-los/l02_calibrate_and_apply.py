"""l02: measure the floor, CALIBRATE the instrument (INVARIANT 3), then apply it (INVARIANT 4).

Order is the registered order: floor -> calibration (null / known-big / permutation) -> live pair
-> full matrix. Uses TOURNAMENT-1's published per-seed ms/char (mspc) for all three pricings, which
my own pipeline reproduces to 4.97e-12 (l01), so no ~400s stroke reload is needed for observed/common.
"""

from __future__ import annotations

import json

import numpy as np
from l00_common import ALL_SEEDS, TOURNAMENT_JSON, PerSeedSurface, assert_provenance, dump, load_boards

from keybo.analysis.los import compute_los, scale_floor_to_n, split_half_floor

prov = assert_provenance()
RNG = np.random.default_rng(20260803)
tj = json.loads(TOURNAMENT_JSON.read_text())
PRICINGS = ("all", "observed", "common")
boards = load_boards()

# ------------------------------------------------------------------------------------------------
# per-seed ms/char panels. all-cells from MY pipeline (14 boards incl. keybo-lsb+lm); observed/common
# from TOURNAMENT-1's mspc (13 boards) -- verified equal to mine at 4.97e-12 on all-cells in l01.
# ------------------------------------------------------------------------------------------------
surf = PerSeedSurface(90.0)
panel = {"all": {}, "observed": {}, "common": {}}
for name in boards:
    panel["all"][name] = surf.per_seed_ms_per_char(boards[name])
for pr in ("observed", "common"):
    for name in tj["boards"]:
        panel[pr][name] = np.array(tj["mspc"][pr][name], dtype=np.float64)
# keybo-lsb+lm only exists in all-cells (not in tournament's mspc); note that where relevant.

# ------------------------------------------------------------------------------------------------
# STEP 1 -- THE FLOOR, measured for MY design (split-half same-board placebo). Borrow nothing.
# ------------------------------------------------------------------------------------------------
floors = {}
for pr in PRICINGS:
    mat = np.stack([panel[pr][n] for n in tj["boards"]])   # 13 boards x 25 seeds
    f = split_half_floor(mat, n_partitions=2000, rng=np.random.default_rng(7 + hash(pr) % 1000))
    f["floor_scaled_to_n25"] = scale_floor_to_n(f["p90"], f["half_n"], len(ALL_SEEDS))
    floors[pr] = f
    print(f"[FLOOR {pr:9s}] p90(half n={f['half_n']}) = {f['p90']:.4f}  "
          f"scaled->n25 = {f['floor_scaled_to_n25']:.4f}  (p50 {f['p50']:.4f}, p99 {f['p99']:.4f})")
# Headline uses the CONSERVATIVE unscaled p90 (a half-sample floor applied at full n).
FLOOR = {pr: floors[pr]["p90"] for pr in PRICINGS}
FLOOR_SCALED = {pr: floors[pr]["floor_scaled_to_n25"] for pr in PRICINGS}
print(f"[FLOOR check] prereg predicted p90 ~ 0.2921/0.2905 +-0.02; got all-cells {FLOOR['all']:.4f}")

# ------------------------------------------------------------------------------------------------
# STEP 2 -- CALIBRATION (INVARIANT 3). Report BEFORE any real pair.
# ------------------------------------------------------------------------------------------------
calib = {}

# NULL-1: board vs ITSELF -> LOS must be exactly 0.5.
r = compute_los(panel["all"]["candidate"], panel["all"]["candidate"], FLOOR["all"],
                "candidate", "candidate")
calib["null1_self"] = {"los_seed": r.los_seed, "los_design": r.los_design,
                       "los_typist": r.los_typist, "mean_margin": r.mean_margin}
print(f"[NULL-1 self]  LOS_design={r.los_design:.4f} (require ==0.5)  margin={r.mean_margin:.2e}")

# NULL-2: same-board split-half pairs -> LOS_design in [0.35,0.65]; median over partitions ~0.5.
null2_los = []
board_ms = panel["all"]["candidate"]
half = len(ALL_SEEDS) // 2
for _ in range(2000):
    perm = RNG.permutation(len(ALL_SEEDS))
    h1, h2 = perm[:half], perm[half:2 * half]
    # two disjoint same-board half-samples, treated as a paired comparison of 'board on H1' vs
    # 'board on H2'. Truth 0 by construction. Pair by within-half seed index (arbitrary but fixed).
    a = board_ms[h1]; b = board_ms[h2]
    rr = compute_los(a, b, FLOOR["all"], "H1", "H2")
    null2_los.append(rr.los_design)
null2_los = np.array(null2_los)
calib["null2_splithalf"] = {
    "median": float(np.median(null2_los)), "p05": float(np.percentile(null2_los, 5)),
    "p95": float(np.percentile(null2_los, 95)), "mean": float(null2_los.mean()),
    "frac_in_0.35_0.65": float(((null2_los >= 0.35) & (null2_los <= 0.65)).mean()),
    "frac_decided": float(((null2_los >= 0.95) | (null2_los <= 0.05)).mean()),
}
print(f"[NULL-2 split] median LOS_design={np.median(null2_los):.4f} "
      f"(require in [0.45,0.55]); frac decided={calib['null2_splithalf']['frac_decided']:.4f} "
      f"(require <=0.05)")

# NULL-3: permutation null -- sign-flip the per-seed margins of a REAL pair, 20000 draws. The LOS
# statistic's null distribution should be ~symmetric about 0.5 with P(LOS>0.95) small.
d_real = panel["all"]["candidate"] - panel["all"]["flagship-c3"]
n = d_real.size
perm_los = np.empty(20000)
for i in range(20000):
    signs = RNG.choice([-1.0, 1.0], size=n)
    dperm = d_real * signs
    # build synthetic ms arrays with this margin against a fixed reference
    ref = np.zeros(n)
    perm_los[i] = compute_los(dperm, ref, FLOOR["all"], "A", "B").los_design
calib["null3_permutation"] = {
    "median": float(np.median(perm_los)), "mean": float(perm_los.mean()),
    "P_los_gt_0.95": float((perm_los > 0.95).mean()),
    "P_los_lt_0.05": float((perm_los < 0.05).mean()),
    "P_decided_either": float(((perm_los > 0.95) | (perm_los < 0.05)).mean()),
}
print(f"[NULL-3 perm]  median={np.median(perm_los):.4f}  P(LOS>0.95)={(perm_los>0.95).mean():.4f} "
      f"(require <=0.05)")

# BIG: every tuned board vs qwerty -> LOS_design >= 0.99, LOS_typist >= 0.95 (capped by 0.12 hazard).
big = {}
for name in ("candidate", "flagship-c3", "arm-B", "F(2.0)", "keybo-lsb"):
    r = compute_los(panel["all"][name], panel["all"]["qwerty"], FLOOR["all"], name, "qwerty")
    big[name] = {"mean_margin": r.mean_margin, "margin_over_floor": r.margin_over_floor,
                 "signs": f"{r.signs_a_faster}/{r.signs_b_faster}",
                 "los_seed": r.los_seed, "los_design": r.los_design, "los_typist": r.los_typist}
    print(f"[BIG {name:12s} vs qwerty] margin={r.mean_margin:+.3f} ({r.margin_over_floor:.1f}x floor) "
          f"signs={r.signs_a_faster}/{r.signs_b_faster} LOS_design={r.los_design:.4f} "
          f"LOS_typist={r.los_typist:.4f}")
calib["big_vs_qwerty"] = big

# Calibration verdict per registered bars.
null1_ok = abs(calib["null1_self"]["los_design"] - 0.5) < 1e-9
null2_ok = 0.45 <= calib["null2_splithalf"]["median"] <= 0.55 and calib["null2_splithalf"]["frac_decided"] <= 0.05
null3_ok = calib["null3_permutation"]["P_decided_either"] <= 0.05
big_ok = all(v["los_design"] >= 0.99 for v in big.values())
calib["PASS"] = {"null1": null1_ok, "null2": null2_ok, "null3": null3_ok, "big": big_ok,
                 "instrument_validated": bool(null1_ok and null2_ok and null3_ok and big_ok)}
print(f"\n[CALIBRATION] null1={null1_ok} null2={null2_ok} null3={null3_ok} big={big_ok} "
      f"=> VALIDATED={calib['PASS']['instrument_validated']}")

# ------------------------------------------------------------------------------------------------
# STEP 3 -- THE LIVE PAIR (INVARIANT 4): candidate vs flagship-c3, all three pricings.
# ------------------------------------------------------------------------------------------------
live = {}
for pr in PRICINGS:
    r = compute_los(panel[pr]["candidate"], panel[pr]["flagship-c3"], FLOOR[pr],
                    "candidate", "flagship-c3")
    live[pr] = r.as_row()
    # also with the SCALED (finer, n=25) floor, to show the choice's effect
    r_sc = compute_los(panel[pr]["candidate"], panel[pr]["flagship-c3"], FLOOR_SCALED[pr],
                       "candidate", "flagship-c3")
    live[pr]["los_design_scaled_floor"] = r_sc.los_design
    live[pr]["los_typist_scaled_floor"] = r_sc.los_typist
    print(f"[LIVE {pr:9s}] margin={r.mean_margin:+.4f} ({r.margin_over_floor:.2f}x floor {FLOOR[pr]:.3f}) "
          f"signs={r.signs_a_faster}/{r.signs_b_faster} LOS_design={r.los_design:.4f} "
          f"LOS_typist={r.los_typist:.4f} verdict={r.verdict}")
decided_all = all(live[pr]["LOS_design"] >= 0.95 for pr in PRICINGS)
live["DECIDED_under_all_3_pricings"] = decided_all
live["candidate_faster_all"] = all(live[pr]["faster"] == "candidate" for pr in PRICINGS)

# ------------------------------------------------------------------------------------------------
# STEP 4 -- FULL 13-BOARD MATRIX (all-cells), every pair (INVARIANT 4). 78 ordered->unordered pairs.
# ------------------------------------------------------------------------------------------------
names13 = list(tj["boards"].keys())
matrix = {"all": [], "observed": [], "common": []}
for pr in PRICINGS:
    for i in range(len(names13)):
        for j in range(i + 1, len(names13)):
            A, B = names13[i], names13[j]
            r = compute_los(panel[pr][A], panel[pr][B], FLOOR[pr], A, B)
            matrix[pr].append(r.as_row())

# decomposition headline: seed-only vs extrapolation-included, per pricing, over all pairs
decomp = {}
for pr in PRICINGS:
    rows = matrix[pr]
    drops = [abs(row["LOS_seed"] - row["LOS_typist"]) for row in rows]
    decomp[pr] = {
        "n_pairs": len(rows),
        "decided_by_LOS_seed": sum(1 for r in rows if r["LOS_seed"] >= 0.95 or r["LOS_seed"] <= 0.05),
        "decided_by_LOS_design": sum(1 for r in rows if r["LOS_design"] >= 0.95 or r["LOS_design"] <= 0.05),
        "decided_by_LOS_typist": sum(1 for r in rows if r["LOS_typist"] >= 0.95 or r["LOS_typist"] <= 0.05),
        "mean_drop_seed_to_typist": float(np.mean(drops)),
        "max_drop_seed_to_typist": float(np.max(drops)),
    }
    print(f"[DECOMP {pr:9s}] decided: seed={decomp[pr]['decided_by_LOS_seed']} "
          f"design={decomp[pr]['decided_by_LOS_design']} typist={decomp[pr]['decided_by_LOS_typist']} "
          f"/ {decomp[pr]['n_pairs']}  (mean seed->typist drop {decomp[pr]['mean_drop_seed_to_typist']:.3f})")

out = {
    "provenance": prov, "n_seeds": len(ALL_SEEDS), "pricings": PRICINGS,
    "floors": floors, "FLOOR_headline_unscaled_p90": FLOOR, "FLOOR_scaled_n25": FLOOR_SCALED,
    "calibration": calib, "live_pair": live, "decomposition": decomp, "matrix": matrix,
}
dump("l02_los_results.json", out)
print("\n=== DONE l02 ===")
