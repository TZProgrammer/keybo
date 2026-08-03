"""l03: DEMONSTRATE the affine-invariance exemption; live-pair q-sensitivity; D3; INVARIANT 6.

calib proved the K31 miscalibration is UNIFORM (0 of 6 class contrasts differential) => a positive
affine map ms -> a*ms + b. This driver PROVES numerically (not asserts) that every LOS is exactly
invariant under it, which is what licenses leaving calibration OUT of the sign uncertainty. Then it
sweeps the live pair over every measured flip-hazard stratum, exhibits a genuinely-undecided real
pair (D3), and runs the seed-count sensitivity that tests CLOSING-2's "diversity not compute" claim.
"""

from __future__ import annotations

import json

import numpy as np
from l00_common import ALL_SEEDS, TOURNAMENT_JSON, PerSeedSurface, assert_provenance, dump, load_boards

from keybo.analysis.los import apply_flip_hazard, compute_los, split_half_floor

prov = assert_provenance()
tj = json.loads(TOURNAMENT_JSON.read_text())
boards = load_boards()
surf = PerSeedSurface(90.0)
panel_all = {n: surf.per_seed_ms_per_char(boards[n]) for n in boards}
names13 = list(tj["boards"].keys())
mat13 = np.stack([panel_all[n] for n in names13])
FLOOR = split_half_floor(mat13, n_partitions=2000, rng=np.random.default_rng(7))["p90"]
print(f"[floor all-cells] {FLOOR:.4f}")

# ------------------------------------------------------------------------------------------------
# (A) AFFINE-INVARIANCE EXEMPTION -- DEMONSTRATED. calib: compression is UNIFORM (positive affine).
# Apply ms -> a*ms + b for a grid of (a,b) INCLUDING calib's over-shrink factor a=1.4618, recompute
# the floor from the transformed panel, and check LOS_design/seed are identical to machine precision.
# ------------------------------------------------------------------------------------------------
inv = {"cases": [], "worst_abs_dev_los_design": 0.0, "worst_abs_dev_los_seed": 0.0}
base_c = panel_all["candidate"]; base_f = panel_all["flagship-c3"]
base = compute_los(base_c, base_f, FLOOR, "candidate", "flagship-c3")
for a, b in [(1.4618, 0.0), (1.4618, -60.7), (0.5, 100.0), (2.0, -254.0), (1.0, 12.3)]:
    # transform the WHOLE panel; the floor is a functional of the same panel, so it scales by |a|.
    tp = a * mat13 + b
    fl = split_half_floor(tp, n_partitions=2000, rng=np.random.default_rng(7))["p90"]
    tc, tf = a * base_c + b, a * base_f + b
    r = compute_los(tc, tf, fl, "candidate", "flagship-c3")
    dd = abs(r.los_design - base.los_design)
    ds = abs(r.los_seed - base.los_seed)
    inv["cases"].append({"a": a, "b": b, "floor_scaled": fl, "floor_expected": abs(a) * FLOOR,
                         "los_design": r.los_design, "los_seed": r.los_seed,
                         "dev_design": dd, "dev_seed": ds,
                         "margin_over_floor": r.margin_over_floor})
    inv["worst_abs_dev_los_design"] = max(inv["worst_abs_dev_los_design"], dd)
    inv["worst_abs_dev_los_seed"] = max(inv["worst_abs_dev_los_seed"], ds)
    print(f"[AFFINE a={a} b={b}] floor {FLOOR:.4f}->{fl:.4f} (expect {abs(a)*FLOOR:.4f}) "
          f"LOS_design dev={dd:.2e} margin/floor={r.margin_over_floor:.3f}")
print(f"[AFFINE] worst |Δ LOS_design| over all maps = {inv['worst_abs_dev_los_design']:.2e} "
      f"(exempt if ~0) ; worst |Δ LOS_seed| = {inv['worst_abs_dev_los_seed']:.2e}")

# ------------------------------------------------------------------------------------------------
# (B) LIVE-PAIR q-SENSITIVITY: LOS_typist over EVERY measured flip-hazard stratum. candidate vs
# flagship gap 0.98-1.13 sits in the 0.97-3.04 band (q=0.30), but show all strata + the crossover.
# ------------------------------------------------------------------------------------------------
qsens = {"pricings": {}}
for pr in ("all", "observed", "common"):
    c = np.array(tj["mspc"][pr]["candidate"]) if pr != "all" else base_c
    f = np.array(tj["mspc"][pr]["flagship-c3"]) if pr != "all" else base_f
    fl = split_half_floor(np.stack([np.array(tj["mspc"][pr][n]) for n in names13]),
                          n_partitions=2000, rng=np.random.default_rng(7))["p90"]
    r = compute_los(c, f, fl, "candidate", "flagship-c3")
    row = {"los_design": r.los_design, "margin": r.mean_margin, "floor": fl,
           "los_typist_by_q": {q: apply_flip_hazard(r.los_design, q)
                               for q in (0.0, 0.12, 0.30, 0.42, 0.50, 0.74, 0.81)}}
    qsens["pricings"][pr] = row
    print(f"[q-SENS {pr:9s}] LOS_design={r.los_design:.4f}; LOS_typist at "
          f"q=0.12:{row['los_typist_by_q'][0.12]:.3f} q=0.30:{row['los_typist_by_q'][0.30]:.3f} "
          f"q=0.74:{row['los_typist_by_q'][0.74]:.3f}")
# the crossover q where LOS_typist drops below 0.95 and where it flips below 0.5
ld = qsens["pricings"]["all"]["los_design"]
qsens["q_where_typist_below_0.95_allcells"] = float((ld - 0.95) / (2 * ld - 1)) if ld != 0.5 else None
qsens["note"] = ("LOS_typist = (1-q)*L + q*(1-L); it hits 0.5 only at q=0.5 and is <0.95 for any "
                 "q above (L-0.95)/(2L-1). For candidate/flagship L~1 so typist<0.95 once q>~0.05.")

# ------------------------------------------------------------------------------------------------
# (C) D3: exhibit a REAL pair with LOS_design genuinely in [0.3,0.7] (the instrument returns ~0.5
# on real data, not only on constructed nulls). Scan all 78 pairs.
# ------------------------------------------------------------------------------------------------
undecided = []
for i in range(len(names13)):
    for j in range(i + 1, len(names13)):
        A, B = names13[i], names13[j]
        r = compute_los(panel_all[A], panel_all[B], FLOOR, A, B)
        if 0.30 <= r.los_design <= 0.70:
            undecided.append({"pair": f"{A} vs {B}", "margin": r.mean_margin,
                              "margin_over_floor": r.margin_over_floor,
                              "los_design": r.los_design, "p_two_sided": r.p_two_sided})
undecided.sort(key=lambda x: abs(x["los_design"] - 0.5))
print(f"[D3] {len(undecided)} real pairs with LOS_design in [0.3,0.7]:")
for u in undecided[:8]:
    print(f"     {u['pair']:26s} margin={u['margin']:+.4f} ({u['margin_over_floor']:.2f}x) "
          f"LOS_design={u['los_design']:.3f} p2={u['p_two_sided']:.2e}")

# ------------------------------------------------------------------------------------------------
# (D) INVARIANT 6 -- seed-count sensitivity. Does adding seeds move a named UNDECIDED pair to
# DECIDED? Test the cross-family pair arm-B vs candidate (SEEDTB-1: needs n~783). Sub-sample seeds.
# ------------------------------------------------------------------------------------------------
inv6 = {"pairs": {}}
for A, B in [("arm-B", "candidate"), ("F(2.0)", "candidate"), ("candidate", "flagship-c3")]:
    dA, dB = panel_all[A], panel_all[B]
    traj = []
    for nn in (3, 5, 10, 15, 20, 25):
        r = compute_los(dA[:nn], dB[:nn], FLOOR, A, B)
        traj.append({"n": nn, "margin": r.mean_margin, "los_design": r.los_design,
                     "margin_over_floor": r.margin_over_floor})
    inv6["pairs"][f"{A} vs {B}"] = traj
    print(f"[INV6 {A} vs {B}] LOS_design by n: " +
          " ".join(f"n{t['n']}={t['los_design']:.3f}" for t in traj))
inv6["closing2_note"] = ("If LOS_design stays ~0.5 as n grows for the cross-family pairs while the "
                         "candidate/flagship pair is already 1.0, the binding constraint is layout "
                         "diversity, not seed count -- CLOSING-2's prediction. Floor is the SAME.")

out = {"provenance": prov, "floor_all": FLOOR, "affine_invariance": inv,
       "live_q_sensitivity": qsens, "D3_real_undecided": undecided, "invariant6_seed_sensitivity": inv6}
dump("l03_invariance_sensitivity.json", out)
print("\n=== DONE l03 ===")
