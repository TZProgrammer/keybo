"""Render the QUADGRAM-1 result JSON into report tables (machine-generated, not transcribed)."""

import json
import sys

R = json.load(open("/tmp/quad_eval_result.json"))


def fmt(x, nd=4):
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


ab = R["A_vs_B_paired"]
hw = R["high_wpm_gate"]
A, B, C = R["arm_A_quad_full"], R["arm_B_quad_trictx"], R["arm_C_tri_incumbent"]

print("=" * 78)
print("QUADGRAM-1 RESULTS")
print("=" * 78)

# --- decisive verdict ---
crit_a = (
    ab["mean_paired_delta"] is not None
    and ab["mean_paired_delta"] > 0
    and ab["n_folds_sign_consistent_win_for_A"] >= 3
)
crit_b = hw["high_wpm_pass"]
beats = crit_a and crit_b
print(f"\nDECISIVE A/B (QUAD-FULL vs matched QUAD-TRICTX, identical cells):")
print(f"  (a) transfer: mean paired delta {fmt(ab['mean_paired_delta'],6)}, "
      f"W/L {ab['overall_W']}/{ab['overall_L']}, "
      f"sign-consistent winning folds for A: {ab['n_folds_sign_consistent_win_for_A']}/4 "
      f"=> {'PASS' if crit_a else 'FAIL'}")
print(f"  (b) high-wpm gate (floor {hw['floor']}): "
      f"{'PASS' if crit_b else 'FAIL ' + str(hw['structural_regressions'])}")
print(f"\n  ==> QUADGRAM {'BEATS' if beats else 'DOES NOT BEAT'} TRIGRAM on held-out transfer.")

print(f"\nPer-fold paired deltas (A - B), per (fold,seed):")
for holdout, d in ab["per_fold"].items():
    print(f"  {holdout:8s} mean {fmt(d['mean_delta'],6):>10s}  "
          f"seeds {d['seed_deltas']}  {'sign-consistent' if d['sign_consistent'] else 'MIXED'} {d['direction']}")

print(f"\nFit / transfer levels (mean over folds×seeds):")
print(f"  {'arm':<22s} {'rho/ceil':>9s} {'rho':>8s} {'wmae':>8s} {'umae':>8s} {'pooled tau_heldout'}")
for name, arm in [("A QUAD-FULL", A), ("B QUAD-TRICTX", B), ("C TRI-INCUMBENT", C)]:
    print(f"  {name:<22s} {fmt(arm['mean_rho_frac_ceiling']):>9s} {fmt(arm['mean_rho']):>8s} "
          f"{fmt(arm['mean_wmae']):>8s} {fmt(arm['mean_umae']):>8s} {arm['pooled_tau_heldout']}")

print(f"\nCeilings (split-half, per layout): A={A['ceilings']}")

print(f"\nHigh-wpm gate detail (A vs B, bucket>=80):")
for holdout, d in hw["per_fold"].items():
    print(f"  {holdout:8s} n_seeds={d['n_seeds']} structural={d['structural_buckets']} "
          f"noise={d['noise_buckets']} counts={d['regressing_bucket_seed_counts']}")

print(f"\nA vs C note: {R['A_vs_C_note']}")
