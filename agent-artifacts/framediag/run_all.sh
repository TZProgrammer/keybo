#!/bin/bash
source /local/home/zegertho/fdwk/mut/mutate.sh
echo "=== MUTATION TESTING: keybo.analysis.frame_collapse + keybo.cli.frame_collapse ==="
echo

# --- M1..M4: the FLOOR ESTIMATOR (the INVARIANT 4 core) ---
run_one "M1 median floor -> use the group MEAN" "$LIB" \
  'wmae = float((weights * np.abs(target - median_of[inverse])).sum() / total)' \
  'wmae = float((weights * np.abs(target - mean_of[inverse])).sum() / total)'
run_one "M2 mean floor -> use the group MEDIAN" "$LIB" \
  'wmae_mean = float((weights * np.abs(target - mean_of[inverse])).sum() / total)' \
  'wmae_mean = float((weights * np.abs(target - median_of[inverse])).sum() / total)'
run_one "M3 weighted median -> UNWEIGHTED median" "$LIB" \
  'return float(v[np.searchsorted(np.cumsum(w), 0.5 * total, side="left")])' \
  'return float(v[len(v) // 2])'
run_one "M4 wrmse: drop the sqrt" "$LIB" \
  'wrmse = float(np.sqrt((weights * (target - mean_of[inverse]) ** 2).sum() / total))' \
  'wrmse = float((weights * (target - mean_of[inverse]) ** 2).sum() / total)'

# --- M5..M7: the WEIGHTING ---
run_one "M5 ignore caller weights (force uniform)" "$LIB" \
  'w = np.asarray(weights, dtype=np.float64).ravel()' \
  'w = np.ones(n_cells, dtype=np.float64)'
run_one "M6 mass share -> cell share" "$LIB" \
  'mass_share = float(w[is_collapsed].sum() / w_total) if w_total > 0 else 0.0' \
  'mass_share = collapsed_cells / n_cells'
run_one "M7 accept negative weights" "$LIB" \
  'if (w < 0).any():' \
  'if False:'

# --- M8..M11: the GROUPING / TOLERANCE ---
run_one "M8 tol ignored (always exact)" "$LIB" \
  'keys = X if tol == 0.0 else np.round(X / tol)' \
  'keys = X'
run_one "M9 tol: round(x/tol) -> round(x*tol)" "$LIB" \
  'keys = X if tol == 0.0 else np.round(X / tol)' \
  'keys = X if tol == 0.0 else np.round(X * tol)'
run_one "M10 allow non-finite feature rows" "$LIB" \
  'if not np.isfinite(X).all():' \
  'if False:'
run_one "M11 collapsed = sizes>=1 (every cell)" "$LIB" \
  'is_collapsed = sizes[inverse] > 1' \
  'is_collapsed = sizes[inverse] >= 1'

# --- M12..M15: the CELL SPACE (the 765-vs-775 core) ---
run_one "M12 include_space ignored (always on)" "$LIB" \
  'return [*slots, geometry.space_position] if include_space else slots' \
  'return [*slots, geometry.space_position]'
run_one "M13 include_space ignored (always off)" "$LIB" \
  'return [*slots, geometry.space_position] if include_space else slots' \
  'return slots'
run_one "M14 cell order: product -> permutations-ish reverse" "$LIB" \
  'for cell in itertools.product(pos, repeat=order)' \
  'for cell in itertools.product(pos[::-1], repeat=order)'
run_one "M15 includes_space field hard-coded True" "$LIB" \
  'includes_space=geometry.space_position in pos,' \
  'includes_space=True,'

# --- M16..M19: the SELF-GENERATED-TARGET FLAG ---
run_one "M16 flag hard-coded False" "$LIB" \
  '"target_is_self_generated": bool(n_cg > 0 and n_spread == 0),' \
  '"target_is_self_generated": False,'
run_one "M17 flag hard-coded True" "$LIB" \
  '"target_is_self_generated": bool(n_cg > 0 and n_spread == 0),' \
  '"target_is_self_generated": bool(n_cg >= 0),'
run_one "M18 flag drops the n_cg>0 guard" "$LIB" \
  '"target_is_self_generated": bool(n_cg > 0 and n_spread == 0),' \
  '"target_is_self_generated": bool(n_spread == 0),'
run_one "M19 spreads include SINGLETON groups" "$LIB" \
  'collapsed_groups = np.flatnonzero(sizes > 1)' \
  'collapsed_groups = np.flatnonzero(sizes >= 1)'

# --- M20..M22: the SWEEP VERDICT ---
run_one "M20 exceeds_exact hard-coded False" "$LIB" \
  '"exceeds_exact": bool(exact is not None and any(c > exact for c in counts)),' \
  '"exceeds_exact": False,'
run_one "M21 rises never detected" "$LIB" \
  'if counts[i + 1] > counts[i]' \
  'if False'
run_one "M22 flat hard-coded True" "$LIB" \
  '"flat": len(set(counts)) == 1,' \
  '"flat": True,'

# --- M23..M26: the CLI surface ---
run_one "M23 CLI mixed-order refusal removed" "$CLI" \
  'if len(orders) > 1:' \
  'if False:'
run_one "M24 CLI --no-space ignored" "$CLI" \
  'include_space = not args.no_space' \
  'include_space = True'
run_one "M25 CLI floor/no-space refusal removed" "$CLI" \
  'if args.no_space:' \
  'if False:'
run_one "M26 CLI geometry mismatch refusal removed" "$CLI" \
  'if surface.geometry is not geometry:' \
  'if False:'

# --- M27..M28: the REPORT ---
run_one "M27 report never warns on self-gen target" "$LIB" \
  'if r.target_is_self_generated:' \
  'if False:'
run_one "M28 report always shows floor columns" "$LIB" \
  'has_floor = first.floor_wmae is not None' \
  'has_floor = True'
