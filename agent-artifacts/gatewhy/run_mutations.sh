#!/usr/bin/env bash
# GATEWHY-1 mutation battery for tests/test_gate_control_vacuity.py.
#
# FM4-1's hazard, handled: a `.bak` restored in the same second at the same byte size satisfies
# CPython's (source_mtime, source_size) .pyc check, so a stale .pyc can run MUTATED BYTECODE against
# RESTORED SOURCE and report a false SURVIVOR. So: purge __pycache__ BEFORE AND AFTER every mutation,
# and run python -B (note: -B is a PYTHON flag; `pytest -B` is an unrecognized-argument error).
#
# Every mutation is ASSERTED to have changed the file (else NOT-APPLIED, never a false RED), rc is
# taken STRAIGHT from pytest (never from a pipe tail -- `cmd | tail; rc=$?` captures tail's rc), and
# the restore is in a trap so an interrupt cannot leave a mutated source on the branch.
set -u
TREE=/local/home/zegertho/repos/keybo-wt-gatewhy
PY=/local/home/zegertho/repos/keybo/.venv/bin/python
TESTS=tests/test_gate_control_vacuity.py
OUT="$TREE/agent-artifacts/gatewhy/mutation-results.txt"
cd "$TREE" || exit 1

purge() { find "$TREE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; }
restore() {
  for f in src/keybo/verdicts.py src/keybo/training/validate.py; do
    [ -f "$f.gwbak" ] && mv -f "$f.gwbak" "$f"
  done
  purge
}
trap restore EXIT INT TERM

: > "$OUT"
{
  echo "GATEWHY-1 mutation battery — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "target: $TESTS  (asserting each mutation applied; python -B; pycache purged before AND after)"
  echo
} >> "$OUT"

# id | file | python-expression sed pattern (old -> new)
run_mutation() {
  local id="$1" file="$2" old="$3" new="$4"
  purge
  cp "$file" "$file.gwbak"
  $PY - "$file" "$old" "$new" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); old, new = sys.argv[2], sys.argv[3]
s = p.read_text()
if old not in s:
    sys.exit(9)          # pattern absent -> NOT-APPLIED (never a false RED)
p.write_text(s.replace(old, new, 1))
PYEOF
  local patched=$?
  if [ $patched -ne 0 ]; then
    printf '%-6s NOT-APPLIED  %s  (pattern absent: %s)\n' "$id" "$file" "$old" >> "$OUT"
    mv -f "$file.gwbak" "$file"; purge; return
  fi
  if cmp -s "$file" "$file.gwbak"; then
    printf '%-6s NOT-APPLIED  %s  (file unchanged)\n' "$id" "$file" >> "$OUT"
    mv -f "$file.gwbak" "$file"; purge; return
  fi
  purge
  PYTHONPATH="$TREE/src" $PY -B -m pytest -q "$TESTS" > /tmp/gatewhy_wk/mut.$id.log 2>&1
  local rc=$?                      # STRAIGHT from pytest, not from a pipe
  mv -f "$file.gwbak" "$file"
  purge
  if [ $rc -eq 0 ]; then
    printf '%-6s SURVIVED     %s  %s -> %s\n' "$id" "$file" "$old" "$new" >> "$OUT"
  else
    printf '%-6s RED          %s  %s -> %s\n' "$id" "$file" "$old" "$new" >> "$OUT"
  fi
}

V=src/keybo/verdicts.py

# --- the gate's thresholds and scope --------------------------------------------------------
run_mutation M01 "$V" 'HIGH_WPM_TOLERANCE = 0.005' 'HIGH_WPM_TOLERANCE = 0.05'
run_mutation M02 "$V" 'HIGH_WPM_TOLERANCE = 0.005' 'HIGH_WPM_TOLERANCE = 0.0'
run_mutation M03 "$V" 'HIGH_WPM_FLOOR = 80' 'HIGH_WPM_FLOOR = 40'
run_mutation M04 "$V" 'HIGH_WPM_FLOOR = 80' 'HIGH_WPM_FLOOR = 120'

# --- the regression predicate itself --------------------------------------------------------
run_mutation M05 "$V" 'regressing = sorted(b for b, d in high.items() if d < -tolerance)' \
                      'regressing = sorted(b for b, d in high.items() if d <= -tolerance)'
run_mutation M06 "$V" 'regressing = sorted(b for b, d in high.items() if d < -tolerance)' \
                      'regressing = sorted(b for b, d in high.items() if d < tolerance)'
run_mutation M07 "$V" 'regressing = sorted(b for b, d in high.items() if d < -tolerance)' \
                      'regressing = sorted(b for b, d in high.items() if abs(d) > tolerance)'
run_mutation M08 "$V" 'high = {b: d for b, d in deltas.items() if b >= floor}' \
                      'high = {b: d for b, d in deltas.items() if b > floor}'
run_mutation M09 "$V" 'high = {b: d for b, d in deltas.items() if b >= floor}' \
                      'high = dict(deltas)'

# --- the delta's direction (a sign flip must not survive) -----------------------------------
run_mutation M10 "$V" 'bucket: float(candidate[bucket]) - float(baseline[bucket])' \
                      'bucket: float(baseline[bucket]) - float(candidate[bucket])'

# --- gated / passed tri-state (TAUGATE-1's ambiguity) ---------------------------------------
run_mutation M11 "$V" 'gated = bool(baseline) and top in deltas and bool(high)' \
                      'gated = True'
run_mutation M12 "$V" 'report["passed"] = not regressing' 'report["passed"] = True'
run_mutation M13 "$V" '        report["passed"] = not regressing' '        report["passed"] = bool(regressing)'

# --- support is recorded but must NOT gate (GATESUPPORT-1's deliberate choice) --------------
run_mutation M14 "$V" '    if gated:
        report["passed"] = not regressing' \
                      '    if gated:
        report["passed"] = not regressing or (
            report["min_regressing_support"] is not None
            and report["min_regressing_support"] < 30
        )'
run_mutation M15 "$V" '"min_regressing_support": (' '"min_regressing_support": (None) or ('

# --- the enforcing wrapper ------------------------------------------------------------------
run_mutation M16 "$V" '    if not report["passed"]:
        offenders' '    if False:
        offenders'
run_mutation M17 "$V" 'f"{what}: rho regresses in {len(offenders)} of {len(report[' \
                      'f"{what}: rho moved in {len(offenders)} of {len(report['
run_mutation M18 "$V" '    if not report["gated"]:
        top = max(baseline) if baseline else None' \
                      '    if False:
        top = max(baseline) if baseline else None'

echo >> "$OUT"
printf 'RED: %s   SURVIVED: %s   NOT-APPLIED: %s\n' \
  "$(grep -c ' RED ' "$OUT")" "$(grep -c ' SURVIVED ' "$OUT")" "$(grep -c ' NOT-APPLIED ' "$OUT")" >> "$OUT"
cat "$OUT"
