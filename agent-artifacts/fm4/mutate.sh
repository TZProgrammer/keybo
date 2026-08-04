#!/bin/bash
# FM4 INVARIANT 5: mutation-test the new assertions.
# Each mutation is a SINGLE targeted edit that a correct test suite MUST catch. A mutation that
# leaves the suite green is a test asserting something other than its own name -- the exact defect
# PRODUCTIZE-1 found three of.
set -u
cd /local/home/zegertho/repos/keybo-wt-fm4
PY=/local/home/zegertho/repos/keybo/.venv/bin/python
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
TESTS="tests/analysis/test_gauge_collisions.py"

# -B AND an explicit cache purge. Both are REQUIRED and the second is the one that bit me:
# restoring a mutated file from a .bak in the same second, at the same byte size, leaves a .pyc
# whose (mtime, size) validation stamp still MATCHES -- so CPython loads the MUTATED bytecode for
# the RESTORED source. That silently reversed three verdicts on the first run of this harness
# (M13/M14 "survived" against code that was no longer on disk). -B stops the mutated run from
# WRITING a .pyc; the purge stops any pre-existing one from being read.
purge_cache() { find . -name '__pycache__' -type d -not -path './.git/*' -exec rm -rf {} + 2>/dev/null; }
run() {
  purge_cache
  timeout 900 $PY -B -m pytest $TESTS -q -p no:randomly >/tmp/fm4_work/mut.txt 2>&1
  local rc=$?
  purge_cache
  echo $rc
}

declare -a NAMES=() RESULTS=()
mutate() {  # name file sed-expr
  local name="$1" file="$2" expr="$3"
  cp "$file" "$file.bak"
  sed -i "$expr" "$file"
  if diff -q "$file" "$file.bak" >/dev/null; then
    echo "SKIP  $name  (sed matched nothing -- mutation not applied)"
    NAMES+=("$name"); RESULTS+=("NOT-APPLIED")
  else
    local rc; rc=$(run)
    if [ "$rc" = "0" ]; then
      echo "GREEN $name   <-- SURVIVED, the test does not catch it"
      NAMES+=("$name"); RESULTS+=("SURVIVED")
    else
      echo "RED   $name   (killed)"
      NAMES+=("$name"); RESULTS+=("KILLED")
    fi
  fi
  mv "$file.bak" "$file"
}

SD=src/keybo/analysis/shap_diff.py

echo "=== baseline (must be GREEN) ==="
rc=$(run); echo "baseline rc=$rc"; [ "$rc" = "0" ] || { echo "BASELINE RED -- abort"; tail -20 /tmp/fm4_work/mut.txt; exit 1; }

echo
echo '=== M1: drop the lsb entry from the map (a genuine collision left un-annotated) ==='
mutate "M1 remove lsb entry" "$SD" '/^    "lsb": ($/,/^    ),$/d'

echo "=== M2: ADD scissor to the map (annotating a TRUTHFUL shared name) ==="
mutate "M2 add scissor entry" "$SD" 's|^GAUGE_COLLISIONS: dict\[str, tuple\[str, str, str\]\] = {$|GAUGE_COLLISIONS: dict[str, tuple[str, str, str]] = {\n    "scissor": ("scissor_x", "the `scissor` gauge", "0 disagreements"),|'

echo "=== M3: display_name becomes the identity (the rename stops happening) ==="
mutate "M3 display_name identity" "$SD" 's|^    entry = _COLLISION_COLUMNS.get(column)$|    entry = None|'

echo "=== M4: gauge_collision_notes returns [] (disclosure silently disappears) ==="
mutate "M4 notes always empty" "$SD" 's|^    notes = \[\]$|    notes = []; return notes|'

echo "=== M5: mirror bg1_/bg2_ for ALL entries incl trigram-level (the bug the test caught) ==="
mutate "M5 mirror all levels" "$SD" 's|^        if col in _BIGRAM_LEVEL_COLLISIONS$|        if True|'

echo "=== M6: a display name that collides with a served column ==="
mutate "M6 display name = dx" "$SD" 's|^        "landing_off_home",$|        "dx",|'

echo "=== M7: a display name that collides with a GAUGE name ==="
mutate "M7 display name = lat-span" "$SD" 's|^        "landing_off_home",$|        "lat-span",|'

echo "=== M8: drop the measured NUMBER from a note (turns evidence into opinion) ==="
mutate "M8 note without a number" "$SD" 's|"index/middle stagger-adjusted dx > 1.5 -- a strict SUPERSET of the gauge (32 vs 24 "|"index/middle stagger-adjusted dx, a superset of the gauge ("|'

echo "=== M9: gauge-side index points at an un-annotated column ==="
mutate "M9 gauge side -> scissor" "$SD" 's|^    "lat-span": "lateral",$|    "lat-span": "scissor",|'

echo "=== M10: gauge-side index names a gauge that is not reported ==="
mutate "M10 gauge side bogus gauge" "$SD" 's|^    "lat-span": "lateral",$|    "not-a-gauge": "lateral",|'

# --- mutations to the PREDICATES themselves: the counts must be load-bearing --------------
NG=src/keybo/features/ngram.py
CL=src/keybo/features/classify.py

echo "=== M11: widen is_lsb threshold 1.5 -> 2.0 (would make the column EQUAL the gauge) ==="
mutate "M11 is_lsb dx > 2.0" "$CL" 's|return index_middle and geometry.stagger_adjusted_dx(a, b) > 1.5|return index_middle and geometry.stagger_adjusted_dx(a, b) > 2.0|'

echo "=== M12: GATE the served redirect (would make the column EQUAL the gauge) ==="
mutate "M12 gate served redirect" "$NG" 's|        redirect = going_in_1 != going_in_2|        redirect = going_in_1 != going_in_2 and not (C.same_finger(g, a, b) or C.same_finger(g, b, c))|'

echo "=== M13: is_lateral drops the K31 pinky column ==="
mutate "M13 is_lateral no |x|==6" "$CL" 's|^    return abs(x) in (1, 6)$|    return abs(x) == 1|'

echo "=== M14: is_scissor dy == 2 -> dy >= 2 (breaks the EQUAL verdict silently?) ==="
mutate "M14 is_scissor dy >= 2" "$CL" 's|^    return abs(a\[1\] - b\[1\]) == 2$|    return abs(a[1] - b[1]) >= 2|'

echo "=== M15: display name = off_home_column (interp.1 already uses it for a DIFFERENT predicate) ==="
mutate "M15 display name = off_home_column" "$SD" 's|^        "landing_off_home",$|        "off_home_column",|'

echo
echo "================ SUMMARY ================"
survived=0
for i in "${!NAMES[@]}"; do
  printf '%-34s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}"
  [ "${RESULTS[$i]}" != "KILLED" ] && survived=$((survived+1))
done
echo "-----------------------------------------"
echo "total=${#NAMES[@]}  killed=$(( ${#NAMES[@]} - survived ))  not-killed=$survived"
