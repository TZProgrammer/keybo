#!/bin/bash
# Mutation harness. rc is captured from PYTEST, never from a pipe tail (the campaign bug that
# silently inverted a sibling's verdicts). Each mutation is applied with python (exact string
# replace, asserted to have changed the file), tests run, file restored unconditionally.
set -u
WT=/local/home/zegertho/repos/keybo-wt-framediag
LIB=$WT/src/keybo/analysis/frame_collapse.py
CLI=$WT/src/keybo/cli/frame_collapse.py
PY=$HOME/repos/keybo/.venv/bin/python
export PYTHONPATH=$WT/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
TESTS="tests/analysis/test_frame_collapse.py tests/cli/test_frame_collapse_cli.py"

apply() {  # apply <file> <old> <new>
  "$PY" - "$1" "$2" "$3" <<'PYEOF'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if old not in s:
    sys.exit(9)
open(p, 'w').write(s.replace(old, new, 1))
PYEOF
}

run_one() {  # run_one <name> <file> <old> <new>
  local name="$1" f="$2" old="$3" new="$4"
  cp "$f" "$f.bak"
  apply "$f" "$old" "$new"
  local arc=$?
  if [ $arc -eq 9 ]; then
    mv "$f.bak" "$f"
    printf '%-46s NOT-APPLIED (pattern absent)\n' "$name"
    return
  fi
  ( cd "$WT" && "$PY" -m pytest $TESTS -x -q >/tmp/mutout.$$ 2>&1 )
  local rc=$?          # <-- pytest's rc, taken directly
  mv "$f.bak" "$f"
  if [ $rc -ne 0 ]; then
    local first
    first=$(grep -oE '^FAILED [^ ]+' /tmp/mutout.$$ | head -1)
    printf '%-46s RED   rc=%s  %s\n' "$name" "$rc" "${first:-(collection/assert error)}"
  else
    printf '%-46s !!!! GREEN -- TEST IS VACUOUS\n' "$name"
  fi
  rm -f /tmp/mutout.$$
}
