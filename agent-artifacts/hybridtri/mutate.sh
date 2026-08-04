#!/usr/bin/env bash
# HYBRIDB-1 INVARIANT 3 — mutate ONE source string, run the target tests, restore, report.
#
# Usage: mutate.sh <id> <file> <from> <to> <test-selector...>
#
# FOUR correctness points, every one of them a bug this campaign actually hit:
#
# 1. __pycache__ IS PURGED BEFORE **AND** AFTER each mutation, and pytest runs with -B.
#    FM4-1's first mutation run reported 3 SURVIVORS THAT WERE ALL FALSE: a .bak restored in the
#    same second at the same byte size satisfies CPython's (source_mtime, source_size) .pyc check,
#    so the interpreter reused the MUTATED bytecode against the RESTORED source. -B alone is not
#    enough, because a stale .pyc from a PREVIOUS run is still eligible.
# 2. THE MUTATION IS ASSERTED TO HAVE CHANGED THE FILE. A pattern that no longer matches (source
#    reformatted, string moved) reports NOT-APPLIED -- never a false RED, which would read as
#    "the test caught it".
# 3. rc IS TAKEN DIRECTLY FROM PYTEST, never from a pipe tail. `cmd | tail; rc=$?` captures TAIL's
#    rc and silently INVERTED a sibling's mutation verdicts.
# 4. THE RESTORE IS IN A trap, so an interrupted run cannot leave a mutated source on the branch.

set -u
if [ $# -lt 5 ]; then
    echo "usage: $0 <id> <file> <from> <to> <test-selector...>" >&2
    exit 2
fi
ID="$1"; FILE="$2"; FROM="$3"; TO="$4"; shift 4
WT=/local/home/zegertho/repos/keybo-wt-hybridtri
PY=/local/home/zegertho/repos/keybo/.venv/bin/python

purge() { find "$WT/src" "$WT/tests" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; }

cd "$WT" || exit 2
[ -f "$FILE" ] || { echo "$ID NOT-APPLIED (no such file: $FILE)"; exit 3; }

BAK="$(mktemp /tmp/hybridtri_wk/mutate.XXXXXX.bak)"
cp "$FILE" "$BAK"
# shellcheck disable=SC2064
trap "cp '$BAK' '$FILE'; rm -f '$BAK'; purge" EXIT INT TERM

purge
"$PY" - "$FILE" "$FROM" "$TO" <<'PYEOF'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
if frm not in src:
    sys.exit(9)                      # pattern absent -> NOT-APPLIED
out = src.replace(frm, to, 1)
if out == src:
    sys.exit(9)                      # replacement was a no-op -> NOT-APPLIED
open(path, "w").write(out)
PYEOF
rc_mut=$?
if [ "$rc_mut" -eq 9 ]; then
    echo "$ID NOT-APPLIED (pattern absent or no-op) :: $FILE :: ${FROM:0:60}"
    exit 3
fi
if [ "$rc_mut" -ne 0 ]; then
    echo "$ID NOT-APPLIED (mutator failed rc=$rc_mut)"
    exit 3
fi

purge
PYTHONPATH="$WT/src" "$PY" -B -m pytest "$@" -q -p no:randomly -x > /tmp/hybridtri_wk/mut_$ID.log 2>&1
rc=$?                                # <-- straight from pytest, NOT from a pipe
purge

if [ "$rc" -eq 0 ]; then
    echo "$ID SURVIVED  (tests still GREEN under the mutation -- the assertion tests nothing) :: ${FROM:0:60}"
else
    echo "$ID RED       (rc=$rc) :: ${FROM:0:60}"
fi
exit 0
