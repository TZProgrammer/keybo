#!/bin/bash
# FIND-pass probe 4 (v2): does the shipped gate BITE?
#
# v1 BUG (self-caught, kept as a lesson): the verdict used
#     grep -qE 'failed|error'
# against pytest's output, which is case-SENSITIVE, so "FAILED"/"AssertionError"
# never matched and EVERY mutation reported SURVIVED — a comparison returning the
# answer that means "no problem". v2 gates on pytest's EXIT CODE, the authoritative
# signal, and additionally prints the parsed counts.
set -u
SRC=/tmp/qapaudit/src/keybo/optimize/qap_bound.py
WORK=/tmp/qap_mut
PY=/tmp/qapaudit-venv/bin/python
rm -rf $WORK && mkdir -p $WORK
cp $SRC $WORK/orig.py
N_CAUGHT=0; N_SURV=0; N_NOOP=0

run_mut () {
  local name="$1" expr="$2"
  cp $WORK/orig.py $SRC
  sed -i "$expr" $SRC
  if diff -q $WORK/orig.py $SRC >/dev/null; then
    printf 'NO-OP     %-46s | sed matched nothing (mutation not applied)\n' "$name"
    N_NOOP=$((N_NOOP+1)); cp $WORK/orig.py $SRC; return
  fi
  out=$(cd /tmp/qapaudit && PYTHONPATH=/tmp/qapaudit/src $PY -m pytest \
        tests/optimize/test_qap_bound.py -p no:cacheprovider -q --no-header 2>&1)
  rc=$?                       # 0 = all passed, 1 = tests failed, 2+ = error
  summary=$(echo "$out" | grep -oE '[0-9]+ (passed|failed)(, [0-9]+ (passed|failed))*' | tail -1)
  if [ $rc -ne 0 ]; then verdict="CAUGHT  "; N_CAUGHT=$((N_CAUGHT+1))
  else verdict="SURVIVED"; N_SURV=$((N_SURV+1)); fi
  printf '%s  %-46s | rc=%d  %s\n' "$verdict" "$name" "$rc" "${summary:-<no summary>}"
  cp $WORK/orig.py $SRC
}

echo "=== baseline (unmutated) ==="
(cd /tmp/qapaudit && PYTHONPATH=/tmp/qapaudit/src $PY -m pytest tests/optimize/test_qap_bound.py -p no:cacheprovider -q --no-header >/dev/null 2>&1; echo "baseline rc=$?")
echo
echo "=== A. perturb the BOUND ==="
run_mut "bound x1.001 (inflate 0.1%)"          's|    return float(cost\[rows, cols\].sum())|    return float(cost[rows, cols].sum()) * 1.001|'
run_mut "bound x1.10 (inflate 10%)"            's|    return float(cost\[rows, cols\].sum())|    return float(cost[rows, cols].sum()) * 1.10|'
run_mut "bound x2.0 (inflate 100%)"            's|    return float(cost\[rows, cols\].sum())|    return float(cost[rows, cols].sum()) * 2.0|'
run_mut "bound x0.999 (deflate 0.1%: still VALID, looser)" 's|    return float(cost\[rows, cols\].sum())|    return float(cost[rows, cols].sum()) * 0.999|'
run_mut "bound x0.5 (deflate 50%: still VALID, vacuous-er)" 's|    return float(cost\[rows, cols\].sum())|    return float(cost[rows, cols].sum()) * 0.5|'
run_mut "bound := 0.0 (VACUOUS)"               's|    return float(cost\[rows, cols\].sum())|    return 0.0|'
run_mut "bound := 1e-12 (vacuous but positive)" 's|    return float(cost\[rows, cols\].sum())|    return 1e-12|'
echo
echo "=== B. break the RELAXATION math ==="
run_mut "drop halving: 0.5*(...) -> 1.0*(...)"  's|cost\[i, k\] = F\[i, i\] \* T\[k, k\] + 0.5 \* (|cost[i, k] = F[i, i] * T[k, k] + 1.0 * (|'
run_mut "sorted_dot MAXIMIZES (drop the [::-1])" 's|    return float(np.sort(f_row)\[::-1\] @ np.sort(t_row))|    return float(np.sort(f_row) @ np.sort(t_row))|'
run_mut "LAP maximize=True"                    's|    rows, cols = linear_sum_assignment(cost)|    rows, cols = linear_sum_assignment(cost, maximize=True)|'
run_mut "drop diagonal term F[i,i]*T[k,k]"     's|cost\[i, k\] = F\[i, i\] \* T\[k, k\] + 0.5 \* (|cost[i, k] = 0.0 + 0.5 * (|'
run_mut "off-diag mask -> full (double-count diag)" 's|    off = ~np.eye(n, dtype=bool)|    off = np.ones((n, n), dtype=bool)|'
run_mut "use OUTGOING twice (drop the incoming leg)" 's|_sorted_dot_min(f_in, t_in)|_sorted_dot_min(f_out, t_out)|'
run_mut "t_in reads row k not column k"        's|            t_in = T\[:, k\]\[off\[:, k\]\]|            t_in = T[k][off[k]]|'
run_mut "f_in reads row i not column i"        's|        f_in = F\[:, i\]\[off\[:, i\]\]|        f_in = F[i][off[i]]|'
echo
echo "=== C. break qap_fitness / direction ==="
run_mut "qap_fitness negated (direction flip)" 's|    return float((F \* T\[np.ix_(p, p)\]).sum())|    return float(-(F * T[np.ix_(p, p)]).sum())|'
run_mut "qap_fitness ignores perm (identity)"  's|    return float((F \* T\[np.ix_(p, p)\]).sum())|    return float((F * T).sum())|'
run_mut "qap_fitness transposes F"             's|    return float((F \* T\[np.ix_(p, p)\]).sum())|    return float((F.T * T[np.ix_(p, p)]).sum())|'
echo
echo "=== D. break the CERTIFICATE arithmetic ==="
run_mut "gap sign flip (lb-found)/lb"          's|    gap = (found_fitness - lb) / lb \* 100 if lb > 0 else float("inf")|    gap = (lb - found_fitness) / lb * 100 if lb > 0 else float("inf")|'
run_mut "gap divides by found, not lb"         's|    gap = (found_fitness - lb) / lb \* 100 if lb > 0 else float("inf")|    gap = (found_fitness - lb) / found_fitness * 100 if lb > 0 else float("inf")|'
run_mut "gap forgets the x100"                 's|    gap = (found_fitness - lb) / lb \* 100 if lb > 0 else float("inf")|    gap = (found_fitness - lb) / lb if lb > 0 else float("inf")|'
run_mut "gap hardcoded 0.0 (perfect cert)"     's|    gap = (found_fitness - lb) / lb \* 100 if lb > 0 else float("inf")|    gap = 0.0|'
run_mut "lb>0 guard -> lb>=0 (0-div reachable)" 's|if lb > 0 else float("inf")|if lb >= 0 else float("inf")|'
run_mut "statement text drops the gap number"  's|f"the found layout is within {gap:.2f}% of the best possible layout "|"the found layout is within 0.00% of the best possible layout "|'
cp $WORK/orig.py $SRC
echo
printf '=== TOTALS: CAUGHT %d | SURVIVED %d | NO-OP %d ===\n' $N_CAUGHT $N_SURV $N_NOOP
echo "=== worktree restored (empty diff = clean) ==="
cd /tmp/qapaudit && git diff --stat -- src/keybo/optimize/qap_bound.py; git diff --quiet -- src/keybo/optimize/qap_bound.py && echo "CLEAN"
