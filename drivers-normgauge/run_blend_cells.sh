#!/bin/bash
# NORMGAUGE-1: the blend-search cells that DON'T need weight-evidence.json (the 5 fixed
# weightings). The 'registered' cell waits for the weights, by design: its weights must come
# from the evidence, not from me.
set -u
cd /tmp/normgauge
export PYTHONPATH=/tmp/normgauge/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/local/home/zegertho/repos/keybo/.venv/bin/python
D=drivers-normgauge
for label in "$@"; do
  for seed in 20260728 20260901 20261015; do
    (
      nice -n 12 $PY $D/blend_cell.py "$label" "$seed" > $D/logs/blend-$label-$seed.log 2>&1
      echo "rc=$? label=$label seed=$seed" > $D/runs/.sentinel-blend-$label-$seed
    ) &
  done
done
wait
echo "CELLS DONE"; cat $D/runs/.sentinel-blend-* 2>/dev/null
