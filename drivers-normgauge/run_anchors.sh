#!/bin/bash
# NORMGAUGE-1: the 9 (model, seed) anchor-search cells, in parallel, single-threaded each.
# rc is read from a SENTINEL file per cell, never from a callback.
set -u
cd /tmp/normgauge
export PYTHONPATH=/tmp/normgauge/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/local/home/zegertho/repos/keybo/.venv/bin/python
D=drivers-normgauge
rm -f $D/runs/.sentinel-*
for pool in AALTO COMMUNITY POOL; do
  for seed in 20260728 20260901 20261015; do
    (
      nice -n 10 $PY $D/build_anchors.py --cell $pool $seed > $D/logs/anchor-$pool-$seed.log 2>&1
      echo "rc=$? pool=$pool seed=$seed" > $D/runs/.sentinel-$pool-$seed
    ) &
  done
done
wait
echo "ALL CELLS DONE"
cat $D/runs/.sentinel-* 
