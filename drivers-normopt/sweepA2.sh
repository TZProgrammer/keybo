#!/bin/bash
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
cd /tmp/normopt
PY=/local/home/zegertho/repos/keybo/.venv/bin/python
export PYTHONPATH=/tmp/normopt/src
for s in 0 1 2 3 4 5 6 7 8 9; do
  $PY -m keybo optimize --model /tmp/normopt-scratch/models/trigram_cond31_seed0.json \
     --ngram trigram --start "qwertyuiopasdfghjkl'zxcvbnm,.-" --seed $s --no-progress \
     --out runs/A2-s$s.json > logs/A2-s$s.log 2>&1
  echo "A2 s$s rc=$?"
done
echo A2-COMPLETE
