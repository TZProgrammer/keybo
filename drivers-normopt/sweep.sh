#!/bin/bash
# 3 arms x seeds 0-9, shipped keybo optimize, hyperparams at DEFAULTS.
set -u
cd /tmp/normopt
PY=/local/home/zegertho/repos/keybo/.venv/bin/python
export PYTHONPATH=/tmp/normopt/src
MODEL=/tmp/normopt-scratch/models/bigram_reg31_seed0.json
START="qwertyuiopasdfghjkl'zxcvbnm,.-"
ANCH=/tmp/normopt/drivers-normgauge/anchors.json
mkdir -p runs logs
for s in 0 1 2 3 4 5 6 7 8 9; do
  # arm A: ms/char control
  $PY -m keybo optimize --model "$MODEL" --ngram bigram --start "$START" \
      --seed $s --no-progress --out runs/A-s$s.json > logs/A-s$s.log 2>&1
  echo "A s$s rc=$?"
  # arm B: normgauge registered (c)
  $PY -m keybo optimize --model "$MODEL" --ngram bigram --start "$START" \
      --seed $s --no-progress \
      --model-weight aalto-n=0.5411 --model-weight comm-n=0.3977 --model-weight pool-n=0.0612 \
      --model-anchors "$ANCH" --out runs/B-s$s.json > logs/B-s$s.log 2>&1
  echo "B s$s rc=$?"
  # arm C: normgauge 50/50 (drop-pool)
  $PY -m keybo optimize --model "$MODEL" --ngram bigram --start "$START" \
      --seed $s --no-progress \
      --model-weight aalto-n=0.5 --model-weight comm-n=0.5 \
      --model-anchors "$ANCH" --out runs/C-s$s.json > logs/C-s$s.log 2>&1
  echo "C s$s rc=$?"
done
echo "SWEEP-COMPLETE"
