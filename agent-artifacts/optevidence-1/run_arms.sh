#!/bin/bash
# Run the three arms sequentially, each at ~10M UNIQUE evals. Writes an rc sentinel.
set -u
S=/local/home/zegertho/agent/state/optevidence/artifacts
cd /tmp/optev || exit 9
RC=0
for ARM in evidence baseline constrained; do
  echo "=== ARM $ARM start $(date -u +%FT%TZ) ==="
  timeout 5400 uv run --no-sync python "$S/drivers/search.py" \
      --arm "$ARM" --budget 10000000 --islands 40 --epochs 55 --overshoot 1.95 \
      --ga-share 0.6 --polish-sweeps 40 --seed 20260728 --resume \
      --out "$S/runs/arm-$ARM.json" > "$S/runs/arm-$ARM.log" 2>&1
  r=$?
  echo "=== ARM $ARM rc=$r $(date -u +%FT%TZ) ==="
  [ $r -ne 0 ] && RC=$r
done
echo "$RC" > "$S/arms-rc.txt.tmp" && mv "$S/arms-rc.txt.tmp" "$S/arms-rc.txt"
ticket optevidence --prompt "[ARMS-DONE] all three arms finished, rc=$RC — sentinel at $S/arms-rc.txt"
