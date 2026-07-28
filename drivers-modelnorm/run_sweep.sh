#!/bin/bash
# MODELNORM-1 step 4 + deliverable D: the blend search at campaign budget, and the PREFERENCE
# SWEEP. Every cell runs at the IDENTICAL budget/islands/epochs/seed as the anchor searches,
# so a difference between two cells is the WEIGHT and not the draw.
# Detached, with the callback fired from the SAME subshell as the work (trap 50) and gated on
# an rc SENTINEL (trap 1: callback-absence is not a result).
set -u
S=/local/home/zegertho/agent/state/modelnorm/artifacts
cd /tmp/modelnorm || exit 9
rc_all=0
# name:weights  — (1,1,1) is the headline equal-weight blend of deliverable C.
for cell in "$@"; do
    NAME="${cell%%:*}"; W="${cell##*:}"
    OUT="$S/runs/blend-$NAME.json"
    echo "=== blend $NAME weights=$W -> $OUT  $(date -u +%FT%TZ) ==="
    uv run --no-sync python "$S/drivers/search_modelnorm.py" \
        --objective blend --anchors "$S/anchors.json" --weights "$W" \
        --budget 10000000 --islands 40 --epochs 55 --overshoot 1.95 --ga-share 0.6 \
        --polish-sweeps 40 --seed 20260728 --processes 40 --resume \
        --out "$OUT" >> "$S/runs/blend-$NAME.log" 2>&1
    r=$?
    echo "=== blend $NAME rc=$r $(date -u +%FT%TZ) ==="
    [ "$r" -ne 0 ] && rc_all=$r
done
echo "$rc_all" > "$S/sweep-rc.txt.tmp" && mv "$S/sweep-rc.txt.tmp" "$S/sweep-rc.txt"
