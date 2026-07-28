#!/bin/bash
# MODELNORM-1 step 2: the per-model "1" anchors, at IDENTICAL budget/islands/epochs for
# every model and every seed (trap 1: a model whose optimum is found less completely gets a
# compressed scale and is silently DOWN-weighted). Two seeds per model so the anchor's
# stability is MEASURED rather than asserted.
# Detached with a push callback fired from the SAME subshell as the work (trap 50: a
# while-pgrep watcher died silently 3 of 3 this session), and gated on an rc SENTINEL
# (trap 1: callback-absence is not a result).
set -u
S=/local/home/zegertho/agent/state/modelnorm/artifacts
cd /tmp/modelnorm || exit 9
rc_all=0
for spec in "$@"; do
    M="${spec%%:*}"; SEED="${spec##*:}"
    case "$SEED" in 20260728) TAG=s1 ;; *) TAG=s2 ;; esac
    OUT="$S/runs/anchor-$M-$TAG.json"
    echo "=== solo:$M seed=$SEED -> $OUT  $(date -u +%FT%TZ) ==="
    uv run --no-sync python "$S/drivers/search_modelnorm.py" \
        --objective "solo:$M" --budget 10000000 --islands 40 --epochs 55 \
        --overshoot 1.95 --ga-share 0.6 --polish-sweeps 40 --seed "$SEED" \
        --processes 40 --resume --out "$OUT" >> "$S/runs/anchor-$M-$TAG.log" 2>&1
    r=$?
    echo "=== solo:$M seed=$SEED rc=$r $(date -u +%FT%TZ) ==="
    [ "$r" -ne 0 ] && rc_all=$r
done
echo "$rc_all" > "$S/anchors-rc.txt.tmp" && mv "$S/anchors-rc.txt.tmp" "$S/anchors-rc.txt"
