#!/bin/bash
# ARM D at the SAME budget/islands/epochs/seed/overshoot/ga-share/polish-sweeps as
# OPTEVIDENCE-1's arm A, differing ONLY in --arm domain (the CLAMP policy).
# Writes an rc sentinel; the sentinel — not the callback — is what any verdict is gated on
# (trap 50: a watcher subshell can die silently, so callback-absence is not a result).
set -u
S=/local/home/zegertho/agent/state/armd/artifacts
cd /tmp/domainfix || exit 9
echo "=== ARM D start $(date -u +%FT%TZ) ==="
uv run --no-sync python "$S/drivers/search_armd.py" \
    --arm domain --budget 10000000 --islands 40 --epochs 55 --overshoot 1.95 \
    --ga-share 0.6 --polish-sweeps 40 --seed 20260728 --resume \
    --out "$S/runs/arm-domain.json" > "$S/runs/arm-domain.log" 2>&1
r=$?
echo "=== ARM D rc=$r $(date -u +%FT%TZ) ==="
echo "$r" > "$S/armd-rc.txt.tmp" && mv "$S/armd-rc.txt.tmp" "$S/armd-rc.txt"
ticket armd --prompt "[ARMD-DONE] arm D finished rc=$r — sentinel $S/armd-rc.txt, result $S/runs/arm-domain.json"
