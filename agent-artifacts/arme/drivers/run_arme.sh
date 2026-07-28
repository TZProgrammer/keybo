#!/bin/bash
# ARM E at the SAME budget/islands/epochs/seed/overshoot/ga-share/polish-sweeps as arm D
# (and therefore as OPTEVIDENCE-1's arm A/B), differing ONLY in --arm archive: the ARCHIVE-fitted
# weights JSON instead of the random400-fitted one. Everything else is arm D's engine, verified by
# gate 2's arm-A AND arm-D positive controls against the frozen drivers.
# Writes an rc sentinel; the sentinel — not the callback — is what any verdict is gated on
# (trap 50: a watcher subshell can die silently, so callback-absence is not a result).
set -u
S=/local/home/zegertho/agent/state/arme/artifacts
cd /tmp/arme || exit 9
echo "=== ARM E start $(date -u +%FT%TZ) ==="
uv run --no-sync python "$S/drivers/search_arme.py" \
    --arm archive --budget 10000000 --islands 40 --epochs 55 --overshoot 1.95 \
    --ga-share 0.6 --polish-sweeps 40 --seed 20260728 --resume \
    --out "$S/runs/arm-archive.json" > "$S/runs/arm-archive.log" 2>&1
r=$?
echo "=== ARM E rc=$r $(date -u +%FT%TZ) ==="
echo "$r" > "$S/arme-rc.txt.tmp" && mv "$S/arme-rc.txt.tmp" "$S/arme-rc.txt"
ticket arme --prompt "[ARME-DONE] arm E finished rc=$r — sentinel $S/arme-rc.txt, result $S/runs/arm-archive.json"
