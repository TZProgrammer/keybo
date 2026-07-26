#!/bin/bash
# REHUNT — the 7 targeted hunts holding the 14 flipped verdicts, re-run against the CORRECTED
# wfd bar, plus the 6 placebo/real frames that WSCISSOR-ARMB-1's "wscissor is inert" null needs.
#
# ONE HUNT PER FROZEN CELL, same (corpus, arm, frame) and the same warm-start field the frozen
# hunt used, so the ONLY thing that changes is the wfd axis. That is the experiment.
#
# Trap 22: this runs DETACHED (a foreground Bash call is clamped ~10 min and a killed run leaves
# no sentinel). Trap 7: rehunt_hunt.py checkpoints EVERY epoch and resumes, so a reboot costs one
# epoch, not the arm.
#
# Budget: 6 seeds x 12 restarts x 60000 iters over 6 targets = 36 workers/cell. The frozen arms
# delivered 9.59-10.12M unique evals each; this delivers a comparable count per cell (measured,
# not multiplied out — `unique_layouts_total` in each output).
set -x
cd /tmp/rehunt || exit 1
export PYTHONPATH=/tmp/rehunt/keybo-e2e
LOGS=/tmp/rehunt/keybo-e2e/runs/logs
RUNS=/tmp/rehunt/keybo-e2e/runs
mkdir -p "$LOGS" "$RUNS/ckpt"

ITERS=60000
RESTARTS=12
SEEDS=6
PROCS=24

# --- the 8 warm starts of the frozen noanchor 10-axis cells (config.extra_warm) --------------
NOANCHOR_TEN_WARM=(
  "pyou'vgdnmheai.cstrlkjz,-wfbxq"
  "uyog.bdnsleiat,pchmrz-'kjfwvxq"
  "pyou,vgdnlheai.cstmrk'zj-wfbqx"
  "pyou,vgdnlheai-cstmrkjz.'wfbxq"
  "mldfbxhae-crstp.nouiwqvgky,jz'"
  "pyou-gfdnlheai.bstrmkj',zwcvxq"
  "pyou-vfdnlheai.cstmrk'z,jgwbqx"
  "uyo,.fdnsleiatkpchmrq-xg'bwvzj"
)
# --- the 6 warm starts of each frozen twelve-axis cell (warm_from_searched_field) ------------
BLEND_TWELVE_WARM=(
  "jf'dcxubvqlieo-yrsnazpmhgt.,kw" "jf'dcxubvqlieo-syrnazpmhgt.,kw"
  "jf',cdubvqerly-oisnazpmhgt.xkw" "jf',cdubvqliey-orsnazpmhgt.xkw"
  "jf',cdubvqeily-orsnazpmhgt.xkw" "jf',cdubvqlieo-yrsnazpmhgt.xkw"
)
IWEB_TWELVE_WARM=(
  "'-bcpmyhfkltuirsaneozwg,.dvxjq" "z-bpcmyhfkrtuilsaneogw',.dvxjq"
  "'-bcpmytfkriuhlsaneogwz,.dvxqj" "'-bcpmytfkriuhslaneogwz,.dvxqj"
  "'-bcpmytfkriuhlsaneogwz,.dvxjq" "'-bcpmytfkrouhlsaneigwz,.dvxjq"
)
NOANCHOR_TWELVE_WARM=(
  "zx,.'cphwkoirnlasteyq-vumdbgfj" "zx,.'cphwkoirylnsteaq-vumdbgfj"
  "zx,.'cphwkoirnlastyeq-vumdbgfj" "zx,.'ychwkaernlosptiq-vumdbgfj"
  "zx,.'ychwkternlospaiq-vumdbgfj" "zx,.'ychwkaernposltiq-vumdbgfj"
)

launch () {  # launch <corpus> <arm> <frame> <tag> [warm...]
  local corpus=$1 arm=$2 frame=$3 tag=$4; shift 4
  (
    uv run --no-sync python keybo-e2e/rehunt_hunt.py \
      --corpus "$corpus" --arm "$arm" --frame "$frame" --wfd corrected \
      --iters $ITERS --restarts $RESTARTS --seeds $SEEDS --procs $PROCS \
      ${1:+--warm "$@"} \
      --ckpt-dir "$RUNS/ckpt" \
      --out "$RUNS/rehunt-${tag}.json"
    echo "REHUNT-DONE tag=${tag} rc=$?"
  ) > "$LOGS/rehunt-${tag}.log" 2>&1 &
}

# ============ THE 7 FROZEN CELLS (all 14 flips live here) ====================================
launch blend    A ten    blend-armA-ten
launch blend    B ten    blend-armB-ten
launch noanchor A ten    noanchor-armA-ten    "${NOANCHOR_TEN_WARM[@]}"
launch noanchor B ten    noanchor-armB-ten    "${NOANCHOR_TEN_WARM[@]}"
launch blend    A twelve blend-armA-twelve    "${BLEND_TWELVE_WARM[@]}"
launch iweb     A twelve iweb-armA-twelve     "${IWEB_TWELVE_WARM[@]}"
launch noanchor A twelve noanchor-armA-twelve "${NOANCHOR_TWELVE_WARM[@]}"
wait

# ============ NULL #2: the wscissor-inert placebo differencing ===============================
# ten -> wide11 changes TWO things (axis added AND frame grew), so the marginal effect must be
# read placebo(narrow11) -> real(wide11), never ten -> wide11 (trap 17). Run in a second wave so
# the first wave gets the whole host.
launch iweb     A narrow11 iweb-armA-narrow11     "${IWEB_TWELVE_WARM[@]}"
launch iweb     A wide11   iweb-armA-wide11       "${IWEB_TWELVE_WARM[@]}"
launch iweb     A ten      iweb-armA-ten          "${IWEB_TWELVE_WARM[@]}"
launch blend    A narrow11 blend-armA-narrow11    "${BLEND_TWELVE_WARM[@]}"
launch blend    A wide11   blend-armA-wide11      "${BLEND_TWELVE_WARM[@]}"
launch noanchor A narrow11 noanchor-armA-narrow11 "${NOANCHOR_TWELVE_WARM[@]}"
launch noanchor A wide11   noanchor-armA-wide11   "${NOANCHOR_TWELVE_WARM[@]}"
wait

{
  echo "ALL-REHUNTS-EXITED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  grep -h '^REHUNT-DONE' "$LOGS"/rehunt-*.log | sort
  echo "--- dominated_targets per cell ---"
  grep -h 'done; dominated=' "$LOGS"/rehunt-*.log | sort
} > "$RUNS/rehunt-complete.marker"
