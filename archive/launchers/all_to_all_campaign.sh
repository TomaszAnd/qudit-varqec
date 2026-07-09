#!/usr/bin/env bash
# Round-12 Commit 13: all-to-all retraining of the campaign codes.
#
# Sequentially trains ((9,3,3))_3, ((6,4,3))_4, ((5,5,3))_5 with
# --connectivity all-to-all (1500 steps, 3 seeds), then LER-benchmarks
# each. 8h per-code kill switch via the `timeout` binary (gtimeout on
# macOS via coreutils). If `gtimeout` / `timeout` is not available, the
# script proceeds without a kill switch and relies on manual monitoring.
#
# Outputs:
#   results/saved_params/all_to_all/<tag>_seed*.npz
#   results/benchmarks_30k_alltoall/ler_*.csv
#   results/benchmarks_100k_alltoall/ler_*.csv
#   results/saved_params/all_to_all/run.log   (live log)
#   results/saved_params/all_to_all/COMPARISON.md (written by the
#     companion all_to_all_compare.py once benchmarks exist)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

OUT_PARAMS="$REPO/results/saved_params/all_to_all"
OUT_BENCH_QUTRIT="$REPO/results/benchmarks_30k_alltoall"
OUT_BENCH_LARGE="$REPO/results/benchmarks_100k_alltoall"
LOG="$OUT_PARAMS/run.log"
mkdir -p "$OUT_PARAMS" "$OUT_BENCH_QUTRIT" "$OUT_BENCH_LARGE"

# Per-code 8h kill switch via scripts/_timed_run.py (Python wrapper
# that kills the process group on timeout; exits 124 on kill, matching
# POSIX `timeout` convention). macOS doesn't ship `timeout` by default
# and we don't want to require coreutils.
KILL_SECONDS=28800  # 8 hours

run_with_kill() {
  python3 "$SCRIPT_DIR/_timed_run.py" "$KILL_SECONDS" -- "$@"
}

echo "==== all-to-all campaign begun $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >> "$LOG"

train_one() {
  local d=$1
  local n=$2
  local layers=$3
  local label="$4"
  local shots=$5
  local bench_dir=$6

  echo "" >> "$LOG"
  echo "--- $label TRAIN begin $(date -u +%H:%M:%SZ) ---" >> "$LOG"
  set +e
  run_with_kill python3 scripts/train.py \
    --d "$d" --n "$n" --distance 3 --layers "$layers" \
    --steps 1500 --seeds 0,1,2 --backend jax \
    --connectivity all-to-all --save_subdir all_to_all \
    --force 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e

  echo "--- $label TRAIN end $(date -u +%H:%M:%SZ) rc=$rc ---" >> "$LOG"

  if [ "$rc" -eq 124 ]; then
    echo "!!! $label EXCEEDED 8h KILL SWITCH; skipping LER benchmark" >> "$LOG"
    return
  elif [ "$rc" -ne 0 ]; then
    echo "!!! $label TRAIN FAILED rc=$rc; skipping LER benchmark" >> "$LOG"
    return
  fi

  # Find the best-seed npz that train.py wrote
  local pattern="$OUT_PARAMS/d${d}_n${n}_dist3_${layers}L_best3s_seed*.npz"
  local npz=$(ls -t $pattern 2>/dev/null | head -1)
  if [ -z "$npz" ]; then
    echo "!!! $label could not locate trained npz under $pattern" >> "$LOG"
    return
  fi

  echo "--- $label LER begin $(date -u +%H:%M:%SZ) npz=$(basename $npz) ---" >> "$LOG"
  set +e
  run_with_kill python3 scripts/benchmark_ler_alltoall.py \
    --params "$npz" --n_shots "$shots" --out_dir "$bench_dir" \
    2>&1 | tee -a "$LOG"
  local rc2=${PIPESTATUS[0]}
  set -e
  echo "--- $label LER end $(date -u +%H:%M:%SZ) rc=$rc2 ---" >> "$LOG"
}

# Run order is wall-clock-aware: ((6,4,3))_4 fits within 8h; ((9,3,3))_3
# and ((5,5,3))_5 are expected to hit the kill switch with the full
# 3-seed protocol (each is ~11h projected). Putting the safe one first
# guarantees at least one complete result lands even if the longer ones
# are killed mid-third-seed.
train_one 4 6 4 "((6,4,3))_4 all-to-all" 100000  "$OUT_BENCH_LARGE"
train_one 3 9 4 "((9,3,3))_3 all-to-all"  30000  "$OUT_BENCH_QUTRIT"
train_one 5 5 4 "((5,5,3))_5 all-to-all" 100000  "$OUT_BENCH_LARGE"

echo "" >> "$LOG"
echo "==== all-to-all campaign ended $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >> "$LOG"
