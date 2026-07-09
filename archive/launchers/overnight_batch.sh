#!/usr/bin/env bash
# Round-12 Commit 25 — overnight prioritized batch.
#
# Priority queue:
#   S-1  (~3.5 h): extend n=9 sampling Option-A from seed 0 to all-3-seed
#                  (runs only seeds 1 and 2; Option-A's seed-0 data is
#                  already on disk and gets aggregated in writeup).
#   AT-1 (~4.5 h): ((6,4,3))_4 depth study — all-to-all at 6 layers +
#                  8 layers, plus ring at 6 layers as the matched-depth
#                  fairness baseline.
#   AT-2 (~6 h):   1M-shot LER disambiguation on ((6,4,3))_4 and
#                  ((5,5,3))_5. Lowest priority — confirmatory, not crux.
#
# Each step is wrapped in scripts/_timed_run.py with a generous per-job
# timeout so a single hang doesn't burn the queue.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

LOG_DIR="$REPO/results/overnight"
LOG="$LOG_DIR/run.log"
mkdir -p "$LOG_DIR"

# Per-job kill switches. Generous; the queue is sequential.
KILL_S1=21600    # 6 h for S-1
KILL_AT1=14400   # 4 h per training inside AT-1 (3 trainings)
KILL_AT2=14400   # 4 h per LER inside AT-2 (4 LERs)

run_with_kill() {
  local secs="$1"; shift
  python3 "$SCRIPT_DIR/_timed_run.py" "$secs" -- "$@"
}

echo "==== overnight batch begun $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >>"$LOG"

# ───────────── S-1: n=9 sampling, seeds 1 + 2 ─────────────
echo "" >>"$LOG"
echo "==== S-1 begin $(date -u +%H:%M:%SZ) ====" >>"$LOG"
set +e
run_with_kill "$KILL_S1" nice -n 10 python3 -u scripts/sampling_strategy_experiment.py \
  --code n9 \
  --variants full,uniform-30,uniform-20,stratified-30,stratified-20 \
  --seeds 1,2 --n_steps 1500 \
  --out_dir results/sampling_strategy \
  --fig figures/sampling_strategy_comparison_n9_1500_seeds12.png \
  2>&1 | tee -a "$LOG"
S1_RC=${PIPESTATUS[0]}
set -e
echo "==== S-1 end $(date -u +%H:%M:%SZ) rc=$S1_RC ====" >>"$LOG"

# ───────────── AT-1: ((6,4,3))_4 depth study ─────────────
echo "" >>"$LOG"
echo "==== AT-1 begin $(date -u +%H:%M:%SZ) ====" >>"$LOG"

# All-to-all 6L
echo "--- AT-1: a2a 6L begin $(date -u +%H:%M:%SZ) ---" >>"$LOG"
set +e
run_with_kill "$KILL_AT1" nice -n 10 python3 scripts/train.py \
  --d 4 --n 6 --distance 3 --layers 6 --steps 1500 --seeds 0,1,2 \
  --backend jax --connectivity all-to-all \
  --save_subdir all_to_all_6L --force 2>&1 | tee -a "$LOG"
echo "--- AT-1: a2a 6L end $(date -u +%H:%M:%SZ) rc=${PIPESTATUS[0]} ---" >>"$LOG"
set -e

# All-to-all 8L
echo "--- AT-1: a2a 8L begin $(date -u +%H:%M:%SZ) ---" >>"$LOG"
set +e
run_with_kill "$KILL_AT1" nice -n 10 python3 scripts/train.py \
  --d 4 --n 6 --distance 3 --layers 8 --steps 1500 --seeds 0,1,2 \
  --backend jax --connectivity all-to-all \
  --save_subdir all_to_all_8L --force 2>&1 | tee -a "$LOG"
echo "--- AT-1: a2a 8L end $(date -u +%H:%M:%SZ) rc=${PIPESTATUS[0]} ---" >>"$LOG"
set -e

# Ring 6L (matched-depth fairness baseline)
echo "--- AT-1: ring 6L begin $(date -u +%H:%M:%SZ) ---" >>"$LOG"
set +e
run_with_kill "$KILL_AT1" nice -n 10 python3 scripts/train.py \
  --d 4 --n 6 --distance 3 --layers 6 --steps 1500 --seeds 0,1,2 \
  --backend jax --connectivity ring \
  --save_subdir ring_6L --force 2>&1 | tee -a "$LOG"
echo "--- AT-1: ring 6L end $(date -u +%H:%M:%SZ) rc=${PIPESTATUS[0]} ---" >>"$LOG"
set -e

echo "==== AT-1 end $(date -u +%H:%M:%SZ) ====" >>"$LOG"

# ───────────── AT-2: 1M LER on ((6,4,3))_4 and ((5,5,3))_5 ─────────────
echo "" >>"$LOG"
echo "==== AT-2 begin $(date -u +%H:%M:%SZ) ====" >>"$LOG"

LER_OUT_RING="$REPO/results/benchmarks_1M_ring"
LER_OUT_A2A="$REPO/results/benchmarks_1M_alltoall"
mkdir -p "$LER_OUT_RING" "$LER_OUT_A2A"
P_LOW="0.01,0.02,0.05"
N1M=1000000

# (6,4,3)_4 ring (existing original-campaign params)
echo "--- AT-2: (6,4,3) ring 1M begin $(date -u +%H:%M:%SZ) ---" >>"$LOG"
set +e
run_with_kill "$KILL_AT2" python3 scripts/benchmark_ler_alltoall.py \
  --params "$REPO/results/params/d4_n6_dist3_4L_best3s_seed0.npz" \
  --n_shots "$N1M" --p_range "$P_LOW" --out_dir "$LER_OUT_RING" \
  2>&1 | tee -a "$LOG"
echo "--- AT-2: (6,4,3) ring 1M end $(date -u +%H:%M:%SZ) rc=${PIPESTATUS[0]} ---" >>"$LOG"
set -e

# (6,4,3)_4 all-to-all (campaign all-to-all params)
echo "--- AT-2: (6,4,3) a2a 1M begin $(date -u +%H:%M:%SZ) ---" >>"$LOG"
set +e
run_with_kill "$KILL_AT2" python3 scripts/benchmark_ler_alltoall.py \
  --params "$REPO/results/saved_params/all_to_all/d4_n6_dist3_4L_best3s_seed2.npz" \
  --n_shots "$N1M" --p_range "$P_LOW" --out_dir "$LER_OUT_A2A" \
  2>&1 | tee -a "$LOG"
echo "--- AT-2: (6,4,3) a2a 1M end $(date -u +%H:%M:%SZ) rc=${PIPESTATUS[0]} ---" >>"$LOG"
set -e

# (5,5,3)_5 ring
echo "--- AT-2: (5,5,3) ring 1M begin $(date -u +%H:%M:%SZ) ---" >>"$LOG"
set +e
run_with_kill "$KILL_AT2" python3 scripts/benchmark_ler_alltoall.py \
  --params "$REPO/results/params/d5_n5_dist3_4L_best3s_seed0.npz" \
  --n_shots "$N1M" --p_range "$P_LOW" --out_dir "$LER_OUT_RING" \
  2>&1 | tee -a "$LOG"
echo "--- AT-2: (5,5,3) ring 1M end $(date -u +%H:%M:%SZ) rc=${PIPESTATUS[0]} ---" >>"$LOG"
set -e

# (5,5,3)_5 all-to-all
echo "--- AT-2: (5,5,3) a2a 1M begin $(date -u +%H:%M:%SZ) ---" >>"$LOG"
set +e
run_with_kill "$KILL_AT2" python3 scripts/benchmark_ler_alltoall.py \
  --params "$REPO/results/saved_params/all_to_all/d5_n5_dist3_4L_best3s_seed2.npz" \
  --n_shots "$N1M" --p_range "$P_LOW" --out_dir "$LER_OUT_A2A" \
  2>&1 | tee -a "$LOG"
echo "--- AT-2: (5,5,3) a2a 1M end $(date -u +%H:%M:%SZ) rc=${PIPESTATUS[0]} ---" >>"$LOG"
set -e

echo "==== AT-2 end $(date -u +%H:%M:%SZ) ====" >>"$LOG"

echo "" >>"$LOG"
echo "==== overnight batch ended $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >>"$LOG"
