#!/usr/bin/env bash
# Round-12 Commit 35: closure-paths overnight orchestrator.
#
# Sequentially runs the three named closure paths from
# results/warm_start/SNR_FLOOR.md:
#
#   A: warm_start_stratified10.py      — stratified-10 sampling at the
#                                        Commit-28 warm-start floor.
#                                        Budget: 90 min kill switch.
#                                        Expected wall-clock: ~30-40 min
#                                        (18 cells × ~95 s/cell).
#
#   B: warm_start_sigma_pert_sweep.py  — sub-0.05 sigma_pert sweep.
#                                        Budget: 90 min kill switch.
#                                        Expected wall-clock: ~25-30 min
#                                        (27 cells × ~50 s/cell).
#
#   C: train_weight1_restricted.py     — weight-1 |E_det| restriction
#                                        train + LER benchmark.
#                                        Budget: 6 h kill switch.
#                                        Expected wall-clock: ~2-3 h.
#
# A → B → C is the right order: A and B are the cheap decisive
# experiments (closure verdict in <90 min combined); C is the long one
# that determines the trade-off (matters even if A or B closes the gap,
# because the §VIII benchmark-correspondence question is independent).
#
# All subprocesses run nice -n 19 so the in-flight adaptive sweep
# (PID 35265) keeps CPU priority — though that sweep finished cleanly
# before launch, the nicing is still appropriate hygiene.
#
# Per-path kill switch via scripts/_timed_run.py (Python wrapper, kills
# process group on timeout, exits 124). Live log at
# results/closure_paths/run.log with timestamped per-path headers;
# Python subprocess stdout is teed to that file.
#
# Outputs:
#   results/warm_start_stratified10/{warm_start_stratified10.csv,SUMMARY.md}
#   results/warm_start_sigma_pert/{sigma_pert_sweep.csv,SUMMARY.md}
#   results/weight1_restricted/{*.npz,*.csv,COMPARISON.md}
#   figures/warm_start_stratified10.png
#   figures/warm_start_sigma_pert.png
#   figures/weight1_restricted_{training,ler}.png
#   results/closure_paths/run.log
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

OUT_DIR="$REPO/results/closure_paths"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

# Kill-switch budgets (seconds)
KILL_A=5400   # 90 min
KILL_B=5400   # 90 min
KILL_C=21600  # 6 h

run_path() {
  local label="$1"
  local kill_seconds="$2"
  shift 2
  echo "" >> "$LOG"
  echo "==== ${label} BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) "    \
       "(budget ${kill_seconds}s) ====" >> "$LOG"
  set +e
  nice -n 19 python3 "$SCRIPT_DIR/_timed_run.py" "$kill_seconds" \
    -- python3 -u "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "==== ${label} END $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} ====" \
       >> "$LOG"
  return "$rc"
}

echo "" >> "$LOG"
echo "######## closure-paths overnight BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG"
echo "host: $(uname -n); cwd: $REPO" >> "$LOG"
echo "Paths queued: A (stratified-10) -> B (sigma_pert sub-0.05) -> C (weight-1 train+LER)" >> "$LOG"

# Path A
run_path "PATH_A stratified-10" "$KILL_A" \
  "$SCRIPT_DIR/warm_start_stratified10.py"
RC_A=$?
echo "PATH_A final rc: $RC_A" >> "$LOG"

# Path B (continues regardless of A's outcome — independent experiment)
run_path "PATH_B sigma_pert sub-0.05" "$KILL_B" \
  "$SCRIPT_DIR/warm_start_sigma_pert_sweep.py"
RC_B=$?
echo "PATH_B final rc: $RC_B" >> "$LOG"

# Path C (also continues — the trade-off measurement is useful even if
# A or B closes the gap, because the §IX benchmark-correspondence
# question is independent)
run_path "PATH_C weight1-restricted" "$KILL_C" \
  "$SCRIPT_DIR/train_weight1_restricted.py"
RC_C=$?
echo "PATH_C final rc: $RC_C" >> "$LOG"

echo "" >> "$LOG"
echo "######## closure-paths overnight END $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG"
echo "  A rc=$RC_A  B rc=$RC_B  C rc=$RC_C" >> "$LOG"
echo "  (rc 124 = kill-switch timeout per POSIX timeout convention;" >> "$LOG"
echo "   non-zero non-124 = subprocess error)" >> "$LOG"
