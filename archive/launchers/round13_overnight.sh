#!/usr/bin/env bash
# Round-13 Commit R13-5: scoping experiments orchestrator.
#
# Sequentially runs the two Round-13 batch experiments:
#
#   Job 1: train_w1inv_w2ortho_sweep.py     — (n,q) sweep on three arms
#                                              (full K3, w1-only, w1-inv +
#                                              uniform-20 w2). 4 configs ×
#                                              3 arms × 3 seeds × 1500
#                                              steps. Plus LER at 100k
#                                              for the (9,3) Arm 3.
#                                              Kill switch: OFF (2026-05-21).
#                                              Expected wall-clock ~15-20 h.
#
#   Job 2: sampling_technique_bakeoff.py    — 8 techniques on weight-≤2
#                                              ((9,3,3))_3 ring 4L,
#                                              1500 steps × 3 seeds.
#                                              Kill switch: OFF (2026-05-21).
#                                              Expected wall-clock ~30-45 h
#                                              (slower techniques like
#                                              stratified-10 / rosalin /
#                                              adaptive take longer
#                                              steps-to-converge).
#
# Job 1 → Job 2 order: Job 1 is the larger experiment whose results
# inform whether the bake-off is worth interpreting carefully; Job 2
# is a deeper measurement at a fixed config that benefits from having
# Job 1's "structural recipe works on (9,3)" baseline in hand.
#
# All subprocesses run nice -n 19 — same convention as
# closure_paths_overnight.sh from Round-12. Kill switches removed
# 2026-05-21 (see scripts/_timed_run.py header comment); _timed_run
# is still in the call chain so the per-job exit code is propagated
# cleanly and the run.log header per job is consistent. To re-enable
# a budget for a specific run, change "off" to a positive seconds
# value on the relevant run_job invocation below.
#
# Live log at results/round13_scoping/run.log with timestamped per-job
# BEGIN/END headers; subprocess stdout is teed via PIPESTATUS.
#
# Outputs (under results/round13_scoping/):
#   w1inv_w2ortho_sweep/{<config>_*.npz, *_summary.json, SUMMARY.md,
#                       ler_d3_n9_dist3_4L_w1inv_w2ortho_100000.csv}
#   technique_bakeoff/{<technique>_*.npz, *_summary.json, BAKEOFF.md}
#   run.log
#   figures/w1inv_w2ortho_sweep_*.png, sampling_technique_bakeoff_n9_1500.png
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

OUT_DIR="$REPO/results/round13_scoping"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

# Kill-switch budgets (seconds, or "off" / "0" to disable)
# Disabled 2026-05-21 — see header comment.
KILL_SWEEP=off
KILL_BAKEOFF=off

run_job() {
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
echo "######## round-13 scoping BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG"
echo "host: $(uname -n); cwd: $REPO" >> "$LOG"
echo "Jobs queued: Job1 (w1inv_w2ortho_sweep) -> Job2 (technique_bakeoff)" >> "$LOG"

# Job 1
run_job "JOB1 w1inv_w2ortho_sweep" "$KILL_SWEEP" \
  "$SCRIPT_DIR/train_w1inv_w2ortho_sweep.py"
RC_SWEEP=$?
echo "JOB1 final rc: $RC_SWEEP" >> "$LOG"

# Job 2 — continues regardless of Job 1's outcome (independent experiments)
run_job "JOB2 sampling_technique_bakeoff" "$KILL_BAKEOFF" \
  "$SCRIPT_DIR/sampling_technique_bakeoff.py"
RC_BAKEOFF=$?
echo "JOB2 final rc: $RC_BAKEOFF" >> "$LOG"

echo "" >> "$LOG"
echo "######## round-13 scoping END $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG"
echo "  JOB1 rc=$RC_SWEEP  JOB2 rc=$RC_BAKEOFF" >> "$LOG"
echo "  (rc 124 = kill-switch timeout per POSIX timeout convention)" >> "$LOG"
