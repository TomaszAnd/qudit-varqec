#!/usr/bin/env bash
# Round-14 Commit R14-2: H4 (Z-only basis) + H5(b) (anti-correlated stratification) overnight.
#
# Per the R14-1 H1 diagnostic verdict (encoder is asymmetric with
# max std/mean = 0.48 across qudits, far above the 15% S_n-symmetry
# threshold), Round 14 pursues the conservative path: H4 and H5(b)
# on a2a ((9,3,3))_3, deferring H6 (permutation-invariant ansatz) to
# Round 15.
#
#   JOB 1 (H4 Z-only): train_h4_h5b_overnight.py --mode h4_z_only
#       Train ((9,3,3))_3 a2a 4L on Z-only weight-≤2 basis (172 ops vs
#       full 685 — 4x reduction by basis restriction). 1500 steps × 3
#       seeds, then inline LER at 100k on the standard Pauli
#       depolarising channel. Wall-clock estimate ~24-30 h.
#
#   JOB 2 (H5b anti-correlated): train_h4_h5b_overnight.py
#       --mode h5b_anticorrelated
#       Train ((9,3,3))_3 a2a 4L with w1-inv + cyclic-stratified-10 on
#       w2. Deterministic cyclic schedule visits every w2 op once per
#       10-step window (vs Round-13 IID stratified-10 which resamples
#       independently). 1500 steps × 3 seeds. Wall-clock ~12-18 h.
#
# nice -n 19, no kill switch (R13-6 policy). Live log at
# results/round14_scoping/run.log via tee through _timed_run.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

OUT_DIR="$REPO/results/round14_scoping"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

run_job() {
  local label="$1"
  shift
  echo "" >> "$LOG"
  echo "==== ${label} BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >> "$LOG"
  set +e
  nice -n 19 python3 "$SCRIPT_DIR/_timed_run.py" off \
    -- python3 -u "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "==== ${label} END $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc} ====" \
       >> "$LOG"
  return "$rc"
}

echo "" >> "$LOG"
echo "######## round-14 overnight BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG"
echo "host: $(uname -n); cwd: $REPO" >> "$LOG"
echo "Jobs: JOB1 (H4 Z-only) -> JOB2 (H5b cyclic-stratified-10)" >> "$LOG"

run_job "JOB1 H4_z_only" \
  "$SCRIPT_DIR/train_h4_h5b_overnight.py" --mode h4_z_only
RC_H4=$?
echo "JOB1 final rc: $RC_H4" >> "$LOG"

run_job "JOB2 H5b_anticorrelated" \
  "$SCRIPT_DIR/train_h4_h5b_overnight.py" --mode h5b_anticorrelated
RC_H5B=$?
echo "JOB2 final rc: $RC_H5B" >> "$LOG"

echo "" >> "$LOG"
echo "######## round-14 overnight END $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG"
echo "  H4 rc=$RC_H4  H5b rc=$RC_H5B" >> "$LOG"
