#!/usr/bin/env bash
# Round-14 R14-3-launch: Meth-weighted training on a2a ((9,3,3))_3 with
# stratified-10 sampling.
#
# Per R14-3-audit-1 verdict (Verdict 2) + r14_3_choices.md (commit
# 22cda9b): use noise_model='physical' (level-dependent f_k=k) as the
# headline arm, matching the VarQEC paper's app:kraus reading and
# R12/13/14's discrete-basis derivation.
#
#   JOB 1 (R14-3a, headline): train_r14_3_meth.py --mode physical
#       3 seeds × 1500 steps, weights from meth_pauli_weights.npz
#       (Pauli-twirled Meth physical channel onto R12/13/14 closure
#       basis). Expected ~18-26h wall-clock.
#
#   JOB 2 (R14-3b, sensitivity): train_r14_3_meth.py --mode simplified
#       1 seed × 1500 steps, weights from meth_pauli_weights_simplified.npz
#       (Meth literal Eq. J3, uniform f=2). Sanity check that the
#       physical/simplified distinction produces operationally
#       different trained codes. Expected ~6-10h wall-clock.
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
echo "######## round-14 R14-3 BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG"
echo "host: $(uname -n); cwd: $REPO" >> "$LOG"
echo "Jobs: R14-3a (Meth physical, headline)" >> "$LOG"
echo "R14-3b (simplified sensitivity) deferred — needs meth_pauli_weights.py" >> "$LOG"
echo "  re-run with --noise_model simplified to generate the weights npz." >> "$LOG"

run_job "R14-3a Meth_physical" \
  "$SCRIPT_DIR/train_r14_3_meth.py" --mode physical
RC_A=$?
echo "R14-3a final rc: $RC_A" >> "$LOG"

echo "" >> "$LOG"
echo "######## round-14 R14-3 END $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG"
echo "  R14-3a rc=$RC_A" >> "$LOG"
