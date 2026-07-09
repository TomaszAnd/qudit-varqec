#!/usr/bin/env bash
# Round-12 Commit 20 — 1M-shot LER disambiguation on ((9,3,3))_3.
#
# Six benchmark runs: {ring, all-to-all} × p ∈ {0.01, 0.02, 0.05}.
# Both arms use the same scripts/benchmark_ler_alltoall.py (it reads
# connectivity / connections from saved npz metadata; the ring file
# saved before the metadata addition defaults to ring connectivity).
#
# Outputs:
#   results/benchmarks_1M_ring/ler_d3_n9_dist3_4L_best3s_seed1_1000000.csv
#   results/benchmarks_1M_alltoall/ler_d3_n9_dist3_4L_best3s_seed1_1000000.csv
#   results/benchmarks_1M_alltoall/DISAMBIGUATION.md (written by separate
#     Python summary script after the runs complete)
#   results/benchmarks_1M_alltoall/run.log (live log)
#
# Wall-clock projection: scaling the 30k all-to-all sweep (8 p-values
# in 29 min ⇒ ~3.6 min/p at 30k) to 1M shots × 3 p-values × 2 arms is
# ~3.6 × (1e6/3e4) × 3 × 2 = ~1440 min = ~24 h naive. Empirically the
# 100k-ring sweep took ~258s per p (1M extrapolated ~43 min/p × 3 ×
# 2 = ~4.3 h). The factored decoder is the same for both arms, so the
# realistic budget is closer to the empirical extrapolation (3-5 h).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

OUT_RING="$REPO/results/benchmarks_1M_ring"
OUT_A2A="$REPO/results/benchmarks_1M_alltoall"
LOG="$OUT_A2A/run.log"
mkdir -p "$OUT_RING" "$OUT_A2A"

PARAMS_RING="$REPO/results/params/d3_n9_dist3_4L_best3s_seed1.npz"
PARAMS_A2A="$REPO/results/saved_params/all_to_all/d3_n9_dist3_4L_best3s_seed1.npz"
P_RANGE="0.01,0.02,0.05"
N_SHOTS=1000000

echo "==== 1M-shot LER disambig begun $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >>"$LOG"
echo "params (ring): $PARAMS_RING" >>"$LOG"
echo "params (a2a):  $PARAMS_A2A" >>"$LOG"
echo "p_range: $P_RANGE; n_shots: $N_SHOTS" >>"$LOG"

run_one() {
  local label="$1"; local params="$2"; local out_dir="$3"
  echo "" >>"$LOG"
  echo "--- $label begin $(date -u +%H:%M:%SZ) ---" >>"$LOG"
  python3 scripts/benchmark_ler_alltoall.py \
    --params "$params" --n_shots "$N_SHOTS" --p_range "$P_RANGE" \
    --out_dir "$out_dir" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "--- $label end $(date -u +%H:%M:%SZ) rc=$rc ---" >>"$LOG"
}

run_one "RING 1M" "$PARAMS_RING" "$OUT_RING"
run_one "A2A 1M"  "$PARAMS_A2A"  "$OUT_A2A"

echo "" >>"$LOG"
echo "==== 1M-shot LER disambig ended $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >>"$LOG"
