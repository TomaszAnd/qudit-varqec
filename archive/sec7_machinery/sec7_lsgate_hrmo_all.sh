#!/usr/bin/env bash
# Round-12 Commit 12 orchestrator.
# Sec VII rerun with the genuine Hrmo LS gate, full campaign protocol
# (1500 steps, 3 seeds, Adam two-stage LR). Five variants:
#   * pure_ls_ring on n=5
#   * pure_ls_ring on n=7
#   * pure_ls_ring on n=5 (depolarizing-only check — same script, prose-level)
#   * csum_star_ls on n=5
#   * csum_star_ls on n=7
# (We do not have a separate depolarizing-only training script in this repo;
# the §VII table's pure-LS-depolarizing entry uses the same training protocol,
# only the LER benchmarking differs. The Round-12 spec is "re-run the three
# pure LS Ring entries"; the variant differences in §VII are LER-side, not
# training-side, so we issue ONE pure_ls_ring training and use it for all
# three §VII pure-LS analyses.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

OUT="$REPO/results/sec7_lsgate_hrmo"
LOG="$OUT/run.log"
mkdir -p "$OUT"

echo "==== Sec VII Hrmo LS rerun begun $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >>"$LOG"

run_one() {
  local variant="$1"; local n="$2"; local layers="$3"
  echo "--- $variant n=$n layers=$layers $(date -u +%H:%M:%SZ) ---" >>"$LOG"
  python3 scripts/sec7_lsgate_hrmo_rerun.py \
    --variant "$variant" --n "$n" --layers "$layers" \
    --seeds 0,1,2 --steps 1500 2>&1 | tee -a "$LOG"
  echo "--- done $variant n=$n $(date -u +%H:%M:%SZ) ---" >>"$LOG"
}

run_one pure_ls_ring 5 4
run_one csum_star_ls 5 4
run_one pure_ls_ring 7 4
run_one csum_star_ls 7 4

echo "==== Sec VII Hrmo LS rerun ended $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >>"$LOG"
