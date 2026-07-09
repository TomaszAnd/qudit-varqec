#!/usr/bin/env bash
# R14-7 §B/§C — overnight training launch script.
#
# DEFERRED from the R14-7 session because the machine was memory-saturated
# (~59 MB free, load 300+, ~30 reachability/loky workers). Launching 1.5 GB
# JAX trainings into that state forces catastrophic swap. Fire this script
# WHEN MEMORY FREES (e.g., after reachability finishes).
#
# Recommended invocation: launch 1 seed of each experiment first to see how
# memory behaves; expand to seeds 1,2 once they're past JIT compile.
#
# Memory gate (informational — script does NOT enforce):
#   require free + inactive  > 4 GB  before launching
#   require loky processes   < 10    before launching
# Check with:
#   vm_stat | awk '/Pages free/||/Pages inactive/{print}'
#   ps -ef | grep "[l]oky" | wc -l
set -euo pipefail
cd /Users/tomas/PycharmProjects/PythonProject8/qudit

mkdir -p results/round14_scoping/r14_7a_hoeffding
mkdir -p results/round14_scoping/r14_7b_full_is
mkdir -p results/round14_scoping/r14_7c_dim_scaling

PY=/Users/tomas/qudit_env/bin/python3
TS=$(date -u +%Y%m%dT%H%M%SZ)

echo "=== R14-7 overnight launch ${TS} ==="
vm_stat | awk '/Pages free/||/Pages inactive/{print}'
echo "loky workers: $(ps -ef | grep '[l]oky' | wc -l)"

# §C — ((7,1,3))_4 dim-scaling, 1 seed (most novel result; ~8h under contention)
echo "launching §C dim-scaling seed 0 ..."
nohup nice -n 19 $PY -u scripts/train_r14_7c_dimscaling.py \
    --d 4 --n_qudit 7 --seed 0 \
    --out_dir results/round14_scoping/r14_7c_dim_scaling/ \
    > results/round14_scoping/r14_7c_dim_scaling/seed0.log 2>&1 &
echo "  pid $!"

# §B.1 — Hoeffding-stop sub-100k, 1 seed
echo "launching §B.1 Hoeffding seed 0 ..."
nohup nice -n 19 $PY -u scripts/train_r14_7a_hoeffding.py \
    --seed 0 --n_steps 1100 \
    --out_dir results/round14_scoping/r14_7a_hoeffding/ \
    > results/round14_scoping/r14_7a_hoeffding/seed0.log 2>&1 &
echo "  pid $!"

# §B.2 — full-basis IS sub-100k, 1 seed
echo "launching §B.2 full-IS seed 0 ..."
nohup nice -n 19 $PY -u scripts/train_r14_7b_full_is.py \
    --seed 0 --w1_budget 12 --w2_budget 54 \
    --out_dir results/round14_scoping/r14_7b_full_is/ \
    > results/round14_scoping/r14_7b_full_is/seed0.log 2>&1 &
echo "  pid $!"

sleep 8
echo "--- launched processes ---"
ps -ef | grep -E "train_r14_7[abc]" | grep -v grep | awk '{print "  pid", $2, "nice", "<-", $NF}'
echo
echo "MONITOR: tail -f results/round14_scoping/r14_7?_*/seed0.log"
echo "When seed 0 is past JIT compile (a few minutes) and stepping smoothly,"
echo "  fire seeds 1 and 2 with the same commands (replace --seed 0 with 1/2)."
echo "R14-8 analysis: LER measurement of these codes + extension of"
echo "  weight-enumerator analysis (§H deferred from R14-7 due to memory crunch)."
