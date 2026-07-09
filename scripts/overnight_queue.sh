#!/usr/bin/env bash
# Overnight sequential queue runner (Stage S2/S3).
# Runs jobs one at a time, nice -19, tees each to a timestamped log, updates a
# status file after each, SKIPS to the next job on failure, and stops before a
# soft wall budget so nothing runs into the workday. NO concurrent jobs.
# FULL Meth noise always (each job passes --noise full via the scripts' defaults).
set -u
REPO="/Users/tomas/PycharmProjects/qudit-varqec"
PY="/Users/tomas/qudit_env/bin/python3"
BPR="$REPO/results/best_practice_runs"
LOGDIR="$BPR/overnight_logs"
STATUS="$BPR/overnight_status.json"
mkdir -p "$LOGDIR" "$BPR/stage_hprime/n9" "$BPR/campaign" "$BPR/budget_nkq"
export PYTHONPATH="$REPO"

START=$(date +%s)
BUDGET=${BUDGET:-32400}     # ~9h soft wall budget for the whole queue
TS() { date +%Y%m%d_%H%M%S; }
elapsed() { echo $(( $(date +%s) - START )); }

# jobs: "name|command"  (command runs from $REPO)
JOBS=(
  "A_n9_budget_sweep|$PY -u scripts/stage_hprime_budget_sweep.py --n 9 --d 3 --layers 16 --fractions 0.2,0.35,0.5,1.0 --seeds 3 --steps 400 --tag n9 --outdir $BPR/stage_hprime/n9 --write-f-campaign --f5-star 0.40"
  "B_fig6_campaign|$PY -u scripts/campaign_train.py --seeds 3 --steps 2000"
  "C_fig8_budget_nkq_d4n6|$PY -u scripts/stage_hprime_budget_sweep.py --n 6 --d 4 --layers 16 --fractions 0.2,0.35,0.5,1.0 --seeds 2 --steps 400 --tag n6d4 --outdir $BPR/budget_nkq"
)

echo "[]" > "$STATUS"
write_status() {  # name pid start_ts exit wall status
  $PY - "$STATUS" "$1" "$2" "$3" "$4" "$5" "$6" << 'PYEOF'
import json,sys
path,name,pid,start,ex,wall,st=sys.argv[1:8]
try: d=json.load(open(path))
except Exception: d=[]
d=[x for x in d if x.get("job")!=name]
d.append({"job":name,"pid":int(pid),"start":start,"exit":ex,"wall_s":wall,"status":st})
json.dump(d,open(path,"w"),indent=2)
PYEOF
}

cd "$REPO" || exit 1
echo "QUEUE START $(TS)  budget=${BUDGET}s  pid=$$" | tee -a "$LOGDIR/queue.log"

for entry in "${JOBS[@]}"; do
  name="${entry%%|*}"; cmd="${entry#*|}"
  rem=$(( BUDGET - $(elapsed) ))
  if [ "$rem" -lt 600 ]; then
    echo "[$(TS)] SKIP $name — wall budget exhausted (${rem}s left)" | tee -a "$LOGDIR/queue.log"
    write_status "$name" 0 "$(TS)" "-" 0 "SKIPPED_WALL"
    continue
  fi
  log="$LOGDIR/${name}_$(TS).log"
  echo "[$(TS)] START $name (rem ${rem}s) -> $log" | tee -a "$LOGDIR/queue.log"
  write_status "$name" 0 "$(TS)" "-" 0 "RUNNING"
  jstart=$(date +%s)
  nice -n 19 bash -c "$cmd" > "$log" 2>&1
  ec=$?
  jwall=$(( $(date +%s) - jstart ))
  if [ "$ec" -eq 0 ]; then
    echo "[$(TS)] DONE  $name (exit 0, ${jwall}s)" | tee -a "$LOGDIR/queue.log"
    write_status "$name" 0 "$(TS)" "0" "$jwall" "DONE"
  else
    echo "[$(TS)] FAIL  $name (exit $ec, ${jwall}s) — skipping to next" | tee -a "$LOGDIR/queue.log"
    write_status "$name" 0 "$(TS)" "$ec" "$jwall" "FAILED"
  fi
done
echo "QUEUE END $(TS)  total $(elapsed)s" | tee -a "$LOGDIR/queue.log"
