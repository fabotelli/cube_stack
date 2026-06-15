#!/usr/bin/env bash
# Non-babysitting watcher: log milestones + GPU until the detached pipeline
# (run_validation_then_full.sh, PID passed as $1) finishes, then exit.
set -uo pipefail
cd ~/mujoco-test/cube_stack
PIPE_PID="${1:?need pipeline pid}"
LOG=pipeline_watch.log
gpu_seen=0
echo "=== watch start $(date -u +%FT%TZ) pid=$PIPE_PID ===" >> "$LOG"
while kill -0 "$PIPE_PID" 2>/dev/null; do
  ts=$(date -u +%FT%TZ)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  if pgrep -f train_bc_cube_stack.py >/dev/null 2>&1; then
    phase="TRAIN"
    if [ "$gpu_seen" = 0 ]; then
      echo "$ts  *** TRAIN STARTED — GPU now: $gpu ***" >> "$LOG"
      gpu_seen=1
    fi
    last=$(tail -1 train_cube_stack.log 2>/dev/null)
  elif [ -f dataset_cube_stack.npz ]; then
    phase="MERGE/POST-COLLECT"; last="dataset_cube_stack.npz present"
  else
    phase="COLLECT"; last=$(tail -1 collect_cube_stack.log 2>/dev/null)
  fi
  echo "$ts  [$phase] gpu=$gpu | $last" >> "$LOG"
  sleep 120
done
echo "=== pipeline pid $PIPE_PID exited $(date -u +%FT%TZ) ===" >> "$LOG"
echo "FINAL train tail:" >> "$LOG"; tail -20 train_cube_stack.log >> "$LOG" 2>/dev/null
echo "watch done"
