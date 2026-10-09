#!/bin/bash
# Run inside the GPU queue wrapper: gpu-job.sh LOGFILE ESTIMATED_SECONDS -- command args...
# Logs the estimated and actual wall seconds of the job (rule: every submit logs its estimated and actual seconds).
LOG=$1
EST=$2
shift 3
{
  echo "== start $(date -u +%FT%TZ) est_s=$EST VK_DRIVER_FILES=${VK_DRIVER_FILES:-system}"
  T0=$(date +%s.%N)
  "$@"
  RC=$?
  T1=$(date +%s.%N)
  echo "== end $(date -u +%FT%TZ) rc=$RC actual_s=$(echo "$T1 - $T0" | bc -l | cut -c1-6)"
} >> "$LOG" 2>&1
