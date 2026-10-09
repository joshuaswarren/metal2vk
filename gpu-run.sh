#!/bin/bash
# Run on the GPU host inside whatever GPU queue wrapper it uses. Usage: gpu-run.sh OUTDIR [cases...]   env: SPVDIR (default ~/scratch/metal2vk/out),
# VK_DRIVER_FILES (ICD; the fork ICD exposes cooperative matrix), RUN_BASELINE=1 to also time the omarchy-mlx kernels.
# Logs the estimated and actual wall seconds of the whole job. Each submit inside is sized by m2v-test.py to about 2 s.
set -u
OUT=${1:?outdir}
shift
mkdir -p "$OUT"
PY=${PYTHON:-python3}
H=$HOME/scratch/metal2vk/repo
SPV=${SPVDIR:-$HOME/scratch/metal2vk/out}
T0=$(date +%s)
{
  echo "== start $(date -u +%FT%TZ) host $(hostname -s) kernel $(uname -r) ICD=${VK_DRIVER_FILES:-system} spv=$SPV"
  vulkaninfo --summary 2>&1 | grep -E "deviceName|driverInfo" | head -4
  echo "== m2v-test"
  "$PY" "$H/m2v-test.py" "$OUT/spike" "$HOME/scratch/metal2vk/m2v-run" "$SPV" "$@" 2>&1
  if [ "${RUN_BASELINE:-0}" = 1 ]; then
    echo "== omarchy-mlx baseline"
    "$PY" "$H/m2v-mlx-base.py" "$OUT/baseline.json" 2>&1 | tail -12
  fi
  echo "== end $(date -u +%FT%TZ) actual_s=$(( $(date +%s) - T0 ))"
} > "$OUT/log.txt" 2>&1
