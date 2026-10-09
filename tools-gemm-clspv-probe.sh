#!/bin/bash
# Diagnose why clspv does not finish on uzu's GEMM IR. Usage: tools-gemm-clspv-probe.sh GEMM.ll OUTDIR [SECONDS]
LL=$1
OUT=$2
CAP=${3:-100}
CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
mkdir -p "$OUT"
run() {
  local name=$1
  shift
  local t0
  t0=$(date +%s.%N)
  timeout "$CAP" "$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 "$@" "$LL" -o "$OUT/$name.spv" > "$OUT/$name.log" 2>&1
  echo "$name rc=$? secs=$(echo "$(date +%s.%N) - $t0" | bc -l | cut -c1-6) size=$(stat -c %s "$OUT/$name.spv" 2>/dev/null)"
}
echo "IR: $(wc -c < "$LL") bytes, $(grep -c '^define' "$LL") functions, $(grep -c 'call ' "$LL") calls"
run noopt --cl-opt-disable
run default
