#!/bin/bash
# Usage: tools-try-flags.sh CASE OUTDIR "label|clang flags|clspv flags" ...   env: UZU_ROOT, M2V_DEFS, CAP (seconds per clspv run, default 60)
# For each variant: clang -> .ll -> sed strip -> clspv; prints rc, seconds, size and spirv-val verdict. Used to find IR shapes clspv accepts.
CASE=$1
OUT=$2
shift 2
H=$(cd "$(dirname "$0")" && pwd)
K=${UZU_ROOT:?}/crates/uzu-engine/src/backends/metal/kernel
CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
mkdir -p "$OUT"
for v in "$@"; do
  IFS='|' read -r label cflags sflags <<< "$v"
  ll="$OUT/$CASE-$label.ll"
  spv="$OUT/$CASE-$label.spv"
  rm -f "$spv"
  # shellcheck disable=SC2086
  clang --target=${TARGET:-spir} -x cl -cl-std=clc++2021 -Xclang -finclude-default-header -cl-ext=-__opencl_c_generic_address_space \
    $cflags -fno-strict-return -cl-kernel-arg-info -w ${M2V_DEFS:-} -I"$H/include" -I"$K" -I"$K/generated" -S -emit-llvm "$H/cases/$CASE.cl" -o "$ll" 2>&1 | head -3
  sed -i -E 's/, !(alias\.scope|noalias) ![0-9]+//g; /llvm\.experimental\.noalias\.scope\.decl/d' "$ll"
  t0=$(date +%s.%N)
  # shellcheck disable=SC2086
  msg=$(timeout "${CAP:-60}" "$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 $sflags "$ll" -o "$spv" 2>&1 | grep -v "^warning" | grep -v "^$" | head -2 | cut -c1-160 | tr '\n' ' ')
  rc=$?
  secs=$(echo "$(date +%s.%N) - $t0" | bc -l | cut -c1-5)
  if [ -s "$spv" ]; then val=$(spirv-val --target-env vulkan1.3 "$spv" 2>&1 | head -1 | cut -c1-150); [ -z "$val" ] && val=VALID; else val="no output"; fi
  echo "[$label] ir=$(wc -c < "$ll")B clspv_secs=$secs size=$(stat -c %s "$spv" 2>/dev/null) val=$val | $msg"
done
