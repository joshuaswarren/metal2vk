#!/bin/bash
# metal2vk: Metal kernel (through the metal_stdlib shim, C++ for OpenCL) -> LLVM IR (clang) -> Vulkan SPIR-V (clspv).
# Usage: compile.sh OUTDIR [case ...]
# env: UZU_ROOT (uzu checkout), CLSPV (patched clspv binary), CLANG, M2V_DEFS (extra -D flags, e.g. -DM2V_NO_COOPMAT)
# clang is the front end because it accepts the generic-address-space-free C++ for OpenCL dialect that makes Metal's `thread`
# qualifier work (see include/metal_stdlib); clspv then lowers the IR, including the m2v_mma_f32 cooperative matrix builtin.
set -u
OUT=${1:?outdir}
shift
H=$(cd "$(dirname "$0")" && pwd)
UZU_ROOT=${UZU_ROOT:?set UZU_ROOT to a uzu checkout}
CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
CLANG=${CLANG:-clang}
K=$UZU_ROOT/crates/uzu-engine/src/backends/metal/kernel
CASES=${*:-activation softmax tilematmul gemm}
mkdir -p "$OUT"
for c in $CASES; do
  echo "== $c"
  # shellcheck disable=SC2086
  "$CLANG" --target=spir -x cl -cl-std=clc++2021 -Xclang -finclude-default-header \
    -cl-ext=-__opencl_c_generic_address_space -O2 -fno-strict-return -cl-kernel-arg-info -w ${M2V_DEFS:-} \
    -I"$H/include" -I"$K" -I"$K/generated" -S -emit-llvm "$H/cases/$c.cl" -o "$OUT/$c.ll" 2>&1 | head -"${LINES_MAX:-20}"
  [ -s "$OUT/$c.ll" ] || continue
  # clang -O2 inlining attaches scoped-alias metadata in a form the LLVM inside clspv rejects; the metadata is only an optimisation hint
  sed -i -E 's/, !(alias\.scope|noalias) ![0-9]+//g' "$OUT/$c.ll"
  "$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 "$OUT/$c.ll" -o "$OUT/$c.spv" 2>&1 \
    | grep -v "^warning: \(overriding the module target\|Linking two modules\)" | grep -v "^$" | head -"${LINES_MAX:-20}"
  [ -s "$OUT/$c.spv" ] && spirv-val --target-env vulkan1.3 "$OUT/$c.spv" 2>&1 | head -5 && echo "spirv-val rc=$? size=$(stat -c %s "$OUT/$c.spv")"
done
