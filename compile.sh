#!/bin/bash
# metal2vk: Metal kernel (through the metal_stdlib shim, C++ for OpenCL) -> LLVM IR (clang) -> Vulkan SPIR-V (clspv).
# Usage: compile.sh OUTDIR [case ...]
# env: UZU_ROOT (uzu checkout), CLSPV (patched clspv binary), CLANG, OPT (LLVM opt, same version as clang),
#      M2V_DEFS (extra clang flags, e.g. -DM2V_NO_COOPMAT), M2V_OPT (replaces the route's clang optimisation flags, used by bench/sweep.py),
#      M2V_PIPELINE  typed (default) | o2   which front-end route to take, see below.
# clang is the front end because it accepts the generic-address-space-free C++ for OpenCL dialect that makes Metal's `thread`
# qualifier work (see include/metal_stdlib); clspv then lowers the IR, including the m2v_mma_f32 cooperative matrix builtin.
#
# Front-end routes. clang's own optimiser (InstCombine) rewrites typed `getelementptr float` into byte-offset `getelementptr i8` forms.
# clspv's pointer passes then lose the element type: vector fragment loads become 8 byte loads per lane (the 8x8 tile matmul runs 4x
# slower) and uzu's Gemm ends in an OpPhi of a float pointer and a [4 x i8] pointer that fails validation. The default route therefore
# keeps the IR typed: clang -O0 without optnone, drop the noinline clang puts on every -O0 function, inline and promote aggregates with
# LLVM's own inliner and SROA (no InstCombine), and let clspv run its optimiser. M2V_PIPELINE=o2 is the previous route (clang -O2 with a
# huge inline threshold and the vectorisers off), kept for comparison.
set -u
OUT=${1:?outdir}
shift
H=$(cd "$(dirname "$0")" && pwd)
UZU_ROOT=${UZU_ROOT:?set UZU_ROOT to a uzu checkout}
CLSPV=${CLSPV:-$HOME/scratch/metal2vk/clspv/build/bin/clspv}
CLANG=${CLANG:-clang}
OPT=${OPT:-opt}
PIPE=${M2V_PIPELINE:-typed}
K=$UZU_ROOT/crates/uzu-engine/src/backends/metal/kernel
CASES=${*:-activation softmax tilematmul tilematmul_rt gemm}
mkdir -p "$OUT"
if [ "$PIPE" = typed ]; then
  FE_OPT="-O0 -Xclang -disable-O0-optnone"
else
  FE_OPT="-O2 -fno-slp-vectorize -fno-vectorize -mllvm -inline-threshold=100000"
fi
FE_OPT=${M2V_OPT:-$FE_OPT}
for c in $CASES; do
  echo "== $c ($PIPE)"
  # shellcheck disable=SC2086
  "$CLANG" --target=spir -x cl -cl-std=clc++2021 -Xclang -finclude-default-header \
    -cl-ext=-__opencl_c_generic_address_space $FE_OPT -fno-strict-return -cl-kernel-arg-info -w ${M2V_DEFS:-} \
    -I"$H/include" -I"$K" -I"$K/generated" -S -emit-llvm "$H/cases/$c.cl" -o "$OUT/$c.ll" 2>&1 | head -"${LINES_MAX:-20}"
  [ -s "$OUT/$c.ll" ] || continue
  if [ "$PIPE" = typed ]; then
    sed -i -E 's/\bnoinline\b//g' "$OUT/$c.ll"
    "$OPT" -passes='cgscc(inline),function(sroa,early-cse,simplifycfg)' -inline-threshold=100000 -S "$OUT/$c.ll" -o "$OUT/$c.opt.ll" 2>&1 \
      | grep -v "failed to create target machine" | head -5
    [ -s "$OUT/$c.opt.ll" ] && mv "$OUT/$c.opt.ll" "$OUT/$c.ll"
  fi
  # scoped-alias metadata (and the noalias.scope.decl calls that carry its domains) is in a form the LLVM inside clspv rejects; it is
  # only an optimisation hint
  sed -i -E 's/, !(alias\.scope|noalias) ![0-9]+//g; /llvm\.experimental\.noalias\.scope\.decl/d' "$OUT/$c.ll"
  "$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 "$OUT/$c.ll" -o "$OUT/$c.spv" 2>&1 \
    | grep -v "^warning: \(overriding the module target\|Linking two modules\)" | grep -v "^$" | head -"${LINES_MAX:-20}"
  [ -s "$OUT/$c.spv" ] && spirv-val --target-env vulkan1.3 "$OUT/$c.spv" 2>&1 | head -5 && echo "spirv-val rc=$? size=$(stat -c %s "$OUT/$c.spv")"
done
