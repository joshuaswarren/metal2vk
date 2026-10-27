#!/bin/bash
# Lower the three precise-rsqrt reductions with the pinned compiler and keep every intermediate.
#   a_shim    rsqrt_precise.metal  -> tools/m2v-compile (shim, clang, opt, clspv)
#   b_direct  rsqrt_direct.cl      -> the same clang, opt and clspv commands, no shim
#   c_glsl    rsqrt_inversesqrt.spvasm -> spirv-as, no compiler at all
#   e_scope, f_half, base/  Phase 1 modules, see the block below. PR 1682 is not built.
# usage: lower.sh OUTDIR   env: B (pinned clspv build dir), SPIRV_AS (assembler), CACHE (m2v-compile cache dir), BASE (base shim commit)
set -euo pipefail
OUT=${1:?outdir}
H=$(cd "$(dirname "$0")/../.." && pwd)
B=${B:?path to the patched clspv build directory}
CLANG=$B/third_party/llvm/bin/clang-24 OPT=$B/third_party/llvm/bin/opt CLSPV=$B/bin/clspv
SPIRV_VAL=$B/third_party/SPIRV-Tools/tools/spirv-val SPIRV_DIS=$B/third_party/SPIRV-Tools/tools/spirv-dis
SPIRV_AS=${SPIRV_AS:?path to spirv-as}
CACHE=${CACHE:-$OUT/cache}
export CLANG OPT CLSPV SPIRV_VAL PATH=$(dirname "$SPIRV_DIS"):$PATH
S=$H/cases/precise_rsqrt
mkdir -p "$OUT/a_shim" "$OUT/b_direct" "$OUT/c_glsl"

# a: the translated source (.cl) and final LLVM IR (.ll) stay in the m2v-compile cache entry; copy them out
key=$(python3 "$H/tools/m2v-compile" --msl "$S/rsqrt_precise.metal" --out "$OUT/a_shim" --cache "$CACHE" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["key"])')
for sub in src ll; do cp "$CACHE"/keys/${key}*/$sub/*.* "$OUT/a_shim/rsqrt_precise.$sub.${sub/src/cl}"; done
mv "$OUT/a_shim/rsqrt_precise.ll.ll" "$OUT/a_shim/rsqrt_precise.ll"; mv "$OUT/a_shim/rsqrt_precise.src.cl" "$OUT/a_shim/rsqrt_precise.translated.cl"

# b: m2v_sweep.process_once's front end and opt step, then its clspv command
D=$OUT/b_direct
"$CLANG" --target=spir -x cl -cl-std=clc++2021 -Xclang -finclude-default-header -cl-ext=-__opencl_c_generic_address_space \
  -O0 -Xclang -disable-O0-optnone -fno-strict-return -cl-kernel-arg-info -w -S -emit-llvm "$S/rsqrt_direct.cl" -o "$D/rsqrt_direct.clang.ll"
perl -pe 's/\bnoinline\b//g' "$D/rsqrt_direct.clang.ll" > "$D/rsqrt_direct.nn.ll"
"$OPT" -passes='cgscc(inline),function(sroa,early-cse,simplifycfg)' -inline-threshold=100000 -S "$D/rsqrt_direct.nn.ll" -o "$D/rsqrt_direct.ll" 2> "$D/opt.log"
"$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 --long-vector "$D/rsqrt_direct.ll" -o "$D/rsqrt_direct.spv" > "$D/clspv.log" 2>&1
rm "$D/rsqrt_direct.nn.ll"

# c: assemble
"$SPIRV_AS" --target-env vulkan1.2 "$S/rsqrt_inversesqrt.spvasm" -o "$OUT/c_glsl/rsqrt_inversesqrt.spv"

# Phase 1 (docs/precise-rsqrt.md): e_scope and f_half probes. PR 1682 is not built here; its failure receipt is phase1/lowering/d_pr1682. Each shim-built module is also built
# under base/ against the unmodified shim at BASE (default the reviewed base), and as <name>.nativefma.spv with clspv's
# --use-native-builtins=fma on the same IR: the native-FMA selection, compiled directly here, not a new m2v-compile option.
BASE=${BASE:-97517a57527a2d6157adbd4756e10b2e3aa2ad95}
mkdir -p "$OUT/base/include"
git -C "$H" archive "$BASE" include | tar -x -C "$OUT/base"
shim() { # SRC STEM OUTDIR [include]: m2v-compile, keep the IR, add the native-FMA module
  local d=$3/$2 k
  mkdir -p "$d"
  k=$(M2V_INCLUDE=${4:-$H/include} python3 "$H/tools/m2v-compile" --msl "$S/$1" --out "$d" --cache "$CACHE" --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["key"])')
  cp "$CACHE"/keys/${k}*/ll/*.ll "$d/${1%.metal}.ll"
  "$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 --long-vector --use-native-builtins=fma \
    "$d/${1%.metal}.ll" -o "$d/${1%.metal}.nativefma.spv" 2> "$d/${1%.metal}.nativefma.clspv.log"
}
shim rsqrt_precise.metal a_shim "$OUT"  # rebuilds a_shim with the native-FMA module; the default module is unchanged
shim rsqrt_scope.metal e_scope "$OUT"
shim rsqrt_half_prop.metal f_half "$OUT"
for p in "rsqrt_precise.metal a_shim" "rsqrt_scope.metal e_scope" "rsqrt_half_prop.metal f_half"; do
  shim ${p% *} ${p#* } "$OUT/base" "$OUT/base/include"
done
rm -rf "$OUT/base/include"

for spv in "$OUT"/b_direct/rsqrt_direct.spv "$OUT"/c_glsl/rsqrt_inversesqrt.spv \
           "$OUT"/{,base/}{a_shim,e_scope,f_half}/*.spv; do
  [ -f "$spv" ] || continue
  "$SPIRV_VAL" --target-env vulkan1.3 "$spv"
  "$SPIRV_DIS" "$spv" -o "${spv%.spv}.dis"
  python3 "$H/m2v-reflect.py" "$spv" > "${spv%.spv}.json"
done
