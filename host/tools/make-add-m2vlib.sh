#!/bin/bash
# Compile host/kernels/add.cl through the repo chain (compile.sh, minus the MSL shim,
# which plain OpenCL C smoke kernels do not need) and pack it into add.m2vlib.
# Usage: make-add-m2vlib.sh OUTDIR   (CLSPV env overrides the binary path)
set -euo pipefail
OUT=${1:?outdir}
H=$(cd "$(dirname "$0")/.." && pwd)   # host/
REPO=$(cd "$H/.." && pwd)
CLSPV=${CLSPV:?set CLSPV to the patched clspv binary}
mkdir -p "$OUT"
clang --target=spir -x cl -cl-std=clc++2021 -Xclang -finclude-default-header \
  -cl-ext=-__opencl_c_generic_address_space -O2 -fno-strict-return -cl-kernel-arg-info -w \
  -S -emit-llvm "$H/kernels/add.cl" -o "$OUT/add.ll"
sed -i -E 's/, !(alias\.scope|noalias) ![0-9]+//g' "$OUT/add.ll"
"$CLSPV" -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 \
  "$OUT/add.ll" -o "$OUT/add.spv"
spirv-val --target-env vulkan1.3 "$OUT/add.spv"
echo "spirv-val ok: $(stat -c %s "$OUT/add.spv") bytes"
python3 "$REPO/m2v-reflect.py" "$OUT/add.spv" > "$OUT/add.json"
# Gate: verified only when the G13C receipt proves add_kernel_end_to_end
# (see add-gate-args.sh); otherwise the kernel stays unverified.
GATE_ARGS=$("$H/tools/add-gate-args.sh" "$OUT")
# shellcheck disable=SC2086
python3 "$H/tools/make-m2vlib.py" "$OUT/add.spv" "$OUT/add.json" "$OUT/add.m2vlib" $GATE_ARGS
