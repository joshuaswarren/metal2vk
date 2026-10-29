#!/bin/bash
# Build the pipeline-variant cache fixture for tests/pipeline_variants.rs:
# assemble each library, validate it for Vulkan 1.3, reflect it and pack it
# with the repo's own tools. Index 7 is FACTOR (SpecId 7). Index 9 is the
# entry the specialization packer needs for threadgroup argument 1 (SpecId 9,
# LENGTH_BYTES). Both kernels stay unverified: the test checks them against
# the CPU reference on first use.
# Usage: build.sh OUTDIR   (needs spirv-as, spirv-val, spirv-dis, python3)
set -euo pipefail
OUT=${1:?outdir}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
mkdir -p "$OUT"
for lib in variant_a variant_b; do
  spirv-as --target-env vulkan1.2 "$HERE/$lib.spvasm" -o "$OUT/$lib.spv"
  spirv-val --target-env vulkan1.3 "$OUT/$lib.spv"
  python3 "$REPO/m2v-reflect.py" "$OUT/$lib.spv" > "$OUT/$lib.json"
  python3 "$REPO/host/tools/make-m2vlib.py" "$OUT/$lib.spv" "$OUT/$lib.json" "$OUT/$lib.m2vlib" \
    --constants "$HERE/constants.json"
done
