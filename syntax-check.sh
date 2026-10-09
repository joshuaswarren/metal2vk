#!/bin/bash
# Front-end check only (no clspv): clang parses the uzu kernel sources through the metal_stdlib shim as C++ for OpenCL.
# Usage: syntax-check.sh [case ...]   env: UZU_ROOT, LINES_MAX
set -u
H=$(cd "$(dirname "$0")" && pwd)
UZU_ROOT=${UZU_ROOT:?set UZU_ROOT to a uzu checkout}
K=$UZU_ROOT/crates/uzu-engine/src/backends/metal/kernel
CASES=${*:-activation softmax tilematmul gemm}
for c in $CASES; do
  echo "== $c"
  clang --target=spir64 -x cl -cl-std=clc++2021 -Xclang -finclude-default-header -cl-ext=-__opencl_c_generic_address_space -fsyntax-only -w \
    -I"$H/include" -I"$K" -I"$K/generated" "$H/cases/$c.cl" 2>&1 | grep -E "error|note: in instantiation" | head -"${LINES_MAX:-25}"
done
