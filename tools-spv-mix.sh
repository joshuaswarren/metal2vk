#!/bin/bash
# Usage: tools-spv-mix.sh MODULE.spv   -> counts of the op kinds that decide GEMM speed
dis=$(spirv-dis --no-header "$1")
c() { printf '%s\n' "$dis" | grep -c -E "$1"; }
echo "module $1 size $(stat -c %s "$1") bytes, $(printf '%s\n' "$dis" | wc -l) disassembly lines"
echo "OpCooperativeMatrixMulAddKHR : $(c 'OpCooperativeMatrixMulAddKHR')"
echo "OpControlBarrier             : $(c 'OpControlBarrier')"
echo "OpLoad  (all)                : $(c ' = OpLoad ')"
echo "OpLoad  float scalar         : $(c ' = OpLoad %float ')"
echo "OpLoad  v4float / v2float    : $(c ' = OpLoad %v4float ') / $(c ' = OpLoad %v2float ')"
echo "OpLoad  uchar/uint byte path : $(c ' = OpLoad %uchar ') / $(c ' = OpLoad %uint ')"
echo "OpStore (all)                : $(c '^ *OpStore ')"
echo "OpAccessChain                : $(c ' = OpAccessChain ')"
echo "OpGroupNonUniform*           : $(c 'OpGroupNonUniform')"
echo "OpPhi                        : $(c ' = OpPhi ')"
echo "Workgroup variables          : $(c ' = OpVariable .*Workgroup')"
echo "Function variables           : $(c ' = OpVariable .*Function')"
echo "OpFMul/OpFAdd/OpExtInst Fma  : $(c ' = OpFMul ') / $(c ' = OpFAdd ') / $(c 'OpExtInst %float %[0-9a-zA-Z_]+ Fma')"
