// The math the precise::rsqrt change must not move: unqualified rsqrt (scalar and float4, the only vector rsqrt the
// shim has), fast::rsqrt, and fma (scalar and float4), whose lowering is clspv's FMA selection. lower.sh compiles this
// file against the base shim and the current one; precise_rsqrt_scoped_math_unchanged compares modules and bits.
// Operands are runtime loads of the frozen corpus, so nothing is constant-folded.
#include <metal_stdlib>
using namespace metal;

[[kernel]] void rsqrt_scope(device const uint* in [[buffer(0)]], device uint* unq [[buffer(1)]], device uint* unq4 [[buffer(2)]],
                            device uint* fst [[buffer(3)]], device uint* fm [[buffer(4)]], device uint* fm4 [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
  const uint j = i & ~3u;
  const float x = as_type<float>(in[i]);
  const float4 v = float4(as_type<float>(in[j]), as_type<float>(in[j + 1]), as_type<float>(in[j + 2]), as_type<float>(in[j + 3]));
  const float4 w = float4(as_type<float>(in[j ^ 4u]), as_type<float>(in[(j ^ 4u) + 1]), as_type<float>(in[(j ^ 4u) + 2]),
                          as_type<float>(in[(j ^ 4u) + 3]));
  unq[i] = as_type<uint>(rsqrt(x));
  unq4[i] = as_type<uint>(rsqrt(v)[i & 3u]);
  fst[i] = as_type<uint>(fast::rsqrt(x));
  fm[i] = as_type<uint>(fma(x, as_type<float>(in[i ^ 1u]), as_type<float>(in[i ^ 2u])));
  fm4[i] = as_type<uint>(fma(v, w, v.wzyx)[i & 3u]);
}
