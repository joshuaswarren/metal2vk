#include <metal_stdlib>
using namespace metal;
kernel void pointer_probe(device const half *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  device const half *p[4];
  float4 sums[4];
  #pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) { p[j] = src + (tid + uint(j)) * count; sums[j] = float4(0.0f); }
  for (uint k = 0; k < count; k += 8) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
      sums[j] += float4(*reinterpret_cast<device const half4 *>(p[j] + k));
      sums[j] += float4(*reinterpret_cast<device const half4 *>(p[j] + k + 4));
    }
  }
  float result = 0;
  #pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) result += sums[j].x;
  dst[tid] = result;
}
