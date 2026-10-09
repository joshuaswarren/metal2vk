// Case 2 (reduction): kernel/softmax/softmax.metal, compiled unmodified (simd_max, simd_sum, threadgroup arrays,
// threadgroup_barrier). Grid: GROUPS(outer_dim) x GROUPS(batch_dim), THREADS(256).
#include "metal_stdlib"
#include "softmax/softmax.metal"

#define SOFTMAX_ENTRY(NAME, T)                                                                                 \
  __kernel __attribute__((reqd_work_group_size(256, 1, 1))) void NAME(                                         \
      __global T* values, __global const T* sinks, __constant uint* meta, uint has_sinks) {                    \
    __local float shared_max[METAL_SIMD_SIZE];                                                                 \
    __local float shared_norm[METAL_SIMD_SIZE];                                                                \
    Softmax<T>(values, sinks, meta[0], meta[1], meta[2], shared_max, shared_norm, has_sinks != 0,              \
               (uint)get_group_id(0), (uint)get_group_id(1), (uint)get_local_id(0));                           \
  }

SOFTMAX_ENTRY(softmax_f32, float)
SOFTMAX_ENTRY(softmax_f16, half)
