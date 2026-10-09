// Case 1 (elementwise): crates/uzu-engine/.../kernel/activation/activation.metal, compiled unmodified.
// The entry below is what uzu's DSL annotations (AXIS(n, 256), SPECIALIZE, constant refs) describe; a generator
// would emit it. Constant refs become __constant pointers; in_place is a plain uint argument here.
#include "metal_stdlib"
#include "activation/activation.metal"

#define ACT_ENTRY(NAME, T)                                                                                    \
  __kernel __attribute__((reqd_work_group_size(256, 1, 1))) void NAME(__global const T* input, __global T* output, __constant uint* n, __constant uint* act, \
                     uint in_place) {                                                                         \
    uint tid = get_global_id(0);                                                                              \
    if (tid >= n[0]) return;                                                                                  \
    Activation<T>(input, output, n[0], *(const __constant ActivationType*)act, in_place != 0, tid);           \
  }

ACT_ENTRY(activation_f32, float)
ACT_ENTRY(activation_f16, half)
