// Case 1 (elementwise): crates/uzu-engine/.../kernel/activation/activation.metal, compiled unmodified.
// The entry below is what uzu's DSL annotations (AXIS(n, 256), SPECIALIZE, constant refs) describe; a generator
// would emit it. Constant refs become __constant pointers; in_place is a plain uint argument here.
#include "metal_stdlib"
#include "activation/activation.metal"

// Tuning knobs (compile-time, driven from compile.sh via M2V_DEFS; e.g. -DM2V_ACT_EPT=2):
//   M2V_ACT_WG   workgroup size         (default 256, uzu's AXIS(n, 256))
//   M2V_ACT_EPT  elements per thread    (default 1, uzu's one element per invocation)
// The defaults reproduce the unmodified uzu mapping exactly.
#ifndef M2V_ACT_WG
#define M2V_ACT_WG 256
#endif
#ifndef M2V_ACT_EPT
#define M2V_ACT_EPT 1
#endif

#define ACT_ENTRY(NAME, T)                                                                                    \
  __kernel __attribute__((reqd_work_group_size(M2V_ACT_WG, 1, 1))) void NAME(__global const T* input, __global T* output, __constant uint* n, __constant uint* act, \
                     uint in_place) {                                                                         \
    for (uint e = 0; e < (uint)M2V_ACT_EPT; ++e) {                                                            \
      uint tid = get_global_id(0) * (uint)M2V_ACT_EPT + e;                                                    \
      if (tid < n[0])                                                                                         \
        Activation<T>(input, output, n[0], *(const __constant ActivationType*)act, in_place != 0, tid);       \
    }                                                                                                         \
  }

ACT_ENTRY(activation_f32, float)
ACT_ENTRY(activation_f16, half)
