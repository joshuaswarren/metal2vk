#include <metal_stdlib>
using namespace metal;

// packed_uint3 is three tightly packed words: size 12, alignment 4. A native OpenCL uint3 is 16 and 16.
// source = [sentinel, a0, a1, a2, b0, b1, b2, sentinel] is read as an array of packed_uint3 starting at byte 4, so the
// array stride decides which words thread 1 reads: stride 12 gives b0 b1 b2, a 16-byte uint3 alias gives b1 b2 sentinel.
// out = [a0, a1, a2, b0, b1, b2, sizeof, alignof]. Constant component indices on purpose: this case tests the type's
// layout, not index lowering. Covers storage and indexing only, not packed-vector arithmetic.
// tools/tests/run.py compiles and validates this source. A hardware runner/oracle is not included.
static_assert(sizeof(packed_uint3) == 12 && alignof(packed_uint3) == 4, "packed_uint3 layout");
static_assert(__is_trivially_copyable(packed_uint3), "packed_uint3 trivial copy");
static_assert(__is_standard_layout(packed_uint3), "packed_uint3 standard layout");

kernel void packed_uint3_stride12(device const uint* source [[buffer(0)]],
                                  device uint* out [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]]) {
  if (tid < 2u) {
    const packed_uint3 words = reinterpret_cast<device const packed_uint3*>(source + 1u)[tid];
    out[3u * tid] = words[0];
    out[3u * tid + 1u] = words[1];
    out[3u * tid + 2u] = words[2];
  }
  if (tid == 0u) {
    out[6] = sizeof(packed_uint3);
    out[7] = alignof(packed_uint3);
  }
}
