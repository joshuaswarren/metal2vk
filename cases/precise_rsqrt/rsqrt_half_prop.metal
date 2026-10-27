// Half-rounding propagation of the FP32 precise::rsqrt result, MLX rms_norm style: n = half(x * inv), y = w * n.
// Inputs are public: s comes from the frozen corpus, x and w are generated in m2v-test.py so that x * inv sits on the
// other side of a half rounding boundary from x * nextup(inv). Lane i packs x (low 16 bits) and w (high 16 bits).
// The output packs n (low) and y (high).
#include <metal_stdlib>
using namespace metal;

[[kernel]] void rsqrt_half_prop(device const uint* s [[buffer(0)]], device const uint* xw [[buffer(1)]],
                                device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  const float inv = precise::rsqrt(as_type<float>(s[i]));
  const uint p = xw[i];
  const half x = as_type<half>(ushort(p & 0xFFFFu));
  const half w = as_type<half>(ushort(p >> 16));
  const half n = half(float(x) * inv);
  const half y = w * n;
  out[i] = uint(as_type<ushort>(n)) | (uint(as_type<ushort>(y)) << 16);
}
