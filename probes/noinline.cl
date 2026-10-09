// Does clspv keep a noinline function with private-pointer arguments as an OpFunction + OpFunctionCall?
__attribute__((noinline)) void m2v_mma_f32(__private float* d, __private const float* a, __private const float* b,
                                           __private const float* c) {
  d[0] = a[0] * b[0] + c[0];
  d[1] = a[1] * b[1] + c[1];
}

__kernel void probe(__global float* o, __global const float* in) {
  float a[2], b[2], c[2], d[2];
  const uint i = get_global_id(0);
  a[0] = in[i]; a[1] = in[i + 1]; b[0] = in[i + 2]; b[1] = in[i + 3]; c[0] = in[i + 4]; c[1] = in[i + 5];
  m2v_mma_f32(d, a, b, c);
  o[i] = d[0];
  o[i + 1] = d[1];
}
