// The same operation written as OpenCL C++ with no metal2vk shim: direct ::rsqrt(float) on a runtime load.
// Lowered with the same clang, opt and clspv commands as tools/m2v-compile (see lower.sh).
__kernel void rsqrt_direct(__global const uint* in, __global uint* out) {
  size_t i = get_global_id(0);
  out[i] = as_uint(rsqrt(as_float(in[i])));
}
