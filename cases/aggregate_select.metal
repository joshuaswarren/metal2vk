#include <metal_stdlib>
using namespace metal;

// Selecting between two private copies of a struct that came from a dynamic threadgroup-array index. Each entry reads
// states g and g+1 from `groups` and writes the one with the larger `best` (select_*), or state g (select_control).
//   select_aggregate: `a.best > b.best ? a : b`, both operands private structs. The copy becomes byte loads; without
//                     the src == src_gep guard in clspv-patches/apply-implicit-casts-lossless-merge.py, clspv's
//                     SimplifyPointerBitcastPass counted the element offset twice for k % 4 != 0 and read bytes
//                     (g << 5) | k instead of (g << 4) | k: a valid module with the wrong state for g > 0.
//   select_scalar:    the same choice one field at a time. Correct.
//   select_control:   no select, plain struct copy. Correct, so the copy alone is not the fault.
// Synthetic input only. out = [best, index, second, sum]; m2v-test.py aggregate_select supplies states and oracle.
struct Conf { float best; uint index; float second; float sum; };

kernel void select_control(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                          constant uint& g [[buffer(2)]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup Conf groups[8];
  if (lane < 8u) groups[lane] = Conf{in[lane * 4u], uint(in[lane * 4u + 1u]), in[lane * 4u + 2u], in[lane * 4u + 3u]};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    Conf a = groups[g];
    Conf b = groups[g + 1u];
    Conf r = a;
    out[0] = r.best; out[1] = float(r.index); out[2] = r.second; out[3] = r.sum;
  }
}

kernel void select_aggregate(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                            constant uint& g [[buffer(2)]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup Conf groups[8];
  if (lane < 8u) groups[lane] = Conf{in[lane * 4u], uint(in[lane * 4u + 1u]), in[lane * 4u + 2u], in[lane * 4u + 3u]};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    Conf a = groups[g];
    Conf b = groups[g + 1u];
    Conf r = a.best > b.best ? a : b;
    out[0] = r.best; out[1] = float(r.index); out[2] = r.second; out[3] = r.sum;
  }
}

kernel void select_scalar(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint& g [[buffer(2)]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup Conf groups[8];
  if (lane < 8u) groups[lane] = Conf{in[lane * 4u], uint(in[lane * 4u + 1u]), in[lane * 4u + 2u], in[lane * 4u + 3u]};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    Conf a = groups[g];
    Conf b = groups[g + 1u];
    const bool w = a.best > b.best;
    Conf r = Conf{w ? a.best : b.best, w ? a.index : b.index, w ? a.second : b.second, w ? a.sum : b.sum};
    out[0] = r.best; out[1] = float(r.index); out[2] = r.second; out[3] = r.sum;
  }
}
