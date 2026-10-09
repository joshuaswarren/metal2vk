// Case 3 (simdgroup_matrix): kernel/matmul/common/simdgroup_multiply_accumulate.h, unmodified (SimdgroupMMA load,
// mma, store with the Apple per-lane fragment layout). uzu's full gemm.metal needs build-generated headers and
// the MXU path, so this driver is a minimal 8x8-tile GEMM built from the same header: C[M,N] = A[M,K] * B[K,N],
// one subgroup per 8x8 output tile.
#include "metal_stdlib"
#include "matmul/common/simdgroup_multiply_accumulate.h"

using MMA = uzu::matmul::SimdgroupMMA<float, 8, 8>;

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void tile_matmul_f32(
    __global const float* A, __global const float* B, __global float* C, __constant uint* dims) {
  const uint M = dims[0], N = dims[1], K = dims[2];
  (void)M;
  const uint lane = get_sub_group_local_id();
  const short2 rc = MMA::get_lane_coordinates(lane);
  const uint tile_x = get_group_id(0), tile_y = get_group_id(1);
  MMA::ThreadDataType acc;
  acc[0] = 0.0f;
  acc[1] = 0.0f;
  for (uint k = 0; k < K; k += 8) {
    MMA::ThreadDataType a, b;
    MMA::load(a, A + (tile_y * 8 + rc.y) * K + k + rc.x, K, 1);
    MMA::load(b, B + (k + rc.y) * N + tile_x * 8 + rc.x, N, 1);
    MMA::mma(acc, a, b, acc);
  }
  MMA::store(acc, C + (tile_y * 8 + rc.y) * N + tile_x * 8 + rc.x, N, 1);
}
