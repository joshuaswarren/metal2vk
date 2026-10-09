// Case 3 (simdgroup_matrix): kernel/matmul/common/simdgroup_multiply_accumulate.h, unmodified (SimdgroupMMA load,
// mma, store with the Apple per-lane fragment layout). uzu's full gemm.metal needs build-generated headers and
// the MXU path, so this driver is a minimal 8x8-tile GEMM built from the same header: C[M,N] = A[M,K] * B[K,N],
// one subgroup per 8x8 output tile.
#include "metal_stdlib"
#include "matmul/common/simdgroup_multiply_accumulate.h"

// Tuning knob (compile-time, driven from compile.sh via M2V_DEFS; e.g. -DM2V_TM_CHAINS=2):
//   M2V_TM_CHAINS independent k-step accumulator chains (default 1 = the kernel as uzu
//   writes it; > 1 interleaves loads and mul-adds to hide mma latency).
// The default reproduces the unmodified loop exactly (the tail loop is empty for it).
#ifndef M2V_TM_CHAINS
#define M2V_TM_CHAINS 1
#endif

using MMA = uzu::matmul::SimdgroupMMA<float, 8, 8>;

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void tile_matmul_f32(
    __global const float* A, __global const float* B, __global float* C, __constant uint* dims) {
  const uint M = dims[0], N = dims[1], K = dims[2];
  (void)M;
  const uint lane = get_sub_group_local_id();
  const short2 rc = MMA::get_lane_coordinates(lane);
  const uint tile_x = get_group_id(0), tile_y = get_group_id(1);
  MMA::ThreadDataType acc[M2V_TM_CHAINS];
  for (uint c = 0; c < M2V_TM_CHAINS; ++c) {
    acc[c][0] = 0.0f;
    acc[c][1] = 0.0f;
  }
  for (uint k = 0; k + 8 * (M2V_TM_CHAINS - 1) < K; k += 8 * M2V_TM_CHAINS) {
    for (uint c = 0; c < M2V_TM_CHAINS; ++c) {
      const uint kk = k + 8 * c;
      MMA::ThreadDataType a, b;
      MMA::load(a, A + (tile_y * 8 + rc.y) * K + kk + rc.x, K, 1);
      MMA::load(b, B + (kk + rc.y) * N + tile_x * 8 + rc.x, N, 1);
      MMA::mma(acc[c], a, b, acc[c]);
    }
  }
  // tail when K is not a multiple of 8 * M2V_TM_CHAINS: finish on chain 0
  for (uint k = K - (K % (8 * M2V_TM_CHAINS)); k < K; k += 8) {
    MMA::ThreadDataType a, b;
    MMA::load(a, A + (tile_y * 8 + rc.y) * K + k + rc.x, K, 1);
    MMA::load(b, B + (k + rc.y) * N + tile_x * 8 + rc.x, N, 1);
    MMA::mma(acc[0], a, b, acc[0]);
  }
  MMA::ThreadDataType total = acc[0];
  for (uint c = 1; c < M2V_TM_CHAINS; ++c) {
    total[0] += acc[c][0];
    total[1] += acc[c][1];
  }
  MMA::store(total, C + (tile_y * 8 + rc.y) * N + tile_x * 8 + rc.x, N, 1);
}
