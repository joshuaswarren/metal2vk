// Case 5 (register-tiled simdgroup_matrix GEMM): the same uzu SimdgroupMMA header as tilematmul.cl, unmodified, but one subgroup owns an
// MR x NR block of 8x8 output tiles. Each k step loads MR left and NR right fragments and issues MR*NR multiply-accumulates, so the
// operand traffic per flop drops by a factor of 2*MR*NR/(MR+NR) against the one-tile kernel. C[M,N] = A[M,K] * B[K,N], f32, row major.
#include "metal_stdlib"
#include "matmul/common/simdgroup_multiply_accumulate.h"

using MMA = uzu::matmul::SimdgroupMMA<float, 8, 8>;

#define RT_ENTRY(NAME, MR, NR)                                                                                        \
  __kernel __attribute__((reqd_work_group_size(32, 1, 1))) void NAME(                                                \
      __global const float* A, __global const float* B, __global float* C, __constant uint* dims) {                  \
    const uint N = dims[1], K = dims[2];                                                                             \
    const uint lane = get_sub_group_local_id();                                                                      \
    const short2 rc = MMA::get_lane_coordinates(lane);                                                               \
    const uint row0 = get_group_id(1) * (8 * MR), col0 = get_group_id(0) * (8 * NR);                                 \
    MMA::ThreadDataType acc[MR][NR];                                                                                 \
    _Pragma("unroll") for (int i = 0; i < MR; i++) {                                                                 \
      _Pragma("unroll") for (int j = 0; j < NR; j++) {                                                               \
        acc[i][j][0] = 0.0f;                                                                                         \
        acc[i][j][1] = 0.0f;                                                                                         \
      }                                                                                                              \
    }                                                                                                                \
    for (uint k = 0; k < K; k += 8) {                                                                                \
      MMA::ThreadDataType a[MR], b[NR];                                                                              \
      _Pragma("unroll") for (int i = 0; i < MR; i++) MMA::load(a[i], A + (row0 + i * 8 + rc.y) * K + k + rc.x, K, 1);   \
      _Pragma("unroll") for (int j = 0; j < NR; j++) MMA::load(b[j], B + (k + rc.y) * N + col0 + j * 8 + rc.x, N, 1);   \
      _Pragma("unroll") for (int i = 0; i < MR; i++) {                                                               \
        _Pragma("unroll") for (int j = 0; j < NR; j++) MMA::mma(acc[i][j], a[i], b[j], acc[i][j]);                   \
      }                                                                                                              \
    }                                                                                                                \
    _Pragma("unroll") for (int i = 0; i < MR; i++) {                                                                 \
      _Pragma("unroll") for (int j = 0; j < NR; j++)                                                                 \
        MMA::store(acc[i][j], C + (row0 + i * 8 + rc.y) * N + col0 + j * 8 + rc.x, N, 1);                            \
    }                                                                                                                \
  }

RT_ENTRY(tile_matmul_rt2x2_f32, 2, 2)
RT_ENTRY(tile_matmul_rt4x2_f32, 4, 2)
RT_ENTRY(tile_matmul_rt4x4_f32, 4, 4)
