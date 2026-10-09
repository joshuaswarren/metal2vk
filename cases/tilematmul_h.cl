// Case 6 (half inputs, f32 accumulate, register-tiled): same structure as tilematmul_rt.cl but A and B are half and the
// multiply-accumulate calls the m2v_mma_f16 builtin directly (struct arrays of simdgroup_matrix trip the clspv pointer passes) and lowers to the f16/f16/f32 8x8x8 cooperative matrix (needs -DM2V_F16_COOPMAT and the f16 clspv patch).
// C[M,N] = A[M,K] * B[K,N], A and B half, C f32, row major.
#include "metal_stdlib"

#define H_ENTRY(NAME, MR, NR)                                                                                         \
  __kernel __attribute__((reqd_work_group_size(32, 1, 1))) void NAME(                                                \
      __global const half* A, __global const half* B, __global float* C, __constant uint* dims) {                    \
    const uint N = dims[1], K = dims[2];                                                                             \
    const uint lane = get_sub_group_local_id();                                                                      \
    const int r = metal::m2v_row(lane), c = metal::m2v_col(lane);                                                    \
    const uint row0 = get_group_id(1) * (8 * MR), col0 = get_group_id(0) * (8 * NR);                                 \
    float2 acc[MR][NR];                                                                \
    _Pragma("unroll") for (int i = 0; i < MR; i++) {                                                                 \
      _Pragma("unroll") for (int j = 0; j < NR; j++) { acc[i][j] = float2(0.0f, 0.0f); }                          \
    }                                                                                                                \
    for (uint k = 0; k < K; k += 8) {                                                                                \
      half2 a[MR], b[NR];                                                              \
      _Pragma("unroll") for (int i = 0; i < MR; i++) {                                                               \
        const __global half* p = A + (row0 + i * 8 + r) * K + k + c;                                                 \
        a[i] = half2(p[0], p[1]);                                                                                 \
      }                                                                                                              \
      _Pragma("unroll") for (int j = 0; j < NR; j++) {                                                               \
        const __global half* p = B + (k + r) * N + col0 + j * 8 + c;                                                 \
        b[j] = half2(p[0], p[1]);                                                                                 \
      }                                                                                                              \
      _Pragma("unroll") for (int i = 0; i < MR; i++) {                                                               \
        _Pragma("unroll") for (int j = 0; j < NR; j++) acc[i][j] = m2v_mma_f16(a[i], b[j], acc[i][j]); \
      }                                                                                                              \
    }                                                                                                                \
    _Pragma("unroll") for (int i = 0; i < MR; i++) {                                                                 \
      _Pragma("unroll") for (int j = 0; j < NR; j++) {                                                               \
        __global float* p = C + (row0 + i * 8 + r) * N + col0 + j * 8 + c;                                           \
        p[0] = acc[i][j].x;                                                                                      \
        p[1] = acc[i][j].y;                                                                                      \
      }                                                                                                              \
    }                                                                                                                \
  }

H_ENTRY(tile_matmul_h2x2, 2, 2)
H_ENTRY(tile_matmul_h4x4, 4, 4)
