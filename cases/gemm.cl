// Case 4 (GEMM): kernel/matmul/gemm/gemm.metal compiled unmodified. SimdgroupMmaCore path (USE_MXU = false), float
// operands, full-precision B (transposed, the nn.Linear weight layout), tiling Tile64x64x32_Simdgroups2x2
// (128 threads: 32 lanes x 2 x 2 simdgroups). The entry below is what uzu's DSL annotations describe
// (GROUPS(group_count_x/y/z), THREADS(32, simdgroups per column, simdgroups per row), threadgroup arrays sized by
// GEMM_TGA/TGB_ELEMENTS); a generator would emit it per variant.
#include "metal_stdlib"
#include "matmul/gemm/gemm.metal"

#define GEMM_ENTRY(NAME, AT_, BT_, DT_, TILING_, TB_, TPC, TPR)                                                       \
  __kernel __attribute__((reqd_work_group_size(32, TPC, TPR))) void NAME(                                          \
      __global const AT_* a, __global const BT_* b, __global DT_* d, __global const BT_* scales,                    \
      __global const BT_* biases, __global const uint8_t* zero_points, __global const BT_* output_bias,            \
      __global const int32_t* rht_factors, __global const int8_t* a_int8, __global const float* a_scales,          \
      __global const int32_t* a_group_sums, __constant uzu::matmul::GemmParams* params,                            \
      __constant uint* counts, uint transform_bits, uint alignment_bits) {                                          \
    using AT = AT_;                                                                                                \
    using BT = BT_;                                                                                                \
    constexpr GemmTiling GEMM_TILING = TILING_;                                                                    \
    constexpr bool USE_MXU = false;                                                                                \
    constexpr uint GROUP_SIZE = 0;                                                                                 \
    constexpr GemmAPrologueKind A_PROLOGUE = GemmAPrologueKind::FullPrecision;                                     \
    constexpr GemmBPrologueKind B_PROLOGUE = GemmBPrologueKind::FullPrecision;                                     \
    (void)GEMM_TILING; (void)USE_MXU; (void)GROUP_SIZE; (void)A_PROLOGUE; (void)B_PROLOGUE;                       \
    __local AT a_shared[GEMM_TGA_ELEMENTS];                                                                        \
    __local BT b_shared[GEMM_TGB_ELEMENTS];                                                                        \
    GemmDTransform tr; tr.raw_value = transform_bits; GemmAlignment al; al.raw_value = alignment_bits;          \
    ThreadContext tc;                                                                                              \
    const uint3 lid = uint3(get_local_id(0), get_local_id(1), get_local_id(2));                                    \
    tc.simd_lane_id = get_sub_group_local_id();                                                                    \
    tc.simdgroup_index = lid.y + TPC * lid.z;                                                                      \
    tc.simdgroup_size = 32;                                                                                        \
    tc.simdgroups_per_threadgroup = TPC * TPR;                                                                     \
    tc.threadgroup_position = uint3(get_group_id(0), get_group_id(1), get_group_id(2));                            \
    tc.threadgroup_size = uint3(32, TPC, TPR);                                                                     \
    tc.threadgroup_count = uint3(get_num_groups(0), get_num_groups(1), get_num_groups(2));                         \
    tc.grid_size = tc.threadgroup_count * tc.threadgroup_size;                                                     \
    Gemm<AT_, BT_, DT_, TILING_, TB_, false, GemmBPrologueKind::FullPrecision, 0, 0,                               \
         GemmAPrologueKind::FullPrecision, 0>(                                                                     \
        a, b, d, scales, biases, zero_points, output_bias, rht_factors, a_int8, a_scales, a_group_sums, params,    \
        counts[0], counts[1], counts[2], tr, al, false,                                                            \
        a_shared, b_shared, tc.threadgroup_position.x,                       \
        tc.threadgroup_position.y, tc.threadgroup_position.z, lid.x, lid.y, lid.z, tc);                            \
  }

#if defined(M2V_GEMM_SMALL)
GEMM_ENTRY(gemm_f32_t8x32x32, float, float, float, GemmTiling::Tile8x32x32_Simdgroups1x1, true, 1, 1)
#elif defined(M2V_GEMM_MID)
GEMM_ENTRY(gemm_f32_t32x32x32, float, float, float, GemmTiling::Tile32x32x32_Simdgroups2x2, true, 2, 2)
#else
GEMM_ENTRY(gemm_f32_t64x64x32, float, float, float, GemmTiling::Tile64x64x32_Simdgroups2x2, true, 2, 2)
#endif
