# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

| set | entries | IR | SPIR-V | valid | runs | matches reference |
|---|---|---|---|---|---|---|
| uzu | 121 | 102 (84.3%) | 102 (84.3%) | 102 (84.3%) | not run here | 0 |
| tf | 488 | 447 (91.6%) | 403 (82.6%) | 339 (69.5%) | not run here | 0 |
| all | 609 | 549 (90.1%) | 505 (82.9%) | 441 (72.4%) | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 46 | 46 |
| 2 | Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d) | 43 | 44 |
| 3 | spirv-val: other | 29 | 29 |
| 4 | clspv: other | 20 | 20 |
| 5 | other | 16 | 20 |
| 6 | clspv does not finish (timeout, crash without a message, SimplifyPointerBitcast loop) | 13 | 13 |
| 7 | vector construction and conversion (mk1, C-style vector casts, scalar initializers) | 1 | 1 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `overload:mma_16x32` | 38 | core/affine_mm.metal `tf_affine_mm` | error: no matching function for call to 'frag_get' |
| 2 | `overload:frag_get_in` | 35 | core/affine_mm.metal `tf_affine_mm` | error: no matching function for call to 'frag_get' |
| 3 | `overload:frag_get` | 34 | core/affine_mm.metal `tf_affine_row_sums` | error: no matching function for call to 'frag_get' |
| 4 | `spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_uchar]'X'N[%N]'X'N[%_ptr_` | 32 | core/row_projection.metal `tf_row_projection_q2_f32` | error: line 908: OpPhi's result type <id> '214[%_ptr_StorageBuffer_uchar]' does not match  |
| 5 | `clspv:ptr addrspace(N)` | 20 | kimi/experts.metal `k3_xp_up_r1` | ptr addrspace(1) |
| 6 | `spirv-val:error: line N: Header block 'X' is contained in the loop construct hea` | 18 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: line 1780: Header block '1428[%1428]' is contained in the loop construct headed by  |
| 7 | `clspv:Invalid bitcast` | 11 | decode/fn_lane.metal `fz_lane` | Invalid bitcast |
| 8 | `spirv-val:error: line N: [VUID-StandaloneSpirv-None-N] Having N components for T` | 11 | flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d-lanes.metal `custom_kernel_lane_qmm_bytes_grouped` | error: line 205: [VUID-StandaloneSpirv-None-12295] Having 8 components for TypeVector requ |
| 9 | `clspv:timeout` | 10 | kimi/dense_mma.metal `k3_slice_mma` | timeout |
| 10 | `overload:load_paired_vectors` | 9 | attention/attention_gemm.metal `AttentionGemm_v1` | error: no matching member function for call to 'row_reduce' |
| 11 | `overload:store_paired_vectors` | 9 | attention/attention_gemm.metal `AttentionGemm_v1` | error: no matching member function for call to 'row_reduce' |
| 12 | `overload:store_hadamard_vector` | 4 | activation_transform/activation_transform.metal `ActivationTransform_v0` | error: no matching function for call to 'store_hadamard_vector' |
| 13 | `overload:row_reduce` | 4 | attention/attention_gemm.metal `AttentionGemm_v0` | error: no matching member function for call to 'row_reduce' |
| 14 | `other: from vector 'X' (vector of N 'X' values) to vector 'X' (aka 'X') of diffe` | 3 | activation_transform/activation_transform.metal `ActivationTransform_v1` | error:  from vector 'float4' (vector of 4 'float' values) to vector 'vec<__bf16, 4>' (aka  |
| 15 | `clspv:MNV: SimplifyPointerBitcast does not converge; changing sub-passes: N N` | 3 | ops/qmv.metal `tf_gather_qmv_b4_g64` | M2V: SimplifyPointerBitcast does not converge; changing sub-passes: 5 7 |
| 16 | `other:incompatible operand types ('X' (aka 'X') and 'X')` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 17 | `other:variable in constant address space must be initialized` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 18 | `overload:__private Logit` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 19 | `overload:const __constant Logit` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 20 | `spirv-val:error: line N: Expected input to be a pointer or int or float vector o` | 2 | glm/router.metal `custom_kernel_tf_glm5_fused_router_t` | error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast |
| 21 | `other: from vector 'X' (aka 'X') to vector 'X' (aka 'X') of different size` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 22 | `other:excess elements in scalar initializer` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 23 | `overload:select` | 1 | matmul/gemm/gemm.metal `Gemm_v1` | error: no matching function for call to 'load_paired_vectors' |
| 24 | `other:static_cast from 'X' (aka 'X') to 'X' (vector of N 'X' values) is not allo` | 1 | matmul/gemv/gemv.metal `Gemv_v0` | error: static_cast from 'I4' (aka 'typename vec_sel<__bf16, 4>::type') to 'float4' (vector |
| 25 | `spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_ushort]'X'N[%N]'X'N[%_ptr` | 1 | flashnext/qa_ple_lookup_c9861329fad34fa4.metal `custom_kernel_qa_ple_lookup_c9861329` | error: line 316: OpPhi's result type <id> '55[%_ptr_StorageBuffer_ushort]' does not match  |
| 26 | `overload:Mark` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 27 | `overload:__private Mark` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 28 | `other:incompatible operand types ('X' and 'X')` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 29 | `other:kernel functions cannot be used in a template declaration, instantiation o` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 30 | `other:redefinition of 'X'` | 1 | ops/nax_gemm.metal `tf_nax_gemm_nn_bf16` | error: redefinition of 'frag_home' |
| 31 | `overload:frag_put_in` | 1 | ops/nax_gemm.metal `tf_nax_gemm_nn_bf16` | error: redefinition of 'frag_home' |
