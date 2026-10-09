# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

| set | entries | IR | SPIR-V | valid | runs | matches reference |
|---|---|---|---|---|---|---|
| uzu | 121 | 102 (84.3%) | 102 (84.3%) | 102 (84.3%) | not run here | 0 |
| tf | 488 | 447 (91.6%) | 415 (85.0%) | 384 (78.7%) | not run here | 0 |
| all | 609 | 549 (90.1%) | 517 (84.9%) | 486 (79.8%) | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d) | 43 | 44 |
| 2 | spirv-val: other | 29 | 29 |
| 3 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 17 | 17 |
| 4 | other | 16 | 20 |
| 5 | clspv does not finish (timeout, crash without a message, SimplifyPointerBitcast loop) | 13 | 13 |
| 6 | clspv: other | 4 | 4 |
| 7 | vector construction and conversion (mk1, C-style vector casts, scalar initializers) | 1 | 1 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `overload:mma_16x32` | 38 | core/affine_mm.metal `tf_affine_mm` | error: no matching function for call to 'frag_get' |
| 2 | `overload:frag_get_in` | 35 | core/affine_mm.metal `tf_affine_mm` | error: no matching function for call to 'frag_get' |
| 3 | `overload:frag_get` | 34 | core/affine_mm.metal `tf_affine_row_sums` | error: no matching function for call to 'frag_get' |
| 4 | `spirv-val:error: line N: Header block 'X' is contained in the loop construct hea` | 18 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: line 1780: Header block '1428[%1428]' is contained in the loop construct headed by  |
| 5 | `clspv:Invalid bitcast` | 15 | decode/fn_lane.metal `fz_lane` | Invalid bitcast |
| 6 | `spirv-val:error: line N: [VUID-StandaloneSpirv-None-N] Having N components for T` | 11 | flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d-lanes.metal `custom_kernel_lane_qmm_bytes_grouped` | error: line 205: [VUID-StandaloneSpirv-None-12295] Having 8 components for TypeVector requ |
| 7 | `clspv:timeout` | 10 | kimi/dense_mma.metal `k3_slice_mma` | timeout |
| 8 | `overload:store_paired_vectors` | 9 | attention/attention_gemm.metal `AttentionGemm_v1` | error: no matching member function for call to 'row_reduce' |
| 9 | `overload:load_paired_vectors` | 9 | attention/attention_gemm.metal `AttentionGemm_v1` | error: no matching member function for call to 'row_reduce' |
| 10 | `overload:store_hadamard_vector` | 4 | activation_transform/activation_transform.metal `ActivationTransform_v0` | error: no matching function for call to 'store_hadamard_vector' |
| 11 | `overload:row_reduce` | 4 | attention/attention_gemm.metal `AttentionGemm_v0` | error: no matching member function for call to 'row_reduce' |
| 12 | `clspv:ptr addrspace(N)` | 4 | kimi/mla.metal `k3_mla_cache` | ptr addrspace(1) |
| 13 | `other: from vector 'X' (vector of N 'X' values) to vector 'X' (aka 'X') of diffe` | 3 | activation_transform/activation_transform.metal `ActivationTransform_v1` | error:  from vector 'float4' (vector of 4 'float' values) to vector 'vec<__bf16, 4>' (aka  |
| 14 | `clspv:MNV: SimplifyPointerBitcast does not converge; changing sub-passes: N N` | 3 | ops/qmv.metal `tf_gather_qmv_b4_g64` | M2V: SimplifyPointerBitcast does not converge; changing sub-passes: 5 7 |
| 15 | `overload:const __constant Logit` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 16 | `other:incompatible operand types ('X' (aka 'X') and 'X')` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 17 | `other:variable in constant address space must be initialized` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 18 | `overload:__private Logit` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 19 | `spirv-val:error: line N: Expected input to be a pointer or int or float vector o` | 2 | glm/router.metal `custom_kernel_tf_glm5_fused_router_t` | error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast |
| 20 | `other: from vector 'X' (aka 'X') to vector 'X' (aka 'X') of different size` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 21 | `other:excess elements in scalar initializer` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 22 | `overload:select` | 1 | matmul/gemm/gemm.metal `Gemm_v1` | error: no matching function for call to 'load_paired_vectors' |
| 23 | `other:static_cast from 'X' (aka 'X') to 'X' (vector of N 'X' values) is not allo` | 1 | matmul/gemv/gemv.metal `Gemv_v0` | error: static_cast from 'I4' (aka 'typename vec_sel<__bf16, 4>::type') to 'float4' (vector |
| 24 | `overload:__private Mark` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 25 | `other:incompatible operand types ('X' and 'X')` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 26 | `other:kernel functions cannot be used in a template declaration, instantiation o` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 27 | `overload:Mark` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 28 | `other:redefinition of 'X'` | 1 | ops/nax_gemm.metal `tf_nax_gemm_nn_bf16` | error: redefinition of 'frag_home' |
| 29 | `overload:frag_put_in` | 1 | ops/nax_gemm.metal `tf_nax_gemm_nn_bf16` | error: redefinition of 'frag_home' |
