# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

| set | entries | IR | SPIR-V | valid | refused | runs | matches reference |
|---|---|---|---|---|---|---|---|
| uzu | 121 | 106 (87.6%) | 106 (87.6%) | 106 (87.6%) | 0 | not run here | 0 |
| tf | 488 | 447 (91.6%) | 425 (87.1%) | 412 (84.4%) | 0 | not run here | 0 |
| all | 609 | 553 (90.8%) | 531 (87.2%) | 518 (85.1%) | 0 | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d) | 43 | 44 |
| 2 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 17 | 17 |
| 3 | other | 12 | 16 |
| 4 | spirv-val: other | 11 | 11 |
| 5 | clspv: other | 4 | 4 |
| 6 | clspv does not finish (timeout, crash without a message, SimplifyPointerBitcast loop) | 3 | 3 |
| 7 | vector construction and conversion (mk1, C-style vector casts, scalar initializers) | 1 | 1 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `overload:mma_16x32` | 38 | core/affine_mm.metal `tf_affine_mm` | error: no matching function for call to 'frag_get' |
| 2 | `overload:frag_get_in` | 35 | core/affine_mm.metal `tf_affine_mm` | error: no matching function for call to 'frag_get' |
| 3 | `overload:frag_get` | 34 | core/affine_mm.metal `tf_affine_row_sums` | error: no matching function for call to 'frag_get' |
| 4 | `clspv:Invalid bitcast` | 15 | decode/fn_lane.metal `fz_lane` | Invalid bitcast |
| 5 | `spirv-val:error: line N: [VUID-StandaloneSpirv-None-N] Having N components for T` | 11 | flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d-lanes.metal `custom_kernel_lane_qmm_bytes_grouped` | error: line 205: [VUID-StandaloneSpirv-None-12295] Having 8 components for TypeVector requ |
| 6 | `overload:store_paired_vectors` | 9 | attention/attention_gemm.metal `AttentionGemm_v1` | error: no matching member function for call to 'row_reduce' |
| 7 | `overload:load_paired_vectors` | 9 | attention/attention_gemm.metal `AttentionGemm_v1` | error: no matching member function for call to 'row_reduce' |
| 8 | `overload:row_reduce` | 4 | attention/attention_gemm.metal `AttentionGemm_v0` | error: no matching member function for call to 'row_reduce' |
| 9 | `clspv:ptr addrspace(N)` | 4 | kimi/mla.metal `k3_mla_cache` | ptr addrspace(1) |
| 10 | `other: from vector 'X' (vector of N 'X' values) to vector 'X' (aka 'X') of diffe` | 3 | activation_transform/activation_transform.metal `ActivationTransform_v1` | error:  from vector 'float4' (vector of 4 'float' values) to vector 'vec<__bf16, 4>' (aka  |
| 11 | `clspv:MNV: SimplifyPointerBitcast does not converge; changing sub-passes: N N` | 3 | ops/qmv.metal `tf_gather_qmv_b4_g64` | M2V: SimplifyPointerBitcast does not converge; changing sub-passes: 5 7 |
| 12 | `spirv-val:error: line N: Expected input to be a pointer or int or float vector o` | 2 | glm/router.metal `custom_kernel_tf_glm5_fused_router_t` | error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast |
| 13 | `other: from vector 'X' (aka 'X') to vector 'X' (aka 'X') of different size` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 14 | `other:excess elements in scalar initializer` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 15 | `overload:select` | 1 | matmul/gemm/gemm.metal `Gemm_v1` | error: no matching function for call to 'load_paired_vectors' |
| 16 | `other:static_cast from 'X' (aka 'X') to 'X' (vector of N 'X' values) is not allo` | 1 | matmul/gemv/gemv.metal `Gemv_v0` | error: static_cast from 'I4' (aka 'typename vec_sel<__bf16, 4>::type') to 'float4' (vector |
| 17 | `other:incompatible operand types ('X' and 'X')` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 18 | `other:kernel functions cannot be used in a template declaration, instantiation o` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 19 | `overload:Mark` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 20 | `overload:__private Mark` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 21 | `other:redefinition of 'X'` | 1 | ops/nax_gemm.metal `tf_nax_gemm_nn_bf16` | error: redefinition of 'frag_home' |
| 22 | `overload:frag_put_in` | 1 | ops/nax_gemm.metal `tf_nax_gemm_nn_bf16` | error: redefinition of 'frag_home' |
