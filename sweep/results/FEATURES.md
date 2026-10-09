# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

Before (the shim and flags of main when the sweep started):

| set | entries | IR | SPIR-V | valid | runs | matches reference |
|---|---|---|---|---|---|---|
| uzu | 121 | 91 (75.2%) | 84 (69.4%) | 84 (69.4%) | not run here | 0 |
| tf | 488 | 335 (68.6%) | 280 (57.4%) | 241 (49.4%) | not run here | 0 |
| all | 609 | 426 (70.0%) | 364 (59.8%) | 325 (53.4%) | not run here | 0 |

After:

| set | entries | IR | SPIR-V | valid | runs | matches reference |
|---|---|---|---|---|---|---|
| uzu | 121 | 102 (84.3%) | 102 (84.3%) | 102 (84.3%) | not run here | 0 |
| tf | 488 | 372 (76.2%) | 336 (68.9%) | 299 (61.3%) | not run here | 0 |
| all | 609 | 474 (77.8%) | 438 (71.9%) | 401 (65.8%) | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d) | 124 | 125 |
| 2 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 39 | 39 |
| 3 | clspv: other | 20 | 20 |
| 4 | clspv does not finish (timeout, crash without a message, SimplifyPointerBitcast loop) | 12 | 12 |
| 5 | other | 10 | 48 |
| 6 | spirv-val: other | 2 | 2 |
| 7 | vector construction and conversion (mk1, C-style vector casts, scalar initializers) | 1 | 1 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `overload:__private tensor<__global bfloat, dextents<int32_t, 2>, tensor_inline>` | 70 | decode/fn_lane.metal `fz_lane` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 2 | `overload:__private dextents<int32_t, 2>` | 70 | decode/fn_lane.metal `fz_lane` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 3 | `member:metal::tensor<__global __bf16, metal::dextents<int, 2>>::slice` | 69 | decode/fn_lane.metal `fz_lane` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 4 | `other:expected 'X' for function-style cast or type construction` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: no member named 'remove_addrspace_t' in namespace 'metal' |
| 5 | `other:expected 'X' at end of declaration` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: no member named 'remove_addrspace_t' in namespace 'metal' |
| 6 | `other:definition or redeclaration of 'X' not allowed inside a function` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: no member named 'remove_addrspace_t' in namespace 'metal' |
| 7 | `other:declaration of variable 'X' with deduced type 'X' requires an initializer` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: no member named 'remove_addrspace_t' in namespace 'metal' |
| 8 | `member:metal::remove_addrspace_t` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: no member named 'remove_addrspace_t' in namespace 'metal' |
| 9 | `overload:mma_16x32` | 38 | core/affine_mm.metal `tf_affine_mm` | error: no member named 'remove_addrspace_t' in namespace 'metal' |
| 10 | `other:expected unqualified-id` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 11 | `ident:execution_simdgroups` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 12 | `other:a type specifier is required for all declarations` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 13 | `other:type-id cannot have a name` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 14 | `ident:b` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 15 | `overload:frag_get_in` | 35 | core/affine_mm.metal `tf_affine_mm` | error: no member named 'remove_addrspace_t' in namespace 'metal' |
| 16 | `overload:frag_get` | 34 | core/affine_mm.metal `tf_affine_row_sums` | error: no member named 'remove_addrspace_t' in namespace 'metal' |
| 17 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_destination` | 34 | decode/fn_lane.metal `fz_lane` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 18 | `spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_uchar]'X'N[%N]'X'N[%_ptr_` | 32 | core/row_projection.metal `tf_row_projection_q2_f32` | error: line 908: OpPhi's result type <id> '214[%_ptr_StorageBuffer_uchar]' does not match  |
| 19 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_left_input_` | 25 | attention/attention_gemm.metal `AttentionGemm_v1` | error: no matching member function for call to 'row_reduce' |
| 20 | `clspv:ptr addrspace(N)` | 20 | kimi/experts.metal `k3_xp_up_r1` | ptr addrspace(1) |
| 21 | `overload:__private tensor<__local uint8_t, dextents<int32_t, 2>, tensor_inline>` | 17 | decode/fn_lane.metal `fz_lane` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 22 | `overload:__private tensor<__local half, dextents<int32_t, 2>, tensor_inline>` | 17 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 23 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_right_input` | 9 | attention/attention_gemm.metal `AttentionGemm_v1` | error: no matching member function for call to 'row_reduce' |
| 24 | `clspv:timeout after N s (clspv does not finish)` | 9 | kimi/dense_mma.metal `k3_slice_mma` | timeout |
| 25 | `overload:store_hadamard_vector` | 4 | activation_transform/activation_transform.metal `ActivationTransform_v0` | error: no matching function for call to 'store_hadamard_vector' |
| 26 | `overload:row_reduce` | 4 | attention/attention_gemm.metal `AttentionGemm_v0` | error: no matching member function for call to 'row_reduce' |
| 27 | `other: from vector 'X' (vector of N 'X' values) to vector 'X' (aka 'X') of diffe` | 3 | activation_transform/activation_transform.metal `ActivationTransform_v1` | error:  from vector 'float4' (vector of 4 'float' values) to vector 'vec<__bf16, 4>' (aka  |
| 28 | `clspv:MNV: SimplifyPointerBitcast does not converge; changing sub-passes: N N` | 3 | ops/qmv.metal `tf_gather_qmv_b4_g64` | M2V: SimplifyPointerBitcast does not converge; changing sub-passes: 5 7 |
| 29 | `other:incompatible operand types ('X' (aka 'X') and 'X')` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 30 | `other:variable in constant address space must be initialized` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 31 | `overload:__private Logit` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 32 | `overload:const __constant Logit` | 2 | sampling/unified_sampling.metal `UnifiedSampling_v0` | error: variable in constant address space must be initialized |
| 33 | `spirv-val:error: line N: Expected input to be a pointer or int or float vector o` | 2 | glm/router.metal `custom_kernel_tf_glm5_fused_router_t` | error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast |
| 34 | `clspv:Invalid bitcast` | 2 | kimi/kda.metal `k3_kda` | Invalid bitcast |
| 35 | `spirv-val:error: line N: Header block 'X' is contained in the loop construct hea` | 2 | ops/gemv.metal `tf_gemv_wide_bf16_v4_kl32` | error: line 455: Header block '359[%359]' is contained in the loop construct headed by '21 |
| 36 | `clspv:error: undefined reference to 'X'` | 2 | prefill/qmm6_nax_b.metal `tf_mm_bf16_f32_t_nax` | error: undefined reference to '_Z0Pfp9mma_16x32ILb0ELb1EEEvRU3AS4Dv8_fS3_RU3AS4KDv8_DF16bS |
| 37 | `other:excess elements in scalar initializer` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 38 | `other: from vector 'X' (aka 'X') to vector 'X' (aka 'X') of different size` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 39 | `overload:select` | 1 | matmul/gemm/gemm.metal `Gemm_v1` | error: no member named 'get_left_input_cooperative_tensor' in 'mpp::tensor_ops::matmul2d<0 |
| 40 | `other:static_cast from 'X' (aka 'X') to 'X' (vector of N 'X' values) is not allo` | 1 | matmul/gemv/gemv.metal `Gemv_v0` | error: static_cast from 'I4' (aka 'typename vec_sel<__bf16, 4>::type') to 'float4' (vector |
