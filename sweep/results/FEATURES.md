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
| uzu | 121 | 92 (76.0%) | 90 (74.4%) | 90 (74.4%) | not run here | 0 |
| tf | 488 | 361 (74.0%) | 329 (67.4%) | 292 (59.8%) | not run here | 0 |
| all | 609 | 453 (74.4%) | 419 (68.8%) | 382 (62.7%) | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d) | 135 | 136 |
| 2 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 37 | 37 |
| 3 | clspv: other | 20 | 20 |
| 4 | other | 14 | 73 |
| 5 | clspv does not finish (timeout, crash without a message, SimplifyPointerBitcast loop) | 12 | 12 |
| 6 | host-injected symbols (a kernel needs a -D, a typedef or a helper the host provides) | 6 | 6 |
| 7 | spirv-val: other | 2 | 2 |
| 8 | vector construction and conversion (mk1, C-style vector casts, scalar initializers) | 1 | 1 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `overload:__private tensor<__global bfloat, dextents<int32_t, 2>, tensor_inline>` | 70 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 2 | `overload:__private dextents<int32_t, 2>` | 70 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 3 | `member:metal::tensor<__global __bf16, metal::dextents<int, 2>>::slice` | 69 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 4 | `ident:mpp` | 60 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 5 | `other:expected 'X' for function-style cast or type construction` | 46 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 6 | `member:metal::remove_addrspace_t` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 7 | `other:definition or redeclaration of 'X' not allowed inside a function` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 8 | `ident:execution_simdgroup` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 9 | `other:declaration of variable 'X' with deduced type 'X' requires an initializer` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 10 | `other:expected 'X' at end of declaration` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 11 | `ident:matmul2d_descriptor` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 12 | `ident:b` | 40 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 13 | `overload:mma_16x32` | 38 | core/affine_mm.metal `tf_affine_mm` | error: use of undeclared identifier 'mpp' |
| 14 | `overload:frag_get_in` | 36 | core/affine_mm.metal `tf_affine_mm` | error: use of undeclared identifier 'mpp' |
| 15 | `ident:execution_simdgroups` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 16 | `other:a type specifier is required for all declarations` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 17 | `other:type-id cannot have a name` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 18 | `other:expected unqualified-id` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgr |
| 19 | `overload:frag_get` | 34 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 20 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_destination` | 34 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 21 | `spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_uchar]'X'N[%N]'X'N[%_ptr_` | 32 | core/row_projection.metal `tf_row_projection_q2_f32` | error: line 908: OpPhi's result type <id> '214[%_ptr_StorageBuffer_uchar]' does not match  |
| 22 | `clspv:ptr addrspace(N)` | 20 | kimi/experts.metal `k3_xp_up_r1` | ptr addrspace(1) |
| 23 | `overload:__private tensor<__local uint8_t, dextents<int32_t, 2>, tensor_inline>` | 17 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 24 | `overload:__private tensor<__local half, dextents<int32_t, 2>, tensor_inline>` | 17 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 25 | `other:'X' keyword not permitted here` | 16 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 26 | `type:MatmulMode` | 16 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 27 | `ident:MatmulMode` | 16 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 28 | `ident:matmul_op` | 16 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 29 | `member:metal::execution_simdgroup` | 16 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 30 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_left_input_` | 16 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 31 | `other:expected external declaration` | 13 | prefill/attention_nax.metal `custom_kernel_tf_attention_nax_bf16_` | error: use of undeclared identifier 'mpp' |
| 32 | `clspv:timeout after N s (clspv does not finish)` | 9 | attention/attention_two_pass.metal `AttentionTwoPass2_v0` | timeout |
| 33 | `ident:simdgroup_load` | 7 | kimi/dense_mma.metal `k3_slice_mma` | error: use of undeclared identifier 'simdgroup_load' |
| 34 | `ident:simdgroup_store` | 7 | kimi/synth.metal `k3_mma_peak` | error: use of undeclared identifier 'simdgroup_store' |
| 35 | `overload:store_hadamard_vector` | 4 | activation_transform/activation_transform.metal `ActivationTransform_v0` | error: no matching function for call to 'store_hadamard_vector' |
| 36 | `ident:sq_acc` | 4 | glm_kda_prompt.metal `glm_kda_pre` | error: use of undeclared identifier 'quad_dot' |
| 37 | `ident:mlx_sigmoid_precise` | 4 | glm_kda_prompt.metal `glm_kda_pre` | error: use of undeclared identifier 'quad_dot' |
| 38 | `other:expected 'X' after expression` | 4 | prefill/qmm6_nax_b.metal `tf_mm_bf16_f32_t_nax` | error: no template named 'frag' |
| 39 | `template:frag` | 4 | prefill/qmm6_nax_b.metal `tf_mm_bf16_f32_t_nax` | error: no template named 'frag' |
| 40 | `ident:frag` | 4 | prefill/qmm6_nax_b.metal `tf_mm_bf16_f32_t_nax` | error: no template named 'frag' |
