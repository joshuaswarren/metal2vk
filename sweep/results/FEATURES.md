# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

Before (the shim and flags of main when the sweep started):

| set | entries | IR | SPIR-V | valid | runs | matches reference |
|---|---|---|---|---|---|---|
| uzu | 121 | 70 (57.9%) | 70 (57.9%) | 70 (57.9%) | not run here | 0 |
| tf | 488 | 239 (49.0%) | 193 (39.5%) | 155 (31.8%) | not run here | 0 |
| all | 609 | 309 (50.7%) | 263 (43.2%) | 225 (36.9%) | not run here | 0 |

After:

| set | entries | IR | SPIR-V | valid | runs | matches reference |
|---|---|---|---|---|---|---|
| uzu | 121 | 87 (71.9%) | 83 (68.6%) | 83 (68.6%) | not run here | 0 |
| tf | 488 | 335 (68.6%) | 303 (62.1%) | 263 (53.9%) | not run here | 0 |
| all | 609 | 422 (69.3%) | 386 (63.4%) | 346 (56.8%) | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d) | 138 | 139 |
| 2 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 66 | 66 |
| 3 | bfloat vs float in one expression (`c ? bfloat : float` is ambiguous: bfloat converts both ways) | 27 | 28 |
| 4 | other | 15 | 63 |
| 5 | clspv: other | 10 | 10 |
| 6 | host-injected symbols (a kernel needs a -D, a typedef or a helper the host provides) | 6 | 6 |
| 7 | vector construction and conversion (mk1, C-style vector casts, scalar initializers) | 1 | 1 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `overload:__private dextents<int32_t, 2>` | 70 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 2 | `overload:__private tensor<__global bfloat, dextents<int32_t, 2>, tensor_inline>` | 70 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 3 | `member:metal::tensor<__global metal::bfloat, metal::dextents<int, 2>>::slice` | 69 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 4 | `ident:mpp` | 63 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 5 | `other:expected 'X' for function-style cast or type construction` | 46 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 6 | `ident:execution_simdgroup` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 7 | `member:metal::remove_addrspace_t` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 8 | `other:declaration of variable 'X' with deduced type 'X' requires an initializer` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 9 | `other:expected 'X' at end of declaration` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 10 | `other:definition or redeclaration of 'X' not allowed inside a function` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 11 | `ident:matmul2d_descriptor` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 12 | `ident:b` | 40 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 13 | `overload:mma_16x32` | 38 | core/affine_mm.metal `tf_affine_mm` | error: use of undeclared identifier 'mpp' |
| 14 | `overload:frag_get_in` | 36 | core/affine_mm.metal `tf_affine_mm` | error: use of undeclared identifier 'mpp' |
| 15 | `ident:execution_simdgroups` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 16 | `other:type-id cannot have a name` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 17 | `other:expected unqualified-id` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 18 | `other:a type specifier is required for all declarations` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 19 | `overload:frag_get` | 34 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 20 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_destination` | 34 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 21 | `spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_uchar]'X'N[%N]'X'N[%_ptr_` | 32 | core/row_projection.metal `tf_row_projection_q2_f32` | error: line 1514: OpPhi's result type <id> '116[%_ptr_StorageBuffer_uchar]' does not match |
| 22 | `other:conditional expression is ambiguous; 'X' can be converted to 'X' and vice ` | 28 | softmax/softmax.metal `Softmax_v1` | error: conditional expression is ambiguous; 'const __global metal::bfloat' can be converte |
| 23 | `clspv:Invalid bitcast` | 26 | gdn/tree_verify/state_advance.metal `StateAdvance_v1` | Invalid bitcast |
| 24 | `type:MatmulMode` | 19 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 25 | `member:metal::execution_simdgroup` | 19 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 26 | `ident:MatmulMode` | 19 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 27 | `ident:matmul_op` | 19 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 28 | `overload:__private tensor<__local uint8_t, dextents<int32_t, 2>, tensor_inline>` | 17 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 29 | `overload:__private tensor<__local half, dextents<int32_t, 2>, tensor_inline>` | 17 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 30 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_left_input_` | 16 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 31 | `other:expected external declaration` | 13 | prefill/attention_nax.metal `custom_kernel_tf_attention_nax_bf16_` | error: use of undeclared identifier 'mpp' |
| 32 | `clspv:MNV: SimplifyPointerBitcast does not converge; changing sub-passes: N N` | 8 | nemotron_round.metal `tf_round_attsel` | M2V: SimplifyPointerBitcast does not converge; changing sub-passes: 3 7 |
| 33 | `ident:simdgroup_load` | 7 | kimi/dense_mma.metal `k3_slice_mma` | error: use of undeclared identifier 'simdgroup_load' |
| 34 | `ident:simdgroup_store` | 7 | kimi/synth.metal `k3_mma_peak` | error: use of undeclared identifier 'simdgroup_store' |
| 35 | `overload:store_hadamard_vector` | 4 | activation_transform/activation_transform.metal `ActivationTransform_v0` | error: no matching function for call to 'store_hadamard_vector' |
| 36 | `overload:row_reduce` | 4 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 37 | `other:clang frontend command failed with exit code N (use -v to see invocation)` | 4 | gdn/chunked/causal_inv.metal `DeltaNetChunkedCausalInv_v0` | clang-19: error: clang frontend command failed with exit code 139 (use -v to see invocatio |
| 38 | `ident:sq_acc` | 4 | glm_kda_prompt.metal `glm_kda_pre` | error: use of undeclared identifier 'quad_dot' |
| 39 | `ident:mlx_sigmoid_precise` | 4 | glm_kda_prompt.metal `glm_kda_pre` | error: use of undeclared identifier 'quad_dot' |
| 40 | `other:expected 'X' after expression` | 4 | prefill/qmm6_nax_b.metal `tf_mm_bf16_f32_t_nax` | error: no template named 'frag' |
