# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

Before (the shim and flags of main when the sweep started):

| set | entries | IR | SPIR-V | valid | runs | matches reference |
|---|---|---|---|---|---|---|
| uzu | 121 | 70 (57.9%) | 70 (57.9%) | 70 (57.9%) | not run here | 0 |
| tf | 488 | 239 (49.0%) | 199 (40.8%) | 162 (33.2%) | not run here | 0 |
| all | 609 | 309 (50.7%) | 269 (44.2%) | 232 (38.1%) | not run here | 0 |

After:

| set | entries | IR | SPIR-V | valid | runs | matches reference |
|---|---|---|---|---|---|---|
| uzu | 121 | 87 (71.9%) | 75 (62.0%) | 75 (62.0%) | not run here | 0 |
| tf | 488 | 335 (68.6%) | 280 (57.4%) | 241 (49.4%) | not run here | 0 |
| all | 609 | 422 (69.3%) | 355 (58.3%) | 316 (51.9%) | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d) | 138 | 139 |
| 2 | clspv does not finish (timeout, crash without a message, SimplifyPointerBitcast loop) | 49 | 49 |
| 3 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 41 | 41 |
| 4 | bfloat vs float in one expression (`c ? bfloat : float` is ambiguous: bfloat converts both ways) | 27 | 28 |
| 5 | clspv: other | 16 | 16 |
| 6 | other | 15 | 63 |
| 7 | host-injected symbols (a kernel needs a -D, a typedef or a helper the host provides) | 6 | 6 |
| 8 | vector construction and conversion (mk1, C-style vector casts, scalar initializers) | 1 | 1 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `overload:__private tensor<__global bfloat, dextents<int32_t, 2>, tensor_inline>` | 70 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 2 | `overload:__private dextents<int32_t, 2>` | 70 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 3 | `member:metal::tensor<__global metal::bfloat, metal::dextents<int, 2>>::slice` | 69 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 4 | `ident:mpp` | 63 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 5 | `other:expected 'X' for function-style cast or type construction` | 46 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 6 | `ident:execution_simdgroup` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 7 | `ident:matmul2d_descriptor` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 8 | `other:definition or redeclaration of 'X' not allowed inside a function` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 9 | `member:metal::remove_addrspace_t` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 10 | `other:expected 'X' at end of declaration` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 11 | `other:declaration of variable 'X' with deduced type 'X' requires an initializer` | 44 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 12 | `ident:b` | 40 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 13 | `overload:mma_16x32` | 38 | core/affine_mm.metal `tf_affine_mm` | error: use of undeclared identifier 'mpp' |
| 14 | `clspv:timeout after N s (clspv does not finish)` | 37 | radix_top_k_small.metal `RadixTopKSmallPass` | timeout |
| 15 | `overload:frag_get_in` | 36 | core/affine_mm.metal `tf_affine_mm` | error: use of undeclared identifier 'mpp' |
| 16 | `ident:execution_simdgroups` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 17 | `other:a type specifier is required for all declarations` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 18 | `other:expected unqualified-id` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 19 | `other:type-id cannot have a name` | 36 | nemotron/coop_down_1_0.metal `custom_kernel_lane_qmm_coop_b8e6f3fa` | error: use of undeclared identifier 'execution_simdgroups' |
| 20 | `overload:frag_get` | 34 | core/affine_mm.metal `tf_affine_row_sums` | error: use of undeclared identifier 'mpp' |
| 21 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_destination` | 34 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 22 | `spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_uchar]'X'N[%N]'X'N[%_ptr_` | 32 | core/row_projection.metal `tf_row_projection_q2_f32` | error: line 1054: OpPhi's result type <id> '313[%_ptr_StorageBuffer_uchar]' does not match |
| 23 | `other:conditional expression is ambiguous; 'X' can be converted to 'X' and vice ` | 28 | softmax/softmax.metal `Softmax_v1` | error: conditional expression is ambiguous; 'const __global metal::bfloat' can be converte |
| 24 | `type:MatmulMode` | 19 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 25 | `ident:MatmulMode` | 19 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 26 | `ident:matmul_op` | 19 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 27 | `member:metal::execution_simdgroup` | 19 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 28 | `overload:__private tensor<__local uint8_t, dextents<int32_t, 2>, tensor_inline>` | 17 | decode/fn_lane.metal `fz_lane` | error: use of undeclared identifier 'fz_tile' |
| 29 | `overload:__private tensor<__local half, dextents<int32_t, 2>, tensor_inline>` | 17 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 30 | `member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_left_input_` | 16 | nemotron/attn_partial_1.metal `custom_kernel_lane_attention_partial` | error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka |
| 31 | `clspv:ptr addrspace(N)` | 15 | kimi/mla.metal `k3_mla_cache` | ptr addrspace(1) |
| 32 | `other:expected external declaration` | 13 | prefill/attention_nax.metal `custom_kernel_tf_attention_nax_bf16_` | error: use of undeclared identifier 'mpp' |
| 33 | `clspv:clspv exit -N without a diagnostic (crash)` | 9 | attention/ancestor_attention.metal `AncestorAttention_v0` | clspv exit -11 without a diagnostic (crash) |
| 34 | `ident:simdgroup_load` | 7 | kimi/dense_mma.metal `k3_slice_mma` | error: use of undeclared identifier 'simdgroup_load' |
| 35 | `ident:simdgroup_store` | 7 | kimi/synth.metal `k3_mma_peak` | error: use of undeclared identifier 'simdgroup_store' |
| 36 | `overload:store_hadamard_vector` | 4 | activation_transform/activation_transform.metal `ActivationTransform_v0` | error: no matching function for call to 'store_hadamard_vector' |
| 37 | `overload:row_reduce` | 4 | attention/attention_gemm.metal `AttentionGemm_v0` | error: use of undeclared identifier 'mpp' |
| 38 | `other:clang frontend command failed with exit code N (use -v to see invocation)` | 4 | gdn/chunked/causal_inv.metal `DeltaNetChunkedCausalInv_v0` | clang-19: error: clang frontend command failed with exit code 139 (use -v to see invocatio |
| 39 | `ident:sq_acc` | 4 | glm_kda_prompt.metal `glm_kda_pre` | error: use of undeclared identifier 'quad_dot' |
| 40 | `ident:mlx_sigmoid_precise` | 4 | glm_kda_prompt.metal `glm_kda_pre` | error: use of undeclared identifier 'quad_dot' |
