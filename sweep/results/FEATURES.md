# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

| set | entries | IR | SPIR-V | valid | refused | runs | matches reference |
|---|---|---|---|---|---|---|---|
| uzu | 121 | 111 (91.7%) | 111 (91.7%) | 110 (90.9%) | 0 | not run here | 0 |
| tf | 488 | 486 (99.6%) | 473 (96.9%) | 440 (90.2%) | 0 | not run here | 0 |
| all | 609 | 597 (98.0%) | 584 (95.9%) | 550 (90.3%) | 0 | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | spirv-val: other | 32 | 32 |
| 2 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 8 | 8 |
| 3 | other | 6 | 7 |
| 4 | Apple MetalPerformancePrimitives / NAX (mpp::tensor_ops, tensor, dextents, cooperative tensors, matmul2d) | 5 | 5 |
| 5 | clspv: other | 4 | 4 |
| 6 | clspv does not finish (timeout, crash without a message, SimplifyPointerBitcast loop) | 3 | 3 |
| 7 | vector construction and conversion (mk1, C-style vector casts, scalar initializers) | 1 | 1 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `spirv-val:error: line N: Header block 'X' is contained in the loop construct hea` | 32 | gdn/chunked/output_and_state.metal `DeltaNetChunkedOutputAndState_v1` | error: line 11317: Header block '9087[%9087]' is contained in the loop construct headed by |
| 2 | `clspv:Invalid bitcast` | 6 | kimi/experts.metal `k3_xp_up_r1` | Invalid bitcast |
| 3 | `overload:row_reduce` | 4 | attention/attention_gemm.metal `AttentionGemm_v0` | error: no matching member function for call to 'row_reduce' |
| 4 | `clspv:ptr addrspace(N)` | 4 | kimi/mla.metal `k3_mla_cache` | ptr addrspace(1) |
| 5 | `other: from vector 'X' (vector of N 'X' values) to vector 'X' (vector of N 'X' v` | 3 | activation_transform/activation_transform.metal `ActivationTransform_v1` | error:  from vector 'float4' (vector of 4 'float' values) to vector 'vec<__bf16, 4>' (vect |
| 6 | `clspv:MNV: SimplifyPointerBitcast does not converge; changing sub-passes: N N` | 3 | ops/qmv.metal `tf_gather_qmv_b4_g64` | M2V: SimplifyPointerBitcast does not converge; changing sub-passes: 5 7 |
| 7 | `spirv-val:error: line N: Expected input to be a pointer or int or float vector o` | 2 | glm/router.metal `custom_kernel_tf_glm5_fused_router_t` | error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast |
| 8 | `other:excess elements in scalar initializer` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 9 | `other: from vector 'X' (aka 'X') to vector 'X' (aka 'X') of different size` | 1 | convolution/separable_causal_conv.metal `SeparableCausalConv_v0` | error: excess elements in scalar initializer |
| 10 | `overload:select` | 1 | matmul/gemm/gemm.metal `Gemm_v1` | error: no matching function for call to 'select' |
| 11 | `other:static_cast from 'X' (aka 'X') to 'X' (vector of N 'X' values) is not allo` | 1 | matmul/gemv/gemv.metal `Gemv_v0` | error: static_cast from 'I4' (aka 'vec<__bf16, 4>') to 'float4' (vector of 4 'float' value |
| 12 | `other:incompatible operand types ('X' and 'X')` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 13 | `overload:__private Mark` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 14 | `overload:Mark` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 15 | `other:kernel functions cannot be used in a template declaration, instantiation o` | 1 | nemotron_sample.metal `tf_sample_full` | error: no matching constructor for initialization of 'Mark' |
| 16 | `overload:mma_16x32` | 1 | prefill/qmm6_nax_b.metal `tf_attn256_nax` | error: no matching function for call to 'mma_16x32' |
