# host/KERNELS.md - uzu kernel translation (vulkan-host pipeline)

Run 1 of OUT_DIR/m2v-kernels.json: **23 translated / 738 failed / 761 total**

A kernel left untranslated is absent from its .m2vlib: creating its pipeline fails at runtime with the kernel name. Policy-only unit tests do not need the kernels.

## Failure reasons (top 5)

- 338 × `redefinition of 'GemmAPrologueKind'`
- 241 × `program scope variable must reside in global or constant address space`
- 47 × `redefinition of '__m2v_spec_0'`
- 16 × `variable in constant address space must be initialized`
- 11 × `unknown type name 'input'`

## Per-source counts

| source | translated | failed |
| --- | --- | --- |
| `activation/activation.metal` | 0 | 3 |
| `activation_transform/activation_transform.metal` | 0 | 4 |
| `activation_transform/trellis.metal` | 0 | 6 |
| `attention/ancestor_attention.metal` | 0 | 1 |
| `attention/attention_fallback.metal` | 0 | 4 |
| `attention/attention_gemm.metal` | 0 | 14 |
| `attention/attention_prepare.metal` | 0 | 1 |
| `attention/attention_single_pass.metal` | 0 | 8 |
| `attention/attention_two_pass.metal` | 0 | 16 |
| `attention/gemm_grouped/attention_gemm_grouped.metal` | 0 | 4 |
| `attention/kv_cache_update.metal` | 2 | 0 |
| `attention/qkv_norm.metal` | 0 | 8 |
| `attention/sigmoid_gate.metal` | 2 | 0 |
| `convolution/separable_causal_conv.metal` | 0 | 1 |
| `embedding/input_embedding_lookup.metal` | 0 | 2 |
| `gated_act_mul/gated_act_mul.metal` | 0 | 2 |
| `gdn/chunked/a_diag_inv.metal` | 2 | 0 |
| `gdn/chunked/causal_inv.metal` | 0 | 4 |
| `gdn/chunked/cumsum.metal` | 2 | 0 |
| `gdn/chunked/gram.metal` | 0 | 2 |
| `gdn/chunked/output_and_state.metal` | 0 | 16 |
| `gdn/conv_scan.metal` | 0 | 2 |
| `gdn/conv_update.metal` | 0 | 2 |
| `gdn/norm_gate.metal` | 0 | 2 |
| `gdn/prefill.metal` | 2 | 0 |
| `gdn/prefill_prep.metal` | 0 | 4 |
| `gdn/tree_verify/conv_scan.metal` | 0 | 2 |
| `gdn/tree_verify/out.metal` | 0 | 12 |
| `gdn/tree_verify/prefix.metal` | 0 | 1 |
| `gdn/tree_verify/state_advance.metal` | 0 | 2 |
| `gdn/tree_verify/tree_gram.metal` | 0 | 4 |
| `gdn/tree_verify/tree_update_solve.metal` | 0 | 4 |
| `gdn/update.metal` | 0 | 2 |
| `logit_transform/logit_transform.metal` | 0 | 2 |
| `matmul/gemm/gemm.metal` | 0 | 334 |
| `matmul/gemm/gemm_split_k_reduce.metal` | 0 | 2 |
| `matmul/gemm/gemm_trellis.metal` | 0 | 4 |
| `matmul/gemv/gemv.metal` | 0 | 187 |
| `normalization/normalization.metal` | 0 | 27 |
| `pooling/pooling.metal` | 3 | 0 |
| `radix_top_k_small.metal` | 0 | 2 |
| `sampling/context_ring_update.metal` | 1 | 0 |
| `sampling/repetition_penalty.metal` | 2 | 0 |
| `sampling/unified_sampling.metal` | 0 | 2 |
| `short_conv/short_conv.metal` | 2 | 10 |
| `softmax/softmax.metal` | 0 | 3 |
| `ssm/conv1d.metal` | 1 | 9 |
| `ssm/split_inproj.metal` | 3 | 0 |
| `ssm/ssd_prefill.metal` | 0 | 6 |
| `ssm/ssd_update.metal` | 0 | 3 |
| `tensor_add_bias/tensor_add_bias.metal` | 0 | 9 |
| `tensor_add_scale/tensor_add_scale.metal` | 0 | 3 |
| `weaver/weaver_frontier_insert_children.metal` | 1 | 0 |
| `weaver/weaver_frontier_select.metal` | 0 | 1 |
| `weaver/weaver_top_children.metal` | 0 | 1 |

