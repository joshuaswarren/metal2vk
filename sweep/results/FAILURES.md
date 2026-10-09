# Per-entry failure classes

The first message of every failing entry, grouped by stage and normalized message, largest first.

## spirv-val: spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_uchar]'X'N[%N]'X'N[%_ptr_StorageBuffer_ushort]'. (32)

```
error: line 908: OpPhi's result type <id> '214[%_ptr_StorageBuffer_uchar]' does not match incoming value <id> '1202[%1202]' type <id> '218[%_ptr_StorageBuffer_ushort]'.
  %626 = OpPhi %_ptr_StorageBuffer_uchar %615 %613 %1202 %266
```

Entries: `core/row_projection.metal:tf_row_projection_q2_f32`, `core/row_projection.metal:tf_row_projection_relu2_q2_f32`, `core/row_projection.metal:tf_row_projection_q2_bf16`, `core/row_projection.metal:tf_row_projection_relu2_q2_bf1`, `core/row_projection.metal:tf_row_projection_q4_f32`, `core/row_projection.metal:tf_row_projection_relu2_q4_f32`, `core/row_projection.metal:tf_row_projection_q4_bf16`, `core/row_projection.metal:tf_row_projection_relu2_q4_bf1` and 24 more

## clspv: clspv:ptr addrspace(N) (20)

```
ptr addrspace(1)
```

Entries: `kimi/experts.metal:k3_xp_up_r1`, `kimi/experts.metal:k3_xp_up_r4`, `kimi/experts.metal:k3_xp_down_r1`, `kimi/experts.metal:k3_xp_down_r4`, `kimi/mla.metal:k3_mla_cache`, `kimi/mla.metal:k3_mla_qlat`, `kimi/mla.metal:k3_mla_merge`, `kimi/mla.metal:k3_mla_uv` and 12 more

## front end: overload:frag_get (19)

```
core_affine_mm_metal__tf_affine_row_sums_6e1dd751.cl:192:13: error: no matching function for call to 'frag_get'
  192 |             frag_get(b[j][i], (const threadgroup bfloat*)tile, PAD, tn + 16 * i, kk + 16 * j, home);
      |             ^~~~~~~~
```

Entries: `core/affine_mm.metal:tf_affine_row_sums`, `core/affine_mm.metal:tf_affine_mm`, `core/affine_mm.metal:tf_affine_gather_32`, `core/affine_mm.metal:tf_affine_gather_64`, `core/affine_mm.metal:tf_affine_gather_scatter_32`, `core/affine_mm.metal:tf_affine_gather_scatter_64`, `glm_absorb_nax.metal:glm_absorb_nax`, `glm_sparse_nax.metal:glm_sparse_nax` and 11 more

## front end: overload:frag_get_in (19)

```
glm_attn_metal__glm_latent_scores_f9e7009f.cl:139:5: error: no matching function for call to 'frag_get_in'
  139 |     frag_get_in(a, q, 512, row0, k0, home, 64, 512);
      |     ^~~~~~~~~~~
```

Entries: `glm_attn.metal:glm_latent_scores`, `glm_attn.metal:glm_latent_values`, `prefill/attention_nax.metal:custom_kernel_tf_attention_nax`, `prefill/attention_nax.metal:custom_kernel_tf_attention_nax`, `prefill/attention_nax.metal:custom_kernel_tf_attention_nax`, `prefill/attention_nax.metal:custom_kernel_tf_attention_nax`, `prefill/fn_attn.metal:tf_idx_scores_nax`, `prefill/fn_attn.metal:tf_sattn_nax` and 11 more

## spirv-val: spirv-val:error: line N: Header block 'X' is contained in the loop construct headed by 'X', but its merge bloc (18)

```
error: line 1780: Header block '1428[%1428]' is contained in the loop construct headed by '515[%515]', but its merge block '1951[%1951]' is not
  %1428 = OpLabel
```

Entries: `nemotron/attn_partial_1.metal:custom_kernel_lane_attention_p`, `nemotron/attn_partial_10.metal:custom_kernel_lane_attention_p`, `nemotron/attn_partial_11.metal:custom_kernel_lane_attention_p`, `nemotron/attn_partial_12.metal:custom_kernel_lane_attention_p`, `nemotron/attn_partial_13.metal:custom_kernel_lane_attention_p`, `nemotron/attn_partial_14.metal:custom_kernel_lane_attention_p`, `nemotron/attn_partial_15.metal:custom_kernel_lane_attention_p`, `nemotron/attn_partial_16.metal:custom_kernel_lane_attention_p` and 10 more

## clspv: clspv:Invalid bitcast (11)

```
Invalid bitcast
  %303 = bitcast <4 x i16> %302 to <8 x i16>
Invalid bitcast
  %351 = bitcast <4 x i16> %350 to <8 x i16>
LLVM ERROR: Broken module found, compilation aborted!
```

Entries: `decode/fn_lane.metal:fz_lane`, `flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_35e293dbee531073.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_68a0697c6f38c26d.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_6dc76fc8a1f7fac9.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_88c1bf5dd9357153.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_98e87a6da79ccc48.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_a803238811388b40.metal:custom_kernel_lane_qmm_bytes_g` and 3 more

## spirv-val: spirv-val:error: line N: [VUID-StandaloneSpirv-None-N] Having N components for TypeVector requires the VectorN (11)

```
error: line 205: [VUID-StandaloneSpirv-None-12295] Having 8 components for TypeVector requires the Vector16 or LongVectorEXT capability
  %v8ushort = OpTypeVector %ushort 8
```

Entries: `flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_35e293dbee531073-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_68a0697c6f38c26d-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_6dc76fc8a1f7fac9-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_88c1bf5dd9357153-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_98e87a6da79ccc48-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_a803238811388b40-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_ba76543b6f461aa8-lanes.metal:custom_kernel_lane_qmm_bytes_g` and 3 more

## clspv: clspv:timeout (10)

```
timeout
```

Entries: `kimi/dense_mma.metal:k3_slice_mma`, `kimi/synth.metal:k3_mma_peak`, `prefill/qmm_nax.metal:custom_kernel_tf_qmm_splitk_pa`, `qwen3_5/qmm_16_2048.metal:custom_kernel_qwen35_qmm_16_20`, `qwen3_5/qmm_2048_2048.metal:custom_kernel_qwen35_qmm_2048_`, `qwen3_5/qmm_2048_6144.metal:custom_kernel_qwen35_qmm_2048_`, `qwen3_5/qmm_248320_2048.metal:custom_kernel_qwen35_qmm_24832`, `qwen3_5/qmm_4096_2048.metal:custom_kernel_qwen35_qmm_4096_` and 2 more

## front end: overload:load_paired_vectors (6)

```
gdn_chunked_output_and_state_metal__DeltaNetChunkedOutputAndState_v1_80824eb5.cl:889:13: error: no matching function for call to 'load_paired_vectors'
  889 |             load_paired_vectors(cooperative_right, right_col_0, right_col_1);
      |             ^~~~~~~~~~~~~~~~~~~
```

Entries: `gdn/chunked/output_and_state.metal:DeltaNetChunkedOutputAndState_`, `gdn/tree_verify/out.metal:BuildTreeOut_v1`, `gdn/tree_verify/tree_gram.metal:BuildTreeGram_v1`, `matmul/gemm/gemm.metal:Gemm_v1`, `matmul/gemm/gemm_trellis.metal:GemmTrellis_v0`, `matmul/gemm/gemm_trellis.metal:GemmTrellis_v1`

## front end: overload:row_reduce (4)

```
attention_attention_gemm_metal__AttentionGemm_v0_637ac307.cl:1502:20: error: no matching member function for call to 'row_reduce'
 1502 |     score_fragment.row_reduce(block_max, -INFINITY, [](AccumType a, AccumType b) { return metal::max(a, b); });
      |     ~~~~~~~~~~~~~~~^~~~~~~~~~
```

Entries: `attention/attention_gemm.metal:AttentionGemm_v0`, `attention/attention_gemm.metal:AttentionGemm_v1`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGrouped_v0`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGrouped_v1`

## clspv: clspv:MNV: SimplifyPointerBitcast does not converge; changing sub-passes: N N (3)

```
M2V: SimplifyPointerBitcast does not converge; changing sub-passes: 5 7
```

Entries: `ops/qmv.metal:tf_gather_qmv_b4_g64`, `ops/qmv.metal:tf_gather_qmv_b6_g64`, `ops/qmv.metal:tf_gather_qmv_b8_g64`

## front end: other: from vector 'X' (vector of N 'X' values) to vector 'X' (aka 'X') of different size (3)

```
ion_transform_activation_transform_metal__ActivationTransform_v1_669cf469.cl:274:25: error:  from vector 'float4' (vector of 4 'float' values) to vector 'vec<__bf16, 4>' (aka 'typename vec_sel<__bf16, 4>::type') of different size
  274 |         values = float4(vec<T, 4>(values)) + float4(load_hadamard_vector(bias + first_index));
      |                         ^~~~~~~~~~~~~~~~~
```

Entries: `activation_transform/activation_transform.metal:ActivationTransform_v1`, `gated_act_mul/gated_act_mul.metal:GatedActMul_v1`, `matmul/gemm/gemm_split_k_reduce.metal:GemmSplitKReduce_v1`

## front end: other:variable in constant address space must be initialized (2)

```
sampling_unified_sampling_metal__UnifiedSampling_v0_1a2c2607.cl:203:31: error: variable in constant address space must be initialized
  203 |   static const constant Logit LOWEST;
      |                               ^
```

Entries: `sampling/unified_sampling.metal:UnifiedSampling_v0`, `sampling/unified_sampling.metal:UnifiedSampling_v1`

## front end: overload:store_hadamard_vector (2)

```
ion_transform_activation_transform_metal__ActivationTransform_v0_cbcb1dd7.cl:276:7: error: no matching function for call to 'store_hadamard_vector'
  276 |       store_hadamard_vector(fp_out + element_index, values);
      |       ^~~~~~~~~~~~~~~~~~~~~
```

Entries: `activation_transform/activation_transform.metal:ActivationTransform_v0`, `gated_act_mul/gated_act_mul.metal:GatedActMul_v0`

## spirv-val: spirv-val:error: line N: Expected input to be a pointer or int or float vector or scalar: Bitcast (2)

```
error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast
  %153 = OpBitcast %uchar %151
```

Entries: `glm/router.metal:custom_kernel_tf_glm5_fused_ro`, `glm/router.metal:custom_kernel_tf_glm5_fused_ro`

## front end: other:excess elements in scalar initializer (1)

```
convolution_separable_causal_conv_metal__SeparableCausalConv_v0_c4e1e48c.cl:93:37: error: excess elements in scalar initializer
   93 |     const AccumulatorBlock weight = AccumulatorBlock(
      |                                     ^
```

Entries: `convolution/separable_causal_conv.metal:SeparableCausalConv_v0`

## front end: other:redefinition of 'X' (1)

```
ops_nax_gemm_metal__tf_nax_gemm_nn_bf16_8d9e6ed0.cl:236:15: error: redefinition of 'frag_home'
  236 | inline short2 frag_home(ushort l) {
      |               ^
```

Entries: `ops/nax_gemm.metal:tf_nax_gemm_nn_bf16`

## front end: other:static_cast from 'X' (aka 'X') to 'X' (vector of N 'X' values) is not allowed (1)

```
matmul_gemv_gemv_metal__Gemv_v0_616987e7.cl:531:35: error: static_cast from 'I4' (aka 'typename vec_sel<__bf16, 4>::type') to 'float4' (vector of 4 'float' values) is not allowed
  531 |       const float4 input_values = static_cast<float4>(*reinterpret_cast<const device I4*>(input));
      |                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Entries: `matmul/gemv/gemv.metal:Gemv_v0`

## front end: overload:Mark (1)

```
nemotron_sample_metal__tf_sample_full_8ad6ba3f.cl:266:44: error: no matching constructor for initialization of 'Mark'
  266 |   if (!(cum + mass(r, m, norm, lo, b, top, END, false, s) >= top_p)) return END;
      |                                            ^~~
```

Entries: `nemotron_sample.metal:tf_sample_full`

## front end: overload:mma_16x32 (1)

```
prefill_qmm6_nax_b_metal__tf_attn256_nax_58fcd5b1.cl:366:9: error: no matching function for call to 'mma_16x32'
  366 |         mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      |         ^~~~~~~~~~~~~~~~~~~~~~~
```

Entries: `prefill/qmm6_nax_b.metal:tf_attn256_nax`

## spirv-val: spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_ushort]'X'N[%N]'X'N[%_ptr_StorageBuffer_uint]'. (1)

```
error: line 316: OpPhi's result type <id> '55[%_ptr_StorageBuffer_ushort]' does not match incoming value <id> '322[%322]' type <id> '52[%_ptr_StorageBuffer_uint]'.
  %177 = OpPhi %_ptr_StorageBuffer_ushort %172 %170 %322 %154
```

Entries: `flashnext/qa_ple_lookup_c9861329fad34fa4.metal:custom_kernel_qa_ple_lookup_c9`
