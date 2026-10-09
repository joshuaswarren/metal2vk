# Per-entry failure classes

The first message of every failing entry, grouped by stage and normalized message, largest first.

## front end: member:metal::remove_addrspace_t (43)

```
core_affine_mm_metal__tf_affine_row_sums_6e1dd751.cl:106:68: error: no member named 'remove_addrspace_t' in namespace 'metal'
  106 |   auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
      |                                                                    ^~~~~~~~~~~~~~~~~~
```

Entries: `core/affine_mm.metal:tf_affine_row_sums`, `core/affine_mm.metal:tf_affine_mm`, `core/affine_mm.metal:tf_affine_gather_32`, `core/affine_mm.metal:tf_affine_gather_64`, `core/affine_mm.metal:tf_affine_gather_scatter_32`, `core/affine_mm.metal:tf_affine_gather_scatter_64`, `glm_absorb_nax.metal:glm_absorb_nax`, `glm_attn.metal:glm_latent_scores` and 35 more

## front end: ident:execution_simdgroups (36)

```
nemotron_coop_down_1_0_metal__custom_kernel_lane_qmm_coop_b8e6f3fa04e5_0403615e.cl:79:18: error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgroup'?
   79 |   matmul2d<desc, execution_simdgroups<2>> op;
      |                  ^~~~~~~~~~~~~~~~~~~~
```

Entries: `nemotron/coop_down_1_0.metal:custom_kernel_lane_qmm_coop_b8`, `nemotron/coop_down_2_0.metal:custom_kernel_lane_qmm_coop_04`, `nemotron/coop_down_2_1.metal:custom_kernel_lane_qmm_coop_0c`, `nemotron/coop_draft_1_0.metal:custom_kernel_lane_qmm_coop_4c`, `nemotron/coop_draft_2_0.metal:custom_kernel_lane_qmm_coop_af`, `nemotron/coop_draft_2_1.metal:custom_kernel_lane_qmm_coop_d7`, `nemotron/coop_eh_1_0.metal:custom_kernel_lane_qmm_coop_9e`, `nemotron/coop_eh_1_0_sk.metal:custom_kernel_lane_qmm_coop_sk` and 28 more

## front end: overload:__private dextents<int32_t, 2> (34)

```
decode_fn_lane_metal__fz_lane_aef0a838.cl:89:84: error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka '__private dextents<int, 2>')
   89 |   tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(64, M));
      |                                                                                    ^                    ~~~~~
```

Entries: `decode/fn_lane.metal:fz_lane`, `flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_35e293dbee531073-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_35e293dbee531073.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_68a0697c6f38c26d-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_68a0697c6f38c26d.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_6dc76fc8a1f7fac9-lanes.metal:custom_kernel_lane_qmm_bytes_g` and 26 more

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

## clspv: clspv:timeout after N s (clspv does not finish) (9)

```
timeout
```

Entries: `kimi/dense_mma.metal:k3_slice_mma`, `kimi/synth.metal:k3_mma_peak`, `qwen3_5/qmm_16_2048.metal:custom_kernel_qwen35_qmm_16_20`, `qwen3_5/qmm_2048_2048.metal:custom_kernel_qwen35_qmm_2048_`, `qwen3_5/qmm_2048_6144.metal:custom_kernel_qwen35_qmm_2048_`, `qwen3_5/qmm_248320_2048.metal:custom_kernel_qwen35_qmm_24832`, `qwen3_5/qmm_4096_2048.metal:custom_kernel_qwen35_qmm_4096_`, `qwen3_5/qmm_512_2048.metal:custom_kernel_qwen35_qmm_512_2` and 1 more

## front end: member:mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>::get_left_input_cooperative_tensor (6)

```
gdn_chunked_output_and_state_metal__DeltaNetChunkedOutputAndState_v1_80824eb5.cl:806:46: error: no member named 'get_left_input_cooperative_tensor' in 'mpp::tensor_ops::matmul2d<0, metal::execution_simdgroup>'
  806 |   auto cooperative_left = matmul_op.template get_left_input_cooperative_tensor<LeftType, RightType, OutputType>();
      |                           ~~~~~~~~~          ^
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

## clspv: clspv:Invalid bitcast (2)

```
Invalid bitcast
  %220 = bitcast float %219 to ptr addrspace(1)
Invalid bitcast
  %263 = bitcast float %224 to ptr addrspace(1)
Invalid bitcast
  %296 = bitcast float %.in.peel to ptr addrspace(1)
```

Entries: `kimi/kda.metal:k3_kda`, `kimi/mla.metal:k3_mla_attend`

## clspv: clspv:error: undefined reference to 'X' (2)

```
error: undefined reference to '_Z0Pfp9mma_16x32ILb0ELb1EEEvRU3AS4Dv8_fS3_RU3AS4KDv8_DF16bS6_S6_'
```

Entries: `prefill/qmm6_nax_b.metal:tf_mm_bf16_f32_t_nax`, `prefill/qmm6_nax_b.metal:tf_qmm6_splitk_nax`

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

## spirv-val: spirv-val:error: line N: Header block 'X' is contained in the loop construct headed by 'X', but its merge bloc (2)

```
error: line 455: Header block '359[%359]' is contained in the loop construct headed by '213[%213]', but its merge block '405[%405]' is not
  %359 = OpLabel
```

Entries: `ops/gemv.metal:tf_gemv_wide_bf16_v4_kl32`, `ops/gemv.metal:tf_gemv_wide_bf16_v5_kl32`

## front end: other:excess elements in scalar initializer (1)

```
convolution_separable_causal_conv_metal__SeparableCausalConv_v0_c4e1e48c.cl:93:37: error: excess elements in scalar initializer
   93 |     const AccumulatorBlock weight = AccumulatorBlock(
      |                                     ^
```

Entries: `convolution/separable_causal_conv.metal:SeparableCausalConv_v0`

## front end: other:redefinition of 'X' (1)

```
ops_nax_gemm_metal__tf_nax_gemm_nn_bf16_8d9e6ed0.cl:199:15: error: redefinition of 'frag_home'
  199 | inline short2 frag_home(ushort l) {
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
prefill_qmm6_nax_b_metal__tf_attn256_nax_58fcd5b1.cl:329:9: error: no matching function for call to 'mma_16x32'
  329 |         mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      |         ^~~~~~~~~~~~~~~~~~~~~~~
```

Entries: `prefill/qmm6_nax_b.metal:tf_attn256_nax`

## spirv-val: spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_ushort]'X'N[%N]'X'N[%_ptr_StorageBuffer_uint]'. (1)

```
error: line 316: OpPhi's result type <id> '55[%_ptr_StorageBuffer_ushort]' does not match incoming value <id> '322[%322]' type <id> '52[%_ptr_StorageBuffer_uint]'.
  %177 = OpPhi %_ptr_StorageBuffer_ushort %172 %170 %322 %154
```

Entries: `flashnext/qa_ple_lookup_c9861329fad34fa4.metal:custom_kernel_qa_ple_lookup_c9`
