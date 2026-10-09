# Per-entry failure classes

The first message of every failing entry, grouped by stage and normalized message, largest first.

## front end: ident:mpp (60)

```
attention_attention_gemm_metal__AttentionGemm_v0_fd07b578.cl:697:22: error: use of undeclared identifier 'mpp'
  697 |   using MatmulMode = mpp::tensor_ops::matmul2d_descriptor::mode;
      |                      ^
```

Entries: `attention/attention_gemm.metal:AttentionGemm_v0`, `attention/attention_gemm.metal:AttentionGemm_v1`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGroupedCombine_v1`, `gdn/chunked/gram.metal:DeltaNetChunkedGram_v0`, `gdn/chunked/gram.metal:DeltaNetChunkedGram_v1`, `gdn/chunked/output_and_state.metal:DeltaNetChunkedOutputAndState_`, `gdn/chunked/output_and_state.metal:DeltaNetChunkedOutputAndState_`, `gdn/tree_verify/out.metal:BuildTreeOut_v0` and 52 more

## front end: ident:execution_simdgroups (36)

```
nemotron_coop_down_1_0_metal__custom_kernel_lane_qmm_coop_b8e6f3fa04e5_0403615e.cl:79:18: error: use of undeclared identifier 'execution_simdgroups'; did you mean 'execution_simdgroup'?
   79 |   matmul2d<desc, execution_simdgroups<2>> op;
      |                  ^~~~~~~~~~~~~~~~~~~~
```

Entries: `nemotron/coop_down_1_0.metal:custom_kernel_lane_qmm_coop_b8`, `nemotron/coop_down_2_0.metal:custom_kernel_lane_qmm_coop_04`, `nemotron/coop_down_2_1.metal:custom_kernel_lane_qmm_coop_0c`, `nemotron/coop_draft_1_0.metal:custom_kernel_lane_qmm_coop_4c`, `nemotron/coop_draft_2_0.metal:custom_kernel_lane_qmm_coop_af`, `nemotron/coop_draft_2_1.metal:custom_kernel_lane_qmm_coop_d7`, `nemotron/coop_eh_1_0.metal:custom_kernel_lane_qmm_coop_9e`, `nemotron/coop_eh_1_0_sk.metal:custom_kernel_lane_qmm_coop_sk` and 28 more

## front end: overload:__private dextents<int32_t, 2> (33)

```
tes_grouped_1134f4f64c06078d_lanes_metal__custom_kernel_lane_qmm_bytes_grouped_113_354ef955.cl:93:102: error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka '__private dextents<int, 2>')
   93 |   tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
      |                                              
```

Entries: `flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_35e293dbee531073-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_35e293dbee531073.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_68a0697c6f38c26d-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_68a0697c6f38c26d.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_6dc76fc8a1f7fac9-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_6dc76fc8a1f7fac9.metal:custom_kernel_lane_qmm_bytes_g` and 25 more

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

Entries: `attention/attention_two_pass.metal:AttentionTwoPass2_v0`, `attention/attention_two_pass.metal:AttentionTwoPass2_v1`, `qwen3_5/qmm_16_2048.metal:custom_kernel_qwen35_qmm_16_20`, `qwen3_5/qmm_2048_2048.metal:custom_kernel_qwen35_qmm_2048_`, `qwen3_5/qmm_2048_6144.metal:custom_kernel_qwen35_qmm_2048_`, `qwen3_5/qmm_248320_2048.metal:custom_kernel_qwen35_qmm_24832`, `qwen3_5/qmm_4096_2048.metal:custom_kernel_qwen35_qmm_4096_`, `qwen3_5/qmm_512_2048.metal:custom_kernel_qwen35_qmm_512_2` and 1 more

## front end: template:frag (4)

```
prefill_qmm6_nax_b_metal__tf_mm_bf16_f32_t_nax_a4a1ffe5.cl:54:32: error: no template named 'frag'
   54 | inline void k_loop_bf16(thread frag<float> (&acc)[TM][2], const device T* x, int K, int live, bool inside,
      |                                ^
```

Entries: `prefill/qmm6_nax_b.metal:tf_mm_bf16_f32_t_nax`, `prefill/qmm6_nax_b.metal:tf_qmm6_splitk_nax`, `prefill/qmm6_nax_b.metal:tf_parts_sum`, `prefill/qmm6_nax_b.metal:tf_attn256_nax`

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

## front end: other:? (3)

```
timeout
```

Entries: `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGrouped_v0`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGrouped_v1`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGroupedCombine_v0`

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

## front end: ident:quad_dot (2)

```
glm_kda_prompt_metal__glm_kda_pre_af477398.cl:96:18: error: use of undeclared identifier 'quad_dot'
   96 |         result = quad_dot<4, PER>(x, wb, s, bb);
      |                  ^~~~~~~~
```

Entries: `glm_kda_prompt.metal:glm_kda_pre`, `glm_kda_prompt.metal:glm_kda_pre_tp`

## front end: ident:sq_acc (2)

```
glm_kda_prompt_metal__glm_kda_post_bf2222ec.cl:71:38: error: use of undeclared identifier 'sq_acc'
   71 |     for (int i = 0; i < 4; ++i) po = sq_acc(po, float(SY[at + base + i]));
      |                                      ^~~~~~
```

Entries: `glm_kda_prompt.metal:glm_kda_post`, `glm_kda_prompt.metal:glm_kda_post_tp`

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

## front end: ident:fsoftplus (1)

```
decode_fn_gdn_metal__fz_gdn_9852eab6.cl:79:61: error: use of undeclared identifier 'fsoftplus'
   79 |     gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
      |                                                             ^~~~~~~~~
```

Entries: `decode/fn_gdn.metal:fz_gdn`

## front end: ident:fz_tile (1)

```
decode_fn_lane_metal__fz_lane_aef0a838.cl:81:20: error: use of undeclared identifier 'fz_tile'
   81 |   const int tile = fz_tile(int(tgx));
      |                    ^~~~~~~
```

Entries: `decode/fn_lane.metal:fz_lane`

## front end: ident:simd_topk_all (1)

```
prefill_fn_prompt_metal__pf_route_13a7f1e4.cl:71:3: error: use of undeclared identifier 'simd_topk_all'
   71 |   simd_topk_all<NE, TOPK>(LG + size_t(r) * NL, lane, ids, picked);
      |   ^~~~~~~~~~~~~
```

Entries: `prefill/fn_prompt.metal:pf_route`

## front end: ident:simdgroup_load (1)

```
kimi_dense_mma_metal__k3_slice_mma_f8aa7f6b.cl:123:35: error: use of undeclared identifier 'simdgroup_load'
  123 |       for (int i = 0; i < 4; ++i) simdgroup_load(wa[i], ws[q & 1] + (32 * sn + 8 * i) * LD + 8 * kk, LD);
      |                                   ^~~~~~~~~~~~~~
```

Entries: `kimi/dense_mma.metal:k3_slice_mma`

## front end: ident:simdgroup_store (1)

```
kimi_synth_metal__k3_mma_peak_1434bf41.cl:110:3: error: use of undeclared identifier 'simdgroup_store'
  110 |   simdgroup_store(acc[0], out + (tg * 8 + sg) * 64, 8);
      |   ^~~~~~~~~~~~~~~
```

Entries: `kimi/synth.metal:k3_mma_peak`

## front end: other:excess elements in scalar initializer (1)

```
convolution_separable_causal_conv_metal__SeparableCausalConv_v0_c4e1e48c.cl:93:37: error: excess elements in scalar initializer
   93 |     const AccumulatorBlock weight = AccumulatorBlock(
      |                                     ^
```

Entries: `convolution/separable_causal_conv.metal:SeparableCausalConv_v0`

## front end: other:static assertion failed due to requirement 'X': group lane slices must contain complete chunks (1)

```
matmul_gemv_gemv_metal__Gemv_v1_8d06dc40.cl:848:17: error: static assertion failed due to requirement 'VALUES_PER_LANE % CHUNK_VALUES == 0': group lane slices must contain complete chunks
  848 |   static_assert(VALUES_PER_LANE % CHUNK_VALUES == 0, "group lane slices must contain complete chunks");
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Entries: `matmul/gemv/gemv.metal:Gemv_v1`

## front end: other:static assertion failed due to requirement 'X': output rows must divide row groups (1)

```
matmul_gemv_gemv_metal__Gemv_v0_ee134002.cl:334:17: error: static assertion failed due to requirement 'OUTPUT_ROWS % (NUM_SIMDGROUPS / K_SPLIT) == 0': output rows must divide row groups
  334 |   static_assert(OUTPUT_ROWS % (NUM_SIMDGROUPS / K_SPLIT) == 0, "output rows must divide row groups");
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Entries: `matmul/gemv/gemv.metal:Gemv_v0`

## front end: overload:Mark (1)

```
nemotron_sample_metal__tf_sample_full_8ad6ba3f.cl:266:44: error: no matching constructor for initialization of 'Mark'
  266 |   if (!(cum + mass(r, m, norm, lo, b, top, END, false, s) >= top_p)) return END;
      |                                            ^~~
```

Entries: `nemotron_sample.metal:tf_sample_full`

## spirv-val: spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_ushort]'X'N[%N]'X'N[%_ptr_StorageBuffer_uint]'. (1)

```
error: line 316: OpPhi's result type <id> '55[%_ptr_StorageBuffer_ushort]' does not match incoming value <id> '322[%322]' type <id> '52[%_ptr_StorageBuffer_uint]'.
  %177 = OpPhi %_ptr_StorageBuffer_ushort %172 %170 %322 %154
```

Entries: `flashnext/qa_ple_lookup_c9861329fad34fa4.metal:custom_kernel_qa_ple_lookup_c9`
