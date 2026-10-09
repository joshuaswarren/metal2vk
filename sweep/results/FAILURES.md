# Per-entry failure classes

The first message of every failing entry, grouped by stage and normalized message, largest first.

## front end: ident:mpp (63)

```
attention_attention_gemm_metal__AttentionGemm_v0_fd07b578.cl:697:22: error: use of undeclared identifier 'mpp'
  697 |   using MatmulMode = mpp::tensor_ops::matmul2d_descriptor::mode;
      |                      ^
```

Entries: `attention/attention_gemm.metal:AttentionGemm_v0`, `attention/attention_gemm.metal:AttentionGemm_v1`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGrouped_v0`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGrouped_v1`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGroupedCombine_v0`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGroupedCombine_v1`, `gdn/chunked/gram.metal:DeltaNetChunkedGram_v0`, `gdn/chunked/gram.metal:DeltaNetChunkedGram_v1` and 55 more

## clspv: clspv:timeout after N s (clspv does not finish) (39)

```
timeout
```

Entries: `attention/ancestor_attention.metal:AncestorAttention_v0`, `gdn/tree_verify/state_advance.metal:StateAdvance_v1`, `radix_top_k_small.metal:RadixTopKSmallPass`, `radix_top_k_small.metal:RadixTopKSmallCollect`, `glm/gemv_scores.metal:custom_kernel_tf_glm5_gemv_row`, `glm/gemv_scores_le32.metal:custom_kernel_tf_glm5_gemv_row`, `glm/gemv_scores_lt4.metal:custom_kernel_tf_glm5_gemv_row`, `glm/gemv_t_igate.metal:custom_kernel_tf_glm5_gemv_t_r` and 31 more

## front end: ident:execution_simdgroups (36)

```
nemotron_coop_down_1_0_metal__custom_kernel_lane_qmm_coop_b8e6f3fa04e5_0403615e.cl:79:18: error: use of undeclared identifier 'execution_simdgroups'
   79 |   matmul2d<desc, execution_simdgroups<2>> op;
      |                  ^
```

Entries: `nemotron/coop_down_1_0.metal:custom_kernel_lane_qmm_coop_b8`, `nemotron/coop_down_2_0.metal:custom_kernel_lane_qmm_coop_04`, `nemotron/coop_down_2_1.metal:custom_kernel_lane_qmm_coop_0c`, `nemotron/coop_draft_1_0.metal:custom_kernel_lane_qmm_coop_4c`, `nemotron/coop_draft_2_0.metal:custom_kernel_lane_qmm_coop_af`, `nemotron/coop_draft_2_1.metal:custom_kernel_lane_qmm_coop_d7`, `nemotron/coop_eh_1_0.metal:custom_kernel_lane_qmm_coop_9e`, `nemotron/coop_eh_1_0_sk.metal:custom_kernel_lane_qmm_coop_sk` and 28 more

## front end: overload:__private dextents<int32_t, 2> (33)

```
tes_grouped_1134f4f64c06078d_lanes_metal__custom_kernel_lane_qmm_bytes_grouped_113_354ef955.cl:93:102: error: no matching constructor for initialization of '__private dextents<int32_t, 2>' (aka '__private dextents<int, 2>')
   93 |   tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
      |                                              
```

Entries: `flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_1134f4f64c06078d.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_35e293dbee531073-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_35e293dbee531073.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_68a0697c6f38c26d-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_68a0697c6f38c26d.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_6dc76fc8a1f7fac9-lanes.metal:custom_kernel_lane_qmm_bytes_g`, `flashnext/lane_qmm_bytes_grouped_6dc76fc8a1f7fac9.metal:custom_kernel_lane_qmm_bytes_g` and 25 more

## spirv-val: spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_uchar]'X'N[%N]'X'N[%_ptr_StorageBuffer__struct_N]'. (32)

```
error: line 911: OpPhi's result type <id> '215[%_ptr_StorageBuffer_uchar]' does not match incoming value <id> '1206[%1206]' type <id> '219[%_ptr_StorageBuffer__struct_15]'.
  %628 = OpPhi %_ptr_StorageBuffer_uchar %618 %616 %1206 %269
```

Entries: `core/row_projection.metal:tf_row_projection_q2_f32`, `core/row_projection.metal:tf_row_projection_relu2_q2_f32`, `core/row_projection.metal:tf_row_projection_q2_bf16`, `core/row_projection.metal:tf_row_projection_relu2_q2_bf1`, `core/row_projection.metal:tf_row_projection_q4_f32`, `core/row_projection.metal:tf_row_projection_relu2_q4_f32`, `core/row_projection.metal:tf_row_projection_q4_bf16`, `core/row_projection.metal:tf_row_projection_relu2_q4_bf1` and 24 more

## front end: other:conditional expression is ambiguous; 'X' can be converted to 'X' and vice versa (27)

```
softmax_softmax_metal__Softmax_v1_e5728a3c.cl:78:42: error: conditional expression is ambiguous; 'const __global metal::bfloat' can be converted to 'float' and vice versa
   78 |   float maxval = (has_sinks && tid == 0) ? sinks[outer_index] : -FLT_MAX;
      |                                          ^ ~~~~~~~~~~~~~~~~~~   ~~~~~~~~
```

Entries: `softmax/softmax.metal:Softmax_v1`, `glm/kda_rows.metal:custom_kernel_tf_glm5_kda_rows`, `glm/kda_rows_tp.metal:custom_kernel_tf_glm5_kda_rows`, `glm/moe_gateup_1.metal:custom_kernel_tf_glm5_fused_mo`, `glm/moe_gateup_2.metal:custom_kernel_tf_glm5_fused_mo`, `glm/moe_gateup_2h.metal:custom_kernel_tf_glm5_fused_mo`, `glm_glue.metal:glm_swiglu`, `glm_glue.metal:glm_act2` and 19 more

## clspv: clspv:ptr addrspace(N) (15)

```
ptr addrspace(1)
```

Entries: `kimi/mla.metal:k3_mla_cache`, `kimi/mla.metal:k3_mla_qlat`, `kimi/mla.metal:k3_mla_merge`, `ops/qmv.metal:tf_qmv_wide_b4_g64_v2`, `ops/qmv.metal:tf_qmv_wide_b4_g64_v3`, `ops/qmv.metal:tf_qmv_wide_b4_g64_v4`, `ops/qmv.metal:tf_qmv_wide_b4_g64_v5`, `ops/qmv.metal:tf_qmv_wide_b6_g64_v2` and 7 more

## front end: other:clang frontend command failed with exit code N (use -v to see invocation) (4)

```
clang-19: error: clang frontend command failed with exit code 139 (use -v to see invocation)
Debian clang version 19.1.7 (3~deb12u1)
Target: spir
```

Entries: `gdn/chunked/causal_inv.metal:DeltaNetChunkedCausalInv_v0`, `gdn/chunked/causal_inv.metal:DeltaNetChunkedCausalInv_v1`, `gdn/tree_verify/tree_update_solve.metal:TreeUpdateSolve_v0`, `gdn/tree_verify/tree_update_solve.metal:TreeUpdateSolve_v1`

## front end: overload:store_hadamard_vector (4)

```
ion_transform_activation_transform_metal__ActivationTransform_v0_cbcb1dd7.cl:276:7: error: no matching function for call to 'store_hadamard_vector'
  276 |       store_hadamard_vector(fp_out + element_index, values);
      |       ^~~~~~~~~~~~~~~~~~~~~
```

Entries: `activation_transform/activation_transform.metal:ActivationTransform_v0`, `activation_transform/activation_transform.metal:ActivationTransform_v1`, `gated_act_mul/gated_act_mul.metal:GatedActMul_v0`, `gated_act_mul/gated_act_mul.metal:GatedActMul_v1`

## front end: template:frag (4)

```
prefill_qmm6_nax_b_metal__tf_mm_bf16_f32_t_nax_a4a1ffe5.cl:54:32: error: no template named 'frag'
   54 | inline void k_loop_bf16(thread frag<float> (&acc)[TM][2], const device T* x, int K, int live, bool inside,
      |                                ^
```

Entries: `prefill/qmm6_nax_b.metal:tf_mm_bf16_f32_t_nax`, `prefill/qmm6_nax_b.metal:tf_qmm6_splitk_nax`, `prefill/qmm6_nax_b.metal:tf_parts_sum`, `prefill/qmm6_nax_b.metal:tf_attn256_nax`

## spirv-val: spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_ushort]'X'N[%N]'X'N[%_ptr_StorageBuffer__struct_N]'. (4)

```
error: line 957: OpPhi's result type <id> '235[%_ptr_StorageBuffer_ushort]' does not match incoming value <id> '1308[%1308]' type <id> '239[%_ptr_StorageBuffer__struct_15]'.
  %631 = OpPhi %_ptr_StorageBuffer_ushort %619 %617 %1308 %294
```

Entries: `nemotron_experts.metal:tf_xup_rows2`, `nemotron_experts.metal:tf_xdown_rows2`, `nemotron_experts.metal:tf_xup_rows4`, `nemotron_experts.metal:tf_xdown_rows4`

## clspv: clspv:MNV: SimplifyPointerBitcast does not converge; changing sub-passes: N N (3)

```
M2V: SimplifyPointerBitcast does not converge; changing sub-passes: 5 7
```

Entries: `ops/qmv.metal:tf_gather_qmv_b4_g64`, `ops/qmv.metal:tf_gather_qmv_b6_g64`, `ops/qmv.metal:tf_gather_qmv_b8_g64`

## clspv: clspv:Invalid bitcast (2)

```
Invalid bitcast
  %220 = bitcast float %219 to ptr addrspace(1)
Invalid bitcast
  %263 = bitcast float %224 to ptr addrspace(1)
Invalid bitcast
  %295 = bitcast float %.in.peel to ptr addrspace(1)
```

Entries: `kimi/kda.metal:k3_kda`, `kimi/mla.metal:k3_mla_attend`

## front end: ident:quad_dot (2)

```
glm_kda_prompt_metal__glm_kda_pre_af477398.cl:96:18: error: use of undeclared identifier 'quad_dot'
   96 |         result = quad_dot<4, PER>(x, wb, s, bb);
      |                  ^
```

Entries: `glm_kda_prompt.metal:glm_kda_pre`, `glm_kda_prompt.metal:glm_kda_pre_tp`

## front end: ident:sq_acc (2)

```
glm_kda_prompt_metal__glm_kda_post_bf2222ec.cl:71:38: error: use of undeclared identifier 'sq_acc'
   71 |     for (int i = 0; i < 4; ++i) po = sq_acc(po, float(SY[at + base + i]));
      |                                      ^
```

Entries: `glm_kda_prompt.metal:glm_kda_post`, `glm_kda_prompt.metal:glm_kda_post_tp`

## front end: other:variable in constant address space must be initialized (2)

```
sampling_unified_sampling_metal__UnifiedSampling_v0_1a2c2607.cl:203:31: error: variable in constant address space must be initialized
  203 |   static const constant Logit LOWEST;
      |                               ^
```

Entries: `sampling/unified_sampling.metal:UnifiedSampling_v0`, `sampling/unified_sampling.metal:UnifiedSampling_v1`

## spirv-val: spirv-val:error: line N: Expected input to be a pointer or int or float vector or scalar: Bitcast (2)

```
error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast
  %153 = OpBitcast %uchar %151
```

Entries: `glm/router.metal:custom_kernel_tf_glm5_fused_ro`, `glm/router.metal:custom_kernel_tf_glm5_fused_ro`

## clspv: clspv:Instruction:   call void @llvm.memcpy.pN.pN.iN(ptr align N %.sroa.N, ptr addrspace(N) align N %N, iN (1)

```
Instruction:   call void @llvm.memcpy.p0.p1.i32(ptr align 8 %.sroa.045, ptr addrspace(1) align 2 %78, i32 8, i1 false)
Instruction:   call void @llvm.memcpy.p0.p1.i32(ptr align 8 %.sroa.044, ptr addrspace(1) align 2 %107, i32 8, i1 false)
Instruction:   call void @llvm.memcpy.p0.p1.i32(ptr align 8 %79, ptr addrspace(1) align 2 %78, i32 8, i1 false)
Instruction:   call void @llvm.memcpy.p0.p1.i32(ptr align 8 %107, ptr
```

Entries: `matmul/gemm/gemm_split_k_reduce.metal:GemmSplitKReduce_v1`

## front end: ident:fsoftplus (1)

```
decode_fn_gdn_metal__fz_gdn_9852eab6.cl:79:61: error: use of undeclared identifier 'fsoftplus'
   79 |     gates[r][0] = metal::exp(-metal::exp(float(ALOG[hv])) * fsoftplus(a + float(DT[hv])));
      |                                                             ^
```

Entries: `decode/fn_gdn.metal:fz_gdn`

## front end: ident:fz_tile (1)

```
decode_fn_lane_metal__fz_lane_aef0a838.cl:81:20: error: use of undeclared identifier 'fz_tile'
   81 |   const int tile = fz_tile(int(tgx));
      |                    ^
```

Entries: `decode/fn_lane.metal:fz_lane`

## front end: ident:simd_topk_all (1)

```
prefill_fn_prompt_metal__pf_route_13a7f1e4.cl:71:3: error: use of undeclared identifier 'simd_topk_all'
   71 |   simd_topk_all<NE, TOPK>(LG + size_t(r) * NL, lane, ids, picked);
      |   ^
```

Entries: `prefill/fn_prompt.metal:pf_route`

## front end: ident:simdgroup_load (1)

```
kimi_dense_mma_metal__k3_slice_mma_f8aa7f6b.cl:123:35: error: use of undeclared identifier 'simdgroup_load'
  123 |       for (int i = 0; i < 4; ++i) simdgroup_load(wa[i], ws[q & 1] + (32 * sn + 8 * i) * LD + 8 * kk, LD);
      |                                   ^
```

Entries: `kimi/dense_mma.metal:k3_slice_mma`

## front end: ident:simdgroup_store (1)

```
kimi_synth_metal__k3_mma_peak_1434bf41.cl:110:3: error: use of undeclared identifier 'simdgroup_store'
  110 |   simdgroup_store(acc[0], out + (tg * 8 + sg) * 64, 8);
      |   ^
```

Entries: `kimi/synth.metal:k3_mma_peak`

## front end: other:excess elements in scalar initializer (1)

```
convolution_separable_causal_conv_metal__SeparableCausalConv_v0_c4e1e48c.cl:93:37: error: excess elements in scalar initializer
   93 |     const AccumulatorBlock weight = AccumulatorBlock(
      |                                     ^
```

Entries: `convolution/separable_causal_conv.metal:SeparableCausalConv_v0`

## front end: other:no viable overloaded 'X' (1)

```
normalization_normalization_metal__Normalization_v1_da5e172d.cl:211:13: error: no viable overloaded '+='
  211 |         val += shortcut[i];
      |         ~~~ ^  ~~~~~~~~~~~
```

Entries: `normalization/normalization.metal:Normalization_v1`

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

## spirv-val: spirv-val:error: line N: OpPhi'X'N[%_ptr_StorageBuffer_uint]'X'N[%N]'X'N[%_ptr_StorageBuffer__struct_N]'. (1)

```
error: line 320: OpPhi's result type <id> '53[%_ptr_StorageBuffer_uint]' does not match incoming value <id> '324[%324]' type <id> '56[%_ptr_StorageBuffer__struct_21]'.
  %179 = OpPhi %_ptr_StorageBuffer_uint %174 %171 %324 %155
```

Entries: `flashnext/qa_ple_lookup_c9861329fad34fa4.metal:custom_kernel_qa_ple_lookup_c9`
