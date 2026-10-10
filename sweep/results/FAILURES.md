# Per-entry failure classes

The first message of every failing entry, grouped by stage and normalized message, largest first.

## clspv: clspv:Invalid bitcast (6)

```
Invalid bitcast
  %121 = bitcast <4 x i8> %120 to ptr addrspace(1)
Invalid bitcast
  %138 = bitcast <4 x i8> %137 to ptr addrspace(1)
Invalid bitcast
  %155 = bitcast <4 x i8> %154 to ptr addrspace(1)
```

Entries: `kimi/experts.metal:k3_xp_up_r1`, `kimi/experts.metal:k3_xp_up_r4`, `kimi/experts.metal:k3_xp_down_r1`, `kimi/experts.metal:k3_xp_down_r4`, `kimi/kda.metal:k3_kda`, `kimi/mla.metal:k3_mla_attend`

## clspv: clspv:ptr addrspace(N) (4)

```
ptr addrspace(1)
```

Entries: `kimi/mla.metal:k3_mla_cache`, `kimi/mla.metal:k3_mla_qlat`, `kimi/mla.metal:k3_mla_merge`, `kimi/mla.metal:k3_mla_uv`

## front end: overload:row_reduce (4)

```
attention_attention_gemm_metal__AttentionGemm_v0_637ac307.cl:1502:20: error: no matching member function for call to 'row_reduce'
 1502 |     score_fragment.row_reduce(block_max, -INFINITY, [](AccumType a, AccumType b) { return metal::max(a, b); });
      |     ~~~~~~~~~~~~~~~^~~~~~~~~~
```

Entries: `attention/attention_gemm.metal:AttentionGemm_v0`, `attention/attention_gemm.metal:AttentionGemm_v1`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGrouped_v0`, `attention/gemm_grouped/attention_gemm_grouped.metal:AttentionGemmGrouped_v1`

## front end: other: from vector 'X' (vector of N 'X' values) to vector 'X' (vector of N 'X' values) of different size (3)

```
ion_transform_activation_transform_metal__ActivationTransform_v1_669cf469.cl:274:25: error:  from vector 'float4' (vector of 4 'float' values) to vector 'vec<__bf16, 4>' (vector of 4 '__bf16' values) of different size
  274 |         values = float4(vec<T, 4>(values)) + float4(load_hadamard_vector(bias + first_index));
      |                         ^~~~~~~~~~~~~~~~~
```

Entries: `activation_transform/activation_transform.metal:ActivationTransform_v1`, `gated_act_mul/gated_act_mul.metal:GatedActMul_v1`, `matmul/gemm/gemm_split_k_reduce.metal:GemmSplitKReduce_v1`

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

## front end: other:static_cast from 'X' (aka 'X') to 'X' (vector of N 'X' values) is not allowed (1)

```
matmul_gemv_gemv_metal__Gemv_v0_616987e7.cl:531:35: error: static_cast from 'I4' (aka 'vec<__bf16, 4>') to 'float4' (vector of 4 'float' values) is not allowed
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

## front end: overload:select (1)

```
matmul_gemm_gemm_metal__Gemm_v1_3053a573.cl:1936:28: error: no matching function for call to 'select'
 1936 |           scales[tile_n] = select(ScaleVector(0), scales[tile_n], live);
      |                            ^~~~~~
```

Entries: `matmul/gemm/gemm.metal:Gemm_v1`
