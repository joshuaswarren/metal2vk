# Per-entry failure classes

The first message of every failing entry, grouped by stage and normalized message, largest first.

## clspv: clspv:clspv pass gave up on a pointer rewrite; the module would be miscompiled (no SPIR-V kept) (8)

```
clspv pass gave up on a pointer rewrite; the module would be miscompiled (no SPIR-V kept)
M2V-GIVEUP: spb-iter-hash
M2V-GIVEUP: spb-iter-hash
```

Entries: `nemotron/expert_down.metal:custom_kernel_nemotron_rows_ex`, `nemotron/expert_up.metal:custom_kernel_nemotron_rows_ex`, `nemotron_expert_down.metal:custom_kernel_nemotron_rows_ex`, `nemotron_expert_up.metal:custom_kernel_nemotron_rows_ex`, `nemotron_experts.metal:tf_xup_rows2`, `nemotron_experts.metal:tf_xdown_rows2`, `nemotron_experts.metal:tf_xup_rows4`, `nemotron_experts.metal:tf_xdown_rows4`

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

## spirv-val: spirv-val:error: line N: Expected input to be a pointer or int or float vector or scalar: Bitcast (2)

```
error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast
  %153 = OpBitcast %uchar %151
```

Entries: `glm/router.metal:custom_kernel_tf_glm5_fused_ro`, `glm/router.metal:custom_kernel_tf_glm5_fused_ro`
