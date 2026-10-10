# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

| set | entries | IR | SPIR-V | valid | refused | runs | matches reference |
|---|---|---|---|---|---|---|---|
| uzu | 121 | 121 (100.0%) | 121 (100.0%) | 121 (100.0%) | 0 | not run here | 0 |
| tf | 489 | 489 (100.0%) | 479 (98.0%) | 477 (97.5%) | 0 | not run here | 0 |
| all | 610 | 610 (100.0%) | 600 (98.4%) | 598 (98.0%) | 0 | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 8 | 8 |
| 2 | clspv: other | 4 | 4 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `clspv:Invalid bitcast` | 6 | kimi/experts.metal `k3_xp_up_r1` | Invalid bitcast |
| 2 | `clspv:ptr addrspace(N)` | 4 | kimi/mla.metal `k3_mla_cache` | ptr addrspace(1) |
| 3 | `spirv-val:error: line N: Expected input to be a pointer or int or float vector o` | 2 | glm/router.metal `custom_kernel_tf_glm5_fused_router_t` | error: line 259: Expected input to be a pointer or int or float vector or scalar: Bitcast |
