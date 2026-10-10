# Missing Metal features, ranked

Entries are kernel variants: one per uzu variant pair (first and last of each VARIANTS list) and one per TensorFold
host_name instantiation. A count is a number of entries.

## Compile rates

Before (the shim and flags of main when the sweep started):

| set | entries | IR | SPIR-V | valid | refused | runs | matches reference |
|---|---|---|---|---|---|---|---|
| uzu | 121 | 121 (100.0%) | 121 (100.0%) | 121 (100.0%) | 0 | not run here | 0 |
| tf | 489 | 489 (100.0%) | 479 (98.0%) | 477 (97.5%) | 0 | not run here | 0 |
| all | 610 | 610 (100.0%) | 600 (98.4%) | 598 (98.0%) | 0 | not run here | 0 |

After:

| set | entries | IR | SPIR-V | valid | refused | runs | matches reference |
|---|---|---|---|---|---|---|---|
| uzu | 121 | 121 (100.0%) | 121 (100.0%) | 121 (100.0%) | 0 | not run here | 0 |
| tf | 489 | 489 (100.0%) | 485 (99.2%) | 485 (99.2%) | 0 | not run here | 0 |
| all | 610 | 610 (100.0%) | 606 (99.3%) | 606 (99.3%) | 0 | not run here | 0 |

## Families, ranked by the entries they block first

| # | family | blocks first | touches | 
|---|---|---|---|
| 1 | clspv pointer passes (Invalid bitcast, undefined reference, OpPhi/OpSelect/AccessChain pointer types) | 4 | 4 |

## Top 40 raw diagnostics (entries affected, any position)

| # | feature | entries | example | first message |
|---|---|---|---|---|
| 1 | `clspv:Invalid bitcast` | 4 | kimi/experts.metal `k3_xp_up_r1` | Invalid bitcast |
