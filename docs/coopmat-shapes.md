# Cooperative matrix shapes on honeykrisp (G13C, fork ICD)

Printed by `probes/coopmat-props.c` (subgroup scope, no saturating accumulate):

| M x N x K | A | B | C / result |
|---|---|---|---|
| 8x8x8 | f32 | f32 | f32 |
| 8x8x8 | f16 | f16 | f32 |
| 8x8x8 | bf16 | bf16 | f32 |
| 8x8x8 | f16 | f16 | f16 |
| 16x16x16 | f32 | f32 | f32 |
| 16x16x16 | f16 | f16 | f32 |
| 16x16x16 | bf16 | bf16 | f32 |
| 16x16x16 | f16 | f16 | f16 |

The translator currently emits only the 8x8x8 f32 shape (`m2v_mma_f32`); half and bfloat inputs are widened to f32 first.
The f16 and bf16 inputs with f32 accumulate are the shapes a half or bfloat `simdgroup_matrix` would use. They have not been
measured against the f32 path yet.
