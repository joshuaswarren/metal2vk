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

## Half inputs measured (G13C, 8x8x8 f16/f16/f32, register-tiled, one ticket)

`m2v_mma_f16` (clspv-patches/apply-coopmat-f16.py, shim overload behind `-DM2V_F16_COOPMAT`) lowers half A and B with an f32
accumulator. Output matches an f64 reference on the half-rounded inputs to 1.3e-6 (max relative error). Speed against the same kernel
with f32 inputs, 1024^3:

| tiling | half A/B | f32 A/B |
|---|---|---|
| 2x2 | 1608 GFLOP/s | 1640 |
| 4x4 | 4131 | 4262 |

No speed gain on G13C: the kernel is bound by the matrix path, not by operand loads. The value of the half path is that half buffers
are consumed without widening, which halves operand memory.
