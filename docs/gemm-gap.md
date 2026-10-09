# uzu Gemm through the translator: where the 0.35x goes (G13C, fork ICD)

Kernel: uzu `Gemm` (SimdgroupMmaCore, f32, transposed B), compiled unmodified through the typed-GEP route. 1024^3, one
dispatch per timed iteration, 500 iterations.

| variant | SPIR-V | agx instrs | gprs | threads/core | 1024^3 GFLOP/s |
|---|---|---|---|---|---|
| 8x32x32, 1 simdgroup | 879 KB | not measured | not measured | not measured | 323 |
| 32x32x32, 2x2 simdgroups | 711 KB | not measured | not measured | not measured | 1443 |
| 64x64x32, 2x2 simdgroups | 2143 KB | 24649 | 179 | 576 | 1439 |
| 64x64x32, only the all-aligned k loop kept | 335 KB | 3724 | 135 | 768 | 1421 |
| 4x4 register-tiled (reference, same shim) | 37 KB | 430 | 127 | 832 | 4273 |
| hand GEMM (omarchy-mlx) | n/a | n/a | n/a | n/a | 4070 |

What this rules out:

- Code size and instruction cache. uzu instantiates the k loop for 9 alignment masks (8 dynamic via
  `dispatch_gemm_alignment` plus the all-aligned one). Keeping only the aligned copy shrinks the shader 6.6x and changes
  nothing (1421 vs 1439 GFLOP/s).
- Tile size. 32x32 and 64x64 tie.
- Spills. 0 spills and 0 scratch in all variants.

What is left: the staging path. uzu copies A and B tiles global -> threadgroup memory with scalar float loads and stores,
then reads each fragment from threadgroup memory per lane (two scalars per lane per 8x8 matrix), with a barrier per k
step. The register-tiled kernel reads fragments straight from global memory as float2 and has no barrier. The SPIR-V
also round-trips every fragment through a ushort array (`OpBitcast` to v4ushort and back) because simdgroup_matrix
copies are lowered to 16-bit moves; the agx compiler appears to remove this (0 scratch) but it has not been checked in the
ISA.

Decision: GEMM and quantized matmul hot paths go to the hand kernels through the host shim. The translated Gemm stays as a
correctness path and for kernels with no hand equivalent. A vectorized float4 loader for the staging copy is the next
experiment if the translated path has to carry weight.

Reproduce: `tools-spv-mix.sh` for the op counts, `AGX_MESA_DEBUG=shaderdb MESA_SHADER_CACHE_DISABLE=true` for the agx
numbers (stderr of the runner, `M2V_STDERR_FILE`).
