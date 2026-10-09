# Coverage burn-down

Goal: every uzu and TensorFold Zig kernel produces valid Vulkan SPIR-V, then runs and matches a reference.

Measured with `sweep/m2v_sweep.py` on the typed-GEP route (clang 23, opt 23, clspv f2b01dd6 plus the cooperative matrix patch), 609 entry
points (uzu 121, TensorFold Zig 488). CI runs the same sweep on every PR and fails when an entry in `sweep/coverage-floor.txt` loses its valid
module; raise the floor with `sweep/check_coverage.py RESULT.json --update` in the PR that gains entries.

| when | valid SPIR-V | uzu | TensorFold | change |
|---|---|---|---|---|
| main's shim, sweep start | 232 of 609 (38.1%) | 70 | 162 | |
| shim additions (m2v_extra.h) | 316 of 609 (51.9%) | 75 | 241 | +84 |
| `m2v::mk` fix (clspv segfault on uzu gemm) | 323 of 609 (53.0%) | 82 | 241 | +7 |
| clang 23 toolchain, re-measured baseline | 325 of 609 (53.4%) | 84 | 241 | +2 |
| bfloat = `__bf16` + the bfloat_to_i16 pass | 382 of 609 (62.7%) | 90 | 292 | +57 |
| per-pointer-type UndefValue cache (clspv patch) | 417 of 609 (68.5%) | 92 | 325 | +35; all 37 OpPhi StorageBuffer entries become valid (the 32 row_projection + 4 nemotron_experts + 1 flashnext), 0 regressions, the 2 `bitcast of a non-numeric type` (glm/router) entries stay failing and move to a separate class |
| cov-frontend: host-preamble injection, CONSTRAINT-aware variants, simdgroup_load/store, vec conversions, `__generic` vector aliases (re-measured after the bfloat and ptr slices landed) | 401 of 609 (65.8%) | 102 | 299 | +19 over the 382 floor |
| cov-mpp: the MetalPerformancePrimitives shim (tensor views, `matmul2d` + cooperative tensors on the MLX/uzu/flashnext fragment layout, fp32 loops with subgroup shuffles), `mma_16x32` live behind M2V_NAX_MPP in the injected preamble, and the bfloat_to_i16 pass proving out on the flashnext modules (the clspv bfloat diagnostic loop is gone - those entries fail fast on class 3 now) | 441 of 609 (72.4%) | 102 | 339 | +40 over the 401 floor, no regressions |
| memcpy + ptr-array IR pre-passes (class 5 slice) | 446 of 609 (73.2%) | 102 | 344 | +45 over the 401 floor; all 12 `tf_qmv_wide` entries become valid (promote_ptr_arrays), the 3 `tf_gather_qmv` and 5 kimi/mla entries remain (pointer-as-runtime-data: needs a driver argument-model or clspv physical-pointer change); 2 new spirv-val loop-structure failures appeared in the uzu set |
| on main after the sib-shim + bfloat reconcile, re-measured with the ptr-slice clspv patches (undef cache, memcpy expansion, ptr-array promotion) | 486 of 609 (79.8%) | 102 | 384 | +45 over the 441 floor, no regressions; the 32 OpPhi and 12 qmv_wide entries hold; still failing: frag_get/frag_get_in overloads (38), spirv-val loop-structure (18), Invalid bitcast (15), 8-component TypeVector (11), timeouts (10), load_paired_vectors (6), ptr addrspace (4), non-convergence (3), bitcast non-numeric (2), plus a small tail |

## Failure classes

Counted by the entry's first error. The per-entry messages are in `sweep/results/FAILURES.md`, the ranked families in
`sweep/results/FEATURES.md`.

| # | class | entries | fix route | owner | status |
|---|---|---|---|---|---|
| 1 | Metal 4 tensor ops (`mpp::tensor_ops`, `tensor`, `dextents`, `execution_simdgroups`, cooperative tensors) | 40 | done for the constructs: the shim (`include/MetalPerformancePrimitives/MetalPerformancePrimitives.h`) implements every construct the entries reach (see the construct table below) and 40 + 8 earlier-class entries went valid with it; the 40 still stopping here all fail on one root cause: template argument deduction through the `vec` alias (`using vec = typename vec_sel<T, N>::type` is a non-deduced context; Apple's `vec` is a class template) in nax.h's frag family and uzu's `load_paired_vectors`/`row_reduce` - a real `vec` class template (needs native-vector swizzle/arithmetic parity) or concrete overloads everywhere | mpp slice | shim landed; 40 remain, all on the vec gap |
| 2 | clspv does not finish (90 s limit; still no result at 400 s for the seven checked) | 42 | clspv patch: find the loop (pointer passes on large unrolled kernels), or an IR pre-pass that shrinks the module | sweep slice | open |
| 3 | clspv pointer passes (`OpPhi` of a `[4 x i8]` pointer and a typed pointer: 37, `Invalid bitcast`: 2, bitcast of a non-numeric type: 2) | 0 (37 fixed) | clspv patch: per-SPIR-V-type UndefValue cache in `SPIRVProducerPass` (`clspv-patches/apply-undef-pointer-type.py`) plus a `getSPIRVValue` re-route so VMap-hits on a pointer undef go through the constant producer when the caller has a TyHint. All 37 OpPhi entries become valid (32 row_projection + 4 nemotron_experts + 1 flashnext). The 2 `Invalid bitcast` (kimi) and 2 `bitcast of a non-numeric type` (glm/router) are separate root causes and become class 8 | sweep slice | closed (37 / 37); the 4 bitcast cases move to class 8 |
| 4 | `c ? bfloat : float` is ambiguous (bfloat converts both ways) | 0 | closed on clang 23: `bfloat` is `__bf16` (a real arithmetic type, so Metal's rule - a conditional of bfloat and float is float - holds in C++), with `sweep/bfloat_to_i16.py` retyping the LLVM bfloat to i16 for clspv; the clang 19 struct fallback keeps the class open there | sweep slice | closed |
| 5 | clspv, other (`ptr addrspace(N)` operand: 15, `llvm.memcpy` with mixed address spaces: 1) | 16 | clspv patch or IR pre-pass | sweep slice | open |
| 6 | small front-end gaps (`template:frag`, clang frontend failures, the two `static_assert`s, `Mark`, `excess elements`) | 15 | shim header, constraint-aware variant picks | sweep slice | mostly closed: the `static_assert` tiles now honor uzu CONSTRAINT (gemv v0/v1 and qmm6 `tf_parts_sum` valid; on the `__bf16` shim the gemv pair moves to a bfloat-vector to float4 `static_cast`, the bfloat slice's type), the `template:frag` and qmm6 `tfq6` helpers come from the injected host preamble (the rest of qmm6_nax_b lands on the clspv timeout), `excess elements` reduced to the 4-arg vector functional cast through a typedef (`AccumulatorBlock(a, b, c, d)`, a clang OpenCL C++ limitation); `Mark` remains (clang rejects copy-constructing a program-scope `constant` struct). store_hadamard_vector, the bfloat `+=` and the constant-address-space initializer moved to fix/uzu-shim-easy |
| 7 | symbols the host provides (`quad_dot`, `sq_acc`, `fsoftplus`, `simd_topk_all`, `fz_tile`, `simdgroup_load`, `simdgroup_store` on a layout the shim lacks) | 6 | shim header, or a stand-in value in the driver | sweep slice | closed: the driver injects the TensorFold host preambles on demand (nax.h frag family, flashnext elementwise/topk, kda_rows quad_dot/sq_acc/mlx_sigmoid_precise, identity fz_tile), the shim gained simdgroup_load/simdgroup_store (Apple 8x8 layout); every entry is valid or on another class's named error |
| 8 | clspv bitcast (kimi `bitcast float -> ptr addrspace(1)`: 2, glm/router `OpBitcast %uchar` on a value clspv cannot type: 2) | 4 | clspv patch: either reject the bitcast in the producer's `I.print(errs())` path and re-emit via a typed cast, or an opt-19 IR pre-pass that lowers the source pattern | sweep slice | open |

Class 1 construct table (entries whose sources reach each construct, over all 609; class 1 blocked first on 98 entries at the c23 sweep, 40 after the shim):

| construct | entries | status |
|---|---|---|
| `tensor<T, dextents<int32_t, 2>, tensor_inline>` views over device/threadgroup memory, `slice`, strides | 77 | shim: implemented |
| `uint4b_format` elements (two per byte) | 36 | shim: implemented |
| `matmul2d_descriptor` (transposes, relaxed flag, multiply / multiply_accumulate) + `matmul2d<desc, scope>` (descriptor as integer NTTP via constexpr conversion; class-type NTTPs need C++20) | 77 | shim: implemented |
| `run(tensor, tensor, coop)` and mixed tensor/cooperative operands, fp32 accumulate | 77 | shim: implemented |
| `get_destination_cooperative_tensor` + per-element `operator[]` | 77 | shim: implemented |
| `execution_simdgroup` scope | 41 | shim: implemented |
| `execution_simdgroups<2>` scope (column bands; 32-row tiles row-banded, incl. one 32-lane group covering a 32-row tile) | 36 | shim: implemented |
| `get_multidimensional_index` (ids[0] = column, ids[1] = row) | 36 | shim: implemented |
| `get_left/right_input_cooperative_tensor`, the left one also initialized from another op's tensor | 16 | shim: implemented |
| `metal::remove_addrspace_t` | 48 | metal_stdlib: implemented |
| nax.h frag family + `mma_16x32` | 48 | `mma_16x32` is live behind M2V_NAX_MPP in the injected preamble (M5 packing, on the shim); the frag helpers remain on the vec-deduction gap |

The fragment layout the shim implements (MLX steel NAX, uzu `MxuFragmentOps<true>`, flashnext and nemotron all pack for it; verified bijective against all four call sites): lane l of a 32-lane group holds x = (l & 8) + 4 * (l & 1), y = 4 * ((l >> 4) & 1) + ((l >> 1) & 3); element i of a 16-row tile is at row y + 8 * ((i >> 2) & 1), column x + (i & 3) + 16 * (i >> 3); with `execution_simdgroups<N>` the 32-lane groups tile columns, or rows for 32-row tiles. uzu's `MxuStrictFragmentOps` packs for the other (strict) hardware layout; under the emulation its mma results land permuted - valid modules, Apple-relative numerics differ.

The clspv bfloat diagnostic loop this slice reported (class 2): the modules reached clspv with the LLVM bfloat type still in them and clspv looped printing `Err: SrcTy = bfloat - DstTy = i16 - CstVal = 16` (~1.4M lines in 120 s). The bfloat_to_i16 pass clears it - after the pass no bfloat type remains (only mangled names and metadata strings) and those entries fail fast on class 3's `Invalid bitcast` instead of timing out; the pass is not missing a construct for these modules.

Order of work: 4, 6 and 7 first (shim only, one commit each), then 3 and 5 (one IR pre-pass), then 2, with 1 following the coopmat slice.
The coverage number is posted after each class lands.

## Rules for this file

- A row closes when the sweep shows its entries valid, not when a fix is written.
- A class that splits gets a new row; counts are re-taken from the last sweep, so rows can change when an earlier class lands.
- A fix that makes SPIR-V valid but wrong is a regression; the run stage (`sweep/m2v_sweep_run.py`) is where those show.
