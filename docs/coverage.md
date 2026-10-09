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
| cov-frontend: host-preamble injection, CONSTRAINT-aware variants, simdgroup_load/store, vec conversions, `__generic` vector aliases | 340 of 609 (55.8%) | 94 | 246 | +15 |

## Failure classes

Counted by the entry's first error. The per-entry messages are in `sweep/results/FAILURES.md`, the ranked families in
`sweep/results/FEATURES.md`.

| # | class | entries | fix route | owner | status |
|---|---|---|---|---|---|
| 1 | Metal 4 tensor ops (`mpp::tensor_ops`, `tensor`, `dextents`, `execution_simdgroups`, cooperative tensors) | 138 | emulation: `matmul2d` and cooperative tensors on top of `simdgroup_matrix` and the cooperative matrix lowering | coopmat slice | open: gap list handed over |
| 2 | clspv does not finish (90 s limit; still no result at 400 s for the seven checked) | 42 | clspv patch: find the loop (pointer passes on large unrolled kernels), or an IR pre-pass that shrinks the module | sweep slice | open |
| 3 | clspv pointer passes (`OpPhi` of a `[4 x i8]` pointer and a typed pointer: 37, `Invalid bitcast`: 2, bitcast of a non-numeric type: 2) | 41 | IR post-pass in front of clspv that retypes the pointer, or a clspv patch | sweep slice | open |
| 4 | `c ? bfloat : float` is ambiguous (bfloat converts both ways) | 0 | closed on clang 23: `bfloat` is `__bf16` (a real arithmetic type, so Metal's rule - a conditional of bfloat and float is float - holds in C++), with `sweep/bfloat_to_i16.py` retyping the LLVM bfloat to i16 for clspv; the clang 19 struct fallback keeps the class open there | sweep slice | closed |
| 5 | clspv, other (`ptr addrspace(N)` operand: 15, `llvm.memcpy` with mixed address spaces: 1) | 16 | clspv patch or IR pre-pass | sweep slice | open |
| 6 | small front-end gaps (`template:frag`, clang frontend failures, the two `static_assert`s, `Mark`, `excess elements`) | 15 | shim header, constraint-aware variant picks | sweep slice | mostly closed: the `static_assert` tiles now honor uzu CONSTRAINT (gemv v0/v1 and qmm6 `tf_parts_sum` valid), the `template:frag` and qmm6 `tfq6` helpers come from the injected host preamble (the rest of qmm6_nax_b lands on the clspv timeout), `excess elements` reduced to the 4-arg vector functional cast through a typedef (`AccumulatorBlock(a, b, c, d)`, a clang OpenCL C++ limitation); `Mark` remains (clang rejects copy-constructing a program-scope `constant` struct). store_hadamard_vector, the bfloat `+=` and the constant-address-space initializer moved to fix/uzu-shim-easy |
| 7 | symbols the host provides (`quad_dot`, `sq_acc`, `fsoftplus`, `simd_topk_all`, `fz_tile`, `simdgroup_load`, `simdgroup_store` on a layout the shim lacks) | 6 | shim header, or a stand-in value in the driver | sweep slice | closed: the driver injects the TensorFold host preambles on demand (nax.h frag family, flashnext elementwise/topk, kda_rows quad_dot/sq_acc/mlx_sigmoid_precise, identity fz_tile), the shim gained simdgroup_load/simdgroup_store (Apple 8x8 layout); every entry is valid or on another class's named error |

Order of work: 4, 6 and 7 first (shim only, one commit each), then 3 and 5 (one IR pre-pass), then 2, with 1 following the coopmat slice.
The coverage number is posted after each class lands.

## Rules for this file

- A row closes when the sweep shows its entries valid, not when a fix is written.
- A class that splits gets a new row; counts are re-taken from the last sweep, so rows can change when an earlier class lands.
- A fix that makes SPIR-V valid but wrong is a regression; the run stage (`sweep/m2v_sweep_run.py`) is where those show.
