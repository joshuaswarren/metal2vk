# Coverage burn-down

Goal: every uzu and TensorFold Zig kernel produces valid Vulkan SPIR-V, then runs and matches a reference.

Measured with `sweep/m2v_sweep.py` on the typed-GEP route (clang 19, opt 19, clspv f2b01dd6 plus the cooperative matrix patch), 609 entry
points (uzu 121, TensorFold Zig 488). CI runs the same sweep on every PR and fails when an entry in `sweep/coverage-floor.txt` loses its valid
module; raise the floor with `sweep/check_coverage.py RESULT.json --update` in the PR that gains entries.

| when | valid SPIR-V | uzu | TensorFold | change |
|---|---|---|---|---|
| main's shim, sweep start | 232 of 609 (38.1%) | 70 | 162 | |
| shim additions (m2v_extra.h) | 316 of 609 (51.9%) | 75 | 241 | +84 |
| `m2v::mk` fix (clspv segfault on uzu gemm) | 323 of 609 (53.0%) | 82 | 241 | +7 |

## Failure classes

Counted by the entry's first error. The per-entry messages are in `sweep/results/FAILURES.md`, the ranked families in
`sweep/results/FEATURES.md`.

| # | class | entries | fix route | owner | status |
|---|---|---|---|---|---|
| 1 | Metal 4 tensor ops (`mpp::tensor_ops`, `tensor`, `dextents`, `execution_simdgroups`, cooperative tensors) | 138 | emulation: `matmul2d` and cooperative tensors on top of `simdgroup_matrix` and the cooperative matrix lowering | coopmat slice | open: gap list handed over |
| 2 | clspv does not finish (90 s limit; still no result at 400 s for the seven checked) | 42 | clspv patch: find the loop (pointer passes on large unrolled kernels), or an IR pre-pass that shrinks the module | sweep slice | open |
| 3 | clspv pointer passes (`OpPhi` of a `[4 x i8]` pointer and a typed pointer: 37, `Invalid bitcast`: 2, bitcast of a non-numeric type: 2) | 41 | IR post-pass in front of clspv that retypes the pointer, or a clspv patch | sweep slice | open |
| 4 | `c ? bfloat : float` is ambiguous (bfloat converts both ways) | 27 | shim, or a bfloat lowering so `bfloat` is a real arithmetic type | sweep slice | open |
| 5 | clspv, other (`ptr addrspace(N)` operand: 15, `llvm.memcpy` with mixed address spaces: 1) | 16 | clspv patch or IR pre-pass | sweep slice | open |
| 6 | small front-end gaps (`store_hadamard_vector`, `template:frag`, clang frontend failure, constant address space variables, two `static_assert`s, `Mark`, one `+=`, `excess elements`) | 15 | shim header | sweep slice | open |
| 7 | symbols the host provides (`quad_dot`, `sq_acc`, `fsoftplus`, `simd_topk_all`, `fz_tile`, `simdgroup_load`, `simdgroup_store` on a layout the shim lacks) | 6 | shim header, or a stand-in value in the driver | sweep slice | open |

Order of work: 4, 6 and 7 first (shim only, one commit each), then 3 and 5 (one IR pre-pass), then 2, with 1 following the coopmat slice.
The coverage number is posted after each class lands.

## Rules for this file

- A row closes when the sweep shows its entries valid, not when a fix is written.
- A class that splits gets a new row; counts are re-taken from the last sweep, so rows can change when an earlier class lands.
- A fix that makes SPIR-V valid but wrong is a regression; the run stage (`sweep/m2v_sweep_run.py`) is where those show.
