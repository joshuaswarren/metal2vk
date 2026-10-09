# Sibling-kernel parity harness

Proves that every kernel of the 32-kernel TensorFold sibling set computes the
same result through two paths:

1. **omarchy-mlx path**: `sweep/parity_mlx.py` dispatches each kernel with
   `mx.fast.metal_kernel`, using the exact source, header, template values and
   launch geometry the assembled artifacts pin. The call contract is recorded
   from the artifact assemblers' `build()` calls at runtime, never re-parsed.
2. **metal2vk path**: `sweep/parity_compare.py` dispatches the module compiled
   by the sweep (`spv/<tag>.spv` from the t1 run) through `m2v-run`, with
   buffers laid out from the sweep row args and the module reflection.

Both sides consume byte-identical inputs: `parity_mlx.py` generates them from
the seeded table in `parity_common.py` and saves `.npy` files;
`parity_compare.py` loads those files. bfloat16 travels as raw `uint16` bit
patterns. Output buffers on the metal2vk side are poisoned before the dispatch
so an unwritten region is caught.

## Running

`parity_mlx.py` runs under the python of the mlx wheel venv, on a GPU host,
through a guard ticket only. `parity_compare.py` runs on the host with
`m2v-run`. Every path is an environment variable; nothing is host-specific.

| variable | side | meaning | default |
|---|---|---|---|
| `PARITY_ARTIFACTS` | mlx | artifact root with the three assembler lanes | required |
| `PARITY_OUT` | both | output directory | `./parity-mlx-out` / `./parity-compare-out` |
| `PARITY_SEED` | both | input seed (must match) | `1234` |
| `PARITY_ONLY` | both | substring filter on kernel names | all |
| `PARITY_FAKE` | both | `1` = CPU dry run with the shared fake kernel | off |
| `PARITY_SWEEP` | compare | sweep result json | `./t1.json` |
| `PARITY_SPV_DIR` | compare | sweep output dir with `spv/` | `./OUT` |
| `PARITY_RUNNER` | compare | `m2v-run` binary | `./m2v-run` |
| `PARITY_MLX_OUT` | compare | mlx side's output directory | `./parity-mlx-out` |
| `PARITY_TSV` | compare | table path | `<PARITY_OUT>/parity.tsv` |

Every submit logs estimated and actual seconds in the timing files and in the
table, and refuses to launch when the estimate exceeds 5 s.

## Kernel table

`parity_common.py` carries one entry per kernel: launch geometry, template
values, per-input dtype/shape/generator and per-output dtype/shape. The
generators keep every access in bounds: floats uniform in [-1, 1), packed
4-bit weights as random uint32 words, expert indices inside a 4-expert weight
pool (the pool is small so parity buffers stay small; the addressing math is
identical), scalars/strides/params as exact constants matching the geometry.

`parity_mlx.py --check-assemblers --artifacts DIR` validates the table against
the artifacts with no mlx import: the three assemblers re-run into a temp dir
(in an isolated subprocess, since they install module import stubs that would
otherwise poison this process's later `import mlx.core`), their `.metal`
output must be byte-identical to the artifacts, every table entry must match
the recorded build() call and the `.launch` file, and every input of every
kernel must generate as plain numpy bytes (generation stays numpy-side; the
check runs in the fake mode too).

## Comparison and statuses

Integers compare exactly; floats by relative tolerance `|a-b| / (1+|b|)`:

| output dtype | tolerance |
|---|---|
| `uint32`, `int32`, `int64` | exact |
| `bfloat16` | 1e-2 |
| `float32`, `float16` | 1e-5 / 1e-3 |

The table reports max abs error, max rel error, first differing index and the
difference count. Positions where both sides are NaN are skipped and counted.

Statuses: `match`, `mismatch`, `refused-by-translator` (the omarchy path
refused the kernel; the four simple ones - router top-k, both softmax top-k
variants, combine row, sigmoid probe - additionally compare against a numpy
reference), `no-reference` (refused and not simple enough for a reference),
`skipped` (metal2vk compile failed in the sweep, or the submit guard tripped).

## Geometry deviations (recorded per kernel in the table)

The artifacts pin launch grids that were never GPU-validated; four of them
index out of the kernel's own buffers. Parity runs the kernel-consistent grid
and the artifact grid stays in `launch_grid` for the check:

- `omlx_qwen35_moe_router_gemv`: s = tg.y*NSG covers 2048 experts against 512
  weight rows; parity runs tg.y = 128.
- `omlx_qwen35_moe_down_combine_decode/window`: grid y writes y rows up to
  18432 against a 2048-row (4x2048-row) output; parity runs the covering 1024
  (2048).
- `omlx_verify_attn_gqa_partial`: grid x = 512 drives 512 kv heads against a
  2-kv-head template; parity runs x = 2.
- `omlx_chain_attn_partial`: grid x = heads*256 indexes q[qh*D] out of bounds
  for 16 heads; parity runs x = heads.
- `qwen35_ragged_sdpa_2p1`: grid (64,8,4) drives 64 kv heads x 8 batches; the
  template pins 2 kv heads; parity runs (2, 2, 4).

## Dry run (no GPU)

`PARITY_FAKE=1 parity_mlx.py` then `parity_compare.py --fake` proves the
plumbing on CPU: input generation, npy exchange, reflection-driven buffer
layout, dump handling, comparison and the table. The fake runner flips one
byte of the first kernel so a `mismatch` row exists. The committed
`sweep/results/parity.tsv` is that dry run's output (fake runner; the real
run overwrites it).

## What a CPU-only dry run cannot test

- real omarchy-mlx output bytes, real Vulkan dispatch and driver behaviour;
- the assumption that template values passed as python ints/dtypes render to
  the same MSL the assembler produced (the ticket's first run should diff the
  omarchy-generated MSL against the assembled artifact once, then trust it);
- wall-clock submit times (only the estimate model runs locally);
- whether the four compile-failed kernels pass after future metal2vk fixes.
