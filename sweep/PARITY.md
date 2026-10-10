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

The bfloat16 tolerance covers the references' one freedom: cross-lane `simd`
reductions are simulated as sequential float32 sums, so summation-order ulps
differ from the GPU; everything else in a reference is op-for-op identical to
the assembled MSL, including every explicit bf16 rounding. Because the
reduction lane order is defined by the hardware and not the source, a bf16
output is only judged a mismatch when it differs by more than 2 bf16 ulps of
its own binade in addition to the relative tolerance - the resolution floor
of the comparison, not a widened relative tolerance. The table reports
max abs error, max rel error, first differing index and the difference count.
Positions where both sides are NaN are skipped and counted.

Statuses: `match`, `mismatch`, `refused-by-translator` (the omarchy path
refused the kernel; the row is judged against the numpy reference in
`sweep/parity_refs.py`), `no-reference` (refused and no reference exists),
`no-cmp-output` (compare-only: stage B saved no dump for this kernel),
`skipped` (metal2vk compile failed in the sweep, or the submit guard tripped).

## References

`sweep/parity_refs.py` registers one numpy reference per kernel the omarchy
path refuses (all 25 at this writing: the five simple ones plus the fused MoE
expert, GDN prework/step/verify and ragged-SDPA kernels). Each reference
simulates the arithmetic of the assembled `.metal` op for op; `refs_common.py`
holds the rounding conventions (bf16 round-to-nearest-even, half via float16,
sequential-sum `simd` reductions) and `refs_router.py` is the worked example.
`python3 sweep/parity_refs.py` generates every seeded input and runs every
reference, checking names, shapes, dtypes and finiteness; it runs as part of
`sweep/tests/run.py`.

Three references settle the disagreements of the G13G runs:

- Router top-k selects over the softmax probabilities ROUNDED TO BF16
  (`vals[s*4+i] = float(static_cast<T>(...))` in the MSL), with ties broken
  toward the LARGER expert index (selection key `((bits+1) << 16) | expert`,
  descending). The pre-reference argsort over unrounded float64 probabilities
  disagreed on exactly the tied slots (22 of 32 indices on the saved inputs;
  the GPU run showed 19 - the gap is `precise::exp` rounding at bf16
  boundaries). `router_topk` (no softmax inside) selects by value with ties to
  the higher index, and renormalizes in f32 with one rounding.
- `combine_row` rounds EVERY op to bf16 (products, each accumulate step, the
  sigmoid chain `T(1+T(exp|g|)) -> T(1/e) -> T(1-y)`, the shared product, the
  final add). The op-for-op simulation is bit-identical to the metal2vk
  output; the earlier float64/one-rounding reference differed by up to 0.094
  purely from the per-op roundings.
- `chain_attn_partial`'s kv head is fixed by the row build, not by a second
  index map: the reference built per-query-head k/v blocks through the
  kv-head row bases and then indexed them AGAIN with the head map, so the
  head-1 query halves (qh 8..15) were scored against kv head 0's keys and
  values in every split, including the tail. With the double map dropped the
  reference matches the metal2vk output on all 20640 elements (max rel
  2.9e-6). An independent op-for-op simulation written from the MSL before
  looking at any dump reproduced the metal2vk output, which is what settled
  reference-vs-kernel.

None of the three is a metal2vk translation defect.

Two more settle the re-derivation of the five GDN step references
(decode/batch decode, the verify pair, the sibling main replay), all
reference-side: the kernels were correct.

- The step kernels' conv activation rounds the SiLU branch value to T
  BEFORE the product: `act = conv * T((conv < T(0)) ? sy : 1 - sy)`. A
  reference multiplying the unrounded f32 branch value lands about 2^-9
  off per activation; bf16 outputs absorb that, the f32 state does not
  (every element of every head with a non-negligible gate diverged).
- The sigmoid-shaped gates are the true sigmoid: `by = 1/(1+exp(|x|))`
  with the branch `(x < 0) ? by : 1 - by` is `sigmoid(x)` for both signs,
  so the reference is that one rounding with no further branch. The old
  verify references wrapped `sigmoid_f32` in a second identical branch,
  re-inverting beta and the norm-gate sig for every positive input; the
  dumps adjudicated the kernel.

For the two decode steps and the two verify steps, `SECONDARY` in
`refs_gdn_prework` / `refs_gdn_verify` keeps a composed-op variant of the
reference (the conv activation without the branch rounding; the sigmoid
transcription error corrected in both, so the check isolates the SiLU
rounding). compare-only runs it for refused kernels and records the
outcome in the table note as `composed-op check:`. A composed-vs-MSL
difference is a finding about the oMLX fallback's semantics, and the
assembled MSL is the Metal-intended behaviour. The sibling main replay
MSL spells the composed replay op for op (its `gdn_decay`/`gdn_beta` round
at exactly the reference's sites), so it has one reference serving both
and no separate secondary. `verify_step_states` never writes
`states[S-1]`; the row is expected to stay flagged on the unwritten hole.

## Compare-only mode

`parity_compare.py --compare-only --mlx-out DIR --cmp-out DIR` re-judges
saved outputs with no runner, no GPU and no sweep json: it loads the mlx
side's inputs and (where that path ran) outputs from DIR, this side's dumps
from DIR2, and falls back to the numpy reference for refused kernels. Stage B
saves every dispatch's dumps as `<out>/<kernel>/out_<name>.npy` (the pre-per-
kernel flat work directory overwrote dumps across kernels - that is why the
first G13G data lost all but the last writer of each filename). A GPU ticket
therefore only has to run stages A and B; the table is produced afterwards on
any machine.

Execution caveat: the modules assume 32-lane subgroups (every lane owns
NE/32 slots with `simd_*` reductions). A CPU submission through a software
ICD whose subgroup is smaller (lavapipe/llvmpipe reports 8) executes but
produces wrong cross-lane results; software-ICC runs are valid only for
kernels without cross-lane ops.

## Launch geometry

`mx.fast.metal_kernel` takes its grid in THREADS: `grid=(gx, gy, gz)` with
`threadgroup=(tx, ty, tz)` launches ceil(g / t) workgroups per axis. The
artifact `.launch` files record those thread grids, and the table keeps them in
`launch_grid` where they differ from the shape the references were written
against. Both sides launch the same thread grid; `parity_compare.py` divides it
by the threadgroup size before calling `m2v-run`, which dispatches workgroup
counts (the first G13G runs passed the thread grid as counts: 256x too many
workgroups for the kernels with a 256 wide threadgroup, writing far outside
their outputs; that produced nondeterministic output blocks and the geometry
"over-launch" findings recorded earlier, all of which were this mistake and not
properties of the artifacts: for example the gate+up grid (32, 2306, 1) with
threadgroup (32, 2, 1) is exactly 1153 workgroups).

A launch can also UNDER-cover its template, and then the hole is a property of
the real call, not a defect: `qwen35_ragged_sdpa_2p1`'s artifact launch is
(64, 8, 4) threads over (32, 8, 1) threadgroups = (2, 1, 4) workgroups, so
`batch_idx` only ever takes 0 and the kernel never writes
`partials[16384:]`, `sums[64:]` or `maxs[64:]` (the batch-1 half the template
could cover with a wider launch). The reference derives its workgroup counts
from `launch_grid / tg`, simulates only the launched workgroups, zeros the
never-written region (the `verify_step_states` convention) and its table row
says exactly which outputs the real launch leaves unwritten. The harness
buffers stay template-sized: the reference slices them to the launched
prefix.

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
