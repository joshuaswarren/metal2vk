# host/UZU-TESTS.md  -  uzu metal unit-test status on Linux/vulkan-host

Baseline: uzu c92723bd + host/uzu-linux.patch at `596b11f`. All 42 tests under
`crates/uzu-engine/unit/backends/metal` are gated `#![cfg(backend = "metal")]`;
the patched build script sets that on Linux with the `vulkan-host` feature.

## Status summary (2026-10-09 session)

- 0 PASS / 0 FAIL / 42 NOT-RUN. The aarch64 test binary (`cargo test -p
  uzu-engine --features metal,vulkan-host --no-run`, CPU-only, chip label
  Apple M1 Max (G13C)) was still linking at session close: the build script
  and all dependency crates compile and the final `uzu-engine` codegen was
  in flight for ~50 min on a host whose compute is saturated by an
  unrelated service (build niced 19 as required). The build unit was left
  running; `TESTBUILD-RC` will appear in its log when linking completes.
- One execution attempt was made before the binary existed:
  `fp_policy_cases` returned rc=124 (300 s timeout) because it waited in
  cargo's target-dir lock behind the unfinished build. Not a test failure.
- Every test's exact command is listed below; rerun them one exact name at
  a time (the custom harness takes a single whole-name filter) once the
  binary links:
  `cargo test -q -p uzu-engine --features metal,vulkan-host -- <name>`
- Kernels needed at runtime come from the translation pipeline
  (host/KERNELS.md: 23/761 translated). Tests that construct a Metal device
  are NOT-RUN here because G13C runs the vulkan-host path with no Metal
  device; they need the exact command above on a host that can satisfy
  `MetalDevice::new()`.

## Per-test inventory (unit/backends/metal)

### kernel/matmul/gemv/policy_test.rs  -  CPU-only, NOT-RUN (binary linking)

- fp_policy_cases  -  NOT-RUN (needs GPU: no; binary still linking; attempted
  once at rc=124, see summary)
- quant_policy_cases  -  NOT-RUN (needs GPU: no; binary still linking)
- gpu_core_count_boundary_selects_large_policy  -  NOT-RUN (same)
- quantized_policy_edges  -  NOT-RUN (same)
- untuned_quantized_io_uses_only_the_generated_fallback  -  NOT-RUN (same)
- block_unaligned_quantized_k_stays_on_gemv  -  NOT-RUN (same)
- specialization_preserves_quantized_route_and_accumulate_tail  -  NOT-RUN (same)
- gathered_group_major_is_a_gemv_route  -  NOT-RUN (same)

### kernel/matmul/qmv/routes_test.rs  -  CPU-only, NOT-RUN (binary linking)

- table_is_complete_and_fingerprint_is_stable  -  NOT-RUN (same)
- exact_lookup_rejects_non_matrix_inputs  -  NOT-RUN (same)
- normal_routing_handles_inputs_outside_the_frozen_matrix  -  NOT-RUN (same)
- family_lookup_requires_one_unanimous_route  -  NOT-RUN (same)

### kernel/matmul/gemm/selection_test.rs  -  CPU-only, NOT-RUN (binary linking)

- policy_boundaries_are_preserved  -  NOT-RUN (same)
- selection_fallbacks_and_split_k_are_preserved  -  NOT-RUN (same)
- trellis_plan_matches_projection_cases  -  NOT-RUN (same)
- forced_engine_errors_are_preserved  -  NOT-RUN (same)
- gemv_gemm_route_boundaries_are_preserved  -  NOT-RUN (same)

### kernel/attention/gemm_grouped_policy_test.rs  -  CPU-only, NOT-RUN (binary linking)

- measured_and_fallback_boundaries  -  NOT-RUN (same)
- should_encode_boundaries  -  NOT-RUN (same)
- long_prefill_split_selection_boundaries  -  NOT-RUN (same)

### kernel/attention/kernel_test.rs  -  NOT-RUN (needs GPU: Metal device)

- test_single_pass_attention_basic  -  NOT-RUN (needs GPU:
  `cargo test -q -p uzu-engine --features metal,vulkan-host -- test_single_pass_attention_basic`)
- test_matrix_attention_matches_vector_and_cpu_seq256  -  NOT-RUN (needs GPU, same form)
- test_single_pass_attention_with_sinks  -  NOT-RUN (needs GPU, same form)
- test_single_pass_attention_with_sinks_long_sequence  -  NOT-RUN (needs GPU, same form)
- test_single_pass_attention_gqa  -  NOT-RUN (needs GPU, same form)
- test_two_pass_attention  -  NOT-RUN (needs GPU, same form)
- test_two_pass_attention_gqa  -  NOT-RUN (needs GPU, same form)
- attention_kernel_matches_cpu  -  NOT-RUN (needs GPU, same form)
- attention_kernel_reuses_instance_for_flat_and_trie  -  NOT-RUN (needs GPU, same form)
- attention_kernel_ring_matches_full_on_wrap  -  NOT-RUN (needs GPU, same form)

### kernel/attention/gemm_test.rs  -  NOT-RUN (needs GPU: Metal device)

- test_basic_f32, test_basic_bf16, test_causal_f32, test_causal_bf16,
  test_gqa_f32, test_gqa_bf16, test_head_dim_128_f32, test_head_dim_128_bf16,
  test_unaligned_f32, test_unaligned_bf16, test_prefill_mxu  -  NOT-RUN
  (needs GPU: `cargo test -q -p uzu-engine --features metal,vulkan-host -- <name>`
  for each)

### kernel/gdn/chunked_test.rs  -  NOT-RUN (needs GPU: Metal device)

- chunked_prefill_matches_recurrent_prefill  -  NOT-RUN (needs GPU, same form)

## What the kernel pipeline must emit on Linux

uzu's macOS `MetalCompiler` (crates/uzu-engine/build/metal/compiler.rs) produces, per
`.metal` source, `<source>.rs` shards containing:

- `const MTLB_<blake3-of-relpath-UPPERCASE>: [&[u8]; N] =
  [include_bytes!(<shard>.metallib), ...];`  -  on Linux the bytes must be the `.m2vlib`
  produced by make-m2vlib.py, one entry per shard (`num_shards` from shard_footers),
- struct `XyzMetalKernel` per kernel holding
  `pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>` plus config fields,
  with `fn new(context: &MetalContext, <config>) -> Result<Self, MetalError>` calling
  `context.compute_pipeline_state(lib_bytes, compressed, cache_key, function_name,
  constants)`,
- per-variant `fn encode(self, command_buffer: &mut MetalCommandBufferEncoding, <args>)`
  calling `command_buffer.compute_encoder.set_compute_pipeline_state(&self.pipeline)`,
  `command_buffer.argument_table.set_address_at_index(addr, idx)` per buffer/constant
  argument (POD args first materialised via `allocate_constant(size)` + `copyin`),
  `command_buffer.compute_encoder.set_threadgroup_memory_length_at_index(len, idx)` for
  shared args, then one of `dispatch_threads_threads_per_threadgroup(MTLSize, MTLSize)` /
  `dispatch_threadgroups_threads_per_threadgroup(MTLSize, MTLSize)` /
  `dispatch_threadgroups_with_indirect_buffer_threads_per_threadgroup(gpu_addr, MTLSize)`.

`traitgen_all` emits the public `<Name>Kernel` traits that the hand-written modules
(`src/backends/metal/kernel/*`) implement, and `bindgen_global` emits `OUT_DIR/metal.rs`
(`include!` of the per-source shards + `autogen_kernels!()`, consumed by
`impl Kernels for MetalKernels` in `src/backends/metal/kernel/mod.rs`).

On Linux these items must be generated with identical names/signatures; only the
library byte constants change (`.m2vlib` instead of `.metallib`).

## Build state (raw tail, session close)

`cargo test -p uzu-engine --features metal,vulkan-host --no-run` on
Apple M1 Max (G13C), aarch64, niced:

```
warning: `uzu-engine` (build script) generated 2 warnings
   |
   |
warning: `mtl-rs` (lib) generated 8 warnings (run `cargo fix --lib -p mtl-rs` to apply 2 suggestions)
```

Compile of the uzu-engine library was still in flight; no TESTBUILD-RC line
yet. The earlier stage of the same log shows the dependency crates and the
build script completed (the build script runs the full kernel translation
pipeline; host/KERNELS.md records its diagnostics).
