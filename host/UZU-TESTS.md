# host/UZU-TESTS.md - uzu metal unit-test status on Linux/vulkan-host

Baseline: uzu c92723bd + host/uzu-linux.patch (`apply-uzu.sh`). All 42 tests under
`crates/uzu-engine/unit/backends/metal` are gated `#![cfg(backend = "metal")]`; the
patched build script sets that on Linux with the `vulkan-host` feature.

## Status summary

- NOT-BUILT: all 42. The metal backend does not compile on Linux yet because the
  macOS-generated kernel bindings (`OUT_DIR/metal.rs`) are absent - blocking error
  below. No test reached the GPU; none was skipped or failed at runtime.

## Per-test inventory (unit/backends/metal, all NOT-BUILT - backend does not compile)

- kernel/matmul/gemv/policy_test.rs: fp_policy_cases, quant_policy_cases,
  gpu_core_count_boundary_selects_large_policy, quantized_policy_edges,
  untuned_quantized_io_uses_only_the_generated_fallback,
  block_unaligned_quantized_k_stays_on_gemv,
  specialization_preserves_quantized_route_and_accumulate_tail,
  gathered_group_major_is_a_gemv_route
- kernel/matmul/qmv/routes_test.rs: table_is_complete_and_fingerprint_is_stable,
  exact_lookup_rejects_non_matrix_inputs,
  normal_routing_handles_inputs_outside_the_frozen_matrix,
  family_lookup_requires_one_unanimous_route
- kernel/matmul/gemm/selection_test.rs: policy_boundaries_are_preserved,
  selection_fallbacks_and_split_k_are_preserved,
  trellis_plan_matches_projection_cases, forced_engine_errors_are_preserved,
  gemv_gemm_route_boundaries_are_preserved
- kernel/attention/gemm_grouped_policy_test.rs: measured_and_fallback_boundaries,
  should_encode_boundaries, long_prefill_split_selection_boundaries
- kernel/attention/kernel_test.rs: test_single_pass_attention_basic,
  test_matrix_attention_matches_vector_and_cpu_seq256,
  test_single_pass_attention_with_sinks,
  test_single_pass_attention_with_sinks_long_sequence, test_single_pass_attention_gqa,
  test_two_pass_attention, test_two_pass_attention_gqa, attention_kernel_matches_cpu,
  attention_kernel_reuses_instance_for_flat_and_trie,
  attention_kernel_ring_matches_full_on_wrap
- kernel/attention/gemm_test.rs: test_basic_f32, test_basic_bf16, test_causal_f32,
  test_causal_bf16, test_gqa_f32, test_gqa_bf16, test_head_dim_128_f32,
  test_head_dim_128_bf16, test_unaligned_f32, test_unaligned_bf16, test_prefill_mxu
- kernel/gdn/chunked_test.rs: chunked_prefill_matches_recurrent_prefill

## What the kernel pipeline must emit on Linux

uzu's macOS `MetalCompiler` (crates/uzu-engine/build/metal/compiler.rs) produces, per
`.metal` source, `<source>.rs` shards containing:

- `const MTLB_<blake3-of-relpath-UPPERCASE>: [&[u8]; N] =
  [include_bytes!(<shard>.metallib), ...];` - on Linux the bytes must be the `.m2vlib`
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

## Blocking error (raw, cargo check tail, 2026-10-09)

`cargo check -p uzu-engine --no-default-features --features metal,vulkan-host`
(build log from the 2026-10-09 run):

```
warning: `mtl-rs` (lib) generated 4 warnings
error: couldn't find file `<target-dir>/debug/build/uzu-engine/<hash>/out/metal.rs`
  --> crates/uzu-engine/src/backends/metal/kernel/mod.rs:24:1
   |
24 | include!(concat!(env!("OUT_DIR"), "/metal.rs"));
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
error: could not compile `uzu-engine` (lib) due to 1 previous error
```

That is the ONLY error: everything else - the compat crates, the whole metal
backend, the Linux `DeviceExt` forwards, LZFSE decode, the manifest swap - compiles.
