# host/STATUS.md — slice/host compat crates

## Compiles (receipt: `cargo check -p objc2 -p m2v-host -p mtl-rs`, nightly, x86_64 CT, 2026-10-09)

- `compat/objc2` — Retained (Arc clone), ProtocolObject (repr(transparent), last-field
  unsize), AnyObject/AnyThread/NSObjectProtocol markers, autoreleasepool (passthrough).
- `compat/mtl-rs` — the mtl-rs 0.3.0 API subset uzu's metal backend + generated kernel
  glue call, backed by m2v-host. All MTL* types/enums/bitflags carry upstream values.

## Backed by the host core

- Buffers = `m2v_host::Buffer` (contents/gpu_address/len), live-byte accounting for
  `current_allocated_size`.
- `set_address_at_index` -> argument table state (first-touch commit order), applied at
  the host encoder before each dispatch (`Encoder::set_address`).
- Pipelines: `new_library_with_data` parses `.m2vlib` bytes; the compiler shim maps
  MTL4 function/specialized descriptors + `MTLFunctionConstantValues` to
  `Device::create_pipeline(library, name, constants)`.
- Dispatch trio -> `Encoder::dispatch_threads/_threadgroups/_indirect` (usize->u32 casts).
- All three Metal 4 barrier forms -> `Encoder::barrier()`.
- fill/copy -> `Encoder::fill`/`copy`; debug groups -> push/pop_debug_group.
- Commit: `Queue::submit(&[&CommandBuffer], on_complete)`; feedback handler receives a
  shim `MTL4CommitFeedback` carrying `Completion` (gpu_start/end_ns as f64 seconds).
- Events -> timeline semaphores (`signal_event`/`wait_event`, `Event::wait`).
- Timestamps: counter heap = `TimestampPool`; `resolve_counter_range` returns
  ns-as-u64-NE, zero for unwritten slots; `query_timestamp_frequency` = 1e9 (already ns).

## Honest behavior (no fake values)

- `supports_placement_sparse_resources()` = false -> uzu `DeviceCapabilities::SPARSE_BUFFERS`
  stays unset, dense path. `new_buffer_with_length_options_placement_sparse_page_size` and
  `new_heap_with_descriptor` return None (uzu maps to CannotCreateBuffer/CannotCreateHeap).
- Residency sets: bookkeeping no-ops. Every host buffer is host-visible coherent for its
  whole life (unified memory), so "resident" is always true; add/remove/commit/request are
  recorded no-ops. Queue add/remove_residency_set likewise.
- Capture: `start_capture_with_descriptor` = Err("GPU capture is not supported by the m2v
  host layer"); `stop_capture` = no-op. `MTLCaptureManager` never silently fakes a trace.
- GPU family: `newest_supported_apple_gpu_family()` = Apple7, `supports_family` true for
  Apple1..=Apple7, `supports_mxu()` = false, `gpu_core_count` = DeviceInfo value
  (0 = driver does not report; uzu's LARGE_MIN_GPU_CORES threshold treats it conservatively).
- `MTLSharedEvent::set_signaled_value` panics (no host-side set on a timeline semaphore;
  reachable only from the sparse teardown, which never runs).

## Deviations from DESIGN.md (coordinated, see the DESIGN note in this commit)

- `MTLDevice::system_default` is the free fn `mtl::system_default_device()`: an associated
  fn on a protocol trait cannot dispatch through `dyn Trait`. uzu's single call site is
  rewritten by `host/uzu-linux.patch`.
- `MTL4FunctionDescriptor::set_library` and
  `MTL4SpecializedFunctionDescriptor::set_constant_values` take `&Retained<...>` (the Arc
  must be cloned, a bare `&ProtocolObject` cannot recover it). uzu's
  `CompilerPipelineExtensions::compute_pipeline_state` parameter type changes to
  `&Retained<ProtocolObject<dyn MTLLibrary>>`; its call sites are unchanged.

## Not done (gaps)

- `host/uzu-linux.patch` + `host/apply-uzu.sh`: NOT WRITTEN yet in this commit. The cfg
  swap design is fixed (see the commit message): feature `vulkan-host` on Linux ->
  `backend="metal"`; `objc2`/`metal` deps un-gated for Linux via `[patch]` redirect to the
  compat crates; `device_extensions.rs` Linux variant forwards to `metal::MTLDeviceExt`;
  `decompression.rs` LZFSE needs a non-Apple decoder; kernel-generation stays macOS-only,
  so `include!(OUT_DIR/metal.rs)` fails with the missing generated bindings — the exact
  list to emit is every `*MetalKernel` struct + `autogen_kernels!` + `MTLB_*` library
  consts from the macOS bindgen run.
- `cargo check -p uzu-engine` with the patch: not run (depends on the patch).
- Unit tests on the Apple silicon host: not run (depends on kernel bindings; without them
  every test in `unit/backends/metal/kernel` is NOT-BUILT).
