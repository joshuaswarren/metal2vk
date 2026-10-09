# metal2vk host layer (slice/host)

Goal: run uzu's Metal 4 backend (crates/uzu-engine/src/backends/metal) on Linux/Honeykrisp with no logic changes, and expose
the same device/queue/pipeline/dispatch model as a C ABI for non-Rust consumers (TensorFold's Zig engine, later MLX/llama.cpp).

```
uzu (cfg swap)  -> compat/mtl-rs (lib name `metal`, MTL* traits) + compat/objc2 (Retained, ProtocolObject stand-in)
                                   |
Zig / C / C++   -> metal2vk.h  -> m2v-host C ABI (src/capi.rs, cbindgen)
                                   |
                              m2v-host Rust core (ash)  -> Vulkan 1.3 (Honeykrisp)
```

## Layout

```
host/Cargo.toml            workspace: m2v-host, compat/mtl-rs, compat/objc2
host/m2v-host/             core + C ABI. crate-type = ["rlib", "cdylib", "staticlib"]. build.rs runs cbindgen -> include/metal2vk.h
host/m2v-host/include/metal2vk.h   checked in (generated), CI check that it is up to date
host/compat/mtl-rs/        package name `mtl-rs`, [lib] name = "metal": only the API subset uzu uses, same signatures as mtl-rs 0.3.0
host/compat/objc2/         package name `objc2`: Retained<T: ?Sized> (Arc-like), runtime::{ProtocolObject, AnyObject}, rc::autoreleasepool (no-op)
host/examples/zig/         20-line Zig program: @cImport("metal2vk.h"), runs one translated kernel, prints PASS
host/uzu-linux.patch       the cfg-swap patch for a uzu checkout (Cargo.toml + build.rs cfgs only), applied by host/apply-uzu.sh
```

## Core Rust API (m2v-host)  -- the contract between the core and the compat crate

All handles are `Arc<...>`; errors are `metal2vk::Error` (Display = message, plus `VkResult` where relevant).

```rust
Device::new() -> Result<Arc<Device>>            // first Vulkan device with a compute queue; env M2V_DEVICE=<index>
Device::name(&self) -> &str
Device::info(&self) -> &DeviceInfo              // subgroup_size, max_threads_per_threadgroup, max_shared_memory,
                                                // has_coopmat, has_timeline, total_memory, gpu_core_count (best effort: 0 if unknown)
Device::create_buffer(&self, size: usize) -> Result<Arc<Buffer>>
                                                // HOST_VISIBLE|HOST_COHERENT (unified memory), persistently mapped, usage = STORAGE|TRANSFER_SRC|DST|INDIRECT|SHADER_DEVICE_ADDRESS
Buffer::contents(&self) -> *mut u8 ; len() ; gpu_address(&self) -> u64
                                                // gpu_address = a VA in a shim-owned space: 16 KiB aligned, unique, never reused while the
                                                // buffer lives (Metal contract: address + offset stays inside one buffer). Device::resolve(addr)
                                                // -> Option<(Arc<Buffer>, offset)> via an interval map (BTreeMap, RwLock).
Library::from_bytes(&Arc<Device>, &[u8]) -> Result<Arc<Library>>         // .m2vlib container, format below
Library::function_names(&self) -> Vec<String>
Device::create_pipeline(&self, &Arc<Library>, name: &str, constants: &[(u32 /*function constant index*/, ConstantValue)])
                                                -> Result<Arc<Pipeline>>   // spec constants + workgroup size; VkPipelineCache on disk optional
Pipeline::max_threads_per_threadgroup() ; thread_execution_width() (= subgroup size) ; static_threadgroup_memory()
Queue = Device::create_queue() -> Result<Arc<Queue>>
Queue::new_command_buffer(&self) -> Result<CommandBuffer>              // pooled VkCommandPool per queue, reset on reuse
CommandBuffer::begin/compute_encoder -> Encoder (one compute encoder per buffer is enough)
Encoder::set_pipeline(&Arc<Pipeline>)
Encoder::set_address(index: u32, gpu_address: u64)                      // Metal 4 argument-table semantics: slot -> address; resolved when dispatching
Encoder::set_bytes(index: u32, &[u8])                                   // small constants, goes to push constants/inline buffer
Encoder::set_threadgroup_memory_length(index: u32, bytes: u32)
Encoder::dispatch_threadgroups(grid: [u32;3], tpt: [u32;3])
Encoder::dispatch_threads(threads: [u32;3], tpt: [u32;3])               // rounds up; kernels bounds-check, as in Metal non-uniform dispatch
Encoder::dispatch_indirect(gpu_address: u64 /*3 x u32*/, tpt: [u32;3])
Encoder::fill(addr_or_buffer, range, value: u8) ; copy(src, src_off, dst, dst_off, size)
Encoder::barrier()                                                      // compute->compute + transfer memory barrier (device visibility)
Encoder::push_debug_group(&str)/pop_debug_group() ; write_timestamp(&TimestampPool, index)
Encoder::end(self)
CommandBuffer::end()
Queue::submit(&self, &[&CommandBuffer], on_complete: Option<Box<dyn FnOnce(Completion) + Send>>)
Completion { error: Option<Error>, gpu_start_ns: u64, gpu_end_ns: u64 } // from timestamp queries around the buffer
Event = Device::create_event() -> Arc<Event>                           // timeline semaphore (Vulkan 1.2)
Queue::signal_event(&Event, value) ; Queue::wait_event(&Event, value)
Event::signaled_value() ; Event::wait(value, timeout) 
Device::limits_gpu_submit_seconds: every submit the shim issues must finish under the rule below
```

Descriptor strategy (decided, see Q1 in the PR): the pipeline reflection (below) lists the storage-buffer bindings and push-constant
words. At dispatch the encoder resolves each bound address to (VkBuffer, offset) and writes a push descriptor set (VK_KHR_push_descriptor,
falls back to a per-encoder descriptor pool). Buffers larger than maxStorageBufferRange are an error with a clear message.

## .m2vlib container (what compile.sh must emit; the core only reads it)

```
bytes 0..4   "M2VL"; u32 LE version = 1; u32 LE json_len; json (UTF-8); then SPIR-V blobs (4-byte aligned), offsets in the json
json = {
  "functions": [ {
     "name": "activation_gelu_f32",            // Metal function name used by new_library/function lookup
     "entry": "activation_gelu_f32",           // OpEntryPoint name in the SPIR-V
     "spv": {"offset": 1234, "len": 5678},
     "workgroup_size": [32,1,1],                // from reqd_work_group_size or spec constants 0..2 defaults
     "workgroup_size_spec_ids": [0,1,2] | null, // when the size is spec-constant driven
     "buffers": [ {"arg": 0, "binding": 0, "set": 0}, ... ],   // Metal argument-table slot -> descriptor binding
     "push": {"size": 24, "words": [ {"arg": 7, "offset": 16, "size": 4} ]}, // POD args, slot -> push constant offset
     "constants": [ {"index": 0, "spec_id": 3, "type": "bool|u32|i32|f32"} ],  // Metal function constant index -> SpecId
     "threadgroup": [ {"arg": 0, "spec_id": 4} ] | [],    // dynamic threadgroup memory: OpenCL local pointer args become spec-sized arrays
     "uses_coopmat": false,
     "gate": {"state": "unverified", "evidence": ""}  // state: verified | unverified (default). verified needs an evidence string: a test name, receipt path or commit. Written by make-m2vlib.py --gates
  } ]
}
```

## Rules

- Any single GPU submit this shim issues must finish under 5 s
  (`M2V_MAX_SUBMIT_MS`, default 5000); the shim aborts over-limit submissions
  in debug builds and reports them through `Completion::error`.
- Build artifacts go to a target directory outside tmpfs.
- No private paths, hostnames, or tokens in committed files.

## Untranslated and unverified kernels (policy.rs)

The host never runs a kernel it cannot trust. `Policy::route(kernel, translated)` runs before every dispatch:

| kernel state | route |
|---|---|
| translated, verified (sweep table row with a matching reference, or matched on first use) | run the translated kernel |
| translated, not verified, a reference exists | run it into shadow outputs, compute the reference into the real outputs, compare, `report_check`; the first dispatch uses the reference result |
| translated, not verified, no reference | refuse, error names the kernel |
| translated, failed its check | fallback, never the translated kernel again in this process |
| not translated (no SPIR-V, or the pipeline failed to build) | fallback |
| no fallback registered | refuse, `Error::Refused` names the kernel (C ABI: `m2v_status` UNSUPPORTED) |

A fallback is an omarchy-mlx hand kernel with the same contract (`{"hand": name}`) or a CPU reference registered by the application
(`{"cpu": name}`). Each routing event other than the verified path is logged once per kernel on stderr with the `m2v:` prefix and
kept in `Policy::events()`. There is no setting that disables the gate. The table is read from `M2V_POLICY_FILE` (a missing or
malformed file is an error); `tools/make-gate.py` builds its `kernels` section from `sweep-run.json`: `verified` only for rows whose
`ref` is ok, `failed` for rows whose `ref` failed, `unverified` for the rest. The `gate` field of a `.m2vlib` function (state verified, unverified or failed, plus evidence) seeds the table in `Device::route_kernel` when the policy file and earlier first-use checks have no entry for that kernel.

`Device::create_pipeline` goes through the gate: it builds a pipeline only for a Translated or TranslatedChecked route and returns
`Error::Refused` for a fallback route, so a kernel the gate does not trust cannot be dispatched by accident. Callers that want the
fallback call `Device::route_kernel` first. C ABI: `m2v_route_kernel(device, library, name, &route, reference, cap)` returns
`m2v_route_TRANSLATED`, `m2v_route_CHECKED`, `m2v_route_FALLBACK_HAND` or `m2v_route_FALLBACK_CPU` (with the reference name), or
`m2v_status` UNSUPPORTED naming the kernel when it is refused; `m2v_report_check(device, name, matched, detail)` reports the first-use check.
The GPU tests register a CPU reference for their kernel and report the result of their own comparison.

