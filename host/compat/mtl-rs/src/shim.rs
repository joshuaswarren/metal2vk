//! Host-backed objects standing in for Metal objects. Every Metal 4 object uzu
//! holds (`Retained<ProtocolObject<dyn MTL...>>`) wraps one of these shims, which
//! owns the matching `m2v_host` handle.

use std::ffi::c_void;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use crate::descriptors::{
    ComputePipelineDescriptorFields, MTL4CommitOptions, MTLFunctionConstantValues,
};
use crate::error::MetalError;
use crate::protocols::{
    MTL4ArgumentTable, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue,
    MTL4CommitFeedback, MTL4Compiler, MTL4ComputeCommandEncoder, MTL4CounterHeap, MTLAllocation, MTLBuffer,
    MTLComputePipelineState, MTLDevice, MTLEvent, MTLLibrary, MTLResidencySet, MTLSharedEvent,
};
use crate::types::*;
use metal2vk::{Fallback, Route};
use objc2::{NSObjectProtocol, ProtocolObject, Retained};

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

pub struct DeviceShim {
    pub(crate) dev: Arc<metal2vk::Device>,
    pub(crate) name: String,
    pub(crate) gpu_core_count: u32,
    pub(crate) allocated: Arc<AtomicUsize>,
    main_queue: Mutex<Option<Arc<metal2vk::Queue>>>,
}

unsafe impl NSObjectProtocol for DeviceShim {}
unsafe impl MTLAllocation for DeviceShim {}

unsafe impl MTLDevice for DeviceShim {
    fn supports_family(&self, family: MTLGPUFamily) -> bool {
        // M1-class part: families Apple1..Apple7.
        matches!(family, MTLGPUFamily::Apple1 | MTLGPUFamily::Apple2 | MTLGPUFamily::Apple3 | MTLGPUFamily::Apple4 | MTLGPUFamily::Apple5 | MTLGPUFamily::Apple6 | MTLGPUFamily::Apple7)
    }

    fn sample_timestamps_gpu_timestamp(&self, cpu_timestamp: &mut u64, gpu_timestamp: &mut u64) {
        // Host timestamps are one clock: nanoseconds since the Unix epoch.
        let now = now_ns();
        *cpu_timestamp = now;
        *gpu_timestamp = now;
    }

    fn current_allocated_size(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    fn host_name(&self) -> String {
        self.name.clone()
    }

    fn host_gpu_core_count(&self) -> u32 {
        self.gpu_core_count
    }

    fn host_new_buffer(
        &self,
        length: usize,
        _options: MTLResourceOptions,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let buf = self.dev.create_buffer(length).ok()?;
        self.allocated.fetch_add(length, Ordering::Relaxed);
        Some({
                                        let arc: Arc<ProtocolObject<dyn MTLBuffer>> =
                    Arc::new(ProtocolObject::from_inner(BufferShim {
                buf,
                len: length,
                allocated: Arc::clone(&self.allocated),
            }));
                let obj: Retained<ProtocolObject<dyn MTLBuffer>> = Retained::from_arc(arc);
            obj
        })
    }

    fn host_new_library_with_data(&self, data: &[u8]) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        let lib = metal2vk::Library::from_bytes(&self.dev, data)?;
        Ok({
                                        let arc: Arc<ProtocolObject<dyn MTLLibrary>> =
                    Arc::new(ProtocolObject::from_inner(LibraryShim { lib }));
                let obj: Retained<ProtocolObject<dyn MTLLibrary>> = Retained::from_arc(arc);
            obj
        })
    }

    fn host_new_event(&self) -> Option<Retained<ProtocolObject<dyn MTLEvent>>> {
        Some({
                                        let arc: Arc<ProtocolObject<dyn MTLEvent>> =
                    Arc::new(ProtocolObject::from_inner(EventShim {
                event: self.dev.create_event(),
            }));
                let obj: Retained<ProtocolObject<dyn MTLEvent>> = Retained::from_arc(arc);
            obj
        })
    }

    fn host_new_shared_event(&self) -> Option<Retained<ProtocolObject<dyn MTLSharedEvent>>> {
        Some({
                                        let arc: Arc<ProtocolObject<dyn MTLSharedEvent>> =
                    Arc::new(ProtocolObject::from_inner(EventShim {
                event: self.dev.create_event(),
            }));
                let obj: Retained<ProtocolObject<dyn MTLSharedEvent>> = Retained::from_arc(arc);
            obj
        })
    }

    fn host_new_queue(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandQueue>>> {
        Some({
                                        let arc: Arc<ProtocolObject<dyn MTL4CommandQueue>> =
                    Arc::new(ProtocolObject::from_inner(QueueShim {
                queue: self.main_queue(),
            }));
                let obj: Retained<ProtocolObject<dyn MTL4CommandQueue>> = Retained::from_arc(arc);
            obj
        })
    }

    fn host_new_command_buffer(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandBuffer>>> {
        let cmd = self.main_queue().new_command_buffer().ok()?;
        Some({
                                        let arc: Arc<ProtocolObject<dyn MTL4CommandBuffer>> =
                    Arc::new(ProtocolObject::from_inner(CommandBufferShim {
                slot: Arc::new(Mutex::new(Some(cmd))),
                label: Mutex::new(None),
            }));
                let obj: Retained<ProtocolObject<dyn MTL4CommandBuffer>> = Retained::from_arc(arc);
            obj
        })
    }

    fn host_new_compiler(&self) -> Result<Retained<ProtocolObject<dyn MTL4Compiler>>, MetalError> {
        Ok({
                                        let arc: Arc<ProtocolObject<dyn MTL4Compiler>> =
                    Arc::new(ProtocolObject::from_inner(CompilerShim {
                dev: Arc::clone(&self.dev),
            }));
                let obj: Retained<ProtocolObject<dyn MTL4Compiler>> = Retained::from_arc(arc);
            obj
        })
    }

    fn host_new_counter_heap(
        &self,
        count: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTL4CounterHeap>>, MetalError> {
        let pool = self.dev.create_timestamp_pool(count.max(1) as u32)?;
        Ok({
                                        let arc: Arc<ProtocolObject<dyn MTL4CounterHeap>> =
                    Arc::new(ProtocolObject::from_inner(CounterHeapShim {
                pool: Arc::new(Mutex::new(pool)),
                count,
                r#type: MTL4CounterHeapType::TIMESTAMP,
            }));
                let obj: Retained<ProtocolObject<dyn MTL4CounterHeap>> = Retained::from_arc(arc);
            obj
        })
    }
}

impl DeviceShim {
    /// One VkQueue backs every Metal queue in the process; submits from all
    /// Metal queues serialize through it in commit order.
    fn main_queue(&self) -> Arc<metal2vk::Queue> {
        Arc::clone(
            self.main_queue
                .lock()
                .get_or_insert_with(|| self.dev.create_queue().expect("m2v-host: cannot create compute queue")),
        )
    }
}

static DEFAULT_DEVICE: LazyLock<Option<Retained<ProtocolObject<dyn MTLDevice>>>> = LazyLock::new(|| {
    let dev = metal2vk::Device::new().ok()?;
    let name = dev.name().to_owned();
    let info = dev.info().clone();
    Some({
                                    let arc: Arc<ProtocolObject<dyn MTLDevice>> =
                    Arc::new(ProtocolObject::from_inner(DeviceShim {
        dev,
        name,
        gpu_core_count: info.gpu_core_count,
        allocated: Arc::new(AtomicUsize::new(0)),
            main_queue: Mutex::new(None),
        }));
                let obj: Retained<ProtocolObject<dyn MTLDevice>> = Retained::from_arc(arc);
        obj
    })
});

// Runtime input (the capture descriptor hands us the device object), so OnceLock.
static CAPTURE_DEVICE: OnceLock<Arc<ProtocolObject<dyn MTLDevice>>> = OnceLock::new();

pub(crate) fn system_default_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    DEFAULT_DEVICE.clone()
}

pub(crate) fn default_device_handle() -> Retained<ProtocolObject<dyn MTLDevice>> {
    DEFAULT_DEVICE.clone().expect("no default device: system_default_device() failed to open a Vulkan device")
}

pub(crate) fn set_capture_device(device: Arc<ProtocolObject<dyn MTLDevice>>) {
    // The capture object is always the default device; keep the identity.
    let _ = CAPTURE_DEVICE.set(device);
}

fn now_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Buffer / library / pipeline / event / residency / allocator / queue
// ---------------------------------------------------------------------------

pub struct BufferShim {
    pub(crate) buf: Arc<metal2vk::Buffer>,
    len: usize,
    allocated: Arc<AtomicUsize>,
}

impl Drop for BufferShim {
    fn drop(&mut self) {
        self.allocated.fetch_sub(self.len, Ordering::Relaxed);
    }
}

unsafe impl NSObjectProtocol for BufferShim {}
unsafe impl MTLAllocation for BufferShim {}

unsafe impl MTLBuffer for BufferShim {
    fn length(&self) -> usize {
        self.len
    }
    fn contents(&self) -> NonNull<c_void> {
        let ptr = self.buf.contents() as *mut c_void;
        NonNull::new(ptr).unwrap_or(NonNull::dangling())
    }
    fn gpu_address(&self) -> u64 {
        self.buf.gpu_address()
    }
    fn host_buffer(&self) -> &Arc<metal2vk::Buffer> {
        &self.buf
    }
}

pub struct LibraryShim {
    pub(crate) lib: Arc<metal2vk::Library>,
}

unsafe impl NSObjectProtocol for LibraryShim {}

unsafe impl MTLLibrary for LibraryShim {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        default_device_handle()
    }
    fn host_library(&self) -> &Arc<metal2vk::Library> {
        &self.lib
    }
}

pub struct PipelineShim {
    pub(crate) pipeline: Arc<metal2vk::Pipeline>,
}

unsafe impl NSObjectProtocol for PipelineShim {}
unsafe impl MTLAllocation for PipelineShim {}

unsafe impl MTLComputePipelineState for PipelineShim {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        default_device_handle()
    }
    fn host_pipeline(&self) -> &Arc<metal2vk::Pipeline> {
        &self.pipeline
    }
}

pub struct EventShim {
    pub(crate) event: Arc<metal2vk::Event>,
}

unsafe impl NSObjectProtocol for EventShim {}

unsafe impl MTLEvent for EventShim {
    fn device(&self) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
        Some(default_device_handle())
    }
    fn host_event(&self) -> &Arc<metal2vk::Event> {
        &self.event
    }
}

unsafe impl MTLSharedEvent for EventShim {
    fn signaled_value(&self) -> u64 {
        self.event.signaled_value().unwrap_or(0)
    }
    fn set_signaled_value(&self, _signaled_value: u64) {
        // Host-side set has no timeline-semaphore equivalent; only the sparse
        // teardown (which never runs) would reach this.
        panic!("MTLSharedEvent::set_signaled_value is not supported by the m2v host layer")
    }
    fn wait_until_signaled_value_timeout_ms(&self, value: u64, milliseconds: u64) -> bool {
        self.event.wait(value, std::time::Duration::from_millis(milliseconds)).is_ok()
    }
}

/// Residency bookkeeping only: every host buffer is host-visible coherent for
/// its whole life (unified memory), so the set is always trivially satisfied.
pub struct ResidencySetShim;

unsafe impl NSObjectProtocol for ResidencySetShim {}

unsafe impl MTLResidencySet for ResidencySetShim {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        default_device_handle()
    }
    fn add_allocation(&self, _allocation: &ProtocolObject<dyn MTLAllocation>) {}
    fn remove_allocation(&self, _allocation: &ProtocolObject<dyn MTLAllocation>) {}
    fn commit(&self) {}
    fn request_residency(&self) {}
}

/// Command buffers are pooled per queue in the host; reset is bookkeeping.
pub struct AllocatorShim;

unsafe impl NSObjectProtocol for AllocatorShim {}

unsafe impl MTL4CommandAllocator for AllocatorShim {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        default_device_handle()
    }
    fn allocated_size(&self) -> u64 {
        0
    }
    fn reset(&self) {}
}

/// Every Metal queue shares the device's one compute queue: submits serialize in
/// commit order and timeline events keep the sparse-vs-main ordering honest.
pub struct QueueShim {
    pub(crate) queue: Arc<metal2vk::Queue>,
}

unsafe impl NSObjectProtocol for QueueShim {}

unsafe impl MTL4CommandQueue for QueueShim {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        default_device_handle()
    }
    fn signal_event_value(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64) {
        self.queue
            .signal_event(event.host_event(), value)
            .unwrap_or_else(|e| panic!("m2v-host: signal_event failed: {e}"));
    }
    fn wait_for_event_value(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64) {
        self.queue
            .wait_event(event.host_event(), value)
            .unwrap_or_else(|e| panic!("m2v-host: wait_event failed: {e}"));
    }
    fn add_residency_set(&self, _residency_set: &ProtocolObject<dyn MTLResidencySet>) {}
    fn remove_residency_set(&self, _residency_set: &ProtocolObject<dyn MTLResidencySet>) {}
    fn shim_queue(&self) -> &Arc<metal2vk::Queue> {
        &self.queue
    }
}

// ---------------------------------------------------------------------------
// Argument table
// ---------------------------------------------------------------------------

/// Argument-table entries in first-touch (commit) order; values update in place.
#[derive(Debug, Default)]
pub struct TableState {
    pub(crate) entries: Vec<(u32, u64)>,
}

#[derive(Debug, Default)]
pub struct ArgumentTableShim {
    pub(crate) state: Arc<Mutex<TableState>>,
}

impl ArgumentTableShim {
    pub fn new() -> Self {
        Self::default()
    }
}

unsafe impl NSObjectProtocol for ArgumentTableShim {}

unsafe impl MTL4ArgumentTable for ArgumentTableShim {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        default_device_handle()
    }
    fn set_address_at_index(&self, gpu_address: MTLGPUAddress, binding_index: usize) {
        let mut state = self.state.lock();
        match state.entries.iter_mut().find(|(index, _)| *index == binding_index as u32) {
            Some(entry) => entry.1 = gpu_address,
            None => state.entries.push((binding_index as u32, gpu_address)),
        }
    }
    fn host_state(&self) -> &Arc<Mutex<TableState>> {
        &self.state
    }
}

pub(crate) fn argument_table_state(
    table: Option<&ProtocolObject<dyn MTL4ArgumentTable>>,
) -> Option<Arc<Mutex<TableState>>> {
    table.map(|t| Arc::clone(t.as_inner().host_state()))
}

// ---------------------------------------------------------------------------
// Command buffer + compute encoder (recorded, flushed at end_encoding)
// ---------------------------------------------------------------------------

pub struct CommandBufferShim {
    pub(crate) slot: Arc<Mutex<Option<metal2vk::CommandBuffer>>>,
    pub(crate) label: Mutex<Option<String>>,
}

unsafe impl NSObjectProtocol for CommandBufferShim {}

unsafe impl MTL4CommandBuffer for CommandBufferShim {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        default_device_handle()
    }
    fn begin_command_buffer_with_allocator(&self, _allocator: &ProtocolObject<dyn MTL4CommandAllocator>) {}
    fn end_command_buffer(&self) {}
    fn compute_command_encoder(&self) -> Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>> {
        Some(encoder_for_slot(&self.slot))
    }
    fn host_slot(&self) -> &Arc<Mutex<Option<metal2vk::CommandBuffer>>> {
        &self.slot
    }
    fn shim_label(&self) -> Option<String> {
        self.label.lock().clone()
    }
    fn shim_set_label(&self, label: Option<&str>) {
        *self.label.lock() = label.map(str::to_owned);
    }
}

pub(crate) enum RecordedOp {
    Barrier,
    SetPipeline(Arc<metal2vk::Pipeline>),
    SetThreadgroupLength { index: u32, bytes: u32 },
    Copy { src: Arc<metal2vk::Buffer>, src_offset: u64, dst: Arc<metal2vk::Buffer>, dst_offset: u64, size: u64 },
    Fill { dst: Arc<metal2vk::Buffer>, range: Range<u64>, value: u8 },
    DispatchThreads([u32; 3], [u32; 3]),
    DispatchThreadgroups([u32; 3], [u32; 3]),
    DispatchIndirect(MTLGPUAddress, [u32; 3]),
    PushDebugGroup(String),
    PopDebugGroup,
    WriteTimestamp { pool: Arc<Mutex<metal2vk::TimestampPool>>, index: u32 },
    SetTable(Option<Arc<Mutex<TableState>>>),
}

/// Records encoder calls and replays them through one `m2v_host` recording
/// session (`begin` .. `end`) at `end_encoding`; the argument table is applied
/// before each dispatch (Metal 4 argument-table commit order).
pub struct ComputeEncoderShim {
    slot: Arc<Mutex<Option<metal2vk::CommandBuffer>>>,
    ops: Mutex<Vec<RecordedOp>>,
    flushed: AtomicBool,
}

unsafe impl NSObjectProtocol for ComputeEncoderShim {}

unsafe impl MTL4CommandEncoder for ComputeEncoderShim {
    fn barrier_after_queue_stages_before_stages_visibility_options(
        &self,
        _after_queue_stages: MTLStages,
        _before_stages: MTLStages,
        _visibility_options: MTL4VisibilityOptions,
    ) {
        self.record(RecordedOp::Barrier);
    }
    fn barrier_after_stages_before_queue_stages_visibility_options(
        &self,
        _after_stages: MTLStages,
        _before_queue_stages: MTLStages,
        _visibility_options: MTL4VisibilityOptions,
    ) {
        self.record(RecordedOp::Barrier);
    }
    fn barrier_after_encoder_stages_before_encoder_stages_visibility_options(
        &self,
        _after_encoder_stages: MTLStages,
        _before_encoder_stages: MTLStages,
        _visibility_options: MTL4VisibilityOptions,
    ) {
        self.record(RecordedOp::Barrier);
    }
    fn pop_debug_group(&self) {
        self.record(RecordedOp::PopDebugGroup);
    }
    fn end_encoding(&self) {
        self.end_encoding_impl();
    }
    fn shim_push_debug_group(&self, label: &str) {
        self.record(RecordedOp::PushDebugGroup(label.to_owned()));
    }
}

unsafe impl MTL4ComputeCommandEncoder for ComputeEncoderShim {
    fn set_compute_pipeline_state(&self, state: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.record(RecordedOp::SetPipeline(Arc::clone(state.as_inner().host_pipeline())));
    }
    fn set_threadgroup_memory_length_at_index(&self, length: usize, index: usize) {
        self.record(RecordedOp::SetThreadgroupLength { index: index as u32, bytes: length as u32 });
    }
    fn dispatch_threads_threads_per_threadgroup(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.record(RecordedOp::DispatchThreads(size3(threads_per_grid), size3(threads_per_threadgroup)));
    }
    fn dispatch_threadgroups_threads_per_threadgroup(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.record(RecordedOp::DispatchThreadgroups(
            size3(threadgroups_per_grid),
            size3(threads_per_threadgroup),
        ));
    }
    fn dispatch_threadgroups_with_indirect_buffer_threads_per_threadgroup(
        &self,
        indirect_buffer: MTLGPUAddress,
        threads_per_threadgroup: MTLSize,
    ) {
        self.record(RecordedOp::DispatchIndirect(indirect_buffer, size3(threads_per_threadgroup)));
    }
    fn copy_from_buffer_source_offset_to_buffer_destination_offset_size(
        &self,
        source_buffer: &ProtocolObject<dyn MTLBuffer>,
        source_offset: usize,
        destination_buffer: &ProtocolObject<dyn MTLBuffer>,
        destination_offset: usize,
        size: usize,
    ) {
        self.record(RecordedOp::Copy {
            src: Arc::clone(source_buffer.as_inner().host_buffer()),
            src_offset: source_offset as u64,
            dst: Arc::clone(destination_buffer.as_inner().host_buffer()),
            dst_offset: destination_offset as u64,
            size: size as u64,
        });
    }
    fn set_argument_table(&self, argument_table: Option<&ProtocolObject<dyn MTL4ArgumentTable>>) {
        self.record(RecordedOp::SetTable(argument_table_state(argument_table)));
    }
    fn shim_fill_buffer_range_value(&self, buffer: &ProtocolObject<dyn MTLBuffer>, range: Range<usize>, value: u8) {
        self.record(RecordedOp::Fill {
            dst: Arc::clone(buffer.as_inner().host_buffer()),
            range: range.start as u64..range.end as u64,
            value,
        });
    }
    fn shim_write_timestamp_with_granularity_into_heap_at_index(
        &self,
        granularity: MTL4TimestampGranularity,
        counter_heap: &ProtocolObject<dyn MTL4CounterHeap>,
        index: usize,
    ) {
        self.record(RecordedOp::WriteTimestamp {
            pool: Arc::clone(counter_heap.as_inner().pool_handle()),
            index: index as u32,
        });
        let _ = granularity;
    }
}

impl ComputeEncoderShim {
    fn record(&self, op: RecordedOp) {
        self.ops.lock().push(op);
    }

    fn end_encoding_impl(&self) {
        if !self.flushed.swap(true, Ordering::AcqRel) {
            self.flush();
        }
    }

    fn flush(&self) {
        let cmd = self.slot.lock().take().expect("m2v-host: command buffer already consumed");
        let mut encoder = cmd.begin().unwrap_or_else(|e| encode_failed("begin", e));
        let mut table: Option<Arc<Mutex<TableState>>> = None;
        for op in self.ops.lock().drain(..) {
            match op {
                RecordedOp::Barrier => encoder.barrier().unwrap_or_else(|e| encode_failed("barrier", e)),
                RecordedOp::SetPipeline(pipeline) => {
                    encoder.set_pipeline(&pipeline).unwrap_or_else(|e| encode_failed("set_pipeline", e))
                },
                RecordedOp::SetThreadgroupLength { index, bytes } => encoder
                    .set_threadgroup_memory_length(index, bytes)
                    .unwrap_or_else(|e| encode_failed("set_threadgroup_memory_length", e)),
                RecordedOp::Copy { src, src_offset, dst, dst_offset, size } => encoder
                    .copy(&src, src_offset, &dst, dst_offset, size)
                    .unwrap_or_else(|e| encode_failed("copy", e)),
                RecordedOp::Fill { dst, range, value } => {
                    encoder.fill(&dst, range, value).unwrap_or_else(|e| encode_failed("fill", e))
                },
                RecordedOp::DispatchThreads(threads, tpt) => {
                    apply_table(&mut encoder, &table);
                    encoder
                        .dispatch_threads(threads, tpt)
                        .unwrap_or_else(|e| encode_failed("dispatch_threads", e));
                },
                RecordedOp::DispatchThreadgroups(groups, tpt) => {
                    apply_table(&mut encoder, &table);
                    encoder
                        .dispatch_threadgroups(groups, tpt)
                        .unwrap_or_else(|e| encode_failed("dispatch_threadgroups", e));
                },
                RecordedOp::DispatchIndirect(address, tpt) => {
                    apply_table(&mut encoder, &table);
                    encoder
                        .dispatch_indirect(address, tpt)
                        .unwrap_or_else(|e| encode_failed("dispatch_indirect", e));
                },
                RecordedOp::PushDebugGroup(name) => encoder.push_debug_group(&name),
                RecordedOp::PopDebugGroup => encoder.pop_debug_group(),
                RecordedOp::WriteTimestamp { pool, index } => {
                    let guard = pool.lock();
                    encoder
                        .write_timestamp(&guard, index)
                        .unwrap_or_else(|e| encode_failed("write_timestamp", e));
                },
                RecordedOp::SetTable(new_table) => table = new_table,
            }
        }
        encoder.end().unwrap_or_else(|e| encode_failed("end encoder", e));
        cmd.end().unwrap_or_else(|e| encode_failed("end command buffer", e));
        *self.slot.lock() = Some(cmd);
    }
}

impl Drop for ComputeEncoderShim {
    fn drop(&mut self) {
        if !self.flushed.swap(true, Ordering::AcqRel) {
            self.flush();
        }
    }
}

fn apply_table(encoder: &mut metal2vk::Encoder, table: &Option<Arc<Mutex<TableState>>>) {
    let Some(state) = table else { return };
    for (index, address) in state.lock().entries.iter().copied() {
        encoder.set_address(index, address).unwrap_or_else(|e| encode_failed("set_address", e));
    }
}

fn encode_failed(what: &str, error: metal2vk::Error) -> ! {
    panic!("m2v-host: {what} failed: {error}")
}

pub(crate) fn encoder_for_slot(
    slot: &Arc<Mutex<Option<metal2vk::CommandBuffer>>>,
) -> Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>> {
    {
                                    let arc: Arc<ProtocolObject<dyn MTL4ComputeCommandEncoder>> =
                    Arc::new(ProtocolObject::from_inner(ComputeEncoderShim {
            slot: Arc::clone(slot),
            ops: Mutex::new(Vec::new()),
            flushed: AtomicBool::new(false),
        }));
                let obj: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>> = Retained::from_arc(arc);
        obj
    }
}

// ---------------------------------------------------------------------------
// Queue commit / feedback
// ---------------------------------------------------------------------------

pub(crate) fn queue_commit(
    queue: &Arc<metal2vk::Queue>,
    command_buffers: &[&ProtocolObject<dyn MTL4CommandBuffer>],
    options: &MTL4CommitOptions,
) {
    let guards: Vec<_> = command_buffers.iter().map(|c| c.as_inner().host_slot().lock()).collect();
    let refs: Vec<&metal2vk::CommandBuffer> = guards
        .iter()
        .map(|guard| guard.as_ref().expect("m2v-host: committing a command buffer that is still encoding"))
        .collect();
    let handlers = options.handlers.lock().clone();
    let on_complete: Option<Box<dyn FnOnce(metal2vk::Completion) + Send>> = if handlers.is_empty() {
        None
    } else {
        Some(Box::new(move |completion| {
            let arc: Arc<ProtocolObject<dyn MTL4CommitFeedback>> = Arc::new(ProtocolObject::from_inner(FeedbackShim { completion }));
            let feedback: Retained<ProtocolObject<dyn MTL4CommitFeedback>> = Retained::from_arc(arc);
            for handler in handlers {
                handler(&feedback);
            }
        }))
    };
    queue.submit(&refs, on_complete).expect("m2v-host: submit failed");
}

pub struct FeedbackShim {
    pub(crate) completion: metal2vk::Completion,
}

unsafe impl NSObjectProtocol for FeedbackShim {}

unsafe impl MTL4CommitFeedback for FeedbackShim {
    fn gpu_start_time(&self) -> f64 {
        self.completion.gpu_start_ns as f64 / 1e9
    }
    fn gpu_end_time(&self) -> f64 {
        self.completion.gpu_end_ns as f64 / 1e9
    }
    fn completion_error(&self) -> Option<MetalError> {
        self.completion.error.as_ref().map(|e| MetalError::new(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

pub struct CompilerShim {
    pub(crate) dev: Arc<metal2vk::Device>,
}

unsafe impl NSObjectProtocol for CompilerShim {}

unsafe impl MTL4Compiler for CompilerShim {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        default_device_handle()
    }
    fn host_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptorFields,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let function = descriptor
            .compute_function_descriptor
            .as_ref()
            .ok_or_else(|| MetalError::new("compute pipeline descriptor has no function descriptor"))?;
        // A specialized descriptor records its inner function descriptor and
        // constant values into its own base at set time; read them back here.
        let (function_fields, constant_values) = match &function.specialized {
            Some(inner) => (inner.as_ref(), function.specialized_constants.as_ref()),
            None => (function, None),
        };
        let library = function_fields
            .library
            .as_ref()
            .ok_or_else(|| MetalError::new("function descriptor has no library"))?;
        let name = function_fields
            .name
            .as_deref()
            .ok_or_else(|| MetalError::new("function descriptor has no name"))?;
        let constants = match constant_values {
            Some(values) => values.host_constants(),
            None => Vec::new(),
        };
        // The gate: ask the policy how this kernel is routed before creating a
        // pipeline. See host/DESIGN.md "Untranslated and unverified kernels".
        if let Some(error) = pipeline_gate_error(self.dev.route_kernel(library.host_library(), name), name) {
            return Err(error);
        }
        let pipeline = self.dev.create_pipeline(library.host_library(), name, &constants)?;
        Ok({
                                        let arc: Arc<ProtocolObject<dyn MTLComputePipelineState>> =
                    Arc::new(ProtocolObject::from_inner(PipelineShim {
                pipeline,
            }));
                let obj: Retained<ProtocolObject<dyn MTLComputePipelineState>> = Retained::from_arc(arc);
            obj
        })
    }
}

// ---------------------------------------------------------------------------
// Counter heaps (timestamps)
// ---------------------------------------------------------------------------

pub struct CounterHeapShim {
    pub(crate) pool: Arc<Mutex<metal2vk::TimestampPool>>,
    pub(crate) count: usize,
    pub(crate) r#type: MTL4CounterHeapType,
}

unsafe impl NSObjectProtocol for CounterHeapShim {}

unsafe impl MTL4CounterHeap for CounterHeapShim {
    fn count(&self) -> usize {
        self.count
    }
    fn r#type(&self) -> MTL4CounterHeapType {
        self.r#type
    }
    fn pool_handle(&self) -> &Arc<Mutex<metal2vk::TimestampPool>> {
        &self.pool
    }
}

// ---------------------------------------------------------------------------
// Function constants
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct FunctionConstant {
    pub(crate) index: usize,
    pub(crate) ty: MTLDataType,
    pub(crate) bytes: Vec<u8>,
}

pub(crate) fn constant_value_bytes(value: NonNull<c_void>, ty: MTLDataType) -> Vec<u8> {
    let size = match ty {
        MTLDataType::Bool => 1,
        MTLDataType::Float | MTLDataType::Int | MTLDataType::UInt => 4,
        other => panic!("unsupported function constant type {other:?}"),
    };
    // SAFETY: uzu passes `NonNull::from(&value)`, so `size` bytes are readable.
    unsafe { std::slice::from_raw_parts(value.as_ptr().cast::<u8>(), size).to_vec() }
}

impl MTLFunctionConstantValues {
    /// Convert the recorded constants into host pipeline specializations, in
    /// constant-index order.
    pub(crate) fn host_constants(&self) -> Vec<(u32, metal2vk::ConstantValue)> {
        let mut constants = self.constants.lock().clone();
        constants.sort_by_key(|c| c.index);
        constants
            .into_iter()
            .map(|c| {
                let value = match c.ty {
                    MTLDataType::Bool => metal2vk::ConstantValue::Bool(c.bytes[0] != 0),
                    MTLDataType::Float => {
                        metal2vk::ConstantValue::F32(f32::from_le_bytes(c.bytes.as_slice().try_into().unwrap()))
                    },
                    MTLDataType::Int => {
                        metal2vk::ConstantValue::I32(i32::from_le_bytes(c.bytes.as_slice().try_into().unwrap()))
                    },
                    MTLDataType::UInt => {
                        metal2vk::ConstantValue::U32(u32::from_le_bytes(c.bytes.as_slice().try_into().unwrap()))
                    },
                    other => panic!("unsupported function constant type {other:?}"),
                };
                (c.index as u32, value)
            })
            .collect()
    }
}

pub(crate) fn size3(size: MTLSize) -> [u32; 3] {
    [size.width as u32, size.height as u32, size.depth as u32]
}

/// Map a [`Route`] (or a routing error) to the `MetalError` the compat layer
/// reports for a compute pipeline, or `None` when the translated kernel may be
/// created. Pure: no device, no GPU.
pub fn pipeline_gate_error(route: Result<Route, metal2vk::Error>, name: &str) -> Option<MetalError> {
    match route {
        // Trusted: create the translated pipeline.
        Ok(Route::Translated) => None,
        // The compat layer has no reference runner; an unchecked kernel is never
        // silently run. The host side (or the C ABI caller) runs the check.
        Ok(Route::TranslatedChecked { reference }) => Some(MetalError::new(format!(
            "kernel '{name}' needs a first-use check against reference {reference:?}; \
             the compat layer cannot run reference kernels, run the check host-side"
        ))),
        // uzu sees MetalError::CannotCreatePipelineState { function_name, error } via
        // CompilerPipelineExtensions, then the kernel `new` and `MetalKernels::new`
        // fail, so `select_backend` reports metal-backend startup failure; uzu's own
        // kernel-selection/CPU path is `UZU_BACKEND=cpu` (or a build without metal).
        Ok(Route::Fallback(fb)) => Some(MetalError::new(format!(
            "kernel '{name}' is routed to fallback {fb:?}; no translated pipeline is \
             created so the kernel-selection/CPU-backend path can take over"
        ))),
        // The policy refused: surface its Display text verbatim ("refused: ...").
        Err(error) => Some(MetalError::new(error.to_string())),
    }
}

#[cfg(test)]
mod gate_tests {
    use super::pipeline_gate_error;
    use metal2vk::{Error, Fallback, Route};

    #[test]
    fn translated_route_creates_the_pipeline() {
        assert!(pipeline_gate_error(Ok(Route::Translated), "add").is_none());
    }

    #[test]
    fn checked_route_names_kernel_and_reference() {
        let error = pipeline_gate_error(
            Ok(Route::TranslatedChecked { reference: Fallback::Cpu("add_cpu".into()) }),
            "add",
        )
        .expect("checked route must fail in the compat layer");
        let text = error.to_string();
        assert!(text.contains("'add'"), "names the kernel: {text}");
        assert!(text.contains("add_cpu"), "names the reference: {text}");
        assert!(text.contains("first-use check"), "says why: {text}");
    }

    #[test]
    fn hand_fallback_names_kernel_and_route() {
        let error =
            pipeline_gate_error(Ok(Route::Fallback(Fallback::Hand("add_hand".into()))), "add")
                .expect("hand fallback must fail in the compat layer");
        let text = error.to_string();
        assert!(text.contains("'add'"), "names the kernel: {text}");
        assert!(text.contains("add_hand"), "names the hand kernel: {text}");
        assert!(text.contains("fallback"), "says fallback: {text}");
    }

    #[test]
    fn cpu_fallback_names_kernel_and_route() {
        let error = pipeline_gate_error(
            Ok(Route::Fallback(Fallback::Cpu("add_cpu".into()))),
            "add_offset",
        )
        .expect("cpu fallback must fail in the compat layer");
        let text = error.to_string();
        assert!(text.contains("'add_offset'"), "names the kernel: {text}");
        assert!(text.contains("add_cpu"), "names the cpu reference: {text}");
    }

    #[test]
    fn refused_error_text_is_verbatim() {
        let refused = Err(Error::Refused("kernel 'add' has no fallback".into()));
        let error = pipeline_gate_error(refused, "add").expect("refused must fail");
        assert_eq!(error.to_string(), "refused: kernel 'add' has no fallback");
    }
}
