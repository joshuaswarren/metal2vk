//! Metal protocol traits (signatures from mtl-rs 0.3.0, restricted to uzu's call
//! surface) and their impls on `ProtocolObject<dyn Trait>`, which delegate into
//! the wrapped host-backed shims.
//!
//! `shim_*`/`host_*` methods are `#[doc(hidden)]` dispatch paths: an inherent
//! shim method is not reachable through `dyn Trait`, so the trait carries it and
//! the `ProtocolObject` impl forwards. `MTLDevice::system_default` is the free
//! [`system_default_device`]: associated functions cannot dispatch through a
//! trait object, so uzu's single call site uses the free fn (uzu-linux.patch).

use std::ffi::c_void;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::descriptors::{
    ComputePipelineDescriptorFields, MTL4ArgumentTableDescriptor, MTL4CommitOptions, MTL4CompilerDescriptor,
    MTL4CompilerTaskOptions, MTL4ComputePipelineDescriptor, MTL4CounterHeapDescriptor,
    MTL4UpdateSparseBufferMappingOperation, MTLHeapDescriptor, MTLResidencySetDescriptor,
    MTLSharedEventNotificationBlock, MTLSharedEventListener,
};
use crate::error::MetalError;
use crate::shim;
use crate::types::*;
use objc2::{NSObjectProtocol, ProtocolObject, Retained};

/// The default device: first Vulkan compute device (`M2V_DEVICE=<index>`).
/// One logical device per process; every `device()` accessor returns it.
pub fn system_default_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    shim::system_default_device()
}

/// Marker protocol for GPU-allocated resources (`MTLAllocation`).
pub unsafe trait MTLAllocation: NSObjectProtocol + Send + Sync {}

unsafe impl<T: MTLAllocation + ?Sized> MTLAllocation for ProtocolObject<T> {}

/// `MTLDevice` protocol plus the hidden dispatch surface the `MTLDeviceExt`
/// factory methods route through.
pub unsafe trait MTLDevice: NSObjectProtocol + Send + Sync {
    fn supports_family(&self, family: MTLGPUFamily) -> bool;
    fn sample_timestamps_gpu_timestamp(&self, cpu_timestamp: &mut u64, gpu_timestamp: &mut u64);
    fn current_allocated_size(&self) -> usize;

    #[doc(hidden)]
    fn host_name(&self) -> String;
    #[doc(hidden)]
    fn host_gpu_core_count(&self) -> u32;
    #[doc(hidden)]
    fn host_new_buffer(&self, length: usize, options: MTLResourceOptions)
        -> Option<Retained<ProtocolObject<dyn MTLBuffer>>>;
    #[doc(hidden)]
    fn host_new_library_with_data(&self, data: &[u8]) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError>;
    #[doc(hidden)]
    fn host_new_event(&self) -> Option<Retained<ProtocolObject<dyn MTLEvent>>>;
    #[doc(hidden)]
    fn host_new_shared_event(&self) -> Option<Retained<ProtocolObject<dyn MTLSharedEvent>>>;
    #[doc(hidden)]
    fn host_new_queue(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandQueue>>>;
    #[doc(hidden)]
    fn host_new_command_buffer(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>;
    #[doc(hidden)]
    fn host_new_compiler(&self) -> Result<Retained<ProtocolObject<dyn MTL4Compiler>>, MetalError>;
    #[doc(hidden)]
    fn host_new_counter_heap(
        &self,
        count: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTL4CounterHeap>>, MetalError>;
}

unsafe impl<P: MTLDevice + ?Sized> MTLDevice for ProtocolObject<P>{
    fn supports_family(&self, family: MTLGPUFamily) -> bool {
        self.as_inner().supports_family(family)
    }
    fn sample_timestamps_gpu_timestamp(&self, cpu_timestamp: &mut u64, gpu_timestamp: &mut u64) {
        self.as_inner().sample_timestamps_gpu_timestamp(cpu_timestamp, gpu_timestamp)
    }
    fn current_allocated_size(&self) -> usize {
        self.as_inner().current_allocated_size()
    }
    fn host_name(&self) -> String {
        self.as_inner().host_name()
    }
    fn host_gpu_core_count(&self) -> u32 {
        self.as_inner().host_gpu_core_count()
    }
    fn host_new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.as_inner().host_new_buffer(length, options)
    }
    fn host_new_library_with_data(&self, data: &[u8]) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        self.as_inner().host_new_library_with_data(data)
    }
    fn host_new_event(&self) -> Option<Retained<ProtocolObject<dyn MTLEvent>>> {
        self.as_inner().host_new_event()
    }
    fn host_new_shared_event(&self) -> Option<Retained<ProtocolObject<dyn MTLSharedEvent>>> {
        self.as_inner().host_new_shared_event()
    }
    fn host_new_queue(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandQueue>>> {
        self.as_inner().host_new_queue()
    }
    fn host_new_command_buffer(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandBuffer>>> {
        self.as_inner().host_new_command_buffer()
    }
    fn host_new_compiler(&self) -> Result<Retained<ProtocolObject<dyn MTL4Compiler>>, MetalError> {
        self.as_inner().host_new_compiler()
    }
    fn host_new_counter_heap(
        &self,
        count: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTL4CounterHeap>>, MetalError> {
        self.as_inner().host_new_counter_heap(count)
    }
}

/// Device factory methods (mtl-rs `MTLDeviceExt`) plus the private-selector
/// queries uzu's `DeviceExt` wraps. Host answers: Apple7-family (M1-class), no
/// MXU, no placement sparse (uzu takes its dense path).
pub trait MTLDeviceExt {
    fn name(&self) -> String;
    fn gpu_core_count(&self) -> u32;
    fn supports_mxu(&self) -> bool;
    fn supports_placement_sparse_resources(&self) -> bool;
    fn newest_supported_apple_gpu_family(&self) -> MTLGPUFamily;
    fn new_buffer(&self, length: usize, options: MTLResourceOptions) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>>;
    fn new_buffer_with_length_options_placement_sparse_page_size(
        &self,
        length: usize,
        options: MTLResourceOptions,
        placement_sparse_page_size: MTLSparsePageSize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>>;
    fn new_library_with_data(&self, data: &[u8]) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError>;
    fn new_event(&self) -> Option<Retained<ProtocolObject<dyn MTLEvent>>>;
    fn new_shared_event(&self) -> Option<Retained<ProtocolObject<dyn MTLSharedEvent>>>;
    fn new_heap_with_descriptor(&self, descriptor: &MTLHeapDescriptor)
        -> Option<Retained<ProtocolObject<dyn MTLHeap>>>;
    fn new_residency_set_with_descriptor(
        &self,
        descriptor: &MTLResidencySetDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTLResidencySet>>, MetalError>;
    fn new_command_allocator(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandAllocator>>>;
    fn new_mtl4_command_queue(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandQueue>>>;
    fn new_mtl4_command_buffer(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>;
    fn new_argument_table_with_descriptor(
        &self,
        descriptor: &MTL4ArgumentTableDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTL4ArgumentTable>>, MetalError>;
    fn new_compiler_with_descriptor(
        &self,
        descriptor: &MTL4CompilerDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTL4Compiler>>, MetalError>;
    fn new_counter_heap_with_descriptor(
        &self,
        descriptor: &MTL4CounterHeapDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTL4CounterHeap>>, MetalError>;
    fn query_timestamp_frequency(&self) -> u64;
}

impl MTLDeviceExt for ProtocolObject<dyn MTLDevice> {
    fn name(&self) -> String {
        self.host_name()
    }

    fn gpu_core_count(&self) -> u32 {
        self.host_gpu_core_count()
    }

    fn supports_mxu(&self) -> bool {
        // MXU is an M5-generation feature; every host part reports false.
        false
    }

    fn supports_placement_sparse_resources(&self) -> bool {
        // No placement-sparse buffers on the host: DeviceCapabilities::SPARSE_BUFFERS
        // stays unset and uzu uses its dense path.
        false
    }

    fn newest_supported_apple_gpu_family(&self) -> MTLGPUFamily {
        // True for the M1-class hardware this host targets: family Apple7.
        MTLGPUFamily::Apple7
    }

    fn new_buffer(&self, length: usize, options: MTLResourceOptions) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.host_new_buffer(length, options)
    }

    fn new_buffer_with_length_options_placement_sparse_page_size(
        &self,
        _length: usize,
        _options: MTLResourceOptions,
        _placement_sparse_page_size: MTLSparsePageSize,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        // No sparse placement on the host; unreachable while
        // supports_placement_sparse_resources() is false (uzu errors CannotCreateBuffer).
        None
    }

    fn new_library_with_data(&self, data: &[u8]) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, MetalError> {
        self.host_new_library_with_data(data)
    }

    fn new_event(&self) -> Option<Retained<ProtocolObject<dyn MTLEvent>>> {
        self.host_new_event()
    }

    fn new_shared_event(&self) -> Option<Retained<ProtocolObject<dyn MTLSharedEvent>>> {
        // Same semaphore shape; the shared-event surface serves the sparse path's
        // teardown, which never runs without sparse support.
        self.host_new_shared_event()
    }

    fn new_heap_with_descriptor(
        &self,
        _descriptor: &MTLHeapDescriptor,
    ) -> Option<Retained<ProtocolObject<dyn MTLHeap>>> {
        // Heaps back sparse page placement only; no Vulkan equivalent. Unreachable
        // while sparse is off (uzu maps None to CannotCreateHeap).
        None
    }

    fn new_residency_set_with_descriptor(
        &self,
        _descriptor: &MTLResidencySetDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTLResidencySet>>, MetalError> {
        Ok({
                                        let arc: Arc<ProtocolObject<dyn MTLResidencySet>> =
                    Arc::new(ProtocolObject::from_inner(shim::ResidencySetShim));
                let obj: Retained<ProtocolObject<dyn MTLResidencySet>> = Retained::from_arc(arc);
            obj
        })
    }

    fn new_command_allocator(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandAllocator>>> {
        // Host command buffers are pooled per queue; the allocator is bookkeeping.
        Some({
                                        let arc: Arc<ProtocolObject<dyn MTL4CommandAllocator>> =
                    Arc::new(ProtocolObject::from_inner(shim::AllocatorShim));
                let obj: Retained<ProtocolObject<dyn MTL4CommandAllocator>> = Retained::from_arc(arc);
            obj
        })
    }

    fn new_mtl4_command_queue(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandQueue>>> {
        self.host_new_queue()
    }

    fn new_mtl4_command_buffer(&self) -> Option<Retained<ProtocolObject<dyn MTL4CommandBuffer>>> {
        self.host_new_command_buffer()
    }

    fn new_argument_table_with_descriptor(
        &self,
        _descriptor: &MTL4ArgumentTableDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTL4ArgumentTable>>, MetalError> {
        Ok({
                                        let arc: Arc<ProtocolObject<dyn MTL4ArgumentTable>> =
                    Arc::new(ProtocolObject::from_inner(shim::ArgumentTableShim::new()));
                let obj: Retained<ProtocolObject<dyn MTL4ArgumentTable>> = Retained::from_arc(arc);
            obj
        })
    }

    fn new_compiler_with_descriptor(
        &self,
        _descriptor: &MTL4CompilerDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTL4Compiler>>, MetalError> {
        self.host_new_compiler()
    }

    fn new_counter_heap_with_descriptor(
        &self,
        descriptor: &MTL4CounterHeapDescriptor,
    ) -> Result<Retained<ProtocolObject<dyn MTL4CounterHeap>>, MetalError> {
        self.host_new_counter_heap(descriptor.count())
    }

    fn query_timestamp_frequency(&self) -> u64 {
        // Host timestamps are already nanoseconds.
        1_000_000_000
    }
}

// ---------------------------------------------------------------------------
// Buffer / heap / residency / library / pipeline
// ---------------------------------------------------------------------------

/// `MTLBuffer` protocol.
pub unsafe trait MTLBuffer: MTLAllocation {
    fn length(&self) -> usize;
    fn contents(&self) -> NonNull<c_void>;
    fn gpu_address(&self) -> u64;

    /// Shim-internal: the host buffer this wraps.
    #[doc(hidden)]
    fn host_buffer(&self) -> &Arc<metal2vk::Buffer>;
}

unsafe impl<P: MTLBuffer + ?Sized> MTLBuffer for ProtocolObject<P>{
    fn length(&self) -> usize {
        self.as_inner().length()
    }
    fn contents(&self) -> NonNull<c_void> {
        self.as_inner().contents()
    }
    fn gpu_address(&self) -> u64 {
        self.as_inner().gpu_address()
    }
    fn host_buffer(&self) -> &Arc<metal2vk::Buffer> {
        self.as_inner().host_buffer()
    }
}

/// `MTLHeap` protocol. Heaps exist for sparse page placement on Metal; the host
/// has no equivalent (unreachable while sparse is off).
pub unsafe trait MTLHeap: MTLAllocation {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;
    fn size(&self) -> usize;
    fn current_allocated_size(&self) -> usize;
}

unsafe impl<P: MTLHeap + ?Sized> MTLHeap for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn size(&self) -> usize {
        0
    }
    fn current_allocated_size(&self) -> usize {
        0
    }
}

/// `MTLResidencySet`: unified-memory bookkeeping only. Every host buffer is
/// host-visible and coherent for its whole life, so "resident" is always true;
/// commit/request are recorded no-ops (see host/STATUS.md).
pub unsafe trait MTLResidencySet: NSObjectProtocol + Send + Sync {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;
    fn add_allocation(&self, allocation: &ProtocolObject<dyn MTLAllocation>);
    fn remove_allocation(&self, allocation: &ProtocolObject<dyn MTLAllocation>);
    fn commit(&self);
    fn request_residency(&self);
}

unsafe impl<P: MTLResidencySet + ?Sized> MTLResidencySet for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn add_allocation(&self, _allocation: &ProtocolObject<dyn MTLAllocation>) {}
    fn remove_allocation(&self, _allocation: &ProtocolObject<dyn MTLAllocation>) {}
    fn commit(&self) {}
    fn request_residency(&self) {}
}

/// `MTLLibrary` protocol: a loaded `.m2vlib`.
pub unsafe trait MTLLibrary: NSObjectProtocol + Send + Sync {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;

    /// Shim-internal: the host library.
    #[doc(hidden)]
    fn host_library(&self) -> &Arc<metal2vk::Library>;
}

unsafe impl<P: MTLLibrary + ?Sized> MTLLibrary for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn host_library(&self) -> &Arc<metal2vk::Library> {
        self.as_inner().host_library()
    }
}

/// `MTLComputePipelineState` protocol.
pub unsafe trait MTLComputePipelineState: MTLAllocation {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;

    /// Shim-internal: the host pipeline. `None` for a refused kernel: the object
    /// exists only to keep uzu's launch maths valid and to refuse at use.
    #[doc(hidden)]
    fn host_pipeline(&self) -> Option<&Arc<metal2vk::Pipeline>>;

    /// For most efficient execution, the threadgroup size should be a multiple
    /// of this (Metal `threadExecutionWidth`; host: device subgroup size).
    fn thread_execution_width(&self) -> usize;
    /// The maximum total number of threads that can be in a single threadgroup.
    fn max_total_threads_per_threadgroup(&self) -> usize;
    /// Threadgroup memory statically allocated by the pipeline, in bytes.
    fn static_threadgroup_memory_length(&self) -> usize;

    /// Shim-internal: kernel this pipeline was built for (trace + refusals).
    #[doc(hidden)]
    fn kernel_name(&self) -> &str;
    /// Shim-internal: refusal reason when there is no host pipeline.
    #[doc(hidden)]
    fn pipeline_reason(&self) -> &str;
}

unsafe impl<P: MTLComputePipelineState + ?Sized> MTLComputePipelineState for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn host_pipeline(&self) -> Option<&Arc<metal2vk::Pipeline>> {
        self.as_inner().host_pipeline()
    }
    fn thread_execution_width(&self) -> usize {
        self.as_inner().thread_execution_width()
    }
    fn max_total_threads_per_threadgroup(&self) -> usize {
        self.as_inner().max_total_threads_per_threadgroup()
    }
    fn static_threadgroup_memory_length(&self) -> usize {
        self.as_inner().static_threadgroup_memory_length()
    }
    fn kernel_name(&self) -> &str {
        self.as_inner().kernel_name()
    }
    fn pipeline_reason(&self) -> &str {
        self.as_inner().pipeline_reason()
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// `MTLEvent` protocol: a Vulkan timeline semaphore.
pub unsafe trait MTLEvent: NSObjectProtocol + Send + Sync {
    fn device(&self) -> Option<Retained<ProtocolObject<dyn MTLDevice>>>;

    /// Shim-internal: the host timeline semaphore.
    #[doc(hidden)]
    fn host_event(&self) -> &Arc<metal2vk::Event>;
}

unsafe impl<P: MTLEvent + ?Sized> MTLEvent for ProtocolObject<P>{
    fn device(&self) -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
        Some(shim::default_device_handle())
    }
    fn host_event(&self) -> &Arc<metal2vk::Event> {
        self.as_inner().host_event()
    }
}

/// `MTLSharedEvent` protocol (same semaphore; only the sparse path uses this).
pub unsafe trait MTLSharedEvent: MTLEvent {
    fn signaled_value(&self) -> u64;
    fn set_signaled_value(&self, signaled_value: u64);
    fn wait_until_signaled_value_timeout_ms(&self, value: u64, milliseconds: u64) -> bool;
}

unsafe impl<P: MTLSharedEvent + ?Sized> MTLSharedEvent for ProtocolObject<P>{
    fn signaled_value(&self) -> u64 {
        self.as_inner().signaled_value()
    }
    fn set_signaled_value(&self, signaled_value: u64) {
        self.as_inner().set_signaled_value(signaled_value)
    }
    fn wait_until_signaled_value_timeout_ms(&self, value: u64, milliseconds: u64) -> bool {
        self.as_inner().wait_until_signaled_value_timeout_ms(value, milliseconds)
    }
}

/// `MTLSharedEventExt`: listener notification. Timeline semaphores here have no
/// dispatch-source listeners; the block is not retained past the call. Only the
/// sparse teardown (which never runs) reaches this.
pub trait MTLSharedEventExt {
    fn notify_listener_at_value(
        &self,
        listener: &MTLSharedEventListener,
        value: u64,
        block: &MTLSharedEventNotificationBlock,
    );
}

impl MTLSharedEventExt for ProtocolObject<dyn MTLSharedEvent> {
    fn notify_listener_at_value(
        &self,
        _listener: &MTLSharedEventListener,
        _value: u64,
        _block: &MTLSharedEventNotificationBlock,
    ) {
    }
}

// ---------------------------------------------------------------------------
// Metal 4: queues, command buffers, allocators
// ---------------------------------------------------------------------------

/// `MTL4CommandQueue` protocol.
pub unsafe trait MTL4CommandQueue: NSObjectProtocol + Send + Sync {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;
    fn signal_event_value(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64);
    fn wait_for_event_value(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64);
    fn add_residency_set(&self, residency_set: &ProtocolObject<dyn MTLResidencySet>);
    fn remove_residency_set(&self, residency_set: &ProtocolObject<dyn MTLResidencySet>);

    /// Shim-internal: the host queue behind this handle.
    #[doc(hidden)]
    fn shim_queue(&self) -> &Arc<metal2vk::Queue>;
}

unsafe impl<P: MTL4CommandQueue + ?Sized> MTL4CommandQueue for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn signal_event_value(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64) {
        self.as_inner().signal_event_value(event, value)
    }
    fn wait_for_event_value(&self, event: &ProtocolObject<dyn MTLEvent>, value: u64) {
        self.as_inner().wait_for_event_value(event, value)
    }
    fn add_residency_set(&self, residency_set: &ProtocolObject<dyn MTLResidencySet>) {
        self.as_inner().add_residency_set(residency_set)
    }
    fn remove_residency_set(&self, residency_set: &ProtocolObject<dyn MTLResidencySet>) {
        self.as_inner().remove_residency_set(residency_set)
    }
    fn shim_queue(&self) -> &Arc<metal2vk::Queue> {
        self.as_inner().shim_queue()
    }
}

/// `MTL4CommandQueueExt`.
pub trait MTL4CommandQueueExt {
    fn commit(&self, command_buffers: &[&ProtocolObject<dyn MTL4CommandBuffer>]);
    fn commit_with_options(
        &self,
        command_buffers: &[&ProtocolObject<dyn MTL4CommandBuffer>],
        options: &MTL4CommitOptions,
    );
    fn update_buffer_mappings(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        heap: Option<&ProtocolObject<dyn MTLHeap>>,
        operations: &[MTL4UpdateSparseBufferMappingOperation],
    );
}

impl MTL4CommandQueueExt for ProtocolObject<dyn MTL4CommandQueue> {
    fn commit(&self, command_buffers: &[&ProtocolObject<dyn MTL4CommandBuffer>]) {
        let options = MTL4CommitOptions::new();
        self.commit_with_options(command_buffers, &options);
    }

    fn commit_with_options(
        &self,
        command_buffers: &[&ProtocolObject<dyn MTL4CommandBuffer>],
        options: &MTL4CommitOptions,
    ) {
        shim::queue_commit(self.shim_queue(), command_buffers, options);
    }

    fn update_buffer_mappings(
        &self,
        _buffer: &ProtocolObject<dyn MTLBuffer>,
        _heap: Option<&ProtocolObject<dyn MTLHeap>>,
        _operations: &[MTL4UpdateSparseBufferMappingOperation],
    ) {
        // Sparse mapping has no host path; unreachable while sparse is off.
    }
}

/// `MTL4CommandBuffer` protocol.
pub unsafe trait MTL4CommandBuffer: NSObjectProtocol + Send + Sync {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;
    fn begin_command_buffer_with_allocator(&self, allocator: &ProtocolObject<dyn MTL4CommandAllocator>);
    fn end_command_buffer(&self);
    fn compute_command_encoder(&self) -> Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>;

    /// Shim-internal: the shared slot holding the host command buffer between
    /// encoding end and submit.
    #[doc(hidden)]
    fn host_slot(&self) -> &Arc<Mutex<Option<metal2vk::CommandBuffer>>>;
    /// Shim-internal: labels live on the shim (Ext dispatch path).
    #[doc(hidden)]
    fn shim_label(&self) -> Option<String>;
    #[doc(hidden)]
    fn shim_set_label(&self, label: Option<&str>);
    /// Shim-internal: kernels refused in this command buffer (name, reason).
    #[doc(hidden)]
    fn host_refused(&self) -> &Arc<Mutex<crate::shim::Refusals>>;
}

unsafe impl<P: MTL4CommandBuffer + ?Sized> MTL4CommandBuffer for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn begin_command_buffer_with_allocator(&self, _allocator: &ProtocolObject<dyn MTL4CommandAllocator>) {}
    fn end_command_buffer(&self) {
        // Recording is closed when the encoder ends; nothing to do here.
    }
    fn compute_command_encoder(&self) -> Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>> {
        Some(shim::encoder_for_slot(self.as_inner().host_slot(), Arc::clone(self.as_inner().host_refused())))
    }
    fn host_slot(&self) -> &Arc<Mutex<Option<metal2vk::CommandBuffer>>> {
        self.as_inner().host_slot()
    }
    fn shim_label(&self) -> Option<String> {
        self.as_inner().shim_label()
    }
    fn shim_set_label(&self, label: Option<&str>) {
        self.as_inner().shim_set_label(label)
    }
    fn host_refused(&self) -> &Arc<Mutex<crate::shim::Refusals>> {
        self.as_inner().host_refused()
    }
}

/// `MTL4CommandBufferExt`.
pub trait MTL4CommandBufferExt {
    fn label(&self) -> Option<String>;
    fn set_label(&self, label: Option<&str>);
    fn push_debug_group(&self, label: &str);
}

impl MTL4CommandBufferExt for ProtocolObject<dyn MTL4CommandBuffer> {
    fn label(&self) -> Option<String> {
        self.shim_label()
    }
    fn set_label(&self, label: Option<&str>) {
        self.shim_set_label(label)
    }
    fn push_debug_group(&self, _label: &str) {}
}

/// `MTL4CommandAllocator` protocol. Host command buffers are pooled per queue, so
/// reset is bookkeeping only.
pub unsafe trait MTL4CommandAllocator: NSObjectProtocol + Send + Sync {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;
    fn allocated_size(&self) -> u64;
    fn reset(&self);
}

unsafe impl<P: MTL4CommandAllocator + ?Sized> MTL4CommandAllocator for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn allocated_size(&self) -> u64 {
        self.as_inner().allocated_size()
    }
    fn reset(&self) {
        self.as_inner().reset()
    }
}

// ---------------------------------------------------------------------------
// Metal 4: encoders
// ---------------------------------------------------------------------------

/// `MTL4CommandEncoder` protocol. Every barrier form maps to one full
/// compute/transfer memory barrier (`metal2vk::Encoder::barrier`).
pub unsafe trait MTL4CommandEncoder: NSObjectProtocol {
    fn barrier_after_queue_stages_before_stages_visibility_options(
        &self,
        after_queue_stages: MTLStages,
        before_stages: MTLStages,
        visibility_options: MTL4VisibilityOptions,
    );
    fn barrier_after_stages_before_queue_stages_visibility_options(
        &self,
        after_stages: MTLStages,
        before_queue_stages: MTLStages,
        visibility_options: MTL4VisibilityOptions,
    );
    fn barrier_after_encoder_stages_before_encoder_stages_visibility_options(
        &self,
        after_encoder_stages: MTLStages,
        before_encoder_stages: MTLStages,
        visibility_options: MTL4VisibilityOptions,
    );
    fn pop_debug_group(&self);
    fn end_encoding(&self);

    /// Shim-internal dispatch path for the Ext `push_debug_group`.
    #[doc(hidden)]
    fn shim_push_debug_group(&self, label: &str);
}

unsafe impl<P: MTL4CommandEncoder + ?Sized> MTL4CommandEncoder for ProtocolObject<P>{
    fn barrier_after_queue_stages_before_stages_visibility_options(
        &self,
        after_queue_stages: MTLStages,
        before_stages: MTLStages,
        visibility_options: MTL4VisibilityOptions,
    ) {
        self.as_inner().barrier_after_queue_stages_before_stages_visibility_options(
            after_queue_stages,
            before_stages,
            visibility_options,
        );
    }
    fn barrier_after_stages_before_queue_stages_visibility_options(
        &self,
        after_stages: MTLStages,
        before_queue_stages: MTLStages,
        visibility_options: MTL4VisibilityOptions,
    ) {
        self.as_inner().barrier_after_stages_before_queue_stages_visibility_options(
            after_stages,
            before_queue_stages,
            visibility_options,
        );
    }
    fn barrier_after_encoder_stages_before_encoder_stages_visibility_options(
        &self,
        after_encoder_stages: MTLStages,
        before_encoder_stages: MTLStages,
        visibility_options: MTL4VisibilityOptions,
    ) {
        self.as_inner().barrier_after_encoder_stages_before_encoder_stages_visibility_options(
            after_encoder_stages,
            before_encoder_stages,
            visibility_options,
        );
    }
    fn pop_debug_group(&self) {
        self.as_inner().pop_debug_group()
    }
    fn end_encoding(&self) {
        self.as_inner().end_encoding()
    }
    fn shim_push_debug_group(&self, label: &str) {
        self.as_inner().shim_push_debug_group(label)
    }
}

/// `MTL4CommandEncoderExt`.
pub trait MTL4CommandEncoderExt {
    fn label(&self) -> Option<String>;
    fn set_label(&self, label: Option<&str>);
    fn push_debug_group(&self, label: &str);
}

impl MTL4CommandEncoderExt for ProtocolObject<dyn MTL4CommandEncoder> {
    fn label(&self) -> Option<String> {
        None
    }
    fn set_label(&self, _label: Option<&str>) {}
    fn push_debug_group(&self, label: &str) {
        self.shim_push_debug_group(label)
    }
}

/// `MTL4ComputeCommandEncoder` protocol.
pub unsafe trait MTL4ComputeCommandEncoder: MTL4CommandEncoder {
    fn set_compute_pipeline_state(&self, state: &ProtocolObject<dyn MTLComputePipelineState>);
    fn set_threadgroup_memory_length_at_index(&self, length: usize, index: usize);
    fn dispatch_threads_threads_per_threadgroup(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize);
    fn dispatch_threadgroups_threads_per_threadgroup(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    );
    fn dispatch_threadgroups_with_indirect_buffer_threads_per_threadgroup(
        &self,
        indirect_buffer: MTLGPUAddress,
        threads_per_threadgroup: MTLSize,
    );
    fn copy_from_buffer_source_offset_to_buffer_destination_offset_size(
        &self,
        source_buffer: &ProtocolObject<dyn MTLBuffer>,
        source_offset: usize,
        destination_buffer: &ProtocolObject<dyn MTLBuffer>,
        destination_offset: usize,
        size: usize,
    );
    fn set_argument_table(&self, argument_table: Option<&ProtocolObject<dyn MTL4ArgumentTable>>);

    /// Shim-internal dispatch paths for the Ext methods.
    #[doc(hidden)]
    fn shim_fill_buffer_range_value(&self, buffer: &ProtocolObject<dyn MTLBuffer>, range: Range<usize>, value: u8);
    #[doc(hidden)]
    fn shim_write_timestamp_with_granularity_into_heap_at_index(
        &self,
        granularity: MTL4TimestampGranularity,
        counter_heap: &ProtocolObject<dyn MTL4CounterHeap>,
        index: usize,
    );
}

unsafe impl<P: MTL4ComputeCommandEncoder + ?Sized> MTL4ComputeCommandEncoder for ProtocolObject<P>{
    fn set_compute_pipeline_state(&self, state: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.as_inner().set_compute_pipeline_state(state)
    }
    fn set_threadgroup_memory_length_at_index(&self, length: usize, index: usize) {
        self.as_inner().set_threadgroup_memory_length_at_index(length, index)
    }
    fn dispatch_threads_threads_per_threadgroup(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.as_inner().dispatch_threads_threads_per_threadgroup(threads_per_grid, threads_per_threadgroup)
    }
    fn dispatch_threadgroups_threads_per_threadgroup(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.as_inner().dispatch_threadgroups_threads_per_threadgroup(threadgroups_per_grid, threads_per_threadgroup)
    }
    fn dispatch_threadgroups_with_indirect_buffer_threads_per_threadgroup(
        &self,
        indirect_buffer: MTLGPUAddress,
        threads_per_threadgroup: MTLSize,
    ) {
        self.as_inner()
            .dispatch_threadgroups_with_indirect_buffer_threads_per_threadgroup(indirect_buffer, threads_per_threadgroup);
    }
    fn copy_from_buffer_source_offset_to_buffer_destination_offset_size(
        &self,
        source_buffer: &ProtocolObject<dyn MTLBuffer>,
        source_offset: usize,
        destination_buffer: &ProtocolObject<dyn MTLBuffer>,
        destination_offset: usize,
        size: usize,
    ) {
        self.as_inner().copy_from_buffer_source_offset_to_buffer_destination_offset_size(
            source_buffer,
            source_offset,
            destination_buffer,
            destination_offset,
            size,
        )
    }
    fn set_argument_table(&self, argument_table: Option<&ProtocolObject<dyn MTL4ArgumentTable>>) {
        self.as_inner().set_argument_table(argument_table)
    }
    fn shim_fill_buffer_range_value(&self, buffer: &ProtocolObject<dyn MTLBuffer>, range: Range<usize>, value: u8) {
        self.as_inner().shim_fill_buffer_range_value(buffer, range, value)
    }
    fn shim_write_timestamp_with_granularity_into_heap_at_index(
        &self,
        granularity: MTL4TimestampGranularity,
        counter_heap: &ProtocolObject<dyn MTL4CounterHeap>,
        index: usize,
    ) {
        self.as_inner().shim_write_timestamp_with_granularity_into_heap_at_index(granularity, counter_heap, index)
    }
}

/// `MTL4ComputeCommandEncoderExt`.
pub trait MTL4ComputeCommandEncoderExt {
    fn fill_buffer_range_value(&self, buffer: &ProtocolObject<dyn MTLBuffer>, range: Range<usize>, value: u8);
    fn write_timestamp_with_granularity_into_heap_at_index(
        &self,
        granularity: MTL4TimestampGranularity,
        counter_heap: &ProtocolObject<dyn MTL4CounterHeap>,
        index: usize,
    );
}

impl MTL4ComputeCommandEncoderExt for ProtocolObject<dyn MTL4ComputeCommandEncoder> {
    fn fill_buffer_range_value(&self, buffer: &ProtocolObject<dyn MTLBuffer>, range: Range<usize>, value: u8) {
        self.shim_fill_buffer_range_value(buffer, range, value)
    }
    fn write_timestamp_with_granularity_into_heap_at_index(
        &self,
        granularity: MTL4TimestampGranularity,
        counter_heap: &ProtocolObject<dyn MTL4CounterHeap>,
        index: usize,
    ) {
        self.shim_write_timestamp_with_granularity_into_heap_at_index(granularity, counter_heap, index)
    }
}

// ---------------------------------------------------------------------------
// Metal 4: argument table, compiler, feedback, counter heaps
// ---------------------------------------------------------------------------

/// `MTL4ArgumentTable` protocol: slot -> GPU address, applied at dispatch in
/// commit order (Metal 4 argument-table semantics).
pub unsafe trait MTL4ArgumentTable: NSObjectProtocol + Send + Sync {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;
    fn set_address_at_index(&self, gpu_address: MTLGPUAddress, binding_index: usize);

    /// Shim-internal.
    #[doc(hidden)]
    fn host_state(&self) -> &Arc<Mutex<shim::TableState>>;
}

unsafe impl<P: MTL4ArgumentTable + ?Sized> MTL4ArgumentTable for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn set_address_at_index(&self, gpu_address: MTLGPUAddress, binding_index: usize) {
        self.as_inner().set_address_at_index(gpu_address, binding_index)
    }
    fn host_state(&self) -> &Arc<Mutex<shim::TableState>> {
        self.as_inner().host_state()
    }
}

/// `MTL4Compiler` protocol.
pub unsafe trait MTL4Compiler: NSObjectProtocol + Send + Sync {
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>>;

    /// Shim-internal: pipeline creation from a descriptor snapshot.
    #[doc(hidden)]
    fn host_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptorFields,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError>;
}

unsafe impl<P: MTL4Compiler + ?Sized> MTL4Compiler for ProtocolObject<P>{
    fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
        shim::default_device_handle()
    }
    fn host_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptorFields,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        self.as_inner().host_compute_pipeline_state(descriptor)
    }
}

/// `MTL4CompilerExt`: synchronous compute pipeline creation.
pub trait MTL4CompilerExt {
    fn new_compute_pipeline_state_with_descriptor_compiler_task_options_error(
        &self,
        descriptor: &MTL4ComputePipelineDescriptor,
        compiler_task_options: Option<&MTL4CompilerTaskOptions>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError>;
}

impl MTL4CompilerExt for ProtocolObject<dyn MTL4Compiler> {
    fn new_compute_pipeline_state_with_descriptor_compiler_task_options_error(
        &self,
        descriptor: &MTL4ComputePipelineDescriptor,
        _compiler_task_options: Option<&MTL4CompilerTaskOptions>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        self.host_compute_pipeline_state(&descriptor.f.lock())
    }
}

/// `MTL4CommitFeedback` protocol: GPU start/end times in seconds, plus the
/// commit error when the submission failed.
pub unsafe trait MTL4CommitFeedback: NSObjectProtocol {
    fn gpu_start_time(&self) -> f64;
    fn gpu_end_time(&self) -> f64;

    /// Shim-internal.
    #[doc(hidden)]
    fn completion_error(&self) -> Option<MetalError>;
}

unsafe impl<P: MTL4CommitFeedback + ?Sized> MTL4CommitFeedback for ProtocolObject<P>{
    fn gpu_start_time(&self) -> f64 {
        self.as_inner().gpu_start_time()
    }
    fn gpu_end_time(&self) -> f64 {
        self.as_inner().gpu_end_time()
    }
    fn completion_error(&self) -> Option<MetalError> {
        self.as_inner().completion_error()
    }
}

/// `MTL4CommitFeedbackExt`.
pub trait MTL4CommitFeedbackExt {
    fn error(&self) -> Option<MetalError>;
}

impl MTL4CommitFeedbackExt for ProtocolObject<dyn MTL4CommitFeedback> {
    fn error(&self) -> Option<MetalError> {
        self.completion_error()
    }
}

/// `MTL4CounterHeap` protocol.
pub unsafe trait MTL4CounterHeap: NSObjectProtocol + Send + Sync {
    fn count(&self) -> usize;
    fn r#type(&self) -> MTL4CounterHeapType;

    /// Shim-internal.
    #[doc(hidden)]
    fn pool_handle(&self) -> &Arc<Mutex<metal2vk::TimestampPool>>;
}

unsafe impl<P: MTL4CounterHeap + ?Sized> MTL4CounterHeap for ProtocolObject<P>{
    fn count(&self) -> usize {
        self.as_inner().count()
    }
    fn r#type(&self) -> MTL4CounterHeapType {
        self.as_inner().r#type()
    }
    fn pool_handle(&self) -> &Arc<Mutex<metal2vk::TimestampPool>> {
        self.as_inner().pool_handle()
    }
}

/// `MTL4CounterHeapExt`.
pub trait MTL4CounterHeapExt {
    fn resolve_counter_range(&self, range: Range<usize>) -> Option<Box<[u8]>>;
    fn invalidate_counter_range(&self, range: Range<usize>);
}

impl MTL4CounterHeapExt for ProtocolObject<dyn MTL4CounterHeap> {
    fn resolve_counter_range(&self, range: Range<usize>) -> Option<Box<[u8]>> {
        let results = self.pool_handle().lock().results().ok()?;
        let mut out = Vec::with_capacity(range.len() * size_of::<u64>());
        for slot in range {
            // Unwritten slots resolve to zero, which uzu treats as "not written".
            let ns = results.get(slot).copied().flatten().unwrap_or(0);
            out.extend_from_slice(&ns.to_ne_bytes());
        }
        Some(out.into())
    }

    fn invalidate_counter_range(&self, _range: Range<usize>) {}
}
