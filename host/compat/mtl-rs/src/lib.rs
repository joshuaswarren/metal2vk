//! Linux host shim for the `metal` (mtl-rs 0.3.0) API subset uzu's Metal backend
//! consumes, backed by the `m2v-host` core (see host/DESIGN.md).
//!
//! Protocol traits dispatch through `ProtocolObject<dyn Trait>`; the `shim`
//! module holds the host-backed objects. Features with no Vulkan equivalent are
//! honest: sparse placement reports unsupported, residency sets are unified-memory
//! bookkeeping, GPU capture errors, timestamps come from Vulkan query pools.

pub mod descriptors;
pub mod error;
pub mod protocols;
pub mod shim;
pub mod types;

pub use descriptors::{
    MTL4ArgumentTableDescriptor, MTL4CommitFeedbackHandler, MTL4CommitOptions, MTL4CompilerDescriptor,
    MTL4CompilerTaskOptions, MTL4ComputePipelineDescriptor, MTL4CounterHeapDescriptor,
    MTL4FunctionDescriptor, MTL4LibraryFunctionDescriptor, MTL4SpecializedFunctionDescriptor,
    MTL4UpdateSparseBufferMappingOperation, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager,
    MTLCaptureTarget, MTLFunctionConstantValues, MTLHeapDescriptor, MTLResidencySetDescriptor,
    MTLSharedEventNotificationBlock, MTLSharedEventListener,
};
pub use error::MetalError;
pub use metal2vk::{Fallback, Route};
pub use protocols::{
    system_default_device, MTL4ArgumentTable, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandBufferExt,
    MTL4CommandEncoder, MTL4CommandEncoderExt, MTL4CommandQueue, MTL4CommandQueueExt, MTL4CommitFeedback,
    MTL4CommitFeedbackExt, MTL4Compiler, MTL4CompilerExt, MTL4ComputeCommandEncoder, MTL4ComputeCommandEncoderExt,
    MTL4CounterHeap, MTL4CounterHeapExt, MTLAllocation, MTLBuffer, MTLComputePipelineState, MTLDevice, MTLDeviceExt,
    MTLEvent, MTLHeap, MTLLibrary, MTLResidencySet, MTLSharedEvent, MTLSharedEventExt,
};
pub use types::{
    MTL4CommandQueueError, MTL4CounterHeapType, MTL4TimestampGranularity, MTL4VisibilityOptions, MTLDataType,
    MTLGPUAddress, MTLGPUFamily, MTLHeapType, MTLResourceOptions, MTLSize, MTLStages, MTLSparsePageSize,
    MTLSparseTextureMappingMode, MTLStorageMode,
};
