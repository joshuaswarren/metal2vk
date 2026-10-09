//! The Metal protocol surface uzu consumes: Rust traits standing in for Objective-C
//! protocols, with signatures copied from mtl-rs 0.3.0.
//!
//! Descriptors (`*Descriptor`, `MTLFunctionConstantValues`, commit options) are
//! classes: interior-mutable, `Retained`-wrapped, cloneable. Protocol traits are
//! implemented for `ProtocolObject<dyn Trait>`, which is what `Retained` hands out;
//! calls delegate into the wrapped host-backed object.

use std::ffi::c_void;
use std::ops::{Deref, Range};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::MetalError;
use crate::protocols::{MTL4CommitFeedback, MTLDevice, MTLSharedEvent, MTLLibrary};
use crate::shim;
use crate::types::*;
use objc2::{ProtocolObject, Retained};

// ---------------------------------------------------------------------------
// Classes: descriptors and mutable option objects
// ---------------------------------------------------------------------------

macro_rules! descriptor_class {
    ($name:ident, $fields:ident, { $($f:ident : $t:ty),* $(,)? }) => {
        #[derive(Debug)]
        pub struct $name {
            pub(crate) f: Mutex<$fields>,
        }
        #[derive(Debug, Clone, Default)]
        pub(crate) struct $fields {
            $(pub $f: $t,)*
        }
        impl Default for $name {
            fn default() -> Self {
                Self { f: Mutex::new($fields::default()) }
            }
        }
        impl Clone for $name {
            fn clone(&self) -> Self {
                Self { f: Mutex::new(self.f.lock().clone()) }
            }
        }
    };
}

descriptor_class!(MTLHeapDescriptor, HeapDescriptorFields, {
    size: usize,
    storage_mode: Option<MTLStorageMode>,
    r#type: Option<MTLHeapType>,
    sparse_page_size: Option<MTLSparsePageSize>,
    max_compatible_placement_sparse_page_size: Option<MTLSparsePageSize>,
});

impl MTLHeapDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
    pub fn set_size(&self, size: usize) {
        self.f.lock().size = size;
    }
    pub fn set_storage_mode(&self, storage_mode: MTLStorageMode) {
        self.f.lock().storage_mode = Some(storage_mode);
    }
    pub fn set_type(&self, r#type: MTLHeapType) {
        self.f.lock().r#type = Some(r#type);
    }
    pub fn set_sparse_page_size(&self, sparse_page_size: MTLSparsePageSize) {
        self.f.lock().sparse_page_size = Some(sparse_page_size);
    }
    pub fn set_max_compatible_placement_sparse_page_size(&self, page_size: MTLSparsePageSize) {
        self.f.lock().max_compatible_placement_sparse_page_size = Some(page_size);
    }
}

descriptor_class!(MTLResidencySetDescriptor, ResidencySetDescriptorFields, {
    initial_capacity: usize,
});

impl MTLResidencySetDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
    pub fn set_initial_capacity(&self, initial_capacity: usize) {
        self.f.lock().initial_capacity = initial_capacity;
    }
}

descriptor_class!(MTL4ArgumentTableDescriptor, ArgumentTableDescriptorFields, {
    max_buffer_bind_count: usize,
});

impl MTL4ArgumentTableDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
    pub fn set_max_buffer_bind_count(&self, max_buffer_bind_count: usize) {
        self.f.lock().max_buffer_bind_count = max_buffer_bind_count;
    }
}

descriptor_class!(MTL4CounterHeapDescriptor, CounterHeapDescriptorFields, {
    r#type: Option<MTL4CounterHeapType>,
    count: usize,
});

impl MTL4CounterHeapDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
    pub fn set_type(&self, r#type: MTL4CounterHeapType) {
        self.f.lock().r#type = Some(r#type);
    }
    pub fn set_count(&self, count: usize) {
        self.f.lock().count = count;
    }
    pub fn count(&self) -> usize {
        self.f.lock().count
    }
}

descriptor_class!(MTL4CompilerDescriptor, CompilerDescriptorFields, {});

impl MTL4CompilerDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
}

/// `MTL4CompilerTaskOptions` (uzu only ever passes `None`).
#[derive(Debug, Clone, Copy, Default)]
pub struct MTL4CompilerTaskOptions;

impl MTL4CompilerTaskOptions {
    pub fn new() -> Retained<Self> {
        Retained::new(Self)
    }
}

descriptor_class!(MTL4FunctionDescriptor, FunctionDescriptorFields, {
    library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    name: Option<String>,
    // Set by `MTL4SpecializedFunctionDescriptor` so the pipeline descriptor's
    // base-class snapshot carries the real function + constants through.
    specialized: Option<Box<FunctionDescriptorFields>>,
    specialized_constants: Option<Retained<MTLFunctionConstantValues>>,
});

impl MTL4FunctionDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
    pub fn set_library(&self, library: Option<&Retained<ProtocolObject<dyn MTLLibrary>>>) {
        self.f.lock().library = library.cloned();
    }
    pub fn set_name(&self, name: Option<&str>) {
        self.f.lock().name = name.map(str::to_owned);
    }
    pub(crate) fn snapshot(&self) -> FunctionDescriptorFields {
        self.f.lock().clone()
    }
}

/// `MTL4LibraryFunctionDescriptor` (deref target: `MTL4FunctionDescriptor`).
#[derive(Debug, Clone, Default)]
pub struct MTL4LibraryFunctionDescriptor {
    pub(crate) base: MTL4FunctionDescriptor,
}

impl Deref for MTL4LibraryFunctionDescriptor {
    type Target = MTL4FunctionDescriptor;
    fn deref(&self) -> &MTL4FunctionDescriptor {
        &self.base
    }
}

impl MTL4LibraryFunctionDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
}

/// `MTL4SpecializedFunctionDescriptor` (deref target: `MTL4FunctionDescriptor`).
#[derive(Debug, Clone, Default)]
pub struct MTL4SpecializedFunctionDescriptor {
    pub(crate) base: MTL4FunctionDescriptor,
}

impl Deref for MTL4SpecializedFunctionDescriptor {
    type Target = MTL4FunctionDescriptor;
    fn deref(&self) -> &MTL4FunctionDescriptor {
        &self.base
    }
}

impl MTL4SpecializedFunctionDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
    pub fn set_function_descriptor(&self, function_descriptor: Option<&MTL4FunctionDescriptor>) {
        // The pipeline descriptor only sees this object's base-class snapshot;
        // record the real function there so the compiler can read it back.
        self.base.f.lock().specialized = function_descriptor.map(|d| Box::new(d.snapshot()));
    }
    pub fn set_constant_values(&self, constant_values: Option<&MTLFunctionConstantValues>) {
        self.base.f.lock().specialized_constants = constant_values.map(|values| {
            Retained::new(MTLFunctionConstantValues {
                constants: Mutex::new(values.constants.lock().clone()),
            })
        });
    }
}

descriptor_class!(MTL4ComputePipelineDescriptor, ComputePipelineDescriptorFields, {
    compute_function_descriptor: Option<FunctionDescriptorFields>,
});

impl MTL4ComputePipelineDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
    pub fn set_compute_function_descriptor(&self, compute_function_descriptor: Option<&MTL4FunctionDescriptor>) {
        self.f.lock().compute_function_descriptor = compute_function_descriptor.map(|d| d.snapshot());
    }
}

/// `MTLFunctionConstantValues` class.
#[derive(Debug, Default)]
pub struct MTLFunctionConstantValues {
    pub(crate) constants: Mutex<Vec<shim::FunctionConstant>>,
}

impl MTLFunctionConstantValues {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }

    pub fn set_constant_value_type_at_index(&self, value: NonNull<c_void>, r#type: MTLDataType, index: usize) {
        let bytes = shim::constant_value_bytes(value, r#type);
        self.constants.lock().push(shim::FunctionConstant { index, ty: r#type, bytes });
    }
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// `MTLCaptureDestination`.
#[repr(i64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MTLCaptureDestination {
    DeveloperTools = 1,
    GPUTraceDocument = 2,
}

/// `MTLCaptureTarget` (uzu only ever captures the device).
#[derive(Clone)]
pub enum MTLCaptureTarget {
    Device(Retained<ProtocolObject<dyn MTLDevice>>),
}

descriptor_class!(MTLCaptureDescriptor, CaptureDescriptorFields, {
    destination: Option<MTLCaptureDestination>,
    output_path: Option<PathBuf>,
});

impl MTLCaptureDescriptor {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }
    pub fn set_destination(&self, destination: MTLCaptureDestination) {
        self.f.lock().destination = Some(destination);
    }
    pub fn set_output_path(&self, output_path: Option<&Path>) {
        self.f.lock().output_path = output_path.map(Path::to_path_buf);
    }
    pub fn set_capture_object(&self, capture_object: Option<&MTLCaptureTarget>) {
        if let Some(MTLCaptureTarget::Device(device)) = capture_object {
            shim::set_capture_device(Arc::clone(device.as_arc()));
        }
    }
}

/// `MTLCaptureManager` class: GPU trace capture has no host equivalent.
#[derive(Debug, Clone, Copy, Default)]
pub struct MTLCaptureManager;

impl MTLCaptureManager {
    pub fn shared_capture_manager() -> Retained<Self> {
        Retained::new(Self)
    }

    /// Always fails: the Vulkan host cannot produce a Metal GPU trace document.
    pub fn start_capture_with_descriptor(&self, _descriptor: &MTLCaptureDescriptor) -> Result<(), MetalError> {
        Err(MetalError::new(
            "GPU capture is not supported by the m2v host layer (no Metal GPU trace document on Vulkan)",
        ))
    }

    pub fn stop_capture(&self) {}
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

type SharedEventBlock = Arc<dyn Fn(&ProtocolObject<dyn MTLSharedEvent>, u64) + Send + Sync>;

/// `MTLSharedEventNotificationBlock`: owns the handler closure.
#[derive(Clone)]
pub struct MTLSharedEventNotificationBlock(pub(crate) SharedEventBlock);

impl MTLSharedEventNotificationBlock {
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(&ProtocolObject<dyn MTLSharedEvent>, u64) + Send + Sync + 'static,
    {
        Self(Arc::new(handler))
    }
}

/// `MTLSharedEventListener`: a no-op singleton stand-in (dispatch sources have no
/// host equivalent; only the sparse teardown touches this).
#[derive(Debug, Clone, Copy, Default)]
pub struct MTLSharedEventListener;

impl MTLSharedEventListener {
    pub fn shared_listener() -> Retained<Self> {
        Retained::new(Self)
    }
}

// ---------------------------------------------------------------------------
// Metal 4: commit feedback plumbing
// ---------------------------------------------------------------------------

type CommitFeedbackBlock = Arc<dyn Fn(&ProtocolObject<dyn MTL4CommitFeedback>) + Send + Sync>;

/// `MTL4CommitFeedbackHandler`.
#[derive(Clone)]
pub struct MTL4CommitFeedbackHandler(pub(crate) CommitFeedbackBlock);

impl MTL4CommitFeedbackHandler {
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(&ProtocolObject<dyn MTL4CommitFeedback>) + Send + Sync + 'static,
    {
        Self(Arc::new(handler))
    }
}

/// `MTL4CommitOptions`: collects feedback handlers for one commit.
#[derive(Default)]
pub struct MTL4CommitOptions {
    pub(crate) handlers: Mutex<Vec<CommitFeedbackBlock>>,
}

impl MTL4CommitOptions {
    pub fn new() -> Retained<Self> {
        Retained::new(Self::default())
    }

    pub fn add_feedback_handler(&self, handler: &MTL4CommitFeedbackHandler) {
        self.handlers.lock().push(Arc::clone(&handler.0));
    }
}

/// Sparse buffer mapping update arguments (`MTL4UpdateSparseBufferMappingOperation`).
#[derive(Clone, Debug, PartialEq)]
pub struct MTL4UpdateSparseBufferMappingOperation {
    pub mode: MTLSparseTextureMappingMode,
    buffer_range: (usize, usize),
    pub heap_offset: usize,
}

impl MTL4UpdateSparseBufferMappingOperation {
    /// `buffer_range` is in bytes.
    pub fn new(mode: MTLSparseTextureMappingMode, buffer_range: Range<usize>, heap_offset: usize) -> Self {
        Self { mode, buffer_range: (buffer_range.start, buffer_range.end), heap_offset }
    }

    pub fn buffer_range(&self) -> Range<usize> {
        self.buffer_range.0..self.buffer_range.1
    }
}
