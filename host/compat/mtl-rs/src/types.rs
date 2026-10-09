//! Metal value types (bit masks, enums, `MTLSize`) with the same variants and
//! discriminants as mtl-rs 0.3.0.

use bitflags::bitflags;

/// 64-bit GPU virtual address.
pub type MTLGPUAddress = u64;

/// `MTLSize`: a three-dimensional extent. All fields `usize`, like upstream.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct MTLSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl MTLSize {
    pub const fn new(width: usize, height: usize, depth: usize) -> Self {
        Self { width, height, depth }
    }
}

/// Stages of GPU work (`MTLStages`). Values match Metal's `MTLStage*` masks.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct MTLStages(pub usize);

bitflags! {
    impl MTLStages: usize {
        const Vertex = 1 << 0;
        const Fragment = 1 << 1;
        const Tile = 1 << 2;
        const Object = 1 << 3;
        const Mesh = 1 << 4;
        const ResourceState = 1 << 26;
        const Dispatch = 1 << 27;
        const Blit = 1 << 28;
        const AccelerationStructure = 1 << 29;
        const MachineLearning = 1 << 30;
        const All = isize::MAX as usize;
    }
}

/// Memory consistency options for synchronization commands (`MTL4VisibilityOptions`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct MTL4VisibilityOptions(pub usize);

bitflags! {
    impl MTL4VisibilityOptions: usize {
        /// Execution-only barrier (no cache flush).
        const None = 0;
        /// Flush caches to the device coherence point.
        const Device = 1 << 0;
        /// Flush caches for aliased virtual addresses.
        const ResourceAlias = 1 << 1;
    }
}

/// Resource creation options (`MTLResourceOptions`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct MTLResourceOptions(pub usize);

bitflags! {
    impl MTLResourceOptions: usize {
        const CPU_CACHE_MODE_DEFAULT_CACHE = 0 << 0;
        const CPU_CACHE_MODE_WRITE_COMBINED = 1 << 0;
        /// Unified memory: CPU and GPU share one allocation (the host layer's only mode).
        const STORAGE_MODE_SHARED = 0 << 4;
        const STORAGE_MODE_MANAGED = 1 << 4;
        const STORAGE_MODE_PRIVATE = 2 << 4;
        const STORAGE_MODE_MEMORYLESS = 3 << 4;
        const HAZARD_TRACKING_MODE_DEFAULT = 0 << 8;
        const HAZARD_TRACKING_MODE_UNTRACKED = 1 << 8;
        const HAZARD_TRACKING_MODE_TRACKED = 2 << 8;
    }
}

/// Storage mode (`MTLStorageMode`).
#[repr(usize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MTLStorageMode {
    Shared = 0,
    Managed = 1,
    Private = 2,
    Memoryless = 3,
}

/// Sparse page size (`MTLSparsePageSize`).
#[repr(isize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MTLSparsePageSize {
    KB16 = 101,
    KB64 = 102,
    KB256 = 103,
}

/// Sparse mapping operation kind (`MTLSparseTextureMappingMode`).
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MTLSparseTextureMappingMode {
    Map = 0,
    Unmap = 1,
}

/// Metal data type (`MTLDataType`). Only the constants uzu's function-constant
/// values use carry meaningful behaviour here; the full upstream enumeration is
/// far larger.
#[repr(u64)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MTLDataType {
    None = 0,
    Struct = 1,
    Array = 2,
    Float = 3,
    Half = 16,
    Int = 29,
    UInt = 33,
    Bool = 53,
}

/// Metal GPU family (`MTLGPUFamily`).
#[repr(isize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MTLGPUFamily {
    Apple1 = 1001,
    Apple2 = 1002,
    Apple3 = 1003,
    Apple4 = 1004,
    Apple5 = 1005,
    Apple6 = 1006,
    Apple7 = 1007,
    Apple8 = 1008,
    Apple9 = 1009,
    Apple10 = 1010,
    Metal3 = 5001,
    Metal4 = 5002,
}

/// Heap type (`MTLHeapType`).
#[repr(isize)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum MTLHeapType {
    Automatic = 0,
    Placement = 1,
    Reference = 2,
}

/// Counter heap kind (`MTL4CounterHeapType`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MTL4CounterHeapType(pub isize);

impl MTL4CounterHeapType {
    pub const INVALID: Self = Self(0);
    pub const TIMESTAMP: Self = Self(1);
}

/// Timestamp counter precision (`MTL4TimestampGranularity`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MTL4TimestampGranularity(pub isize);

impl MTL4TimestampGranularity {
    pub const RELAXED: Self = Self(0);
    pub const PRECISE: Self = Self(1);
}

/// Kinds of errors a command-queue commit can report (`MTL4CommandQueueError`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MTL4CommandQueueError(pub isize);

impl MTL4CommandQueueError {
    pub const NONE: Self = Self(0);
    pub const TIMEOUT: Self = Self(1);
    pub const NOT_PERMITTED: Self = Self(2);
    pub const OUT_OF_MEMORY: Self = Self(3);
    pub const ACCESS_REVOKED: Self = Self(5);
}
