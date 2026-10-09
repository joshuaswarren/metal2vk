//! metal2vk host layer: a Metal-style device/queue/pipeline/dispatch model over Vulkan 1.3.
//!
//! Contract: see `host/DESIGN.md`. All handles are `Arc<...>`; errors are [`Error`].
//! The same model is exposed to C through `metal2vk::capi` (see `include/metal2vk.h`);
//! the cdylib/staticlib link name is `metal2vk` (`-lmetal2vk`).

mod addrmap;
pub mod capi;
mod buffer;
mod device;
mod encoder;
mod error;
mod library;
mod pipeline;
mod queue;

pub use buffer::Buffer;
pub use device::{Device, DeviceInfo, TimestampPool};
pub use encoder::Encoder;
pub use error::Error;
pub use library::{
    BufferBindingJson, ConstantJson, ConstantType, ConstantValue, FunctionJson, Library, M2vLibJson,
    PushJson, PushWordJson, SpvRange, ThreadgroupJson, M2VLIB_MAGIC, M2VLIB_VERSION,
};
pub use pipeline::Pipeline;
pub use queue::{CommandBuffer, Completion, Event, Queue};

pub(crate) use device::DeviceInner;

pub type Result<T, E = Error> = std::result::Result<T, E>;
