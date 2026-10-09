use std::sync::Arc;

use crate::{Device, Error, Result};

/// A host-visible, persistently mapped storage/transfer buffer.
///
/// `gpu_address` is a VA in the shim-owned address space: 16 KiB aligned, unique,
/// never reused while the buffer lives (Metal contract: address + offset stays
/// inside one buffer). It is NOT the raw Vulkan device address; use
/// [`Device::resolve`] to map it back to (VkBuffer, offset) at dispatch time.
pub struct Buffer {
    _priv: (),
}

impl Buffer {
    /// Persistently mapped host pointer; write through it directly.
    pub fn contents(&self) -> *mut u8 {
        unimplemented_stub("Buffer::contents")
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        unimplemented_stub("Buffer::len")
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shim-owned 16 KiB-aligned virtual address (stable for the buffer's lifetime).
    pub fn gpu_address(&self) -> u64 {
        unimplemented_stub("Buffer::gpu_address")
    }
}

fn unimplemented_stub(what: &str) -> ! {
    // temporary: bodies are filled in by the implementation commit
    let _ = what;
    unreachable!()
}

impl Device {
    /// Unified host-visible coherent allocation, persistently mapped.
    pub fn create_buffer(&self, size: usize) -> Result<Arc<Buffer>> {
        let _ = size;
        Err(Error::Unsupported("create_buffer".into()))
    }

    /// Interval-map lookup: which live buffer owns `addr`, and the offset into it.
    pub fn resolve(&self, addr: u64) -> Option<(Arc<Buffer>, u64)> {
        let _ = addr;
        None
    }
}
