use std::sync::Arc;

use ash::vk;

use crate::{DeviceInner, Error, Result};

/// A host-visible, persistently mapped storage/transfer buffer.
///
/// `gpu_address` is a VA in the shim-owned address space: 16 KiB aligned, unique,
/// never reused while the buffer lives (Metal contract: address + offset stays
/// inside one buffer). It is NOT the raw Vulkan device address; use
/// [`crate::Device::resolve`] to map it back to (VkBuffer, offset) at dispatch time.
pub struct Buffer {
    pub(crate) inner: BufferInner,
}

impl std::ops::Deref for Buffer {
    type Target = BufferInner;
    fn deref(&self) -> &BufferInner {
        &self.inner
    }
}

impl std::ops::DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut BufferInner {
        &mut self.inner
    }
}

#[doc(hidden)]
pub struct BufferInner {
    /// Weak: buffers must not keep the device alive past vkDestroyDevice.
    pub(crate) device: std::sync::Weak<DeviceInner>,
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
    pub(crate) va: u64,
}

unsafe impl Send for BufferInner {}
unsafe impl Sync for BufferInner {}

impl Drop for BufferInner {
    fn drop(&mut self) {
        let Some(device) = self.device.upgrade() else {
            return; // device already gone; Vulkan teardown frees everything
        };
        device.map.remove(self.va);
        unsafe {
            device.device.destroy_buffer(self.buffer, None);
            device.device.free_memory(self.memory, None);
        }
    }
}

impl Buffer {
    pub(crate) fn allocate(device: &Arc<DeviceInner>, size: usize) -> Result<Arc<Buffer>> {
        if size == 0 {
            return Err(Error::Invalid("zero-size buffer".into()));
        }
        unsafe {
            let buffer = device.device.create_buffer(
                &vk::BufferCreateInfo {
                    size: size as u64,
                    // No SHADER_DEVICE_ADDRESS: the shim's gpu_address() is a
                    // virtual VA resolved by Device::resolve, and honeykrisp
                    // segfaults at dispatch on device-address buffers
                    // (G13C, found against m2v-run, 2026-10-09).
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    ..Default::default()
                },
                None,
            )?;
            let reqs = device.device.get_buffer_memory_requirements(buffer);
            let memory_type = device.find_host_memory_type(reqs)?;
            let memory = device.device.allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: memory_type,
                    ..Default::default()
                },
                None,
            )?;
            device.device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr = device
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;

            let va = device.map.reserve(size as u64);
            let buffer = Arc::new(Buffer {
                inner: BufferInner {
                    device: Arc::downgrade(device),
                    buffer,
                    memory,
                    ptr: ptr.cast(),
                    len: size,
                    va,
                },
            });
            device.map.register(va, &buffer, size as u64);
            Ok(buffer)
        }
    }

    /// Persistently mapped host pointer; write through it directly.
    pub fn contents(&self) -> *mut u8 {
        self.ptr
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Shim-owned 16 KiB-aligned virtual address (stable for the buffer's lifetime).
    pub fn gpu_address(&self) -> u64 {
        self.va
    }

    /// Vulkan-side view used by the encoder at dispatch: buffer + offset.
    pub(crate) fn descriptor_range(&self, offset: u64) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset,
            range: vk::WHOLE_SIZE,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_probe() -> BufferInner {
        // Address tests only exercise identity (Weak upgrade) and never touch
        // Vulkan state; every handle is null and the device link is empty.
        BufferInner {
            device: std::sync::Weak::new(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            ptr: std::ptr::null_mut(),
            len: 0,
            va: 0,
        }
    }
}
