use std::sync::Arc;

use crate::{Buffer, Error, Pipeline, Result, TimestampPool};

/// A compute command encoder. Created from [`crate::CommandBuffer::begin`]; end it
/// with [`Encoder::end`] before ending the command buffer.
pub struct Encoder<'a> {
    _cmd: &'a mut crate::CommandBuffer,
}

impl<'a> Encoder<'a> {
    pub(crate) fn new(cmd: &'a mut crate::CommandBuffer) -> Self {
        Encoder { _cmd: cmd }
    }

    pub fn set_pipeline(&mut self, pipeline: &Arc<Pipeline>) -> Result<()> {
        let _ = pipeline;
        Err(Error::Unsupported("Encoder::set_pipeline".into()))
    }

    /// Metal 4 argument-table semantics: slot -> shim GPU address. Resolved to
    /// (VkBuffer, offset) and written as a push descriptor set at dispatch time.
    pub fn set_address(&mut self, index: u32, gpu_address: u64) -> Result<()> {
        let _ = (index, gpu_address);
        Err(Error::Unsupported("Encoder::set_address".into()))
    }

    /// Small POD constants into the push-constant block (clspv layout: after the
    /// 16-byte group/region offsets block, offset from the library's push words).
    pub fn set_bytes(&mut self, index: u32, bytes: &[u8]) -> Result<()> {
        let _ = (index, bytes);
        Err(Error::Unsupported("Encoder::set_bytes".into()))
    }

    /// Dynamic threadgroup memory length for an OpenCL local pointer argument.
    pub fn set_threadgroup_memory_length(&mut self, index: u32, bytes: u32) -> Result<()> {
        let _ = (index, bytes);
        Err(Error::Unsupported("Encoder::set_threadgroup_memory_length".into()))
    }

    pub fn dispatch_threadgroups(&mut self, grid: [u32; 3], threads_per_threadgroup: [u32; 3]) -> Result<()> {
        let _ = (grid, threads_per_threadgroup);
        Err(Error::Unsupported("Encoder::dispatch_threadgroups".into()))
    }

    /// Non-uniform dispatch: rounds up to whole threadgroups; kernels bounds-check,
    /// as in Metal.
    pub fn dispatch_threads(&mut self, threads: [u32; 3], threads_per_threadgroup: [u32; 3]) -> Result<()> {
        let _ = (threads, threads_per_threadgroup);
        Err(Error::Unsupported("Encoder::dispatch_threads".into()))
    }

    /// Indirect dispatch: `gpu_address` points at 3 x u32 {threadgroups x,y,z}.
    pub fn dispatch_indirect(&mut self, gpu_address: u64, threads_per_threadgroup: [u32; 3]) -> Result<()> {
        let _ = (gpu_address, threads_per_threadgroup);
        Err(Error::Unsupported("Encoder::dispatch_indirect".into()))
    }

    /// vkCmdFillBuffer.
    pub fn fill(&mut self, dst: &Arc<Buffer>, range: std::ops::Range<u64>, value: u8) -> Result<()> {
        let _ = (dst, range, value);
        Err(Error::Unsupported("Encoder::fill".into()))
    }

    /// vkCmdCopyBuffer.
    pub fn copy(
        &mut self,
        src: &Arc<Buffer>,
        src_offset: u64,
        dst: &Arc<Buffer>,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        let _ = (src, src_offset, dst, dst_offset, size);
        Err(Error::Unsupported("Encoder::copy".into()))
    }

    /// Compute -> compute and transfer memory barrier (device visibility).
    pub fn barrier(&mut self) -> Result<()> {
        Err(Error::Unsupported("Encoder::barrier".into()))
    }

    /// VK_EXT_debug_utils named group when the instance has it; documented no-op otherwise.
    pub fn push_debug_group(&mut self, name: &str) {
        let _ = name;
    }

    pub fn pop_debug_group(&mut self) {}

    pub fn write_timestamp(&mut self, pool: &TimestampPool, index: u32) -> Result<()> {
        let _ = (pool, index);
        Err(Error::Unsupported("Encoder::write_timestamp".into()))
    }

    /// Close the encoder; the command buffer stays open until [`crate::CommandBuffer::end`].
    pub fn end(self) -> Result<()> {
        Err(Error::Unsupported("Encoder::end".into()))
    }
}
