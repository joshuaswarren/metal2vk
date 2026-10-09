use std::sync::Arc;

use ash::vk;

use crate::{ConstantValue, DeviceInner, Error, Pipeline, Result, TimestampPool};

/// A compute command encoder. Created from [`CommandBuffer::begin`]; close it
/// with [`Encoder::end`] before ending the command buffer.
pub struct Encoder {
    pub(crate) cb: Arc<crate::queue::CbInner>,
    pipeline: Option<Arc<Pipeline>>,
    addresses: Vec<(u32, u64)>,
    push_data: Vec<u8>,
    tg_lengths: Vec<(u32, u32)>,
    desc_set: Option<vk::DescriptorSet>,
}

/// Single source of truth for push-vs-fallback descriptors: the queue creates
/// the fallback pool exactly when this returns false, and the encoder uses
/// push descriptors exactly when it returns true.
pub(crate) fn use_push_descriptors(device: &DeviceInner) -> bool {
    device.has_push_descriptor && std::env::var_os("M2V_NO_PUSH_DESCRIPTOR").is_none()
}

impl Encoder {
pub(crate) fn new(cb: Arc<crate::queue::CbInner>) -> Self {
        Encoder {
            cb,
            pipeline: None,
            addresses: Vec::new(),
            push_data: Vec::new(),
            tg_lengths: Vec::new(),
            desc_set: None,
        }
    }

    pub fn set_pipeline(&mut self, pipeline: &Arc<Pipeline>) -> Result<()> {
        if self.pipeline.as_ref().map(|p| Arc::ptr_eq(p, pipeline)) != Some(true) {
            self.pipeline = Some(pipeline.clone());
            self.push_data = vec![0; pipeline.push_size as usize];
            self.addresses.clear();
            self.tg_lengths.clear();
            self.desc_set = None;
        }
        Ok(())
    }

    /// Metal 4 argument-table semantics: slot -> shim GPU address. Resolved to
    /// (VkBuffer, offset) and written as a push descriptor set at dispatch time.
    pub fn set_address(&mut self, index: u32, gpu_address: u64) -> Result<()> {
        match self.addresses.iter_mut().find(|(a, _)| *a == index) {
            Some(slot) => slot.1 = gpu_address,
            None => self.addresses.push((index, gpu_address)),
        }
        Ok(())
    }

    /// Small POD constants into the push-constant block (clspv layout: after the
    /// 16-byte group/region offsets block, which stays zero, offsets from the
    /// library's push words).
    pub fn set_bytes(&mut self, index: u32, bytes: &[u8]) -> Result<()> {
        let pipeline = self.pipeline()?;
        let Some((_, offset, size)) = pipeline.push_words.iter().find(|(a, _, _)| *a == index) else {
            return Err(Error::Invalid(format!(
                "set_bytes: argument {index} is not a POD push-constant argument of {:?}",
                pipeline.name
            )));
        };
        if bytes.len() as u32 > *size {
            return Err(Error::Invalid(format!(
                "set_bytes: {} bytes for argument {index} exceeds the declared {size}",
                bytes.len()
            )));
        }
        let offset = *offset as usize;
        self.push_data[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Dynamic threadgroup memory length for an OpenCL local pointer argument.
    /// Applied when the dispatch is recorded: the matching pipeline variant is
    /// specialised with the length as a u32 spec constant (clspv ArgumentWorkgroup).
    pub fn set_threadgroup_memory_length(&mut self, index: u32, bytes: u32) -> Result<()> {
        if bytes % 4 != 0 {
            return Err(Error::Invalid(format!(
                "threadgroup memory length {bytes} must be a multiple of 4"
            )));
        }
        match self.tg_lengths.iter_mut().find(|(a, _)| *a == index) {
            Some(slot) => slot.1 = bytes,
            None => self.tg_lengths.push((index, bytes)),
        }
        Ok(())
    }

    pub fn dispatch_threadgroups(&mut self, grid: [u32; 3], _threads_per_threadgroup: [u32; 3]) -> Result<()> {
        let pipeline = self.resolve_pipeline()?;
        unsafe { self.record_dispatch(&pipeline, grid) }
    }

    /// Non-uniform dispatch: rounds up to whole threadgroups; kernels bounds-check,
    /// as in Metal.
    pub fn dispatch_threads(&mut self, threads: [u32; 3], threads_per_threadgroup: [u32; 3]) -> Result<()> {
        let pipeline = self.resolve_pipeline()?;
        let mut grid = [0u32; 3];
        for i in 0..3 {
            let tpt = threads_per_threadgroup[i].max(1);
            grid[i] = threads[i].div_ceil(tpt);
        }
        unsafe { self.record_dispatch(&pipeline, grid) }
    }

    /// Indirect dispatch: `gpu_address` points at 3 x u32 {threadgroups x,y,z}.
    pub fn dispatch_indirect(&mut self, gpu_address: u64, threads_per_threadgroup: [u32; 3]) -> Result<()> {
        let _ = threads_per_threadgroup; // already baked into the pipeline's spec constants
        let pipeline = self.resolve_pipeline()?;
        let (buffer, offset) = self
            .cb
            .device
            .map
            .resolve(gpu_address)
            .ok_or_else(|| Error::Invalid(format!("dispatch_indirect: no buffer at address {gpu_address:#x}")))?;
        if offset % 4 != 0 || buffer.len() < (offset as usize + 12) {
            return Err(Error::Invalid(format!(
                "dispatch_indirect: address {gpu_address:#x} does not cover 3 x u32"
            )));
        }
        self.bind_descriptors(&pipeline)?;
        unsafe {
            self.cb.device.device.cmd_push_constants(
                self.cb.raw,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.push_data,
            );
            self.cb.device.device.cmd_dispatch_indirect(
                self.cb.raw,
                buffer.buffer,
                offset,
            );
        }
        Ok(())
    }

    /// vkCmdFillBuffer; the Vulkan command requires 4-byte-aligned offset/size.
    pub fn fill(&mut self, dst: &Arc<crate::Buffer>, range: std::ops::Range<u64>, value: u8) -> Result<()> {
        if range.start % 4 != 0 || range.end % 4 != 0 {
            return Err(Error::Invalid(format!(
                "fill: range {range:?} must be 4-byte aligned (Vulkan constraint; Metal allows any byte)"
            )));
        }
        if range.end > dst.len() as u64 {
            return Err(Error::Invalid("fill: range outside buffer".into()));
        }
        unsafe {
            self.cb.device.device.cmd_fill_buffer(
                self.cb.raw,
                dst.buffer,
                range.start,
                range.end - range.start,
                u32::from_ne_bytes([value; 4]),
            );
        }
        Ok(())
    }

    /// vkCmdCopyBuffer; 4-byte-aligned offsets and size.
    pub fn copy(
        &mut self,
        src: &Arc<crate::Buffer>,
        src_offset: u64,
        dst: &Arc<crate::Buffer>,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        if src_offset % 4 != 0 || dst_offset % 4 != 0 || size % 4 != 0 {
            return Err(Error::Invalid("copy: offsets and size must be 4-byte aligned".into()));
        }
        if src_offset + size > src.len() as u64 || dst_offset + size > dst.len() as u64 {
            return Err(Error::Invalid("copy: range outside a buffer".into()));
        }
        unsafe {
            self.cb.device.device.cmd_copy_buffer(
                self.cb.raw,
                src.buffer,
                dst.buffer,
                &[vk::BufferCopy { src_offset, dst_offset, size }],
            );
        }
        Ok(())
    }

    /// Compute -> compute and transfer memory barrier (device visibility).
    pub fn barrier(&mut self) -> Result<()> {
        unsafe {
            self.cb.device.device.cmd_pipeline_barrier(
                self.cb.raw,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                }],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// VK_EXT_debug_utils named group when the instance has it; documented no-op otherwise.
    pub fn push_debug_group(&mut self, name: &str) {
        if !self.cb.device.has_debug_utils {
            return;
        }
        let c = std::ffi::CString::new(name).unwrap_or_default();
        unsafe {
            self.cb.device.debug_utils.cmd_begin_debug_utils_label(
                self.cb.raw,
                &vk::DebugUtilsLabelEXT {
                    p_label_name: c.as_ptr(),
                    ..Default::default()
                },
            );
        }
    }

    pub fn pop_debug_group(&mut self) {
        if !self.cb.device.has_debug_utils {
            return;
        }
        unsafe {
            self.cb.device.debug_utils.cmd_end_debug_utils_label(self.cb.raw);
        }
    }

    /// Records a GPU timestamp into a user pool when the GPU reaches this point.
    /// The pool is single-shot: it starts reset and `results` reads it once.
    pub fn write_timestamp(&mut self, pool: &TimestampPool, index: u32) -> Result<()> {
        if index >= pool.count {
            return Err(Error::Invalid(format!(
                "write_timestamp: index {index} outside pool of {} queries",
                pool.count
            )));
        }
        unsafe {
            self.cb.device.device.cmd_write_timestamp(
                self.cb.raw,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                **pool,
                index,
            );
        }
        Ok(())
    }

    /// Close the encoder; the command buffer stays open until [`CommandBuffer::end`].
    pub fn end(self) -> Result<()> {
        unsafe {
            self.cb.device.device.cmd_write_timestamp(
                self.cb.raw,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.cb.query_pool,
                1,
            );
        }
        {
            let mut st = self.cb.state.lock();
            st.encoder_open = false;
        }
        Ok(())
    }

    fn pipeline(&self) -> Result<&Arc<Pipeline>> {
        self.pipeline
            .as_ref()
            .ok_or_else(|| Error::Invalid("no pipeline set (call set_pipeline first)".into()))
    }

    fn resolve_pipeline(&mut self) -> Result<Arc<Pipeline>> {
        let base = self.pipeline()?.clone();
        if base.threadgroup.is_empty() {
            return Ok(base);
        }
        // Dynamic threadgroup memory: the lengths must be specialised into the
        // pipeline (u32 spec constants). Variant pipelines are cached on the queue.
        let mut combined = base.constants.clone();
        let mut key = Vec::with_capacity(64);
        key.extend_from_slice(base.name.as_bytes());
        key.push(0);
        for (arg, spec_id) in &base.threadgroup {
            let len = self
                .tg_lengths
                .iter()
                .find(|(a, _)| a == arg)
                .map(|(_, l)| *l)
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "argument {arg} needs set_threadgroup_memory_length before dispatch"
                    ))
                })?;
            combined.push((*spec_id, ConstantValue::U32(len)));
            key.extend_from_slice(&[*arg as u8, 0, 0, 0]);
            key.extend_from_slice(&len.to_le_bytes());
        }
        let queue = self
            .cb
            .queue
            .upgrade()
            .ok_or_else(|| Error::Invalid("queue is gone".into()))?;
        let mut variants = queue.variants.lock();
        if let Some(p) = variants.get(&key) {
            return Ok(p.clone());
        }
        let variant = Pipeline::create(&self.cb.device, &base.library, &base.name, &combined)?;
        variants.insert(key, variant.clone());
        Ok(variant)
    }

    fn bind_descriptors(&mut self, pipeline: &Arc<Pipeline>) -> Result<()> {
        let device: &DeviceInner = &self.cb.device;
        // Honeykrisp on M2 Max (G14C) segfaults in vkCmdPushDescriptorSetKHR at
        // record time (2026-10-09); M2V_NO_PUSH_DESCRIPTOR=1 forces the plain
        // descriptor-set path on such drivers.
        let use_push = use_push_descriptors(device);
        let mut infos = Vec::with_capacity(pipeline.buffer_bindings.len());
        for (arg, _binding, _set) in &pipeline.buffer_bindings {
            let addr = self
                .addresses
                .iter()
                .find(|(a, _)| a == arg)
                .map(|(_, v)| *v)
                .ok_or_else(|| {
                    Error::Invalid(format!("dispatch: no address bound for argument {arg}"))
                })?;
            let (buffer, offset) = device.map.resolve(addr).ok_or_else(|| {
                Error::Invalid(format!("dispatch: address {addr:#x} (argument {arg}) is not a live buffer"))
            })?;
            if buffer.len() as u64 > device.max_storage_buffer_range {
                return Err(Error::Invalid(format!(
                    "buffer for argument {arg} is {} bytes, above maxStorageBufferRange ({})",
                    buffer.len(),
                    device.max_storage_buffer_range
                )));
            }
            infos.push(buffer.descriptor_range(offset));
        }
        unsafe {
            if use_push {
                let writes: Vec<vk::WriteDescriptorSet> = infos
                    .iter()
                    .zip(&pipeline.buffer_bindings)
                    .map(|(info, (_, binding, _))| vk::WriteDescriptorSet {
                        dst_set: vk::DescriptorSet::null(),
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: info,
                        ..Default::default()
                    })
                    .collect();
                device.push_descriptor.as_ref().expect("push descriptor loader").cmd_push_descriptor_set(
                    self.cb.raw,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &writes,
                );
            } else if !infos.is_empty() {
                let set = match self.desc_set {
                    Some(s) => s,
                    None => {
                        let pool = self.cb.desc_pool.ok_or_else(|| {
                            Error::Unsupported("no push descriptors and no fallback pool".into())
                        })?;
                        let set_layouts = [pipeline.desc_layout.expect("set layout")];
                        let set = device
                            .device
                            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                                descriptor_pool: pool,
                                descriptor_set_count: 1,
                                p_set_layouts: set_layouts.as_ptr(),
                                ..vk::DescriptorSetAllocateInfo::default()
                            })?[0];
                        self.desc_set = Some(set);
                        set
                    }
                };
                let writes: Vec<vk::WriteDescriptorSet> = infos
                    .iter()
                    .zip(&pipeline.buffer_bindings)
                    .map(|(info, (_, binding, _))| vk::WriteDescriptorSet {
                        dst_set: set,
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: info,
                        ..Default::default()
                    })
                    .collect();
                device.device.update_descriptor_sets(&writes, &[]);
                device.device.cmd_bind_descriptor_sets(
                    self.cb.raw,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[set],
                    &[],
                );
            }
        }
        Ok(())
    }

    unsafe fn record_dispatch(&mut self, pipeline: &Arc<Pipeline>, grid: [u32; 3]) -> Result<()> {
        // Bind at every dispatch so mid-encoder pipeline switches just work.
        self.cb.device.device.cmd_bind_pipeline(
            self.cb.raw,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        self.bind_descriptors(pipeline)?;
        // Kernels without POD arguments have an empty push block; do not call
        // vkCmdPushConstants with a zero-size range (honeykrisp G13C/G14C
        // segfault on it, 2026-10-09).
        if !self.push_data.is_empty() {
            self.cb.device.device.cmd_push_constants(
                self.cb.raw,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.push_data,
            );
        }
        self.cb.device.device.cmd_dispatch(self.cb.raw, grid[0], grid[1], grid[2]);
        Ok(())
    }


}

impl Drop for Encoder {
    fn drop(&mut self) {
        // If the user dropped the encoder without end(), close the state so the
        // command buffer reports a clear error instead of hanging in "open".
        let mut st = self.cb.state.lock();
        st.encoder_open = false;
    }
}
