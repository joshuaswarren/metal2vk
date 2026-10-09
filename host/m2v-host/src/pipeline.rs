use std::sync::Arc;

use ash::vk;

use crate::library::{ConstantType, ConstantValue, FunctionJson, Library};
use crate::{DeviceInner, Error, Result};

/// A compiled compute pipeline: one SPIR-V entry point with specialization applied.
pub struct Pipeline {
    pub(crate) inner: PipelineInner,
}

impl std::ops::Deref for Pipeline {
    type Target = PipelineInner;
    fn deref(&self) -> &PipelineInner {
        &self.inner
    }
}

#[doc(hidden)]
pub struct PipelineInner {
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) library: Arc<Library>,
    /// Metal function name this pipeline was built from (variant key part).
    pub(crate) name: String,
    /// Function constants the pipeline was specialised with (variant key part).
    pub(crate) constants: Vec<(u32, ConstantValue)>,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) module: vk::ShaderModule,
    /// Set 0 layout (storage buffers); `None` when the kernel has no buffers and
    /// push descriptors are unavailable.
    pub(crate) desc_layout: Option<vk::DescriptorSetLayout>,
    pub(crate) push_size: u32,
    pub(crate) push_words: Vec<(u32, u32, u32)>, // (arg, offset, size)
    pub(crate) buffer_bindings: Vec<(u32, u32, u32)>, // (arg, binding, set)
    pub(crate) threadgroup: Vec<(u32, u32)>,     // (arg, spec_id)
    pub(crate) workgroup_size: [u32; 3],
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device.device.destroy_pipeline_layout(self.layout, None);
            self.device.device.destroy_shader_module(self.module, None);
            if let Some(l) = self.desc_layout {
                self.device.device.destroy_descriptor_set_layout(l, None);
            }
        }
    }
}

impl Pipeline {
    pub fn max_threads_per_threadgroup(&self) -> u32 {
        self.workgroup_size.iter().product()
    }

    /// Metal threadExecutionWidth (the device subgroup size).
    pub fn thread_execution_width(&self) -> u32 {
        self.device.info.subgroup_size
    }

    /// Threadgroup memory the pipeline declares statically. clspv lowers all
    /// OpenCL local pointers to spec-sized dynamic arrays, so this is 0 unless a
    /// future emitter adds fixed shared memory; dynamic lengths come through
    /// `Encoder::set_threadgroup_memory_length`.
    pub fn static_threadgroup_memory(&self) -> u32 {
        0
    }

    pub(crate) fn create(
        device: &Arc<DeviceInner>,
        library: &Arc<Library>,
        name: &str,
        constants: &[(u32, ConstantValue)],
    ) -> Result<Arc<Pipeline>> {
        let f = library.function(name)?;
        let spv = library.spv_blob(f)?;
        unsafe { create_pipeline(device, library, f, spv, constants) }
    }
}

/// Pure packing of Metal function constants + workgroup size into Vulkan
/// specialization data. GPU-free and unit tested.
pub(crate) fn pack_spec_data(
    workgroup_size: [u32; 3],
    workgroup_size_spec_ids: Option<[u32; 3]>,
    available: &[crate::ConstantJson],
    requested: &[(u32, ConstantValue)],
) -> Result<(Vec<vk::SpecializationMapEntry>, Vec<u8>)> {
    // clspv always emits spec constants 0..2 for the workgroup size (defaults 1,1,1).
    let wg_ids = workgroup_size_spec_ids.unwrap_or([0, 1, 2]);
    let mut data = Vec::with_capacity(16 + 4 * requested.len());
    let mut entries = Vec::with_capacity(3 + requested.len());
    for (i, id) in wg_ids.into_iter().enumerate() {
        entries.push(vk::SpecializationMapEntry {
            constant_id: id,
            offset: (i * 4) as u32,
            size: 4,
        });
    }
    data.extend_from_slice(&workgroup_size[0].to_le_bytes());
    data.extend_from_slice(&workgroup_size[1].to_le_bytes());
    data.extend_from_slice(&workgroup_size[2].to_le_bytes());

    for (index, value) in requested {
        let c = available.iter().find(|c| c.index == *index).ok_or_else(|| {
            Error::Invalid(format!(
                "function constant {index} is not declared by this kernel (declared: {:?})",
                available.iter().map(|c| c.index).collect::<Vec<_>>()
            ))
        })?;
        let ok = matches!(
            (c.ty, value),
            (ConstantType::Bool, ConstantValue::Bool(_))
                | (ConstantType::U32, ConstantValue::U32(_))
                | (ConstantType::I32, ConstantValue::I32(_))
                | (ConstantType::F32, ConstantValue::F32(_))
        );
        if !ok {
            return Err(Error::Invalid(format!(
                "function constant {index} is {:?} but a {value:?} was passed",
                c.ty
            )));
        }
        let off = data.len() as u32;
        match *value {
            ConstantValue::Bool(b) => data.extend_from_slice(&(b as u32).to_le_bytes()),
            ConstantValue::U32(v) => data.extend_from_slice(&v.to_le_bytes()),
            ConstantValue::I32(v) => data.extend_from_slice(&v.to_le_bytes()),
            ConstantValue::F32(v) => data.extend_from_slice(&v.to_le_bytes()),
        }
        entries.push(vk::SpecializationMapEntry { constant_id: c.spec_id, offset: off, size: 4 });
    }
    Ok((entries, data))
}

unsafe fn create_pipeline(
    device: &Arc<DeviceInner>,
    library: &Arc<Library>,
    f: &FunctionJson,
    spv: &[u8],
    constants: &[(u32, ConstantValue)],
) -> Result<Arc<Pipeline>> {
    let (spec_entries, spec_data) = pack_spec_data(
        f.workgroup_size,
        f.workgroup_size_spec_ids,
        &f.constants,
        constants,
    )?;
    let spec_info = vk::SpecializationInfo {
        map_entry_count: spec_entries.len() as u32,
        p_map_entries: spec_entries.as_ptr(),
        data_size: spec_data.len(),
        p_data: spec_data.as_ptr().cast(),
        ..Default::default()
    };

    let module = device.device.create_shader_module(
        &vk::ShaderModuleCreateInfo {
            code_size: spv.len(),
            p_code: spv.as_ptr().cast(),
            ..Default::default()
        },
        None,
    )?;

    let entry = std::ffi::CString::new(f.entry.as_str())
        .map_err(|_| Error::Invalid(format!("entry name {:?} has a NUL byte", f.entry)))?;

    // Set 0: storage buffers, push descriptor when the device offers it.
    let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = f
        .buffers
        .iter()
        .map(|b| vk::DescriptorSetLayoutBinding {
            binding: b.binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        })
        .collect();
    bindings.sort_by_key(|b| b.binding);
    let needs_set0 = !bindings.is_empty() || device.has_push_descriptor;
    let desc_layout = if needs_set0 {
        Some(device.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo {
                flags: if device.has_push_descriptor {
                    vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
                } else {
                    vk::DescriptorSetLayoutCreateFlags::empty()
                },
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
                ..Default::default()
            },
            None,
        )?)
    } else {
        None
    };

    let max_push = device.max_push_constants_size;
    let push_size = f.push.size;
    if push_size > max_push {
        return Err(Error::Unsupported(format!(
            "push constant block {push_size} bytes exceeds the device maximum {max_push}"
        )));
    }
    let push_ranges = if push_size > 0 {
        [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: (push_size + 3) & !3, // range sizes are 4-byte multiples
        }]
    } else {
        [vk::PushConstantRange::default()]
    };
    let set_layouts: Vec<vk::DescriptorSetLayout> = desc_layout.into_iter().collect();
    let layout = device.device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo {
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: if push_size > 0 { 1 } else { 0 },
            p_push_constant_ranges: push_ranges.as_ptr(),
            ..Default::default()
        },
        None,
    )?;

    let stage = vk::PipelineShaderStageCreateInfo {
        stage: vk::ShaderStageFlags::COMPUTE,
        module,
        p_name: entry.as_ptr(),
        p_specialization_info: &spec_info,
        ..Default::default()
    };
    let pipeline = device
        .device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo {
                stage,
                layout,
                ..Default::default()
            }],
            None,
        )
        .map_err(|(_, e)| e)?[0];

    Ok(Arc::new(Pipeline {
        inner: PipelineInner {
            device: device.clone(),
            library: library.clone(),
            name: f.name.clone(),
            constants: constants.to_vec(),
            pipeline,
            layout,
            module,
            desc_layout,
            push_size,
            push_words: f.push.words.iter().map(|w| (w.arg, w.offset, w.size)).collect(),
            buffer_bindings: f.buffers.iter().map(|b| (b.arg, b.binding, b.set)).collect(),
            threadgroup: f.threadgroup.iter().map(|t| (t.arg, t.spec_id)).collect(),
            workgroup_size: f.workgroup_size,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstantJson;

    fn c(index: u32, spec_id: u32, ty: ConstantType) -> ConstantJson {
        ConstantJson { index, spec_id, ty }
    }

    #[test]
    fn workgroup_size_is_always_packed_at_ids_0_to_2() {
        let (entries, data) = pack_spec_data([32, 4, 1], None, &[], &[]).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(u32::from_le_bytes(data[0..4].try_into().unwrap()), 32);
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(data[8..12].try_into().unwrap()), 1);
        assert!(entries[..3].iter().all(|e| e.size == 4));
        assert_eq!(entries[0].constant_id, 0);
        assert_eq!(entries[1].constant_id, 1);
        assert_eq!(entries[2].constant_id, 2);
    }

    #[test]
    fn constants_pack_after_the_workgroup_block() {
        let available = [c(0, 7, ConstantType::F32), c(3, 9, ConstantType::I32)];
        let (entries, data) = pack_spec_data(
            [1, 1, 1],
            None,
            &available,
            &[(3, ConstantValue::I32(-5)), (0, ConstantValue::F32(1.5))],
        )
        .unwrap();
        assert_eq!(entries.len(), 5);
        assert_eq!(u32::from_le_bytes(data[12..16].try_into().unwrap()), -5i32 as u32);
        assert_eq!(f32::from_le_bytes(data[16..20].try_into().unwrap()), 1.5);
        assert_eq!(entries[3].constant_id, 9);
        assert_eq!(entries[3].offset, 12);
        assert_eq!(entries[4].constant_id, 7);
        assert_eq!(entries[4].offset, 16);
    }

    #[test]
    fn unknown_or_mistyped_constants_are_clear_errors() {
        let available = [c(0, 7, ConstantType::F32)];
        let e = pack_spec_data([1, 1, 1], None, &available, &[(5, ConstantValue::U32(1))])
            .unwrap_err()
            .to_string();
        assert!(e.contains("not declared"), "{e}");
        assert!(e.contains("5"), "{e}");
        let e = pack_spec_data([1, 1, 1], None, &available, &[(0, ConstantValue::U32(1))])
            .unwrap_err()
            .to_string();
        assert!(e.contains("F32") && e.contains("U32"), "{e}");
    }
}
