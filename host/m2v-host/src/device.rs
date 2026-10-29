use std::sync::Arc;

use ash::vk;

use crate::addrmap::AddressMap;
use crate::{Buffer, Error, Event, Library, Pipeline, Queue, Result};

/// Everything shared between handles. `Device` is the public wrapper.
#[doc(hidden)]
pub struct DeviceInner {
    pub(crate) _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) queue_family: u32,
    pub(crate) info: DeviceInfo,
    pub(crate) timestamp_period: f32,
    pub(crate) max_storage_buffer_range: u64,
    pub(crate) max_push_constants_size: u32,
    pub(crate) has_push_descriptor: bool,
    pub(crate) has_debug_utils: bool,
    /// Loader for VK_KHR_push_descriptor; `Some` iff the extension is enabled.
    pub(crate) push_descriptor: Option<ash::khr::push_descriptor::Device>,
    pub(crate) debug_utils: ash::ext::debug_utils::Device,
    /// Validation messenger, kept alive for the life of the device.
    #[allow(dead_code)]
    pub(crate) debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub(crate) map: Arc<AddressMap>,
}

impl std::ops::Deref for Device {
    type Target = DeviceInner;
    fn deref(&self) -> &DeviceInner {
        &self.inner
    }
}

/// Validation messages go to stderr, prefixed so a test can find them.
unsafe extern "system" fn validation_callback(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !data.is_null() && !(*data).p_message.is_null() {
        let text = std::ffi::CStr::from_ptr((*data).p_message).to_string_lossy();
        eprintln!("m2v-validation: {text}");
    }
    vk::FALSE
}

/// The C pointer array Vulkan wants for extension names. A `&[&CStr]` cannot be
/// passed directly: `&CStr` is a fat pointer (address and length), so reading it
/// as `*const c_char` entries makes every second name the length of the first.
fn c_string_ptrs(names: &[&std::ffi::CStr]) -> Vec<*const std::ffi::c_char> {
    names.iter().map(|n| n.as_ptr()).collect()
}

#[cfg(test)]
mod tests {
    use super::c_string_ptrs;

    #[test]
    fn extension_name_pointers_are_one_per_name() {
        let names = [c"VK_KHR_cooperative_matrix", c"VK_KHR_push_descriptor"];
        let ptrs = c_string_ptrs(&names);
        assert_eq!(ptrs.len(), 2);
        for (ptr, name) in ptrs.iter().zip(names) {
            assert_eq!(unsafe { std::ffi::CStr::from_ptr(*ptr) }, name);
        }
    }
}

/// Static device properties, best effort (`0`/`false` when unknown).
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Subgroup size the driver guarantees for compute (Metal: threadExecutionWidth).
    pub subgroup_size: u32,
    /// maxTotalThreadsPerThreadgroup.
    pub max_threads_per_threadgroup: u32,
    /// Max threadgroup (shared) memory in bytes.
    pub max_shared_memory: u32,
    /// VK_KHR_cooperative_matrix available and enabled.
    pub has_coopmat: bool,
    /// Timeline semaphores available (Vulkan 1.2 core feature bit).
    pub has_timeline: bool,
    /// Total device-local+host-visible heap memory in bytes (best effort).
    pub total_memory: u64,
    /// GPU core count when the driver exposes it (best effort: 0 if unknown).
    pub gpu_core_count: u32,
}

/// Query pool for GPU timestamps; results are nanoseconds (timestampPeriod applied).
pub struct TimestampPool {
    pub(crate) device: ash::Device,
    pub(crate) pool: vk::QueryPool,
    pub(crate) count: u32,
    pub(crate) period_ns: f32,
}

impl std::ops::Deref for TimestampPool {
    type Target = vk::QueryPool;
    fn deref(&self) -> &vk::QueryPool {
        &self.pool
    }
}

/// The first Vulkan device with a compute queue (env `M2V_DEVICE=<index>` selects).
pub struct Device {
    pub(crate) inner: Arc<DeviceInner>,
    /// Physical device name; queried lazily, once.
    name_str: std::sync::OnceLock<String>,
    policy: crate::Policy,
}

impl Device {
    pub(crate) fn inner(&self) -> &Arc<DeviceInner> {
        &self.inner
    }

    /// First Vulkan 1.3 device with a compute queue; `M2V_DEVICE=<index>` selects
    /// among the compute-capable physical devices. Instance/device features follow
    /// `m2v-run.c`: enable everything the device offers that the translated
    /// kernels can use.
    pub fn new() -> Result<Arc<Device>> {
        let wanted: u32 = std::env::var("M2V_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        unsafe { Device::open(wanted) }
    }

    // The feature structs are walked through raw p_next pointers by
    // vkCreateDevice, so rustc cannot see that the field writes are read.
    #[allow(unused_assignments)]
    unsafe fn open(wanted: u32) -> Result<Arc<Device>> {
        use ash::ext;
        use ash::khr;

        let entry = ash::Entry::load().map_err(|e| Error::Invalid(format!("vulkan loader: {e}")))?;
        let instance_exts = available_instance_exts(&entry)?;
        let debug_utils_flag = instance_exts.iter().any(|e| e.as_c_str() == ext::debug_utils::NAME);
        // M2V_VALIDATION=1 turns on the Khronos validation layer; its messages
        // go to stderr with a "m2v-validation:" prefix.
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        let validation_flag = std::env::var_os("M2V_VALIDATION").is_some()
            && debug_utils_flag
            && entry
                .enumerate_instance_layer_properties()
                .map(|props| {
                    props.iter().any(|p| p.layer_name_as_c_str().is_ok_and(|name| name == validation_layer))
                })
                .unwrap_or(false);
        let layer_names = [validation_layer.as_ptr()];
        let instance_ext_names = [ext::debug_utils::NAME.as_ptr()];
        let create = vk::InstanceCreateInfo {
            p_application_info: &vk::ApplicationInfo {
                api_version: vk::API_VERSION_1_3,
                ..Default::default()
            },
            enabled_extension_count: u32::try_from(if debug_utils_flag { 1 } else { 0 }).unwrap(),
            pp_enabled_extension_names: if debug_utils_flag {
                instance_ext_names.as_ptr()
            } else {
                std::ptr::null()
            },
            enabled_layer_count: u32::try_from(if validation_flag { 1 } else { 0 }).unwrap(),
            pp_enabled_layer_names: if validation_flag {
                layer_names.as_ptr()
            } else {
                std::ptr::null()
            },
            ..Default::default()
        };
        let instance = entry.create_instance(&create, None)?;
        let debug_messenger = if validation_flag {
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(validation_callback));
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None) }.ok();
            eprintln!("m2v: {} enabled", validation_layer.to_string_lossy());
            messenger
        } else {
            None
        };

        let physicals = instance.enumerate_physical_devices()?;
        let mut candidates = Vec::new();
        for ph in physicals {
            if instance
                .get_physical_device_queue_family_properties(ph)
                .iter()
                .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                candidates.push(ph);
            }
        }
        let physical_device = *candidates
            .get(wanted as usize)
            .ok_or_else(|| Error::Invalid(format!("M2V_DEVICE={wanted}: only {} compute-capable device(s)", candidates.len())))?;

        // Feature chain: query what exists, enable the pieces the kernels need.
        let mut cm_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f11 = vk::PhysicalDeviceVulkan11Features {
            p_next: &mut f12 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        f12.p_next = &mut f13 as *mut _ as *mut std::ffi::c_void;
        f13.p_next = &mut cm_features as *mut _ as *mut std::ffi::c_void;
        let mut f2 = vk::PhysicalDeviceFeatures2 {
            p_next: &mut f11 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        instance.get_physical_device_features2(physical_device, &mut f2);

        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2 {
            p_next: &mut p11 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        instance.get_physical_device_properties2(physical_device, &mut props2);

        let dev_exts_avail: std::collections::HashSet<Vec<u8>> = instance
            .enumerate_device_extension_properties(physical_device)?
            .iter()
            .filter_map(|p| p.extension_name_as_c_str().ok().map(|s| s.to_bytes().to_vec()))
            .collect();
        let has_coopmat = dev_exts_avail.contains(khr::cooperative_matrix::NAME.to_bytes())
            && cm_features.cooperative_matrix == vk::TRUE;
        let has_push_descriptor = dev_exts_avail.contains(khr::push_descriptor::NAME.to_bytes());
        let mut enabled_exts: Vec<&std::ffi::CStr> = Vec::new();
        if has_coopmat {
            enabled_exts.push(khr::cooperative_matrix::NAME);
        }
        if has_push_descriptor {
            enabled_exts.push(khr::push_descriptor::NAME);
        }

        f11.storage_buffer16_bit_access = f11.storage_buffer16_bit_access.min(vk::TRUE);
        f11.uniform_and_storage_buffer16_bit_access = f11.uniform_and_storage_buffer16_bit_access.min(vk::TRUE);
        f11.variable_pointers = f11.variable_pointers.min(vk::TRUE);
        f12.storage_buffer8_bit_access = f12.storage_buffer8_bit_access.min(vk::TRUE);
        f12.uniform_and_storage_buffer8_bit_access = f12.uniform_and_storage_buffer8_bit_access.min(vk::TRUE);
        f12.shader_float16 = f12.shader_float16.min(vk::TRUE);
        f12.shader_int8 = f12.shader_int8.min(vk::TRUE);
        f12.shader_subgroup_extended_types = f12.shader_subgroup_extended_types.min(vk::TRUE);
        f12.timeline_semaphore = f12.timeline_semaphore.min(vk::TRUE);
        f12.buffer_device_address = f12.buffer_device_address.min(vk::TRUE);
        f12.scalar_block_layout = f12.scalar_block_layout.min(vk::TRUE);
        f2.features.shader_storage_image_write_without_format =
            f2.features.shader_storage_image_write_without_format.min(vk::TRUE);
        f13.subgroup_size_control = f13.subgroup_size_control.min(vk::TRUE);
        f13.compute_full_subgroups = f13.compute_full_subgroups.min(vk::TRUE);
        f13.shader_demote_to_helper_invocation = f13.shader_demote_to_helper_invocation.min(vk::TRUE);
        f13.maintenance4 = f13.maintenance4.min(vk::TRUE);
        cm_features.cooperative_matrix = cm_features.cooperative_matrix.min(vk::TRUE);
        let enabled_ext_ptrs = c_string_ptrs(&enabled_exts);

        let queue_family = instance
            .get_physical_device_queue_family_properties(physical_device)
            .iter()
            .enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .expect("filtered above");
        let prio = [1.0f32];
        let queue_create = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: prio.as_ptr(),
            ..Default::default()
        };
        let device_create = vk::DeviceCreateInfo {
            p_next: &mut f2 as *mut _ as *mut std::ffi::c_void,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create,
            enabled_extension_count: enabled_ext_ptrs.len() as u32,
            pp_enabled_extension_names: enabled_ext_ptrs.as_ptr(),
            ..Default::default()
        };
        let device = instance.create_device(physical_device, &device_create, None)?;

        let heaps = instance.get_physical_device_memory_properties(physical_device);
        let total_memory: u64 = heaps.memory_heaps[..heaps.memory_heap_count as usize]
            .iter()
            .map(|h| h.size)
            .sum();

        let info = DeviceInfo {
            subgroup_size: p11.subgroup_size,
            max_threads_per_threadgroup: props2.properties.limits.max_compute_work_group_invocations,
            max_shared_memory: props2.properties.limits.max_compute_shared_memory_size,
            has_coopmat,
            has_timeline: f12.timeline_semaphore == vk::TRUE,
            total_memory,
            gpu_core_count: 0, // no portable query on this driver yet
        };

        let debug_utils = ash::ext::debug_utils::Device::new(&instance, &device);
        let push_descriptor =
            has_push_descriptor.then(|| ash::khr::push_descriptor::Device::new(&instance, &device));
        let inner = Arc::new(DeviceInner {
            _entry: entry,
            debug_utils,
            debug_messenger,
            push_descriptor,
            instance,
            physical_device,
            device,
            queue_family,
            info,
            timestamp_period: props2.properties.limits.timestamp_period,
            max_storage_buffer_range: props2.properties.limits.max_storage_buffer_range as u64,
            max_push_constants_size: props2.properties.limits.max_push_constants_size,
            has_push_descriptor,
            has_debug_utils: debug_utils_flag,
            map: Arc::new(AddressMap::new()),
        });
        Ok(Arc::new(Device { inner, name_str: std::sync::OnceLock::new(), policy: crate::Policy::from_env()? }))
    }

    /// Kernel routing policy of this device (gate table from `M2V_POLICY_FILE`, fallbacks, first-use checks).
    pub fn policy(&self) -> &crate::Policy {
        &self.policy
    }

    /// How to run `kernel` from `lib`: translated, translated with a first-use check, a fallback, or `Error::Refused`.
    /// Call this before creating the pipeline; only [`crate::Route::Translated`] and
    /// [`crate::Route::TranslatedChecked`] may be followed by `create_pipeline`.
    pub fn route_kernel(&self, lib: &crate::Library, kernel: &str) -> Result<crate::Route> {
        let func = lib.function(kernel).ok();
        // the library's own gate seeds the table unless the policy file or an earlier check already decided
        if let Some(f) = func {
            if !self.policy.knows(kernel) && f.gate.state != crate::GateState::Unverified {
                self.policy.set_state(kernel, f.gate.state, &f.gate.evidence);
            }
        }
        self.policy.route(kernel, func.is_some())
    }

    pub fn name(&self) -> &str {
        self.name_str.get_or_init(|| {
            let props = unsafe { self.instance.get_physical_device_properties(self.physical_device) };
            props
                .device_name_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Every submit this shim issues must finish under this many seconds
    /// (`M2V_MAX_SUBMIT_MS`, default 5000).
    pub fn limits_gpu_submit_seconds(&self) -> f64 {
        self.inner.limits_gpu_submit_seconds()
    }

    /// Unified host-visible coherent allocation, persistently mapped.
    pub fn create_buffer(&self, size: usize) -> Result<Arc<Buffer>> {
        Buffer::allocate(self.inner(), size)
    }

    /// Interval-map lookup: which live buffer owns `addr`, and the offset into it.
    pub fn resolve(&self, addr: u64) -> Option<(Arc<Buffer>, u64)> {
        self.map.resolve(addr)
    }

    pub fn create_pipeline(
        &self,
        library: &Arc<Library>,
        name: &str,
        constants: &[(u32, crate::ConstantValue)],
    ) -> Result<Arc<Pipeline>> {
        // The gate: only a verified kernel, or one that will be checked against a reference on its first use, gets a
        // pipeline. A fallback route is an error here; callers that want the fallback ask `route_kernel` first.
        match self.route_kernel(library, name)? {
            crate::Route::Translated | crate::Route::TranslatedChecked { .. } => {
                Pipeline::create(self.inner(), library, name, constants, &[])
            }
            crate::Route::Fallback(fb) => Err(Error::Refused(format!(
                "kernel '{name}' is routed to fallback {fb:?}; dispatch the fallback, no translated pipeline is created"
            ))),
        }
    }

    pub fn create_queue(&self) -> Result<Arc<Queue>> {
        Queue::create(self.inner())
    }

    /// Timeline-semaphore backed event (Vulkan 1.2).
    pub fn create_event(self: &Arc<Self>) -> Arc<Event> {
        Event::create(self.inner())
    }

    pub fn create_timestamp_pool(&self, count: u32) -> Result<TimestampPool> {
        unsafe {
            let pool = self
                .device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo {
                        query_type: vk::QueryType::TIMESTAMP,
                        query_count: count,
                        ..Default::default()
                    },
                    None,
                )?;
            Ok(TimestampPool {
                device: self.device.clone(),
                pool,
                count,
                period_ns: self.timestamp_period,
            })
        }
    }
}

impl DeviceInner {
    /// Every submit this shim issues must finish under this many seconds
    /// (`M2V_MAX_SUBMIT_MS`, default 5000).
    pub(crate) fn limits_gpu_submit_seconds(&self) -> f64 {
        std::env::var("M2V_MAX_SUBMIT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|ms| ms as f64 / 1000.0)
            .unwrap_or(5.0)
    }

    pub(crate) fn submit_seconds_ns(&self) -> u64 {
        (self.limits_gpu_submit_seconds() * 1e9) as u64
    }

    pub(crate) fn find_host_memory_type(&self, flags: vk::MemoryRequirements) -> Result<u32> {
        let props = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut best = None;
        for i in 0..props.memory_type_count {
            let t = props.memory_types[i as usize];
            if flags.memory_type_bits & (1 << i) == 0 || !t.property_flags.contains(wanted) {
                continue;
            }
            let cached = t.property_flags.contains(vk::MemoryPropertyFlags::HOST_CACHED);
            if cached {
                return Ok(i); // cached beats uncached on unified memory
            }
            best = best.or(Some(i));
        }
        best.ok_or_else(|| Error::Unsupported("no host-visible coherent memory type".into()))
    }
}

fn available_instance_exts(entry: &ash::Entry) -> Result<Vec<std::ffi::CString>> {
    Ok(unsafe { entry
        .enumerate_instance_extension_properties(None)?
        .iter()
        .filter_map(|p| p.extension_name_as_c_str().ok().map(|s| s.to_owned()))
        .collect()
    })
}

impl TimestampPool {
    /// `None` for slots never written. Values in ns (timestampPeriod applied).
    pub fn results(&self) -> Result<Vec<Option<u64>>> {
        let mut raw = vec![0u64; self.count as usize];
        unsafe {
            self.device
                .get_query_pool_results(self.pool, 0, &mut raw, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
        }
        Ok(raw
            .into_iter()
            .map(|v| if v == 0 { None } else { Some((v as f64 * self.period_ns as f64) as u64) })
            .collect())
    }
}
