use std::time::Duration;
use std::sync::Arc;

use crate::{Error, Library, Queue, Result};

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
    _priv: (),
}

impl TimestampPool {
    /// `None` for slots never written or still pending.
    pub fn results(&self) -> Result<Vec<Option<u64>>> {
        unimplemented_stub("TimestampPool::results")
    }
}

/// The first Vulkan device with a compute queue (env `M2V_DEVICE=<index>` selects).
pub struct Device {
    _priv: (),
}

impl Device {
    pub fn new() -> Result<Arc<Device>> {
        Err(Error::Unsupported("Device::new".into()))
    }

    pub fn name(&self) -> &str {
        unimplemented_stub("Device::name")
    }

    pub fn info(&self) -> &DeviceInfo {
        unimplemented_stub("Device::info")
    }

    /// Every submit this shim issues must finish under this many seconds
    /// (`M2V_MAX_SUBMIT_MS`, default 5). Enforced as an abort in debug builds only.
    pub fn limits_gpu_submit_seconds(&self) -> f64 {
        5.0
    }

    pub fn create_pipeline(
        &self,
        library: &Arc<Library>,
        name: &str,
        constants: &[(u32, crate::ConstantValue)],
    ) -> Result<Arc<crate::Pipeline>> {
        let _ = (library, name, constants);
        Err(Error::Unsupported("create_pipeline".into()))
    }

    pub fn create_queue(&self) -> Result<Arc<Queue>> {
        Err(Error::Unsupported("create_queue".into()))
    }

    /// Timeline-semaphore backed event (Vulkan 1.2).
    pub fn create_event(self: &Arc<Self>) -> Arc<Event> {
        Arc::new(Event { _priv: () })
    }

    pub fn create_timestamp_pool(&self, count: u32) -> Result<TimestampPool> {
        let _ = count;
        Err(Error::Unsupported("create_timestamp_pool".into()))
    }
}

impl Default for Device {
    fn default() -> Self {
        Device { _priv: () }
    }
}

fn unimplemented_stub<T>(what: &str) -> T {
    // temporary: bodies are filled in by the implementation commit
    let _ = what;
    unreachable!()
}


/// Timeline semaphore. Values are monotonic; `wait` blocks the host.
pub struct Event {
    pub(crate) _priv: (),
}

impl Event {
    /// Last signalled value (0 if never signalled).
    pub fn signaled_value(&self) -> Result<u64> {
        unimplemented_stub("Event::signaled_value")
    }

    /// Block until the semaphore reaches `value` or `timeout` elapses.
    pub fn wait(&self, value: u64, timeout: Duration) -> Result<()> {
        let _ = (value, timeout);
        unimplemented_stub("Event::wait")
    }
}

