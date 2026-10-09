use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Weak};
use std::time::Duration;

use ash::vk;
use parking_lot::Mutex;

use crate::{DeviceInner, Error, Result};

/// A compute queue: owns the VkQueue, its command pool, and its waiter thread.
pub struct Queue {
    pub(crate) inner: Arc<QueueInner>,
}

impl std::ops::Deref for Queue {
    type Target = QueueInner;
    fn deref(&self) -> &QueueInner {
        &self.inner
    }
}

/// An in-flight or recorded command buffer. Pooled per queue; reset on reuse.
/// Cheap to clone (shared inner handle) so the encoder and the submit list can
/// both refer to it.
#[derive(Clone)]
pub struct CommandBuffer {
    pub(crate) inner: Arc<CbInner>,
}

impl std::ops::Deref for CommandBuffer {
    type Target = CbInner;
    fn deref(&self) -> &CbInner {
        &self.inner
    }
}

impl CommandBuffer {
    /// Start recording. One compute encoder per buffer is enough.
    pub fn begin(&self) -> Result<crate::Encoder> {
        let device = &self.inner.device;
        let mut st = self.inner.state.lock();
        if st.recording || st.ended {
            return Err(Error::Invalid(
                "command buffer is already recorded; take a new one from the queue".into(),
            ));
        }
        unsafe {
            device.device.begin_command_buffer(
                self.inner.raw,
                &vk::CommandBufferBeginInfo::default(),
            )?;
            device.device.cmd_reset_query_pool(self.inner.raw, self.inner.query_pool, 0, 2);
            device.device.cmd_write_timestamp(
                self.inner.raw,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.inner.query_pool,
                0,
            );
        }
        st.recording = true;
        st.encoder_open = true;
        st.ended = false;
        Ok(crate::Encoder::new(self.inner.clone()))
    }

    /// Alias of [`CommandBuffer::begin`] (a compute encoder is all this crate offers).
    pub fn compute_encoder(&self) -> Result<crate::Encoder> {
        self.begin()
    }

    /// Finish recording. The buffer must not be submitted twice; take a fresh
    /// one from the queue for the next round.
    pub fn end(&self) -> Result<()> {
        let device = &self.inner.device;
        let mut st = self.inner.state.lock();
        if st.encoder_open {
            return Err(Error::Invalid("end(): the compute encoder is still open".into()));
        }
        if !st.recording {
            return Err(Error::Invalid("end(): begin() was never called".into()));
        }
        if st.ended {
            return Err(Error::Invalid("end(): already ended".into()));
        }
        unsafe { device.device.end_command_buffer(self.inner.raw)? };
        st.ended = true;
        st.recording = false;
        Ok(())
    }
}

#[doc(hidden)]
pub struct QueueInner {
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) vk_queue: vk::Queue,
    pub(crate) pool: Mutex<vk::CommandPool>,
    pub(crate) free_cbs: Mutex<Vec<FreeCb>>,
    /// Timeline waits applied to the next submit (Queue::wait_event).
    pub(crate) pending_waits: Mutex<Vec<(vk::Semaphore, u64)>>,
    /// Pipeline variants specialised for dynamic threadgroup lengths, keyed by
    /// canonical bytes (name + constants + lengths). See Encoder::dispatch_threadgroups.
    pub(crate) variants: Mutex<HashMap<Vec<u8>, Arc<crate::Pipeline>>>,
    tx: Sender<SubmitJob>,
    waiter: Option<std::thread::JoinHandle<()>>,
}

pub(crate) struct FreeCb {
    raw: vk::CommandBuffer,
    query_pool: vk::QueryPool,
    desc_pool: Option<vk::DescriptorPool>,
}

/// Reported to the `on_complete` callback of [`Queue::submit`].
/// GPU start/end are device timestamps converted to nanoseconds.
pub struct Completion {
    pub error: Option<Error>,
    pub gpu_start_ns: u64,
    pub gpu_end_ns: u64,
}

pub(crate) struct SubmitJob {
    pub(crate) vk_queue: vk::Queue,
    pub(crate) cbs: Vec<Arc<CbInner>>,
    pub(crate) on_complete: Option<Box<dyn FnOnce(crate::Completion) + Send>>,
}

pub(crate) struct CbState {
    pub(crate) recording: bool,
    pub(crate) encoder_open: bool,
    pub(crate) ended: bool,
}

#[doc(hidden)]
pub struct CbInner {
    pub(crate) queue: Weak<QueueInner>,
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) raw: vk::CommandBuffer,
    pub(crate) query_pool: vk::QueryPool,
    /// Fallback descriptor pool when VK_KHR_push_descriptor is unavailable.
    pub(crate) desc_pool: Option<vk::DescriptorPool>,
    pub(crate) state: Mutex<CbState>,
}

impl Drop for QueueInner {
    fn drop(&mut self) {
        // tx (a field, dropped after drop()) closes the channel; join first so
        // the waiter never touches a half-dropped queue.
        if let Some(h) = self.waiter.take() {
            let _ = h.join();
        }
        let pool = *self.pool.lock();
        let frees: Vec<FreeCb> = self.free_cbs.lock().drain(..).collect();
        unsafe {
            self.device.device.destroy_command_pool(pool, None);
            for free in frees {
                self.device.device.destroy_query_pool(free.query_pool, None);
                if let Some(p) = free.desc_pool {
                    self.device.device.destroy_descriptor_pool(p, None);
                }
            }
        }
    }
}

impl Queue {
    pub(crate) fn create(device: &Arc<DeviceInner>) -> Result<Arc<Queue>> {
        unsafe {
            let pool = device.device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index: device.queue_family,
                    ..Default::default()
                },
                None,
            )?;
            let (tx, rx) = std::sync::mpsc::channel::<SubmitJob>();
            let waiter = {
                let device = device.clone();
                std::thread::Builder::new()
                    .name("m2v-queue".into())
                    .spawn(move || waiter_loop(device, rx))
                    .map_err(|e| Error::Invalid(format!("queue thread: {e}")))?
            };
            let vk_queue = device.device.get_device_queue(device.queue_family, 0);
            Ok(Arc::new(Queue {
                inner: Arc::new(QueueInner {
                    device: device.clone(),
                    vk_queue,
                    pool: Mutex::new(pool),
                    free_cbs: Mutex::new(Vec::new()),
                    pending_waits: Mutex::new(Vec::new()),
                    variants: Mutex::new(HashMap::new()),
                    tx,
                    waiter: Some(waiter),
                }),
            }))
        }
    }

    pub fn new_command_buffer(self: &Arc<Self>) -> Result<CommandBuffer> {
        let q_arc: Arc<QueueInner> = self.inner.clone();
        let q: &QueueInner = &q_arc;
        unsafe {
            let free = q.free_cbs.lock().pop();
            let (raw, query_pool, desc_pool) = match free {
                Some(f) => {
                    q.device
                        .device
                        .reset_command_buffer(f.raw, vk::CommandBufferResetFlags::RELEASE_RESOURCES)?;
                    if let Some(p) = f.desc_pool {
                        q.device.device.reset_descriptor_pool(
                            p,
                            vk::DescriptorPoolResetFlags::empty(),
                        )?;
                    }
                    (f.raw, f.query_pool, f.desc_pool)
                }
                None => {
                    let raw = q
                        .device
                        .device
                        .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                            command_pool: *q.pool.lock(),
                            level: vk::CommandBufferLevel::PRIMARY,
                            command_buffer_count: 1,
                            ..Default::default()
                        })?[0];
                    let query_pool = q.device.device.create_query_pool(
                        &vk::QueryPoolCreateInfo {
                            query_type: vk::QueryType::TIMESTAMP,
                            query_count: 2,
                            ..Default::default()
                        },
                        None,
                    )?;
                    // Fallback only: each recycled command buffer owns a pool
                    // sized for several encoders. Created whenever dispatch
                    // will use plain descriptor sets (no extension, or the
                    // M2V_NO_PUSH_DESCRIPTOR override).
                    let desc_pool =
                        (!crate::encoder::use_push_descriptors(q.device.as_ref())).then(|| {
                        let sizes = [vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 256,
                        }];
                        q.device
                            .device
                            .create_descriptor_pool(
                                &vk::DescriptorPoolCreateInfo {
                                    max_sets: 64,
                                    pool_size_count: 1,
                                    p_pool_sizes: sizes.as_ptr(),
                                    ..Default::default()
                                },
                                None,
                            )
                            .expect("descriptor pool")
                    });
                    (raw, query_pool, desc_pool)
                }
            };
            Ok(CommandBuffer {
                inner: Arc::new(CbInner {
                    queue: Arc::downgrade(&q_arc),
                    device: q.device.clone(),
                    raw,
                    query_pool,
                    desc_pool,
                    state: Mutex::new(CbState {
                        recording: false,
                        encoder_open: false,
                        ended: false,
                    }),
                }),
            })
        }
    }

    pub fn submit(
        &self,
        command_buffers: &[&CommandBuffer],
        on_complete: Option<Box<dyn FnOnce(crate::Completion) + Send>>,
    ) -> Result<()> {
        if command_buffers.is_empty() {
            return Err(Error::Invalid("submit of zero command buffers".into()));
        }
        for cb in command_buffers {
            let st = cb.inner.state.lock();
            if st.recording || st.encoder_open || !st.ended {
                return Err(Error::Invalid(
                    "submit: command buffer is not ended (call end() first)".into(),
                ));
            }
        }
        let job = SubmitJob {
            vk_queue: self.inner.vk_queue,
            cbs: command_buffers.iter().map(|c| c.inner.clone()).collect(),
            on_complete,
        };
        self.inner
            .tx
            .send(job)
            .map_err(|_| Error::Invalid("queue thread is gone".into()))
    }

    /// Signal a timeline event from the GPU side: a tiny empty submit that
    /// signals at `value`, so GPU-ordered consumers observe it in order.
    pub fn signal_event(&self, event: &Event, value: u64) -> Result<()> {
        let q: &QueueInner = &self.inner;
        unsafe {
            let fence = q
                .device
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            let signal = [vk::SemaphoreSubmitInfo::default()
                .semaphore(event.semaphore)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let infos = [vk::SubmitInfo2::default().signal_semaphore_infos(&signal)];
            let res = q.device.device.queue_submit2(q.vk_queue, &infos, fence);
            if let Err(e) = res {
                q.device.device.destroy_fence(fence, None);
                return Err(e.into());
            }
            let wait = q.device.device.wait_for_fences(&[fence], true, q.device.submit_seconds_ns());
            q.device.device.destroy_fence(fence, None);
            wait?;
        }
        Ok(())
    }

    /// Make the NEXT submit on this queue wait until the event reaches `value`
    /// (Metal-style: the wait is queued on the GPU timeline, not host-blocking).
    pub fn wait_event(&self, event: &Event, value: u64) -> Result<()> {
        self.inner
            .pending_waits
            .lock()
            .push((event.semaphore, value));
        Ok(())
    }
}

/// One waiter thread per queue: submits serially, waits the fence, converts the
/// timestamps, recycles the command buffers, then runs the completion callback.
fn waiter_loop(device: Arc<DeviceInner>, rx: Receiver<SubmitJob>) {
    while let Ok(job) = rx.recv() {
        let queue = job.cbs[0].queue.upgrade();
        let deadline_ns = device.submit_seconds_ns();
        let fence = unsafe { device.device.create_fence(&vk::FenceCreateInfo::default(), None) };
        let fence = match fence {
            Ok(f) => f,
            Err(e) => {
                finish_with_error(job, e.into());
                continue;
            }
        };
        let waits: Vec<(vk::Semaphore, u64)> = queue
            .as_ref()
            .map(|q| std::mem::take(&mut *q.pending_waits.lock()))
            .unwrap_or_default();
        let raws: Vec<vk::CommandBuffer> = job.cbs.iter().map(|c| c.raw).collect();
        let wait_infos: Vec<vk::SemaphoreSubmitInfo> = waits
            .iter()
            .map(|(s, v)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*s)
                    .value(*v)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        let cb_infos: Vec<vk::CommandBufferSubmitInfo> = raws
            .iter()
            .map(|c| vk::CommandBufferSubmitInfo::default().command_buffer(*c))
            .collect();
        let submit_infos = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&cb_infos)];
        let submit_result =
            unsafe { device.device.queue_submit2(job.vk_queue, &submit_infos, fence) };
        if let Err(e) = submit_result {
            unsafe { device.device.destroy_fence(fence, None) };
            finish_with_error(job, e.into());
            continue;
        }
        let outcome = unsafe { device.device.wait_for_fences(&[fence], true, deadline_ns) };
        let error = match outcome {
            Ok(()) => None,
            Err(vk::Result::TIMEOUT) => {
                // Standing rule: every submit must finish inside the limit.
                #[cfg(debug_assertions)]
                eprintln!(
                    "m2v: submit exceeded the {:.1}s limit (M2V_MAX_SUBMIT_MS); waiting idle",
                    deadline_ns as f64 / 1e9
                );
                let _ = unsafe { device.device.device_wait_idle() };
                Some(Error::Invalid(format!(
                    "submit exceeded M2V_MAX_SUBMIT_MS ({:.1}s)",
                    deadline_ns as f64 / 1e9
                )))
            }
            Err(e) => Some(e.into()),
        };
        unsafe { device.device.destroy_fence(fence, None) };

        // GPU timestamps: TOP of the first buffer to BOTTOM of the last.
        let (gpu_start_ns, gpu_end_ns) =
            if error.is_none() { unsafe { read_span_ns(&device, &job.cbs) } } else { (0, 0) };

        if let Some(q) = queue {
            recycle(&q, job.cbs);
        }
        if let Some(cb) = job.on_complete {
            cb(crate::Completion { error, gpu_start_ns, gpu_end_ns });
        }
    }
}

fn finish_with_error(job: SubmitJob, error: Error) {
    if let Some(cb) = job.on_complete {
        cb(crate::Completion { error: Some(error), gpu_start_ns: 0, gpu_end_ns: 0 });
    }
}

unsafe fn read_span_ns(device: &DeviceInner, cbs: &[Arc<CbInner>]) -> (u64, u64) {
    let read = |pool: vk::QueryPool, idx: u32| -> u64 {
        let mut v = [0u64; 1];
        let _ = device.device.get_query_pool_results(
            pool,
            idx,
            &mut v,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        );
        v[0]
    };
    let start = read(cbs[0].query_pool, 0);
    let end = read(cbs[cbs.len() - 1].query_pool, 1);
    let p = device.timestamp_period as f64;
    ((start as f64 * p) as u64, (end as f64 * p) as u64)
}

fn recycle(queue: &QueueInner, cbs: Vec<Arc<CbInner>>) {
    let mut free = queue.free_cbs.lock();
    for cb in cbs {
        free.push(FreeCb {
            raw: cb.raw,
            query_pool: cb.query_pool,
            desc_pool: cb.desc_pool,
        });
    }
}

/// Timeline semaphore. Values are monotonic; `wait` blocks the host.
pub struct Event {
    pub(crate) device: Weak<DeviceInner>,
    pub(crate) semaphore: vk::Semaphore,
}

impl Event {
    pub(crate) fn create(device: &Arc<DeviceInner>) -> Arc<Event> {
        let semaphore = unsafe {
            let mut ty = vk::SemaphoreTypeCreateInfo {
                semaphore_type: vk::SemaphoreType::TIMELINE,
                initial_value: 0,
                ..Default::default()
            };
            device.device.create_semaphore(
                &vk::SemaphoreCreateInfo {
                    p_next: &mut ty as *mut _ as *mut std::ffi::c_void,
                    ..Default::default()
                },
                None,
            )
        }
        .expect("timeline semaphore");
        Arc::new(Event { device: Arc::downgrade(device), semaphore })
    }

    /// Last signalled value (0 if never signalled).
    pub fn signaled_value(&self) -> Result<u64> {
        let device = self
            .device
            .upgrade()
            .ok_or_else(|| Error::Invalid("device is gone".into()))?;
        unsafe { Ok(device.device.get_semaphore_counter_value(self.semaphore)?) }
    }

    /// Block until the semaphore reaches `value` or `timeout` elapses.
    pub fn wait(&self, value: u64, timeout: Duration) -> Result<()> {
        let device = self
            .device
            .upgrade()
            .ok_or_else(|| Error::Invalid("device is gone".into()))?;
        unsafe {
            device.device.wait_semaphores(
                &vk::SemaphoreWaitInfo {
                    semaphore_count: 1,
                    p_semaphores: &self.semaphore,
                    p_values: &value,
                    ..Default::default()
                },
                timeout.as_nanos() as u64,
            )?;
        }
        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Some(device) = self.device.upgrade() {
            unsafe { device.device.destroy_semaphore(self.semaphore, None) };
        }
    }
}
