use std::sync::Arc;

use crate::{Error, Event, Result};

/// Reported to the `on_complete` callback of [`Queue::submit`].
/// GPU start/end are device timestamps converted to nanoseconds.
pub struct Completion {
    pub error: Option<Error>,
    pub gpu_start_ns: u64,
    pub gpu_end_ns: u64,
}

/// An in-flight or recorded command buffer. Pooled per queue; reset on reuse.
pub struct CommandBuffer {
    _priv: (),
}

impl CommandBuffer {
    /// Start recording. One compute encoder per buffer is enough.
    pub fn begin(&mut self) -> Result<crate::Encoder<'_>> {
        unimplemented_stub("CommandBuffer::begin")
    }

    /// Alias of [`CommandBuffer::begin`] (a compute encoder is all this crate offers).
    pub fn compute_encoder(&mut self) -> Result<crate::Encoder<'_>> {
        self.begin()
    }

    /// Finish recording. The buffer must not be touched afterwards until
    /// the completion callback of its submit fires (or the pool is reset).
    pub fn end(&mut self) -> Result<()> {
        unimplemented_stub("CommandBuffer::end")
    }
}

/// A compute queue: owns the VkQueue, its command pool, and its waiter thread.
pub struct Queue {
    _priv: (),
}

impl Queue {
    pub fn new_command_buffer(self: &Arc<Self>) -> Result<CommandBuffer> {
        unimplemented_stub("Queue::new_command_buffer")
    }

    /// Submit command buffers in order; `on_complete` fires once, on the queue's
    /// waiter thread, after the fence (or timeline value) is signalled, with GPU
    /// start/end timestamps around the whole submission.
    pub fn submit(
        &self,
        command_buffers: &[&CommandBuffer],
        on_complete: Option<Box<dyn FnOnce(Completion) + Send>>,
    ) -> Result<()> {
        let _ = (command_buffers, on_complete);
        Err(Error::Unsupported("Queue::submit".into()))
    }

    /// Signal a timeline event from the GPU side (appended as its own tiny submit).
    pub fn signal_event(&self, event: &Event, value: u64) -> Result<()> {
        let _ = (event, value);
        Err(Error::Unsupported("Queue::signal_event".into()))
    }

    /// Make subsequent submits wait for a timeline event value (host-side tracking only
    /// until the next submit; see DESIGN.md).
    pub fn wait_event(&self, event: &Event, value: u64) -> Result<()> {
        let _ = (event, value);
        Err(Error::Unsupported("Queue::wait_event".into()))
    }
}

fn unimplemented_stub<T>(what: &str) -> T {
    // temporary: bodies are filled in by the implementation commit
    let _ = what;
    unreachable!()
}
