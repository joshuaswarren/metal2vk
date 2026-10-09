/// A compiled compute pipeline: one SPIR-V entry point with specialization applied.
pub struct Pipeline {
    _priv: (),
}

impl Pipeline {
    /// maxTotalThreadsPerThreadgroup for this pipeline's workgroup size.
    pub fn max_threads_per_threadgroup(&self) -> u32 {
        unimplemented_stub("Pipeline::max_threads_per_threadgroup")
    }

    /// Metal threadExecutionWidth (the device subgroup size).
    pub fn thread_execution_width(&self) -> u32 {
        unimplemented_stub("Pipeline::thread_execution_width")
    }

    /// Threadgroup memory the pipeline declares statically (excluding dynamic
    /// set_threadgroup_memory_length arguments), in bytes.
    pub fn static_threadgroup_memory(&self) -> u32 {
        unimplemented_stub("Pipeline::static_threadgroup_memory")
    }
}

fn unimplemented_stub<T>(what: &str) -> T {
    // temporary: bodies are filled in by the implementation commit
    let _ = what;
    unreachable!()
}
