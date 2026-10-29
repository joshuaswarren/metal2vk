//! Shared helpers for the GPU tests: fixture loading, the `M2V_GPU` gate,
//! submit-and-wait, and the child-process runner the bounded tests use.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use metal2vk::{CommandBuffer, Completion, Device, Library, Queue};

/// The GPU tests only run where a Vulkan device is wanted; the build host skips them.
pub fn gpu_enabled() -> bool {
    std::env::var("M2V_GPU").as_deref() == Ok("1")
}

/// True inside a child process spawned by [`run_child`].
pub fn is_child(name: &str) -> bool {
    std::env::var("M2V_CHILD").as_deref() == Ok(name)
}

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

pub fn load(device: &Arc<Device>, name: &str) -> Arc<Library> {
    let path = fixture(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    Library::from_bytes(device, &bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Submit and wait for the one completion the queue delivers.
pub fn submit_and_wait(queue: &Arc<Queue>, cbs: &[&CommandBuffer], timeout: Duration) -> Completion {
    let (tx, rx) = mpsc::channel();
    queue
        .submit(cbs, Some(Box::new(move |completion| {
            let _ = tx.send(completion);
        })))
        .expect("submit");
    rx.recv_timeout(timeout).expect("completion callback")
}

/// Fill a float buffer from the CPU.
pub fn fill_f32(buffer: &Arc<metal2vk::Buffer>, values: &[f32]) {
    assert_eq!(values.len() * 4, buffer.len(), "buffer size vs values");
    unsafe {
        let p = buffer.contents() as *mut f32;
        for (i, v) in values.iter().enumerate() {
            *p.add(i) = *v;
        }
    }
}

/// Read a float buffer back.
pub fn read_f32(buffer: &Arc<metal2vk::Buffer>, len: usize) -> Vec<f32> {
    unsafe {
        let p = buffer.contents() as *const f32;
        (0..len).map(|i| *p.add(i)).collect()
    }
}

/// What a child process left behind.
pub struct ChildOutcome {
    pub success: bool,
    pub timed_out: bool,
    pub output: String,
}

/// Run one test of this binary in a child process with a parent-enforced deadline.
/// The child gets `M2V_GPU=1` and the extra environment; everything else is inherited.
pub fn run_child(test_name: &str, env: &[(&str, &str)], deadline: Duration) -> ChildOutcome {
    let exe = std::env::current_exe().expect("current_exe");
    let log = std::env::temp_dir().join(format!(
        "m2v-child-{test_name}-{}-{}.log",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let file = std::fs::File::create(&log).expect("child log");
    let err_file = file.try_clone().expect("child log clone");
    let mut cmd = Command::new(exe);
    cmd.arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("M2V_GPU", "1")
        .env("M2V_CHILD", test_name)
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(err_file));
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn child");
    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => break Some(status),
            None => {
                if start.elapsed() > deadline {
                    timed_out = true;
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    };
    let output = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    // `--exact` with a name that matches nothing exits 0 with "0 passed".
    let ran_one = output.contains("1 passed");
    let success = ran_one && matches!(status, Some(s) if s.success());
    ChildOutcome { success, timed_out, output }
}
