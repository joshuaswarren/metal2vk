//! Bounded queue-lifecycle tests (issue #95).
//!
//! Every case runs in its own child process with a parent-enforced deadline:
//! the queue teardown joins its waiter thread, and a failure that never
//! finishes must show up as a killed child, not as a passing assertion. The
//! child has to exit normally (exit status 0) for the case to pass.

mod common;

use std::time::{Duration, Instant};

use metal2vk::{Completion, Device, Fallback};

const DEADLINE: Duration = Duration::from_secs(120);
const SUBMIT_LIMIT_MS: &str = "1500";

/// Create a queue and drop it again, with nothing submitted.
#[test]
fn child_empty_queue_drop() {
    if !common::is_child("child_empty_queue_drop") {
        return;
    }
    let device = Device::new().expect("device");
    let queue = device.create_queue().expect("queue");
    drop(queue);
    drop(device);
}

/// One submit, exactly one completion, then every handle released.
#[test]
fn child_one_add_one_completion() {
    if !common::is_child("child_one_add_one_completion") {
        return;
    }
    const N: usize = 256;
    let device = Device::new().expect("device");
    let library = common::load(&device, "descriptors.m2vlib");
    device
        .policy()
        .set_fallback("add_scalar", Fallback::Cpu("cpu_add_scalar".into()));
    let queue = device.create_queue().expect("queue");
    let a = device.create_buffer(N * 4).expect("buffer a");
    let b = device.create_buffer(N * 4).expect("buffer b");
    let values: Vec<f32> = (0..N).map(|i| i as f32).collect();
    common::fill_f32(&a, &values);

    let pipeline = device.create_pipeline(&library, "add_scalar", &[]).expect("pipeline");
    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let mut enc = cb.begin().expect("encoder");
        enc.set_pipeline(&pipeline).expect("set pipeline");
        enc.set_address(0, a.gpu_address()).expect("set address 0");
        enc.set_address(1, b.gpu_address()).expect("set address 1");
        enc.set_bytes(2, &0.0f32.to_le_bytes()).expect("set bytes 2");
        enc.dispatch_threadgroups([4, 1, 1], [64, 1, 1]).expect("dispatch");
        enc.end().expect("encoder end");
    }
    cb.end().expect("command buffer end");

    let completion = common::submit_and_wait(&queue, &[&cb], Duration::from_secs(60));
    assert!(completion.error.is_none(), "GPU error: {:?}", completion.error);
    let out = common::read_f32(&b, N);
    for i in 0..N {
        assert!((out[i] - values[i]).abs() < 1e-6, "mismatch at {i}: got {}", out[i]);
    }

    drop(pipeline);
    drop(cb);
    drop(a);
    drop(b);
    drop(library);
    drop(queue);
    drop(device);
}

/// A wait that nothing will ever satisfy: the submit must fail with a named
/// error, report no successful timestamps, and tear down cleanly.
#[test]
fn child_submit_failure_error_completion() {
    if !common::is_child("child_submit_failure_error_completion") {
        return;
    }
    let device = Device::new().expect("device");
    let event = device.create_event();
    let queue = device.create_queue().expect("queue");
    // Nothing signals this event: the next submit waits for a value that never comes.
    queue.wait_event(&event, 1).expect("wait event");

    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let enc = cb.begin().expect("encoder");
        enc.end().expect("encoder end");
    }
    cb.end().expect("command buffer end");

    let completion = common::submit_and_wait(&queue, &[&cb], Duration::from_secs(60));
    let error = completion.error.expect("an error completion");
    let message = error.to_string();
    assert!(
        message.contains("M2V_MAX_SUBMIT_MS"),
        "named timeout outcome expected, got: {message}"
    );
    assert_eq!(
        (completion.gpu_start_ns, completion.gpu_end_ns),
        (0, 0),
        "a failed submit must not report successful timestamps"
    );

    drop(cb);
    drop(event);
    drop(queue);
    drop(device);
}

/// The same unsatisfied wait, but the whole child must stay bounded: the
/// failure has to arrive inside the configured submit limit, not whenever the
/// teardown happens to finish.
#[test]
fn child_unsatisfied_timeline_wait_bounded() {
    if !common::is_child("child_unsatisfied_timeline_wait_bounded") {
        return;
    }
    let start = Instant::now();
    let device = Device::new().expect("device");
    let event = device.create_event();
    let queue = device.create_queue().expect("queue");
    queue.wait_event(&event, 7).expect("wait event");

    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let enc = cb.begin().expect("encoder");
        enc.end().expect("encoder end");
    }
    cb.end().expect("command buffer end");

    let completion = common::submit_and_wait(&queue, &[&cb], Duration::from_secs(60));
    assert!(completion.error.is_some(), "the wait can never be satisfied");
    let elapsed = start.elapsed();
    let limit = Duration::from_millis(SUBMIT_LIMIT_MS.parse().expect("limit"));
    assert!(
        elapsed < limit * 3,
        "the failure took {elapsed:?}, far past the {limit:?} submit limit"
    );

    drop(cb);
    drop(event);
    drop(queue);
    drop(device);
}

/// A queue dropped while its submit is still in flight: the waiter thread
/// ends up as the last owner of the queue state, and the teardown must not
/// try to join the waiter from the waiter itself. The child must exit
/// normally, which also proves the pools were destroyed without incident.
///
/// Releasing a GPU-side timeline wait from this same queue is not possible
/// (a signalling submit queues behind the waiting submit, so the wait can
/// never be satisfied from its own stream), so the in-flight window comes
/// from the test-only waiter delay hook. Run with
/// `cargo test -p m2v-host --features test-hooks`: without the feature the
/// window does not exist and this test fails instead of passing vacuously.
#[test]
fn child_queue_drop_while_job_inflight() {
    if !common::is_child("child_queue_drop_while_job_inflight") {
        return;
    }
    assert!(
        cfg!(feature = "test-hooks"),
        "needs --features test-hooks (the waiter delay hook)"
    );
    let device = Device::new().expect("device");
    let queue = device.create_queue().expect("queue");

    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let enc = cb.begin().expect("encoder");
        enc.end().expect("encoder end");
    }
    cb.end().expect("command buffer end");

    let (tx, rx) = std::sync::mpsc::channel::<Completion>();
    std::env::set_var("M2V_TEST_WAITER_DELAY_MS", "400");
    queue
        .submit(&[&cb], Some(Box::new(move |c| {
            let _ = tx.send(c);
        })))
        .expect("submit");

    // The waiter picks the job up and parks inside the delay hook; dropping
    // the caller's handle here leaves the waiter's Arc as the last strong one.
    std::thread::sleep(Duration::from_millis(150));
    drop(queue);

    // The job must still complete normally before the child exits.
    let c = rx.recv_timeout(Duration::from_secs(15)).expect("completion never arrived");
    assert!(c.error.is_none(), "GPU error: {:?}", c.error);

    drop(cb);
    drop(device);
}

/// Parent: an empty queue must drop and exit normally.
#[test]
fn empty_queue_drop_exits_cleanly() {
    if !common::gpu_enabled() {
        eprintln!("SKIP queue lifecycle tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    let outcome = common::run_child("child_empty_queue_drop", &[], DEADLINE);
    assert!(!outcome.timed_out, "child timed out:\n{}", outcome.output);
    assert!(outcome.success, "child did not exit normally:\n{}", outcome.output);
}

/// Parent: a queue dropped mid-flight still exits cleanly.
#[test]
fn queue_drop_while_job_inflight_exits_cleanly() {
    if !common::gpu_enabled() {
        eprintln!("SKIP queue lifecycle tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    let outcome = common::run_child("child_queue_drop_while_job_inflight", &[], DEADLINE);
    assert!(!outcome.timed_out, "child timed out:\n{}", outcome.output);
    assert!(outcome.success, "child did not exit normally:\n{}", outcome.output);
}

/// Parent: one add, exactly one completion, clean exit.
#[test]
fn one_add_one_completion_exits_cleanly() {
    if !common::gpu_enabled() {
        eprintln!("SKIP queue lifecycle tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    let outcome = common::run_child("child_one_add_one_completion", &[], DEADLINE);
    assert!(!outcome.timed_out, "child timed out:\n{}", outcome.output);
    assert!(outcome.success, "child did not exit normally:\n{}", outcome.output);
}

/// Parent: a failing submit reports one error completion and tears down cleanly.
#[test]
fn submit_failure_reports_one_error_completion() {
    if !common::gpu_enabled() {
        eprintln!("SKIP queue lifecycle tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    let outcome = common::run_child(
        "child_submit_failure_error_completion",
        &[("M2V_MAX_SUBMIT_MS", SUBMIT_LIMIT_MS)],
        DEADLINE,
    );
    assert!(!outcome.timed_out, "child timed out:\n{}", outcome.output);
    assert!(outcome.success, "child did not exit normally:\n{}", outcome.output);
}

/// Parent: the unsatisfied-wait failure stays inside the submit limit.
#[test]
fn unsatisfied_timeline_wait_is_bounded() {
    if !common::gpu_enabled() {
        eprintln!("SKIP queue lifecycle tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    let outcome = common::run_child(
        "child_unsatisfied_timeline_wait_bounded",
        &[("M2V_MAX_SUBMIT_MS", SUBMIT_LIMIT_MS)],
        DEADLINE,
    );
    assert!(!outcome.timed_out, "child timed out:\n{}", outcome.output);
    assert!(outcome.success, "child did not exit normally:\n{}", outcome.output);
}
