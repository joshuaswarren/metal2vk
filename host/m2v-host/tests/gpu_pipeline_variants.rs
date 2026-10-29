//! Dynamic threadgroup memory lengths (follow-up to #97, which PR 107 fixed).
//!
//! The lengths are SPIR-V spec ids, not Metal function constants, and a
//! kernel's constant table does not declare them. They must reach
//! `Pipeline::create` as lengths, and a variant built for one length must not
//! serve another. Every case checks the numbers.

mod common;

use std::time::Duration;

use metal2vk::{ConstantValue, Device, Fallback};

const N: usize = 256;
const GROUPS: u32 = 4;
const THREADS: u32 = 64;
/// 64 floats of local memory, in bytes: the length the host specialises.
const LEN_64: u32 = THREADS * 4;
const LEN_128: u32 = THREADS * 8;
const GUARD: f32 = -1.0e30;

fn device_and_queue(name: &str) -> (std::sync::Arc<Device>, std::sync::Arc<metal2vk::Queue>, std::sync::Arc<metal2vk::Library>) {
    let device = Device::new().expect("device");
    let library = common::load(&device, name);
    device
        .policy()
        .set_fallback("vary", Fallback::Cpu("cpu_vary".into()));
    let queue = device.create_queue().expect("queue");
    (device, queue, library)
}

fn pattern(scale: f32) -> Vec<f32> {
    (0..N).map(|i| (i as f32 + 1.0) * scale).collect()
}

/// out[i] = in[i] + k + len_bytes (the add library).
fn want_add(in_values: &[f32], k: f32, len: u32) -> Vec<f32> {
    in_values.iter().map(|v| v + k + len as f32).collect()
}

fn check(label: &str, got: &[f32], want: &[f32]) {
    for i in 0..N {
        assert!(
            (got[i] - want[i]).abs() < 1e-3 * want[i].abs().max(1.0),
            "{label} at {i}: got {}, want {}",
            got[i],
            want[i]
        );
        assert!(got[i] != GUARD, "{label} left the guard value at {i}");
    }
}

/// Different threadgroup lengths on one queue, then the first length again.
#[test]
fn threadgroup_length_variants_are_distinct() {
    if !common::gpu_enabled() {
        eprintln!("SKIP pipeline variant tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    let (device, queue, library) = device_and_queue("vary_add.m2vlib");
    let pipeline = device
        .create_pipeline(&library, "vary", &[(0, ConstantValue::F32(1.0))])
        .expect("pipeline");

    let values = pattern(1.0);
    let buf1 = device.create_buffer(N * 4).expect("buffer 1");
    let buf2 = device.create_buffer(N * 4).expect("buffer 2");
    let buf3 = device.create_buffer(N * 4).expect("buffer 3");
    for buf in [&buf1, &buf2, &buf3] {
        common::fill_f32(buf, &values);
    }

    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let mut enc = cb.begin().expect("encoder");
        enc.set_pipeline(&pipeline).expect("set pipeline");
        enc.set_address(0, buf1.gpu_address()).expect("set address");
        enc.set_threadgroup_memory_length(1, LEN_64).expect("set length");
        enc.dispatch_threadgroups([GROUPS, 1, 1], [THREADS, 1, 1]).expect("dispatch 64");
        enc.set_address(0, buf2.gpu_address()).expect("set address");
        enc.set_threadgroup_memory_length(1, LEN_128).expect("set length");
        enc.dispatch_threadgroups([GROUPS, 1, 1], [THREADS, 1, 1]).expect("dispatch 128");
        enc.set_address(0, buf3.gpu_address()).expect("set address");
        enc.set_threadgroup_memory_length(1, LEN_64).expect("set length");
        enc.dispatch_threadgroups([GROUPS, 1, 1], [THREADS, 1, 1]).expect("dispatch 64 again");
        enc.end().expect("encoder end");
    }
    cb.end().expect("command buffer end");

    let completion = common::submit_and_wait(&queue, &[&cb], Duration::from_secs(60));
    assert!(completion.error.is_none(), "GPU error: {:?}", completion.error);

    check("length 64", &common::read_f32(&buf1, N), &want_add(&values, 1.0, LEN_64));
    check("length 128", &common::read_f32(&buf2, N), &want_add(&values, 1.0, LEN_128));
    check("length 64 again", &common::read_f32(&buf3, N), &want_add(&values, 1.0, LEN_64));
    device
        .policy()
        .report_check("vary", true, "length scenario matched the CPU result");
}

/// A missing length and a length that is not a multiple of four are refused by
/// name, and neither records a dispatch.
#[test]
fn bad_threadgroup_lengths_are_refused() {
    if !common::gpu_enabled() {
        eprintln!("SKIP pipeline variant tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    let (device, queue, library) = device_and_queue("vary_add.m2vlib");
    let pipeline = device
        .create_pipeline(&library, "vary", &[(0, ConstantValue::F32(1.0))])
        .expect("pipeline");
    let out = device.create_buffer(N * 4).expect("buffer");
    common::fill_f32(&out, &vec![GUARD; N]);

    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let mut enc = cb.begin().expect("encoder");
        enc.set_pipeline(&pipeline).expect("set pipeline");
        enc.set_address(0, out.gpu_address()).expect("set address");
        let missing = enc
            .dispatch_threadgroups([GROUPS, 1, 1], [THREADS, 1, 1])
            .expect_err("a dispatch without a threadgroup length must fail");
        assert!(
            missing.to_string().contains("set_threadgroup_memory_length"),
            "named error expected, got: {missing}"
        );
        let odd = enc
            .set_threadgroup_memory_length(1, 6)
            .expect_err("a length that is not a multiple of four must fail");
        assert!(
            odd.to_string().contains("multiple of 4"),
            "named error expected, got: {odd}"
        );
        enc.end().expect("encoder end");
    }
    cb.end().expect("command buffer end");

    let completion = common::submit_and_wait(&queue, &[&cb], Duration::from_secs(60));
    assert!(completion.error.is_none(), "GPU error: {:?}", completion.error);
    let got = common::read_f32(&out, N);
    for (i, v) in got.iter().enumerate() {
        assert!(*v == GUARD, "a refused dispatch wrote {v} at {i}");
    }
}
