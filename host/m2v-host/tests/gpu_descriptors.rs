//! Descriptor rebinding between dispatches in one encoder (issue #96).
//!
//! Two dispatches in one encoder, with the input, output and scalar rebound
//! between them, must each use the bindings that were live when they were
//! recorded. The scenario runs twice: once with push descriptors and once in a
//! child process with `M2V_NO_PUSH_DESCRIPTOR=1`, which forces the plain
//! descriptor-set path.

mod common;

use std::time::Duration;

use metal2vk::{Device, Fallback};

const N: usize = 256;
const GUARD: f32 = -1.0e30;
const DEADLINE: Duration = Duration::from_secs(120);

/// Two dispatches with rebound buffers and scalar, then a second pipeline.
fn scenario() {
    let device = Device::new().expect("device");
    let library = common::load(&device, "descriptors.m2vlib");
    device
        .policy()
        .set_fallback("add_scalar", Fallback::Cpu("cpu_add_scalar".into()));
    device
        .policy()
        .set_fallback("mul_scalar", Fallback::Cpu("cpu_mul_scalar".into()));
    let queue = device.create_queue().expect("queue");
    let p_add = device
        .create_pipeline(&library, "add_scalar", &[])
        .expect("pipeline add_scalar");
    let p_mul = device
        .create_pipeline(&library, "mul_scalar", &[])
        .expect("pipeline mul_scalar");

    let a1 = device.create_buffer(N * 4).expect("buffer a1");
    let a2 = device.create_buffer(N * 4).expect("buffer a2");
    let out1 = device.create_buffer(N * 4).expect("buffer out1");
    let out2 = device.create_buffer(N * 4).expect("buffer out2");
    let out3 = device.create_buffer(N * 4).expect("buffer out3");
    let in1: Vec<f32> = (0..N).map(|i| i as f32 + 1.0).collect();
    let in2: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) * 1000.0).collect();
    common::fill_f32(&a1, &in1);
    common::fill_f32(&a2, &in2);
    for out in [&out1, &out2, &out3] {
        common::fill_f32(out, &vec![GUARD; N]);
    }

    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let mut enc = cb.begin().expect("encoder");
        // dispatch 1: a1 -> out1 with k = 7
        enc.set_pipeline(&p_add).expect("set pipeline");
        enc.set_address(0, a1.gpu_address()).expect("set address 0");
        enc.set_address(1, out1.gpu_address()).expect("set address 1");
        enc.set_bytes(2, &7.0f32.to_le_bytes()).expect("set bytes 2");
        enc.dispatch_threadgroups([4, 1, 1], [64, 1, 1]).expect("dispatch 1");
        // dispatch 2: same pipeline again, everything rebound
        enc.set_pipeline(&p_add).expect("set pipeline again");
        enc.set_address(0, a2.gpu_address()).expect("rebind address 0");
        enc.set_address(1, out2.gpu_address()).expect("rebind address 1");
        enc.set_bytes(2, &13.0f32.to_le_bytes()).expect("rebind bytes 2");
        enc.dispatch_threadgroups([4, 1, 1], [64, 1, 1]).expect("dispatch 2");
        // dispatch 3: the compatible second pipeline, which resets the arguments
        enc.set_pipeline(&p_mul).expect("set pipeline mul");
        enc.set_address(0, a1.gpu_address()).expect("set address 0");
        enc.set_address(1, out3.gpu_address()).expect("set address 1");
        enc.set_bytes(2, &2.0f32.to_le_bytes()).expect("set bytes 2");
        enc.dispatch_threadgroups([4, 1, 1], [64, 1, 1]).expect("dispatch 3");
        enc.end().expect("encoder end");
    }
    cb.end().expect("command buffer end");

    let completion = common::submit_and_wait(&queue, &[&cb], Duration::from_secs(60));
    assert!(completion.error.is_none(), "GPU error: {:?}", completion.error);

    let got1 = common::read_f32(&out1, N);
    let got2 = common::read_f32(&out2, N);
    let got3 = common::read_f32(&out3, N);
    for i in 0..N {
        let want1 = in1[i] + 7.0;
        let want2 = in2[i] + 13.0;
        let want3 = in1[i] * 2.0;
        assert!(
            (got1[i] - want1).abs() < 1e-4 * want1.abs().max(1.0),
            "dispatch 1 at {i}: got {}, want {want1}",
            got1[i]
        );
        assert!(
            (got2[i] - want2).abs() < 1e-4 * want2.abs().max(1.0),
            "dispatch 2 at {i}: got {}, want {want2}",
            got2[i]
        );
        assert!(
            (got3[i] - want3).abs() < 1e-4 * want3.abs().max(1.0),
            "dispatch 3 at {i}: got {}, want {want3}",
            got3[i]
        );
        assert!(got1[i] != GUARD, "dispatch 1 left the guard value at {i}");
        assert!(got2[i] != GUARD, "dispatch 2 left the guard value at {i}");
        assert!(got3[i] != GUARD, "dispatch 3 left the guard value at {i}");
    }
    device
        .policy()
        .report_check("add_scalar", true, "rebind scenario matched the CPU result");
    device
        .policy()
        .report_check("mul_scalar", true, "rebind scenario matched the CPU result");
}

/// The scenario in a child process with the descriptor-set fallback forced.
#[test]
fn child_descriptor_rebind_without_push_descriptors() {
    if !common::is_child("child_descriptor_rebind_without_push_descriptors") {
        return;
    }
    scenario();
}

/// Push descriptors (or whatever the device does by default).
#[test]
fn descriptor_rebind_keeps_each_dispatch() {
    if !common::gpu_enabled() {
        eprintln!("SKIP descriptor rebind tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    eprintln!("running the rebind scenario with the default descriptor path");
    scenario();
}

/// The same scenario with the plain descriptor-set path, in its own process.
#[test]
fn descriptor_rebind_keeps_each_dispatch_without_push_descriptors() {
    if !common::gpu_enabled() {
        eprintln!("SKIP descriptor rebind tests (set M2V_GPU=1 on a GPU host)");
        return;
    }
    let outcome = common::run_child(
        "child_descriptor_rebind_without_push_descriptors",
        &[("M2V_NO_PUSH_DESCRIPTOR", "1"), ("M2V_VALIDATION", "1")],
        DEADLINE,
    );
    assert!(!outcome.timed_out, "child timed out:\n{}", outcome.output);
    assert!(
        outcome.success,
        "the descriptor-set path failed:\n{}",
        outcome.output
    );
    // The validation layer prints VUID-VkDeviceCreateInfo-pNext-pNext on some
    // loaders at device creation; only descriptor and layout messages count.
    let descriptor_messages: Vec<&str> = outcome
        .output
        .lines()
        .filter(|l| {
            l.contains("m2v-validation:")
                && (l.contains("VkDescriptorSet") || l.contains("VkDescriptorPool") || l.contains("pSetLayouts"))
        })
        .collect();
    assert!(
        descriptor_messages.is_empty(),
        "the descriptor-set path reported descriptor validation errors:\n{}",
        descriptor_messages.join("\n")
    );
}
