//! Dynamic-threadgroup pipeline variants cached on the queue (#97): real dispatches
//! through `Encoder::resolve_pipeline`, compared word for word with CPU results.
//!
//! Skips (prints SKIP, exits 0) unless M2V_GPU=1 and M2V_VARIANT_DIR names a directory
//! holding variant_a.m2vlib and variant_b.m2vlib from tests/fixtures/variant_cache/build.sh.
//! The fixture kernels write `FACTOR * shared[i % (LENGTH_BYTES / 4)] + BIAS`, with
//! `shared[j] = j + 1`: FACTOR is function constant 7, LENGTH_BYTES is threadgroup
//! argument 1, BIAS is a literal in the shader body (A `variant` 0, A `variant_alt` 500,
//! B `variant` 1000).

use std::sync::Arc;

use metal2vk::{Buffer, ConstantValue, Device, Error, Fallback, Library, Pipeline, Queue};

/// Active output words: two workgroups of 32.
const N: usize = 64;
const GUARDS: usize = 4;
const GUARD: u32 = 0xDEAD_BEEF;
const FACTOR_INDEX: u32 = 7;
const TG_ARG: u32 = 1;

struct Fixture {
    device: Arc<Device>,
    a: Arc<Library>,
    b: Arc<Library>,
    queue: Arc<Queue>,
}

fn setup() -> Option<Fixture> {
    if std::env::var("M2V_GPU").as_deref() != Ok("1") {
        eprintln!("SKIP gpu tests (set M2V_GPU=1 on a GPU host)");
        return None;
    }
    let dir =
        std::env::var("M2V_VARIANT_DIR").expect("M2V_VARIANT_DIR=<output of tests/fixtures/variant_cache/build.sh>");
    let device = Device::new().expect("Vulkan 1.3 device");
    eprintln!("device: {}", device.name());
    let load = |file: &str| {
        let bytes = std::fs::read(format!("{dir}/{file}")).expect("read .m2vlib");
        Library::from_bytes(&device, &bytes).expect("parse .m2vlib")
    };
    let (a, b) = (load("variant_a.m2vlib"), load("variant_b.m2vlib"));
    // The CPU results below are the first-use references; the gate refuses an unverified kernel without one.
    device.policy().set_fallback("variant", Fallback::Cpu("cpu_variant".into()));
    device.policy().set_fallback("variant_alt", Fallback::Cpu("cpu_variant_alt".into()));
    let queue = device.create_queue().expect("queue");
    Some(Fixture { device, a, b, queue })
}

/// The gate is per kernel name, so every pipeline is created before the first `report_check`.
fn pipeline(f: &Fixture, library: &Arc<Library>, name: &str, factor: u32) -> Arc<Pipeline> {
    f.device
        .create_pipeline(library, name, &[(FACTOR_INDEX, ConstantValue::U32(factor))])
        .unwrap_or_else(|e| panic!("pipeline {name} FACTOR={factor}: {e}"))
}

fn expected(factor: u32, length_bytes: u32, bias: u32) -> Vec<u32> {
    let words = (length_bytes / 4) as usize;
    (0..N).map(|i| factor * ((i % words) as u32 + 1) + bias).collect()
}

/// N active words plus trailing guards, all set to GUARD.
fn output(f: &Fixture) -> Arc<Buffer> {
    let buf = f.device.create_buffer((N + GUARDS) * 4).expect("output buffer");
    let p = buf.contents() as *mut u32;
    for i in 0..N + GUARDS {
        unsafe { *p.add(i) = GUARD };
    }
    buf
}

fn words(buf: &Buffer) -> Vec<u32> {
    let p = buf.contents() as *const u32;
    (0..N + GUARDS).map(|i| unsafe { *p.add(i) }).collect()
}

/// Exact comparison of all active words, and the guards stay untouched.
fn check(buf: &Buffer, want: &[u32], what: &str) {
    let got = words(buf);
    for i in 0..N {
        assert_eq!(got[i], want[i], "{what}: word {i}");
    }
    assert_eq!(got[N..], [GUARD; GUARDS], "{what}: guards");
}

fn untouched(buf: &Buffer, what: &str) {
    assert!(words(buf).iter().all(|&w| w == GUARD), "{what}: written by a dispatch that failed");
}

fn submit(f: &Fixture, cb: &metal2vk::CommandBuffer) {
    let (tx, rx) = std::sync::mpsc::channel();
    f.queue
        .submit(&[cb], Some(Box::new(move |completion| tx.send(completion).expect("completion channel"))))
        .expect("submit");
    let completion = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("completion callback");
    assert!(completion.error.is_none(), "GPU error: {:?}", completion.error);
}

/// One dispatch (2 x 32 threads) into a fresh output, in its own command buffer on the shared queue.
fn run(f: &Fixture, pipeline: &Arc<Pipeline>, length_bytes: u32) -> Arc<Buffer> {
    let out = output(f);
    let cb = f.queue.new_command_buffer().expect("command buffer");
    {
        let mut enc = cb.begin().expect("encoder");
        enc.set_pipeline(pipeline).unwrap();
        enc.set_address(0, out.gpu_address()).unwrap();
        enc.set_threadgroup_memory_length(TG_ARG, length_bytes).unwrap();
        enc.dispatch_threadgroups([2, 1, 1], [32, 1, 1]).unwrap();
        enc.end().unwrap();
    }
    cb.end().unwrap();
    submit(f, &cb);
    out
}

#[test]
fn threadgroup_variants_distinguish_function_constants() {
    let Some(f) = setup() else { return };
    let (a2, a5) = (pipeline(&f, &f.a, "variant", 2), pipeline(&f, &f.a, "variant", 5));
    check(&run(&f, &a2, 128), &expected(2, 128, 0), "A FACTOR=2");
    check(&run(&f, &a5, 128), &expected(5, 128, 0), "A FACTOR=5 after FACTOR=2");
    check(&run(&f, &a2, 128), &expected(2, 128, 0), "A FACTOR=2 again");
    f.device.policy().report_check("variant", true, "FACTOR 2/5/2 at 128 bytes, 64 words each");
}

#[test]
fn threadgroup_variants_distinguish_library_identity() {
    let Some(f) = setup() else { return };
    let (a, b) = (pipeline(&f, &f.a, "variant", 2), pipeline(&f, &f.b, "variant", 2));
    check(&run(&f, &a, 128), &expected(2, 128, 0), "library A");
    check(&run(&f, &b, 128), &expected(2, 128, 1000), "library B after A");
    check(&run(&f, &a, 128), &expected(2, 128, 0), "library A again");
    f.device.policy().report_check("variant", true, "libraries A/B/A at FACTOR 2, 128 bytes, 64 words each");
}

#[test]
fn threadgroup_variant_lengths_remain_observable() {
    let Some(f) = setup() else { return };
    let a = pipeline(&f, &f.a, "variant", 2);
    check(&run(&f, &a, 128), &expected(2, 128, 0), "128 bytes");
    let mid = run(&f, &a, 256);
    check(&mid, &expected(2, 256, 0), "256 bytes after 128");
    // indices 32-63 tell 256 bytes from a stale 128 (2 * (i % 64 + 1) vs 2 * (i % 32 + 1))
    assert_eq!(words(&mid)[32..N], expected(2, 256, 0)[32..N]);
    check(&run(&f, &a, 128), &expected(2, 128, 0), "128 bytes again");
    f.device.policy().report_check("variant", true, "lengths 128/256/128 at FACTOR 2, 64 words each");
}

/// Beyond the #97 table: one library, two functions, same constants and lengths.
#[test]
fn threadgroup_variants_distinguish_function_names() {
    let Some(f) = setup() else { return };
    let (v, alt) = (pipeline(&f, &f.a, "variant", 2), pipeline(&f, &f.a, "variant_alt", 2));
    check(&run(&f, &v, 128), &expected(2, 128, 0), "A variant");
    check(&run(&f, &alt, 128), &expected(2, 128, 500), "A variant_alt after variant");
    check(&run(&f, &v, 128), &expected(2, 128, 0), "A variant again");
    f.device.policy().report_check("variant", true, "variant/variant_alt/variant at FACTOR 2, 128 bytes");
    f.device.policy().report_check("variant_alt", true, "variant/variant_alt/variant at FACTOR 2, 128 bytes");
}

#[test]
fn threadgroup_length_errors_precede_dispatch() {
    let Some(f) = setup() else { return };
    let a = pipeline(&f, &f.a, "variant", 2);
    for case in ["missing length", "length 130"] {
        let (failed, valid) = (output(&f), output(&f));
        let cb = f.queue.new_command_buffer().expect("command buffer");
        {
            let mut enc = cb.begin().expect("encoder");
            enc.set_pipeline(&a).unwrap();
            enc.set_address(0, failed.gpu_address()).unwrap();
            if case == "length 130" {
                match enc.set_threadgroup_memory_length(TG_ARG, 130) {
                    Err(Error::Invalid(m)) => assert!(m.contains("130") && m.contains("multiple of 4"), "{m}"),
                    other => panic!("length 130: want Error::Invalid, got {other:?}"),
                }
            }
            // A rejected length is not kept, so both cases now lack argument 1's length.
            match enc.dispatch_threadgroups([2, 1, 1], [32, 1, 1]) {
                Err(Error::Invalid(m)) => {
                    assert!(m.contains("argument 1") && m.contains("set_threadgroup_memory_length"), "{case}: {m}")
                }
                other => panic!("{case}: want Error::Invalid, got {other:?}"),
            }
            enc.set_address(0, valid.gpu_address()).unwrap();
            enc.set_threadgroup_memory_length(TG_ARG, 128).unwrap();
            enc.dispatch_threadgroups([2, 1, 1], [32, 1, 1]).unwrap();
            enc.end().unwrap();
        }
        cb.end().unwrap();
        submit(&f, &cb);
        untouched(&failed, case);
        check(&valid, &expected(2, 128, 0), &format!("valid dispatch after {case}"));
    }
    f.device.policy().report_check("variant", true, "128-byte dispatch after each rejected length");
}
