//! End-to-end GPU tests: add kernels through a real .m2vlib. Skips (prints
//! SKIP, exits 0) unless M2V_GPU=1 and M2V_LIB point at a lab host with the
//! Vulkan driver; `cargo test` on the build CT takes the skip path.

use metal2vk::{Device, Fallback, Library};

fn setup() -> Option<(
    std::sync::Arc<Device>,
    std::sync::Arc<Library>,
    std::sync::Arc<metal2vk::Queue>,
)> {
    if std::env::var("M2V_GPU").as_deref() != Ok("1") {
        eprintln!("SKIP gpu tests (set M2V_GPU=1 on a GPU host)");
        return None;
    }
    let lib_path = std::env::var("M2V_LIB").expect("M2V_LIB=<path to add.m2vlib>");
    let device = Device::new().expect("Vulkan 1.3 device");
    eprintln!("device: {}", device.name());
    let info = device.info();
    eprintln!(
        "subgroup={} max_tpt={} shared={}B coopmat={} timeline={} mem={} GiB",
        info.subgroup_size,
        info.max_threads_per_threadgroup,
        info.max_shared_memory,
        info.has_coopmat,
        info.has_timeline,
        info.total_memory / (1 << 30)
    );
    let bytes = std::fs::read(&lib_path).expect("read .m2vlib");
    let library = Library::from_bytes(&device, &bytes).expect("parse .m2vlib");
    eprintln!("functions: {:?}", library.function_names());
    let queue = device.create_queue().expect("queue");
    Some((device, library, queue))
}

fn run(queue: &std::sync::Arc<metal2vk::Queue>, cb: &metal2vk::CommandBuffer) -> (u64, u64) {
    let (tx, rx) = std::sync::mpsc::channel();
    queue
        .submit(
            &[cb],
            Some(Box::new(move |completion| {
                tx.send(completion).expect("completion channel");
            })),
        )
        .expect("submit");
    let completion = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("completion callback");
    assert!(completion.error.is_none(), "GPU error: {:?}", completion.error);
    (completion.gpu_start_ns, completion.gpu_end_ns)
}

#[test]
fn add_kernel_end_to_end() {
    let Some((device, library, queue)) = setup() else { return };
    const N: usize = 4096;
    let a = device.create_buffer(N * 4).expect("buffer a");
    let b = device.create_buffer(N * 4).expect("buffer b");
    unsafe {
        let pa = a.contents() as *mut f32;
        for i in 0..N {
            *pa.add(i) = i as f32; // b stays zero: add gives b[i] = a[i]
        }
    }

    // the CPU result below is the reference for the first-use check (the gate refuses an unverified kernel without one)
    device.policy().set_fallback("add", Fallback::Cpu("cpu_add".into()));
    let pipeline = device.create_pipeline(&library, "add", &[]).expect("pipeline add");
    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let mut enc = cb.begin().expect("encoder");
        enc.set_pipeline(&pipeline).unwrap();
        enc.set_address(0, a.gpu_address()).unwrap();
        enc.set_address(1, b.gpu_address()).unwrap();
        enc.dispatch_threadgroups([(N as u32) / 128, 1, 1], [128, 1, 1]).unwrap();
        enc.end().unwrap();
    }
    cb.end().unwrap();
    let (start, end) = run(&queue, &cb);
    eprintln!("add gpu span: {} us", (end.saturating_sub(start)) / 1000);

    let pb = b.contents() as *const f32;
    for i in 0..N {
        let got = unsafe { *pb.add(i) };
        assert!((got - i as f32).abs() < 1e-6, "mismatch at {i}: got {got}");
    }
    device.policy().report_check("add", true, "b[i] == a[i] for 4096 elements");
    println!("PASS add_kernel_end_to_end ({} elements, {} us GPU span)", N, (end.saturating_sub(start)) / 1000);
}

#[test]
fn push_constant_kernel_end_to_end() {
    let Some((device, library, queue)) = setup() else { return };
    const N: usize = 4096;
    let a = device.create_buffer(N * 4).expect("buffer a");
    let b = device.create_buffer(N * 4).expect("buffer b");
    unsafe {
        let pa = a.contents() as *mut f32;
        for i in 0..N {
            *pa.add(i) = i as f32;
        }
    }

    device.policy().set_fallback("add_offset", Fallback::Cpu("cpu_add_offset".into()));
    let pipeline = device
        .create_pipeline(&library, "add_offset", &[])
        .expect("pipeline add_offset");
    let cb = queue.new_command_buffer().expect("command buffer");
    {
        let mut enc = cb.begin().expect("encoder");
        enc.set_pipeline(&pipeline).unwrap();
        enc.set_address(0, a.gpu_address()).unwrap();
        enc.set_address(1, b.gpu_address()).unwrap();
        enc.set_bytes(2, &7.0f32.to_le_bytes()).unwrap();
        enc.dispatch_threads([N as u32, 1, 1], [128, 1, 1]).unwrap();
        enc.end().unwrap();
    }
    cb.end().unwrap();
    let (start, end) = run(&queue, &cb);
    eprintln!("add_offset gpu span: {} us", (end.saturating_sub(start)) / 1000);

    let pb = b.contents() as *const f32;
    for i in 0..N {
        let got = unsafe { *pb.add(i) };
        let want = i as f32 + 7.0;
        assert!((got - want).abs() < 1e-6 * want.abs().max(1.0), "mismatch at {i}: got {got}, want {want}");
    }
    device.policy().report_check("add_offset", true, "b[i] == a[i] + 7 for 4096 elements");
    println!(
        "PASS push_constant_kernel_end_to_end ({} elements, {} us GPU span)",
        N,
        (end.saturating_sub(start)) / 1000
    );
}
