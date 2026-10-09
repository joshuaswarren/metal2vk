//! Step-marker probe to bisect the M2 segfault; not part of the API surface.

use metal2vk::{Device, Library};
use std::sync::Arc;

fn step(name: &str) {
    eprintln!("STEP {name}");
}

#[test]
fn probe_steps() {
    if std::env::var("M2V_GPU").as_deref() != Ok("1") {
        return;
    }
    step("device-new");
    let device = Device::new().expect("device");
    step("device-ok");
    let a = device.create_buffer(16384).expect("buffer a");
    step("buffer-a");
    let b = device.create_buffer(16384).expect("buffer b");
    step("buffer-b");
    unsafe {
        let pa = a.contents() as *mut f32;
        for i in 0..4096 {
            *pa.add(i) = i as f32;
        }
    }
    step("host-write");
    let bytes = std::fs::read(std::env::var("M2V_LIB").unwrap()).unwrap();
    let library = Library::from_bytes(&device, &bytes).expect("library");
    step("library-ok");
    let pipeline = device.create_pipeline(&library, "add", &[]).expect("pipeline");
    step("pipeline-ok");
    let queue = device.create_queue().expect("queue");
    step("queue-ok");
    let cb = queue.new_command_buffer().expect("cb");
    step("cb-ok");
    {
        let mut enc = cb.begin().expect("encoder");
        step("begin-ok");
        enc.set_pipeline(&pipeline).unwrap();
        step("set-pipeline");
        enc.set_address(0, a.gpu_address()).unwrap();
        enc.set_address(1, b.gpu_address()).unwrap();
        step("set-address");
        enc.dispatch_threadgroups([32, 1, 1], [128, 1, 1]).unwrap();
        step("dispatch-recorded");
        enc.end().unwrap();
        step("encoder-end");
    }
    cb.end().unwrap();
    step("cb-end");
    let (tx, rx) = std::sync::mpsc::channel();
    queue.submit(&[&cb], Some(Box::new(move |c| {
        step("completion");
        eprintln!("completion error={:?} start={} end={}", c.error.is_some(), c.gpu_start_ns, c.gpu_end_ns);
        tx.send(()).unwrap();
    }))).expect("submit");
    step("submitted");
    rx.recv_timeout(std::time::Duration::from_secs(10)).expect("completion");
    step("done");
    let _ = Arc::strong_count(&pipeline);
}
