//! Zig consumer demo: run one translated kernel through the C ABI and verify.
//! Zig 0.17: the C header is translated by the build system (build.zig uses
//! addTranslateC on ../../m2v-host/include/metal2vk.h) and imported as "m2v_c".
//! Build: M2V_LIB_DIR=<cargo target>/debug zig build -Doptimize=ReleaseSafe

const std = @import("std");
const c = @import("m2v_c");
const libc = struct {
    extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
    extern "c" fn fread(ptr: [*]u8, size: usize, n: usize, f: *anyopaque) usize;
    extern "c" fn fseek(f: *anyopaque, off: i64, whence: c_int) c_int;
    extern "c" fn ftell(f: *anyopaque) i64;
    extern "c" fn fclose(f: *anyopaque) i32;
};
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;

const N = 4096;

fn check(status: c.m2v_status) !void {
    if (status != c.m2v_status_OK) {
        const msg = c.m2v_last_error();
        std.debug.print("m2v error: {s}\n", .{if (msg != null) std.mem.span(msg.?) else "?"});
        return error.M2v;
    }
}

fn fill(buf: ?*c.m2v_buffer, comptime T: type, value: *const fn (usize) T) void {
    const p: [*]T = @ptrCast(@alignCast(c.m2v_buffer_contents(buf.?)));
    for (0..N) |i| p[i] = value(i);
}

pub fn main() !void {
    // library path from the environment, like the Rust tests (M2V_LIB)
    const lib_path = getenv("M2V_LIB") orelse return error.Usage;

    var dev: ?*c.m2v_device = null;
    try check(c.m2v_device_new(&dev));
    defer c.m2v_device_release(dev);

    var a: ?*c.m2v_buffer = null;
    var b: ?*c.m2v_buffer = null;
    try check(c.m2v_device_create_buffer(dev.?, N * 4, &a));
    try check(c.m2v_device_create_buffer(dev.?, N * 4, &b));
    fill(a.?, f32, struct {
        fn f(i: usize) f32 {
            return @floatFromInt(i);
        }
    }.f);
    fill(b.?, f32, struct {
        fn f(_: usize) f32 {
            return 0;
        }
    }.f);

    const lf = libc.fopen(lib_path, "rb") orelse return error.Usage;
    _ = libc.fseek(lf, 0, 2); // SEEK_END
    const lib_len: usize = @intCast(libc.ftell(lf));
    _ = libc.fseek(lf, 0, 0); // SEEK_SET
    const lib_bytes = std.heap.page_allocator.alloc(u8, lib_len) catch return error.OutOfMemory;
    const got = libc.fread(lib_bytes.ptr, 1, lib_len, lf);
    _ = libc.fclose(lf);
    if (got != lib_len) return error.ShortRead;
    var lib: ?*c.m2v_library = null;
    try check(c.m2v_library_from_bytes(dev.?, lib_bytes.ptr, lib_bytes.len, &lib));

    var pipe: ?*c.m2v_pipeline = null;
    try check(c.m2v_device_create_pipeline(dev.?, lib.?, "add", null, 0, &pipe));
    var q: ?*c.m2v_queue = null;
    try check(c.m2v_device_create_queue(dev.?, &q));

    var cb: ?*c.m2v_cmdbuf = null;
    try check(c.m2v_queue_new_command_buffer(q.?, &cb));
    var enc: ?*c.m2v_encoder = null;
    try check(c.m2v_cmdbuf_begin(cb.?, &enc));
    try check(c.m2v_encoder_set_pipeline(enc.?, pipe.?));
    try check(c.m2v_encoder_set_address(enc.?, 0, c.m2v_buffer_gpu_address(a.?)));
    try check(c.m2v_encoder_set_address(enc.?, 1, c.m2v_buffer_gpu_address(b.?)));
    try check(c.m2v_encoder_dispatch_threadgroups(enc.?, N / 128, 1, 1, 128, 1, 1));
    try check(c.m2v_encoder_end(enc.?));
    try check(c.m2v_cmdbuf_end(cb.?));

    var ev: ?*c.m2v_event = null;
    try check(c.m2v_device_create_event(dev.?, &ev));
    try check(c.m2v_queue_submit(q.?, @ptrCast(&cb.?), 1, null, null));
    try check(c.m2v_queue_signal_event(q.?, ev.?, 1)); // queue is serial: fires after add
    try check(c.m2v_event_wait(ev.?, 1, 10_000_000_000));

    const pa: [*]const f32 = @ptrCast(@alignCast(c.m2v_buffer_contents(a.?)));
    const pb: [*]const f32 = @ptrCast(@alignCast(c.m2v_buffer_contents(b.?)));
    for (0..N) |i| {
        if (@abs(pb[i] - pa[i]) > 1e-6) {
            std.debug.print("mismatch at {d}: {d} != {d}\n", .{ i, pb[i], pa[i] });
            return error.Wrong;
        }
    }
    c.m2v_report_check(dev.?, "add", true, "b[i] == a[i] verified in the demo");
    std.debug.print("PASS zig demo ({d} elements)\n", .{N});
}
