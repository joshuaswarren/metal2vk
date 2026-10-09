//! Zig consumer demo: @cImport metal2vk.h, run one translated kernel, verify.
//! Build on the Apple silicon host:
//!   zig build-exe demo.zig -O ReleaseSafe -lc \
//!     -I../../m2v-host/include -L<dir with libmetal2vk> -lmetal2vk \
//!     -femit-bin=demo
//!   ./demo <add.m2vlib>

const std = @import("std");
const c = @cImport({
    @cInclude("metal2vk.h");
});

const N = 4096;

fn check(status: c.m2v_status) !void {
    if (status != c.m2v_status_OK) {
        const msg = c.m2v_last_error();
        std.debug.print("m2v error: {s}\n", .{if (msg != null) std.mem.span(msg) else "?"});
        return error.M2v;
    }
}

fn fill(buf: *c.m2v_buffer, comptime T: type, value: *const fn (usize) T) void {
    const p: [*]T = @ptrCast(@alignCast(c.m2v_buffer_contents(buf)));
    for (0..N) |i| p[i] = value(i);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    var args = try std.process.argsWithAllocator(alloc);
    defer args.deinit();
    _ = args.next();
    const lib_path = args.next() orelse return error.Usage; // ./demo add.m2vlib

    var dev: *c.m2v_device = undefined;
    try check(c.m2v_device_new(&dev));
    defer c.m2v_device_release(dev);

    var a: *c.m2v_buffer = undefined;
    var b: *c.m2v_buffer = undefined;
    try check(c.m2v_device_create_buffer(dev, N * 4, &a));
    try check(c.m2v_device_create_buffer(dev, N * 4, &b));
    fill(a, f32, struct {
        fn f(i: usize) f32 {
            return @floatFromInt(i);
        }
    }.f);
    fill(b, f32, struct {
        fn f(_: usize) f32 {
            return 0;
        }
    }.f);

    const lib_bytes = try std.fs.cwd().readFileAlloc(alloc, lib_path, 1 << 24);
    defer alloc.free(lib_bytes);
    var lib: *c.m2v_library = undefined;
    try check(c.m2v_library_from_bytes(dev, lib_bytes.ptr, lib_bytes.len, &lib));

    var pipe: *c.m2v_pipeline = undefined;
    try check(c.m2v_device_create_pipeline(dev, lib, "add", null, 0, &pipe));
    var q: *c.m2v_queue = undefined;
    try check(c.m2v_device_create_queue(dev, &q));

    var cb: *c.m2v_cmdbuf = undefined;
    try check(c.m2v_queue_new_command_buffer(q, &cb));
    var enc: *c.m2v_encoder = undefined;
    try check(c.m2v_cmdbuf_begin(cb, &enc));
    try check(c.m2v_encoder_set_pipeline(enc, pipe));
    try check(c.m2v_encoder_set_address(enc, 0, c.m2v_buffer_gpu_address(a)));
    try check(c.m2v_encoder_set_address(enc, 1, c.m2v_buffer_gpu_address(b)));
    try check(c.m2v_encoder_dispatch_threadgroups(enc, N / 128, 1, 1, 128, 1, 1));
    try check(c.m2v_encoder_end(enc));
    try check(c.m2v_cmdbuf_end(cb));

    var ev: *c.m2v_event = undefined;
    try check(c.m2v_device_create_event(dev, &ev));
    try check(c.m2v_queue_submit(q, @ptrCast(&cb), 1, null, null));
    try check(c.m2v_queue_signal_event(q, ev, 1)); // queue is serial: fires after add
    try check(c.m2v_event_wait(ev, 1, 10_000_000_000));

    const pa: [*]const f32 = @ptrCast(@alignCast(c.m2v_buffer_contents(a)));
    const pb: [*]const f32 = @ptrCast(@alignCast(c.m2v_buffer_contents(b)));
    for (0..N) |i| {
        if (@abs(pb[i] - pa[i]) > 1e-6) {
            std.debug.print("mismatch at {d}: {d} != {d}\n", .{ i, pb[i], pa[i] });
            return error.Wrong;
        }
    }
    std.debug.print("PASS zig demo ({d} elements)\n", .{N});
}
