const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSafe });

    // The C header itself is the source of truth: translate it into a module.
    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("../../m2v-host/include/metal2vk.h"),
        .target = target,
        .optimize = optimize,
    });
    const m2v_c = translate_c.createModule();

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("demo.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    exe_mod.addImport("m2v_c", m2v_c);
    // libmetal2vk comes from the cargo build: CARGO_TARGET_DIR/debug
    if (b.graph.environ_map.get("M2V_LIB_DIR")) |dir| {
        exe_mod.addLibraryPath(.{ .cwd_relative = dir });
        exe_mod.linkSystemLibrary("metal2vk", .{});
    }

    const exe = b.addExecutable(.{ .name = "demo", .root_module = exe_mod });
    b.installArtifact(exe);
}
