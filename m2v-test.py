#!/usr/bin/env python3
"""metal2vk spike test driver. Run on the GPU host (one small pilot submit, then a timed batch under 20 s).
Usage: m2v-test.py OUTDIR RUNNER SPVDIR [case ...]   cases: activation softmax tilematmul
Writes OUTDIR/results.json. CPU reference is numpy float64 -> float32/16 compare."""
import json, subprocess, sys, os, struct
import numpy as np

out, runner, spvdir = sys.argv[1:4]
cases = sys.argv[4:] or ["activation", "softmax", "tilematmul"]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)
res = {}


LOCAL = {"activation": "256 1 1", "softmax": "256 1 1", "tile_matmul": "32 1 1"}


def run(spv, entry, grid, bufs, push=None, iters=1, dump=None):
    loc = next(v for k, v in LOCAL.items() if entry.startswith(k))
    cmd = [runner, spv, entry, *map(str, grid), "--iters", str(iters), "--local", *loc.split()]
    if push is not None:
        pf = os.path.join(out, entry + ".push")
        open(pf, "wb").write(push)
        cmd += ["--push", pf]
    for b in bufs:
        cmd += ["--buf", b]
    if dump:
        cmd += ["--dump", dump]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode:
        return None, r.stderr.strip()[-400:]
    line = [l for l in r.stdout.splitlines() if l.startswith("iters=")][0]
    return (float(line.split("per_dispatch_us=")[1].split()[0]), float(line.split("wall_us=")[1])), r.stderr.strip()


def timed(spv, entry, grid, bufs, push=None):
    """pilot with 1 dispatch, then size the batch so the submit stays near 1 s (well under the 20 s rule)."""
    p1, err = run(spv, entry, grid, bufs, push, 1)
    if p1 is None:
        return None, err
    wall1 = p1[1]  # wall clock of a 1-dispatch submit: includes launch latency, so it over-estimates the cost
    iters = int(max(1, min(500, 2e6 / max(wall1, 1.0))))  # target about 2 s per submit, hard ceiling well under the 20 s rule
    t, err = run(spv, entry, grid, bufs, push, iters)
    if t is None:
        return None, err
    ts_per, wall_total = t
    wall_per = wall_total / iters
    # GPU timestamps can under-read (seen on the fork ICD); the wall clock of the whole submit is the conservative figure.
    per = max(ts_per, wall_per) if iters < 50 else wall_per
    return per, (f"iters={iters} est_submit_s={iters * wall1 / 1e6:.2f} actual_submit_s={wall_total / 1e6:.2f} "
                 f"ts_per_us={ts_per:.2f} wall_per_us={wall_per:.2f}")


def w(name, arr):
    p = os.path.join(out, name)
    arr.tofile(p)
    return p


def silu(x):
    return x / (1.0 + np.exp(-x))


for case in cases:
    try:
        if case == "activation":
            n = 1 << 23  # 32768 workgroups of 256, under the 65535 dispatch limit
            for dt, npdt, tol in (("f32", np.float32, 2e-5), ("f16", np.float16, 2e-3)):
                x = rng.standard_normal(n).astype(npdt)
                y = np.zeros(n, npdt)
                bufs = [w(f"act_in_{dt}.bin", x), w(f"act_out_{dt}.bin", y), w("act_n.bin", np.array([n], np.uint32)),
                        w("act_type.bin", np.array([0], np.uint32))]  # ActivationType::SILU = 0
                push = struct.pack("<4I", 0, 0, 0, 0) + struct.pack("<I", 0)  # clspv push block: 16 bytes group/region offsets, then POD in_place = false
                t, info = timed(os.path.join(spvdir, "activation.spv"), f"activation_{dt}", ((n + 255) // 256, 1, 1), bufs, push)
                # dump pass (one dispatch) for correctness
                run(os.path.join(spvdir, "activation.spv"), f"activation_{dt}", ((n + 255) // 256, 1, 1), bufs, push, 1,
                    dump=f"1:{os.path.join(out, f'act_res_{dt}.bin')}")
                got = np.fromfile(os.path.join(out, f"act_res_{dt}.bin"), npdt)
                ref = silu(x.astype(np.float64))
                err = float(np.max(np.abs(got.astype(np.float64) - ref) / (1.0 + np.abs(ref))))
                res[f"activation_{dt}"] = {"n": n, "max_rel_err": err, "tol": tol, "ok": bool(err < tol), "us": t, "info": info,
                                          "GBps": (None if t is None else 2 * n * np.dtype(npdt).itemsize / (t * 1e3))}
        elif case == "softmax":
            rows, cols = 4096, 4096
            for dt, npdt, tol in (("f32", np.float32, 1e-5), ("f16", np.float16, 2e-3)):
                x = (rng.standard_normal((rows, cols)) * 3).astype(npdt)
                bufs = [w(f"sm_vals_{dt}.bin", x.copy()), w(f"sm_sinks_{dt}.bin", np.zeros(1, npdt)),
                        w("sm_meta.bin", np.array([cols, rows, 1], np.uint32))]
                push = struct.pack("<4I", 0, 0, 0, 0) + struct.pack("<I", 0)  # offsets + has_sinks = false
                t, info = timed(os.path.join(spvdir, "softmax.spv"), f"softmax_{dt}", (rows, 1, 1), bufs, push)
                # the timed run softmaxes in place repeatedly; correctness needs a fresh single dispatch
                w(f"sm_vals_{dt}.bin", x.copy())
                run(os.path.join(spvdir, "softmax.spv"), f"softmax_{dt}", (rows, 1, 1), bufs, push, 1,
                    dump=f"0:{os.path.join(out, f'sm_res_{dt}.bin')}")
                got = np.fromfile(os.path.join(out, f"sm_res_{dt}.bin"), npdt).reshape(rows, cols).astype(np.float64)
                xr = x.astype(np.float64)
                ref = np.exp(xr - xr.max(1, keepdims=True))
                ref /= ref.sum(1, keepdims=True)
                err = float(np.max(np.abs(got - ref)))
                res[f"softmax_{dt}"] = {"rows": rows, "cols": cols, "max_abs_err": err, "tol": tol, "ok": bool(err < tol), "us": t,
                                       "info": info, "GBps": (None if t is None else 2 * rows * cols * np.dtype(npdt).itemsize / (t * 1e3))}
        elif case == "tilematmul_rt":
            # register-tiled variants: one subgroup owns an MR x NR block of 8x8 tiles (cases/tilematmul_rt.cl)
            for name, mr, nr in (("tile_matmul_rt2x2_f32", 2, 2), ("tile_matmul_rt4x2_f32", 4, 2), ("tile_matmul_rt4x4_f32", 4, 4)):
                for M in [int(x) for x in os.environ.get("M2V_SIZES", "256,1024").split(",")]:
                    N = K = M
                    A = rng.standard_normal((M, K)).astype(np.float32)
                    B = rng.standard_normal((K, N)).astype(np.float32)
                    bufs = [w("mm_a.bin", A), w("mm_b.bin", B), w("mm_c.bin", np.zeros((M, N), np.float32)),
                            w("mm_dims.bin", np.array([M, N, K], np.uint32))]
                    grid = (N // (8 * nr), M // (8 * mr), 1)
                    push0 = struct.pack("<4I", 0, 0, 0, 0)
                    t, info = timed(os.path.join(spvdir, "tilematmul_rt.spv"), name, grid, bufs, push0)
                    run(os.path.join(spvdir, "tilematmul_rt.spv"), name, grid, bufs, push0, 1, dump=f"2:{os.path.join(out, 'mm_res.bin')}")
                    got = np.fromfile(os.path.join(out, "mm_res.bin"), np.float32).reshape(M, N).astype(np.float64)
                    ref = A.astype(np.float64) @ B.astype(np.float64)
                    err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
                    res[f"{name}_{M}"] = {"M": M, "max_rel_err": err, "tol": 1e-5, "ok": bool(err < 1e-5), "us": t, "info": info,
                                          "GFLOPs": (None if t is None else 2.0 * M * N * K / (t * 1e3))}
        elif case == "tilematmul":
            for M in (256, 1024):
                N = K = M
                A = rng.standard_normal((M, K)).astype(np.float32)
                B = rng.standard_normal((K, N)).astype(np.float32)
                bufs = [w("mm_a.bin", A), w("mm_b.bin", B), w("mm_c.bin", np.zeros((M, N), np.float32)),
                        w("mm_dims.bin", np.array([M, N, K], np.uint32))]
                grid = (N // 8, M // 8, 1)
                push0 = struct.pack("<4I", 0, 0, 0, 0)
                t, info = timed(os.path.join(spvdir, "tilematmul.spv"), "tile_matmul_f32", grid, bufs, push0)
                run(os.path.join(spvdir, "tilematmul.spv"), "tile_matmul_f32", grid, bufs, push0, 1, dump=f"2:{os.path.join(out, 'mm_res.bin')}")
                got = np.fromfile(os.path.join(out, "mm_res.bin"), np.float32).reshape(M, N).astype(np.float64)
                ref = A.astype(np.float64) @ B.astype(np.float64)
                err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
                res[f"tilematmul_{M}"] = {"M": M, "max_rel_err": err, "tol": 1e-5, "ok": bool(err < 1e-5), "us": t, "info": info,
                                         "GFLOPs": (None if t is None else 2.0 * M * N * K / (t * 1e3))}
    except Exception as e:  # keep going so one failure does not hide the others
        res[case] = {"error": repr(e)}

json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
for k, v in res.items():
    print(k, json.dumps(v))
