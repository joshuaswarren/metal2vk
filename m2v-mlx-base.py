"""Baseline for the metal2vk spike: same ops, same shapes, omarchy-mlx hand-written Vulkan kernels (via mlx).
Run on the GPU host with a python that has the omarchy-mlx wheel installed. Usage: m2v-mlx-base.py OUT.json
Per-op time = wall time of a batch of independent evals / batch size (one sync at the end), so it includes
whatever command-buffer overhead mlx has and the spike runner does not; both are reported with that caveat."""
import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn

res = {}


def bench(name, fn, nbytes=None, flops=None, batch=40):
    mx.eval(fn())  # warm: compiles pipelines
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(batch)]
        mx.eval(outs)
        dt = (time.perf_counter() - t0) / batch * 1e6
        best = dt if best is None or dt < best else best
    r = {"us": round(best, 2)}
    if nbytes:
        r["GBps"] = round(nbytes / (best * 1e3), 1)
    if flops:
        r["GFLOPs"] = round(flops / (best * 1e3), 1)
    res[name] = r
    print(name, r, flush=True)


n = 1 << 23
for dt, nb in ((mx.float32, 4), (mx.float16, 2)):
    x = mx.random.normal((n,)).astype(dt)
    mx.eval(x)
    bench(f"activation_{'f32' if nb == 4 else 'f16'}", lambda x=x: nn.silu(x), nbytes=2 * n * nb)
    bench(f"activation_unfused_{'f32' if nb == 4 else 'f16'}", lambda x=x: x * mx.sigmoid(x), nbytes=2 * n * nb)
for dt, nb in ((mx.float32, 4), (mx.float16, 2)):
    x = (mx.random.normal((4096, 4096)) * 3).astype(dt)
    mx.eval(x)
    bench(f"softmax_{'f32' if nb == 4 else 'f16'}", lambda x=x: mx.softmax(x, axis=-1), nbytes=2 * 4096 * 4096 * nb)
for m in (256, 1024):
    a = mx.random.normal((m, m))
    b = mx.random.normal((m, m))
    mx.eval(a, b)
    bench(f"tilematmul_{m}", lambda a=a, b=b: a @ b, flops=2.0 * m * m * m)
json.dump(res, open(sys.argv[1], "w"), indent=1)
