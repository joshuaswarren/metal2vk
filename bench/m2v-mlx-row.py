#!/usr/bin/env python3
"""One mlx hand-kernel row for m2v-bench.py (runs in whatever python has mlx).
Usage: m2v-mlx-row.py CASE DTYPE N   -> single JSON object on stdout.
Standalone on purpose: the mlx venv may not have numpy."""
import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn

NPDTS = {"f32": 4, "f16": 2}


def bench(fn, batch=40):
    mx.eval(fn())
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(batch)]
        mx.eval(outs)
        dt = (time.perf_counter() - t0) / batch * 1e6
        best = dt if best is None or dt < best else best
    return best


def main():
    case, dt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    itemsize = NPDTS[dt]
    mdt = mx.float32 if dt == "f32" else mx.float16
    mx.random.seed(7)
    out = {}
    if case == "activation":
        x = mx.random.normal((n,)).astype(mdt)
        mx.eval(x)
        for key, fn in (("fused", lambda: nn.silu(x)), ("unfused", lambda: x * mx.sigmoid(x))):
            us = bench(fn)
            r = {"us": round(us, 2), "GBps": round(2 * n * itemsize / (us * 1e3), 1)}
            out[key] = r
    elif case in ("softmax", "tilematmul", "tilematmul_rt"):
        if case == "softmax":
            x = (mx.random.normal((n, n)) * 3).astype(mdt)
            mx.eval(x)
            us = bench(lambda: mx.softmax(x, axis=-1))
            out["fused"] = {"us": round(us, 2), "GBps": round(2 * n * n * itemsize / (us * 1e3), 1)}
        else:
            a = mx.random.normal((n, n))
            b = mx.random.normal((n, n))
            mx.eval(a, b)
            us = bench(lambda: a @ b)
            out["fused"] = {"us": round(us, 2), "GFLOPs": round(2.0 * n * n * n / (us * 1e3), 1)}
    else:
        raise ValueError(case)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
