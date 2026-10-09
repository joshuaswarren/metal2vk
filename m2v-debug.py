"""Tiny cases for m2v-run: prints first elements got vs expected. Usage: m2v-debug.py OUTDIR RUNNER SPVDIR"""
import os
import struct
import subprocess
import sys

import numpy as np

out, runner, spvdir = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
Z16 = struct.pack("<4I", 0, 0, 0, 0)


def w(name, arr):
    p = os.path.join(out, name)
    arr.tofile(p)
    return p


def go(spv, entry, grid, bufs, push, dump_idx, dtype, local="256 1 1"):
    pf = os.path.join(out, entry + ".push")
    open(pf, "wb").write(push)
    df = os.path.join(out, entry + ".res")
    cmd = [runner, os.path.join(spvdir, spv), entry, *map(str, grid), "--local", *local.split(), "--push", pf, "--dump", f"{dump_idx}:{df}"]
    for b in bufs:
        cmd += ["--buf", b]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    print(entry, "rc", r.returncode, r.stdout.strip(), r.stderr.strip()[-300:])
    return np.fromfile(df, dtype) if r.returncode == 0 else None


rng = np.random.default_rng(1)
# activation, n = 1024
n = 1024
x = rng.standard_normal(n).astype(np.float32)
res = go("activation.spv", "activation_f32", (n // 256, 1, 1),
         [w("a_in", x), w("a_out", np.zeros(n, np.float32)), w("a_n", np.array([n], np.uint32)), w("a_t", np.array([0], np.uint32))],
         Z16 + struct.pack("<I", 0), 1, np.float32)
if res is not None:
    print("act got", res[:6], "want", (x / (1 + np.exp(-x)))[:6], "nonzero", int(np.count_nonzero(res)))
# softmax rows=8 cols=256
rows, cols = 8, 256
v = rng.standard_normal((rows, cols)).astype(np.float32)
res = go("softmax.spv", "softmax_f32", (rows, 1, 1),
         [w("s_v", v), w("s_sink", np.zeros(1, np.float32)), w("s_meta", np.array([cols, rows, 1], np.uint32))],
         Z16 + struct.pack("<I", 0), 0, np.float32)
if res is not None:
    e = np.exp(v - v.max(1, keepdims=True))
    e /= e.sum(1, keepdims=True)
    print("softmax got", res[:4], "want", e.ravel()[:4], "rowsum", res.reshape(rows, cols).sum(1)[:3])
# tile matmul 16x16x16
M = 16
A = rng.standard_normal((M, M)).astype(np.float32)
B = rng.standard_normal((M, M)).astype(np.float32)
res = go("tilematmul.spv", "tile_matmul_f32", (M // 8, M // 8, 1),
         [w("m_a", A), w("m_b", B), w("m_c", np.zeros((M, M), np.float32)), w("m_d", np.array([M, M, M], np.uint32))],
         Z16, 2, np.float32, "32 1 1")
if res is not None:
    C = A @ B
    print("matmul got", res[:4], "want", C.ravel()[:4], "maxerr", float(np.max(np.abs(res.reshape(M, M) - C))))
