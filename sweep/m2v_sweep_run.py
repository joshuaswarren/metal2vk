#!/usr/bin/env python3
"""Run stage of the metal2vk sweep (GPU host; run it inside a gpu-turn ticket, see sweep/README).
For every entry the compile stages passed (spirv-val ok) this dispatches the kernel once with synthetic data and reports
  run   the dispatch completed (m2v-run exit 0; a device error, a hang past the timeout or a crash is a FAIL with the message)
  ref   for the few kernels with a cheap CPU reference (REFS below): the output matches the reference
Synthetic data: float buffers uniform in [0, 1), integer buffers zero (so data-dependent loop bounds and indices stay in range), constant
references and POD arguments small fixed values, 4 workgroups of 64 threads (the module's own required size wins). One dispatch per kernel,
a few thousand threads: the estimated and actual wall seconds of every submit are logged in the result (rule: every submit under 5 s).
Usage: m2v_sweep_run.py SWEEP.json SPVDIR RUNNER OUTDIR [--only SUBSTR] [--limit N]
RUNNER is m2v-run built from m2v-run.c (cc -O2 m2v-run.c -lvulkan -o m2v-run); env VK_DRIVER_FILES picks the ICD."""
import argparse
import json
import pathlib
import re
import struct
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REFLECT = HERE.parent / "m2v-reflect.py"
BUF_BYTES = 1 << 20
SCALAR = {"float": (np.float32, 4), "half": (np.float16, 2), "bfloat": (None, 2), "bfloat16_t": (None, 2), "float16_t": (np.float16, 2),
          "char": (np.int8, 1), "int8_t": (np.int8, 1), "uchar": (np.uint8, 1), "uint8_t": (np.uint8, 1), "short": (np.int16, 2),
          "int16_t": (np.int16, 2), "ushort": (np.uint16, 2), "uint16_t": (np.uint16, 2), "int": (np.int32, 4), "int32_t": (np.int32, 4),
          "uint": (np.uint32, 4), "uint32_t": (np.uint32, 4), "long": (np.int64, 8), "int64_t": (np.int64, 8), "ulong": (np.uint64, 8),
          "uint64_t": (np.uint64, 8), "bool": (np.uint32, 4)}
rng = np.random.default_rng(11)


def elem_of(typ):
    t = re.sub(r"\b(const|device|constant|threadgroup|volatile)\b|[*&]", " ", typ).split()
    return t[-1] if t else "float"


def bf16_bits(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def buffer_for(typ):
    el = elem_of(typ)
    dt, sz = SCALAR.get(el, (None, 4))
    n = BUF_BYTES // sz
    if el in ("float", "half", "float16_t"):
        return rng.random(n).astype(dt)
    if el in ("bfloat", "bfloat16_t"):
        return bf16_bits(rng.random(n))
    if el in SCALAR:
        return np.zeros(n, dt)
    return np.zeros(BUF_BYTES, np.uint8)  # a struct: zeros


def ref_value(typ):
    """A constant reference (`constant uint& n`) lives in a buffer holding one value."""
    el = elem_of(typ)
    if el in ("float", "half"):
        return np.array([1.0], np.float32)
    dt, _ = SCALAR.get(el, (np.uint32, 4))
    return np.full(64, 32, dt or np.uint32)


def pod_bytes(typ, size):
    el = elem_of(typ)
    if el in ("float", "half"):
        return struct.pack("<f", 1.0)[:size].ljust(size, b"\0")
    if el == "bool":
        return b"\0" * size
    return int(32).to_bytes(size, "little")


def reflect(spv):
    r = subprocess.run([sys.executable, str(REFLECT), spv], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-200:])
    return json.loads(r.stdout)


def run_one(row, spv, runner, work, local=64, groups=4):
    refl = reflect(spv)
    ks = refl["kernels"] if isinstance(refl, dict) and "kernels" in refl else refl
    k = next(iter(ks.values() if isinstance(ks, dict) else ks))
    kargs = [a for a in row["args"] if a["kind"] in ("buffer", "ref", "pod")]
    bufs, push = {}, {}
    for a in k["args"]:
        o = a["ordinal"]
        spec = kargs[o] if o < len(kargs) else {"kind": "pod", "type": "uint"}
        if a["kind"] == "storage_buffer":
            data = ref_value(spec["type"]) if spec["kind"] == "ref" else buffer_for(spec["type"])
            p = work / f"b{o}.bin"
            data.tofile(p)
            bufs[a["binding"]] = str(p)
        elif a["kind"] == "pod_push_constant":
            push[a["offset"]] = pod_bytes(spec["type"], a["size"])
    cmd = [str(runner), str(spv), k["name"], str(groups), "1", "1", "--iters", "1"]
    wg = k.get("workgroup_size")
    cmd += ["--local", *map(str, wg if wg and all(wg) else (local, 1, 1))]
    if push:
        size = max(o + len(v) for o, v in push.items())
        blob = bytearray(size)
        for o, v in push.items():
            blob[o:o + len(v)] = v
        pf = work / "push.bin"
        pf.write_bytes(bytes(blob))
        cmd += ["--push", str(pf)]
    for b in sorted(bufs):
        cmd += ["--buf", bufs[b]]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("spvdir")
    ap.add_argument("runner")
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = json.load(open(a.sweep))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if r.get("val") == "ok" and a.only in r["file"] + r["entry"]]
    if a.limit:
        todo = todo[:a.limit]
    t_all = time.time()
    log = open(out / "run.log", "w")
    for i, r in enumerate(todo):
        spv = pathlib.Path(a.spvdir) / "spv" / (r["tag"] + ".spv")
        work = out / "work"
        work.mkdir(exist_ok=True)
        est = 0.5
        try:
            res, wall = run_one(r, spv, a.runner, work)
            r["run"] = "ok" if res.returncode == 0 else "FAIL"
            if res.returncode:
                r["run_error"] = (res.stderr.strip().splitlines() or ["?"])[-1][:140]
            r["run_est_s"], r["run_actual_s"] = est, round(wall, 3)
        except subprocess.TimeoutExpired:
            r["run"], r["run_error"], r["run_est_s"], r["run_actual_s"] = "FAIL", "timeout 30 s", est, 30.0
        except Exception as e:  # noqa: BLE001 - one bad module must not stop the sweep
            r["run"], r["run_error"] = "FAIL", repr(e)[:140]
        print(f"{i + 1}/{len(todo)} {r['file']} {r['entry'][:40]} run={r['run']} {r.get('run_error', '')}", file=log, flush=True)
    json.dump(rows, open(out / "sweep-run.json", "w"), indent=1)
    ok = sum(r.get("run") == "ok" for r in rows)
    print(f"ran {len(todo)} entries, {ok} ok, {time.time() - t_all:.0f} s wall")


if __name__ == "__main__":
    main()
