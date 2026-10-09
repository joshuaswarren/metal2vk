#!/usr/bin/env python3
"""metal2vk bench: translated (m2v-run) vs hand (mlx) on one table.

Usage (on the GPU host, under the GPU queue wrapper or flock):
    m2v-bench.py OUTDIR [--spvdir DIR] [--runner PATH] [--cases c,c,...]
                        [--device TAG] [--icd PATH] [--mlx-python PATH]

Prints one table per kernel+dtype and writes OUTDIR/results-<device>-<date>.json.
Registered cases: activation (f32/f16), softmax (f32/f16), tilematmul (256,1024),
plus anything else that has merged to main as cases/*.cl with a driver here.
`--device g14c` only changes the output tag and the ICD env (M2V_ICD, or the tag
lookup in bench/devices.json); without it the tag is derived from vulkaninfo
deviceName. Receipts carry the device name, driver string and relative paths only.
"""
import argparse
import datetime
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (case, dtype, rows/cols or n), same shapes as m2v-test.py / m2v-mlx-base.py.
CASES = [
    ("activation", "f32", 1 << 23),
    ("activation", "f16", 1 << 23),
    ("softmax", "f32", 4096),
    ("softmax", "f16", 4096),
    ("tilematmul", "f32", 256),
    ("tilematmul", "f32", 1024),
    ("tilematmul_rt", "f32", 256),
    ("tilematmul_rt", "f32", 1024),
]

NPDT = {"f32": np.float32, "f16": np.float16}
TOL = {"f32": 2e-5, "f16": 2e-3}  # activation/softmax; matmul uses its own
LOCAL = {"activation": (256, 1, 1), "softmax": (256, 1, 1), "tilematmul": (32, 1, 1),
         "tilematmul_rt": (32, 1, 1)}


def device_info():
    """(tag, deviceName, driverInfo) from vulkaninfo; tag is `g13c` from `Apple M1 Max (G13C C0)`."""
    try:
        out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True,
                             timeout=30).stdout
        name = (re.search(r"deviceName\s*=\s*(.+)", out) or [None, ""])[1].strip() if "deviceName" in out else ""
        drv = (re.search(r"driverInfo\s*=\s*(.+)", out) or [None, ""])[1].strip()
        t = re.search(r"\(([A-Za-z0-9]+)", name)
        tag = (t.group(1) if t else name or "unknown").lower().replace(" ", "")
        return tag, name, drv
    except Exception:
        return "unknown", "", ""


def load_device_map():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devices.json")
    try:
        return json.load(open(p))
    except Exception:
        return {}


def run_once(runner, spv, entry, grid, bufs, push, local, iters, dump=None):
    cmd = [runner, spv, entry, *map(str, grid), "--iters", str(iters),
           "--local", *map(str, local)]
    if push is not None:
        pf = os.path.join(os.path.dirname(bufs[0]), entry + ".push")
        open(pf, "wb").write(push)
        cmd += ["--push", pf]
    for b in bufs:
        cmd += ["--buf", b]
    if dump:
        cmd += ["--dump", dump]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode:
        return None, r.stderr.strip()[-400:]
    line = next((l for l in r.stdout.splitlines() if l.startswith("iters=")), None)
    if not line:
        return None, r.stdout + r.stderr
    return (float(line.split("per_dispatch_us=")[1].split()[0]),
            float(line.split("wall_us=")[1])), r.stderr.strip()


def timed(runner, spv, entry, grid, bufs, push, local):
    """Pilot once, then size the batch so the submit stays near 2 s (same rule as m2v-test.py)."""
    p1, err = run_once(runner, spv, entry, grid, bufs, push, local, 1)
    if p1 is None:
        return None, err
    wall1 = p1[1]
    iters = int(max(1, min(500, 2e6 / max(wall1, 1.0))))
    t, err = run_once(runner, spv, entry, grid, bufs, push, local, iters)
    if t is None:
        return None, err
    ts_per, wall_total = t
    wall_per = wall_total / iters
    per = max(ts_per, wall_per) if iters < 50 else wall_per
    return per, f"iters={iters} ts_per_us={ts_per:.2f} wall_per_us={wall_per:.2f}"


def w(outdir, name, arr):
    p = os.path.join(outdir, name)
    arr.tofile(p)
    return p


def silu(x):
    return x / (1.0 + np.exp(-x))


# --- translated-side case drivers (mirror m2v-test.py) ----------------------

def translated(outdir, runner, spvdir, case, dt, n, grid=None, local=None):
    npdt = NPDT[dt]
    rng = np.random.default_rng(7)
    local = local or LOCAL[case]
    if case == "activation":
        x = rng.standard_normal(n).astype(npdt)
        bufs = [w(outdir, f"act_in_{dt}.bin", x), w(outdir, f"act_out_{dt}.bin", np.zeros(n, npdt)),
                w(outdir, "act_n.bin", np.array([n], np.uint32)),
                w(outdir, "act_type.bin", np.array([0], np.uint32))]
        push = struct.pack("<4I", 0, 0, 0, 0) + struct.pack("<I", 0)
        entry = f"activation_{dt}"
        grid = grid or ((n + local[0] - 1) // local[0], 1, 1)
        t, info = timed(runner, f"{spvdir}/activation.spv", entry, grid, bufs, push, local)
        run_once(runner, f"{spvdir}/activation.spv", entry, grid, bufs, push, local, 1,
                 dump=f"1:{os.path.join(outdir, f'act_res_{dt}.bin')}")
        got = np.fromfile(os.path.join(outdir, f"act_res_{dt}.bin"), npdt)
        ref = silu(x.astype(np.float64))
        err = float(np.max(np.abs(got.astype(np.float64) - ref) / (1.0 + np.abs(ref))))
        ok, tol = err < TOL[dt], TOL[dt]
        gbps = (None if t is None else 2 * n * npdt().itemsize / (t * 1e3))
        return dict(us=t, info=info, err=err, ok=ok, tol=tol, metric="GBps", value=gbps)
    if case == "softmax":
        rows = cols = n
        x = (rng.standard_normal((rows, cols)) * 3).astype(npdt)
        bufs = [w(outdir, f"sm_vals_{dt}.bin", x.copy()), w(outdir, f"sm_sinks_{dt}.bin", np.zeros(1, npdt)),
                w(outdir, "sm_meta.bin", np.array([cols, rows, 1], np.uint32))]
        push = struct.pack("<4I", 0, 0, 0, 0) + struct.pack("<I", 0)
        entry, grid = f"softmax_{dt}", (rows, 1, 1)
        t, info = timed(runner, f"{spvdir}/softmax.spv", entry, grid, bufs, push, LOCAL["softmax"])
        w(outdir, f"sm_vals_{dt}.bin", x.copy())  # timed pass softmaxes in place
        run_once(runner, f"{spvdir}/softmax.spv", entry, grid, bufs, push, LOCAL["softmax"], 1,
                 dump=f"0:{os.path.join(outdir, f'sm_res_{dt}.bin')}")
        got = np.fromfile(os.path.join(outdir, f"sm_res_{dt}.bin"), npdt).reshape(rows, cols).astype(np.float64)
        xr = x.astype(np.float64)
        ref = np.exp(xr - xr.max(1, keepdims=True))
        ref /= ref.sum(1, keepdims=True)
        err = float(np.max(np.abs(got - ref)))
        tol = 1e-5 if dt == "f32" else TOL[dt]
        ok = err < tol
        gbps = (None if t is None else 2 * rows * cols * npdt().itemsize / (t * 1e3))
        return dict(us=t, info=info, err=err, ok=ok, tol=tol, metric="GBps", value=gbps)
    if case == "tilematmul":
        M = N = K = n
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        bufs = [w(outdir, "mm_a.bin", A), w(outdir, "mm_b.bin", B),
                w(outdir, "mm_c.bin", np.zeros((M, N), np.float32)),
                w(outdir, "mm_dims.bin", np.array([M, N, K], np.uint32))]
        grid = (N // 8, M // 8, 1)
        push = struct.pack("<4I", 0, 0, 0, 0)
        entry = "tile_matmul_f32"
        t, info = timed(runner, f"{spvdir}/tilematmul.spv", entry, grid, bufs, push, LOCAL["tilematmul"])
        run_once(runner, f"{spvdir}/tilematmul.spv", entry, grid, bufs, push, LOCAL["tilematmul"], 1,
                 dump=f"2:{os.path.join(outdir, 'mm_res.bin')}")
        got = np.fromfile(os.path.join(outdir, "mm_res.bin"), np.float32).reshape(M, N).astype(np.float64)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
        ok = err < 1e-5
        gf = (None if t is None else 2.0 * M * N * K / (t * 1e3))
        return dict(us=t, info=info, err=err, ok=ok, tol=1e-5, metric="GFLOPs", value=gf)
    if case == "tilematmul_rt":
        return rt_rows(outdir, runner, spvdir, n)
    raise ValueError(case)


RT_SHAPES = (("rt2x2", 2, 2), ("rt4x2", 4, 2), ("rt4x4", 4, 4))


def rt_rows(outdir, runner, spvdir, n):
    """One subgroup owns an MR x NR block of 8x8 tiles (cases/tilematmul_rt.cl); one row per shape."""
    M = N = K = n
    rng = np.random.default_rng(7)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bufs = [w(outdir, "mm_a.bin", A), w(outdir, "mm_b.bin", B),
            w(outdir, "mm_c.bin", np.zeros((M, N), np.float32)),
            w(outdir, "mm_dims.bin", np.array([M, N, K], np.uint32))]
    push = struct.pack("<4I", 0, 0, 0, 0)
    ref = A.astype(np.float64) @ B.astype(np.float64)
    out = {}
    for name, mr, nr in RT_SHAPES:
        entry = f"tile_matmul_{name}_f32"
        grid = (N // (8 * nr), M // (8 * mr), 1)
        t, info = timed(runner, f"{spvdir}/tilematmul_rt.spv", entry, grid, bufs, push, LOCAL["tilematmul"])
        run_once(runner, f"{spvdir}/tilematmul_rt.spv", entry, grid, bufs, push, LOCAL["tilematmul"], 1,
                 dump=f"2:{os.path.join(outdir, 'mm_res.bin')}")
        got = np.fromfile(os.path.join(outdir, "mm_res.bin"), np.float32).reshape(M, N).astype(np.float64)
        err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
        out[f"{name}_{n}"] = dict(us=t, info=info, err=err, ok=bool(err < 1e-5), tol=1e-5, metric="GFLOPs",
                                  value=(None if t is None else 2.0 * M * N * K / (t * 1e3)))
    return out


# --- hand-side (mlx) case drivers -------------------------------------------

def mlx_case(case, dt, n, batch=40):
    import mlx.core as mx
    import mlx.nn as nn
    npdt = NPDT[dt]
    if case == "activation":
        x = mx.random.normal((n,)).astype(mx.float32 if dt == "f32" else mx.float16)
        mx.eval(x)
        return dict(fused=lambda: nn.silu(x), unfused=lambda: x * mx.sigmoid(x), bytes=2 * n * npdt().itemsize)
    if case == "softmax":
        x = (mx.random.normal((n, n)) * 3).astype(mx.float32 if dt == "f32" else mx.float16)
        mx.eval(x)
        return dict(fused=lambda: mx.softmax(x, axis=-1), unfused=None, bytes=2 * n * n * npdt().itemsize)
    if case in ("tilematmul", "tilematmul_rt"):
        a = mx.random.normal((n, n))
        b = mx.random.normal((n, n))
        mx.eval(a, b)
        return dict(fused=lambda: a @ b, unfused=None, flops=2.0 * n * n * n)
    raise ValueError(case)


def hand(case, dt, n, batch=40):
    """Wall time of a batch of independent evals / batch (one sync), best of 3, same as m2v-mlx-base.py."""
    import time
    import mlx.core as mx  # noqa: F401  (fail loudly here if mlx is missing)
    spec = mlx_case(case, dt, n, batch)
    out = {}
    for key, fn in (("fused", spec["fused"]), ("unfused", spec.get("unfused"))):
        if fn is None:
            continue
        mx.eval(fn())
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            outs = [fn() for _ in range(batch)]
            mx.eval(outs)
            dt_s = (time.perf_counter() - t0) / batch * 1e6
            best = dt_s if best is None or dt_s < best else best
        r = {"us": round(best, 2)}
        if "bytes" in spec:
            r["GBps"] = round(spec["bytes"] / (best * 1e3), 1)
        if "flops" in spec:
            r["GFLOPs"] = round(spec["flops"] / (best * 1e3), 1)
        out[key] = r
    return out


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--spvdir", default=os.environ.get("M2V_SPVDIR", "bench-out"))
    ap.add_argument("--runner", default=os.environ.get("M2V_RUNNER", "m2v-run"))
    ap.add_argument("--cases", default=",".join(sorted({c for c, _, _ in CASES})))
    ap.add_argument("--device", default=None, help="output tag + ICD env override, e.g. g14c")
    ap.add_argument("--icd", default=None, help="VK_DRIVER_FILES value (overrides devices.json)")
    ap.add_argument("--mlx-python", default=sys.executable, help="python with mlx (defaults to this one)")
    ap.add_argument("--skip-hand", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = set(args.cases.split(","))

    tag, devname, drv = (args.device or device_info()[0]).lower(), None, ""
    if not args.device:
        tag, devname, drv = device_info()
    dm = load_device_map().get(tag, {})
    icd = args.icd or os.environ.get("M2V_ICD") or dm.get("icd") or os.environ.get("VK_DRIVER_FILES")
    # the tag switch only changes the output tag and the ICD env (M2V_ICD / --icd / devices.json)
    if args.icd or os.environ.get("M2V_ICD") or (dm.get("icd") and not os.environ.get("VK_DRIVER_FILES")):
        os.environ["VK_DRIVER_FILES"] = icd

    date = datetime.date.today().isoformat()
    # receipts carry device name, driver string and relative paths only (no host names or home paths)
    results = {"device": tag, "device_name": devname, "driver": drv, "icd": "M2V_ICD" if icd else None,
               "date": date, "runner": os.path.basename(args.runner),
               "spvdir": os.path.relpath(args.spvdir) if not os.path.isabs(args.spvdir) else os.path.basename(args.spvdir),
               "rows": {}}

    print(f"# metal2vk bench  device={tag} ({devname or 'tag override'})  driver={drv or 'n/a'}  icd={'M2V_ICD' if icd else 'system'}")
    hdr = f"{'kernel':<22}{'translated':>14}{'hand':>12}{'ratio':>8}  {'check':<18}{'metric':>22}"
    print(hdr)
    print("-" * len(hdr))

    mlxpy = [args.mlx_python, os.path.join(REPO, "bench", "m2v-mlx-row.py")]
    for case, dt, n in CASES:
        key = f"{case}_{dt}" + (f"_{n}" if case in ("tilematmul", "tilematmul_rt") else "")
        if case not in want:
            continue
        spv = os.path.join(args.spvdir, f"{case}.spv")
        if not os.path.exists(spv):
            results["rows"][key] = {"skip": f"no {os.path.basename(spv)} in spvdir (compile failed or case not merged)"}
            print(f"{key:<22}{'-':>14}{'-':>12}{'-':>8}  {'-':<18}  skip")
            continue
        got = translated(args.outdir, args.runner, args.spvdir, case, dt, n)
        rows = got if case == "tilematmul_rt" else {key: got}
        handrow = None
        if not args.skip_hand:
            r = subprocess.run(mlxpy + [case, dt, str(n)], capture_output=True, text=True, timeout=600)
            if r.returncode == 0:
                handrow = json.loads(r.stdout)
            elif case == "tilematmul_rt":
                r = subprocess.run(mlxpy + ["tilematmul", dt, str(n)], capture_output=True, text=True, timeout=600)
                handrow = json.loads(r.stdout) if r.returncode == 0 else None
        for rkey, row in rows.items():
            row["dtype"], row["n"] = dt, n
            if handrow is not None:
                row["hand"] = handrow
            elif not args.skip_hand:
                row["hand_err"] = "mlx run failed"
            results["rows"][rkey] = row
            tus = row.get("us")
            h = row.get("hand") or {}
            # comparator: the fastest hand figure available (fused silu vs unfused differ a lot;
            # the faster one is the honest "best hand implementation" bar)
            hk = min((k for k in h if isinstance(h[k], dict) and "us" in h[k]),
                     key=lambda k: h[k]["us"], default=None)
            hus = h[hk]["us"] if hk else None
            ratio = (tus / hus) if (tus and hus) else None
            chk = f"err={row['err']:.2e} {'OK' if row['ok'] else 'FAIL'}"
            m = f"{row['metric']}={row['value']:.1f}" if row.get("value") else ""
            hm = ""
            if hk:
                v = h[hk]
                hm = f"{next((k for k in ('GBps', 'GFLOPs') if k in v), '')}={v.get('GBps') or v.get('GFLOPs')}"
            print(f"{rkey:<22}{tus if tus else -1:>14.1f}{hus if hus else -1:>12.1f}"
                  f"{ratio if ratio else -1:>8.2f}x  {chk:<18}{m:>12} {hm:>10}")

    out = os.path.join(args.outdir, f"results-{tag}-{date}.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
