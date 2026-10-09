#!/usr/bin/env python3
"""omarchy-mlx side of the sibling-kernel parity harness.

For every kernel in sweep/parity_common.KERNELS this generates the seeded
inputs, dispatches the kernel through omarchy-mlx mx.fast.metal_kernel with
the exact source/header/template/grid the assembled artifacts pin, and saves
inputs and outputs as .npy for parity_compare.py.

Runs under the python of the mlx wheel venv, on the GPU host, through a guard
ticket ONLY. Never dispatches anything without the ticket.

  PARITY_ARTIFACTS  artifact root with the three assembler lanes (required;
                    the recorded build() calls are the source of the call
                    contract, so no signature is re-parsed here)
  PARITY_OUT        output directory (default ./parity-mlx-out)
  PARITY_SEED       input seed (default 1234)
  PARITY_ONLY       substring filter on kernel names
  PARITY_FAKE=1     CPU dry run: no mlx import, outputs come from the shared
                    fake kernel in parity_common (plumbing proof only)

Every submit logs estimated and actual seconds and refuses to launch above 5 s.

--check-assemblers validates the committed kernel table against the artifacts
without any mlx import: the three assemblers re-run into a temp dir (their
.metal output must be byte-identical to the artifacts) and every table entry
must match the recorded build() call and the .launch file.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import parity_common as pc

LANES = [("sibling-scan", "assemble_sibling_kernels"),
         ("qwen35-gdn", "assemble_gdn_kernels"),
         ("moe-gate-up", "assemble_m2_kernels")]
DTYPE_TO_MSL = {"bf16": "bfloat16_t", "f32": "float", "f16": "float16_t",
                "u32": "uint32_t", "i32": "int32_t", "i64": "int64_t"}
MAX_SUBMIT_S = 5.0


# ---------------------------------------------------------------------------
# assembler recording (shared by --check-assemblers and the real run)
# ---------------------------------------------------------------------------
def record_builds(artifacts):
    """Re-run the three assemblers with build() recorded, in an isolated
    subprocess. The assemblers install import stubs (mlx, numpy, ...) at
    module scope and leave them behind: in-process, every later
    `import mlx.core` in this process returns the stub instead of the wheel.
    -> ({kernel name: record}, {lane: {msl file name: sha256}})."""
    out = pathlib.Path(tempfile.mkstemp(prefix="parity-records-", suffix=".json")[1])
    try:
        r = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--record-builds", "--artifacts", str(artifacts), "--records", str(out)],
            capture_output=True, text=True, timeout=600)
        if r.returncode:
            raise SystemExit("assembler recording failed:\n" + r.stderr.strip()[-2000:])
        payload = json.loads(out.read_text())
    finally:
        out.unlink(missing_ok=True)
    return payload["records"], payload["shas"]


def record_builds_inprocess(artifacts, records_out):
    """Child of record_builds: runs in this file's interpreter with the
    assembler import stubs confined here; writes records and rebuilt-file
    sha256s as JSON."""
    import hashlib
    import importlib.util

    records, shas = {}, {}
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="parity-assemblers-"))
    try:
        for lane, modname in LANES:
            src = pathlib.Path(artifacts) / lane
            dst = tmp / lane
            dst.mkdir(parents=True)
            for py in src.glob("*.py"):
                shutil.copy(py, dst / py.name)
            spec = importlib.util.spec_from_file_location(modname, dst / f"{modname}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
            real_build = mod.build

            def recorded(kernel, header, source, tnames, tvalues, params,
                         grid, threads, _lane=lane, _rb=real_build):
                ps = []
                for p in params:
                    if len(p) == 3:  # sibling: (tcode, name, kind)
                        tcode, name, kind = p
                        ps.append((tcode, name, kind == "out", kind == "ref"))
                    else:           # gdn/m2: (mtype, name, is_output, scalar)
                        ps.append(tuple(p))
                records[kernel] = dict(lane=_lane, header=header, source=source,
                                       tnames=list(tnames), tvalues=list(tvalues),
                                       params=ps, grid=tuple(grid), threads=tuple(threads))
                return _rb(kernel, header, source, tnames, tvalues, params,
                           grid, threads)

            mod.build = recorded
            mod.main()
            mod.build = real_build
            shas[lane] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (dst / "msl").glob("*")}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    records_out.write_text(json.dumps(dict(records=records, shas=shas)))


def rendered(value):
    """Table template value -> the string the assembler rendered."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return DTYPE_TO_MSL.get(value, value)


def norm_tval(value):
    """Record template value -> rendered form (ints arrive as ints or strings)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def check_table(kernels, records, rebuilt, artifacts):
    """Validate the committed table against the assembler records and the
    artifact bytes. Raises SystemExit on the first drift."""
    seen = set()
    for k in kernels:
        rec = records.get(k["name"])
        if rec is None:
            raise SystemExit(f"{k['name']}: no assembler builds this kernel")
        seen.add(k["name"])
        # the sibling lane renders template values at build time, the gdn/m2
        # lanes pass pre-rendered strings; compare both in rendered form
        want_t = [(n, norm_tval(rendered(v))) for n, v in k["tmpl"]]
        got_t = [(n, norm_tval(v)) for n, v in zip(rec["tnames"], rec["tvalues"])]
        if got_t != want_t:
            raise SystemExit(f"{k['name']}: template {got_t} != table {want_t}")
        want_names = [(DTYPE_TO_MSL.get(i[1], i[1]), i[0]) for i in k["inputs"]]
        got_names = []
        for p in rec["params"]:
            if p[2]:
                continue
            got_names.append((DTYPE_TO_MSL.get(p[0], p[0]), p[1]))
            # the assemblers emit shape/stride/ndim buffers after any input
            # whose source references the suffix
            for suffix, t in (("_shape", "int32_t"), ("_strides", "int64_t"),
                              ("_ndim", "int32_t")):
                if f"{p[1]}{suffix}" in rec["source"]:
                    got_names.append((t, p[1] + suffix))
        if want_names != got_names:
            raise SystemExit(f"{k['name']}: inputs {got_names} != table {want_names}")
        want_out = [(DTYPE_TO_MSL.get(i[1], i[1]), i[0]) for i in k["outputs"]]
        got_out = [(DTYPE_TO_MSL.get(p[0], p[0]), p[1]) for p in rec["params"] if p[2]]
        if want_out != got_out:
            raise SystemExit(f"{k['name']}: outputs {got_out} != table {want_out}")
        want_grid = tuple(k["launch_grid"] or k["grid"])
        if tuple(rec["grid"]) != want_grid:
            raise SystemExit(f"{k['name']}: grid {rec['grid']} != table {want_grid}")
        if tuple(rec["threads"]) != tuple(k["tg"]):
            raise SystemExit(f"{k['name']}: threads {rec['threads']} != table {k['tg']}")
        metal = f"{k['name']}.metal"
        art = pathlib.Path(artifacts) / rec["lane"] / "msl" / metal
        rebuilt_sha = rebuilt[rec["lane"]].get(metal)
        if rebuilt_sha != hashlib.sha256(art.read_bytes()).hexdigest():
            raise SystemExit(f"{k['name']}: rebuilt {metal} differs from the artifact")
        launch = art.with_suffix(".launch")
        g = launch.read_text().split()
        want = [str(x) for x in want_grid] + [str(x) for x in k["tg"]] + \
               [str(len(k["outputs"]))]
        if g != want:
            raise SystemExit(f"{k['name']}: .launch {g} != table {want}")
    missing = set(records) - seen
    if missing:
        raise SystemExit(f"assembler kernels missing from the table: {sorted(missing)}")
    print(f"table check ok: {len(kernels)} kernels, byte-identical rebuild, "
          f".launch and template values match")


# ---------------------------------------------------------------------------
# input/output exchange
# ---------------------------------------------------------------------------
def gen_kernel_inputs(kern, seed, outdir):
    files = {}
    for i, spec in enumerate(kern["inputs"]):
        data = pc.gen_input(kern["name"], i, spec, seed)
        pc.save_arr(outdir, spec[0], spec[1], data, spec[2])
        files[spec[0]] = (spec, data)
    meta = {"kernel": kern["name"], "seed": seed, "grid": list(kern["grid"]),
            "tg": list(kern["tg"]),
            "inputs": [{"name": s[0], "dtype": s[1], "shape": list(s[2]),
                        "gen": s[3]} for s in kern["inputs"]],
            "outputs": [{"name": s[0], "dtype": s[1], "shape": list(s[2])}
                        for s in kern["outputs"]]}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    return files


def mx_array(code, data_bytes):
    """buffer bytes -> mlx array with the logical dtype (bit-exact)."""
    import mlx.core as mx

    if code == "bf16":
        bits = np.frombuffer(data_bytes, np.uint16).astype(np.uint32)
        f32 = (bits << 16).view(np.float32)
        return mx.array(f32).astype(mx.bfloat16)
    dt = {"f32": np.float32, "u32": np.uint32, "i32": np.int32,
          "i64": np.int64, "f16": np.float16}[code]
    return mx.array(np.frombuffer(data_bytes, dt))


def mx_dtype(code):
    import mlx.core as mx

    return {"bf16": mx.bfloat16, "f32": mx.float32, "f16": mx.float16,
            "u32": mx.uint32, "i32": mx.int32, "i64": mx.int64}[code]


def mx_template(tvals):
    import mlx.core as mx

    mapped = {"bf16": mx.bfloat16, "f32": mx.float32, "f16": mx.float16,
              "u32": mx.uint32, "i32": mx.int32, "i64": mx.int64}
    return [(n, mapped.get(v, v)) for n, v in tvals]


def submit_seconds(nbytes):
    return max(0.02, round(nbytes / 2e10, 4))


# ---------------------------------------------------------------------------
# run modes
# ---------------------------------------------------------------------------
def run_real(kernels, records, outdir, seed):
    import mlx.core as mx

    statuses, rows = {}, []
    for kern in kernels:
        name = kern["name"]
        kdir = outdir / name
        kdir.mkdir(parents=True, exist_ok=True)
        rec = records[name]
        files = gen_kernel_inputs(kern, seed, kdir)
        in_bytes_total = sum(len(d) for _s, d in files.values())
        out_bytes = sum(int(np.prod(o[2])) * pc.DTYPES[o[1]][0] for o in kern["outputs"])
        est = submit_seconds(in_bytes_total + out_bytes)
        t = {"est_s": est, "actual_s": None}
        try:
            if est > MAX_SUBMIT_S:
                raise RuntimeError(f"estimated submit {est} s above {MAX_SUBMIT_S} s")
            kernel = mx.fast.metal_kernel(
                name=name,
                input_names=[i[0] for i in kern["inputs"]],
                output_names=[o[0] for o in kern["outputs"]],
                source=rec["source"], header=rec["header"])
            args = [mx_array(spec[1], data).reshape(spec[2])
                    for spec, data in files.values()]
            t0 = time.perf_counter()
            outs = kernel(
                inputs=args,
                template=mx_template(kern["tmpl"]),
                grid=kern["grid"],
                threadgroup=kern["tg"],
                output_shapes=[list(o[2]) for o in kern["outputs"]],
                output_dtypes=[mx_dtype(o[1]) for o in kern["outputs"]])
            mx.eval(outs)
            t["actual_s"] = round(time.perf_counter() - t0, 4)
            if t["actual_s"] > MAX_SUBMIT_S:
                raise RuntimeError(f"actual submit {t['actual_s']} s above {MAX_SUBMIT_S} s")
            for o, arr in zip(kern["outputs"], outs):
                if o[1] == "bf16":
                    bits = np.frombuffer(memoryview(arr.view(mx.uint16)), np.uint16)
                else:
                    bits = np.frombuffer(memoryview(arr), pc.DTYPES[o[1]][1])
                np.save(kdir / f"out_{o[0]}.npy", bits.reshape(o[2]))
            statuses[name] = "ok"
        except Exception as e:  # noqa: BLE001 - a refusal is data, not a crash
            statuses[name] = "refused"
            t["error"] = repr(e)[:300]
        t.update(kernel=name)
        rows.append(t)
        print(f"{name}: {statuses[name]} est={t['est_s']} actual={t['actual_s']}"
              + (f" err={t['error'][:80]}" if "error" in t else ""), flush=True)
    (outdir / "mlx_status.json").write_text(json.dumps(statuses, indent=1))
    (outdir / "mlx_timing.json").write_text(json.dumps(rows, indent=1))


def run_fake(kernels, outdir, seed):
    statuses = {}
    for kern in kernels:
        kdir = outdir / kern["name"]
        kdir.mkdir(parents=True, exist_ok=True)
        files = gen_kernel_inputs(kern, seed, kdir)
        outs = pc.fake_outputs(kern, {n: d for n, (_s, d) in files.items()})
        for oname, obytes in outs.items():
            spec = next(o for o in kern["outputs"] if o[0] == oname)
            pc.save_arr(kdir, f"out_{oname}", spec[1], obytes, spec[2])
        statuses[kern["name"]] = "ok"
        print(f"{kern['name']}: fake ok", flush=True)
    (outdir / "mlx_status.json").write_text(json.dumps(statuses, indent=1))
    (outdir / "mlx_timing.json").write_text(json.dumps(
        [{"kernel": k["name"], "est_s": 0.0, "actual_s": 0.0, "fake": True}
         for k in kernels], indent=1))


def check_generation(kernels, seed):
    """Generate every input of every kernel: catches non-ndarray leaks and
    shape errors without any dispatch."""
    n = 0
    for kern in kernels:
        for i, spec in enumerate(kern["inputs"]):
            data = pc.gen_input(kern["name"], i, spec, seed)
            want = int(np.prod(spec[2])) * pc.DTYPES[spec[1]][0]
            if not isinstance(data, bytes) or len(data) != want:
                raise SystemExit(f"{kern['name']} input {spec[0]}: "
                                 f"{type(data).__name__} of {len(data)} bytes, "
                                 f"expected {want}")
            n += 1
    print(f"generation check ok: {n} input buffers over {len(kernels)} kernels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-assemblers", action="store_true",
                    help="validate the table against the artifacts, then exit")
    ap.add_argument("--artifacts", default=os.environ.get("PARITY_ARTIFACTS", ""))
    ap.add_argument("--record-builds", action="store_true",
                    help=argparse.SUPPRESS)  # child of record_builds()
    ap.add_argument("--records", default="")
    a = ap.parse_args()

    if a.record_builds:
        if not (a.artifacts and a.records):
            raise SystemExit("--record-builds needs --artifacts and --records")
        record_builds_inprocess(a.artifacts, pathlib.Path(a.records))
        return

    outdir = pc.env_path("PARITY_OUT", "./parity-mlx-out")
    seed = int(os.environ.get("PARITY_SEED", "1234"))
    only = os.environ.get("PARITY_ONLY", "")
    kernels = [k for k in pc.KERNELS if only in k["name"]]

    if a.check_assemblers:
        if not a.artifacts:
            raise SystemExit("--check-assemblers needs --artifacts / PARITY_ARTIFACTS")
        records, rebuilt = record_builds(pathlib.Path(a.artifacts))
        check_table(pc.KERNELS, records, rebuilt, pathlib.Path(a.artifacts))
        check_generation(pc.KERNELS, seed)
        return

    if os.environ.get("PARITY_FAKE") == "1":
        outdir.mkdir(parents=True, exist_ok=True)
        check_generation(kernels, seed)
        run_fake(kernels, outdir, seed)
        return

    if not a.artifacts:
        raise SystemExit("real runs need --artifacts / PARITY_ARTIFACTS")
    records, _ = record_builds(pathlib.Path(a.artifacts))
    outdir.mkdir(parents=True, exist_ok=True)
    run_real(kernels, records, outdir, seed)


if __name__ == "__main__":
    main()
