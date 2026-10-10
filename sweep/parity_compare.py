#!/usr/bin/env python3
"""metal2vk side of the sibling-kernel parity harness: dispatch every compiled
module through m2v-run with the same inputs the mlx side used, then compare
outputs and write the parity table.

Runs on the host with the runner, through a guard ticket ONLY (except with
--fake, the CPU dry run with a fake runner that proves the plumbing).

  PARITY_SWEEP    sweep result json (default ./t1.json); val != ok -> skipped
  PARITY_SPV_DIR  sweep output dir with spv/<tag>.spv (default ./OUT)
  PARITY_RUNNER   m2v-run binary (default ./m2v-run)
  PARITY_MLX_OUT  parity_mlx.py output directory (default ./parity-mlx-out)
  PARITY_OUT      this side's output directory (default ./parity-compare-out)
  PARITY_TSV      table path (default <PARITY_OUT>/parity.tsv)
  PARITY_SEED     input seed, must match the mlx side (default 1234)
  PARITY_ONLY     substring filter on kernel names

--fake simulates the dispatch with the shared fake kernel in parity_common
(no Vulkan, no GPU) and flips one byte of one kernel so the mismatch path is
exercised end to end.

Output buffers are poisoned with a fixed pattern before the dispatch so an
unwritten region is detected. Integers compare exactly, floats by relative
tolerance (bf16 1e-2, f32 1e-5); max abs/rel errors and the first differing
index land in the table. Every submit logs estimated and actual seconds and
refuses to launch above 5 s.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parity_common as pc
import parity_refs

MAX_SUBMIT_S = 5.0
REFLECT = HERE.parent / "m2v-reflect.py"
POISON = {"bf16": b"\xc1\x7f", "f32": b"\xde\xc0\xfc\x7f",
          "u32": b"\xde\xc0\xad\xde", "i32": b"\xde\xc0\xad\xde"}


def reflect(spv):
    r = subprocess.run([sys.executable, str(REFLECT), str(spv)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-200:])
    return json.loads(r.stdout)


def first_kernel(refl):
    ks = refl["kernels"] if isinstance(refl, dict) and "kernels" in refl else refl
    return ks[0] if isinstance(ks, list) else next(iter(ks.values()))


def poison_bytes(code, shape):
    size = int(np.prod(shape)) * pc.DTYPES[code][0]
    unit = POISON[code]
    return unit * (size // len(unit))


def const_bytes(spec):
    """Exact bytes of a c:[...] constant/ref buffer (seed-independent)."""
    return pc.gen_input("const", 0, spec, 0)


def dispatch(kern, row, spv, entry, inputs, runner, work, fake, perturb):
    """One dispatch. inputs: {name: bytes}.
    -> (status, error, {output name: bytes}, actual s)."""
    reflk = first_kernel(reflect(spv))
    work.mkdir(parents=True, exist_ok=True)
    out_by_name = {o[0]: o for o in kern["outputs"]}
    in_by_name = {i[0]: i for i in kern["inputs"]}
    bufs = {}
    for a in reflk["args"]:
        if a["kind"] != "storage_buffer":
            continue
        o = a["ordinal"]
        name = row["args"][o]["name"] if o < len(row["args"]) else f"arg{o}"
        if name in inputs:
            data = inputs[name]
        elif name in out_by_name:
            data = poison_bytes(out_by_name[name][1], out_by_name[name][2])
        elif name in in_by_name:
            data = const_bytes(in_by_name[name])
        else:
            return "FAIL", f"no data for buffer {name} (arg {o})", {}, 0.0
        p = work / f"b{o}.bin"
        p.write_bytes(data)
        bufs[a["binding"]] = (p, name)
    if len(bufs) > 64:
        return "FAIL", f"{len(bufs)} buffers exceed the m2v-run cap", {}, 0.0

    order = sorted(bufs)
    dump_paths = {}
    for oname in out_by_name:
        binding = next(b for b, (_p, n) in bufs.items() if n == oname)
        dump_paths[oname] = work / f"dump_{oname}.bin"
    if fake:
        outs = pc.fake_outputs(kern, inputs)
        for oname, p in dump_paths.items():
            data = outs[oname]
            if perturb and oname == perturb[0]:
                data = data[:-1] + bytes([data[-1] ^ perturb[1]])
            p.write_bytes(data)
        return "ok", "", {n: p.read_bytes() for n, p in dump_paths.items()}, 0.0

    # kern["grid"] is the launch grid in THREADS (mx.fast.metal_kernel semantics, which the mlx side passes through);
    # m2v-run dispatches workgroup COUNTS, so divide by the threadgroup size (rounding up, as dispatchThreads does).
    launch = kern.get("launch_grid") or kern["grid"]  # the same thread grid the mlx side launches
    groups = [-(-int(g) // int(t)) for g, t in zip(launch, kern["tg"])]
    cmd = [str(runner), str(spv), entry, *map(str, groups),
           "--iters", "1", "--local", *map(str, kern["tg"])]
    for b in order:
        cmd += ["--buf", str(bufs[b][0])]
    for oname, p in dump_paths.items():
        binding = next(b for b, (_p, n) in bufs.items() if n == oname)
        cmd += ["--dump", f"{order.index(binding)}:{p}"]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "FAIL", "timeout 20 s", {}, 20.0
    wall = time.perf_counter() - t0
    if r.returncode:
        return "FAIL", (r.stderr.strip().splitlines() or ["?"])[-1][:140], {}, wall
    return "ok", "", {n: p.read_bytes() for n, p in dump_paths.items()}, wall


def ref_outputs(kern, kdir):
    """numpy reference outputs for a refused kernel (saved-output form)."""
    inputs = {}
    for spec in kern["inputs"]:
        inputs[spec[0]] = np.load(kdir / f"{spec[0]}.npy")
    return _ref_outputs(kern, inputs, parity_refs.REFS[kern["name"]])


def _ref_outputs(kern, inputs, fn):
    outs = {}
    for name, code, shape, *_ in kern["outputs"]:
        vals = np.asarray(fn(kern, inputs)[name]).reshape(-1)
        dt, npdt = pc.DTYPES[code]
        if npdt is None:
            assert vals.dtype == np.uint16, f"{name}: reference must give bf16 bits"
            bits = vals
        elif code == "f32":
            bits = vals.astype(np.float32)
        else:
            bits = vals.astype(npdt)
        outs[name] = bits
    return outs


def compare_only(mlx_out, cmp_out, only, tsv):
    """Compare saved metal2vk outputs against the references. No runner, no
    GPU, no sweep json: mlx-out supplies the inputs (and the reference where
    the omarchy path ran), cmp-out supplies this side's saved outputs."""
    rows = []
    for kern in pc.KERNELS:
        name = kern["name"]
        if only and only not in name:
            continue
        rec = dict(kernel=name, status="", note=kern.get("note") or "",
                   max_abs="", max_rel="", first_diff="", n_diff="", n_nan="",
                   tol="", run_est_s="", run_actual_s="")
        kdir = mlx_out / name
        missing = [s[0] for s in kern["inputs"]
                   if not (kdir / f"{s[0]}.npy").exists()]
        if missing:
            rec["status"] = "skipped"
            rec["note"] += f"; mlx inputs missing: {missing[0]}"
            rows.append(rec)
            print(f"{name}: skipped (no saved inputs)", flush=True)
            continue
        got_all, gone = {}, []
        for o in kern["outputs"]:
            p = cmp_out / name / f"out_{o[0]}.npy"
            if p.exists():
                got_all[o[0]] = np.load(p)
            else:
                gone.append(o[0])
        if gone:
            rec["status"] = "no-cmp-output"
            rec["note"] += "; stage B saved no dump for " + gone[0]
            rows.append(rec)
            print(f"{name}: no-cmp-output ({gone[0]})", flush=True)
            continue
        mstat = json.loads((mlx_out / "mlx_status.json").read_text()).get(name, "missing") \
            if (mlx_out / "mlx_status.json").exists() else "missing"
        if mstat == "refused":
            if name not in parity_refs.REFS:
                rec["status"] = "no-reference"
                rows.append(rec)
                print(f"{name}: no-reference", flush=True)
                continue
            want_all = ref_outputs(kern, kdir)
            src = "reference"
            sec = parity_refs.SECONDARY.get(name)
            if sec is not None:
                # labelled secondary: the composed-op fallback reference.
                # The primary judge is the assembled-MSL reference; a
                # composed-op disagreement is a finding about the oMLX
                # fallback's semantics, and the MSL side is the
                # Metal-intended behaviour.
                try:
                    inputs = {}
                    for spec in kern["inputs"]:
                        inputs[spec[0]] = np.load(kdir / f"{spec[0]}.npy")
                    sworst = "match"
                    for oname, ocode, *_ in kern["outputs"]:
                        want = _ref_outputs(
                            kern, inputs, sec)[oname].reshape(-1)
                        r = pc.compare(ocode, got_all[oname].reshape(-1), want)
                        if r["status"] == "mismatch":
                            sworst = f"mismatch n={r['n_diff']}"
                            break
                    if sworst.startswith("mismatch"):
                        rec["note"] += (f"; composed-op check: {sworst} "
                                        "(oMLX fallback semantics differ from "
                                        "the assembled MSL; the MSL is the "
                                        "Metal-intended behaviour)")
                    else:
                        rec["note"] += f"; composed-op check: {sworst}"
                except Exception as e:  # a broken secondary must not mask the verdict
                    rec["note"] += f"; composed-op check: error ({e})"
        else:
            want_all, src = {}, "omarchy output"
            for o in kern["outputs"]:
                p = kdir / f"out_{o[0]}.npy"
                if p.exists():
                    want_all[o[0]] = np.load(p)
            if len(want_all) != len(kern["outputs"]):
                rec["status"] = "no-reference"
                rec["note"] += f"; mlx outputs missing ({mstat})"
                rows.append(rec)
                print(f"{name}: no-reference (omarchy {mstat})", flush=True)
                continue
        worst = None
        for oname, ocode, *_ in kern["outputs"]:
            cmp_r = pc.compare(ocode, got_all[oname].reshape(-1),
                               want_all[oname].reshape(-1))
            if cmp_r["status"] == "mismatch" or worst is None:
                worst = cmp_r
        rec.update(worst)
        rec["status"] = ("refused-by-translator" if mstat == "refused"
                         else worst["status"])
        rec["note"] += f"; vs {src}: {worst['status']}"
        rows.append(rec)
        print(f"{name}: {rec['status']} tol={rec['tol']} abs={rec['max_abs']} "
              f"rel={rec['max_rel']} first={rec['first_diff']} n={rec['n_diff']}",
              flush=True)
    pc.write_tsv(tsv, rows)
    counts = Counter(r["status"] for r in rows)
    print(f"{sum(counts.values())} kernels: {dict(counts)}")
    print(f"table: {tsv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--compare-only", action="store_true",
                    help="compare saved outputs against the references; "
                         "no runner, no GPU, no sweep json")
    ap.add_argument("--mlx-out", default="")
    ap.add_argument("--cmp-out", default="")
    a = ap.parse_args()
    fake = a.fake or os.environ.get("PARITY_FAKE") == "1"

    mlx_out = pc.env_path("PARITY_MLX_OUT", "./parity-mlx-out")
    out = pc.env_path("PARITY_OUT", "./parity-compare-out")
    tsv = pc.env_path("PARITY_TSV", str(out / "parity.tsv"))
    only = os.environ.get("PARITY_ONLY", "")
    if a.compare_only:
        if not (a.mlx_out and a.cmp_out):
            raise SystemExit("--compare-only needs --mlx-out and --cmp-out")
        tsv.parent.mkdir(parents=True, exist_ok=True)
        compare_only(pathlib.Path(a.mlx_out), pathlib.Path(a.cmp_out),
                     only, tsv)
        return

    sweep = json.load(open(pc.env_path("PARITY_SWEEP", "./t1.json")))
    spvdir = pc.env_path("PARITY_SPV_DIR", "./OUT")
    runner = pc.env_path("PARITY_RUNNER", "./m2v-run")
    out.mkdir(parents=True, exist_ok=True)
    mlx_status = {}
    if (mlx_out / "mlx_status.json").exists():
        mlx_status = json.load(open(mlx_out / "mlx_status.json"))
    byfile = {r["file"]: r for r in sweep if r.get("set") == "tf"}

    rows = []
    first_name = pc.KERNELS[0]["name"]
    for kern in pc.KERNELS:
        name = kern["name"]
        if only and only not in name:
            continue
        rec = dict(kernel=name, status="", note=kern.get("note") or "",
                   max_abs="", max_rel="", first_diff="", n_diff="", n_nan="",
                   tol="", mlx_est_s="", mlx_actual_s="", run_est_s="",
                   run_actual_s="")
        row = byfile.get(f"{kern['group']}/{name}.metal")
        if row is None:
            rec["status"], rec["note"] = "skipped", rec["note"] + "; not in the sweep"
            rows.append(rec)
            continue
        if row.get("val") != "ok":
            rec["status"] = "skipped"
            rec["note"] += f"; metal2vk: {row.get('error', '')[:110]}"
            rows.append(rec)
            print(f"{name}: skipped (metal2vk invalid)", flush=True)
            continue

        # inputs come from the mlx side's saved npy: byte-identical by construction
        kdir = mlx_out / name
        inputs = {}
        missing = [s[0] for s in kern["inputs"]
                   if not (kdir / f"{s[0]}.npy").exists()]
        if missing:
            rec["status"] = "skipped"
            rec["note"] += f"; mlx inputs missing: {missing[0]}"
            rows.append(rec)
            continue
        for spec in kern["inputs"]:
            inputs[spec[0]] = np.load(kdir / f"{spec[0]}.npy").tobytes()

        n_bytes = sum(len(b) for b in inputs.values()) + \
            sum(int(np.prod(o[2])) * pc.DTYPES[o[1]][0] for o in kern["outputs"])
        est = max(0.02, round(n_bytes / 2e10, 4))
        rec["run_est_s"] = est
        if est > MAX_SUBMIT_S:
            rec["status"] = "skipped"
            rec["note"] += f"; estimated submit {est} s above {MAX_SUBMIT_S} s"
            rows.append(rec)
            continue

        perturb = (kern["outputs"][0][0], 0x55) if fake and name == first_name else None
        status, err, dumps, wall = dispatch(
            kern, row, spvdir / "spv" / (row["tag"] + ".spv"), row["entry"],
            inputs, runner, out / name / "work", fake, perturb)
        rec["run_actual_s"] = round(wall, 4)
        if status != "ok":
            rec["status"] = "skipped" if fake else "mismatch"
            rec["note"] += f"; runner: {err}"
            rows.append(rec)
            print(f"{name}: {rec['status']} ({err[:60]})", flush=True)
            continue
        # save the dumps in the same form the mlx side uses, so the
        # comparison can be re-run later with --compare-only and no runner
        kout = out / name
        for oname, obytes in dumps.items():
            spec = next(o for o in kern["outputs"] if o[0] == oname)
            pc.save_arr(kout, f"out_{oname}", spec[1], obytes, spec[2])

        mstat = mlx_status.get(name, "missing")
        if mstat == "refused":
            if kern["name"] not in parity_refs.REFS:
                rec["status"] = "no-reference"
                rows.append(rec)
                print(f"{name}: no-reference (omarchy path refused)", flush=True)
                continue
            want_all = ref_outputs(kern, kdir)
        else:
            want_all = None
            for o in kern["outputs"]:
                p = kdir / f"out_{o[0]}.npy"
                if p.exists():
                    want_all = want_all or {}
                    want_all[o[0]] = np.load(p)
        if want_all is None:
            rec["status"] = "no-reference"
            rec["note"] += f"; mlx outputs missing ({mstat})"
            rows.append(rec)
            continue

        worst = None
        for oname, ocode, *_ in kern["outputs"]:
            got = np.frombuffer(dumps[oname], pc.DTYPES[ocode][1] or np.uint16)
            cmp = pc.compare(ocode, got, want_all[oname].reshape(-1))
            if cmp["status"] == "mismatch" or worst is None:
                worst = cmp
        rec.update(worst)
        if mstat == "refused":
            rec["status"] = "refused-by-translator"
            rec["note"] += f"; vs cpu reference: {worst['status']}"
        else:
            rec["status"] = worst["status"]
        rows.append(rec)
        print(f"{name}: {rec['status']} abs={rec['max_abs']} rel={rec['max_rel']} "
              f"first={rec['first_diff']} n={rec['n_diff']}", flush=True)

    pc.write_tsv(tsv, rows)
    counts = Counter(r["status"] for r in rows)
    print(f"{sum(counts.values())} kernels: {dict(counts)}")
    print(f"table: {tsv}")


if __name__ == "__main__":
    main()
