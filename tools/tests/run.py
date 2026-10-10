#!/usr/bin/env python3
"""Self-checks for the m2v-compile CLI front door; run from the repo root: python3 tools/tests/run.py

Every check drives tools/m2v-compile as a subprocess, exactly as omarchy-mlx's C++ will: MSL in, <name>.spv plus
<name>.json out, exit codes 0 ok / 2 usage / 3 refused / 4 compile error / 5 spirv-val failure, one JSON line with
--json, content-addressed cache under a temporary directory.

The mlx-style kernels (a plain [[kernel]], a template kernel with two host_name instantiations, a Metal 4
tensor< refusal, a syntax error) are generated into a temporary directory. Checks that produce SPIR-V also compile
the same file through sweep/m2v_sweep.py --dir and require byte-identical modules.

Set M2V_SIBLING_ROOT to a TensorFold-layout checkout root (its zig/kernels/metal holds the sibling kernels) to also
run the 32-kernel sibling parity block: 30 entries must exit 0, the dextents entry 3, the invalid wide_partial
entry 5, and every module must be byte-identical to the sweep's.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
CLI = REPO / "tools" / "m2v-compile"

PLAIN = """\
#include <metal_stdlib>
using namespace metal;

[[kernel]] void add_one(device const float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        float scale,
                        uint2 tid [[thread_position_in_grid]]) {
  if (tid.x < 16u && tid.y == 0u) out[tid.x] = in[tid.x] * scale + 1.0f;
}
"""

TEMPLATE = """\
#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void scaled(device const T* in [[buffer(0)]],
                       device T* out [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
  if (tid < 16u) out[tid] = in[tid] * T(2);
}
template [[host_name("scaled_f")]] [[kernel]] decltype(scaled<float>) scaled<float>;
template [[host_name("scaled_h")]] [[kernel]] decltype(scaled<half>) scaled<half>;
"""

REFUSED = """\
#include <metal_stdlib>
using namespace metal;

template <int N>
[[kernel]] void tensorish(device const tensor<int, N>* in [[buffer(0)]],
                          device int* out [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
  if (tid < 16u) out[tid] = int(in[tid]);
}
template [[host_name("tensorish")]] [[kernel]] decltype(tensorish<2>) tensorish<2>;
"""

BROKEN = "kernel void broken( uninitialized ) {\n"

fails = 0


def check(ok, msg):
    global fails
    if ok:
        print(f"ok {msg}")
    else:
        fails += 1
        print(f"FAIL {msg}")


def run(msl, out, cache, name=None, as_json=True):
    cmd = [sys.executable, str(CLI), "--msl", str(msl), "--out", str(out), "--cache", str(cache)]
    if name:
        cmd += ["--name", name]
    if as_json:
        cmd += ["--json"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    line = json.loads(r.stdout.splitlines()[-1]) if as_json and r.stdout.strip() else None
    return r.returncode, r.stdout, r.stderr, line


def main():
    with tempfile.TemporaryDirectory(prefix="m2v-cli-test-") as td:
        root = pathlib.Path(td)
        src = root / "msl"
        src.mkdir()
        for n, text in (("plain.metal", PLAIN), ("template.metal", TEMPLATE), ("refused.metal", REFUSED),
                        ("broken.metal", BROKEN)):
            (src / n).write_text(text)
        cache = root / "cache"
        out = root / "out"

        # plain kernel: ok, two output files, JSON shape, reflection fields
        rc, so, se, line = run(src / "plain.metal", out, cache)
        check(rc == 0, "plain kernel exits 0")
        check(bool(line) and line["status"] == "ok" and line["cache"] == "miss", "plain kernel reports ok/miss")
        files = sorted(p.name for p in out.iterdir())
        check(files == ["add_one.json", "add_one.spv"], f"out holds exactly spv and json ({files})")
        doc = json.loads((out / "add_one.json").read_text())
        check(doc["name"] == "add_one", "reflection names the kernel")
        kinds = [a["kind"] for a in doc["kernel"]["args"]]
        check(kinds[0] == "storage_buffer" and kinds[1] == "storage_buffer" and kinds[2] == "pod_push_constant",
              f"argument kinds in kernel-argument order ({kinds})")
        check(doc["kernel"]["args"][0]["binding"] == 0 and doc["kernel"]["args"][1]["binding"] == 1
              and doc["kernel"]["args"][2]["offset"] >= 0, "buffer bindings and push-constant offsets present")
        check(isinstance(doc["uses_cooperative_matrix"], bool), "uses_cooperative_matrix flag present")
        wg = doc["kernel"]["workgroup_size"]
        spec = doc["workgroup_size_spec_constant_ids"]
        check((isinstance(wg, list) and len(wg) == 3) or spec, f"workgroup size or its spec ids present ({wg}, {spec})")
        check(all(k in doc["toolchain"] for k in ("clang", "opt", "clspv", "include_sha256", "clspv_patches_sha256")),
              "toolchain fingerprint present")
        for a in doc["kernel"]["args"]:
            if a["kind"] == "storage_buffer":
                check(a.get("metal_name") in ("in", "out"), "Metal-side argument names recorded")
                break

        # cache: same inputs hit, byte-identical copies, changed source misses
        rc, so, se, line = run(src / "plain.metal", out, cache)
        check(rc == 0 and line["cache"] == "hit", "second run is a cache hit")
        spv1 = (out / "add_one.spv").read_bytes()
        rc, so, se, _ = run(src / "plain.metal", root / "out2", cache)
        check((root / "out2" / "add_one.spv").read_bytes() == spv1, "hit copies byte-identical artifacts")
        (src / "plain.metal").write_text(PLAIN.replace("* scale + 1.0f", "* scale + 2.0f"))
        rc, so, se, line = run(src / "plain.metal", out, cache)
        check(rc == 0 and line["cache"] == "miss", "changed MSL text is a new key")

        # name selection: template kernel with two instantiations
        rc, so, se, line = run(src / "template.metal", out, cache)
        check(rc == 2, "ambiguous template without --name exits 2")
        rc, so, se, line = run(src / "template.metal", out, cache, name="scaled_f")
        check(rc == 0 and line["name"] == "scaled_f", "--name picks the instantiation by host name")
        rc, so, se, line = run(src / "template.metal", out, cache, name="scaled")
        check(rc == 2, "--name by entry function name with two instantiations is ambiguous, exits 2")
        rc, so, se, line = run(src / "template.metal", out, cache, name="nope")
        check(rc == 2, "unknown --name exits 2")
        rc, so, se, line = run(src / "template.metal", out, cache, name="scaled_h")
        check(rc == 0 and line["name"] == "scaled_h", "--name picks the second instantiation")
        check((out / "scaled_f.spv").read_bytes() != (out / "scaled_h.spv").read_bytes(),
              "the two instantiations differ")

        # refusal and failures leave no artifacts in the output directory
        rref = root / "refout"
        rc, so, se, line = run(src / "refused.metal", rref, cache)
        check(rc == 3 and line["status"] == "refused", "tensor< kernel exits 3 refused")
        check(line.get("refused") == "tensor<" and not list(rref.iterdir()), "refusal names the construct, out empty")
        rbad = root / "badout"
        rc, so, se, line = run(src / "broken.metal", rbad, cache)
        check(rc == 4 and not list(rbad.iterdir()), "syntax error exits 4 with no output files")
        check("error" in se and (cache / "logs").is_dir() and list((cache / "logs").glob("*.log")),
              "compile error prints a diagnostic and keeps a log under the cache")

        # missing file and the --json error shape
        rc, so, se, line = run(src / "absent.metal", out, cache)
        check(rc == 2 and line["status"] == "error" and line["exit"] == 2, "missing --msl exits 2 with an error line")

        # byte-identity with the sweep for the same entry (clspv present, as in CI)
        (src / "plain.metal").write_text(PLAIN)
        clspv = os.environ.get("CLSPV", str(pathlib.Path.home() / "scratch/metal2vk/clspv/build/bin/clspv"))
        if pathlib.Path(clspv).exists():
            sout = root / "sweepout"
            env = dict(os.environ, TF_ROOT="", UZU_ROOT="")
            r = subprocess.run([sys.executable, str(REPO / "sweep" / "m2v_sweep.py"), str(sout), "--dir", str(src),
                                "--only", "plain.metal"], capture_output=True, text=True, env=env)
            rows = json.loads((sout / "sweep.json").read_text())
            row = next(r2 for r2 in rows if r2["entry"] == "add_one")
            check(row["val"] == "ok", "sweep compiles the same plain kernel")
            if row["val"] == "ok":
                check((sout / "spv" / (row["tag"] + ".spv")).read_bytes() == spv1,
                      "CLI SPIR-V is byte-identical to the sweep's for the same entry")
        else:
            print("skip byte-identity (no clspv)")

        # two concurrent invocations of one cold key: both succeed, one cache entry
        (src / "plain.metal").write_text(PLAIN)
        ccache = root / "ccache"
        procs = [subprocess.Popen([sys.executable, str(CLI), "--msl", str(src / "plain.metal"),
                                   "--out", str(root / f"c{i}"), "--cache", str(ccache), "--json"],
                                  stdout=subprocess.PIPE, text=True) for i in range(2)]
        rcs = [p.wait() for p in procs]
        keys = list((ccache / "keys").iterdir()) if (ccache / "keys").is_dir() else []
        check(rcs == [0, 0], f"concurrent invocations both exit 0 ({rcs})")
        check(len(keys) == 1, f"concurrent invocations leave one cache entry ({len(keys)})")
        check((root / "c0" / "add_one.spv").read_bytes() == (root / "c1" / "add_one.spv").read_bytes(),
              "concurrent outputs are identical")

    sibling = os.environ.get("M2V_SIBLING_ROOT", "")
    if sibling:
        run_sibling(pathlib.Path(sibling))
    else:
        print("skip sibling parity (M2V_SIBLING_ROOT not set)")
    sys.exit(1 if fails else 0)


def run_sibling(root):
    """The 32 sibling kernels: 30 ok, the dextents one refused (3), wide_partial invalid (5); every module
    byte-identical to what the sweep produces for the same entry."""
    metal = root / "zig" / "kernels" / "metal"
    files = sorted(metal.glob("*/*.metal"))
    check(len(files) == 32, f"sibling tree holds 32 kernels ({len(files)})")
    with tempfile.TemporaryDirectory(prefix="m2v-cli-sib-") as td:
        td = pathlib.Path(td)
        cache = td / "cache"
        env = dict(os.environ, TF_ROOT=str(root))
        cold, warm = {}, {}
        statuses = {}
        for round_no, times in (("cold", cold), ("warm", warm)):
            for f in files:
                out = td / round_no
                t0 = time.time()
                r = subprocess.run([sys.executable, str(CLI), "--msl", str(f), "--out", str(out),
                                    "--cache", str(cache), "--json"], capture_output=True, text=True, env=env)
                line = json.loads(r.stdout.splitlines()[-1])
                name = line.get("name") or line.get("kernel") or f.stem
                if line["status"] == "ok":
                    times[name] = time.time() - t0
                statuses[str(f.relative_to(metal))] = (r.returncode, line, name, line.get("spv"))
        ok = [f for f, (rc, line, _, _) in statuses.items() if rc == 0]
        refused = [f for f, (rc, line, _, _) in statuses.items() if rc == 3]
        invalid = [f for f, (rc, line, _, _) in statuses.items() if rc == 5]
        check(len(ok) == 30, f"30 sibling kernels compile ({len(ok)})")
        check(refused == ["sibling/omlx_verify_attn_gqa_partial.metal"], f"the dextents kernel is refused ({refused})")
        check(invalid == ["sibling/omlx_verify_attn_wide_partial.metal"], f"wide_partial exits 5 ({invalid})")
        check(all(line["cache"] == "hit" for rc, line, _, _ in statuses.values() if rc == 0),
              "warm round is all cache hits")
        # byte-identity: sweep the same tree and compare every ok module
        sout = td / "sweep"
        r = subprocess.run([sys.executable, str(REPO / "sweep" / "m2v_sweep.py"), str(sout), "--set", "tf", "--jobs", "8"],
                           capture_output=True, text=True, env=env)
        rows = {r2["entry"]: r2 for r2 in json.loads((sout / "sweep.json").read_text())}
        same, diff = 0, []
        for f, (rc, line, name, spv) in statuses.items():
            if rc != 0:
                continue
            row = rows.get(name)
            if row and row.get("tag") and (sout / "spv" / (row["tag"] + ".spv")).read_bytes() == pathlib.Path(spv).read_bytes():
                same += 1
            else:
                diff.append(f)
        check(same == 30 and not diff, f"every CLI module is byte-identical to the sweep's ({same}, {diff})")
        for label, times in (("cold", cold), ("warm", warm)):
            ts = sorted(times.values())
            med = ts[len(ts) // 2]
            print(f"sibling {label}: n={len(ts)} median {med:.3f} s min {ts[0]:.3f} s max {ts[-1]:.3f} s")


if __name__ == "__main__":
    main()
