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
import runpy
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

# The ggml corpus staging path on the shim's metal::matrix: column subscript, device -> thread
# copy, matrix-to-matrix cast (narrow and bfloat widen), scalar multiply on both sides, and a
# threadgroup array of matrices. Before the matrix types landed (aee0a1d) this fails at parse
# with `unknown type name 'float4x4'`; it must stay a full-pipeline check (clang -> opt -> clspv
# -> spirv-val), not a source-text assertion.
MATRIX = """\
#include <metal_stdlib>
using namespace metal;

kernel void matrix_cols(device const char* src [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
  threadgroup float4x4 tile[2];
  float4x4 a;
  a[0][0] = 1.0f;
  a[1] = float4(2.0f);
  device const float4x4* dp = (device const float4x4*)(src);
  float4x4 b = *dp;
  half4x4 h = (half4x4)b;
  bfloat4x4 bf;
  bf[0][0] = bfloat(1.5f);
  float4x4 f = (float4x4)bf;
  float4x4 s = b * 2.0f;
  float4x4 s2 = 0.5f * b;
  tile[tid % 2u] = b;
  out[tid] = dot(s[3], tile[tid % 2u][0]) + (float)f[0][0] + (float)s[1][2] + (float)s2[2][1];
}
"""

# Independently written synthetic compile case: Metal's three-parameter spelling,
# and the existing two-parameter shim spelling retained for compatibility.
FILLED_MATRIX = """\
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void filled_matrix(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  simdgroup_float8x8 a = make_filled_simdgroup_matrix<float, 8, 8>(2.0f);
  simdgroup_float8x8 b = make_filled_simdgroup_matrix<float, 8>(3.0f);
  out[tid] = a.thread_elements()[0] + a.thread_elements()[1] + b.thread_elements()[0];
}
"""

SHARED_SHIM = """\
#include <metal_stdlib>
using namespace metal;
constexpr bfloat bf = bfloat(1.5f);
static_assert(float(bf) == 1.5f, "constexpr bfloat roundtrip");
kernel void shared_shim(device ulong* out [[buffer(0)]],
                        uint tid [[thread_position_in_grid]]) {
  uint2 words = uint2(tid, tid + 1u);
  out[tid * 3u] = as_type<ulong>(words);
  out[tid * 3u + 1u] = metal::select(1, 2, false);
  out[tid * 3u + 2u] = metal::select(1, 2, true);
}
"""

# fma() with three distinct operands: software expansion by default, one native instruction with --native-builtins fma.
FMA = """\
#include <metal_stdlib>
using namespace metal;
kernel void fma_case(device const float* in [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
  if (tid < 16u) out[tid] = fma(in[3u * tid], in[3u * tid + 1u], in[3u * tid + 2u]);
}
"""

fails = 0


def check(ok, msg):
    global fails
    if ok:
        print(f"ok {msg}")
    else:
        fails += 1
        print(f"FAIL {msg}")


def run(msl, out, cache, name=None, as_json=True, extra=(), env=None):
    cmd = [sys.executable, str(CLI), "--msl", str(msl), "--out", str(out), "--cache", str(cache), *extra]
    if name:
        cmd += ["--name", name]
    if as_json:
        cmd += ["--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    line = json.loads(r.stdout.splitlines()[-1]) if as_json and r.stdout.strip() else None
    return r.returncode, r.stdout, r.stderr, line


def fma_ops(spv):
    """GLSL.std.450 Fma instruction count and disassembly line count. The software recipe still emits one Fma on its
    normal-operand fast path, wrapped in a large special-case fallback; the native recipe is the lone instruction."""
    r = subprocess.run(["spirv-dis", str(spv)], capture_output=True, text=True)  # from PATH, as m2v-reflect.py
    check(r.returncode == 0, f"spirv-dis reads {spv.name} {r.stderr.strip()[:120]}".rstrip())
    return r.stdout.count(" Fma "), len(r.stdout.splitlines())


def check_native_builtins(root):
    """--native-builtins: separate cache entries, recorded choice, hits, rejection, and precedence over the environment."""
    # own source directory: the cache key hashes the MSL's directory and its parent, so sibling outputs must stay out of both
    src = root / "fma" / "src"
    src.mkdir(parents=True)
    (src / "fma.metal").write_text(FMA)
    cache = root / "fma-cache"
    base_env = {k: v for k, v in os.environ.items() if k != "M2V_NATIVE_BUILTINS"}
    fma_env = dict(base_env, M2V_NATIVE_BUILTINS="fma")
    results = {}
    for label, extra, env in (("default", (), base_env), ("none", ("--native-builtins", "none"), base_env),
                              ("fma", ("--native-builtins", "fma"), base_env),
                              ("none under env fma", ("--native-builtins", "none"), fma_env),
                              ("env fma", (), fma_env)):
        rc, so, se, line = run(src / "fma.metal", root / ("o-" + label.replace(" ", "-")), cache, extra=extra, env=env)
        check(rc == 0 and bool(line) and line["status"] == "ok", f"fma kernel compiles ({label})")
        results[label] = line
    if not all(results.values()):
        return  # a compile already printed FAIL; the key checks below need every result
    key = {k: v["key"] for k, v in results.items()}
    check(key["default"] != key["fma"], "fma and the default recipe get different cache keys")
    check(key["none"] == key["default"] and key["none under env fma"] == key["default"],
          "explicit none is the default recipe, even with M2V_NATIVE_BUILTINS=fma inherited")
    check(key["env fma"] == key["fma"], "M2V_NATIVE_BUILTINS=fma without the argument selects fma")
    check(results["default"]["cache"] == "miss" and results["fma"]["cache"] == "miss"
          and results["none"]["cache"] == "hit" and results["none under env fma"]["cache"] == "hit"
          and results["env fma"]["cache"] == "hit", "each recipe misses once, then hits")
    check(len(list((cache / "keys").iterdir())) == 2, "the cache holds exactly two entries")
    refl = {l: json.loads((root / ("o-" + l.replace(" ", "-")) / "fma_case.json").read_text())["toolchain"]
            for l in ("default", "fma", "none under env fma")}
    check([refl[l]["native_builtins"] for l in ("default", "fma", "none under env fma")] == ["none", "fma", "none"],
          "reflection records the native-builtins choice")
    sw, sw_lines = fma_ops(root / "o-default" / "fma_case.spv")
    nat, nat_lines = fma_ops(root / "o-fma" / "fma_case.spv")
    check(sw == 1 and nat == 1 and nat_lines * 2 < sw_lines,
          f"fma drops the software fallback ({sw_lines} lines by default, {nat_lines} with fma)")
    check((root / "o-none-under-env-fma" / "fma_case.spv").read_bytes() == (root / "o-default" / "fma_case.spv").read_bytes(),
          "explicit none under an inherited fma variable is byte-identical to the default")
    for bad_extra, bad_env, what, named in ((("--native-builtins", "all"), base_env, "unsupported argument", "--native-builtins"),
                                            (("--native-builtins", ""), base_env, "empty argument", "--native-builtins"),
                                            ((), dict(base_env, M2V_NATIVE_BUILTINS="all"), "unsupported variable",
                                             "M2V_NATIVE_BUILTINS")):
        rc, so, se, line = run(src / "fma.metal", root / "o-bad", cache, extra=bad_extra, env=bad_env)
        check(rc == 2 and f"error: {named} must be" in se, f"{what} exits 2 naming {named} ({rc}: {se.strip()[-80:]})")
        check(rc != 0 and not list((root / "o-bad").glob("*.spv")), f"{what} writes no module")
    rc, so, se, line = run(src / "fma.metal", root / "o-bad", cache, extra=("--native-builtins", "none"),
                           env=dict(base_env, M2V_NATIVE_BUILTINS="all"))
    check(rc == 0, "an explicit argument ignores an invalid inherited variable")


def main():
    corpus = runpy.run_path(str(REPO / "sweep" / "mlx-core" / "mlx_core_run.py"))
    diagnostic = ("/private/cache/kernels/attention.h:12:7: error: invalid vector element type 'bool'"
                  " | /private/cache/context.h:4:1: note: instantiated here")
    check(corpus["scrub_diag"](diagnostic)
          == "attention.h:12:7: invalid vector element type 'bool'",
          "corpus diagnostic keeps the error and removes private paths and continuation")
    check(corpus["classify"](4, "error", diagnostic, "") == "shim_gap",
          "invalid bool vector reports a shim gap")
    tparam_names = runpy.run_path(str(REPO / "sweep" / "m2v_sweep.py"))["tparam_names"]
    check(tparam_names("void (*dequantize_func)(device const void *, short, thread SA_4x4 &)")
          == ["dequantize_func"], "function-pointer template parameter keeps its declarator name")
    check(tparam_names("typename T, int N") == ["T", "N"], "ordinary template parameter names stay intact")
    with tempfile.TemporaryDirectory(prefix="m2v-cli-test-") as td:
        root = pathlib.Path(td)
        src = root / "msl"
        src.mkdir()
        for n, text in (("plain.metal", PLAIN), ("template.metal", TEMPLATE), ("refused.metal", REFUSED),
                        ("broken.metal", BROKEN), ("matrix.metal", MATRIX), ("filled_matrix.metal", FILLED_MATRIX),
                        ("shared_shim.metal", SHARED_SHIM)):
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

        # matrix staging path (the ggml corpus shape): column subscript, device -> thread copy,
        # matrix-to-matrix casts, scalar multiply on both sides, threadgroup array of matrices.
        # Fails at parse on the pre-matrix shim (aee0a1d): unknown type name 'float4x4'.
        rc, so, se, line = run(src / "matrix.metal", out, cache)
        check(rc == 0 and bool(line) and line.get("status") == "ok", "matrix staging kernel compiles to valid SPIR-V")
        check(bool(line) and line.get("name") == "matrix_cols", "matrix kernel reflection names matrix_cols")
        rc, so, se, line = run(src / "filled_matrix.metal", out, cache)
        check(rc == 0 and bool(line) and line.get("status") == "ok",
              f"filled simdgroup matrix template spellings compile ({se.strip()})")
        rc, so, se, line = run(src / "shared_shim.metal", out, cache)
        check(rc == 0 and bool(line) and line.get("status") == "ok",
              f"shared shim kernel compiles to valid SPIR-V ({se.strip()})")

        # Two loads share one dynamically indexed pointer-array address.
        # Previously promotion declined it and clspv failed on ptr addrspace(1).
        (src / "pointer_array_reuse.metal").write_text((REPO / "cases" / "pointer_array_reuse.metal").read_text())
        rc, so, se, line = run(src / "pointer_array_reuse.metal", out, cache)
        check(rc == 0 and bool(line) and line.get("status") == "ok",
              f"reused pointer-array address compiles to valid SPIR-V ({se.strip()})")

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

        check_native_builtins(root)
        # packed_uint3 (three tightly packed words, size 12, alignment 4): the case file's own static_assert guards
        # the layout, so a native 16-byte uint3 alias or a missing type fails here. Fails at parse on the base shim:
        # unknown type name 'packed_uint3'. The fixture also tests a hardware read at byte offset 4.
        packed = (REPO / "cases" / "packed_uint3_stride12.metal").read_text()
        (src / "packed.metal").write_text(packed)
        rc, so, se, line = run(src / "packed.metal", out, cache)
        check(rc == 0 and bool(line) and line.get("name") == "packed_uint3_stride12",
              "packed_uint3 stride-12 kernel compiles to valid SPIR-V" + (f" ({se.strip()[:200]})" if rc else ""))
        if rc == 0:
            args = json.loads((out / f"{line['name']}.json").read_text())["kernel"]["args"]
            check([(a["kind"], a["binding"], a["metal_type"]) for a in args]
                  == [("storage_buffer", 0, "device const uint*"), ("storage_buffer", 1, "device uint*")],
                  "packed_uint3 kernel reflects two storage buffers")
        (src / "packed_wrong.metal").write_text(packed.replace("sizeof(packed_uint3) == 12", "sizeof(packed_uint3) == 16"))
        rc, so, se, line = run(src / "packed_wrong.metal", root / "packed_wrong_out", cache)
        check(rc == 4 and "packed_uint3 layout" in se, f"a wrong packed_uint3 size assertion fails the compile ({rc})")

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
