#!/usr/bin/env python3
"""Self-checks for the sweep's IR text passes; run from the repo root: python3 sweep/tests/run.py

Each case is an LLVM IR file under sweep/tests/ plus the pass (or passes) that must handle it. A case fails on
an unexpected pass exit, on a missing expected rewrite, on leftover text that must have been rewritten, or when
the output does not verify. opt (when on PATH) verifies every output; clspv (env CLSPV, or the built binary
next to a clspv checkout) plus spirv-val are used when present, which is how CI runs it.

The bfloat case fails if the pass ever regresses on -inf / +inf / nan operands, on the bare integer constant in
a float phi, or leaves a bfloat token behind. The dynamic array case fails if the flatten pass stops rewriting
the single-member wrapper that clspv cannot lower.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SWEEP = HERE.parent


def run_pass(script, ll):
    r = subprocess.run([sys.executable, str(SWEEP / script), str(ll)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL {script} exited {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")
        return False
    return True


def verify(ll, name):
    """Structural asserts per case, then opt / clspv / spirv-val when available."""
    txt = ll.read_text()
    if name == "bfloat_consts":
        if any("bfloat" in ln.split(";")[0] for ln in txt.splitlines()):
            print("FAIL bfloat_consts: a bfloat token survived the rewrite")
            return False
        for ln in txt.splitlines():
            if "fcmp" in ln and re.search(r"[-+]?inf\b|\bnan\b", ln):
                print(f"FAIL bfloat_consts: literal inf/nan survived in: {ln.strip()}")
                return False
        for ln in txt.splitlines():
            if "phi float" in ln and "[ 0, " in ln:
                print(f"FAIL bfloat_consts: bare integer float phi constant survived: {ln.strip()}")
                return False
        if "0xFFF0000000000000" not in txt or "0x7FF0000000000000" not in txt:
            print("FAIL bfloat_consts: the inf operands were not rewritten to the exact f64 hex form")
            return False
    if name == "dynamic_struct_array":
        if '%"' + 'struct.metal::simdgroup_matrix" = type {' in txt:
            print("FAIL dynamic_struct_array: the wrapper type was not flattened")
            return False
    opts = [shutil.which("opt"), shutil.which("opt-23")]
    opt = next((o for o in opts if o), None)
    if opt:
        r = subprocess.run([opt, "-passes=verify", str(ll), "-o", str(ll) + ".bc"], capture_output=True, text=True)
        (ll.parent / (ll.name + ".bc")).unlink(missing_ok=True)
        if r.returncode != 0:
            print(f"FAIL {name}: opt verify rejected the pass output: {r.stderr.strip()[:300]}")
            return False
    clspv = os.environ.get("CLSPV", "")
    if clspv and pathlib.Path(clspv).exists():
        spv = ll.with_suffix(".spv")
        r = subprocess.run([clspv, "-x", "ir", "--cl-std=CLC++2021", "--fp16", "--inline-entry-points",
                            "--spv-version=1.5", str(ll), "-o", str(spv)], capture_output=True, text=True)
        if r.returncode != 0 or not spv.exists() or spv.stat().st_size == 0:
            print(f"FAIL {name}: clspv failed: {(r.stderr or r.stdout).strip()[:300]}")
            return False
        val = shutil.which("spirv-val")
        if val:
            r = subprocess.run([val, "--target-env", "vulkan1.3", str(spv)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL {name}: spirv-val rejected clspv output: {r.stdout.strip()[:300]}")
                return False
        spv.unlink(missing_ok=True)
    print(f"ok {name}")
    return True


import re  # noqa: E402  (used by the bfloat assertions)

CASES = {
    "bfloat_consts.ll": ["bfloat_to_i16.py", "flatten_single_member_structs.py"],
    "dynamic_struct_array.ll": ["flatten_single_member_structs.py"],
}

fails = 0
with tempfile.TemporaryDirectory(prefix="m2v-selftest-") as td:
    for name, passes in CASES.items():
        work = pathlib.Path(td) / name
        work.write_text((HERE / name).read_text())
        if not all(run_pass(s, work) for s in passes):
            fails += 1
            continue
        if not verify(work, name.removesuffix(".ll")):
            fails += 1
sys.exit(1 if fails else 0)
