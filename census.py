"""Front-end census: run clang (C++ for OpenCL, through the metal_stdlib shim) over every uzu .metal file and bucket
the first errors. Parse level only: templates are not instantiated, so this under-counts semantic problems.
Usage: census.py [ROOT] [KERNEL_DIR] > census.txt   (KERNEL_DIR defaults to uzu's kernel dir under ROOT; for another project pass its directory of .metal files)"""
import collections
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()  # a tree containing .metal files
K = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "crates/uzu-engine/src/backends/metal/kernel"
H = pathlib.Path(__file__).parent.resolve()
files = sorted(K.rglob("*.metal"))
buckets = collections.Counter()
ok = 0
rows = []
for f in files:
    cmd = ["clang", "--target=spir64", "-x", "cl", "-cl-std=clc++2021", "-Xclang", "-finclude-default-header", "-cl-ext=-__opencl_c_generic_address_space", "-fsyntax-only", "-w",
           f"-I{H}/include", f"-I{K}", f"-I{K}/generated", "-ferror-limit=200", str(f)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    errs = [ln for ln in r.stderr.splitlines() if " error: " in ln]
    rel = str(f.relative_to(K))
    if r.returncode == 0:
        ok += 1
        rows.append((rel, "OK", ""))
        continue
    msg = errs[0].split(" error: ", 1)[1] if errs else r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?"
    key = re.sub(r"'[^']*'", "'X'", msg)[:90]
    for e in errs:
        buckets[re.sub(r"'[^']*'", "'X'", e.split(" error: ", 1)[1])[:90]] += 1
    rows.append((rel, f"FAIL({len(errs)})", msg[:110]))
print(f"files={len(files)} parse_ok={ok}")
for r in rows:
    print(f"{r[0]:62s} {r[1]:9s} {r[2]}")
print("\nerror buckets (all errors, normalized):")
for k, v in buckets.most_common(25):
    print(f"{v:5d}  {k}")
