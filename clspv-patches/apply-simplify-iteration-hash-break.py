"""Break the SimplifyPointerBitcast convergence loop at iteration granularity.

The first hash break (apply-implicit-casts-hash-break.py) guarded runOnImplicitCasts
alone; the tf_gather_qmv modules kept cycling because sub-passes 5 (runOnImplicitCasts)
and 7 (runOnGEPFromGEP) flip a GEP pair between two valid shapes TOGETHER - each pass
changes the module, so per-sub-pass hashes never repeat. This patch hashes the module at
iteration boundaries: if a full pass over all sub-passes leaves the module byte-identical
to the previous iteration, the loop reports no progress and converges; the module at that
point is valid IR and compiles on.

Tested against clspv f2b01dd6 plus the other metal2vk patch scripts.

Usage: python3 apply-simplify-iteration-hash-break.py /path/to/clspv   (idempotent)
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
pass_cpp = root / "lib/SimplifyPointerBitcastPass.cpp"

sha = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if not sha.startswith("f2b01dd"):
    print(f"NOTE: clspv HEAD is {sha[:8]} (tested f2b01dd6); anchors may need adjustment",
          file=sys.stderr)

s = pass_cpp.read_text()
if "M2V-GIVEUP: spb-iter-hash" in s:
    print("already patched")
    sys.exit(0)

# v1 upgrade: a tree carrying the v1 guard (hash state without the marker) is
# upgraded in place - only the marker line is inserted.
v1_guard = (
    "      if (M2vIterHashValid && M2vIterHash == M2vIterHashPrev) {\n"
    "        changed = false;\n"
    "      }\n"
)
if "M2vIterHash" in s:
    if v1_guard not in s:
        raise SystemExit("anchor missing: v1 iteration-hash guard")
    s = s.replace(
        v1_guard,
        "      if (M2vIterHashValid && M2vIterHash == M2vIterHashPrev) {\n"
        '        errs() << "M2V-GIVEUP: spb-iter-hash\\n";\n'
        "        changed = false;\n"
        "      }\n",
        1,
    )
    pass_cpp.write_text(s)
    print(f"patched (clspv HEAD {sha[:8]})")
    sys.exit(0)

a = (
    "    for (bool x : c)\n"
    "      changed |= x;\n"
    "    if (++M2vIter > 200) {\n"
)
if a not in s:
    raise SystemExit("anchor missing: changed accumulation before the iteration cap")
s = s.replace(
    a,
    "    for (bool x : c)\n"
    "      changed |= x;\n"
    "    // metal2vk: iteration-level progress guard - sub-passes can flip a GEP pair\n"
    "    // between two valid shapes together, so per-sub-pass hashes never repeat. If\n"
    "    // a full pass leaves the module byte-identical, report no progress and let\n"
    "    // the (valid) IR compile on.\n"
    "    if (changed) {\n"
    "      std::string M2vIterBuf;\n"
    "      llvm::raw_string_ostream M2vIterStream(M2vIterBuf);\n"
    "      M.print(M2vIterStream, nullptr);\n"
    "      size_t M2vIterHash = std::hash<std::string>{}(M2vIterStream.str());\n"
    "      if (M2vIterHashValid && M2vIterHash == M2vIterHashPrev) {\n"
    '        errs() << "M2V-GIVEUP: spb-iter-hash\\n";\n'
    "        changed = false;\n"
    "      }\n"
    "      M2vIterHashPrev = M2vIterHash;\n"
    "      M2vIterHashValid = true;\n"
    "    }\n"
    "    if (++M2vIter > 200) {\n",
    1,
)

a = (
    "  size_t M2vCastHash = 0;\n"
    "  bool M2vCastHashValid = false;\n"
)
if a not in s:
    raise SystemExit("anchor missing: cast hash locals")
s = s.replace(
    a,
    a
    + "  size_t M2vIterHashPrev = 0;\n"
    "  bool M2vIterHashValid = false;\n",
    1,
)

pass_cpp.write_text(s)
print(f"patched (clspv HEAD {sha[:8]})")
