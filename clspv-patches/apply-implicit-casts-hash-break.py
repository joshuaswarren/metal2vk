"""Break the runOnImplicitCasts shape-flip loop with a module-hash guard.

The 3 tf_gather_qmv entries exit with "SimplifyPointerBitcast does not converge; changing
sub-passes: 5": runOnImplicitCasts flips a GEP pair between two valid-looking shapes
without net progress, so its `changed` stays true forever. This patch hashes the module
after each runOnImplicitCasts invocation and clears its changed flag when the hash repeats -
the loop then converges and the (valid) IR compiles on.

Tested against clspv f2b01dd6 plus apply-coopmat-lowering.py, apply-undef-pointer-type.py,
apply-simplify-ptr-bitcast-converge.py, apply-untyped-gep-predicate.py and
apply-implicit-casts-converge.py.

Usage: python3 apply-implicit-casts-hash-break.py /path/to/clspv   (idempotent)
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
if "M2vCastHash" in s:
    print("already patched")
    sys.exit(0)

# 1. includes
a = '#include "llvm/IR/Module.h"\n'
if a not in s:
    raise SystemExit("anchor missing: Module.h include")
s = s.replace(a, a + "#include <functional>\n", 1)

# 2. hash state, reset each convergence loop (local to run(), per loop is correct: the
#    guard only needs to see flips within one pass invocation)
a = (
    "  bool changed = true;\n"
    "  unsigned M2vIter = 0;\n"
    "  while (changed) {\n"
    "    changed = false;\n"
    "    bool c[9];\n"
)
if a not in s:
    raise SystemExit("anchor missing: convergence loop head")
s = s.replace(
    a,
    "  bool changed = true;\n"
    "  unsigned M2vIter = 0;\n"
    "  // metal2vk: hash of the module after the last runOnImplicitCasts run; a repeat\n"
    "  // hash means the sub-pass is flipping shapes without net progress.\n"
    "  size_t M2vCastHash = 0;\n"
    "  bool M2vCastHashValid = false;\n"
    "  while (changed) {\n"
    "    changed = false;\n"
    "    bool c[9];\n",
    1,
)

# 3. the guard
a = "    c[5] = runOnImplicitCasts(M);\n"
if a not in s:
    raise SystemExit("anchor missing: c[5] assignment")
s = s.replace(
    a,
    "    c[5] = runOnImplicitCasts(M);\n"
    "    // metal2vk: break the shape-flip loop: if the module is unchanged after two\n"
    "    // consecutive runs, the sub-pass is oscillating and must not report progress.\n"
    "    if (c[5]) {\n"
    "      std::string M2vBuf;\n"
    "      llvm::raw_string_ostream M2vStream(M2vBuf);\n"
    "      M.print(M2vStream, nullptr);\n"
    "      size_t M2vH = std::hash<std::string>{}(M2vStream.str());\n"
    "      if (M2vCastHashValid && M2vH == M2vCastHash) {\n"
    "        c[5] = false;\n"
    "      }\n"
    "      M2vCastHash = M2vH;\n"
    "      M2vCastHashValid = true;\n"
    "    }\n",
    1,
)

pass_cpp.write_text(s)
print(f"patched (clspv HEAD {sha[:8]})")
