"""Replace SimplifyPointerBitcast's exit(3) non-convergence abort with a clean give-up.

The tf_gather_qmv modules do not reach a fixed point: sub-passes 5 (runOnImplicitCasts) and
7 (runOnGEPFromGEP) keep rewriting a GEP pair with progressive drift - the module changes
every iteration, so hash guards (apply-implicit-casts-hash-break.py,
apply-simplify-iteration-hash-break.py) never fire, the 200-iteration cap is reached, and
the compiler aborted with exit(3) on a module that is valid IR at that point. Compiling on
from the current IR is strictly better than aborting: the output may be valid SPIR-V, and
if it is not, the failure is diagnosable instead of a hard abort.

Tested against clspv f2b01dd6 plus the other metal2vk patch scripts.

Usage: python3 apply-simplify-no-exit.py /path/to/clspv   (idempotent)
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
marker = 'errs() << "M2V-GIVEUP: spb-200iter\\n";'
if marker in s:
    print("already patched")
    sys.exit(0)

old = (
    '      errs() << "M2V: SimplifyPointerBitcast does not converge; changing sub-passes:";\n'
    "      for (int i = 0; i < 9; i++)\n"
    "        if (c[i])\n"
    '          errs() << " " << i;\n'
    '      errs() << "\\n";\n'
    "      exit(3);\n"
)
new = (
    '      errs() << "M2V-GIVEUP: spb-200iter\\n";\n'
    '      errs() << "M2V: SimplifyPointerBitcast did not converge after 200 iterations;'
    ' continuing with the current IR\\n";\n'
    "      // metal2vk: the module at this point is valid IR; aborting (exit 3) turned\n"
    "      // a slow pass into a hard failure. Break out and compile on.\n"
    "      break;\n"
)
if old not in s:
    # v1 upgrade: a tree carrying the v1 text (cap break without the marker)
    # is upgraded in place.
    old1 = (
        '      errs() << "M2V: SimplifyPointerBitcast did not converge after 200 iterations;'
        ' continuing with the current IR\\n";\n'
    )
    if old1 not in s:
        raise SystemExit("anchor missing: exit(3) block in SimplifyPointerBitcastPass::run")
    s = s.replace(old1, '      errs() << "M2V-GIVEUP: spb-200iter\\n";\n' + old1, 1)
    pass_cpp.write_text(s)
    print(f"patched (clspv HEAD {sha[:8]})")
    sys.exit(0)
s = s.replace(old, new, 1)
pass_cpp.write_text(s)
print(f"patched (clspv HEAD {sha[:8]})")
