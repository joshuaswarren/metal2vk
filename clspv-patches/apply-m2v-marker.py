"""Add an `--m2v-spec-constants` cl option as a no-op marker.

metal2vk's Linux toolchain passes this flag to confirm the clspv binary
knows the m2v spec-constant protocol (the patch in this directory). An
unpatched clspv rejects the flag and exits nonzero, which the toolchain
interprets as a safe empty-library fallback instead of a silently wrong
build.
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
comp = root / "lib/Compiler.cpp"
s = comp.read_text()
if "m2v_spec_constants" in s:
    print("clspv marker already present")
else:
    anchor = 'static llvm::cl::opt<bool> cl_single_precision_constants(\n'
    assert anchor in s, "clspv option anchor not found"
    s = s.replace(
        anchor,
        '// metal2vk: no-op marker flag the toolchain passes to confirm this\n'
        '// clspv binary understands the m2v spec-constant protocol.\n'
        'static llvm::cl::opt<bool> m2v_spec_constants(\n'
        '    "m2v-spec-constants", llvm::cl::init(false),\n'
        '    llvm::cl::desc("No-op marker for the metal2vk kernel pipeline."));\n\n'
        + anchor,
        1,
    )
    comp.write_text(s)
    print("clspv marker added")
