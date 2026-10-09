"""Make runOnImplicitCasts skip un-representable folds (convergence, part 2).

The 3 tf_gather_qmv entries exited with "SimplifyPointerBitcast does not converge;
changing sub-passes: 5" - sub-pass 5 is runOnImplicitCasts. Like runOnGEPFromGEP (fixed by
apply-simplify-ptr-bitcast-converge.py), it rewrote every worklist entry unconditionally and
reported `changed = true` even when GetIdxsForTyFromOffset could not represent the offset:
the partial result left the same implicit-cast pattern in place and the pass re-processed it
every iteration. With the Ok out-param added by that earlier patch, this pass now asks for
representability and skips the entry when it is not - the GEP stays as-is, which is valid IR,
and the pass stops reporting progress on it.

Tested against clspv f2b01dd6 plus apply-coopmat-lowering.py, apply-undef-pointer-type.py and
apply-simplify-ptr-bitcast-converge.py (which supplies the bool* parameter).

Usage: python3 apply-implicit-casts-converge.py /path/to/clspv   (idempotent)
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
if "M2vFoldOk" in s.split("bool clspv::SimplifyPointerBitcastPass::runOnImplicitCasts")[-1]:
    print("already patched")
    sys.exit(0)

# 1. declare the flag inside the fold loop (after the builder, before the branches)
old = (
    "  for (auto inst_gep : Worklist) {\n"
    "    GetElementPtrInst *src_gep =\n"
    "        cast<GetElementPtrInst>(inst_gep->getPointerOperand());\n"
    "\n"
    "    IRBuilder<> Builder{inst_gep};\n"
    "    SmallVector<Value *> Idxs;\n"
)
new = (
    "  for (auto inst_gep : Worklist) {\n"
    "    GetElementPtrInst *src_gep =\n"
    "        cast<GetElementPtrInst>(inst_gep->getPointerOperand());\n"
    "\n"
    "    IRBuilder<> Builder{inst_gep};\n"
    "    SmallVector<Value *> Idxs;\n"
    "    // metal2vk: the fold is skipped when the offset cannot be represented\n"
    "    // through the source type path (GetIdxsForTyFromOffset clears this flag).\n"
    "    bool M2vFoldOk = true;\n"
)
if old not in s:
    raise SystemExit("anchor missing: runOnImplicitCasts fold loop head")
s = s.replace(old, new, 1)

# 2. ask for representability
old = (
    "      Idxs = GetIdxsForTyFromOffset(\n"
    "          DL, Builder, src_ty, inst_gep->getResultElementType(), CstVal, DynVal,\n"
    "          SmallerBitWidths, src_gep->getPointerOperand());\n"
)
new = (
    "      Idxs = GetIdxsForTyFromOffset(\n"
    "          DL, Builder, src_ty, inst_gep->getResultElementType(), CstVal, DynVal,\n"
    "          SmallerBitWidths, src_gep->getPointerOperand(), &M2vFoldOk);\n"
)
if old not in s:
    raise SystemExit("anchor missing: GetIdxsForTyFromOffset call in runOnImplicitCasts")
s = s.replace(old, new, 1)

# 3. skip before the replacement GEP is built
old = (
    "    auto new_gep = GetElementPtrInst::Create(src_ty, src, Idxs, \"\",\n"
    "                                             inst_gep->getIterator());\n"
)
new = (
    "    if (!M2vFoldOk) {\n"
    "      // metal2vk: leave the entry alone; it is valid IR and the pass must not\n"
    "      // report progress on a fold it did not perform.\n"
    "      continue;\n"
    "    }\n"
    "    auto new_gep = GetElementPtrInst::Create(src_ty, src, Idxs, \"\",\n"
    "                                             inst_gep->getIterator());\n"
)
if old not in s:
    raise SystemExit("anchor missing: new_gep creation in runOnImplicitCasts")
s = s.replace(old, new, 1)

pass_cpp.write_text(s)
print(f"patched (clspv HEAD {sha[:8]})")
