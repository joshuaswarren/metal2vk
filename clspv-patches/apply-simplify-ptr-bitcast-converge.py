"""Make SimplifyPointerBitcastPass converge on un-representable GEP folds.

The qwen3_5_qmm and tf_gather_qmv hangs: runOnGEPFromGEP reports `Changed` from the
worklist size (`Changed = !WorkList.empty()`) BEFORE folding, and folds every pair even
when BitcastUtils::GetIdxsForTyFromOffset cannot represent the offset (the
`Err: SrcTy = ... - CstVal = N` path; llvm_unreachable is compiled out in Release = UB).
The same pairs are re-collected every pass, so the outer while(changed) churns until the
200-iteration exit(3) (the "does not converge" entries) or the sweep timeout (the
qwen3_5_qmm_* entries; one core dump showed 63,536 Err prints inside one compile).

Fix:
- GetIdxsForTyFromOffset gains a trailing `bool *Ok = nullptr`. On the un-representable
  path it sets `*Ok = false` (when asked) and returns the partial indices instead of
  running into UB. Existing callers pass nothing and keep working.
- runOnGEPFromGEP passes `&M2vFoldOk` to its three fold paths, skips the pair when the
  offset is un-representable (the two GEPs stay split - valid, correct IR), and counts
  `Changed` only for GEPs it actually erased.

Tested against clspv f2b01dd6 plus apply-coopmat-lowering.py and
apply-undef-pointer-type.py.

Usage: python3 apply-simplify-ptr-bitcast-converge.py /path/to/clspv   (idempotent)
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
utils_h = root / "lib/BitcastUtils.h"
utils_cpp = root / "lib/BitcastUtils.cpp"
pass_cpp = root / "lib/SimplifyPointerBitcastPass.cpp"

sha = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if not sha.startswith("f2b01dd"):
    print(f"NOTE: clspv HEAD is {sha[:8]} (tested f2b01dd6); anchors may need adjustment",
          file=sys.stderr)

producer = (root / "lib/SPIRVProducerPass.cpp").read_text()
if "M2vFoldOk" in pass_cpp.read_text() and "requires the capability" in producer:
    print("already patched")
    sys.exit(0)


def step(path, old, new, label):
    s = path.read_text()
    if new in s:
        return  # already applied by a previous (partial) run
    if old not in s:
        raise SystemExit(f"anchor missing: {label}")
    path.write_text(s.replace(old, new, 1))


# 1. header signature: trailing Ok out-param (defaulted, so every existing caller builds)
step(
    utils_h,
    "GetIdxsForTyFromOffset(const DataLayout &DataLayout, IRBuilder<> &Builder,\n"
    "                       Type *SrcTy, Type *DstTy, int64_t CstVal, Value *DynVal,\n"
    "                       size_t SmallerBitWidths, Value *Src);",
    "GetIdxsForTyFromOffset(const DataLayout &DataLayout, IRBuilder<> &Builder,\n"
    "                       Type *SrcTy, Type *DstTy, int64_t CstVal, Value *DynVal,\n"
    "                       size_t SmallerBitWidths, Value *Src,\n"
    "                       bool *Ok = nullptr);",
    "BitcastUtils.h declaration",
)

# 2. definition signature
step(
    utils_cpp,
    "GetIdxsForTyFromOffset(const DataLayout &DataLayout, IRBuilder<> &Builder,\n"
    "                       Type *SrcTy, Type *DstTy, int64_t CstVal, Value *DynVal,\n"
    "                       size_t SmallerBitWidths, Value *Src) {",
    "GetIdxsForTyFromOffset(const DataLayout &DataLayout, IRBuilder<> &Builder,\n"
    "                       Type *SrcTy, Type *DstTy, int64_t CstVal, Value *DynVal,\n"
    "                       size_t SmallerBitWidths, Value *Src, bool *Ok) {",
    "BitcastUtils.cpp definition",
)

# 3. the un-representable path returns instead of UB
step(
    utils_cpp,
    '      errs() << "Err: SrcTy = ";\n'
    "      SrcTy->print(errs());\n"
    '      errs() << " - DstTy = ";\n'
    "      DstTy->print(errs());\n"
    '      errs() << " - Ty = ";\n'
    "      Ty->print(errs());\n"
    '      errs() << " - CstVal = " << CstVal << "\\n";\n'
    '      llvm_unreachable("Unexpected offset for type in GetIdxsForTyFromOffset");',
    "      // metal2vk: the offset is not representable through this type path. Report\n"
    "      // failure to callers that ask and return the partial indices; folding this\n"
    "      // pair would produce a wrong GEP (and in Release llvm_unreachable is\n"
    "      // compiled out, so the old code fell through into undefined behaviour).\n"
    "      if (Ok) {\n"
    "        *Ok = false;\n"
    "      }\n"
    "      return Idxs;",
    "Err block in GetIdxsForTyFromOffset",
)

# 4. Changed starts false; only real folds set it. The worklist-tail context makes the
#    idiom unique (other sub-passes share the bare Changed line).
step(
    pass_cpp,
    "  const bool Changed = !WorkList.empty();\n\n"
    "  for (GetElementPtrInst *GEP : WorkList) {",
    "  bool Changed = false;\n\n"
    "  for (GetElementPtrInst *GEP : WorkList) {",
    "Changed initialiser in runOnGEPFromGEP",
)

# 5. the per-pair fold flag
step(
    pass_cpp,
    "    SmallVector<Value *, 8> Idxs;\n    if (OtherGEPPrevIsCstToStruct) {",
    "    SmallVector<Value *, 8> Idxs;\n"
    "    // metal2vk: the fold is skipped when the offset cannot be represented\n"
    "    // through the source type path (GetIdxsForTyFromOffset clears this flag).\n"
    "    bool M2vFoldOk = true;\n"
    "    if (OtherGEPPrevIsCstToStruct) {",
    "Idxs declaration in runOnGEPFromGEP",
)

# 6. the three fold paths ask for the flag. The first shape matches two call sites
#    (the constant-index and the constant-only branches share it).
s = pass_cpp.read_text()
for old, new, label in [
    (
        "          GEP->getResultElementType(), cstVal, dynVal, smallerBitWidths,\n"
        "          OtherGEP->getPointerOperand());",
        "          GEP->getResultElementType(), cstVal, dynVal, smallerBitWidths,\n"
        "          OtherGEP->getPointerOperand(), &M2vFoldOk);",
        "fold call sites (2x)",
    ),
    (
        "              OtherGEP->getResultElementType(), cstVal, dynVal,\n"
        "              smallerBitWidths, OtherGEP->getPointerOperand());",
        "              OtherGEP->getResultElementType(), cstVal, dynVal,\n"
        "              smallerBitWidths, OtherGEP->getPointerOperand(), &M2vFoldOk);",
        "fold call site (dynamic branch)",
    ),
]:
    if new in s and old not in s:
        continue
    if old not in s:
        raise SystemExit(f"anchor missing: {label}")
    s = s.replace(old, new)
pass_cpp.write_text(s)

# 7. skip before the replacement GEP is built
step(
    pass_cpp,
    "    Value *NewGEP = nullptr;",
    "    if (!M2vFoldOk) {\n"
    "      // metal2vk: leave the split GEPs in place; they are valid IR and the next\n"
    "      // pass will not re-report them as progress.\n"
    "      continue;\n"
    "    }\n"
    "    Value *NewGEP = nullptr;",
    "NewGEP declaration in runOnGEPFromGEP",
)

# 8. Changed counts only real folds
step(
    pass_cpp,
    "    // Remove the GEP as it has no users now.\n    GEP->eraseFromParent();\n",
    "    // Remove the GEP as it has no users now.\n"
    "    GEP->eraseFromParent();\n"
    "    Changed = true;\n",
    "GEP erase in runOnGEPFromGEP",
)

# 9. the producer emits OpUntypedAccessChainKHR on paths where the capability was never
#    declared (the only other addCapability sits inside getSPIRVPointerType's untyped
#    branch, which some base types bypass). Declaring a used capability is required;
#    extra declarations are harmless.
prod = root / "lib/SPIRVProducerPass.cpp"
step(
    prod,
    "    if (untyped) {\n      Opcode = spv::OpUntypedAccessChainKHR;\n",
    "    if (untyped) {\n"
    "      Opcode = spv::OpUntypedAccessChainKHR;\n"
    "      // metal2vk: this opcode requires the capability; some base types reach\n"
    "      // here without passing through the untyped branch of getSPIRVPointerType.\n"
    "      addCapability(spv::CapabilityUntypedPointersKHR);\n",
    "UntypedAccessChainKHR emission in SPIRVProducerPass",
)

print(f"patched (clspv HEAD {sha[:8]})")
