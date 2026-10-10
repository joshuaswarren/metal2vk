"""Merge implicit-cast GEP pairs at the finer granularity instead of dividing.

runOnImplicitCasts folded a pair like `gep i8, (gep i32 base, %a), %b` by
regenerating indices through GetIdxsForTyFromOffset at the inner element's
granularity. For a dynamic offset that path divides (%b / 4) and drops the
sub-word remainder, and the reverse conversion (i32 units back to i8 units)
multiplies again - so the same pair re-folds every outer-loop iteration,
appending one lshr/shl link per pass and decaying the offset (the 3
ops/qmv.metal:tf_gather_qmv entries never converged and finished on an
OpAccessChain result-type mismatch).

Fix: when the outer GEP's offset is dynamic, merge the pair at the finer of
the two granularities. Both offsets are scaled UP to that unit - a multiply,
which is lossless - and summed into one index on the inner GEP's base. The
resulting byte-granular chain is the shape ReplacePointerBitcastPass lowers
into typed word loads. Aggregate and unsized element types keep the old
regenerated-index path.

Tested against clspv f2b01dd6 plus apply-coopmat-lowering.py,
apply-coopmat-f16.py, apply-undef-pointer-type.py,
apply-simplify-ptr-bitcast-converge.py, apply-untyped-gep-predicate.py,
apply-untyped-capability-gate.py, apply-implicit-casts-converge.py,
apply-implicit-casts-hash-break.py, apply-simplify-iteration-hash-break.py
and apply-simplify-no-exit.py.

Usage: python3 apply-implicit-casts-lossless-merge.py /path/to/clspv   (idempotent)
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
p = root / "lib/SimplifyPointerBitcastPass.cpp"

sha = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if not sha.startswith("f2b01dd"):
    print(f"NOTE: clspv HEAD is {sha[:8]} (tested f2b01dd6); anchors may need adjustment",
          file=sys.stderr)

s = p.read_text()
old = (
    "      Idxs = GetIdxsForTyFromOffset(\n"
    "          DL, Builder, src_ty, inst_gep->getResultElementType(), CstVal, DynVal,\n"
    "          SmallerBitWidths, src_gep->getPointerOperand(), &M2vFoldOk);\n"
    "    }\n"
)
new = (
    "      if (DynVal != nullptr && SizeInBits(DL, src_ty) != 0 &&\n"
    "          SizeInBits(DL, inst_gep->getResultElementType()) != 0 &&\n"
    "          !src_ty->isAggregateType() &&\n"
    "          !inst_gep->getResultElementType()->isAggregateType()) {\n"
    "        // metal2vk: merge the pair at the finer of the two granularities.\n"
    "        // The regenerated-index path below divides a dynamic offset to the\n"
    "        // inner element's granularity (losing the sub-word remainder) and\n"
    "        // the next conversion multiplies it back, so the pair re-folds on\n"
    "        // every outer-loop iteration and the offset decays. Scaling both\n"
    "        // offsets up to the finer unit is a multiply, which is lossless.\n"
    "        int64_t InnerCst;\n"
    "        Value *InnerDyn;\n"
    "        size_t InnerSmaller;\n"
    "        ExtractOffsetFromGEP(DL, Builder, src_gep, InnerCst, InnerDyn,\n"
    "                             InnerSmaller);\n"
    "        const size_t Target =\n"
    "            InnerSmaller < SmallerBitWidths ? InnerSmaller : SmallerBitWidths;\n"
    "        Type *FineTy = SmallerBitWidths <= InnerSmaller\n"
    "                           ? inst_gep->getResultElementType()\n"
    "                           : src_ty;\n"
    "        const size_t KIn = InnerSmaller / Target;\n"
    "        const size_t KOut = SmallerBitWidths / Target;\n"
    "        int64_t Cst = InnerCst * static_cast<int64_t>(KIn) +\n"
    "                      CstVal * static_cast<int64_t>(KOut);\n"
    "        Value *Dyn = nullptr;\n"
    "        if (InnerDyn) {\n"
    "          Dyn = KIn == 1 ? InnerDyn : CreateMul(Builder, KIn, InnerDyn);\n"
    "        }\n"
    "        if (DynVal) {\n"
    "          Value *M = KOut == 1 ? DynVal : CreateMul(Builder, KOut, DynVal);\n"
    "          Dyn = Dyn ? Builder.CreateAdd(Dyn, M) : M;\n"
    "        }\n"
    "        if (Cst != 0) {\n"
    "          auto *IdxTy = Dyn ? Dyn->getType() : Builder.getInt32Ty();\n"
    "          Value *CstV = ConstantInt::get(IdxTy, Cst);\n"
    "          Dyn = Dyn ? Builder.CreateAdd(Dyn, CstV) : CstV;\n"
    "        }\n"
    "        if (!Dyn) {\n"
    "          Dyn = Builder.getInt32(0);\n"
    "        }\n"
    "        Idxs.clear();\n"
    "        Idxs.push_back(Dyn);\n"
    "        src = src_gep->getPointerOperand();\n"
    "        src_ty = FineTy;\n"
    "      } else {\n"
    "        Idxs = GetIdxsForTyFromOffset(\n"
    "            DL, Builder, src_ty, inst_gep->getResultElementType(), CstVal,\n"
    "            DynVal, SmallerBitWidths, src_gep->getPointerOperand(),\n"
    "            &M2vFoldOk);\n"
    "      }\n"
    "    }\n"
)
if new in s:
    print("already patched")
    sys.exit(0)
if s.count(old) != 1:
    raise SystemExit(f"anchor count mismatch: {s.count(old)} (want 1)")
p.write_text(s.replace(old, new, 1))
print(f"patched (clspv HEAD {sha[:8]})")
