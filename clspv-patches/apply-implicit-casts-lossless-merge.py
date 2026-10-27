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

Guard (v2): the merge runs only while `src` is still the inner GEP. When the
outer offset is a constant that does not land on a field of the inner element
(a byte copy at k % 4 != 0 out of `gep [8 x Conf], 0, %g`), the branch above
has already folded the inner offset into DynVal and moved `src` to the base;
merging again counted `%g * 16` twice and read bytes at `(%g << 5) | k`: a
valid module with a wrong result (cases/aggregate_select.metal). A tree
patched by v1 is upgraded in place. The marker string lets
sweep/tests/run.py (lossless_merge_guard) recognise a binary built with v2.

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
# v2: the merge must not run when the constant-offset branch above already moved src to the base
guard_old = "      if (DynVal != nullptr && SizeInBits(DL, src_ty) != 0 &&\n"
guard_new = (
    "      if (DynVal != nullptr && src == src_gep &&\n"
    "          SizeInBits(DL, src_ty) != 0 &&\n"
)
mark_at = "        // offsets up to the finer unit is a multiply, which is lossless.\n"
mark = (
    "        // metal2vk: only while src is still src_gep. Otherwise the branch\n"
    "        // above already added src_gep's offset to DynVal; adding it again\n"
    "        // doubles the element offset (cases/aggregate_select.metal).\n"
    "        static const char *const volatile M2vLosslessMergeGuard =\n"
    "            \"metal2vk:apply-implicit-casts-lossless-merge:v2\";\n"
    "        (void)M2vLosslessMergeGuard;\n"
)
if "metal2vk:apply-implicit-casts-lossless-merge:v2" in s:
    print("already patched")
    sys.exit(0)
if new not in s:
    if s.count(old) != 1:
        raise SystemExit(f"anchor count mismatch: {s.count(old)} (want 1)")
    s = s.replace(old, new, 1)
    action = "patched"
else:
    action = "upgraded v1 to v2"
if s.count(guard_old) != 1 or s.count(mark_at) != 1:
    raise SystemExit(f"v2 anchor count mismatch: {s.count(guard_old)}, {s.count(mark_at)} (want 1, 1)")
s = s.replace(guard_old, guard_new, 1).replace(mark_at, mark_at + mark, 1)
p.write_text(s)
print(f"{action} (clspv HEAD {sha[:8]})")
