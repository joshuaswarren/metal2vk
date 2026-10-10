"""Lower bool-vector <-> small-integer bitcasts to legal SPIR-V in the producer.

The two tf_glm5_fused_router variants fail spirv-val with "Expected input to be
a pointer or int or float vector or scalar: Bitcast" on `OpBitcast %uchar %151`
where %151 is a %v4bool. LLVM's LoopVectorize writes the early-exit test of a
vectorized loop as

  %29 = icmp ugt <4 x i32> %vec.ind, splat(i32 -225)
  %30 = bitcast <4 x i1> %29 to i4
  %.not = icmp eq i4 %30, 0

SPIR-V has no OpBitcast from a bool vector, and the producer's CanonicalType
rounds the i4 result to i8 while the <4 x i1> operand cannot round, so the
plain cast mapping emits OpBitcast %uchar from %v4bool and the module is
rejected.

Fix (SPIRVProducerPassImpl, cast visitor): a BitCast whose operand is an
integer vector of i1 and whose result is a wider integer (or the mirror image)
is lowered instead of mapped:

  bool vector -> int:  OpSelect widens every lane to i32, the lanes are
                       shift-ored into a scalar (lane i at bit i, exactly the
                       little-endian packing of LLVM's bitcast) and the scalar
                       is converted to the result's canonical type.
  int -> bool vector:  the scalar is widened to i32, each lane is extracted
                       with shift-right/and-1/OpINotEqual and the vector is
                       built with OpCompositeConstruct.

Every emitted opcode is legal for bool and integer types in Vulkan SPIR-V, so
the fold survives type canonicalisation instead of reaching the producer as an
unrepresentable bitcast. Vectors longer than 64 lanes are left untouched (no
such module exists in the corpus).

Usage: python3 apply-bool-vector-bitcast.py /path/to/clspv   (idempotent)
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])

sha = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if not sha.startswith("f2b01dd"):
    print(f"NOTE: clspv HEAD is {sha[:8]} (tested f2b01dd6); anchors may need adjustment",
          file=sys.stderr)

prod = root / "lib/SPIRVProducerPass.cpp"
s = prod.read_text()
if "GenerateBoolVectorBitcast" in s:
    print("already patched")
    sys.exit(0)


def step(old, new, label):
    global s
    if old not in s:
        raise SystemExit(f"anchor missing: {label}")
    s = s.replace(old, new, 1)


# 1. the branch: inserted after the trunc-to-i1 special case, before the
#    generic OpBitcast fallback
step(
    """        Ops.clear();
        Ops << Ty << oAnd1 << getSPIRVValue(ConstantInt::get(OpTy, 0));
        RID = addSPIRVInst(spv::OpINotEqual, Ops);
      } else {
""",
    """        Ops.clear();
        Ops << Ty << oAnd1 << getSPIRVValue(ConstantInt::get(OpTy, 0));
        RID = addSPIRVInst(spv::OpINotEqual, Ops);
      } else if (I.getOpcode() == Instruction::BitCast &&
                 OpTy->isIntOrIntVectorTy(1) && OpTy->isVectorTy() &&
                 Ty->isIntOrIntVectorTy() && !Ty->isIntOrIntVectorTy(1)) {
        // metal2vk: LLVM's LoopVectorize writes the early-exit test of a
        // vectorized loop as a bitcast of a bool vector to a small integer
        // (icmp <4 x i32> ... -> bitcast <4 x i1> to i4 -> icmp eq i4 0).
        // SPIR-V has no OpBitcast from a bool vector, and the canonical type
        // of i4 is i8 while the bool vector cannot round, so the plain cast
        // mapping would emit OpBitcast %uchar from %v4bool. Lower the pack
        // (and its mirror-image unpack) instead.
        RID = GenerateBoolVectorBitcast(I);
      } else {
""",
    "cast visitor branch",
)

# 2. the declaration next to the other Generate* helpers
step(
    "  SPIRVID GenerateFabs(SPIRVID Input, Type *InputTy);\n",
    "  SPIRVID GenerateFabs(SPIRVID Input, Type *InputTy);\n"
    "  // metal2vk: legal lowering for bool-vector <-> small-integer bitcasts.\n"
    "  SPIRVID GenerateBoolVectorBitcast(CastInst &I);\n",
    "helper declaration",
)

# 3. the definition, placed before GenerateFabs's definition
step(
    "SPIRVID SPIRVProducerPassImpl::GenerateFabs(SPIRVID Input, Type *InputTy) {",
    """SPIRVID SPIRVProducerPassImpl::GenerateBoolVectorBitcast(CastInst &I) {
  auto *VecTy = cast<FixedVectorType>(I.getOperand(0)->getType());
  const unsigned lanes = VecTy->getNumElements();
  auto *I32 = Type::getInt32Ty(Context);
  auto *WideTy = (lanes <= 32) ? I32 : Type::getInt64Ty(Context);
  SPIRVID pack = getSPIRVConstant(ConstantInt::get(WideTy, 0));
  SPIRVID result;
  SPIRVOperandVec Ops;

  if (auto *DstVec = dyn_cast<FixedVectorType>(I.getType())) {
    // scalar integer -> bool vector: widen, then peel one bit per lane
    Ops << WideTy << I.getOperand(0);
    auto widened = addSPIRVInst(spv::OpUConvert, Ops);
    SmallVector<SPIRVID, 8> bits;
    for (unsigned i = 0; i < DstVec->getNumElements(); i++) {
      Ops.clear();
      Ops << WideTy << widened << getSPIRVConstant(ConstantInt::get(WideTy, i));
      auto shifted = addSPIRVInst(spv::OpShiftRightLogical, Ops);
      Ops.clear();
      Ops << WideTy << shifted << getSPIRVConstant(ConstantInt::get(WideTy, 1));
      auto masked = addSPIRVInst(spv::OpBitwiseAnd, Ops);
      Ops.clear();
      Ops << WideTy << masked
          << getSPIRVConstant(ConstantInt::get(WideTy, 0));
      bits.push_back(addSPIRVInst(spv::OpINotEqual, Ops));
    }
    Ops.clear();
    Ops << I.getType();
    for (auto b : bits) {
      Ops << b;
    }
    result = addSPIRVInst(spv::OpCompositeConstruct, Ops);
  } else {
    // bool vector -> scalar integer: widen every lane with OpSelect, then
    // shift-or them into the scalar exactly as LLVM packs them (lane i at
    // bit i)
    auto *SelTy = FixedVectorType::get(WideTy, lanes);
    Ops << SelTy << I.getOperand(0)
        << ConstantVector::getSplat(VecTy->getElementCount(),
                                    ConstantInt::get(WideTy, 1))
        << Constant::getNullValue(SelTy);
    auto widened = addSPIRVInst(spv::OpSelect, Ops);
    for (unsigned i = 0; i < lanes; i++) {
      Ops.clear();
      Ops << WideTy << widened << i;
      auto lane = addSPIRVInst(spv::OpCompositeExtract, Ops);
      Ops.clear();
      Ops << WideTy << lane << getSPIRVConstant(ConstantInt::get(WideTy, i));
      auto shifted = addSPIRVInst(spv::OpShiftLeftLogical, Ops);
      Ops.clear();
      Ops << WideTy << pack << shifted;
      pack = addSPIRVInst(spv::OpBitwiseOr, Ops);
    }
    if (I.getType() == WideTy) {
      result = pack;
    } else {
      Ops.clear();
      Ops << I.getType() << pack;
      result = addSPIRVInst(spv::OpUConvert, Ops);
    }
  }
  return result;
}

SPIRVID SPIRVProducerPassImpl::GenerateFabs(SPIRVID Input, Type *InputTy) {""",
    "helper definition",
)

prod.write_text(s)
print(f"patched (clspv HEAD {sha[:8]})")
