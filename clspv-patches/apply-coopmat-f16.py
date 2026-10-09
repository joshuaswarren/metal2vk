"""Add m2v_mma_f16(half2 a, half2 b, float2 c) -> float2 on top of apply-coopmat-lowering.py.

Same lowering as m2v_mma_f32 but A and B are 8x8 half cooperative matrices (accumulator and result stay f32, the shape
Honeykrisp advertises as 8x8x8 f16/f16/f32). Run after apply-coopmat-lowering.py. Usage: python3 apply-coopmat-f16.py /path/to/clspv
(idempotent)"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prod = root / "lib/SPIRVProducerPass.cpp"
s = prod.read_text()
assert "M2vGenerateMma" in s, "run apply-coopmat-lowering.py first"
if "m2v_mma_f16" in s:
    print("f16 already patched")
    sys.exit(0)

# 1. members
a = "  SPIRVID M2vVar[4];\n"
assert a in s
s = s.replace(a, a + "  bool M2vHalfReady = false;\n  SPIRVID M2vHalfTy, M2vPtrHalf, M2vCmTyH[2], M2vPtrCmTyH[2], M2vVarH[2];\n", 1)

# 2. name test
a = 'static bool M2vIsMma(const Function *Callee) { return Callee && Callee->getName() == "m2v_mma_f32"; }\n'
assert a in s
s = s.replace(a, 'static bool M2vIsMma(const Function *Callee) { return Callee && (Callee->getName() == "m2v_mma_f32" || Callee->getName() == "m2v_mma_f16"); }\n'
              'static bool M2vIsMmaF16(const Function *Callee) { return Callee && Callee->getName() == "m2v_mma_f16"; }\n', 1)

# 3. half types and variables, appended to the end of M2vEmitCoopVars
a = "    M2vVar[i] = addSPIRVInst(spv::OpVariable, Ops);\n  }\n}\n"
assert a in s
s = s.replace(a, """    M2vVar[i] = addSPIRVInst(spv::OpVariable, Ops);
  }
  bool found16 = false;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *C = dyn_cast<CallInst>(&I))
        if (M2vIsMmaF16(C->getCalledFunction()))
          found16 = true;
  if (!found16)
    return;
  addCapability(spv::CapabilityFloat16);
  if (!M2vHalfReady) {
    M2vHalfReady = true;
    Type *HalfTy = Type::getHalfTy(Ctx);
    M2vHalfTy = getSPIRVType(HalfTy);
    M2vPtrHalf = getSPIRVPointerType(PointerType::get(Ctx, 0), HalfTy);
    SPIRVID scope = getSPIRVInt32Constant(3);
    SPIRVID n8 = getSPIRVInt32Constant(8);
    for (uint32_t use = 0; use < 2; use++) {
      SPIRVOperandVec Ops;
      Ops << M2vHalfTy << scope << n8 << n8 << getSPIRVInt32Constant(use);
      M2vCmTyH[use] = addSPIRVInst<kTypes>(spv::OpTypeCooperativeMatrixKHR, Ops);
      SPIRVOperandVec P;
      P << (uint32_t)spv::StorageClassFunction << M2vCmTyH[use];
      M2vPtrCmTyH[use] = addSPIRVInst<kTypes>(spv::OpTypePointer, P);
    }
  }
  for (int i = 0; i < 2; i++) {
    SPIRVOperandVec Ops;
    Ops << M2vPtrCmTyH[i] << (uint32_t)spv::StorageClassFunction;
    M2vVarH[i] = addSPIRVInst(spv::OpVariable, Ops);
  }
}
""", 1)

# 4. replace M2vGenerateMma with a version that takes the A/B element type from the callee
start = s.index("SPIRVID SPIRVProducerPassImpl::M2vGenerateMma(CallInst *Call) {")
end = s.index("\n}\n", start) + 3
new = """SPIRVID SPIRVProducerPassImpl::M2vGenerateMma(CallInst *Call) {
  const bool h = M2vIsMmaF16(Call->getCalledFunction());
  SPIRVID idx[2] = {getSPIRVInt32Constant(0), getSPIRVInt32Constant(1)};
  auto store_pair = [&](SPIRVID var, Value *vec, SPIRVID eltTy, SPIRVID ptrTy) {
    for (uint32_t e = 0; e < 2; e++) {
      SPIRVOperandVec X;
      X << eltTy << vec << e;
      SPIRVID elt = addSPIRVInst(spv::OpCompositeExtract, X);
      SPIRVOperandVec AC;
      AC << ptrTy << var << idx[e];
      SPIRVID ptr = addSPIRVInst(spv::OpAccessChain, AC);
      SPIRVOperandVec ST;
      ST << ptr << elt;
      addSPIRVInst(spv::OpStore, ST);
    }
  };
  SPIRVID var[3] = {h ? M2vVarH[0] : M2vVar[0], h ? M2vVarH[1] : M2vVar[1], M2vVar[2]};
  SPIRVID cty[3] = {h ? M2vCmTyH[0] : M2vCmTy[0], h ? M2vCmTyH[1] : M2vCmTy[1], M2vCmTy[2]};
  store_pair(var[0], Call->getArgOperand(0), h ? M2vHalfTy : M2vFloatTy, h ? M2vPtrHalf : M2vPtrFloat);
  store_pair(var[1], Call->getArgOperand(1), h ? M2vHalfTy : M2vFloatTy, h ? M2vPtrHalf : M2vPtrFloat);
  store_pair(var[2], Call->getArgOperand(2), M2vFloatTy, M2vPtrFloat);
  SPIRVID m[3];
  for (int i = 0; i < 3; i++) {
    SPIRVOperandVec L;
    L << cty[i] << var[i];
    m[i] = addSPIRVInst(spv::OpLoad, L);
  }
  SPIRVOperandVec MA;
  MA << M2vCmTy[2] << m[0] << m[1] << m[2];
  SPIRVID d = addSPIRVInst(spv::OpCooperativeMatrixMulAddKHR, MA);
  SPIRVOperandVec SD;
  SD << M2vVar[3] << d;
  addSPIRVInst(spv::OpStore, SD);
  SPIRVID el[2];
  for (int e = 0; e < 2; e++) {
    SPIRVOperandVec AC;
    AC << M2vPtrFloat << M2vVar[3] << idx[e];
    SPIRVID ptr = addSPIRVInst(spv::OpAccessChain, AC);
    SPIRVOperandVec LD;
    LD << M2vFloatTy << ptr;
    el[e] = addSPIRVInst(spv::OpLoad, LD);
  }
  SPIRVOperandVec CC;
  CC << Call->getType() << el[0] << el[1];
  return addSPIRVInst(spv::OpCompositeConstruct, CC);
}
"""
s = s[:start] + new + s[end:]

# 5. the Coherent guard looks for either builtin
a = 'return M && M->getFunction("m2v_mma_f32") != nullptr; }'
assert a in s
s = s.replace(a, 'return M && (M->getFunction("m2v_mma_f32") != nullptr || M->getFunction("m2v_mma_f16") != nullptr); }', 1)
prod.write_text(s)
print("patched f16 lowering")
