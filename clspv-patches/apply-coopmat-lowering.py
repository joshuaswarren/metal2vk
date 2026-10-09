"""Add a clspv builtin, m2v_mma_f32(float2 a, float2 b, float2 c) -> float2, lowered to VK_KHR_cooperative_matrix.

Each lane passes its two fragment elements (Apple simdgroup_matrix lane layout, which Honeykrisp's 8x8 f32 cooperative
matrix shares: see probes/layout-run.py). The lowering writes them into function-scope coopmat variables with
OpAccessChain, issues OpCooperativeMatrixMulAddKHR (A*B+C, 8x8x8, subgroup scope), and reads the two result elements
back. Also drops llvm.trap (Metal kernels have fall-off-the-end paths that clang turns into traps at -O0).
Usage: python3 apply-coopmat-lowering.py /path/to/clspv   (idempotent)"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prod = root / "lib/SPIRVProducerPass.cpp"
s = prod.read_text()
if "M2vGenerateMma" in s:
    print("producer already patched")
else:
    # 1. declarations next to GenerateFuncBody
    a = "  void GenerateFuncBody(Function &F);\n"
    assert a in s
    s = s.replace(a, a + """  // metal2vk: lowering of m2v_mma_f32 to KHR cooperative matrix (8x8x8 f32, subgroup scope)
  void M2vEmitCoopVars(Function &F);
  SPIRVID M2vGenerateMma(CallInst *Call);
  bool M2vTypesReady = false;
  SPIRVID M2vCmTy[3], M2vPtrCmTy[3], M2vPtrFloat, M2vFloatTy;
  SPIRVID M2vVar[4];
""", 1)
    # 2. definitions before GenerateFuncBody
    b = "void SPIRVProducerPassImpl::GenerateFuncBody(Function &F) {\n"
    assert b in s
    s = s.replace(b, """static bool M2vIsMma(const Function *Callee) { return Callee && Callee->getName() == "m2v_mma_f32"; }

void SPIRVProducerPassImpl::M2vEmitCoopVars(Function &F) {
  bool found = false;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *C = dyn_cast<CallInst>(&I))
        if (M2vIsMma(C->getCalledFunction()))
          found = true;
  if (!found)
    return;
  addCapability(spv::CapabilityCooperativeMatrixKHR);
  LLVMContext &Ctx = module->getContext();
  if (!M2vTypesReady) {
    M2vTypesReady = true;
    Type *FloatTy = Type::getFloatTy(Ctx);
    M2vFloatTy = getSPIRVType(FloatTy);
    M2vPtrFloat = getSPIRVPointerType(PointerType::get(Ctx, 0), FloatTy);
    SPIRVID scope = getSPIRVInt32Constant(3); // Subgroup
    SPIRVID n8 = getSPIRVInt32Constant(8);
    for (uint32_t use = 0; use < 3; use++) { // 0 = A, 1 = B, 2 = Accumulator
      SPIRVOperandVec Ops;
      Ops << M2vFloatTy << scope << n8 << n8 << getSPIRVInt32Constant(use);
      M2vCmTy[use] = addSPIRVInst<kTypes>(spv::OpTypeCooperativeMatrixKHR, Ops);
      SPIRVOperandVec P;
      P << (uint32_t)spv::StorageClassFunction << M2vCmTy[use];
      M2vPtrCmTy[use] = addSPIRVInst<kTypes>(spv::OpTypePointer, P);
    }
  }
  const uint32_t uses[4] = {0, 1, 2, 2};
  for (int i = 0; i < 4; i++) {
    SPIRVOperandVec Ops;
    Ops << M2vPtrCmTy[uses[i]] << (uint32_t)spv::StorageClassFunction;
    M2vVar[i] = addSPIRVInst(spv::OpVariable, Ops);
  }
}

SPIRVID SPIRVProducerPassImpl::M2vGenerateMma(CallInst *Call) {
  SPIRVID idx[2] = {getSPIRVInt32Constant(0), getSPIRVInt32Constant(1)};
  auto store_pair = [&](SPIRVID var, Value *vec) {
    for (uint32_t e = 0; e < 2; e++) {
      SPIRVOperandVec X;
      X << M2vFloatTy << vec << e;
      SPIRVID elt = addSPIRVInst(spv::OpCompositeExtract, X);
      SPIRVOperandVec AC;
      AC << M2vPtrFloat << var << idx[e];
      SPIRVID ptr = addSPIRVInst(spv::OpAccessChain, AC);
      SPIRVOperandVec ST;
      ST << ptr << elt;
      addSPIRVInst(spv::OpStore, ST);
    }
  };
  store_pair(M2vVar[0], Call->getArgOperand(0));
  store_pair(M2vVar[1], Call->getArgOperand(1));
  store_pair(M2vVar[2], Call->getArgOperand(2));
  SPIRVID m[3];
  for (int i = 0; i < 3; i++) {
    SPIRVOperandVec L;
    L << M2vCmTy[i] << M2vVar[i];
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

""" + b, 1)
    # 3. call hook
    c = "  auto &func_info = Builtins::Lookup(Call->getCalledFunction());\n  auto func_type = func_info.getType();\n\n  if (BUILTIN_IN_GROUP(func_type, Clspv)) {"
    assert c in s
    s = s.replace(c, "  if (M2vIsMma(Call->getCalledFunction()))\n    return M2vGenerateMma(Call);\n\n" + c, 1)
    # 4. variables before the alloca loop of the entry block
    d = "    // OpVariable instructions must come first.\n"
    assert d in s
    s = s.replace(d, "    if (&BB == &F.getEntryBlock())\n      M2vEmitCoopVars(F);\n\n" + d, 1)
    # 5. extension
    e = "  if (CapabilitySet.count(spv::CapabilityFMAKHR)) {"
    assert e in s
    s = s.replace(e, "  if (CapabilitySet.count(spv::CapabilityCooperativeMatrixKHR)) {\n    addSPIRVInst<kExtensions>(spv::OpExtension, \"SPV_KHR_cooperative_matrix\");\n  }\n\n" + e, 1)
    prod.write_text(s)
    print("patched", prod)

# 6. binary writer must know the two new opcodes
s = prod.read_text()
if "case spv::OpTypeCooperativeMatrixKHR:" not in s:
    x = "    case spv::OpTypeVector:\n    case spv::OpTypeFunction:"
    y = "    case spv::OpFmaKHR:\n    case spv::OpFunction:"
    assert x in s and y in s
    s = s.replace(x, "    case spv::OpTypeVector:\n    case spv::OpTypeCooperativeMatrixKHR:\n    case spv::OpTypeFunction:", 1)
    s = s.replace(y, "    case spv::OpFmaKHR:\n    case spv::OpCooperativeMatrixMulAddKHR:\n    case spv::OpFunction:", 1)
    prod.write_text(s)
    print("patched writer cases")

# 7. Shader + CooperativeMatrixKHR requires the Vulkan memory model
s = prod.read_text()
if "MemoryModelVulkan" not in s:
    x = "  addCapability(spv::CapabilityCooperativeMatrixKHR);\n  LLVMContext &Ctx = module->getContext();"
    assert x in s
    s = s.replace(x, "  addCapability(spv::CapabilityCooperativeMatrixKHR);\n  addCapability(spv::CapabilityVulkanMemoryModel);\n  LLVMContext &Ctx = module->getContext();", 1)
    y = "      << spv::MemoryModelGLSL450;"
    assert y in s
    s = s.replace(y, "      << (CapabilitySet.count(spv::CapabilityVulkanMemoryModel) ? spv::MemoryModelVulkan : spv::MemoryModelGLSL450);", 1)
    z = "  if (CapabilitySet.count(spv::CapabilityCooperativeMatrixKHR)) {\n    addSPIRVInst<kExtensions>(spv::OpExtension, \"SPV_KHR_cooperative_matrix\");\n  }"
    assert z in s
    s = s.replace(z, z + "\n  if (CapabilitySet.count(spv::CapabilityVulkanMemoryModel)) {\n    addSPIRVInst<kExtensions>(spv::OpExtension, \"SPV_KHR_vulkan_memory_model\");\n  }", 1)
    prod.write_text(s)
    print("patched memory model")

# 8. C++ for OpenCL without the generic address space: an unqualified `this`, pointer or reference is private, which is
# what Metal's `thread` means (uzu declares thread-qualified methods/constructors that otherwise cannot be called).
comp = root / "lib/Compiler.cpp"
c = comp.read_text()
if "__opencl_c_generic_address_space" not in c:
    x = '  if (!clspv::Option::FP64()) {\n    Opts["cl_khr_fp64"] = false;\n  }\n'
    assert x in c
    c = c.replace(x, x + '  if (instance.getLangOpts().OpenCLCPlusPlus) {\n    Opts["__opencl_c_generic_address_space"] = false;\n    Opts["__opencl_c_pipes"] = false;\n    Opts["__opencl_c_device_enqueue"] = false;\n  }\n', 1)
    comp.write_text(c)
    print("patched Compiler.cpp")

# 9. GroupVectorUntilSizeEquals loops forever when a step does not grow the vector (and its llvm_unreachable is undefined
# behaviour in a Release build). Fail loudly with the offending types instead of hanging.
bu = root / "lib/BitcastUtils.cpp"
b = bu.read_text()
if "M2V: cannot group" not in b:
    x = "  while ((ValueEleSize * ValueNumEle) < TySize) {\n    if (ValueNumEle == 2) {\n      // <2 x i16> -> <4 x i16>"
    assert x in b
    b = b.replace(x, "  unsigned M2vIter = 0;\n  while ((ValueEleSize * ValueNumEle) < TySize) {\n    if (++M2vIter > 64) {\n      errs() << \"M2V: cannot group a \" << *ValueTy << \" value up to \" << *Ty << \" (now \" << ValueNumEle << \" x \" << ValueEleSize << \" bits, values \" << Values.size() << \")\\n\";\n      exit(3);\n    }\n    if (ValueNumEle == 2) {\n      // <2 x i16> -> <4 x i16>", 1)
    bu.write_text(b)
    print("patched BitcastUtils.cpp")

# 10. SimplifyPointerBitcastPass iterates its sub-passes to a fixpoint; on some inputs two of them undo each other and it
# never ends. Report which ones change things (and stop) instead of spinning.
sp = root / "lib/SimplifyPointerBitcastPass.cpp"
q = sp.read_text()
if "M2V: SimplifyPointerBitcast" not in q:
    a = q.index("  bool changed = true;\n  while (changed) {\n    changed = false;\n\n    changed |= runOnTrivialBitcast(M);")
    b = q.index("  return PA;", a)
    body = """  bool changed = true;
  unsigned M2vIter = 0;
  while (changed) {
    changed = false;
    bool c[9];
    c[0] = runOnTrivialBitcast(M);
    c[1] = runOnBitcastFromBitcast(M);
    c[2] = runOnImplicitGEP(M);
    c[3] = false;
    while (runOnUpgradeableConstantCasts(M)) {
      c[3] = true;
    }
    c[4] = runOnUnneededIndices(M);
    c[5] = runOnImplicitCasts(M);
    c[6] = runOnAllocaNotAliasing(M);
    c[7] = runOnPHIFromGEP(M);
    c[8] = runOnGEPFromGEP(M);
    for (bool x : c)
      changed |= x;
    if (++M2vIter > 200) {
      errs() << "M2V: SimplifyPointerBitcast does not converge; changing sub-passes:";
      for (int i = 0; i < 9; i++)
        if (c[i])
          errs() << " " << i;
      errs() << "\\n";
      exit(3);
    }
  }

"""
    q = q[:a] + body + q[b:]
    sp.write_text(q)
    print("patched SimplifyPointerBitcastPass.cpp")

rl = root / "lib/ReplaceLLVMIntrinsicsPass.cpp"
t = rl.read_text()
if "case Intrinsic::trap:" in t:
    print("trap already handled")
else:
    old = "  case Intrinsic::lifetime_end:\n    return removeIntrinsicDeclaration(F);"
    assert old in t
    rl.write_text(t.replace(old, "  case Intrinsic::lifetime_end:\n  // llvm.trap marks a fall-off-the-end path; SPIR-V has no trap, drop the call.\n  case Intrinsic::trap:\n    return removeIntrinsicDeclaration(F);", 1))
    print("patched", rl)
