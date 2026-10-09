"""Teach clspv about metal2vk function constants: `constant volatile <ty> __m2v_spec_<idx> = <def>;` globals become
OpSpecConstant (SpecId = 1000 + idx, default = the initializer value) and every load of such a global is replaced by the
spec constant value. Nothing folds earlier in the pipeline (the volatile load survives instcombine, HideConstantLoads
protects it, UnhideConstantLoads restores a plain load before SPIR-V production), so the whole branch structure that
depends on the constant reaches SPIR-V and Vulkan pipeline specialization drives it.

Sites: SPIRVProducerPass (skip the global, intercept its loads), ClusterConstants and NormalizeGlobalVariable (leave
the global alone). Inert for modules without __m2v_spec_ globals.
Usage: python3 apply-spec-constants.py /path/to/clspv   (idempotent; apply after apply-coopmat-lowering.py)"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
# The marker is added by a separate script (apply-m2v-marker.py); apply
# that one first so the spec-constant compile test can prove the flag
# round-trips.
import subprocess as _sp
_sp.check_call([sys.executable, str(pathlib.Path(__file__).parent / "apply-m2v-marker.py"), str(root)])
prod = root / "lib/SPIRVProducerPass.cpp"
s = prod.read_text()
if "M2vGetSpecConstant" in s:
    print("producer already patched (spec constants)")
else:
    # 1. member state + declarations, next to the local-arg spec id maps
    a = """  // A mapping from Argument to its assigned SpecId.
  DenseMap<const Argument *, int> LocalArgSpecIds;
"""
    assert a in s, "producer member anchor not found"
    s = s.replace(a, a + """
  // metal2vk: `constant volatile <ty> __m2v_spec_<idx>` globals become
  // OpSpecConstant (SpecId = 1000 + idx, default = initializer value) and
  // loads of them read the spec constant value directly.
  DenseMap<const GlobalVariable *, SPIRVID> M2vSpecConstIDs;
  SPIRVID M2vGetSpecConstant(GlobalVariable *GV);
  static bool M2vIsSpecGlobal(const Value *V) {
    const auto *GV = dyn_cast<GlobalVariable>(V);
    return GV != nullptr && GV->getName().starts_with("__m2v_spec_");
  }
""", 1)

    # 2. definitions, before GenerateFuncBody (same anchor the coopmat patch uses)
    b = "void SPIRVProducerPassImpl::GenerateFuncBody(Function &F) {\n"
    assert b in s, "producer definition anchor not found"
    s = s.replace(b, """SPIRVID SPIRVProducerPassImpl::M2vGetSpecConstant(GlobalVariable *GV) {
  auto it = M2vSpecConstIDs.find(GV);
  if (it != M2vSpecConstIDs.end()) {
    return it->second;
  }
  unsigned idx = 0;
  GV->getName().drop_front(strlen("__m2v_spec_")).getAsInteger(10, idx);
  const uint32_t spec_id = 1000u + idx;
  SPIRVID ty_id = getSPIRVType(GV->getValueType());
  uint32_t literal = 0;
  if (auto *CI = dyn_cast<ConstantInt>(GV->getInitializer())) {
    literal = static_cast<uint32_t>(CI->getZExtValue());
  } else if (auto *FP = dyn_cast<ConstantFP>(GV->getInitializer())) {
    literal = static_cast<uint32_t>(FP->getValueAPF().bitcastToAPInt().getZExtValue());
  } else {
    llvm_unreachable("__m2v_spec_ globals need an i32/u32/f32 initializer");
  }
  SPIRVOperandVec Ops;
  Ops << ty_id << literal;
  SPIRVID id = addSPIRVInst<kConstants>(spv::OpSpecConstant, Ops);
  SPIRVOperandVec DOps;
  DOps << id << spv::DecorationSpecId << spec_id;
  addSPIRVInst<kAnnotations>(spv::OpDecorate, DOps);
  M2vSpecConstIDs[GV] = id;
  return id;
}

""" + b, 1)

    # 3. load interception
    c = """  case Instruction::Load: {
    LoadInst *LD = cast<LoadInst>(&I);
"""
    assert c in s, "producer load anchor not found"
    s = s.replace(c, c + """    // metal2vk: loads of __m2v_spec_ globals read the spec constant directly.
    if (auto *SpecGV = dyn_cast<GlobalVariable>(LD->getPointerOperand()->stripPointerCasts())) {
      if (M2vIsSpecGlobal(SpecGV)) {
        RID = M2vGetSpecConstant(SpecGV);
        break;
      }
    }
""", 1)

    # 4. keep the marked globals out of module-scope constant handling
    d = """  for (GlobalVariable &GV : module->globals()) {
    if (GV.getType()->getAddressSpace() == AddressSpace::Constant) {
      if (GV.use_empty() &&
          GV.getName() != clspv::ClusteredConstantsVariableName()) {
        DeadGVList.push_back(&GV);
      } else {
        GVList.push_back(&GV);
      }
    }
  }
"""
    assert d in s, "producer FindGlobalConstVars anchor not found"
    s = s.replace(d, """  for (GlobalVariable &GV : module->globals()) {
    if (M2vIsSpecGlobal(&GV)) {
      continue; // metal2vk: emitted as spec constants on first load
    }
    if (GV.getType()->getAddressSpace() == AddressSpace::Constant) {
      if (GV.use_empty() &&
          GV.getName() != clspv::ClusteredConstantsVariableName()) {
        DeadGVList.push_back(&GV);
      } else {
        GVList.push_back(&GV);
      }
    }
  }
""", 1)
    prod.write_text(s)
    print("patched SPIRVProducerPass.cpp")

cc = root / "lib/ClusterConstants.cpp"
s = cc.read_text()
if "__m2v_spec_" in s:
    print("cluster constants already patched")
else:
    e = """  for (GlobalVariable &GV : M.globals()) {
    if (GV.hasInitializer() && GV.getType()->getPointerAddressSpace() ==
                                   clspv::AddressSpace::Constant) {
"""
    assert e in s, "cluster anchor not found"
    s = s.replace(e, e + """      // metal2vk: spec-constant globals are emitted directly by the producer.
      if (GV.getName().starts_with("__m2v_spec_")) {
        continue;
      }
""", 1)
    cc.write_text(s)
    print("patched ClusterConstants.cpp")

ng = root / "lib/NormalizeGlobalVariable.cpp"
s = ng.read_text()
if "__m2v_spec_" in s:
    print("normalize globals already patched")
else:
    f = """  for (auto *GV : globals) {
    NormalizeVariableUsers(GV);
  }
"""
    assert f in s, "normalize anchor not found"
    s = s.replace(f, """  for (auto *GV : globals) {
    if (GV->getName().starts_with("__m2v_spec_")) {
      continue; // metal2vk: spec-constant globals normalize to themselves
    }
    NormalizeVariableUsers(GV);
  }
""", 1)
    ng.write_text(s)
    print("patched NormalizeGlobalVariable.cpp")
