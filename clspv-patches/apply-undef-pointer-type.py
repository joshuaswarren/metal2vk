"""Fix clspv's SPIRVProducerPass for OpPhi pointer-type mismatches in SPIR-V.

Root cause: clspv's own loop transform pipeline (LoopRotate, SimplifyCFG, etc.)
creates PHINode values whose entry edge carries an LLVM `UndefValue` pointer.
Several phis that share the same LLVM value but feed different storage buffer
views (e.g. one phi follows a `char*` byte chain on arg X, another follows a
`bfloat` GEP chain on arg Y) end up with different inferred SPIR-V pointer
types. The producer caches constants by LLVM value in VMap and emits one
OpUndef for the shared undef. The first type wins and every later OpPhi that
reuses the cached ID gets an incoming value whose type does not match its
result type, which spirv-val rejects.

Fix: key a second map by the emitted SPIR-V pointer type and emit one
OpUndef per distinct type. When the caller passes a TyHint (phi emission
always does), use it directly: it is the type the phi was emitted with, and
the incoming value must match it. The producer's shared InferredTypeCache is
only consulted when no hint is available.

Tested against clspv f2b01dd6 plus the apply-coopmat-lowering.py patch.

Usage: python3 apply-undef-pointer-type.py /path/to/clspv   (idempotent, upgrades in place)
"""
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
prod = root / "lib/SPIRVProducerPass.cpp"


def head_sha(repo: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


sha = head_sha(root)
sha_short = sha[:8]
if not sha_short.startswith("f2b01dd"):
    print(
        f"NOTE: clspv HEAD is {sha_short} (tested f2b01dd6); anchors may need adjustment",
        file=sys.stderr,
    )

s = prod.read_text()

# ---- 1. member map ----------------------------------------------------------
a1 = "  DenseMap<Value *, Type *> InferredTypeCache;\n"
if a1 not in s:
    raise SystemExit("anchor missing: InferredTypeCache member")
if "M2vUndefPtrMap" not in s:
    s = s.replace(
        a1,
        a1
        + "  // metal2vk: per-SPIR-V-type cache for UndefValue pointer constants so\n"
        "  // phis that share one LLVM undef but infer different storage buffer\n"
        "  // views each get a matching OpUndef type. See getSPIRVValue below.\n"
        "  DenseMap<std::pair<const Value *, uint32_t>, SPIRVID> M2vUndefPtrMap;\n",
        1,
    )

# ---- 2. getSPIRVValue head: route hinted pointer-undef hits into the --------
# constant producer. Marker-delimited so re-running upgrades the block.
gv_new = (
    "SPIRVID SPIRVProducerPassImpl::getSPIRVValue(Value *V, Type *TyHint) {\n"
    "  auto II = ValueMap.find(V);\n"
    "  if (II != ValueMap.end()) {\n"
    "    assert(II->second.isValid());\n"
    "    // metal2vk: a pointer undef may need a fresh OpUndef under a different\n"
    "    // inferred pointee type; VMap is keyed by LLVM value alone, so without\n"
    "    // this re-route the cached ID's type is reused regardless of TyHint.\n"
    "    // See getSPIRVConstant.\n"
    "    if (TyHint && isa<UndefValue>(V) && V->getType()->isPointerTy()) {\n"
    "      return getSPIRVConstant(cast<Constant>(V), TyHint);\n"
    "    }\n"
    "    return II->second;\n"
    "  }\n"
)
gv_re = re.compile(
    r"SPIRVID SPIRVProducerPassImpl::getSPIRVValue\(Value \*V, Type \*TyHint\) \{\n"
    r"  auto II = ValueMap\.find\(V\);\n"
    r"  if \(II != ValueMap\.end\(\)\) \{\n"
    r"    assert\(II->second\.isValid\(\)\);\n"
    r"(?:.*?\n)*?"
    r"    return II->second;\n"
    r"  \}\n",
)
m = gv_re.search(s)
if not m:
    raise SystemExit("anchor missing: getSPIRVValue head")
s = s[: m.start()] + gv_new + s[m.end():]

# ---- 3. getSPIRVConstant: self-contained pointer-undef block keyed by -------
# (Cst, SPIR-V pointer-type id), placed before the VMap early return.
gc_new = (
    "  // metal2vk: an UndefValue pointer is one LLVM value but can need several\n"
    "  // SPIR-V pointer types. clspv's own loop transforms share a single undef\n"
    "  // across phis whose users infer different storage buffer views (e.g. an\n"
    "  // i8 view of one buffer and a bfloat struct view of another). VMap is\n"
    "  // keyed by the LLVM value alone, so the first emission won and every\n"
    "  // later OpPhi inherited a mismatched incoming type. Key a second map by\n"
    "  // the emitted SPIR-V pointer-type id and emit one OpUndef per type.\n"
    "  // The caller's TyHint is authoritative when present (phi emission passes\n"
    "  // the phi's own inferred type); the shared InferredTypeCache must not be\n"
    "  // consulted first, it pins the first-emitted type for every later user.\n"
    "  if (isa<UndefValue>(Cst) && Cst->getType()->isPointerTy()) {\n"
    "    Type *M2vInferredTy =\n"
    "        TyHint ? TyHint\n"
    "               : clspv::InferType(Cst, Cst->getContext(), &InferredTypeCache,\n"
    "                                  nullptr);\n"
    "    if (M2vInferredTy) {\n"
    "      SPIRVID M2vTyID = getSPIRVPointerType(Cst->getType(), M2vInferredTy);\n"
    "      std::pair<const Value *, uint32_t> M2vKey(Cst, M2vTyID.get());\n"
    "      auto M2vUI = M2vUndefPtrMap.find(M2vKey);\n"
    "      if (M2vUI != M2vUndefPtrMap.end()) {\n"
    "        return M2vUI->second;\n"
    "      }\n"
    "      SPIRVOperandVec M2vOps;\n"
    "      M2vOps << M2vTyID;\n"
    "      spv::Op M2vOpcode = spv::OpUndef;\n"
    "      if (hack_undef && IsTypeNullable(Cst->getType())) {\n"
    "        M2vOpcode = spv::OpConstantNull;\n"
    "      }\n"
    "      SPIRVID M2vRID = addSPIRVInst<kConstants>(M2vOpcode, M2vOps);\n"
    "      M2vUndefPtrMap[M2vKey] = M2vRID;\n"
    "      if (!VMap.count(Cst)) {\n"
    "        VMap[Cst] = M2vRID;\n"
    "      }\n"
    "      return M2vRID;\n"
    "    }\n"
    "  }\n"
    "\n"
)
gc_re = re.compile(
    r"  // metal2vk: an UndefValue pointer is one LLVM value but can need several\n"
    r"(?:.*?\n)*?"
    r"    \}\n"
    r"  \}\n"
    r"\n",
)
m = gc_re.search(s)
if m:
    s = s[: m.start()] + gc_new + s[m.end():]
else:
    a3 = (
        "  // Treat poison as an undef.\n"
        "  auto *Cst = C;\n"
        "  if (isa<PoisonValue>(Cst)) {\n"
        "    Cst = UndefValue::get(Cst->getType());\n"
        "  }\n"
        "\n"
    )
    if a3 not in s:
        raise SystemExit("anchor missing: getSPIRVConstant poison block")
    s = s.replace(a3, a3 + gc_new, 1)

prod.write_text(s)
print(f"patched {prod} (clspv HEAD {sha_short})")
