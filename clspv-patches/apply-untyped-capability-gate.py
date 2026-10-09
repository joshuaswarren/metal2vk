"""Gate the untyped-pointer capability on the --untyped-pointers option.

SPIRVProducerPassImpl::getSPIRVPointerType's untyped branch emitted
CapabilityUntypedPointersKHR and OpTypeUntypedPointerKHR whenever the address space
classified as untyped - even with --untyped-pointers OFF, where every OpVariable is typed
(UntypedPointerStorageClass returns false). With the GEP predicate fixed
(apply-untyped-gep-predicate.py) the qwen3_5_qmm / kimi-mma modules still failed spirv-val
with "Capability UntypedPointersKHR is not allowed by Vulkan": the capability leaked in
through this branch for pointer types created from storage-buffer address spaces.

Fix: the branch requires Option::UntypedPointers(), so with the option off the caller falls
through to the typed OpTypePointer path (with the inferred pointee type), consistent with
the OpVariable path. With the option on, behavior is unchanged.

Tested against clspv f2b01dd6 plus apply-coopmat-lowering.py, apply-undef-pointer-type.py,
apply-simplify-ptr-bitcast-converge.py, apply-untyped-gep-predicate.py and
apply-implicit-casts-converge.py.

Usage: python3 apply-untyped-capability-gate.py /path/to/clspv   (idempotent)
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
prod = root / "lib/SPIRVProducerPass.cpp"

sha = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if not sha.startswith("f2b01dd"):
    print(f"NOTE: clspv HEAD is {sha[:8]} (tested f2b01dd6); anchors may need adjustment",
          file=sys.stderr)

s = prod.read_text()
old = (
    "    addCapability(spv::CapabilityUntypedPointersKHR);\n"
    "    SPIRVOperandVec ops;\n"
    "    ops << GetStorageClass(canonical_aspace);\n"
    "    auto ptr_id = addSPIRVInst<kTypes>(spv::OpTypeUntypedPointerKHR, ops);\n"
    "\n"
    "    auto &entry = aspace_map[PtrTy];\n"
    "    if (entry.empty())\n"
    "      entry.resize(2);\n"
    "    entry[1] = ptr_id;\n"
    "\n"
    "    return ptr_id;\n"
)
new = (
    "    if (Option::UntypedPointers()) {\n"
    "      // metal2vk: with the option off every OpVariable is typed (see\n"
    "      // UntypedPointerStorageClass); emitting the untyped capability and type\n"
    "      // here anyway produced modules spirv-val rejects. Fall through to the\n"
    "      // typed pointer path below.\n"
    "      addCapability(spv::CapabilityUntypedPointersKHR);\n"
    "      SPIRVOperandVec ops;\n"
    "      ops << GetStorageClass(canonical_aspace);\n"
    "      auto ptr_id = addSPIRVInst<kTypes>(spv::OpTypeUntypedPointerKHR, ops);\n"
    "\n"
    "      auto &entry = aspace_map[PtrTy];\n"
    "      if (entry.empty())\n"
    "        entry.resize(2);\n"
    "      entry[1] = ptr_id;\n"
    "\n"
    "      return ptr_id;\n"
    "    }\n"
)
if new in s:
    print("already patched")
    sys.exit(0)
if s.count(old) != 1:
    raise SystemExit(f"anchor count mismatch: {s.count(old)} (want 1)")
prod.write_text(s.replace(old, new, 1))
print(f"patched (clspv HEAD {sha[:8]})")
