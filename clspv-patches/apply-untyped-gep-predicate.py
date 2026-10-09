"""Make the GEP producer path agree with the OpVariable path on untyped pointers.

With --untyped-pointers OFF (our default), SPIRVProducerPassImpl::UntypedPointerStorageClass
returns false and every OpVariable is typed. The GEP instruction path, however, classified
`untyped` with a different predicate - Option::UntypedPointerAddressSpace - which only looks
at the address space and ignores the option. A GEP on a storage-buffer pointer then emitted
OpUntypedAccessChainKHR on an otherwise typed module; spirv-val rejects it ("Capability
UntypedPointersKHR is not allowed by Vulkan" or the missing-capability variant, depending on
what else declared). These are the qwen3_5_qmm / kimi mma modules whose SimplifyPointerBitcast
folds are now skipped by apply-simplify-ptr-bitcast-converge.py: the surviving split GEPs are
exactly the shape that reached this disagreeing path.

Fix: the GEP path uses the same two-part predicate as the OpVariable path - the option must
be on AND the address space must be an untyped one.

Tested against clspv f2b01dd6 plus apply-coopmat-lowering.py, apply-undef-pointer-type.py and
apply-simplify-ptr-bitcast-converge.py.

Usage: python3 apply-untyped-gep-predicate.py /path/to/clspv   (idempotent)
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
new = (
    "    const bool untyped =\n"
    "        Option::UntypedPointers() &&\n"
    "        Option::UntypedPointerAddressSpace(ResultType->getAddressSpace());\n"
    "    // metal2vk: this path must agree with UntypedPointerStorageClass, which\n"
    "    // gates on the option as well; with the option off the old predicate still\n"
    "    // classified storage-buffer address spaces as untyped and emitted\n"
    "    // OpUntypedAccessChainKHR on typed modules, which spirv-val rejects.\n"
)
old = (
    "    const bool untyped =\n"
    "        Option::UntypedPointerAddressSpace(ResultType->getAddressSpace());\n"
)
if new in s:
    print("already patched")
    sys.exit(0)
if s.count(old) != 1:
    raise SystemExit(f"anchor count mismatch in SPIRVProducerPass.cpp: {s.count(old)} (want 1)")
prod.write_text(s.replace(old, new, 1))
print(f"patched (clspv HEAD {sha[:8]})")
