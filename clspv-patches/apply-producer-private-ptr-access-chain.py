"""Handle private-storage OpPtrAccessChain in the producer's GEP path.

SPIRVProducerPassImpl's GEP case classifies an access chain with a non-zero
first index as OpPtrAccessChain and then switches on the storage class. The
switch has no case for Function/Private, so a vector GEP with a non-zero first
index on private memory (the simdgroup accumulator tiles in the qwen3_5_qmm,
kimi mma and tf_qmm_splitk modules) falls into `default: llvm_unreachable`,
which is compiled out in Release. The miscompiled fall-through executed the
untyped needs_array path with --untyped-pointers OFF and emitted
OpCapability UntypedPointersKHR plus OpUntypedAccessChainKHR on an otherwise
typed module, which spirv-val rejects.

Fix: Function and Private are legal storage classes for OpPtrAccessChain with
the VariablePointers capability (the !untyped path already calls
setVariablePointersCapabilities for this opcode). Handle them with a plain
break so the typed OpPtrAccessChain is emitted.

Tested against clspv f2b01dd6 plus apply-coopmat-lowering.py,
apply-coopmat-f16.py, apply-undef-pointer-type.py,
apply-simplify-ptr-bitcast-converge.py, apply-untyped-gep-predicate.py,
apply-untyped-capability-gate.py, apply-implicit-casts-converge.py,
apply-implicit-casts-hash-break.py, apply-simplify-iteration-hash-break.py
and apply-simplify-no-exit.py.

Usage: python3 apply-producer-private-ptr-access-chain.py /path/to/clspv   (idempotent)
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
    "      case spv::StorageClassUniform:\n"
    "      case spv::StorageClassPushConstant:\n"
    "        if (untyped) {\n"
    "          needs_array = true;\n"
    "        } else {\n"
    "          llvm_unreachable(\n"
    "              \"OpPtrAccessChain is not supported for this storage class\");\n"
    "        }\n"
    "        break;\n"
    "      default:\n"
    "        llvm_unreachable(\n"
    "            \"OpPtrAccessChain is not supported for this storage class\");\n"
    "        break;\n"
    "      }\n"
    "    }\n"
)
new = (
    "      case spv::StorageClassUniform:\n"
    "      case spv::StorageClassPushConstant:\n"
    "        if (untyped) {\n"
    "          needs_array = true;\n"
    "        } else {\n"
    "          llvm_unreachable(\n"
    "              \"OpPtrAccessChain is not supported for this storage class\");\n"
    "        }\n"
    "        break;\n"
    "      case spv::StorageClassFunction:\n"
    "      case spv::StorageClassPrivate:\n"
    "        // metal2vk: vector GEPs with a non-zero first index on private\n"
    "        // memory (simdgroup accumulator tiles) reach OpPtrAccessChain.\n"
    "        // The unreachable default compiled out in Release and the\n"
    "        // fall-through ran the untyped needs_array path with\n"
    "        // --untyped-pointers OFF, emitting OpUntypedAccessChainKHR and the\n"
    "        // capability on typed modules. OpPtrAccessChain is legal here with\n"
    "        // VariablePointers, which the !untyped path below requests.\n"
    "        break;\n"
    "      default:\n"
    "        llvm_unreachable(\n"
    "            \"OpPtrAccessChain is not supported for this storage class\");\n"
    "        break;\n"
    "      }\n"
    "    }\n"
)
if new in s:
    print("already patched")
    sys.exit(0)
if s.count(old) != 1:
    raise SystemExit(f"anchor count mismatch: {s.count(old)} (want 1)")
prod.write_text(s.replace(old, new, 1))
print(f"patched (clspv HEAD {sha[:8]})")
