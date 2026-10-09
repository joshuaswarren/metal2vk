# clspv bug: OpPhi with one incoming edge in a two predecessor block

## Symptom

GatedActMul_v1 (uzu gated_act_mul.metal, bfloat variant) compiles through clang and the
patched clspv, but spirv-val rejects the module:

    error: line 7003: OpPhi's number of incoming blocks (1) does not match block's predecessor count (2).
      %6439 = OpPhi %uint %uint_0 %6433

The block holding the phi has two predecessors (`OpBranchConditional %6430 %6433 %6437`
into `%6437`, and `OpBranch %6437` from `%6433`). Three sibling phis in the same block
list both predecessors; `%6439` lists only `%6433`.

## Repro

`repro/gated_act_mul_v1_opphi.ll` (1368 lines, reduced from the failing module with
llvm-reduce; the interestingness test is clspv producing a module that spirv-val rejects
with exactly this message). With `CLSPV` pointing at the patched clspv:

    $CLSPV -x ir --cl-std=CLC++2021 --fp16 --inline-entry-points --spv-version=1.5 \
        repro/gated_act_mul_v1_opphi.ll -o out.spv
    spirv-val --target-env vulkan1.3 out.spv   # OpPhi predecessor count error

The same error text and phi id appear as for the unreduced module, so the reduction kept
the failing region intact.

## Analysis

The input IR is valid: every phi has one incoming per predecessor, confirmed on the final
module dump (`clspv ... -print-after-all`, last dump) which feeds codegen. The corruption
is therefore introduced during the LLVM SPIR-V backend codegen stage inside the clspv
tree (`third_party/llvm/llvm/lib/Target/SPIRV`), not by a clspv IR pass. The malformed
OpPhi keeps the constant incoming of one edge and drops the other edge of a
two-predecessor block.

The failing region is the activation type dispatch: a switch on ActivationType lowered to
compare chains, restructured by clspv's StructurizeCFG into `Flow` blocks, with the
helper `activate<vec<bfloat, 4>::type::ref>` threaded through. The block merges the silu
branch and the dispatch fallthrough; during codegen the phi bookkeeping of that merge
loses the fallthrough edge for one value while sibling values in the same block keep both
edges.

## Proposed fix

In the SPIR-V backend, at OpPhi emission, rebuild the incoming list from the
MachineBasicBlock predecessor set at emission time instead of trusting the phi operand
list accumulated before legalization: emit one incoming per predecessor, using the
existing value where present and an undef (or the phi's other constant) for any edge the
phi lost. That shape is valid by construction. The relevant code paths are the phi
handling in `llvm/lib/Target/SPIRV` (`SPIRVPreLegalizer.cpp` CFG surgery and the
`G_PHI`/`PHI` paths in `SPIRVInstructionSelector.cpp`).

A clspv-side workaround is not available: the IR the clspv passes see is already valid,
and the breakage happens below the pass pipeline. Until the backend is fixed, the entry
stays in the failing set with this error string.
