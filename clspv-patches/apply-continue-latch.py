"""Pick the loop latch itself as the SPIR-V continue target when a unique latch exists, and refresh the cached
analyses around FixupStructuredCFGPass's CFG surgery.

Root cause: clspv's SPIRVProducerPassImpl::PopulateStructuredCFGMaps and
clspv::ComputeStructuredOrder choose a loop's continue target by scanning the
loop's blocks for "the last block that dominates the back-edge block". In a
loop whose body ends with a nested loop, the nested loop's exit block
dominates the latch through the single exit chain (nested exit -> latch), and
in block layout order it comes after the latch, so the scan picks it. The
nested loop's merge block is that same exit block, and a merge block must not
double as the continue target of the enclosing loop. spirv-val rejects the
module with "Header block 'X' is contained in the loop construct headed by
'Y', but its merge block 'Z' is not" (nemotron/attn_partial_*, ops/gemv wide,
and on the structurized trees of the vec-deduction slice many more).

But preferring the latch is not enough on its own, because in the LLVM IR the
nested loop's unique exit block can BE the parent loop's latch (one block,
both roles), and clspv's own FixupStructuredCFGPass::isolateContinue exists
to insert a separating block for exactly that case - it just never fires on
the current pinned LLVM: the three CFG-rewriting helpers in
FixupStructuredCFGPass::run share one cached LoopInfo (the pass claims every
analysis preserved), the cached loop objects predate StructurizeCFG's
rewrite, isolateContinue's loop->contains() finds no predecessors and the
split silently no-ops, and isolateConvergentLatch's latch surgery can
re-create the collision afterwards. So the second half of this patch
recomputes the analyses around every helper and runs isolateContinue once
more on the settled CFG.

Both halves are required: without the analysis refresh the isolateContinue
split never happens; without the latch preference the producer's continue
scan can still pick the nested exit block on shapes where the blocks differ.

Tested against clspv f2b01dd6 plus the apply-coopmat-lowering.py,
apply-coopmat-f16.py, apply-m2v-marker.py and apply-spec-constants.py
patches (the CI order of .github/workflows/sweep.yml).

Usage: python3 apply-continue-latch.py /path/to/clspv   (idempotent)
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
prod = root / "lib/SPIRVProducerPass.cpp"
order = root / "lib/ComputeStructuredOrder.cpp"


def head_sha(repo: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""


sha = head_sha(root)
sha_short = sha[:8]
if not sha_short.startswith("f2b01dd"):
    print(
        f"NOTE: clspv HEAD is {sha_short or '(not a git tree)'} (tested f2b01dd6); anchors may need adjustment",
        file=sys.stderr,
    )

s = prod.read_text()
if "metal2vk: prefer the latch itself as the continue target" in s:
    print("producer already patched")
else:
    a = """          // From SPIR-V spec 2.11, Continue Target must dominate that back-edge
          // block.
          BasicBlock *Header = L->getHeader();
          BasicBlock *Latch = L->getLoopLatch();
          for (auto *loop_block : L->blocks()) {
            if (loop_block == Header) {
              continue;
            }

            // Check whether block dominates block with back-edge.
            // The loop latch is the single block with a back-edge. If it was
            // possible, StructurizeCFG made the loop conform to this
            // requirement, otherwise |Latch| is a nullptr.
            if (DT.dominates(loop_block, Latch)) {
              ContinueBB = loop_block;
            }
          }
"""
    assert a in s, "SPIRVProducerPass.cpp continue scan anchor not found"
    b = """          // From SPIR-V spec 2.11, Continue Target must dominate that back-edge
          // block.
          BasicBlock *Header = L->getHeader();
          BasicBlock *Latch = L->getLoopLatch();
          // metal2vk: prefer the latch itself as the continue target. The scan
          // below can pick a nested loop's exit block (it dominates the latch
          // through the single exit chain), and that block is then also the
          // nested loop's merge block; a merge block must not double as the
          // continue target of the enclosing loop (spirv-val: "Header block
          // 'X' is contained in the loop construct headed by 'Y', but its
          // merge block 'Z' is not"). The latch dominates itself, so it always
          // satisfies the spec requirement; isolateConvergentLatch has already
          // split any convergent content out of it. Fall back to the scan only
          // when the loop has no unique latch.
          if (Latch) {
            ContinueBB = Latch;
          } else {
            for (auto *loop_block : L->blocks()) {
              if (loop_block == Header) {
                continue;
              }

              // Check whether block dominates block with back-edge.
              // The loop latch is the single block with a back-edge. If it was
              // possible, StructurizeCFG made the loop conform to this
              // requirement, otherwise |Latch| is a nullptr.
              if (Latch && DT.dominates(loop_block, Latch)) {
                ContinueBB = loop_block;
              }
            }
          }
"""
    s = s.replace(a, b, 1)
    prod.write_text(s)
    print("patched SPIRVProducerPass.cpp")

s = order.read_text()
if "metal2vk: prefer the latch itself" in s:
    print("ComputeStructuredOrder already patched")
else:
    a = """      auto *header = loop->getHeader();
      auto *latch = loop->getLoopLatch();
      for (auto *bb : loop->blocks()) {
        if (bb == header)
          continue;

        // Several block might dominate the latch, we can pick any.
        if (DT->dominates(bb, latch))
          continue_block = bb;
      }
"""
    assert a in s, "ComputeStructuredOrder.cpp continue scan anchor not found"
    b = """      auto *header = loop->getHeader();
      auto *latch = loop->getLoopLatch();
      // metal2vk: prefer the latch itself (see PopulateStructuredCFGMaps in
      // SPIRVProducerPass.cpp for the reasoning); it always dominates the
      // back-edge block. Only scan when the loop has no unique latch.
      if (latch) {
        continue_block = latch;
      } else {
        for (auto *bb : loop->blocks()) {
          if (bb == header)
            continue;

          // Several block might dominate the latch, we can pick any.
          if (latch && DT->dominates(bb, latch))
            continue_block = bb;
        }
      }
"""
    assert a in s, "ComputeStructuredOrder.cpp continue scan body not found"
    s = s.replace(a, b, 1)
    order.write_text(s)
    print("patched ComputeStructuredOrder.cpp")

# The analysis-staleness half of the fix. FixupStructuredCFGPass rewrites the
# CFG with three helpers that all share one cached LoopInfo (the pass claims
# every analysis preserved), so on the current pinned LLVM the helpers run on
# pre-structurizecfg loop objects: isolateContinue collects zero predecessors
# and silently no-ops, and isolateConvergentLatch's latch split can re-create
# the exit == parent-latch collision afterwards. Refresh the cached analyses
# before every helper, run isolateContinue once more on the settled CFG, and
# drop the stale results again so the SPIR-V producer recomputes them. Without
# this the latch preference above is unreachable in the maps.
fix = root / "lib/FixupStructuredCFGPass.cpp"
s = fix.read_text()
if "metal2vk: the helpers below rewrite the CFG" in s:
    print("FixupStructuredCFGPass already patched")
else:
    a = """PreservedAnalyses
clspv::FixupStructuredCFGPass::run(Function &F, FunctionAnalysisManager &FAM) {
  // Assumes CFG has been structurized.
  isolateContinue(F, FAM);
  isolateConvergentLatch(F, FAM);
  breakConditionalHeader(F, FAM);

  removeUndefPHI(F);

  PreservedAnalyses PA;
  return PA;
}
"""
    assert a in s, "FixupStructuredCFGPass.cpp run() anchor not found"
    b = """PreservedAnalyses
clspv::FixupStructuredCFGPass::run(Function &F, FunctionAnalysisManager &FAM) {
  // metal2vk: the helpers below rewrite the CFG but do not update the cached
  // analyses, and a helper running on stale loop objects silently no-ops or,
  // worse, re-creates the shape a previous helper fixed (a nested loop whose
  // unique exit block is the parent latch - a merge block that doubles as the
  // enclosing loop's continue target, which spirv-val rejects). Recompute the
  // analyses before every analysis-driven step, and run isolateContinue once
  // more on the settled CFG. The whole-manager invalidate is used because
  // clearAnalysis erases end() and crashes when the analysis is not cached.
  FAM.invalidate(F, PreservedAnalyses::none());
  isolateContinue(F, FAM);

  FAM.invalidate(F, PreservedAnalyses::none());
  isolateConvergentLatch(F, FAM);

  FAM.invalidate(F, PreservedAnalyses::none());
  breakConditionalHeader(F, FAM);

  FAM.invalidate(F, PreservedAnalyses::none());
  isolateContinue(F, FAM);

  removeUndefPHI(F);

  // Force the reorder pass and the SPIR-V producer to recompute as well.
  FAM.invalidate(F, PreservedAnalyses::none());
  PreservedAnalyses PA;
  return PA;
}
"""
    s = s.replace(a, b, 1)
    fix.write_text(s)
    print("patched FixupStructuredCFGPass.cpp")
