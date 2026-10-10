"""Widen FixupStructuredCFGPass::isolateContinue: check every exit block of an
inner loop against the latch of EVERY enclosing loop, not just the unique exit
against the immediate parent's latch.

Root cause for the 32 remaining spirv-val loop-structure failures
(16 flashnext/lane_qmm_bytes_grouped, 4 prefill/attention_nax,
11 prefill/gemm_nax, gdn/chunked/output_and_state v0+v1): after LLVM's
StructurizeCFG, an inner loop can reach the latch (the SPIR-V continue target)
of any enclosing loop through one of several exit blocks - or the unique exit
can belong to a grandparent loop rather than the immediate parent. The
enclosing loop's latch then doubles as the inner loop's merge block, and
spirv-val rejects the module: the merge block of an inner construct is not
allowed to be classified inside the enclosing loop's continue construct
("Header block 'X' is contained in the loop construct headed by 'Y', but its
merge block 'Z' is not"; the loop construct excludes blocks dominated by the
continue target). The original isolateContinue only fired for the
unique-exit-equals-immediate-parent-latch case, which none of these are.

The split is the same one the original case performs: inner-loop edges into
the shared exit are redirected through a fresh block that falls through to it.
The fresh block becomes the inner loop's merge and stays inside the enclosing
loop construct; the old block keeps the latch role and the continue target.

apply-continue-latch.py already re-runs isolateContinue after every
FixupStructuredCFGPass helper, so splits destroyed by isolateConvergentLatch
or breakConditionalHeader are re-created here on the settled CFG.

Loops without a unique latch are not covered: the producer's continue scan
falls back to the last non-header block, which this pass cannot predict.

Usage: python3 apply-isolate-continue-exits.py /path/to/clspv   (idempotent)
Tested against clspv f2b01dd6 with the CI patch order of
.github/workflows/sweep.yml up to and including apply-continue-latch.py.
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
fix = root / "lib/FixupStructuredCFGPass.cpp"

NEW_MARK = "metal2vk: check every exit block of an inner loop against the latch"

s = fix.read_text()

if NEW_MARK in s:
    print("FixupStructuredCFGPass.cpp already patched")
    sys.exit(0)

a = """    loops.pop_back();
    // Look for cases where the merge block (unique exit) of the inner loop is
    // the same block as the outer loop's continue target (loop latch).
    if (auto parent = loop->getParentLoop()) {
      if (auto exit_block = loop->getUniqueExitBlock()) {
        if (exit_block == parent->getLoopLatch()) {
          // Create a new basic block to act as the merge of the loop.
          auto new_exit =
              BasicBlock::Create(F.getContext(), "", &F, exit_block);
          UncondBrInst::Create(exit_block, new_exit);
          parent->addBlockEntry(new_exit);

          // Collect the exit's predecessors from within the loop.
          SmallVector<BasicBlock *, 4> loop_preds;
          for (auto iter = pred_begin(exit_block); iter != pred_end(exit_block);
               ++iter) {
            if (loop->contains(*iter)) {
              loop_preds.push_back(*iter);
            }
          }
          // Split the phi nodes so that all predecessors from within the loop
          // are part of a new phi in the new exit block.
          for (auto iter = exit_block->begin();
               iter != exit_block->getFirstNonPHIIt(); ++iter) {
            PHINode *phi = cast<PHINode>(&*iter);
            SmallVector<Value *, 4> phi_values;
            for (auto pred : loop_preds) {
              auto inc = phi->getIncomingValueForBlock(&*pred);
              if (inc) {
                phi_values.push_back(inc);
              }
            }
            assert(phi_values.size() == loop_preds.size());
            if (phi_values.size() == 1) {
              // Special case: don't bother creating a single input phi.
              // Instead, add the single value as an incoming value for the new
              // exit.
              phi->addIncoming(phi_values[0], new_exit);
            } else if (!phi_values.empty()) {
              auto new_phi =
                  PHINode::Create(phi->getType(), phi_values.size(), "",
                                  new_exit->getTerminator()->getIterator());
              for (size_t i = 0; i < phi_values.size(); ++i) {
                new_phi->addIncoming(phi_values[i], loop_preds[i]);
              }
              phi->addIncoming(new_phi, new_exit);
            }
          }
          // Remove the loop predecessors from the old exit block.
          for (auto pred : loop_preds) {
            exit_block->removePredecessor(&*pred);
            pred->getTerminator()->replaceUsesOfWith(exit_block, new_exit);
          }
        }
      }
    }
  }
}
"""

b = """    loops.pop_back();
    // metal2vk: check every exit block of an inner loop against the latch of
    // every enclosing loop, not just the unique exit against the immediate
    // parent's latch. After StructurizeCFG an inner loop can reach the latch
    // (the SPIR-V continue target) of any enclosing loop through one of
    // several exit blocks, and the unique exit can belong to a loop above the
    // immediate parent. The enclosing loop's latch then doubles as the inner
    // loop's merge block, and spirv-val rejects the module ("Header block 'X'
    // is contained in the loop construct headed by 'Y', but its merge block
    // 'Z' is not": the loop construct excludes blocks dominated by the
    // continue target). Split every offending exit: the inner loop merges
    // into a fresh block that falls through to the old exit, so the merge
    // stays inside the enclosing loop construct while the old block keeps the
    // latch role and the continue target.
    SmallVector<BasicBlock *, 4> exits;
    loop->getExitBlocks(exits);
    DenseSet<BasicBlock *> latches;
    for (auto enclosing = loop->getParentLoop(); enclosing;
         enclosing = enclosing->getParentLoop()) {
      if (auto *latch = enclosing->getLoopLatch()) {
        latches.insert(latch);
      }
    }
    DenseSet<BasicBlock *> handled_exits;
    for (auto *exit_block : exits) {
      if (!latches.count(exit_block) ||
          !handled_exits.insert(exit_block).second) {
        continue;
      }
      {
        // Create a new basic block to act as the merge of the loop.
        auto new_exit =
            BasicBlock::Create(F.getContext(), "", &F, exit_block);
        UncondBrInst::Create(exit_block, new_exit);
        // The new exit belongs to the innermost enclosing loop that already
        // contains the old exit block.
        for (auto enclosing = loop->getParentLoop(); enclosing;
             enclosing = enclosing->getParentLoop()) {
          if (enclosing->contains(exit_block)) {
            enclosing->addBlockEntry(new_exit);
            break;
          }
        }

        // Collect the exit's predecessors from within the loop.
        SmallVector<BasicBlock *, 4> loop_preds;
        for (auto iter = pred_begin(exit_block); iter != pred_end(exit_block);
             ++iter) {
          if (loop->contains(*iter)) {
            loop_preds.push_back(*iter);
          }
        }
        // Split the phi nodes so that all predecessors from within the loop
        // are part of a new phi in the new exit block.
        for (auto iter = exit_block->begin();
             iter != exit_block->getFirstNonPHIIt(); ++iter) {
          PHINode *phi = cast<PHINode>(&*iter);
          SmallVector<Value *, 4> phi_values;
          for (auto pred : loop_preds) {
            auto inc = phi->getIncomingValueForBlock(&*pred);
            if (inc) {
              phi_values.push_back(inc);
            }
          }
          assert(phi_values.size() == loop_preds.size());
          if (phi_values.size() == 1) {
            // Special case: don't bother creating a single input phi.
            // Instead, add the single value as an incoming value for the new
            // exit.
            phi->addIncoming(phi_values[0], new_exit);
          } else if (!phi_values.empty()) {
            auto new_phi =
                PHINode::Create(phi->getType(), phi_values.size(), "",
                                new_exit->getTerminator()->getIterator());
            for (size_t i = 0; i < phi_values.size(); ++i) {
              new_phi->addIncoming(phi_values[i], loop_preds[i]);
            }
            phi->addIncoming(new_phi, new_exit);
          }
        }
        // Remove the loop predecessors from the old exit block.
        for (auto pred : loop_preds) {
          exit_block->removePredecessor(&*pred);
          pred->getTerminator()->replaceUsesOfWith(exit_block, new_exit);
        }
      }
    }
  }
}
"""

if a not in s:
    print("ERROR: isolateContinue anchor not found in FixupStructuredCFGPass.cpp",
          file=sys.stderr)
    sys.exit(1)

s = s.replace(a, b, 1)
fix.write_text(s)
print("patched FixupStructuredCFGPass.cpp (isolateContinue every-exit every-ancestor)")
