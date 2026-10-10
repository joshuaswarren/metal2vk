"""Producer-level guarantee that a nested loop's merge block never doubles as
the continue target of an enclosing loop.

spirv-val rejects a module with "Header block 'X' is contained in the loop
construct headed by 'Y', but its merge block 'Z' is not" when a nested loop's
merge block is the enclosing loop's continue target: the loop construct of Y
excludes merge blocks of nested constructs, so Z is not in it. The structured
IR passes before the producer split the known shapes of this collision
(FixupStructuredCFGPass::isolateContinue), but the producer is the last
defense: whatever the earlier passes left behind is what gets emitted. This
script teaches SPIRVProducerPassImpl::PopulateStructuredCFGMaps to repair any
remaining collision on the cloned module before the merge and continue maps
are assigned: the inner loop's exit is split into a fresh fallthrough block
(the distinct merge) ahead of the old block, which keeps its latch or
continue role. No instruction moves, so a threadgroup barrier in the old
block stays inside its loop and nothing crosses it.

The continue candidates mirror the choices made by the map assignment: for
each enclosing loop its unique latch, or, when the loop has no unique latch,
the last non-header block that the fallback scan picks. A fresh fallthrough
block can never become an exit again, so the repair scan terminates.
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
prod = root / "lib/SPIRVProducerPass.cpp"


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

NEW_MARK = "RepairNestedMergeContinueCollisions"

s = prod.read_text()
if NEW_MARK in s:
    print("producer already patched")
else:
    a = """  // Populate the merge and continue block maps.
  void PopulateStructuredCFGMaps();
"""
    b = """  // Populate the merge and continue block maps.
  void PopulateStructuredCFGMaps();

  // metal2vk: split any nested loop exit that would double as the continue
  // target of an enclosing loop, so the map assignment below always emits a
  // distinct merge block for such a loop.
  void RepairNestedMergeContinueCollisions();
"""
    assert a in s, "SPIRVProducerPass.cpp declaration anchor not found"
    s = s.replace(a, b, 1)

    a = """void SPIRVProducerPassImpl::PopulateStructuredCFGMaps() {
  // First, track loop merges and continues.
"""
    b = """// metal2vk: a nested loop whose merge block is the continue target of an
// enclosing loop is rejected by the validator ("Header block 'X' is
// contained in the loop construct headed by 'Y', but its merge block 'Z' is
// not": the loop construct of Y excludes the merge blocks of nested
// constructs). The structured passes before the producer split the known
// shapes of this collision, but this pass is the last defense: whatever the
// earlier passes left behind is what gets emitted. Split every remaining
// collision on the cloned module before the maps below are assigned: the
// inner loop merges into a fresh fallthrough block ahead of the old block,
// which keeps its latch or continue role. No instruction moves, so a
// threadgroup barrier in the old block stays inside its loop and nothing
// crosses it. The continue candidates mirror the choices made by the map
// assignment: for each enclosing loop its unique latch, or, when it has no
// unique latch, the last non-header block the fallback scan picks. A fresh
// fallthrough block can never become a loop exit again, so the rescan
// terminates.
void SPIRVProducerPassImpl::RepairNestedMergeContinueCollisions() {
  auto &FAM = MAM->getResult<FunctionAnalysisManagerModuleProxy>(*module)
                  .getManager();
  for (Function &F : *module) {
    if (F.isDeclaration()) {
      continue;
    }

    // A split invalidates the cached loop analyses, so rescan from scratch
    // and never walk stale loop objects. The guard only caps the scan.
    for (unsigned guard = 0; guard < 128; ++guard) {
      auto &LI = FAM.getResult<LoopAnalysis>(F);
      Loop *offender_loop = nullptr;
      BasicBlock *offender_exit = nullptr;

      SmallVector<Loop *, 16> loops;
      for (Loop *L : LI.getLoopsInPreorder()) {
        loops.push_back(L);
      }
      // Innermost first: splitting an inner exit cannot re-collide it.
      for (auto iter = loops.rbegin();
           iter != loops.rend() && !offender_exit; ++iter) {
        Loop *L = *iter;

        DenseSet<BasicBlock *> continue_candidates;
        for (Loop *A = L->getParentLoop(); A; A = A->getParentLoop()) {
          if (BasicBlock *Latch = A->getLoopLatch()) {
            continue_candidates.insert(Latch);
            continue;
          }
          BasicBlock *Header = A->getHeader();
          BasicBlock *fallback = nullptr;
          for (BasicBlock *BB : A->blocks()) {
            if (BB != Header) {
              fallback = BB;
            }
          }
          if (fallback) {
            continue_candidates.insert(fallback);
          }
        }
        if (continue_candidates.empty()) {
          continue;
        }

        SmallVector<BasicBlock *, 4> exits;
        L->getExitBlocks(exits);
        for (BasicBlock *exit : exits) {
          if (continue_candidates.count(exit)) {
            offender_loop = L;
            offender_exit = exit;
            break;
          }
        }
      }

      if (!offender_exit) {
        break;
      }

      {
        // Same recipe as FixupStructuredCFGPass::isolateContinue: the loop
        // predecessors move to the fresh block and the phis of the old exit
        // gain one incoming edge for it, so every value reaches the old
        // block exactly as before.
        auto *new_exit =
            BasicBlock::Create(F.getContext(), "", &F, offender_exit);
        UncondBrInst::Create(offender_exit, new_exit);

        SmallVector<BasicBlock *, 4> loop_preds;
        for (auto iter = pred_begin(offender_exit);
             iter != pred_end(offender_exit); ++iter) {
          if (offender_loop->contains(*iter)) {
            loop_preds.push_back(*iter);
          }
        }
        for (auto iter = offender_exit->begin();
             iter != offender_exit->getFirstNonPHIIt(); ++iter) {
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
        for (auto pred : loop_preds) {
          offender_exit->removePredecessor(&*pred);
          pred->getTerminator()->replaceUsesOfWith(offender_exit, new_exit);
        }
      }

      FAM.invalidate(F, PreservedAnalyses::none());
    }
  }
}

// metal2vk: the producer-level repair landed by apply-structured-merge-continue.py.
// Baked as a string so the linked binary can be identified from the outside.
static const char *const kMetal2vkStructuredMergeContinueMarker =
    "metal2vk:apply-structured-merge-continue:v1";
// A volatile sink with external linkage keeps the string in the binary even in
// release builds, so the linked producer can be identified from the outside.
const char *const volatile kMetal2vkStructuredMergeContinueMarkerSink =
    kMetal2vkStructuredMergeContinueMarker;

void SPIRVProducerPassImpl::PopulateStructuredCFGMaps() {
  // The volatile read is observable behavior the compiler cannot fold away,
  // so the marker string stays in the linked binary even in release builds.
  const char *const marker = kMetal2vkStructuredMergeContinueMarkerSink;
  (void)marker;
  // metal2vk: enforce the merge/continue separation before the maps are
  // assigned (see RepairNestedMergeContinueCollisions above).
  RepairNestedMergeContinueCollisions();

  // First, track loop merges and continues.
"""
    assert a in s, "SPIRVProducerPass.cpp definition anchor not found"
    s = s.replace(a, b, 1)
    prod.write_text(s)
    print("patched SPIRVProducerPass.cpp")
