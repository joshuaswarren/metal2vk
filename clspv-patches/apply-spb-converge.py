"""Make SimplifyPointerBitcastPass reach a correct fixed point on the gate_up tail drift.

The drift (proven on the omlx_qwen35_moe_gate_up_decode shared-gate scalar in PR 47):
a loop-carried pointer phi with mixed-typed incoming geps (one at i8, one at i16)
was rewritten forever between sub-passes 3 (runOnUpgradeableConstantCasts) and 7
(runOnPHIFromGEP), appending one lshr or shl to the entry-index chain per
convergence round. The module never reached a fixed point, the 200-iteration
cap fired, the IR was half-rewritten, and the SPIR-V's loop-exit tail read one
full block too far (+256 on the ushort storage chain). A separate fold in
sub-pass 8 (runOnGEPFromGEP) re-associated a tail gep with the latch advance
across the loop-carried phi and committed the +256 into the producer's
pointer offset.

Three independent repairs:

  A. runOnGEPFromGEP: skip the pair fold GEP(OtherGEP, i) -> GEP(Phi, i+off)
     exactly when it is address-changing: OtherGEP's pointer operand is a
     PHINode Phi defined in H, the phi's incoming values are not all the same
     SSA value, the inner gep's block B_O differs from the outer gep's block
     B_G and from H, B_O can reach H, and H can reach B_G over a path that
     avoids B_O. On such a path the phi is re-selected (latch advance) after
     OtherGEP's last execution and the new value reaches the outer gep
     without OtherGEP re-running, so the re-associated index changes the
     address (the gate_up tail's +256 ushort access chain). Every other
     shape is exact and folds (the gemm_nax software-pipelined pair folds).
  B. runOnUpgradeableConstantCasts: in the main Worklist collection, do not
     push entries whose consumer is a PHINode. The phi's incoming shapes are
     owned by runOnPHIFromGEP; widening them here fights that canonicalization
     (each pass rewrites the same gep back to its own type, appending one
     shift or division per round, so the module grows forever and the hash
     guards never fire).
  B'. Same pass, in the GEPsDefiningPHIs walk: do not re-type values that are
     an incoming of a walked phi. (The walk already collected the non-phi
     incomings into an 'incoming_seen' set; we skip them in the values loop
     so user geps can still be widened.)
  C. runOnPHIFromGEP Worklist rewrite: pass &Ok to GetIdxsForTyFromOffset and
     skip + emit a M2V-GIVEUP marker when the offset is not representable
     through the phi's type path. Folding anyway would build a wrong gep
     (Release compiles out the upstream llvm_unreachable).

Every remaining give-up path of the pointer-bitcast pass prints a unique
'M2V-GIVEUP: <name>' line; sweep/m2v_sweep.py and compile.sh treat any such
line as a failure (no SPIR-V kept). The drivers' NONCONVERGED check is
extended in the same change.

Tested against clspv f2b01dd6 plus the 16 CI-order patches (apply-coopmat-*.
through apply-isolate-continue-exits.py). Anchor-guarded, idempotent.

Usage: python3 apply-spb-converge.py /path/to/clspv
"""
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
pp = root / "lib" / "SimplifyPointerBitcastPass.cpp"

sha = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
if not sha.startswith("f2b01dd"):
    print(f"NOTE: clspv HEAD is {sha[:8]} (tested f2b01dd6); anchors may need adjustment",
          file=sys.stderr)

s = pp.read_text()
if "spb-converge" in s:
    print("already patched")
    sys.exit(0)


def step(label, old, new):
    global s
    if new in s:
        return
    if old not in s:
        raise SystemExit(f"anchor missing: {label}")
    s = s.replace(old, new, 1)


# --- Fix A: cross-block phi-base pair fold in runOnGEPFromGEP ---
step(
    "runOnGEPFromGEP phi cross-block guard",
    "          // ... whose operand is also a GEP instruction...\n"
    "          if (auto OtherGEP =\n"
    "                  dyn_cast<GetElementPtrInst>(GEP->getPointerOperand())) {\n"
    "            // ... with no implicit cast between them...\n"
    "            if (OtherGEP->getResultElementType() ==\n"
    "                GEP->getSourceElementType()) {\n",
    "          // ... whose operand is also a GEP instruction...\n"
    "          if (auto OtherGEP =\n"
    "                  dyn_cast<GetElementPtrInst>(GEP->getPointerOperand())) {\n"
    "            // ... with no implicit cast between them...\n"
    "            if (OtherGEP->getResultElementType() ==\n"
    "                GEP->getSourceElementType()) {\n"
    "              // metal2vk (apply-spb-converge): a loop-carried phi reads\n"
    "              // a different incoming value at the fold's block than at the\n"
    "              // inner gep, so re-associating the inner gep's offset onto\n"
    "              // its base across blocks changes the address (the gate_up\n"
    "              // tail was advanced by one extra block stride). Leave the\n"
    "              // pair split; split geps are valid IR.\n"
    "              if (isa<PHINode>(OtherGEP->getPointerOperand()) &&\n"
    "                  OtherGEP->getParent() != GEP->getParent()) {\n"
    "                continue;\n"
    "              }\n",
)

# --- Fix B + B': runOnUpgradeableConstantCasts ownership ---
step(
    "upgradeable casts main-worklist phi guard",
    "          dest_ty = findBiggerTyToUpdate(dest_ty);\n"
    "          if (dest_ty != source_ty) {\n\n"
    "            Worklist.push_back(\n"
    "                {&I, cstVal, dynVal, smallerBitWidths, dest_ty, gep});\n"
    "          }\n",
    "          dest_ty = findBiggerTyToUpdate(dest_ty);\n"
    "          // metal2vk (apply-spb-converge): the phi's incoming shapes are\n"
    "          // owned by runOnPHIFromGEP (it canonicalizes every incoming to\n"
    "          // the phi's use-inferred type once and stops). Widening an\n"
    "          // incoming here fights that canonicalization, appending one\n"
    "          // shift or division per convergence round - the module grows\n"
    "          // forever and the hash guards never fire.\n"
    "          if (isa<PHINode>(&I)) {\n"
    "            continue;\n"
    "          }\n"
    "          if (dest_ty != source_ty) {\n\n"
    "            Worklist.push_back(\n"
    "                {&I, cstVal, dynVal, smallerBitWidths, dest_ty, gep});\n"
    "          }\n",
)

step(
    "upgradeable casts GEPsDefiningPHIs incoming_seen",
    "            SmallVector<UpgradeInfo> geps;\n"
    "            SmallVector<Value *> values;\n"
    "            SmallVector<PHINode *> phis;\n"
    "            DenseSet<PHINode *> phis_seen;\n"
    "            phis.push_back(phi);\n",
    "            SmallVector<UpgradeInfo> geps;\n"
    "            SmallVector<Value *> values;\n"
    "            SmallVector<PHINode *> phis;\n"
    "            DenseSet<PHINode *> phis_seen;\n"
    "            DenseSet<Value *> incoming_seen;\n"
    "            phis.push_back(phi);\n",
)
step(
    "upgradeable casts walk collect incomings into set",
    "              for (auto &incoming_value : current_phi->incoming_values()) {\n"
    "                auto val = incoming_value.get();\n"
    "                if (auto phi_node = dyn_cast<PHINode>(val)) {\n"
    "                  phis.push_back(phi_node);\n"
    "                } else {\n"
    "                  values.push_back(val);\n"
    "                }\n"
    "              }\n",
    "              for (auto &incoming_value : current_phi->incoming_values()) {\n"
    "                auto val = incoming_value.get();\n"
    "                if (auto phi_node = dyn_cast<PHINode>(val)) {\n"
    "                  phis.push_back(phi_node);\n"
    "                } else {\n"
    "                  // metal2vk (apply-spb-converge): see main Worklist guard.\n"
    "                  incoming_seen.insert(val);\n"
    "                }\n"
    "              }\n",
)
step(
    "upgradeable casts filter incomings from values loop",
    "            for (auto value : values) {\n"
    "              auto user_ty = clspv::InferType(value, context, &type_cache);\n",
    "            for (auto value : values) {\n"
    "              if (incoming_seen.count(value) != 0) {\n"
    "                continue;\n"
    "              }\n"
    "              auto user_ty = clspv::InferType(value, context, &type_cache);\n",
)

# --- Fix B'': runOnUpgradeableConstantCasts push-side phi-incoming guard ---
# The rewrite loop's replaceAllUsesWith rewrites a phi's incoming through the
# back door when the old gep's only user is a pointer phi (I is a load or
# store, so the phi-source guard never sees it); blocking the push is
# per-gep and independent of worklist order.
step(
    "upgradeable casts push-side phi-incoming guard",
    "          if (isa<PHINode>(&I)) {\n"
    "            continue;\n"
    "          }\n"
    "          if (dest_ty != source_ty) {\n\n"
    "            Worklist.push_back(\n"
    "                {&I, cstVal, dynVal, smallerBitWidths, dest_ty, gep});\n"
    "          }\n",
    "          if (isa<PHINode>(&I)) {\n"
    "            continue;\n"
    "          }\n"
    "          // metal2vk (apply-spb-converge): the same rule for the rewrite\n"
    "          // side. When the source gep IS a phi's incoming, the rewrite\n"
    "          // loop's replaceAllUsesWith rewrites the phi's incoming operand\n"
    "          // through the back door even though I is a load or store, and\n"
    "          // each pass clones the gep instead of converging. Refuse the\n"
    "          // push here, where the check is per gep and independent of\n"
    "          // worklist order.\n"
    "          bool M2vPhiIncomingSrc = false;\n"
    "          for (auto *U : gep->users()) {\n"
    "            if (isa<PHINode>(U)) {\n"
    "              M2vPhiIncomingSrc = true;\n"
    "              break;\n"
    "            }\n"
    "          }\n"
    "          if (M2vPhiIncomingSrc) {\n"
    "            continue;\n"
    "          }\n"
    "          if (dest_ty != source_ty) {\n\n"
    "            Worklist.push_back(\n"
    "                {&I, cstVal, dynVal, smallerBitWidths, dest_ty, gep});\n"
    "          }\n",
)

# --- Fix B''': runOnImplicitCasts push-side phi-incoming guard ---
# Same back door through the implicit-casts fold: replacing inst_gep
# rewrites the phi's incoming when inst_gep feeds a pointer phi.
step(
    "implicit casts push-side phi-incoming guard",
    "            if (!(VecSrcTy && VecDstTy &&\n"
    "                  (VecSrcTy->getNumElements() == 3 ||\n"
    "                   VecDstTy->getNumElements() == 3))) {\n"
    "              Worklist.emplace_back(inst_gep);\n"
    "            }\n",
    "            if (!(VecSrcTy && VecDstTy &&\n"
    "                  (VecSrcTy->getNumElements() == 3 ||\n"
    "                   VecDstTy->getNumElements() == 3))) {\n"
    "              // metal2vk (apply-spb-converge): a gep that is a pointer\n"
    "              // phi's incoming is owned by runOnPHIFromGEP. Folding it\n"
    "              // here re-types the phi's incoming through the back door\n"
    "              // (the fold replaces inst_gep everywhere, including the\n"
    "              // phi's operand), and the c[5]/c[7] flip on that incoming\n"
    "              // starts again. Refuse the push.\n"
    "              bool M2vPhiIncoming = false;\n"
    "              for (auto *U : inst_gep->users()) {\n"
    "                if (isa<PHINode>(U)) {\n"
    "                  M2vPhiIncoming = true;\n"
    "                  break;\n"
    "                }\n"
    "              }\n"
    "              if (!M2vPhiIncoming) {\n"
    "                Worklist.emplace_back(inst_gep);\n"
    "              }\n"
    "            }\n",
)

# --- Fix C: runOnPHIFromGEP Worklist representability ---
step(
    "phi-from-gep Ok flag and giveup",
    "    auto Idxs =\n"
    "        GetIdxsForTyFromOffset(DL, Builder, Ty, nullptr, CstVal, DynVal,\n"
    "                               SmallerBitWidths, gep->getPointerOperand());\n"
    "    auto new_gep = GetElementPtrInst::Create(Ty, gep->getPointerOperand(), Idxs,\n"
    "                                             \"\", gep->getIterator());\n",
    "    // metal2vk (apply-spb-converge): the offset may not be representable\n"
    "    // through the phi's type path; folding it anyway would build a wrong\n"
    "    // GEP (Release compiles out the upstream llvm_unreachable). Skip and\n"
    "    // mark the give-up so the driver keeps no SPIR-V.\n"
    "    bool M2vFoldOk = true;\n"
    "    auto Idxs =\n"
    "        GetIdxsForTyFromOffset(DL, Builder, Ty, nullptr, CstVal, DynVal,\n"
    "                               SmallerBitWidths, gep->getPointerOperand(),\n"
    "                               &M2vFoldOk);\n"
    "    if (!M2vFoldOk) {\n"
    "      errs() << \"M2V-GIVEUP: phi-from-gep-skip\\n\";\n"
    "      continue;\n"
    "    }\n"
    "    auto new_gep = GetElementPtrInst::Create(Ty, gep->getPointerOperand(), Idxs,\n"
    "                                             \"\", gep->getIterator());\n",
)

# --- Fix A refined: the cross-block guard becomes a per-edge reachability
# predicate; the coarse different-block guard also blocked exact folds (the
# seven gemm_nax prefill entries lost valid SPIR-V to it).
step(
    "CFG include for the reachability helper",
    "#include \"llvm/IR/Module.h\"\n",
    "#include \"llvm/IR/Module.h\"\n"
    "#include \"llvm/IR/CFG.h\"\n",
)
step(
    "reachability helper before runOnGEPFromGEP",
    "bool clspv::SimplifyPointerBitcastPass::runOnGEPFromGEP(Module &M) const {",
    "// metal2vk (apply-spb-converge): CFG reachability from Src to Dst; when\n"
    "// Avoid is non-null, paths through Avoid do not count.\n"
    "static bool M2vReach(BasicBlock *Src, BasicBlock *Dst, BasicBlock *Avoid) {\n"
    "  if (Src == Dst) {\n"
    "    return true;\n"
    "  }\n"
    "  SmallPtrSet<BasicBlock *, 32> Visited;\n"
    "  SmallVector<BasicBlock *, 32> Work;\n"
    "  Visited.insert(Src);\n"
    "  Work.push_back(Src);\n"
    "  while (!Work.empty()) {\n"
    "    BasicBlock *BB = Work.pop_back_val();\n"
    "    for (BasicBlock *Succ : successors(BB)) {\n"
    "      if (Succ == Dst) {\n"
    "        return true;\n"
    "      }\n"
    "      if (Avoid && Succ == Avoid) {\n"
    "        continue;\n"
    "      }\n"
    "      if (Visited.insert(Succ).second) {\n"
    "        Work.push_back(Succ);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  return false;\n"
    "}\n"
    "\n"
    "bool clspv::SimplifyPointerBitcastPass::runOnGEPFromGEP(Module &M) const {",
)
step(
    "Fix A refined to the per-edge predicate",
    "              // metal2vk (apply-spb-converge): a loop-carried phi reads\n"
    "              // a different incoming value at the fold's block than at the\n"
    "              // inner gep, so re-associating the inner gep's offset onto\n"
    "              // its base across blocks changes the address (the gate_up\n"
    "              // tail was advanced by one extra block stride). Leave the\n"
    "              // pair split; split geps are valid IR.\n"
    "              if (isa<PHINode>(OtherGEP->getPointerOperand()) &&\n"
    "                  OtherGEP->getParent() != GEP->getParent()) {\n"
    "                continue;\n"
    "              }\n",
    "              // metal2vk (apply-spb-converge): per-edge safety predicate\n"
    "              // for re-associating GEP(OtherGEP, i) into GEP(Phi, i+off)\n"
    "              // when OtherGEP's pointer operand is the PHINode Phi\n"
    "              // (defined in H). The fold re-reads Phi at the outer GEP's\n"
    "              // block (B_G) instead of at the inner GEP's block (B_O),\n"
    "              // which preserves the address iff Phi's value cannot change\n"
    "              // between OtherGEP's last execution before the outer GEP\n"
    "              // and the outer GEP itself. A pointer phi changes its\n"
    "              // register value exactly when control enters H (an incoming\n"
    "              // selection), and re-entering B_O re-executes OtherGEP,\n"
    "              // which re-syncs the pair with the phi's new value. So the\n"
    "              // fold is unsafe iff the phi's incoming values are not all\n"
    "              // the same SSA value, the two geps live in different\n"
    "              // blocks, B_O is not H itself, B_O can reach H (an incoming\n"
    "              // edge is crossed after OtherGEP's last execution), and H\n"
    "              // can reach B_G over a path that avoids B_O (the phi's new\n"
    "              // value reaches the outer GEP without OtherGEP re-running;\n"
    "              // the gate_up tail shape: body -> latch -> header -> exit\n"
    "              // tail). Otherwise the pair fold is exact and must not be\n"
    "              // blocked.\n"
    "              if (auto M2vPhi =\n"
    "                      dyn_cast<PHINode>(OtherGEP->getPointerOperand())) {\n"
    "                BasicBlock *M2vBO = OtherGEP->getParent();\n"
    "                BasicBlock *M2vBG = GEP->getParent();\n"
    "                BasicBlock *M2vH = M2vPhi->getParent();\n"
    "                bool M2vAllSame = true;\n"
    "                Value *M2vFirst = M2vPhi->getIncomingValue(0);\n"
    "                for (unsigned M2vK = 1;\n"
    "                     M2vK < M2vPhi->getNumIncomingValues(); ++M2vK) {\n"
    "                  if (M2vPhi->getIncomingValue(M2vK) != M2vFirst) {\n"
    "                    M2vAllSame = false;\n"
    "                    break;\n"
    "                  }\n"
    "                }\n"
    "                if (!M2vAllSame && M2vBO != M2vBG && M2vBO != M2vH &&\n"
    "                    M2vReach(M2vBO, M2vH, nullptr) &&\n"
    "                    M2vReach(M2vH, M2vBG, M2vBO)) {\n"
    "                  continue;\n"
    "                }\n"
    "              }\n",
)

# --- Fix B''' refined: refuse the push only when runOnPHIFromGEP actually
# owns the incoming, i.e. the phi user is itself an implicit-cast site. A phi
# that is not an implicit-cast site never rewrites its incomings, so the fold
# cannot start the c[5]/c[7] flip; the gemm_nax refusals were all such phis
# and the folds were needed for a valid OpPtrAccessChain.
step(
    "implicit casts push-side guard refined to phi-cast-site check",
    "              bool M2vPhiIncoming = false;\n"
    "              for (auto *U : inst_gep->users()) {\n"
    "                if (isa<PHINode>(U)) {\n"
    "                  M2vPhiIncoming = true;\n"
    "                  break;\n"
    "                }\n"
    "              }\n"
    "              if (!M2vPhiIncoming) {\n"
    "                Worklist.emplace_back(inst_gep);\n"
    "              }\n",
    "              bool M2vPhiIncoming = false;\n"
    "              for (auto *U : inst_gep->users()) {\n"
    "                if (auto *M2vPU = dyn_cast<PHINode>(U)) {\n"
    "                  // metal2vk (apply-spb-converge): refuse only when\n"
    "                  // runOnPHIFromGEP owns this incoming, i.e. the phi\n"
    "                  // itself is an implicit-cast site (the only way c[7]\n"
    "                  // rewrites the incoming and can fight this fold). A phi\n"
    "                  // that is not an implicit-cast site never rewrites its\n"
    "                  // incomings, so folding here cannot start the c[5]/c[7]\n"
    "                  // flip.\n"
    "                  Value *M2vS = nullptr;\n"
    "                  Type *M2vST = nullptr;\n"
    "                  Type *M2vDT = nullptr;\n"
    "                  if (IsImplicitCasts(M, type_cache, *M2vPU, M2vS, M2vST,\n"
    "                                      M2vDT, true)) {\n"
    "                    M2vPhiIncoming = true;\n"
    "                    break;\n"
    "                  }\n"
    "                }\n"
    "              }\n"
    "              if (!M2vPhiIncoming) {\n"
    "                Worklist.emplace_back(inst_gep);\n"
    "              }\n",
)

# --- docstring: Fix A's description follows the refined predicate (the
# module docstring is this script's own text, not clspv source; edited in
# place above this comment) ---

pp.write_text(s)
print(f"patched (clspv HEAD {sha[:8]})")
