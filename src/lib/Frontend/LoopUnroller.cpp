#include "proteus/Frontend/LoopUnroller.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>

namespace proteus {

void LoopUnroller::enable() { Enabled = true; }

void LoopUnroller::enable(int C) {
  Enabled = true;
  Count = C;
}

bool LoopUnroller::isEnabled() const { return Enabled; }

void LoopUnroller::attachMetadata(llvm::BasicBlock *LatchBB) const {
  if (!Enabled)
    return;
  using namespace llvm;
  auto *BackEdgeBr = dyn_cast<BranchInst>(LatchBB->getTerminator());

  LLVMContext &Ctx = BackEdgeBr->getContext();

  // From LanguageRef: "For legacy reasons, the first item
  // of a loop metadata node must be a reference to itself."
  SmallVector<Metadata *, 4> LoopMDOperands;
  LoopMDOperands.push_back(nullptr);

  MDNode *UnrollEnableMD =
      MDNode::get(Ctx, MDString::get(Ctx, "llvm.loop.unroll.enable"));
  LoopMDOperands.push_back(UnrollEnableMD);

  if (Count.has_value()) {
    Metadata *CountOperands[] = {MDString::get(Ctx, "llvm.loop.unroll.count"),
                                 ConstantAsMetadata::get(ConstantInt::get(
                                     Type::getInt32Ty(Ctx), *Count))};
    MDNode *UnrollCountMD = MDNode::get(Ctx, CountOperands);
    LoopMDOperands.push_back(UnrollCountMD);
  }

  MDNode *LoopMD = MDNode::getDistinct(Ctx, LoopMDOperands);
  LoopMD->replaceOperandWith(0, LoopMD);
  BackEdgeBr->setMetadata(LLVMContext::MD_loop, LoopMD);
}

} // namespace proteus
