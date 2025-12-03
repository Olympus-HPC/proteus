#include "proteus/Frontend/LoopUnroller.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>

#include "proteus/Error.h"

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

  auto *BackEdgeBr =
      llvm::dyn_cast<llvm::BranchInst>(LatchBB->getTerminator());
  if (!BackEdgeBr)
    PROTEUS_FATAL_ERROR("Expected back-edge branch in latch block");

  using namespace llvm;
  LLVMContext &Ctx = BackEdgeBr->getContext();

  SmallVector<Metadata *, 4> LoopMDOperands;
  LoopMDOperands.push_back(nullptr); // Self-reference placeholder.

  MDNode *UnrollEnableMD =
      MDNode::get(Ctx, MDString::get(Ctx, "llvm.loop.unroll.enable"));
  LoopMDOperands.push_back(UnrollEnableMD);

  if (Count.has_value()) {
    Metadata *CountOperands[] = {
        MDString::get(Ctx, "llvm.loop.unroll.count"),
        ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(Ctx), *Count))};
    MDNode *UnrollCountMD = MDNode::get(Ctx, CountOperands);
    LoopMDOperands.push_back(UnrollCountMD);
  }

  MDNode *LoopMD = MDNode::getDistinct(Ctx, LoopMDOperands);
  LoopMD->replaceOperandWith(0, LoopMD); // Loop metadata self-reference.
  BackEdgeBr->setMetadata(LLVMContext::MD_loop, LoopMD);
}

} // namespace proteus
