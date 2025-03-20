#include "proteus/JitFrontend.hpp"

namespace proteus {

Func::Func(FunctionCallee FC) : FC(FC), IRB{FC.getCallee()->getContext()} {
  Function *F = cast<Function>(FC.getCallee());
  IP = IRBuilderBase::InsertPoint(&F->back(), F->back().end());
}

Function *Func::getFunction() {
  Function *F = dyn_cast<Function>(FC.getCallee());
  if (!F)
    PROTEUS_FATAL_ERROR("Expected LLVM Function");
  return F;
}

Var &Func::arg(unsigned int ArgNo) {
  Function *F = getFunction();
  auto *Arg = F->getArg(ArgNo);
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  IRB.restoreIP(AllocaIP);
  auto *Alloca =
      IRB.CreateAlloca(Arg->getType(), nullptr, "arg." + std::to_string(ArgNo));
  IRB.CreateStore(Arg, Alloca);

  Variables.emplace_back(Alloca, *this);
  return Variables.back();
}

void Func::ret(std::optional<std::reference_wrapper<Var>> OptRet) {
  IRB.restoreIP(IP);
  if (OptRet == std::nullopt) {
    IRB.CreateRetVoid();
    return;
  }

  auto *RetAlloca = OptRet->get().Alloca;
  auto *Ret = IRB.CreateLoad(RetAlloca->getAllocatedType(), RetAlloca);
  IRB.CreateRet(Ret);
}

void Func::beginIf(Var &CondVar) {
  Function *F = getFunction();
  BasicBlock *ThenBlock = BasicBlock::Create(F->getContext(), "if.then", F);
  BasicBlock *ContBlock = BasicBlock::Create(F->getContext(), "cont", F);

  // Update the terminator of the current basic block due to the split
  // control-flow.
  BasicBlock *CurBlock = IP.getBlock();
  if (auto *TermI = CurBlock->getTerminator()) {
    TermI->eraseFromParent();
    IP = IRBuilderBase::InsertPoint(CurBlock, CurBlock->end());
  }
  IRB.restoreIP(IP);
  Value *Cond =
      IRB.CreateLoad(CondVar.Alloca->getAllocatedType(), CondVar.Alloca);
  IRB.CreateCondBr(Cond, ThenBlock, ContBlock);

  IRB.SetInsertPoint(ThenBlock);
  IRB.CreateBr(ContBlock);

  if (!BlockIPs.empty()) {
    IRB.SetInsertPoint(ContBlock);
    IRB.CreateBr(BlockIPs.back().getBlock());
  }
  auto ContIP = IRBuilderBase::InsertPoint(ContBlock, ContBlock->begin());
  BlockIPs.push_back(ContIP);

  IP = IRBuilderBase::InsertPoint(ThenBlock, ThenBlock->begin());

  dbgs() << "Function " << *F << "\n";
  getchar();
}

void Func::endIf() {
  IP = BlockIPs.back();
  BlockIPs.pop_back();
}

} // namespace proteus