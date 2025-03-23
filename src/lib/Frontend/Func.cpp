#include "proteus/JitFrontend.hpp"

namespace proteus {

Func::Func(FunctionCallee FC) : FC(FC), IRB{FC.getCallee()->getContext()} {
  Function *F = cast<Function>(FC.getCallee());
  IP = IRBuilderBase::InsertPoint(&F->back(), F->back().end());
  IRB.restoreIP(IP);
}

Function *Func::getFunction() {
  Function *F = dyn_cast<Function>(FC.getCallee());
  if (!F)
    PROTEUS_FATAL_ERROR("Expected LLVM Function");
  return F;
}

Var &Func::arg(unsigned int ArgNo) { return Arguments.at(ArgNo); }

AllocaInst *Func::emitAlloca(Type *Ty, StringRef Name) {
  auto SaveIP = IRB.saveIP();
  Function *F = getFunction();
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  IRB.restoreIP(AllocaIP);
  auto *Alloca = IRB.CreateAlloca(Ty, nullptr, Name);

  IRB.restoreIP(IP);
  return Alloca;
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
  dbgs() << "Set IP to BB " << IP.getBlock()->getName() << "\n";
  getchar();
  BlockIPs.pop_back();
}

} // namespace proteus