#include "proteus/JitFrontend.hpp"

namespace proteus {

Func::Func(FunctionCallee FC) : FC(FC), IRB{FC.getCallee()->getContext()} {
  Function *F = cast<Function>(FC.getCallee());
  BasicBlock *EntryBB = BasicBlock::Create(F->getContext(), "entry", F);
}

IRBuilderBase &Func::getIRB() {
  if (!IRB.GetInsertBlock())
    PROTEUS_FATAL_ERROR("Insert point is not set");
  return IRB;
}

Var &Func::declVarInternal(StringRef Name, Type *Ty, Type *PointerElemType) {
  auto *Alloca = emitAlloca(Ty, Name);
  return Variables.emplace_back(Alloca, *this, PointerElemType);
}

void Func::beginFunction() {
  Function *F = cast<Function>(FC.getCallee());
  BasicBlock *BodyBB = BasicBlock::Create(F->getContext(), "body", F);
  IP =
      IRBuilderBase::InsertPoint(&F->getEntryBlock(), F->getEntryBlock().end());
  IRB.restoreIP(IP);
  IRB.CreateBr(BodyBB);

  IP = IRBuilderBase::InsertPoint(BodyBB, BodyBB->end());
  IRB.restoreIP(IP);
}

void Func::endFunction() {}

Function *Func::getFunction() {
  Function *F = dyn_cast<Function>(FC.getCallee());
  if (!F)
    PROTEUS_FATAL_ERROR("Expected LLVM Function");
  return F;
}

Var &Func::getArg(unsigned int ArgNo) { return Arguments.at(ArgNo); }

AllocaInst *Func::emitAlloca(Type *Ty, StringRef Name) {
  auto SaveIP = IRB.saveIP();
  Function *F = getFunction();
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  IRB.restoreIP(AllocaIP);
  auto *Alloca = IRB.CreateAlloca(Ty, nullptr, Name);

  IRB.restoreIP(SaveIP);
  return Alloca;
}

void Func::ret(std::optional<std::reference_wrapper<Var>> OptRet) {
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
  }
  IRB.SetInsertPoint(CurBlock);
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
  IRB.restoreIP(IP);
}

void Func::endIf() {
  IP = BlockIPs.back();
  BlockIPs.pop_back();
  IRB.restoreIP(IP);
}

void Func::beginLoop(Var &IterVar, Var &Init, Var &UpperBound, Var &Inc) {
  Function *F = getFunction();
  BasicBlock *Header = BasicBlock::Create(F->getContext(), "loop.header", F);
  BasicBlock *Body = BasicBlock::Create(F->getContext(), "loop.body", F);
  BasicBlock *Latch = BasicBlock::Create(F->getContext(), "loop.latch", F);
  BasicBlock *LoopExit = BasicBlock::Create(F->getContext(), "loop.cont", F);

  // Update the terminator of the current basic block due to the split
  // control-flow.
  BasicBlock *CurBlock = IP.getBlock();
  if (auto *TermI = CurBlock->getTerminator()) {
    TermI->eraseFromParent();
  }
  IRB.SetInsertPoint(CurBlock);
  IRB.CreateBr(Header);

  IRB.SetInsertPoint(Header);
  {
    IterVar = Init;
    auto &CondVar = IterVar < UpperBound;
    Value *Cond =
        IRB.CreateLoad(CondVar.Alloca->getAllocatedType(), CondVar.Alloca);
    IRB.CreateCondBr(Cond, Body, LoopExit);
  }

  IRB.SetInsertPoint(Body);
  IRB.CreateBr(Latch);

  IRB.SetInsertPoint(Latch);
  {
    IterVar = IterVar + Inc;
    auto &CondVar = IterVar < UpperBound;
    Value *Cond =
        IRB.CreateLoad(CondVar.Alloca->getAllocatedType(), CondVar.Alloca);
    IRB.CreateCondBr(Cond, Body, LoopExit);
  }

  if (!BlockIPs.empty()) {
    IRB.SetInsertPoint(LoopExit);
    IRB.CreateBr(BlockIPs.back().getBlock());
  }
  auto ContIP = IRBuilderBase::InsertPoint(LoopExit, LoopExit->begin());
  BlockIPs.push_back(ContIP);

  IP = IRBuilderBase::InsertPoint(Body, Body->begin());
  IRB.restoreIP(IP);
}

void Func::endLoop() {
  IP = BlockIPs.back();
  BlockIPs.pop_back();
  IRB.restoreIP(IP);
}

} // namespace proteus
