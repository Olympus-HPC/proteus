#include "proteus/JitFrontend.hpp"

namespace proteus {

FuncBase::FuncBase(JitModule &J, FunctionCallee FC)
    : J(J), FC(FC), IRB{FC.getCallee()->getContext()} {
  Function *F = cast<Function>(FC.getCallee());
  BasicBlock::Create(F->getContext(), "entry", F);
  Name = F->getName();
}

IRBuilderBase &FuncBase::getIRBuilder() {
  if (!IRB.GetInsertBlock())
    PROTEUS_FATAL_ERROR("Insert point is not set");
  return IRB;
}

Var &FuncBase::declVarInternal(StringRef Name, Type *Ty,
                               Type *PointerElemType) {
  auto *Alloca = emitAlloca(Ty, Name);
  return Variables.emplace_back(Alloca, *this, PointerElemType);
}

void FuncBase::beginFunction(const char *File, int Line) {
  Function *F = cast<Function>(FC.getCallee());
  BasicBlock *BodyBB = BasicBlock::Create(F->getContext(), "body", F);
  BasicBlock *ExitBB = BasicBlock::Create(F->getContext(), "exit", F);
  IP =
      IRBuilderBase::InsertPoint(&F->getEntryBlock(), F->getEntryBlock().end());
  IRB.restoreIP(IP);
  IRB.CreateBr(BodyBB);

  IP = IRBuilderBase::InsertPoint(BodyBB, BodyBB->end());
  IRB.restoreIP(IP);
  IRB.CreateBr(ExitBB);

  IRB.SetInsertPoint(ExitBB);
  { IRB.CreateUnreachable(); }

  IP = IRBuilderBase::InsertPoint(BodyBB, BodyBB->begin());
  IRB.restoreIP(IP);

  Scopes.emplace_back(File, Line, ScopeKind::FUNCTION,
                      IRBuilderBase::InsertPoint(ExitBB, ExitBB->begin()));
}

void FuncBase::endFunction() {
  if (Scopes.empty())
    PROTEUS_FATAL_ERROR("Expected FUNCTION scope");

  Scope S = Scopes.back();
  if (S.Kind != ScopeKind::FUNCTION)
    PROTEUS_FATAL_ERROR("Syntax error, expected FUNCTION end scope but found "
                        "unterminated scope " +
                        toString(S.Kind) + " @ " + S.File + ":" +
                        std::to_string(S.Line));
  Scopes.pop_back();
}

Function *FuncBase::getFunction() {
  Function *F = dyn_cast<Function>(FC.getCallee());
  if (!F)
    PROTEUS_FATAL_ERROR("Expected LLVM Function");
  return F;
}

Var &FuncBase::getArg(unsigned int ArgNo) { return Arguments.at(ArgNo); }

AllocaInst *FuncBase::emitAlloca(Type *Ty, StringRef Name) {
  auto SaveIP = IRB.saveIP();
  Function *F = getFunction();
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  IRB.restoreIP(AllocaIP);
  auto *Alloca = IRB.CreateAlloca(Ty, nullptr, Name);

  IRB.restoreIP(SaveIP);
  return Alloca;
}

void FuncBase::ret(std::optional<std::reference_wrapper<Var>> OptRet) {
  auto *CurBB = IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    PROTEUS_FATAL_ERROR("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  if (OptRet == std::nullopt) {
    IRB.CreateRetVoid();
  } else {
    auto *RetAlloca = OptRet->get().Alloca;
    auto *Ret = IRB.CreateLoad(RetAlloca->getAllocatedType(), RetAlloca);
    IRB.CreateRet(Ret);
  }

  TermI->eraseFromParent();
}

void FuncBase::beginIf(Var &CondVar, const char *File, int Line) {
  Function *F = getFunction();
  // Update the terminator of the current basic block due to the split
  // control-flow.
  BasicBlock *CurBlock = IP.getBlock();
  BasicBlock *NextBlock =
      CurBlock->splitBasicBlock(IP.getPoint(), CurBlock->getName() + ".split");

  auto ContIP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  Scopes.emplace_back(File, Line, ScopeKind::IF, ContIP);

  BasicBlock *ThenBlock =
      BasicBlock::Create(F->getContext(), "if.then", F, NextBlock);
  BasicBlock *ExitBlock =
      BasicBlock::Create(F->getContext(), "if.cont", F, NextBlock);

  CurBlock->getTerminator()->eraseFromParent();
  IRB.SetInsertPoint(CurBlock);
  {
    Value *Cond =
        IRB.CreateLoad(CondVar.Alloca->getAllocatedType(), CondVar.Alloca);
    IRB.CreateCondBr(Cond, ThenBlock, ExitBlock);
  }

  IRB.SetInsertPoint(ThenBlock);
  { IRB.CreateBr(ExitBlock); }

  IRB.SetInsertPoint(ExitBlock);
  { IRB.CreateBr(NextBlock); }

  IP = IRBuilderBase::InsertPoint(ThenBlock, ThenBlock->begin());
  IRB.restoreIP(IP);
}

void FuncBase::endIf() {
  if (Scopes.empty())
    PROTEUS_FATAL_ERROR("Expected IF scope");
  Scope S = Scopes.back();
  if (S.Kind != ScopeKind::IF)
    PROTEUS_FATAL_ERROR("Syntax error, expected IF end scope but "
                        "found unterminated scope " +
                        toString(S.Kind) + " @ " + S.File + ":" +
                        std::to_string(S.Line));

  IP = S.ContIP;
  Scopes.pop_back();

  IRB.restoreIP(IP);
}

void FuncBase::beginFor(Var &IterVar, Var &Init, Var &UpperBound, Var &Inc,
                        const char *File, int Line) {
  Function *F = getFunction();
  // Update the terminator of the current basic block due to the split
  // control-flow.
  BasicBlock *CurBlock = IP.getBlock();
  BasicBlock *NextBlock =
      CurBlock->splitBasicBlock(IP.getPoint(), CurBlock->getName() + ".split");

  auto ContIP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  Scopes.emplace_back(File, Line, ScopeKind::FOR, ContIP);

  BasicBlock *Header =
      BasicBlock::Create(F->getContext(), "loop.header", F, NextBlock);
  BasicBlock *LoopCond =
      BasicBlock::Create(F->getContext(), "loop.cond", F, NextBlock);
  BasicBlock *Body =
      BasicBlock::Create(F->getContext(), "loop.body", F, NextBlock);
  BasicBlock *Latch =
      BasicBlock::Create(F->getContext(), "loop.inc", F, NextBlock);
  BasicBlock *LoopExit =
      BasicBlock::Create(F->getContext(), "loop.end", F, NextBlock);

  // Erase the old terminator and branch to the header.
  CurBlock->getTerminator()->eraseFromParent();
  IRB.SetInsertPoint(CurBlock);
  { IRB.CreateBr(Header); }

  IRB.SetInsertPoint(Header);
  {
    IterVar = Init;
    IRB.CreateBr(LoopCond);
  }

  IRB.SetInsertPoint(LoopCond);
  {
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
    IRB.CreateBr(LoopCond);
  }

  IRB.SetInsertPoint(LoopExit);
  { IRB.CreateBr(NextBlock); }

  IP = IRBuilderBase::InsertPoint(Body, Body->begin());
  IRB.restoreIP(IP);
}

void FuncBase::endFor() {
  if (Scopes.empty())
    PROTEUS_FATAL_ERROR("Expected FOR scope");

  Scope S = Scopes.back();
  if (S.Kind != ScopeKind::FOR)
    PROTEUS_FATAL_ERROR("Syntax error, expected FOR end scope but "
                        "found unterminated scope " +
                        toString(S.Kind) + " @ " + S.File + ":" +
                        std::to_string(S.Line));

  IP = S.ContIP;
  Scopes.pop_back();

  IRB.restoreIP(IP);
}

} // namespace proteus
