#include "proteus/JitFrontend.hpp"

namespace proteus {

FuncBase::FuncBase(JitModule &J, FunctionCallee FC)
    : J(J), FC(FC), IRB{FC.getCallee()->getContext()} {
  Function *F = cast<Function>(FC.getCallee());
  BasicBlock::Create(F->getContext(), "entry", F);
  Name = F->getName();

  // Clang enables the 'contract' rewrite rule by default to enable FMA
  // instructions.
  // (Controllable via the '-ffp-contract' flag.)
  // Without this, PJ-DSL performance does not match JIT frontends
  // that use FMA instructions.
  // TODO: Make such things configurable.
  FastMathFlags FMF;
  FMF.setAllowContract(true);
  IRB.setFastMathFlags(FMF);
}

IRBuilderBase &FuncBase::getIRBuilder() {
  if (!IRB.GetInsertBlock())
    PROTEUS_FATAL_ERROR("Insert point is not set");
  return IRB;
}

Var &FuncBase::declVarInternal(StringRef Name, Type *Ty,
                               Type *PointerElemType) {
  auto *Alloca = emitAlloca(Ty, Name);

  if (PointerElemType) {
    Variables.emplace_back(
        std::make_unique<PointerVar>(Alloca, *this, PointerElemType));
    return *Variables.back();
  }

  Variables.emplace_back(std::make_unique<ScalarVar>(Alloca, *this));
  return *Variables.back();
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

Var &FuncBase::getArg(unsigned int ArgNo) { return *Arguments.at(ArgNo); }

AllocaInst *FuncBase::emitAlloca(Type *Ty, StringRef Name, AddressSpace AS) {
  auto SaveIP = IRB.saveIP();
  Function *F = getFunction();
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  IRB.restoreIP(AllocaIP);
  auto *Alloca = IRB.CreateAlloca(Ty, static_cast<unsigned>(AS), nullptr, Name);

  IRB.restoreIP(SaveIP);
  return Alloca;
}

Value *FuncBase::emitArrayCreate(Type *Ty, AddressSpace AT, StringRef Name) {
  if (!Ty || !Ty->isArrayTy())
    PROTEUS_FATAL_ERROR("Expected LLVM ArrayType for emitArrayCreate");

  auto *ArrTy = cast<ArrayType>(Ty);

  switch (AT) {
  case AddressSpace::SHARED:
  case AddressSpace::GLOBAL: {
    Module *M = getFunction()->getParent();
    auto *GV = new GlobalVariable(
        *M, ArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
        UndefValue::get(ArrTy), Name, /*InsertBefore=*/nullptr,
        GlobalValue::NotThreadLocal, static_cast<unsigned>(AT),
        /*ExternallyInitialized=*/false);

    return GV;
  }
  case AddressSpace::DEFAULT:
  case AddressSpace::LOCAL: {
    auto *Alloca = emitAlloca(ArrTy, Name, AT);
    return Alloca;
  }
  case AddressSpace::CONSTANT:
    PROTEUS_FATAL_ERROR("Constant arrays are not supported");
  default:
    PROTEUS_FATAL_ERROR("Unsupported AddressSpace");
  }
}

void FuncBase::ret(std::optional<std::reference_wrapper<Var>> OptRet) {
  auto *CurBB = IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    PROTEUS_FATAL_ERROR("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  if (OptRet == std::nullopt) {
    IRB.CreateRetVoid();
  } else {
    auto *RetAlloca = OptRet->get().getAlloca();
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
    Value *Cond = IRB.CreateLoad(CondVar.getAlloca()->getAllocatedType(),
                                 CondVar.getAlloca());
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

void FuncBase::beginIfTT(const VarTT<bool> &CondVar, const char *File, int Line) {
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
    Value *Cond = CondVar.Storage->loadValue();
    IRB.CreateCondBr(Cond, ThenBlock, ExitBlock);
  }

  IRB.SetInsertPoint(ThenBlock);
  { IRB.CreateBr(ExitBlock); }

  IRB.SetInsertPoint(ExitBlock);
  { IRB.CreateBr(NextBlock); }

  IP = IRBuilderBase::InsertPoint(ThenBlock, ThenBlock->begin());
  IRB.restoreIP(IP);
}

void FuncBase::endIfTT() {
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
    Value *Cond = IRB.CreateLoad(CondVar.getAlloca()->getAllocatedType(),
                                 CondVar.getAlloca());
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

Var &FuncBase::emitAtomic(AtomicRMWInst::BinOp Op, Var &Addr, Var &Val) {
  if (Addr.kind() != VarKind::Pointer)
    PROTEUS_FATAL_ERROR("Atomic ops require a pointer variable");

  if (Addr.getValueType() != Val.getValueType())
    PROTEUS_FATAL_ERROR("Atomic op require values of same type");

  auto &IRB = getIRBuilder();
  Type *ValueType = Val.getValueType();
  auto *Result = IRB.CreateAtomicRMW(
      Op, Addr.getPointerValue(), Val.getValue(), MaybeAlign(),
      AtomicOrdering::SequentiallyConsistent, SyncScope::SingleThread);
  Var &Ret = declVarInternal("res.", ValueType);
  Ret.storeValue(Result);
  return Ret;
}

Var &FuncBase::atomicAdd(Var &Addr, Var &Val) {
  Type *ValueType = Val.getValueType();
  auto Op =
      ValueType->isFloatingPointTy() ? AtomicRMWInst::FAdd : AtomicRMWInst::Add;
  return emitAtomic(Op, Addr, Val);
}

Var &FuncBase::atomicSub(Var &Addr, Var &Val) {
  Type *ValueType = Val.getValueType();
  auto Op =
      ValueType->isFloatingPointTy() ? AtomicRMWInst::FSub : AtomicRMWInst::Sub;
  return emitAtomic(Op, Addr, Val);
}

Var &FuncBase::atomicMax(Var &Addr, Var &Val) {
  Type *ValueType = Val.getValueType();
  auto Op =
      ValueType->isFloatingPointTy() ? AtomicRMWInst::FMax : AtomicRMWInst::Max;
  return emitAtomic(Op, Addr, Val);
}

Var &FuncBase::atomicMin(Var &Addr, Var &Val) {
  Type *ValueType = Val.getValueType();
  auto Op =
      ValueType->isFloatingPointTy() ? AtomicRMWInst::FMin : AtomicRMWInst::Min;
  return emitAtomic(Op, Addr, Val);
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
