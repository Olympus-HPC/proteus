#include "proteus/Frontend/LLVMCodeBuilder.h"
#include "proteus/Frontend/LLVMTypeMap.h"
#include "proteus/Frontend/LLVMValueMap.h"
#include "proteus/Frontend/TargetModel.h"
#include "proteus/impl/CoreLLVMDevice.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {

using namespace llvm;

struct LLVMCodeBuilder::Impl {
  // Owned context and module (only set via the owning constructor).
  std::unique_ptr<LLVMContext> OwnedCtx;
  std::unique_ptr<Module> OwnedMod;

  IRBuilder<> IRB;
  IRBuilderBase::InsertPoint IP;

  struct Scope {
    std::string File;
    int Line;
    ScopeKind Kind;
    IRBuilderBase::InsertPoint ContIP;
    Scope(const char *File, int Line, ScopeKind Kind,
          IRBuilderBase::InsertPoint ContIP)
        : File(File), Line(Line), Kind(Kind), ContIP(ContIP) {}
  };
  std::vector<Scope> Scopes;

  explicit Impl(LLVMContext &Ctx) : IRB{Ctx} { init(); }

  Impl(std::unique_ptr<LLVMContext> Ctx, std::unique_ptr<Module> Mod)
      : OwnedCtx{std::move(Ctx)}, OwnedMod{std::move(Mod)}, IRB{*OwnedCtx} {
    init();
  }

  void init() {
    // Initialize IP.
    IP = IRB.saveIP();
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
};

LLVMCodeBuilder::LLVMCodeBuilder(std::unique_ptr<LLVMContext> Ctx,
                                 std::unique_ptr<Module> Mod,
                                 TargetModelType TM)
    : PImpl{std::make_unique<Impl>(std::move(Ctx), std::move(Mod))}, F(nullptr),
      TargetModel(TM) {
  getModule().setTargetTriple(getTargetTriple(TM));
}

LLVMCodeBuilder::~LLVMCodeBuilder() = default;

IRBuilderBase &LLVMCodeBuilder::getIRBuilder() { return PImpl->IRB; }

Function &LLVMCodeBuilder::getFunction() {
  if (!F)
    reportFatalError("LLVMCodeBuilder: no active function");
  return *F;
}

Module &LLVMCodeBuilder::getModule() {
  if (PImpl->OwnedMod)
    return *PImpl->OwnedMod;
  if (!F)
    reportFatalError("LLVMCodeBuilder: no active function or owned module");
  return *F->getParent();
}

LLVMContext &LLVMCodeBuilder::getContext() {
  if (PImpl->OwnedCtx)
    return *PImpl->OwnedCtx;
  if (!F)
    reportFatalError("LLVMCodeBuilder: no active function or owned context");
  return F->getContext();
}

Function *LLVMCodeBuilder::addFunction(const std::string &Name, IRType RetTy,
                                       const std::vector<IRType> &ArgTys) {
  if (!PImpl->OwnedMod)
    reportFatalError("addFunction requires an owning LLVMCodeBuilder");
  auto &Ctx = getContext();
  Type *LLVMRetTy = toLLVMType(RetTy, Ctx);
  std::vector<Type *> LLVMArgTys;
  LLVMArgTys.reserve(ArgTys.size());
  for (const auto &T : ArgTys)
    LLVMArgTys.push_back(toLLVMType(T, Ctx));
  auto FC = PImpl->OwnedMod->getOrInsertFunction(
      Name, FunctionType::get(LLVMRetTy, LLVMArgTys, false));
  Function *Fn = dyn_cast<Function>(FC.getCallee());
  if (!Fn)
    reportFatalError("Expected LLVM Function");
  BasicBlock::Create(Fn->getContext(), "entry", Fn);
  F = Fn;
  return Fn;
}

std::unique_ptr<LLVMContext> LLVMCodeBuilder::takeLLVMContext() {
  return std::move(PImpl->OwnedCtx);
}

std::unique_ptr<Module> LLVMCodeBuilder::takeModule() {
  return std::move(PImpl->OwnedMod);
}

// Insert point management.
void LLVMCodeBuilder::setInsertPoint(BasicBlock *BB) {
  PImpl->IRB.SetInsertPoint(BB);
}

void LLVMCodeBuilder::setInsertPointBegin(BasicBlock *BB) {
  PImpl->IP = IRBuilderBase::InsertPoint(BB, BB->begin());
  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::setInsertPointAtEntry() {
  auto &EntryBB = F->getEntryBlock();
  PImpl->IP = IRBuilderBase::InsertPoint(&EntryBB, EntryBB.end());
  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::clearInsertPoint() { PImpl->IRB.ClearInsertionPoint(); }

BasicBlock *LLVMCodeBuilder::getInsertBlock() {
  return PImpl->IRB.GetInsertBlock();
}

// Basic block operations.
std::tuple<BasicBlock *, BasicBlock *> LLVMCodeBuilder::splitCurrentBlock() {
  BasicBlock *CurBlock = PImpl->IP.getBlock();
  BasicBlock *NextBlock = CurBlock->splitBasicBlock(
      PImpl->IP.getPoint(), CurBlock->getName() + ".split");
  return {CurBlock, NextBlock};
}

BasicBlock *LLVMCodeBuilder::createBasicBlock(const std::string &Name,
                                              BasicBlock *InsertBefore) {
  BasicBlock *BB = BasicBlock::Create(getContext(), Name, F, InsertBefore);
  return BB;
}

void LLVMCodeBuilder::eraseTerminator(BasicBlock *BB) {
  if (!BB->getTerminator())
    reportFatalError("Basic block has no terminator to erase");

  BB->getTerminator()->eraseFromParent();
}

BasicBlock *LLVMCodeBuilder::getUniqueSuccessor(BasicBlock *BB) {
  auto *Succ = BB->getUniqueSuccessor();
  if (!Succ)
    reportFatalError("Expected unique successor for basic block " +
                     BB->getName().str());
  return Succ;
}

// Scope management.
void LLVMCodeBuilder::pushScope(const char *File, int Line, ScopeKind Kind,
                                BasicBlock *NextBlock) {
  IRBuilderBase::InsertPoint ContIP{NextBlock, NextBlock->begin()};
  PImpl->Scopes.emplace_back(File, Line, Kind, ContIP);
}

// High-level scope operations.
void LLVMCodeBuilder::beginFunction(Function &Fn, const char *File, int Line) {
  F = &Fn;
  BasicBlock *BodyBB = BasicBlock::Create(F->getContext(), "body", F);
  BasicBlock *ExitBB = BasicBlock::Create(F->getContext(), "exit", F);
  PImpl->IP =
      IRBuilderBase::InsertPoint(&F->getEntryBlock(), F->getEntryBlock().end());
  PImpl->IRB.restoreIP(PImpl->IP);
  PImpl->IRB.CreateBr(BodyBB);

  PImpl->IP = IRBuilderBase::InsertPoint(BodyBB, BodyBB->end());
  PImpl->IRB.restoreIP(PImpl->IP);
  PImpl->IRB.CreateBr(ExitBB);

  PImpl->IRB.SetInsertPoint(ExitBB);
  { PImpl->IRB.CreateUnreachable(); }

  PImpl->IP = IRBuilderBase::InsertPoint(BodyBB, BodyBB->begin());
  PImpl->IRB.restoreIP(PImpl->IP);

  PImpl->Scopes.emplace_back(
      File, Line, ScopeKind::FUNCTION,
      IRBuilderBase::InsertPoint(ExitBB, ExitBB->begin()));
}

void LLVMCodeBuilder::endFunction() {
  if (PImpl->Scopes.empty())
    reportFatalError("Expected FUNCTION scope");

  Impl::Scope S = PImpl->Scopes.back();
  if (S.Kind != ScopeKind::FUNCTION)
    reportFatalError("Syntax error, expected FUNCTION end scope but found "
                     "unterminated scope " +
                     toString(S.Kind) + " @ " + S.File + ":" +
                     std::to_string(S.Line));
  PImpl->Scopes.pop_back();
}

void LLVMCodeBuilder::beginIf(IRValue Cond, const char *File, int Line) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  BasicBlock *CurBlock = PImpl->IP.getBlock();
  BasicBlock *NextBlock = CurBlock->splitBasicBlock(
      PImpl->IP.getPoint(), CurBlock->getName() + ".split");

  auto ContIP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  PImpl->Scopes.emplace_back(File, Line, ScopeKind::IF, ContIP);

  BasicBlock *ThenBlock =
      BasicBlock::Create(F->getContext(), "if.then", F, NextBlock);
  BasicBlock *ExitBlock =
      BasicBlock::Create(F->getContext(), "if.cont", F, NextBlock);

  CurBlock->getTerminator()->eraseFromParent();
  PImpl->IRB.SetInsertPoint(CurBlock);
  { PImpl->IRB.CreateCondBr(toLLVMValue(Cond), ThenBlock, ExitBlock); }

  PImpl->IRB.SetInsertPoint(ThenBlock);
  { PImpl->IRB.CreateBr(ExitBlock); }

  PImpl->IRB.SetInsertPoint(ExitBlock);
  { PImpl->IRB.CreateBr(NextBlock); }

  PImpl->IP = IRBuilderBase::InsertPoint(ThenBlock, ThenBlock->begin());
  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::endIf() {
  if (PImpl->Scopes.empty())
    reportFatalError("Expected IF scope");
  Impl::Scope S = PImpl->Scopes.back();
  if (S.Kind != ScopeKind::IF)
    reportFatalError("Syntax error, expected IF end scope but "
                     "found unterminated scope " +
                     toString(S.Kind) + " @ " + S.File + ":" +
                     std::to_string(S.Line));

  PImpl->IP = S.ContIP;
  PImpl->Scopes.pop_back();

  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::beginFor(IRValue IterSlot, IRType IterTy, IRValue InitVal,
                               IRValue UpperBoundVal, IRValue IncVal,
                               bool IsSigned, const char *File, int Line) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  auto [CurBlock, NextBlock] = splitCurrentBlock();
  pushScope(File, Line, ScopeKind::FOR, NextBlock);

  llvm::BasicBlock *Header = createBasicBlock("loop.header", NextBlock);
  llvm::BasicBlock *LoopCond = createBasicBlock("loop.cond", NextBlock);
  llvm::BasicBlock *Body = createBasicBlock("loop.body", NextBlock);
  llvm::BasicBlock *Latch = createBasicBlock("loop.inc", NextBlock);
  llvm::BasicBlock *LoopExit = createBasicBlock("loop.end", NextBlock);

  // Erase the old terminator and branch to the header.
  eraseTerminator(CurBlock);
  setInsertPoint(CurBlock);
  { createBr(Header); }

  setInsertPoint(Header);
  {
    createStore(InitVal, IterSlot);
    createBr(LoopCond);
  }

  setInsertPoint(LoopCond);
  {
    IRValue Iter = createLoad(IterTy, IterSlot);
    IRValue Cond = IsSigned ? createICmpSLT(Iter, UpperBoundVal)
                            : createICmpULT(Iter, UpperBoundVal);
    createCondBr(Cond, Body, LoopExit);
  }

  setInsertPoint(Body);
  createBr(Latch);

  setInsertPoint(Latch);
  {
    IRValue Iter = createLoad(IterTy, IterSlot);
    IRValue Next = createAdd(Iter, IncVal);
    createStore(Next, IterSlot);
    createBr(LoopCond);
  }

  setInsertPoint(LoopExit);
  { createBr(NextBlock); }

  setInsertPointBegin(Body);
}

void LLVMCodeBuilder::endFor() {
  if (PImpl->Scopes.empty())
    reportFatalError("Expected FOR scope");

  Impl::Scope S = PImpl->Scopes.back();
  if (S.Kind != ScopeKind::FOR)
    reportFatalError("Syntax error, expected FOR end scope but "
                     "found unterminated scope " +
                     toString(S.Kind) + " @ " + S.File + ":" +
                     std::to_string(S.Line));

  PImpl->IP = S.ContIP;
  PImpl->Scopes.pop_back();

  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::beginWhile(std::function<IRValue()> CondFn,
                                 const char *File, int Line) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  auto [CurBlock, NextBlock] = splitCurrentBlock();
  pushScope(File, Line, ScopeKind::WHILE, NextBlock);

  llvm::BasicBlock *LoopCond = createBasicBlock("while.cond", NextBlock);
  llvm::BasicBlock *Body = createBasicBlock("while.body", NextBlock);
  llvm::BasicBlock *LoopExit = createBasicBlock("while.end", NextBlock);

  eraseTerminator(CurBlock);
  setInsertPoint(CurBlock);
  { createBr(LoopCond); }

  setInsertPoint(LoopCond);
  {
    IRValue CondV = CondFn();
    createCondBr(CondV, Body, LoopExit);
  }

  setInsertPoint(Body);
  createBr(LoopCond);

  setInsertPoint(LoopExit);
  { createBr(NextBlock); }

  setInsertPointBegin(Body);
}

void LLVMCodeBuilder::endWhile() {
  if (PImpl->Scopes.empty())
    reportFatalError("Expected WHILE scope");

  Impl::Scope S = PImpl->Scopes.back();
  if (S.Kind != ScopeKind::WHILE)
    reportFatalError("Syntax error, expected WHILE end scope but "
                     "found unterminated scope " +
                     toString(S.Kind) + " @ " + S.File + ":" +
                     std::to_string(S.Line));

  PImpl->IP = S.ContIP;
  PImpl->Scopes.pop_back();

  PImpl->IRB.restoreIP(PImpl->IP);
}

// Arithmetic operations.
IRValue LLVMCodeBuilder::createAdd(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateAdd(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFAdd(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFAdd(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createSub(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateSub(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFSub(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFSub(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createMul(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateMul(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFMul(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFMul(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createUDiv(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateUDiv(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createSDiv(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateSDiv(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFDiv(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFDiv(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createURem(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateURem(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createSRem(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateSRem(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFRem(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFRem(toLLVMValue(LHS), toLLVMValue(RHS)));
}

// Atomic operations.
IRValue LLVMCodeBuilder::createAtomicAdd(IRValue Addr, IRValue Val) {
  Value *LLVMVal = toLLVMValue(Val);
  auto Op = LLVMVal->getType()->isFloatingPointTy() ? AtomicRMWInst::FAdd
                                                    : AtomicRMWInst::Add;

  return fromLLVMValue(PImpl->IRB.CreateAtomicRMW(
      Op, toLLVMValue(Addr), LLVMVal, MaybeAlign(),
      AtomicOrdering::SequentiallyConsistent, SyncScope::SingleThread));
}

IRValue LLVMCodeBuilder::createAtomicSub(IRValue Addr, IRValue Val) {
  Value *LLVMVal = toLLVMValue(Val);
  auto Op = LLVMVal->getType()->isFloatingPointTy() ? AtomicRMWInst::FSub
                                                    : AtomicRMWInst::Sub;

  return fromLLVMValue(PImpl->IRB.CreateAtomicRMW(
      Op, toLLVMValue(Addr), LLVMVal, MaybeAlign(),
      AtomicOrdering::SequentiallyConsistent, SyncScope::SingleThread));
}

IRValue LLVMCodeBuilder::createAtomicMax(IRValue Addr, IRValue Val) {
  Value *LLVMVal = toLLVMValue(Val);
  auto Op = LLVMVal->getType()->isFloatingPointTy() ? AtomicRMWInst::FMax
                                                    : AtomicRMWInst::Max;

  return fromLLVMValue(PImpl->IRB.CreateAtomicRMW(
      Op, toLLVMValue(Addr), LLVMVal, MaybeAlign(),
      AtomicOrdering::SequentiallyConsistent, SyncScope::SingleThread));
}

IRValue LLVMCodeBuilder::createAtomicMin(IRValue Addr, IRValue Val) {
  Value *LLVMVal = toLLVMValue(Val);
  auto Op = LLVMVal->getType()->isFloatingPointTy() ? AtomicRMWInst::FMin
                                                    : AtomicRMWInst::Min;

  return fromLLVMValue(PImpl->IRB.CreateAtomicRMW(
      Op, toLLVMValue(Addr), LLVMVal, MaybeAlign(),
      AtomicOrdering::SequentiallyConsistent, SyncScope::SingleThread));
}

// Comparison operations.
IRValue LLVMCodeBuilder::createICmpEQ(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpEQ(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpNE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpNE(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpSLT(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpSLT(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpSGT(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpSGT(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpSGE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpSGE(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpSLE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpSLE(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpUGT(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpUGT(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpUGE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpUGE(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpULT(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpULT(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createICmpULE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateICmpULE(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFCmpOEQ(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFCmpOEQ(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFCmpONE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFCmpONE(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFCmpOLT(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFCmpOLT(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFCmpOLE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFCmpOLE(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFCmpOGT(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFCmpOGT(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFCmpOGE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFCmpOGE(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFCmpULT(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFCmpULT(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createFCmpULE(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateFCmpULE(toLLVMValue(LHS), toLLVMValue(RHS)));
}

// Logical operations.
IRValue LLVMCodeBuilder::createAnd(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateAnd(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createOr(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(PImpl->IRB.CreateOr(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createXor(IRValue LHS, IRValue RHS) {
  return fromLLVMValue(
      PImpl->IRB.CreateXor(toLLVMValue(LHS), toLLVMValue(RHS)));
}
IRValue LLVMCodeBuilder::createNot(IRValue Val) {
  return fromLLVMValue(PImpl->IRB.CreateNot(toLLVMValue(Val)));
}

// Load/Store operations.
IRValue LLVMCodeBuilder::createLoad(IRType Ty, IRValue Ptr,
                                    const std::string &Name) {
  return fromLLVMValue(PImpl->IRB.CreateLoad(toLLVMType(Ty, getContext()),
                                             toLLVMValue(Ptr), Name));
}

void LLVMCodeBuilder::createStore(IRValue Val, IRValue Ptr) {
  PImpl->IRB.CreateStore(toLLVMValue(Val), toLLVMValue(Ptr));
}

// Control flow operations.
void LLVMCodeBuilder::createBr(BasicBlock *Dest) { PImpl->IRB.CreateBr(Dest); }

void LLVMCodeBuilder::createCondBr(IRValue Cond, BasicBlock *True,
                                   BasicBlock *False) {
  PImpl->IRB.CreateCondBr(toLLVMValue(Cond), True, False);
}

void LLVMCodeBuilder::createRetVoid() {
  auto *CurBB = PImpl->IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    reportFatalError("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  PImpl->IRB.CreateRetVoid();

  TermI->eraseFromParent();
}

void LLVMCodeBuilder::createRet(IRValue V) {
  auto *CurBB = PImpl->IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    reportFatalError("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  PImpl->IRB.CreateRet(toLLVMValue(V));

  TermI->eraseFromParent();
}

// Cast operations.
IRValue LLVMCodeBuilder::createIntCast(IRValue V, IRType DestTy,
                                       bool IsSigned) {
  return fromLLVMValue(PImpl->IRB.CreateIntCast(
      toLLVMValue(V), toLLVMType(DestTy, getContext()), IsSigned));
}

IRValue LLVMCodeBuilder::createFPCast(IRValue V, IRType DestTy) {
  return fromLLVMValue(PImpl->IRB.CreateFPCast(
      toLLVMValue(V), toLLVMType(DestTy, getContext())));
}

IRValue LLVMCodeBuilder::createSIToFP(IRValue V, IRType DestTy) {
  return fromLLVMValue(PImpl->IRB.CreateSIToFP(
      toLLVMValue(V), toLLVMType(DestTy, getContext())));
}
IRValue LLVMCodeBuilder::createUIToFP(IRValue V, IRType DestTy) {
  return fromLLVMValue(PImpl->IRB.CreateUIToFP(
      toLLVMValue(V), toLLVMType(DestTy, getContext())));
}
IRValue LLVMCodeBuilder::createFPToSI(IRValue V, IRType DestTy) {
  return fromLLVMValue(PImpl->IRB.CreateFPToSI(
      toLLVMValue(V), toLLVMType(DestTy, getContext())));
}
IRValue LLVMCodeBuilder::createFPToUI(IRValue V, IRType DestTy) {
  return fromLLVMValue(PImpl->IRB.CreateFPToUI(
      toLLVMValue(V), toLLVMType(DestTy, getContext())));
}
IRValue LLVMCodeBuilder::createBitCast(IRValue V, IRType DestTy) {
  Value *LLVMv = toLLVMValue(V);
  Type *DestLLVMTy = toLLVMType(DestTy, getContext());
  if (LLVMv->getType() == DestLLVMTy)
    return V;

  return fromLLVMValue(PImpl->IRB.CreateBitCast(LLVMv, DestLLVMTy));
}

IRValue LLVMCodeBuilder::createZExt(IRValue V, IRType DestTy) {
  return fromLLVMValue(
      PImpl->IRB.CreateZExt(toLLVMValue(V), toLLVMType(DestTy, getContext())));
}

// Constant creation.
IRValue LLVMCodeBuilder::getConstantInt(IRType Ty, uint64_t Val) {
  return fromLLVMValue(ConstantInt::get(toLLVMType(Ty, getContext()), Val));
}
IRValue LLVMCodeBuilder::getConstantFP(IRType Ty, double Val) {
  return fromLLVMValue(ConstantFP::get(toLLVMType(Ty, getContext()), Val));
}

// GEP operations.
IRValue LLVMCodeBuilder::createInBoundsGEP(IRType Ty, IRValue Ptr,
                                           const std::vector<IRValue> IdxList,
                                           const std::string &Name) {
  std::vector<Value *> LLVMIdxList;
  LLVMIdxList.reserve(IdxList.size());
  for (const auto &I : IdxList)
    LLVMIdxList.push_back(toLLVMValue(I));
  return fromLLVMValue(PImpl->IRB.CreateInBoundsGEP(
      toLLVMType(Ty, getContext()), toLLVMValue(Ptr), LLVMIdxList, Name));
}
IRValue LLVMCodeBuilder::createConstInBoundsGEP1_64(IRType Ty, IRValue Ptr,
                                                    size_t Idx) {
  return fromLLVMValue(PImpl->IRB.CreateConstInBoundsGEP1_64(
      toLLVMType(Ty, getContext()), toLLVMValue(Ptr), Idx));
}

IRValue LLVMCodeBuilder::createConstInBoundsGEP2_64(IRType Ty, IRValue Ptr,
                                                    size_t Idx0, size_t Idx1) {
  return fromLLVMValue(PImpl->IRB.CreateConstInBoundsGEP2_64(
      toLLVMType(Ty, getContext()), toLLVMValue(Ptr), Idx0, Idx1));
}

// Type accessors.
Type *LLVMCodeBuilder::getPointerType(Type *ElemTy, unsigned AS) {
  return PointerType::get(ElemTy, AS);
}
Type *LLVMCodeBuilder::getPointerTypeUnqual(Type *ElemTy) {
  return PointerType::getUnqual(ElemTy);
}
Type *LLVMCodeBuilder::getInt16Ty() { return PImpl->IRB.getInt16Ty(); }
Type *LLVMCodeBuilder::getInt32Ty() { return PImpl->IRB.getInt32Ty(); }
Type *LLVMCodeBuilder::getInt64Ty() { return PImpl->IRB.getInt64Ty(); }
Type *LLVMCodeBuilder::getFloatTy() { return PImpl->IRB.getFloatTy(); }

// Type queries.
unsigned LLVMCodeBuilder::getAddressSpace(Type *Ty) {
  auto *PtrTy = dyn_cast<PointerType>(Ty);
  if (!PtrTy)
    reportFatalError("Expected LLVM PointerType for getAddressSpace");

  return PtrTy->getAddressSpace();
}

unsigned LLVMCodeBuilder::getAddressSpaceFromValue(IRValue PtrVal) {
  return getAddressSpace(toLLVMValue(PtrVal)->getType());
}

bool LLVMCodeBuilder::isIntegerTy(Type *Ty) { return Ty->isIntegerTy(); }
bool LLVMCodeBuilder::isFloatingPointTy(Type *Ty) {
  return Ty->isFloatingPointTy();
}

// Call operations.
IRValue LLVMCodeBuilder::createCall(const std::string &FName, IRType RetTy,
                                    const std::vector<IRType> &ArgTys,
                                    const std::vector<IRValue> &Args) {
  auto &Ctx = getContext();
  Type *LLVMRetTy = toLLVMType(RetTy, Ctx);
  std::vector<Type *> LLVMArgTys;
  LLVMArgTys.reserve(ArgTys.size());
  for (const auto &T : ArgTys)
    LLVMArgTys.push_back(toLLVMType(T, Ctx));
  std::vector<Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (const auto &A : Args)
    LLVMArgs.push_back(toLLVMValue(A));
  Module *M = &getModule();
  FunctionType *FnTy = FunctionType::get(LLVMRetTy, LLVMArgTys, false);
  FunctionCallee Callee = M->getOrInsertFunction(FName, FnTy);
  return fromLLVMValue(PImpl->IRB.CreateCall(Callee, LLVMArgs));
}

IRValue LLVMCodeBuilder::createCall(const std::string &FName, IRType RetTy) {
  Type *LLVMRetTy = toLLVMType(RetTy, getContext());
  Module *M = &getModule();
  FunctionType *FnTy = FunctionType::get(LLVMRetTy, {}, false);
  FunctionCallee Callee = M->getOrInsertFunction(FName, FnTy);
  return fromLLVMValue(PImpl->IRB.CreateCall(Callee));
}

// Alloca/array emission.
IRValue LLVMCodeBuilder::emitAlloca(Type *Ty, const std::string &Name,
                                    AddressSpace AS) {
  auto SaveIP = PImpl->IRB.saveIP();
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  PImpl->IRB.restoreIP(AllocaIP);
  auto *Alloca =
      PImpl->IRB.CreateAlloca(Ty, static_cast<unsigned>(AS), nullptr, Name);

  PImpl->IRB.restoreIP(SaveIP);
  return fromLLVMValue(Alloca);
}

IRValue LLVMCodeBuilder::emitArrayCreate(Type *Ty, AddressSpace AT,
                                         const std::string &Name) {
  if (!Ty || !Ty->isArrayTy())
    reportFatalError("Expected LLVM ArrayType for emitArrayCreate");

  auto *ArrTy = cast<ArrayType>(Ty);

  switch (AT) {
  case AddressSpace::SHARED:
  case AddressSpace::GLOBAL: {
    Module *M = &getModule();
    auto *GV = new GlobalVariable(
        *M, ArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
        UndefValue::get(ArrTy), Name, /*InsertBefore=*/nullptr,
        GlobalValue::NotThreadLocal, static_cast<unsigned>(AT),
        /*ExternallyInitialized=*/false);

    return fromLLVMValue(GV);
  }
  case AddressSpace::DEFAULT:
  case AddressSpace::LOCAL: {
    return emitAlloca(ArrTy, Name, AT);
  }
  case AddressSpace::CONSTANT:
    reportFatalError("Constant arrays are not supported");
  default:
    reportFatalError("Unsupported AddressSpace");
  }
}

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
void LLVMCodeBuilder::setKernel(Function &Fn) {
  LLVMContext &Ctx = getContext();
  switch (TargetModel) {
  case TargetModelType::CUDA: {
    NamedMDNode *MD = getModule().getOrInsertNamedMetadata("nvvm.annotations");

    Metadata *MDVals[] = {
        ConstantAsMetadata::get(&Fn), MDString::get(Ctx, "kernel"),
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1))};
    // Append metadata to nvvm.annotations.
    MD->addOperand(MDNode::get(Ctx, MDVals));

    // Add a function attribute for the kernel.
    Fn.addFnAttr(Attribute::get(Ctx, "kernel"));
#if LLVM_VERSION_MAJOR >= 20
    Fn.setCallingConv(CallingConv::PTX_Kernel);
#endif
    return;
  }
  case TargetModelType::HIP:
    Fn.setCallingConv(CallingConv::AMDGPU_KERNEL);
    return;
  case TargetModelType::HOST:
    reportFatalError("Host does not support setKernel");
  default:
    reportFatalError("Unsupported target " + getTargetTriple(TargetModel) +
                     " for setKernel");
  }
}

void LLVMCodeBuilder::setLaunchBoundsForKernel(Function &Fn,
                                               int MaxThreadsPerBlock,
                                               int MinBlocksPerSM) {
  proteus::setLaunchBoundsForKernel(Fn, MaxThreadsPerBlock, MinBlocksPerSM);
}
#endif

std::unique_ptr<ScalarStorage>
LLVMCodeBuilder::createScalarStorage(const std::string &Name,
                                     IRType AllocaIRTy) {
  Type *LLVMTy = toLLVMType(AllocaIRTy, getContext());
  return std::make_unique<ScalarStorage>(emitAlloca(LLVMTy, Name),
                                         getIRBuilder(), AllocaIRTy);
}

std::unique_ptr<PointerStorage>
LLVMCodeBuilder::createPointerStorage(const std::string &Name, IRType ElemIRTy,
                                      unsigned AddrSpace) {
  auto &Ctx = getContext();
  Type *AllocaTy = PointerType::get(Ctx, AddrSpace);
  return std::make_unique<PointerStorage>(emitAlloca(AllocaTy, Name),
                                          getIRBuilder(), ElemIRTy);
}

std::unique_ptr<ArrayStorage>
LLVMCodeBuilder::createArrayStorage(const std::string &Name, AddressSpace AS,
                                    IRType ArrIRTy) {
  auto &Ctx = getContext();
  Type *ElemLLVMTy = toLLVMType(IRType{ArrIRTy.ElemKind}, Ctx);
  Type *ArrTy = ArrayType::get(ElemLLVMTy, ArrIRTy.NElem);
  IRType ElemIRTy{ArrIRTy.ElemKind, ArrIRTy.Signed};
  IRValue BasePointer = emitArrayCreate(ArrTy, AS, Name);
  return std::make_unique<ArrayStorage>(BasePointer, getIRBuilder(),
                                        ArrIRTy.NElem, ElemIRTy);
}

} // namespace proteus
