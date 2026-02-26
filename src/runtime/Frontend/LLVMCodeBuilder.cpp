#include "proteus/Frontend/LLVMCodeBuilder.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {

using namespace llvm;

struct LLVMCodeBuilder::Impl {
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

  Impl(LLVMContext &Ctx) : IRB{Ctx} {
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

LLVMCodeBuilder::LLVMCodeBuilder(Function &F)
    : PImpl{std::make_unique<Impl>(F.getContext())}, F(F) {}

LLVMCodeBuilder::~LLVMCodeBuilder() = default;

IRBuilderBase &LLVMCodeBuilder::getIRBuilder() { return PImpl->IRB; }

Function &LLVMCodeBuilder::getFunction() { return F; }

Module &LLVMCodeBuilder::getModule() { return *F.getParent(); }

LLVMContext &LLVMCodeBuilder::getContext() { return F.getContext(); }

// Insert point management.
void LLVMCodeBuilder::setInsertPoint(BasicBlock *BB) {
  PImpl->IRB.SetInsertPoint(BB);
}

void LLVMCodeBuilder::setInsertPointBegin(BasicBlock *BB) {
  PImpl->IP = IRBuilderBase::InsertPoint(BB, BB->begin());
  PImpl->IRB.restoreIP(PImpl->IP);
}

void LLVMCodeBuilder::setInsertPointAtEntry() {
  auto &EntryBB = F.getEntryBlock();
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
  BasicBlock *BB = BasicBlock::Create(getContext(), Name, &F, InsertBefore);
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
void LLVMCodeBuilder::beginFunction(const char *File, int Line) {
  BasicBlock *BodyBB = BasicBlock::Create(F.getContext(), "body", &F);
  BasicBlock *ExitBB = BasicBlock::Create(F.getContext(), "exit", &F);
  PImpl->IP =
      IRBuilderBase::InsertPoint(&F.getEntryBlock(), F.getEntryBlock().end());
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

void LLVMCodeBuilder::beginIf(Value *Cond, const char *File, int Line) {
  // Update the terminator of the current basic block due to the split
  // control-flow.
  BasicBlock *CurBlock = PImpl->IP.getBlock();
  BasicBlock *NextBlock = CurBlock->splitBasicBlock(
      PImpl->IP.getPoint(), CurBlock->getName() + ".split");

  auto ContIP = IRBuilderBase::InsertPoint(NextBlock, NextBlock->begin());
  PImpl->Scopes.emplace_back(File, Line, ScopeKind::IF, ContIP);

  BasicBlock *ThenBlock =
      BasicBlock::Create(F.getContext(), "if.then", &F, NextBlock);
  BasicBlock *ExitBlock =
      BasicBlock::Create(F.getContext(), "if.cont", &F, NextBlock);

  CurBlock->getTerminator()->eraseFromParent();
  PImpl->IRB.SetInsertPoint(CurBlock);
  { PImpl->IRB.CreateCondBr(Cond, ThenBlock, ExitBlock); }

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
Value *LLVMCodeBuilder::createAdd(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateAdd(LHS, RHS);
}
Value *LLVMCodeBuilder::createFAdd(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFAdd(LHS, RHS);
}
Value *LLVMCodeBuilder::createSub(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateSub(LHS, RHS);
}
Value *LLVMCodeBuilder::createFSub(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFSub(LHS, RHS);
}
Value *LLVMCodeBuilder::createMul(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateMul(LHS, RHS);
}
Value *LLVMCodeBuilder::createFMul(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFMul(LHS, RHS);
}
Value *LLVMCodeBuilder::createUDiv(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateUDiv(LHS, RHS);
}
Value *LLVMCodeBuilder::createSDiv(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateSDiv(LHS, RHS);
}
Value *LLVMCodeBuilder::createFDiv(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFDiv(LHS, RHS);
}
Value *LLVMCodeBuilder::createURem(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateURem(LHS, RHS);
}
Value *LLVMCodeBuilder::createSRem(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateSRem(LHS, RHS);
}
Value *LLVMCodeBuilder::createFRem(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFRem(LHS, RHS);
}

// Atomic operations.
Value *LLVMCodeBuilder::createAtomicAdd(Value *Addr, Value *Val) {
  auto Op = Val->getType()->isFloatingPointTy() ? AtomicRMWInst::FAdd
                                                : AtomicRMWInst::Add;

  return PImpl->IRB.CreateAtomicRMW(Op, Addr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent,
                                    SyncScope::SingleThread);
}

Value *LLVMCodeBuilder::createAtomicSub(Value *Addr, Value *Val) {
  auto Op = Val->getType()->isFloatingPointTy() ? AtomicRMWInst::FSub
                                                : AtomicRMWInst::Sub;

  return PImpl->IRB.CreateAtomicRMW(Op, Addr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent,
                                    SyncScope::SingleThread);
}

Value *LLVMCodeBuilder::createAtomicMax(Value *Addr, Value *Val) {
  auto Op = Val->getType()->isFloatingPointTy() ? AtomicRMWInst::FMax
                                                : AtomicRMWInst::Max;

  return PImpl->IRB.CreateAtomicRMW(Op, Addr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent,
                                    SyncScope::SingleThread);
}

Value *LLVMCodeBuilder::createAtomicMin(Value *Addr, Value *Val) {
  auto Op = Val->getType()->isFloatingPointTy() ? AtomicRMWInst::FMin
                                                : AtomicRMWInst::Min;

  return PImpl->IRB.CreateAtomicRMW(Op, Addr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent,
                                    SyncScope::SingleThread);
}

// Comparison operations.
Value *LLVMCodeBuilder::createICmpEQ(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpEQ(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpNE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpNE(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpSLT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpSLT(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpSGT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpSGT(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpSGE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpSGE(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpSLE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpSLE(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpUGT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpUGT(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpUGE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpUGE(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpULT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpULT(LHS, RHS);
}
Value *LLVMCodeBuilder::createICmpULE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpULE(LHS, RHS);
}
Value *LLVMCodeBuilder::createFCmpOEQ(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOEQ(LHS, RHS);
}
Value *LLVMCodeBuilder::createFCmpONE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpONE(LHS, RHS);
}
Value *LLVMCodeBuilder::createFCmpOLT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOLT(LHS, RHS);
}
Value *LLVMCodeBuilder::createFCmpOLE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOLE(LHS, RHS);
}
Value *LLVMCodeBuilder::createFCmpOGT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOGT(LHS, RHS);
}
Value *LLVMCodeBuilder::createFCmpOGE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOGE(LHS, RHS);
}
Value *LLVMCodeBuilder::createFCmpULT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpULT(LHS, RHS);
}
Value *LLVMCodeBuilder::createFCmpULE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpULE(LHS, RHS);
}

// Logical operations.
Value *LLVMCodeBuilder::createAnd(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateAnd(LHS, RHS);
}
Value *LLVMCodeBuilder::createOr(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateOr(LHS, RHS);
}
Value *LLVMCodeBuilder::createXor(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateXor(LHS, RHS);
}
Value *LLVMCodeBuilder::createNot(Value *Val) {
  return PImpl->IRB.CreateNot(Val);
}

// Load/Store operations.
Value *LLVMCodeBuilder::createLoad(Type *Ty, Value *Ptr,
                                   const std::string &Name) {
  return PImpl->IRB.CreateLoad(Ty, Ptr, Name);
}

void LLVMCodeBuilder::createStore(Value *Val, Value *Ptr) {
  PImpl->IRB.CreateStore(Val, Ptr);
}

// Control flow operations.
void LLVMCodeBuilder::createBr(BasicBlock *Dest) { PImpl->IRB.CreateBr(Dest); }

void LLVMCodeBuilder::createCondBr(Value *Cond, BasicBlock *True,
                                   BasicBlock *False) {
  PImpl->IRB.CreateCondBr(Cond, True, False);
}

void LLVMCodeBuilder::createRetVoid() {
  auto *CurBB = PImpl->IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    reportFatalError("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  PImpl->IRB.CreateRetVoid();

  TermI->eraseFromParent();
}

void LLVMCodeBuilder::createRet(Value *V) {
  auto *CurBB = PImpl->IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    reportFatalError("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  PImpl->IRB.CreateRet(V);

  TermI->eraseFromParent();
}

// Cast operations.
Value *LLVMCodeBuilder::createIntCast(Value *V, Type *DestTy, bool IsSigned) {
  return PImpl->IRB.CreateIntCast(V, DestTy, IsSigned);
}

Value *LLVMCodeBuilder::createFPCast(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateFPCast(V, DestTy);
}

Value *LLVMCodeBuilder::createSIToFP(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateSIToFP(V, DestTy);
}
Value *LLVMCodeBuilder::createUIToFP(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateUIToFP(V, DestTy);
}
Value *LLVMCodeBuilder::createFPToSI(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateFPToSI(V, DestTy);
}
Value *LLVMCodeBuilder::createFPToUI(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateFPToUI(V, DestTy);
}
Value *LLVMCodeBuilder::createBitCast(Value *V, Type *DestTy) {
  if (V->getType() == DestTy)
    return V;

  return PImpl->IRB.CreateBitCast(V, DestTy);
}

Value *LLVMCodeBuilder::createZExt(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateZExt(V, DestTy);
}

// Constant creation.
Value *LLVMCodeBuilder::getConstantInt(Type *Ty, uint64_t Val) {
  return ConstantInt::get(Ty, Val);
}
Value *LLVMCodeBuilder::getConstantFP(Type *Ty, double Val) {
  return ConstantFP::get(Ty, Val);
}

// GEP operations.
Value *LLVMCodeBuilder::createInBoundsGEP(Type *Ty, Value *Ptr,
                                          const std::vector<Value *> IdxList,
                                          const std::string &Name) {
  return PImpl->IRB.CreateInBoundsGEP(Ty, Ptr, IdxList, Name);
}
Value *LLVMCodeBuilder::createConstInBoundsGEP1_64(Type *Ty, Value *Ptr,
                                                   size_t Idx) {
  return PImpl->IRB.CreateConstInBoundsGEP1_64(Ty, Ptr, Idx);
}

Value *LLVMCodeBuilder::createConstInBoundsGEP2_64(Type *Ty, Value *Ptr,
                                                   size_t Idx0, size_t Idx1) {
  return PImpl->IRB.CreateConstInBoundsGEP2_64(Ty, Ptr, Idx0, Idx1);
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

unsigned LLVMCodeBuilder::getAddressSpaceFromValue(Value *PtrVal) {
  return getAddressSpace(PtrVal->getType());
}

bool LLVMCodeBuilder::isIntegerTy(Type *Ty) { return Ty->isIntegerTy(); }
bool LLVMCodeBuilder::isFloatingPointTy(Type *Ty) {
  return Ty->isFloatingPointTy();
}

// Call operations.
Value *LLVMCodeBuilder::createCall(const std::string &FName, Type *RetTy,
                                   const std::vector<Type *> &ArgTys,
                                   const std::vector<Value *> &Args) {
  Module *M = F.getParent();
  FunctionType *FnTy = FunctionType::get(RetTy, ArgTys, false);
  FunctionCallee Callee = M->getOrInsertFunction(FName, FnTy);
  return PImpl->IRB.CreateCall(Callee, Args);
}

Value *LLVMCodeBuilder::createCall(const std::string &FName, Type *RetTy) {
  Module *M = F.getParent();
  FunctionType *FnTy = FunctionType::get(RetTy, {}, false);
  FunctionCallee Callee = M->getOrInsertFunction(FName, FnTy);
  return PImpl->IRB.CreateCall(Callee);
}

// Alloca/array emission.
AllocaInst *LLVMCodeBuilder::emitAlloca(Type *Ty, const std::string &Name,
                                        AddressSpace AS) {
  auto SaveIP = PImpl->IRB.saveIP();
  auto AllocaIP =
      IRBuilderBase::InsertPoint(&F.getEntryBlock(), F.getEntryBlock().begin());
  PImpl->IRB.restoreIP(AllocaIP);
  auto *Alloca =
      PImpl->IRB.CreateAlloca(Ty, static_cast<unsigned>(AS), nullptr, Name);

  PImpl->IRB.restoreIP(SaveIP);
  return Alloca;
}

Value *LLVMCodeBuilder::emitArrayCreate(Type *Ty, AddressSpace AT,
                                        const std::string &Name) {
  if (!Ty || !Ty->isArrayTy())
    reportFatalError("Expected LLVM ArrayType for emitArrayCreate");

  auto *ArrTy = cast<ArrayType>(Ty);

  switch (AT) {
  case AddressSpace::SHARED:
  case AddressSpace::GLOBAL: {
    Module *M = F.getParent();
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
    reportFatalError("Constant arrays are not supported");
  default:
    reportFatalError("Unsupported AddressSpace");
  }
}

} // namespace proteus
