#include "proteus/Frontend/Func.hpp"

#include "proteus/CoreLLVMDevice.hpp"
#include "proteus/JitFrontend.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {

struct FuncBase::Impl {
  FunctionCallee FC;
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

  Impl(FunctionCallee FC) : FC(FC), IRB{FC.getCallee()->getContext()} {
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

FuncBase::FuncBase(JitModule &J, const std::string &Name, Type *RetTy,
                   const std::vector<Type *> &ArgTys)
    : J(J), Name(Name),
      PImpl{std::make_unique<Impl>(J.getModule().getOrInsertFunction(
          Name, FunctionType::get(RetTy, ArgTys, false)))} {
  BasicBlock::Create(getContext(), "entry", getFunction());
}

FuncBase::~FuncBase() = default;

LLVMContext &FuncBase::getContext() { return getFunction()->getContext(); }

void FuncBase::setName(const std::string &NewName) {
  Name = NewName;
  Function *F = getFunction();
  F->setName(Name);
}

TargetModelType FuncBase::getTargetModel() const { return J.getTargetModel(); }

Value *FuncBase::getArg(size_t Idx) {
  Function *F = getFunction();
  return F->getArg(Idx);
}

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
void FuncBase::setKernel() {
  LLVMContext &Ctx = J.getModule().getContext();
  switch (getTargetModel()) {
  case TargetModelType::CUDA: {
    NamedMDNode *MD =
        J.getModule().getOrInsertNamedMetadata("nvvm.annotations");

    Metadata *MDVals[] = {
        ConstantAsMetadata::get(getFunction()), MDString::get(Ctx, "kernel"),
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1))};
    // Append metadata to nvvm.annotations.
    MD->addOperand(MDNode::get(Ctx, MDVals));

    // Add a function attribute for the kernel.
    getFunction()->addFnAttr(Attribute::get(Ctx, "kernel"));
    return;
  }
  case TargetModelType::HIP:
    getFunction()->setCallingConv(CallingConv::AMDGPU_KERNEL);
    return;
  case TargetModelType::HOST:
    reportFatalError("Host does not support setKernel");
  default:
    reportFatalError("Unsupported target " + J.getTargetTriple() +
                     " for setKernel");
  }
}

void FuncBase::setLaunchBoundsForKernel(int MaxThreadsPerBlock,
                                        int MinBlocksPerSM) {
  proteus::setLaunchBoundsForKernel(*getFunction(), MaxThreadsPerBlock,
                                    MinBlocksPerSM);
}
#endif

std::unique_ptr<ScalarStorage>
FuncBase::createScalarStorage(const std::string &Name, Type *AllocaTy) {
  auto *Alloca = emitAlloca(AllocaTy, Name);

  return std::make_unique<ScalarStorage>(Alloca, PImpl->IRB);
}

std::unique_ptr<PointerStorage>
FuncBase::createPointerStorage(const std::string &Name, Type *AllocaTy,
                               Type *ElemTy) {
  auto *Alloca = emitAlloca(AllocaTy, Name);

  return std::make_unique<PointerStorage>(Alloca, PImpl->IRB, ElemTy);
}

std::unique_ptr<ArrayStorage>
FuncBase::createArrrayStorage(const std::string &Name, AddressSpace AS,
                              Type *Ty) {
  ArrayType *ArrTy = cast<ArrayType>(Ty);
  Value *BasePointer = emitArrayCreate(ArrTy, AS, Name);

  return std::make_unique<ArrayStorage>(BasePointer, PImpl->IRB, ArrTy);
}

// Insert point management.
void FuncBase::setInsertPoint(BasicBlock *BB) { PImpl->IRB.SetInsertPoint(BB); }
void FuncBase::setInsertPointBegin(BasicBlock *BB) {
  PImpl->IP = IRBuilderBase::InsertPoint(BB, BB->begin());
  PImpl->IRB.restoreIP(PImpl->IP);
}
void FuncBase::setInsertPointAtEntry() {
  Function *F = getFunction();

  auto &EntryBB = F->getEntryBlock();
  PImpl->IP = IRBuilderBase::InsertPoint(&EntryBB, EntryBB.end());
  PImpl->IRB.restoreIP(PImpl->IP);
}
void FuncBase::clearInsertPoint() { PImpl->IRB.ClearInsertionPoint(); }
BasicBlock *FuncBase::getInsertBlock() { return PImpl->IRB.GetInsertBlock(); }

// Basic block operations.
std::tuple<BasicBlock *, BasicBlock *> FuncBase::splitCurrentBlock() {
  BasicBlock *CurBlock = PImpl->IP.getBlock();
  BasicBlock *NextBlock = CurBlock->splitBasicBlock(
      PImpl->IP.getPoint(), CurBlock->getName() + ".split");
  return {CurBlock, NextBlock};
}

BasicBlock *FuncBase::createBasicBlock(const std::string &Name,
                                       BasicBlock *InsertBefore) {
  Function *F = getFunction();
  BasicBlock *BB = BasicBlock::Create(getContext(), Name, F, InsertBefore);
  return BB;
}

void FuncBase::eraseTerminator(BasicBlock *BB) {
  if (!BB->getTerminator())
    reportFatalError("Basic block has no terminator to erase");

  BB->getTerminator()->eraseFromParent();
}

BasicBlock *FuncBase::getUniqueSuccessor(BasicBlock *BB) {
  auto *Succ = BB->getUniqueSuccessor();
  if (!Succ)
    reportFatalError("Expected unique successor for basic block " +
                     BB->getName().str());
  return Succ;
}

// Scope management.
void FuncBase::pushScope(const char *File, const int Line, ScopeKind Kind,
                         BasicBlock *NextBlock) {
  IRBuilderBase::InsertPoint ContIP{NextBlock, NextBlock->begin()};
  PImpl->Scopes.emplace_back(File, Line, Kind, ContIP);
}

// Arithmetic operations.
Value *FuncBase::createAdd(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateAdd(LHS, RHS);
}
Value *FuncBase::createFAdd(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFAdd(LHS, RHS);
}
Value *FuncBase::createSub(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateSub(LHS, RHS);
}
Value *FuncBase::createFSub(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFSub(LHS, RHS);
}
Value *FuncBase::createMul(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateMul(LHS, RHS);
}
Value *FuncBase::createFMul(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFMul(LHS, RHS);
}
Value *FuncBase::createUDiv(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateUDiv(LHS, RHS);
}
Value *FuncBase::createSDiv(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateSDiv(LHS, RHS);
}
Value *FuncBase::createFDiv(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFDiv(LHS, RHS);
}
Value *FuncBase::createURem(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateURem(LHS, RHS);
}
Value *FuncBase::createSRem(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateSRem(LHS, RHS);
}
Value *FuncBase::createFRem(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFRem(LHS, RHS);
}

// Atomic operations.
Value *FuncBase::createAtomicAdd(Value *Addr, Value *Val) {
  auto Op = Val->getType()->isFloatingPointTy() ? AtomicRMWInst::FAdd
                                                : AtomicRMWInst::Add;

  return PImpl->IRB.CreateAtomicRMW(Op, Addr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent,
                                    SyncScope::SingleThread);
}

Value *FuncBase::createAtomicSub(Value *Addr, Value *Val) {
  auto Op = Val->getType()->isFloatingPointTy() ? AtomicRMWInst::FSub
                                                : AtomicRMWInst::Sub;

  return PImpl->IRB.CreateAtomicRMW(Op, Addr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent,
                                    SyncScope::SingleThread);
}

Value *FuncBase::createAtomicMax(Value *Addr, Value *Val) {
  auto Op = Val->getType()->isFloatingPointTy() ? AtomicRMWInst::FMax
                                                : AtomicRMWInst::Max;

  return PImpl->IRB.CreateAtomicRMW(Op, Addr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent,
                                    SyncScope::SingleThread);
}

Value *FuncBase::createAtomicMin(Value *Addr, Value *Val) {
  auto Op = Val->getType()->isFloatingPointTy() ? AtomicRMWInst::FMin
                                                : AtomicRMWInst::Min;

  return PImpl->IRB.CreateAtomicRMW(Op, Addr, Val, MaybeAlign(),
                                    AtomicOrdering::SequentiallyConsistent,
                                    SyncScope::SingleThread);
}

// Comparison operations.
Value *FuncBase::createICmpEQ(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpEQ(LHS, RHS);
}
Value *FuncBase::createICmpNE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpNE(LHS, RHS);
}
Value *FuncBase::createICmpSLT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpSLT(LHS, RHS);
}
Value *FuncBase::createICmpSGT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpSGT(LHS, RHS);
}
Value *FuncBase::createICmpSGE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpSGE(LHS, RHS);
}
Value *FuncBase::createICmpSLE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpSLE(LHS, RHS);
}
Value *FuncBase::createICmpUGT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpUGT(LHS, RHS);
}
Value *FuncBase::createICmpUGE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpUGE(LHS, RHS);
}
Value *FuncBase::createICmpULT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpULT(LHS, RHS);
}
Value *FuncBase::createICmpULE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateICmpULE(LHS, RHS);
}
Value *FuncBase::createFCmpOEQ(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOEQ(LHS, RHS);
}
Value *FuncBase::createFCmpONE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpONE(LHS, RHS);
}
Value *FuncBase::createFCmpOLT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOLT(LHS, RHS);
}
Value *FuncBase::createFCmpOLE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOLE(LHS, RHS);
}
Value *FuncBase::createFCmpOGT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOGT(LHS, RHS);
}
Value *FuncBase::createFCmpOGE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpOGE(LHS, RHS);
}
Value *FuncBase::createFCmpULT(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpULT(LHS, RHS);
}
Value *FuncBase::createFCmpULE(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateFCmpULE(LHS, RHS);
}

// Cast Implementations
Value *FuncBase::createIntCast(Value *V, Type *DestTy, bool IsSigned) {
  // IRBuilder::CreateIntCast automatically handles SExt, ZExt, or Trunc
  return PImpl->IRB.CreateIntCast(V, DestTy, IsSigned);
}

Value *FuncBase::createFPCast(Value *V, Type *DestTy) {
  // IRBuilder::CreateFPCast automatically handles FPExt, FPTrunc, or BitCast
  return PImpl->IRB.CreateFPCast(V, DestTy);
}

Value *FuncBase::createSIToFP(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateSIToFP(V, DestTy);
}
Value *FuncBase::createUIToFP(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateUIToFP(V, DestTy);
}
Value *FuncBase::createFPToSI(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateFPToSI(V, DestTy);
}
Value *FuncBase::createFPToUI(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateFPToUI(V, DestTy);
}
Value *FuncBase::createBitCast(Value *V, Type *DestTy) {
  if (V->getType() == DestTy)
    return V;

  return PImpl->IRB.CreateBitCast(V, DestTy);
}

Value *FuncBase::getConstantInt(Type *Ty, uint64_t Val) {
  return ConstantInt::get(Ty, Val);
}
Value *FuncBase::getConstantFP(Type *Ty, double Val) {
  return ConstantFP::get(Ty, Val);
}

// Control flow operations.
void FuncBase::createBr(llvm::BasicBlock *Dest) { PImpl->IRB.CreateBr(Dest); }
void FuncBase::createCondBr(llvm::Value *Cond, llvm::BasicBlock *True,
                            llvm::BasicBlock *False) {
  PImpl->IRB.CreateCondBr(Cond, True, False);
}
void FuncBase::createRetVoid() {
  auto *CurBB = PImpl->IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    reportFatalError("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  PImpl->IRB.CreateRetVoid();

  TermI->eraseFromParent();
}

void FuncBase::createRet(llvm::Value *V) {
  auto *CurBB = PImpl->IP.getBlock();
  if (!CurBB->getSingleSuccessor())
    reportFatalError("Expected single successor for current block");
  auto *TermI = CurBB->getTerminator();

  PImpl->IRB.CreateRet(V);

  TermI->eraseFromParent();
}

// Logical operations.
Value *FuncBase::createAnd(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateAnd(LHS, RHS);
}
Value *FuncBase::createOr(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateOr(LHS, RHS);
}
Value *FuncBase::createXor(Value *LHS, Value *RHS) {
  return PImpl->IRB.CreateXor(LHS, RHS);
}
Value *FuncBase::createNot(Value *Val) { return PImpl->IRB.CreateNot(Val); }

// Load/Store operations.
Value *FuncBase::createLoad(Type *Ty, Value *Ptr, const std::string &Name) {
  return PImpl->IRB.CreateLoad(Ty, Ptr, Name);
}

void FuncBase::createStore(Value *Val, Value *Ptr) {
  PImpl->IRB.CreateStore(Val, Ptr);
}

// Conversion operations.
Value *FuncBase::createZExt(Value *V, Type *DestTy) {
  return PImpl->IRB.CreateZExt(V, DestTy);
}

// GEP operations.
Value *FuncBase::createInBoundsGEP(Type *Ty, Value *Ptr,
                                   const std::vector<Value *> IdxList,
                                   const std::string &Name) {
  return PImpl->IRB.CreateInBoundsGEP(Ty, Ptr, IdxList, Name);
}
Value *FuncBase::createConstInBoundsGEP1_64(Type *Ty, Value *Ptr, size_t Idx) {
  return PImpl->IRB.CreateConstInBoundsGEP1_64(Ty, Ptr, Idx);
}

Value *FuncBase::createConstInBoundsGEP2_64(Type *Ty, Value *Ptr, size_t Idx0,
                                            size_t Idx1) {
  return PImpl->IRB.CreateConstInBoundsGEP2_64(Ty, Ptr, Idx0, Idx1);
}

// Type accessors.
unsigned FuncBase::getAddressSpace(Type *Ty) {
  auto *PtrTy = dyn_cast<PointerType>(Ty);
  if (!PtrTy)
    reportFatalError("Expected LLVM PointerType for getAddressSpace");

  return PtrTy->getAddressSpace();
}
Type *FuncBase::getPointerType(Type *ElemTy, unsigned AS) {
  return PointerType::get(ElemTy, AS);
}
Type *FuncBase::getPointerTypeUnqual(Type *ElemTy) {
  return PointerType::getUnqual(ElemTy);
}
Type *FuncBase::getInt16Ty() { return PImpl->IRB.getInt16Ty(); }
Type *FuncBase::getInt32Ty() { return PImpl->IRB.getInt32Ty(); }
Type *FuncBase::getInt64Ty() { return PImpl->IRB.getInt64Ty(); }
Type *FuncBase::getFloatTy() { return PImpl->IRB.getFloatTy(); }
bool FuncBase::isIntegerTy(Type *Ty) { return Ty->isIntegerTy(); }
bool FuncBase::isFloatingPointTy(Type *Ty) { return Ty->isFloatingPointTy(); }

// Call operations.
Value *FuncBase::createCall(const std::string &FName, Type *RetTy,
                            const std::vector<Type *> &ArgTys,
                            const std::vector<Value *> &Args) {
  Module *M = getFunction()->getParent();
  FunctionType *FnTy = FunctionType::get(RetTy, ArgTys, false);
  FunctionCallee Callee = M->getOrInsertFunction(FName, FnTy);
  return PImpl->IRB.CreateCall(Callee, Args);
}

void FuncBase::beginFunction(const char *File, int Line) {
  Function *F = getFunction();
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

void FuncBase::endFunction() {
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

Function *FuncBase::getFunction() {
  Function *F = dyn_cast<Function>(PImpl->FC.getCallee());
  if (!F)
    reportFatalError("Expected LLVM Function");
  return F;
}

AllocaInst *FuncBase::emitAlloca(Type *Ty, const std::string &Name,
                                 AddressSpace AS) {
  auto SaveIP = PImpl->IRB.saveIP();
  Function *F = getFunction();
  auto AllocaIP = IRBuilderBase::InsertPoint(&F->getEntryBlock(),
                                             F->getEntryBlock().begin());
  PImpl->IRB.restoreIP(AllocaIP);
  auto *Alloca =
      PImpl->IRB.CreateAlloca(Ty, static_cast<unsigned>(AS), nullptr, Name);

  PImpl->IRB.restoreIP(SaveIP);
  return Alloca;
}

Value *FuncBase::emitArrayCreate(Type *Ty, AddressSpace AT,
                                 const std::string &Name) {
  if (!Ty || !Ty->isArrayTy())
    reportFatalError("Expected LLVM ArrayType for emitArrayCreate");

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
    reportFatalError("Constant arrays are not supported");
  default:
    reportFatalError("Unsupported AddressSpace");
  }
}

void FuncBase::beginIf(const Var<bool> &CondVar, const char *File, int Line) {
  Function *F = getFunction();
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
  {
    Value *Cond = CondVar.loadValue();
    PImpl->IRB.CreateCondBr(Cond, ThenBlock, ExitBlock);
  }

  PImpl->IRB.SetInsertPoint(ThenBlock);
  { PImpl->IRB.CreateBr(ExitBlock); }

  PImpl->IRB.SetInsertPoint(ExitBlock);
  { PImpl->IRB.CreateBr(NextBlock); }

  PImpl->IP = IRBuilderBase::InsertPoint(ThenBlock, ThenBlock->begin());
  PImpl->IRB.restoreIP(PImpl->IP);
}

void FuncBase::endIf() {
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

void FuncBase::endFor() {
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

void FuncBase::endWhile() {
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
} // namespace proteus
