#include "proteus/Frontend/Func.h"
#include "proteus/Frontend/LLVMCodeBuilder.h"
#include "proteus/JitFrontend.h"
#include "proteus/impl/CoreLLVMDevice.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {

FuncBase::FuncBase(JitModule &J, const std::string &Name, Type *RetTy,
                   const std::vector<Type *> &ArgTys)
    : J(J), Name(Name) {
  auto FC = J.getModule().getOrInsertFunction(
      Name, FunctionType::get(RetTy, ArgTys, false));
  Function *F = dyn_cast<Function>(FC.getCallee());
  if (!F)
    reportFatalError("Expected LLVM Function");
  BasicBlock::Create(F->getContext(), "entry", F);
  CB = std::make_unique<LLVMCodeBuilder>(*F);
}

FuncBase::~FuncBase() = default;

LLVMContext &FuncBase::getContext() { return CB->getContext(); }

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
  auto *Alloca = CB->emitAlloca(AllocaTy, Name);

  return std::make_unique<ScalarStorage>(Alloca, CB->getIRBuilder());
}

std::unique_ptr<PointerStorage>
FuncBase::createPointerStorage(const std::string &Name, Type *AllocaTy,
                               Type *ElemTy) {
  auto *Alloca = CB->emitAlloca(AllocaTy, Name);

  return std::make_unique<PointerStorage>(Alloca, CB->getIRBuilder(), ElemTy);
}

std::unique_ptr<ArrayStorage>
FuncBase::createArrayStorage(const std::string &Name, AddressSpace AS,
                             Type *Ty) {
  ArrayType *ArrTy = cast<ArrayType>(Ty);
  Value *BasePointer = CB->emitArrayCreate(ArrTy, AS, Name);

  return std::make_unique<ArrayStorage>(BasePointer, CB->getIRBuilder(), ArrTy);
}

// Delegating methods to LLVMCodeBuilder.
void FuncBase::setInsertPoint(BasicBlock *BB) { CB->setInsertPoint(BB); }
void FuncBase::setInsertPointBegin(BasicBlock *BB) {
  CB->setInsertPointBegin(BB);
}
void FuncBase::setInsertPointAtEntry() { CB->setInsertPointAtEntry(); }
void FuncBase::clearInsertPoint() { CB->clearInsertPoint(); }
BasicBlock *FuncBase::getInsertBlock() { return CB->getInsertBlock(); }

std::tuple<BasicBlock *, BasicBlock *> FuncBase::splitCurrentBlock() {
  return CB->splitCurrentBlock();
}

BasicBlock *FuncBase::createBasicBlock(const std::string &Name,
                                       BasicBlock *InsertBefore) {
  return CB->createBasicBlock(Name, InsertBefore);
}

void FuncBase::eraseTerminator(BasicBlock *BB) { CB->eraseTerminator(BB); }

BasicBlock *FuncBase::getUniqueSuccessor(BasicBlock *BB) {
  return CB->getUniqueSuccessor(BB);
}

void FuncBase::pushScope(const char *File, const int Line, ScopeKind Kind,
                         BasicBlock *NextBlock) {
  CB->pushScope(File, Line, Kind, NextBlock);
}

// Arithmetic operations.
Value *FuncBase::createAdd(Value *LHS, Value *RHS) {
  return CB->createAdd(LHS, RHS);
}
Value *FuncBase::createFAdd(Value *LHS, Value *RHS) {
  return CB->createFAdd(LHS, RHS);
}
Value *FuncBase::createSub(Value *LHS, Value *RHS) {
  return CB->createSub(LHS, RHS);
}
Value *FuncBase::createFSub(Value *LHS, Value *RHS) {
  return CB->createFSub(LHS, RHS);
}
Value *FuncBase::createMul(Value *LHS, Value *RHS) {
  return CB->createMul(LHS, RHS);
}
Value *FuncBase::createFMul(Value *LHS, Value *RHS) {
  return CB->createFMul(LHS, RHS);
}
Value *FuncBase::createUDiv(Value *LHS, Value *RHS) {
  return CB->createUDiv(LHS, RHS);
}
Value *FuncBase::createSDiv(Value *LHS, Value *RHS) {
  return CB->createSDiv(LHS, RHS);
}
Value *FuncBase::createFDiv(Value *LHS, Value *RHS) {
  return CB->createFDiv(LHS, RHS);
}
Value *FuncBase::createURem(Value *LHS, Value *RHS) {
  return CB->createURem(LHS, RHS);
}
Value *FuncBase::createSRem(Value *LHS, Value *RHS) {
  return CB->createSRem(LHS, RHS);
}
Value *FuncBase::createFRem(Value *LHS, Value *RHS) {
  return CB->createFRem(LHS, RHS);
}

// Atomic operations.
Value *FuncBase::createAtomicAdd(Value *Addr, Value *Val) {
  return CB->createAtomicAdd(Addr, Val);
}

Value *FuncBase::createAtomicSub(Value *Addr, Value *Val) {
  return CB->createAtomicSub(Addr, Val);
}

Value *FuncBase::createAtomicMax(Value *Addr, Value *Val) {
  return CB->createAtomicMax(Addr, Val);
}

Value *FuncBase::createAtomicMin(Value *Addr, Value *Val) {
  return CB->createAtomicMin(Addr, Val);
}

// Comparison operations.
Value *FuncBase::createICmpEQ(Value *LHS, Value *RHS) {
  return CB->createICmpEQ(LHS, RHS);
}
Value *FuncBase::createICmpNE(Value *LHS, Value *RHS) {
  return CB->createICmpNE(LHS, RHS);
}
Value *FuncBase::createICmpSLT(Value *LHS, Value *RHS) {
  return CB->createICmpSLT(LHS, RHS);
}
Value *FuncBase::createICmpSGT(Value *LHS, Value *RHS) {
  return CB->createICmpSGT(LHS, RHS);
}
Value *FuncBase::createICmpSGE(Value *LHS, Value *RHS) {
  return CB->createICmpSGE(LHS, RHS);
}
Value *FuncBase::createICmpSLE(Value *LHS, Value *RHS) {
  return CB->createICmpSLE(LHS, RHS);
}
Value *FuncBase::createICmpUGT(Value *LHS, Value *RHS) {
  return CB->createICmpUGT(LHS, RHS);
}
Value *FuncBase::createICmpUGE(Value *LHS, Value *RHS) {
  return CB->createICmpUGE(LHS, RHS);
}
Value *FuncBase::createICmpULT(Value *LHS, Value *RHS) {
  return CB->createICmpULT(LHS, RHS);
}
Value *FuncBase::createICmpULE(Value *LHS, Value *RHS) {
  return CB->createICmpULE(LHS, RHS);
}
Value *FuncBase::createFCmpOEQ(Value *LHS, Value *RHS) {
  return CB->createFCmpOEQ(LHS, RHS);
}
Value *FuncBase::createFCmpONE(Value *LHS, Value *RHS) {
  return CB->createFCmpONE(LHS, RHS);
}
Value *FuncBase::createFCmpOLT(Value *LHS, Value *RHS) {
  return CB->createFCmpOLT(LHS, RHS);
}
Value *FuncBase::createFCmpOLE(Value *LHS, Value *RHS) {
  return CB->createFCmpOLE(LHS, RHS);
}
Value *FuncBase::createFCmpOGT(Value *LHS, Value *RHS) {
  return CB->createFCmpOGT(LHS, RHS);
}
Value *FuncBase::createFCmpOGE(Value *LHS, Value *RHS) {
  return CB->createFCmpOGE(LHS, RHS);
}
Value *FuncBase::createFCmpULT(Value *LHS, Value *RHS) {
  return CB->createFCmpULT(LHS, RHS);
}
Value *FuncBase::createFCmpULE(Value *LHS, Value *RHS) {
  return CB->createFCmpULE(LHS, RHS);
}

// Cast operations.
Value *FuncBase::createIntCast(Value *V, Type *DestTy, bool IsSigned) {
  return CB->createIntCast(V, DestTy, IsSigned);
}

Value *FuncBase::createFPCast(Value *V, Type *DestTy) {
  return CB->createFPCast(V, DestTy);
}

Value *FuncBase::createSIToFP(Value *V, Type *DestTy) {
  return CB->createSIToFP(V, DestTy);
}
Value *FuncBase::createUIToFP(Value *V, Type *DestTy) {
  return CB->createUIToFP(V, DestTy);
}
Value *FuncBase::createFPToSI(Value *V, Type *DestTy) {
  return CB->createFPToSI(V, DestTy);
}
Value *FuncBase::createFPToUI(Value *V, Type *DestTy) {
  return CB->createFPToUI(V, DestTy);
}
Value *FuncBase::createBitCast(Value *V, Type *DestTy) {
  return CB->createBitCast(V, DestTy);
}

Value *FuncBase::getConstantInt(Type *Ty, uint64_t Val) {
  return CB->getConstantInt(Ty, Val);
}
Value *FuncBase::getConstantFP(Type *Ty, double Val) {
  return CB->getConstantFP(Ty, Val);
}

// Control flow operations.
void FuncBase::createBr(llvm::BasicBlock *Dest) { CB->createBr(Dest); }
void FuncBase::createCondBr(llvm::Value *Cond, llvm::BasicBlock *True,
                            llvm::BasicBlock *False) {
  CB->createCondBr(Cond, True, False);
}
void FuncBase::createRetVoid() { CB->createRetVoid(); }

void FuncBase::createRet(llvm::Value *V) { CB->createRet(V); }

// Logical operations.
Value *FuncBase::createAnd(Value *LHS, Value *RHS) {
  return CB->createAnd(LHS, RHS);
}
Value *FuncBase::createOr(Value *LHS, Value *RHS) {
  return CB->createOr(LHS, RHS);
}
Value *FuncBase::createXor(Value *LHS, Value *RHS) {
  return CB->createXor(LHS, RHS);
}
Value *FuncBase::createNot(Value *Val) { return CB->createNot(Val); }

// Load/Store operations.
Value *FuncBase::createLoad(Type *Ty, Value *Ptr, const std::string &Name) {
  return CB->createLoad(Ty, Ptr, Name);
}

void FuncBase::createStore(Value *Val, Value *Ptr) {
  CB->createStore(Val, Ptr);
}

// Conversion operations.
Value *FuncBase::createZExt(Value *V, Type *DestTy) {
  return CB->createZExt(V, DestTy);
}

// GEP operations.
Value *FuncBase::createInBoundsGEP(Type *Ty, Value *Ptr,
                                   const std::vector<Value *> IdxList,
                                   const std::string &Name) {
  return CB->createInBoundsGEP(Ty, Ptr, IdxList, Name);
}
Value *FuncBase::createConstInBoundsGEP1_64(Type *Ty, Value *Ptr, size_t Idx) {
  return CB->createConstInBoundsGEP1_64(Ty, Ptr, Idx);
}

Value *FuncBase::createConstInBoundsGEP2_64(Type *Ty, Value *Ptr, size_t Idx0,
                                            size_t Idx1) {
  return CB->createConstInBoundsGEP2_64(Ty, Ptr, Idx0, Idx1);
}

// Type accessors.
unsigned FuncBase::getAddressSpace(Type *Ty) { return CB->getAddressSpace(Ty); }

unsigned FuncBase::getAddressSpaceFromValue(Value *PtrVal) {
  return CB->getAddressSpaceFromValue(PtrVal);
}

Type *FuncBase::getPointerType(Type *ElemTy, unsigned AS) {
  return CB->getPointerType(ElemTy, AS);
}
Type *FuncBase::getPointerTypeUnqual(Type *ElemTy) {
  return CB->getPointerTypeUnqual(ElemTy);
}
Type *FuncBase::getInt16Ty() { return CB->getInt16Ty(); }
Type *FuncBase::getInt32Ty() { return CB->getInt32Ty(); }
Type *FuncBase::getInt64Ty() { return CB->getInt64Ty(); }
Type *FuncBase::getFloatTy() { return CB->getFloatTy(); }
bool FuncBase::isIntegerTy(Type *Ty) { return CB->isIntegerTy(Ty); }
bool FuncBase::isFloatingPointTy(Type *Ty) { return CB->isFloatingPointTy(Ty); }

// Call operations.
Value *FuncBase::createCall(const std::string &FName, Type *RetTy,
                            const std::vector<Type *> &ArgTys,
                            const std::vector<Value *> &Args) {
  return CB->createCall(FName, RetTy, ArgTys, Args);
}
Value *FuncBase::createCall(const std::string &FName, Type *RetTy) {
  return CB->createCall(FName, RetTy);
}

void FuncBase::beginFunction(const char *File, int Line) {
  CB->beginFunction(File, Line);
}

void FuncBase::endFunction() { CB->endFunction(); }

Function *FuncBase::getFunction() { return &CB->getFunction(); }

AllocaInst *FuncBase::emitAlloca(Type *Ty, const std::string &Name,
                                 AddressSpace AS) {
  return CB->emitAlloca(Ty, Name, AS);
}

Value *FuncBase::emitArrayCreate(Type *Ty, AddressSpace AT,
                                 const std::string &Name) {
  return CB->emitArrayCreate(Ty, AT, Name);
}

void FuncBase::beginIf(const Var<bool> &CondVar, const char *File, int Line) {
  Value *Cond = CondVar.loadValue();
  CB->beginIf(Cond, File, Line);
}

void FuncBase::endIf() { CB->endIf(); }

void FuncBase::endFor() { CB->endFor(); }

void FuncBase::endWhile() { CB->endWhile(); }
} // namespace proteus
