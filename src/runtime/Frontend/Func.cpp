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
LLVMCodeBuilder &FuncBase::getCodeBuilder() { return *CB; }

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

void FuncBase::eraseTerminator(BasicBlock *BB) { CB->eraseTerminator(BB); }

BasicBlock *FuncBase::getUniqueSuccessor(BasicBlock *BB) {
  return CB->getUniqueSuccessor(BB);
}

void FuncBase::pushScope(const char *File, const int Line, ScopeKind Kind,
                         BasicBlock *NextBlock) {
  CB->pushScope(File, Line, Kind, NextBlock);
}

Value *FuncBase::getConstantInt(Type *Ty, uint64_t Val) {
  return CB->getConstantInt(Ty, Val);
}
Value *FuncBase::getConstantFP(Type *Ty, double Val) {
  return CB->getConstantFP(Ty, Val);
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

void FuncBase::beginFunction(const char *File, int Line) {
  CB->beginFunction(File, Line);
}

void FuncBase::endFunction() { CB->endFunction(); }

Function *FuncBase::getFunction() { return &CB->getFunction(); }

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
