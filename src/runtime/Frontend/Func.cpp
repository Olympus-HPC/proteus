#include "proteus/Frontend/Func.h"
#include "proteus/Frontend/LLVMCodeBuilder.h"
#include "proteus/JitFrontend.h"
#include "proteus/impl/CoreLLVMDevice.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

namespace proteus {

FuncBase::FuncBase(JitModule &J, LLVMCodeBuilder &CBParam,
                   const std::string &Name, Type *RetTy,
                   const std::vector<Type *> &ArgTys)
    : J(J), Name(Name), CB(&CBParam) {
  LLVMFunc = CBParam.addFunction(Name, RetTy, ArgTys);
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

Value *FuncBase::getConstantInt(Type *Ty, uint64_t Val) {
  return CB->getConstantInt(Ty, Val);
}
Value *FuncBase::getConstantFP(Type *Ty, double Val) {
  return CB->getConstantFP(Ty, Val);
}

void FuncBase::beginFunction(const char *File, int Line) {
  CB->beginFunction(*LLVMFunc, File, Line);
}

void FuncBase::endFunction() { CB->endFunction(); }

Function *FuncBase::getFunction() { return LLVMFunc; }

void FuncBase::beginIf(const Var<bool> &CondVar, const char *File, int Line) {
  Value *Cond = CondVar.loadValue();
  CB->beginIf(Cond, File, Line);
}

void FuncBase::endIf() { CB->endIf(); }

void FuncBase::endFor() { CB->endFor(); }

void FuncBase::endWhile() { CB->endWhile(); }
} // namespace proteus
