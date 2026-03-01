#include "proteus/Frontend/Func.h"
#include "proteus/Frontend/LLVMCodeBuilder.h"
#include "proteus/JitFrontend.h"
#include "proteus/impl/CoreLLVMDevice.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace proteus {

FuncBase::FuncBase(JitModule &J, LLVMCodeBuilder &CBParam,
                   const std::string &Name, IRType RetTy,
                   const std::vector<IRType> &ArgTys)
    : J(J), Name(Name), CB(&CBParam) {
  LLVMFunc = CBParam.addFunction(Name, RetTy, ArgTys);
}

FuncBase::~FuncBase() = default;

void FuncBase::setName(const std::string &NewName) {
  Name = NewName;
  Function *F = getFunction();
  F->setName(Name);
}

Value *FuncBase::getArg(size_t Idx) {
  Function *F = getFunction();
  return F->getArg(Idx);
}

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
void FuncBase::setKernel() { CB->setKernel(*getFunction()); }

void FuncBase::setLaunchBoundsForKernel(int MaxThreadsPerBlock,
                                        int MinBlocksPerSM) {
  CB->setLaunchBoundsForKernel(*getFunction(), MaxThreadsPerBlock,
                               MinBlocksPerSM);
}
#endif

// Delegating methods to LLVMCodeBuilder.
LLVMCodeBuilder &FuncBase::getCodeBuilder() { return *CB; }

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
