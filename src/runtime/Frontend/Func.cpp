#include "proteus/Frontend/Func.h"
#include "proteus/Frontend/LLVMCodeBuilder.h"
#include "proteus/Frontend/LoopUnroller.h"
#include "proteus/JitFrontend.h"

namespace proteus {

FuncBase::FuncBase(JitModule &J, CodeBuilder &CBParam, const std::string &Name,
                   IRType RetTy, const std::vector<IRType> &ArgTys)
    : J(J), Name(Name), CB(&CBParam) {
  Func = CBParam.addFunction(Name, RetTy, ArgTys);
}

FuncBase::~FuncBase() = default;

void FuncBase::setName(const std::string &NewName) {
  Name = NewName;
  CB->setFunctionName(Func, Name);
}

IRValue *FuncBase::getArg(size_t Idx) { return CB->getArg(Func, Idx); }

#if defined(PROTEUS_ENABLE_CUDA) || defined(PROTEUS_ENABLE_HIP)
void FuncBase::setKernel() { CB->setKernel(Func); }

void FuncBase::setLaunchBoundsForKernel(int MaxThreadsPerBlock,
                                        int MinBlocksPerSM) {
  CB->setLaunchBoundsForKernel(Func, MaxThreadsPerBlock, MinBlocksPerSM);
}
#endif

// Delegating methods to CodeBuilder.
CodeBuilder &FuncBase::getCodeBuilder() { return *CB; }

void FuncBase::beginFunction(const char *File, int Line) {
  CB->beginFunction(Func, File, Line);
}

void FuncBase::endFunction() { CB->endFunction(); }

IRFunction *FuncBase::getFunction() { return Func; }

void FuncBase::captureForLoopLatch() {
  auto &LCB = static_cast<LLVMCodeBuilder &>(*CB);
  llvm::BasicBlock *BodyBB = LCB.getInsertBlock();
  PendingLatchBB = LCB.getUniqueSuccessor(BodyBB);
}

void FuncBase::attachLoopUnrollMetadata(LoopUnroller &U) {
  if (U.isEnabled() && PendingLatchBB)
    U.attachMetadata(static_cast<llvm::BasicBlock *>(PendingLatchBB));
}

void FuncBase::beginIf(const Var<bool> &CondVar, const char *File, int Line) {
  IRValue *Cond = CondVar.loadValue();
  CB->beginIf(Cond, File, Line);
}

void FuncBase::endIf() { CB->endIf(); }

void FuncBase::endFor() { CB->endFor(); }

void FuncBase::endWhile() { CB->endWhile(); }
} // namespace proteus
