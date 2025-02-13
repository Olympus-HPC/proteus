#ifndef PROTEUS_CORE_LLVM_DEVICE_HPP
#define PROTEUS_CORE_LLVM_DEVICE_HPP

#endif

#if PROTEUS_ENABLE_HIP
#include "proteus/CoreLLVMHIP.hpp"
#endif

#if PROTEUS_ENABLE_CUDA
#include "proteus/CoreLLVMCUDA.hpp"
#endif

namespace proteus {

static inline void setKernelDims(Module &M, dim3 &GridDim, dim3 &BlockDim) {
  auto ReplaceIntrinsicDim = [&](ArrayRef<StringRef> IntrinsicNames,
                                 uint32_t DimValue) {
    auto CollectCallUsers = [](Function &F) {
      SmallVector<CallInst *> CallUsers;
      for (auto *User : F.users()) {
        auto *Call = dyn_cast<CallInst>(User);
        if (!Call)
          continue;
        CallUsers.push_back(Call);
      }

      return CallUsers;
    };

    for (auto IntrinsicName : IntrinsicNames) {

      Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction)
        continue;

      for (auto *Call : CollectCallUsers(*IntrinsicFunction)) {
        Value *ConstantValue =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), DimValue);
        Call->replaceAllUsesWith(ConstantValue);
        Call->eraseFromParent();
      }
    }
  };

  ReplaceIntrinsicDim(detail::gridDimXFnName(), GridDim.x);
  ReplaceIntrinsicDim(detail::gridDimYFnName(), GridDim.y);
  ReplaceIntrinsicDim(detail::gridDimZFnName(), GridDim.z);

  ReplaceIntrinsicDim(detail::blockDimXFnName(), BlockDim.x);
  ReplaceIntrinsicDim(detail::blockDimYFnName(), BlockDim.y);
  ReplaceIntrinsicDim(detail::blockDimZFnName(), BlockDim.z);

  auto InsertAssume = [&](ArrayRef<StringRef> IntrinsicNames, int DimValue) {
    for (auto IntrinsicName : IntrinsicNames) {
      Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction || IntrinsicFunction->use_empty())
        continue;

      // Iterate over all uses of the intrinsic.
      for (auto *U : IntrinsicFunction->users()) {
        auto *Call = dyn_cast<CallInst>(U);
        if (!Call)
          continue;

        // Insert the llvm.assume intrinsic.
        IRBuilder<> Builder(Call->getNextNode());
        Value *Bound = ConstantInt::get(Call->getType(), DimValue);
        Value *Cmp = Builder.CreateICmpULT(Call, Bound);

        Function *AssumeIntrinsic =
            Intrinsic::getDeclaration(&M, Intrinsic::assume);
        Builder.CreateCall(AssumeIntrinsic, Cmp);
      }
    }
  };

  // Inform LLVM about the range of possible values of threadIdx.*.
  InsertAssume(detail::threadIdxXFnName(), BlockDim.x);
  InsertAssume(detail::threadIdxYFnName(), BlockDim.y);
  InsertAssume(detail::threadIdxZFnName(), BlockDim.z);

  // Inform LLVdetailut the range of possible values of blockIdx.*.
  InsertAssume(detail::blockIdxXFnName(), GridDim.x);
  InsertAssume(detail::blockIdxYFnName(), GridDim.y);
  InsertAssume(detail::blockIdxZFnName(), GridDim.z);
}

} // namespace proteus
