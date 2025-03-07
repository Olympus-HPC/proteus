#ifndef PROTEUS_CORE_LLVM_DEVICE_HPP
#define PROTEUS_CORE_LLVM_DEVICE_HPP

#if PROTEUS_ENABLE_HIP
#include "proteus/CoreLLVMHIP.hpp"
#endif

#if PROTEUS_ENABLE_CUDA
#include "proteus/CoreLLVMCUDA.hpp"
#endif

#if defined(PROTEUS_ENABLE_HIP) || defined(PROTEUS_ENABLE_CUDA)

#include <llvm/IR/ReplaceConstant.h>
#include <llvm/Object/ELFObjectFile.h>

#include "proteus/CoreDevice.hpp"
#include "proteus/LambdaRegistry.hpp"
#include "proteus/TransformArgumentSpecialization.hpp"
#include "proteus/TransformLambdaSpecialization.hpp"

namespace proteus {

inline void setKernelDims(Module &M, dim3 &GridDim, dim3 &BlockDim) {
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

inline void replaceGlobalVariablesWithPointers(
    Module &M,
    const std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
  // Re-link globals to fixed addresses provided by registered
  // variables.
  for (auto RegisterVar : VarNameToDevPtr) {
    auto &VarName = RegisterVar.first;
    auto *GV = M.getNamedGlobal(VarName);
    // Skip linking if the GV does not exist in the module.
    if (!GV)
      continue;

    // This will convert constant users of GV to instructions so that we can
    // replace with the GV ptr.
    convertUsersOfConstantsToInstructions({GV});

    Constant *Addr =
        ConstantInt::get(Type::getInt64Ty(M.getContext()), 0xDEADBEEFDEADBEEF);
    auto *CE = ConstantExpr::getIntToPtr(Addr, GV->getType()->getPointerTo());
    auto *GVarPtr = new GlobalVariable(
        M, GV->getType()->getPointerTo(), false, GlobalValue::ExternalLinkage,
        CE, GV->getName() + "$ptr", nullptr, GV->getThreadLocalMode(),
        GV->getAddressSpace(), true);

    SmallVector<Instruction *> ToReplace;
    for (auto *User : GV->users()) {
      auto *Inst = dyn_cast<Instruction>(User);
      if (!Inst)
        PROTEUS_FATAL_ERROR("Expected Instruction User for GV");

      ToReplace.push_back(Inst);
    }

    for (auto *Inst : ToReplace) {
      IRBuilder Builder{Inst};
      auto *Load = Builder.CreateLoad(GV->getType(), GVarPtr);
      Inst->replaceUsesOfWith(GV, Load);
    }
  }
}

inline void relinkGlobalsObject(
    MemoryBufferRef Object,
    const std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
  Expected<object::ELF64LEObjectFile> DeviceElfOrErr =
      object::ELF64LEObjectFile::create(Object);
  if (DeviceElfOrErr.takeError())
    PROTEUS_FATAL_ERROR("Cannot create the device elf");
  auto &DeviceElf = *DeviceElfOrErr;

  for (auto &[GlobalName, DevPtr] : VarNameToDevPtr) {
    for (auto &Symbol : DeviceElf.symbols()) {
      auto SymbolNameOrErr = Symbol.getName();
      if (!SymbolNameOrErr)
        continue;
      auto SymbolName = *SymbolNameOrErr;

      if (!SymbolName.equals(GlobalName + "$ptr"))
        continue;

      Expected<uint64_t> ValueOrErr = Symbol.getValue();
      if (!ValueOrErr)
        PROTEUS_FATAL_ERROR("Expected symbol value");
      uint64_t SymbolValue = *ValueOrErr;

      // Get the section containing the symbol
      auto SectionOrErr = Symbol.getSection();
      if (!SectionOrErr)
        PROTEUS_FATAL_ERROR("Cannot retrieve section");
      const auto &Section = *SectionOrErr;
      if (Section == DeviceElf.section_end())
        PROTEUS_FATAL_ERROR("Expected sybmol in section");

      // Get the section's address and data
      Expected<StringRef> SectionDataOrErr = Section->getContents();
      if (!SectionDataOrErr)
        PROTEUS_FATAL_ERROR("Error retrieving section data");
      StringRef SectionData = *SectionDataOrErr;

      // Calculate offset within the section
      uint64_t SectionAddr = Section->getAddress();
      uint64_t Offset = SymbolValue - SectionAddr;
      if (Offset >= SectionData.size())
        PROTEUS_FATAL_ERROR("Expected offset within section size");

      uint64_t *Data = (uint64_t *)(SectionData.data() + Offset);
      *Data = reinterpret_cast<uint64_t>(DevPtr);
      break;
    }
  }
}

inline void specializeIR(Module &M, StringRef FnName, StringRef Suffix,
                         dim3 &BlockDim, dim3 &GridDim,
                         const SmallVector<int32_t> &RCIndices,
                         const SmallVector<RuntimeConstant> &RCVec,
                         bool SpecializeArgs, bool SpecializeDims,
                         bool SpecializeLaunchBounds) {
  Function *F = M.getFunction(FnName);

  assert(F && "Expected non-null function!");
  // Replace argument uses with runtime constants.
  if (SpecializeArgs)
    TransformArgumentSpecialization::transform(M, *F, RCIndices, RCVec);

  if (!LambdaRegistry::instance().empty()) {
    PROTEUS_DBG(Logger::logs("proteus")
                << "=== LAMBDA MATCHING\n"
                << "F trigger " << F->getName() << " -> "
                << demangle(F->getName().str()) << "\n");
    for (auto &F : M.getFunctionList()) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << " Trying F " << demangle(F.getName().str()) << "\n ");
      if (auto OptionalMapIt =
              LambdaRegistry::instance().matchJitVariableMap(F.getName())) {
        auto &RCVec = OptionalMapIt.value()->getSecond();
        TransformLambdaSpecialization::transform(M, F, RCVec);
        LambdaRegistry::instance().erase(OptionalMapIt.value());
        PROTEUS_DBG(Logger::logs("proteus") << "Found match!\n");
        break;
      }
    }
    PROTEUS_DBG(Logger::logs("proteus") << "=== END OF MATCHING\n");
  }

  // Replace uses of blockDim.* and gridDim.* with constants.
  if (SpecializeDims)
    setKernelDims(M, GridDim, BlockDim);

  PROTEUS_DBG(Logger::logs("proteus") << "=== JIT Module\n"
                                      << M << "=== End of JIT Module\n");

  F->setName(FnName + Suffix);

  if (SpecializeLaunchBounds)
    setLaunchBoundsForKernel(M, *F, GridDim.x * GridDim.y * GridDim.z,
                             BlockDim.x * BlockDim.y * BlockDim.z);

  runCleanupPassPipeline(M);
}

} // namespace proteus

#endif

#endif
