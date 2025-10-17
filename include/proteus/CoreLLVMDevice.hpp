#ifndef PROTEUS_CORE_LLVM_DEVICE_HPP
#define PROTEUS_CORE_LLVM_DEVICE_HPP

#if PROTEUS_ENABLE_HIP
#include "proteus/CoreLLVMHIP.hpp"
#endif

#if PROTEUS_ENABLE_CUDA
#include "proteus/CoreLLVMCUDA.hpp"
#endif

#if defined(PROTEUS_ENABLE_HIP) || defined(PROTEUS_ENABLE_CUDA)

#include <llvm/Analysis/CallGraph.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "proteus/CoreDevice.hpp"
#include "proteus/LambdaRegistry.hpp"
#include "proteus/TransformArgumentSpecialization.hpp"
#include "proteus/TransformLambdaSpecialization.hpp"
#include "proteus/TransformSharedArray.hpp"

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

      auto TraceOut = [](Function *F, Value *C) {
        SmallString<128> S;
        raw_svector_ostream OS(S);
        OS << "[DimSpec] Replace call to " << F->getName() << " with constant "
           << *C << "\n";

        return S;
      };

      for (auto *Call : CollectCallUsers(*IntrinsicFunction)) {
        Value *ConstantValue =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), DimValue);
        Call->replaceAllUsesWith(ConstantValue);
        if (Config::get().ProteusTraceOutput >= 1)
          Logger::trace(TraceOut(IntrinsicFunction, ConstantValue));
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
}

inline void setKernelDimsAssume(Module &M, dim3 &GridDim, dim3 &BlockDim) {
  auto InsertAssume = [&](ArrayRef<StringRef> IntrinsicNames, int DimValue) {
    for (auto IntrinsicName : IntrinsicNames) {
      Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction || IntrinsicFunction->use_empty())
        continue;

      auto TraceOut = [](Function *IntrinsicF, int DimValue) {
        SmallString<128> S;
        raw_svector_ostream OS(S);
        OS << "[DimSpec] Assume " << IntrinsicF->getName() << " with "
           << DimValue << "\n";

        return S;
      };

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
        if (Config::get().ProteusTraceOutput >= 1)
          Logger::trace(TraceOut(IntrinsicFunction, DimValue));
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

    // Find all Constant users that refer to the global variable.
    SmallPtrSet<Value *, 16> ValuesToReplace;
    SmallVector<Value *> Worklist;
    // Seed with the global variable.
    Worklist.push_back(GV);
    ValuesToReplace.insert(GV);
    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      for (auto *User : V->users()) {
        if (auto *C = dyn_cast<Constant>(User)) {
          if (ValuesToReplace.insert(C).second)
            Worklist.push_back(C);

          continue;
        }

        // Skip instructions to be handled when replacing.
        if (isa<Instruction>(User))
          continue;

        PROTEUS_FATAL_ERROR(
            "Expected Instruction or Constant user for Value: " + toString(*V) +
            " , User: " + toString(*User));
      }
    }

    for (Value *V : ValuesToReplace) {
      SmallPtrSet<Instruction *, 16> Insts;
      // Find instruction users to replace value.
      for (User *U : V->users()) {
        if (auto *I = dyn_cast<Instruction>(U)) {
          Insts.insert(I);
        }
      }

      // Replace value in instructions.
      for (auto *I : Insts) {
        IRBuilder Builder{I};
        auto *Load = Builder.CreateLoad(GV->getType(), GVarPtr);
        Value *Replacement = Load;
        Type *ExpectedTy = V->getType();
        if (Load->getType() != ExpectedTy)
          Replacement =
              Builder.CreatePointerBitCastOrAddrSpaceCast(Load, ExpectedTy);

        I->replaceUsesOfWith(V, Replacement);
      }
    }
  }

  if (Config::get().ProteusDebugOutput) {
    if (verifyModule(M, &errs()))
      PROTEUS_FATAL_ERROR("Broken module found, JIT compilation aborted!");
  }
}

inline void relinkGlobalsObject(
    MemoryBufferRef Object,
    const std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
  Expected<object::ELF64LEObjectFile> DeviceElfOrErr =
      object::ELF64LEObjectFile::create(Object);
  if (auto E = DeviceElfOrErr.takeError())
    PROTEUS_FATAL_ERROR("Cannot create the device elf: " +
                        toString(std::move(E)));
  auto &DeviceElf = *DeviceElfOrErr;

  for (auto &[GlobalName, DevPtr] : VarNameToDevPtr) {
    for (auto &Symbol : DeviceElf.symbols()) {
      auto SymbolNameOrErr = Symbol.getName();
      if (!SymbolNameOrErr)
        continue;
      auto SymbolName = *SymbolNameOrErr;

      if (!(SymbolName == (GlobalName + "$ptr")))
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

inline void specializeIR(
    Module &M, StringRef FnName, StringRef Suffix, dim3 &BlockDim,
    dim3 &GridDim, ArrayRef<RuntimeConstant> RCArray,
    const SmallVector<std::pair<std::string, StringRef>> LambdaCalleeInfo,
    bool SpecializeArgs, bool SpecializeDims, bool SpecializeDimsAssume,
    bool SpecializeLaunchBounds, int MinBlocksPerSM) {
  Timer T;
  Function *F = M.getFunction(FnName);

  assert(F && "Expected non-null function!");
  // Replace argument uses with runtime constants.
  if (SpecializeArgs)
    TransformArgumentSpecialization::transform(M, *F, RCArray);

  auto &LR = LambdaRegistry::instance();
  for (auto &[FnName, LambdaType] : LambdaCalleeInfo) {
    const SmallVector<RuntimeConstant> &RCVec = LR.getJitVariables(LambdaType);
    Function *F = M.getFunction(FnName);
    if (!F)
      PROTEUS_FATAL_ERROR("Expected non-null Function");
    TransformLambdaSpecialization::transform(M, *F, RCVec);
  }

  // Run the shared array transform after any value specialization (arguments,
  // captures) to propagate any constants.
  TransformSharedArray::transform(M);

  // Replace uses of blockDim.* and gridDim.* with constants.
  if (SpecializeDims)
    setKernelDims(M, GridDim, BlockDim);
  if (SpecializeDimsAssume)
    setKernelDimsAssume(M, GridDim, BlockDim);
  F->setName(FnName + Suffix);

  if (SpecializeLaunchBounds) {
    int BlockSize = BlockDim.x * BlockDim.y * BlockDim.z;
    auto TraceOut = [](int BlockSize, int MinBlocksPerSM) {
      SmallString<128> S;
      raw_svector_ostream OS(S);
      OS << "[LaunchBoundSpec] MaxThreads " << BlockSize << " MinBlocksPerSM "
         << MinBlocksPerSM << "\n";

      return S;
    };
    if (Config::get().ProteusTraceOutput >= 1)
      Logger::trace(TraceOut(BlockSize, MinBlocksPerSM));
    setLaunchBoundsForKernel(*F, BlockSize, MinBlocksPerSM);
  }

  runCleanupPassPipeline(M);

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "specializeIR " << T.elapsed() << " ms\n");
}

} // namespace proteus

#endif

#endif
