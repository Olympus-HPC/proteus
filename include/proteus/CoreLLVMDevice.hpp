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

inline void specializeIR(
    Module &M, StringRef FnName, StringRef Suffix, dim3 &BlockDim,
    dim3 &GridDim, const SmallVector<int32_t> &RCIndices,
    const SmallVector<RuntimeConstant> &RCVec,
    const SmallVector<std::pair<std::string, StringRef>> LambdaCalleeInfo,
    bool SpecializeArgs, bool SpecializeDims, bool SpecializeLaunchBounds) {
  Function *F = M.getFunction(FnName);

  assert(F && "Expected non-null function!");
  // Replace argument uses with runtime constants.
  if (SpecializeArgs)
    TransformArgumentSpecialization::transform(M, *F, RCIndices, RCVec);

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

  PROTEUS_DBG(Logger::logs("proteus") << "=== JIT Module\n"
                                      << M << "=== End of JIT Module\n");

  F->setName(FnName + Suffix);

  if (SpecializeLaunchBounds)
    setLaunchBoundsForKernel(M, *F, GridDim.x * GridDim.y * GridDim.z,
                             BlockDim.x * BlockDim.y * BlockDim.z);

  runCleanupPassPipeline(M);
}

inline std::unique_ptr<Module> cloneKernelFromModule(Module &M, LLVMContext &C,
                                                     const std::string &Name) {
#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "Full module \n" << M << "\n";
#endif
  auto KernelModule = std::make_unique<Module>("JitModule", C);
  KernelModule->setSourceFileName(M.getSourceFileName());
  KernelModule->setDataLayout(M.getDataLayout());
  KernelModule->setTargetTriple(M.getTargetTriple());
  KernelModule->setModuleInlineAsm(M.getModuleInlineAsm());
#if LLVM_VERSION_MAJOR >= 18
  KernelModule->IsNewDbgInfoFormat = M.IsNewDbgInfoFormat;
#endif

  auto KernelFunction = M.getFunction(Name);
  if (!KernelFunction)
    PROTEUS_FATAL_ERROR("Expected function " + Name);

  SmallPtrSet<Function *, 8> ReachableFunctions;
  SmallPtrSet<GlobalVariable *, 16> ReachableGlobals;
  SmallPtrSet<Function *, 8> ReachableDeclarations;
  SmallVector<Function *, 8> ToVisit;
  ReachableFunctions.insert(KernelFunction);
  ToVisit.push_back(KernelFunction);
  CallGraphWrapperPass CG;
  CG.runOnModule(M);
  while (!ToVisit.empty()) {
    Function *VisitF = ToVisit.pop_back_val();
    CallGraphNode *CGNode = CG[VisitF];

    for (const auto &Callee : *CGNode) {
      Function *CalleeF = Callee.second->getFunction();
      if (!CalleeF)
        continue;
      if (CalleeF->isDeclaration()) {
        ReachableDeclarations.insert(CalleeF);
        continue;
      }
      if (ReachableFunctions.contains(CalleeF))
        continue;
      ReachableFunctions.insert(CalleeF);
      ToVisit.push_back(CalleeF);
    }
  }

  auto ProcessInstruction = [&](GlobalVariable &GV, const Instruction *I) {
    const Function *ParentF = I->getParent()->getParent();
    if (ReachableFunctions.contains(ParentF))
      ReachableGlobals.insert(&GV);
  };

  for (auto &GV : M.globals()) {
    for (const User *Usr : GV.users()) {
      const Instruction *I = dyn_cast<Instruction>(Usr);

      if (I) {
        ProcessInstruction(GV, I);
      } else {
        for (const User *NextUser : Usr->users()) {
          I = dyn_cast<Instruction>(NextUser);
          if (I)
            ProcessInstruction(GV, I);
        }
      }
    }
  }

  ValueToValueMapTy VMap;

  for (auto *GV : ReachableGlobals) {
    // We will set the initializer later, after VMap has been populated.
    GlobalVariable *NewGV =
        new GlobalVariable(*KernelModule, GV->getValueType(), GV->isConstant(),
                           GV->getLinkage(), nullptr, GV->getName(), nullptr,
                           GV->getThreadLocalMode(), GV->getAddressSpace());
    NewGV->copyAttributesFrom(GV);
    VMap[GV] = NewGV;
  }

  for (auto *F : ReachableFunctions) {
    auto *NewFunction = Function::Create(F->getFunctionType(), F->getLinkage(),
                                         F->getAddressSpace(), F->getName(),
                                         KernelModule.get());
    NewFunction->copyAttributesFrom(F);
    VMap[F] = NewFunction;
  }

  for (auto *F : ReachableDeclarations) {
    auto *NewFunction = Function::Create(F->getFunctionType(), F->getLinkage(),
                                         F->getAddressSpace(), F->getName(),
                                         KernelModule.get());
    NewFunction->copyAttributesFrom(F);
    NewFunction->setLinkage(GlobalValue::ExternalLinkage);
    VMap[F] = NewFunction;
  }

  for (GlobalVariable *GV : ReachableGlobals) {
    GlobalVariable *NewGV = cast<GlobalVariable>(VMap[GV]);
    NewGV->setInitializer(MapValue(GV->getInitializer(), VMap));
  }

  for (auto *F : ReachableFunctions) {
    SmallVector<ReturnInst *, 8> Returns;
    auto *NewFunction = dyn_cast<Function>(VMap[F]);
    Function::arg_iterator DestI = NewFunction->arg_begin();
    for (const Argument &I : F->args())
      if (VMap.count(&I) == 0) {
        DestI->setName(I.getName());
        VMap[&I] = &*DestI++;
      }
    llvm::CloneFunctionInto(NewFunction, F, VMap,
                            CloneFunctionChangeType::DifferentModule, Returns);
  }

  // Copy annotations from M into KernelModule now that VMap has been populated.
  const std::string AnnotationsToCopy[] = {"llvm.annotations",
                                           "nvvm.annotations", "nvvmir.version",
                                           "llvm.module.flags"};
  for (auto &AnnotationName : AnnotationsToCopy) {
    NamedMDNode *Annotations = M.getNamedMetadata(AnnotationName);
    if (!Annotations)
      continue;

    auto *KernelAnnotations =
        KernelModule->getOrInsertNamedMetadata(AnnotationName);
    for (unsigned I = 0, E = Annotations->getNumOperands(); I < E; ++I) {
      auto *Annotation = Annotations->getOperand(I);
      bool ShouldClone = true;
      // Skip if the operands of an MDNode refer to non-existing,
      // unreachable global values.
      for (auto &Operand : Annotation->operands()) {
        Metadata *MD = Operand.get();
        auto *CMD = dyn_cast<ConstantAsMetadata>(MD);
        if (!CMD)
          continue;

        auto *GV = dyn_cast<GlobalValue>(CMD->getValue());
        if (!GV)
          continue;

        if (!VMap.count(GV)) {
          ShouldClone = false;
          break;
        }
      }

      if (!ShouldClone)
        continue;

      KernelAnnotations->addOperand(
          MapMetadata(Annotations->getOperand(I), VMap));
    }
  }

  if (verifyModule(*KernelModule, &errs()))
    PROTEUS_FATAL_ERROR("Broken mini-module found, JIT compilation aborted!");

#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "Mini-module \n" << *KernelModule << "\n";
#endif

  return std::move(KernelModule);
}

} // namespace proteus

#endif

#endif
