#ifndef PROTEUS_CORE_LLVM_DEVICE_HPP
#define PROTEUS_CORE_LLVM_DEVICE_HPP

#if PROTEUS_ENABLE_HIP
#include "proteus/CoreLLVMHIP.hpp"
#endif

#if PROTEUS_ENABLE_CUDA
#include "proteus/CoreLLVMCUDA.hpp"
#endif

#if defined(PROTEUS_ENABLE_HIP) || defined(PROTEUS_ENABLE_CUDA)

#include <llvm/ADT/StringSet.h>
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
  Timer T;
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

  F->setName(FnName + Suffix);

  if (SpecializeLaunchBounds)
    setLaunchBoundsForKernel(M, *F, GridDim.x * GridDim.y * GridDim.z,
                             BlockDim.x * BlockDim.y * BlockDim.z);

  runCleanupPassPipeline(M);

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                       << "specializeIR " << T.elapsed() << " ms\n");
  PROTEUS_DBG(Logger::logfile(FnName.str() + ".specialized.ll", M));
}

inline std::unique_ptr<Module>
cloneKernelFromModule(Module &M, const std::string &Name, CallGraph &CG) {
  auto KernelModuleTmp = std::make_unique<Module>("JitModule", M.getContext());
  KernelModuleTmp->setSourceFileName(M.getSourceFileName());
  KernelModuleTmp->setDataLayout(M.getDataLayout());
  KernelModuleTmp->setTargetTriple(M.getTargetTriple());
  KernelModuleTmp->setModuleInlineAsm(M.getModuleInlineAsm());
#if LLVM_VERSION_MAJOR >= 18
  KernelModuleTmp->IsNewDbgInfoFormat = M.IsNewDbgInfoFormat;
#endif

  auto *KernelFunction = M.getFunction(Name);
  if (!KernelFunction)
    PROTEUS_FATAL_ERROR("Expected function " + Name);

  SmallPtrSet<Function *, 8> ReachableFunctions;
  SmallPtrSet<GlobalVariable *, 16> ReachableGlobals;
  SmallPtrSet<Function *, 8> ReachableDeclarations;
  SmallVector<Function *, 8> ToVisit;
  ReachableFunctions.insert(KernelFunction);
  ToVisit.push_back(KernelFunction);
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
    const Function *ParentF = I->getFunction();
    if (ReachableFunctions.contains(ParentF))
      ReachableGlobals.insert(&GV);
  };

  for (auto &GV : M.globals()) {
    for (const User *Usr : GV.users()) {
      const Instruction *I = dyn_cast<Instruction>(Usr);

      if (I) {
        ProcessInstruction(GV, I);
        continue;
      }

      // We follow non-instructions users to process them if those are
      // instructions.
      // TODO: We may need to follow deeper than just users of user and also
      // expand to non-instruction users.
      for (const User *NextUser : Usr->users()) {
        I = dyn_cast<Instruction>(NextUser);
        if (!I)
          continue;

        ProcessInstruction(GV, I);
      }
    }
  }

  ValueToValueMapTy VMap;

  for (auto *GV : ReachableGlobals) {
    // We will set the initializer later, after VMap has been populated.
    GlobalVariable *NewGV = new GlobalVariable(
        *KernelModuleTmp, GV->getValueType(), GV->isConstant(),
        GV->getLinkage(), nullptr, GV->getName(), nullptr,
        GV->getThreadLocalMode(), GV->getAddressSpace());
    NewGV->copyAttributesFrom(GV);
    VMap[GV] = NewGV;
  }

  for (auto *F : ReachableFunctions) {
    auto *NewFunction = Function::Create(F->getFunctionType(), F->getLinkage(),
                                         F->getAddressSpace(), F->getName(),
                                         KernelModuleTmp.get());
    NewFunction->copyAttributesFrom(F);
    VMap[F] = NewFunction;
  }

  for (auto *F : ReachableDeclarations) {
    auto *NewFunction = Function::Create(F->getFunctionType(), F->getLinkage(),
                                         F->getAddressSpace(), F->getName(),
                                         KernelModuleTmp.get());
    NewFunction->copyAttributesFrom(F);
    NewFunction->setLinkage(GlobalValue::ExternalLinkage);
    VMap[F] = NewFunction;
  }

  for (GlobalVariable *GV : ReachableGlobals) {
    if (GV->hasInitializer()) {
      GlobalVariable *NewGV = cast<GlobalVariable>(VMap[GV]);
      NewGV->setInitializer(MapValue(GV->getInitializer(), VMap));
    }
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

  // Copy annotations from M into KernelModuleTmp now that VMap has been
  // populated.
  const std::string MetadataToCopy[] = {"llvm.annotations", "nvvm.annotations",
                                        "nvvmir.version", "llvm.module.flags"};
  for (auto &MetadataName : MetadataToCopy) {
    NamedMDNode *NamedMD = M.getNamedMetadata(MetadataName);
    if (!NamedMD)
      continue;

    auto *NewNamedMD = KernelModuleTmp->getOrInsertNamedMetadata(MetadataName);
    for (unsigned I = 0, E = NamedMD->getNumOperands(); I < E; ++I) {
      MDNode *MDEntry = NamedMD->getOperand(I);
      bool ShouldClone = true;
      // Skip if the operands of an MDNode refer to non-existing,
      // unreachable global values.
      for (auto &Operand : MDEntry->operands()) {
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

      NewNamedMD->addOperand(MapMetadata(MDEntry, VMap));
    }
  }

#if PROTEUS_ENABLE_DEBUG
  Logger::logfile(Name + ".mini.ll", *KernelModuleTmp);
  if (verifyModule(*KernelModuleTmp, &errs()))
    PROTEUS_FATAL_ERROR("Broken mini-module found, JIT compilation aborted!");
#endif

  return KernelModuleTmp;
}

struct LinkingCloner {
  // Definitions maps that map symbol name to GlobalValue.
  struct DefMaps {
    StringMap<Function *> FuncDefs;
    StringMap<GlobalVariable *> GlobDefs;
  };

  // Stores declaration prototype and GVs that map to it.
  struct FuncDeclInfo {
    FunctionType *FuncTy;
    GlobalValue::LinkageTypes Linkage;
    unsigned int AddrSpace;
    AttributeList Attributes;
    SmallPtrSet<GlobalValue *, 32> GVs;
  };

  struct GlobDeclInfo {
    Type *ValueType;
    bool IsConstant;
    GlobalValue::LinkageTypes Linkage;
    Constant *Initializer;
    GlobalValue::ThreadLocalMode TLM;
    unsigned int AddrSpace;
    AttributeSet Attributes;
    SmallPtrSet<GlobalValue *, 32> GVs;
  };

  // Maps a resolved GV to cross-module GV references.
  DenseMap<GlobalValue *, SmallVector<GlobalValue *>> ResolvedMap;
  // Stores declaration info by symbol name to clone and update references.
  StringMap<FuncDeclInfo> FuncDecls;
  StringMap<GlobDeclInfo> GlobDecls;

  DefMaps buildDefMaps(ArrayRef<std::unique_ptr<Module>> Mods) {
    DefMaps SymbolMaps;
    for (auto &M : Mods) {
      for (Function &F : M->functions())
        if (!F.isDeclaration())
          SymbolMaps.FuncDefs[F.getName()] = &F;
      for (GlobalVariable &G : M->globals())
        if (G.hasInitializer())
          SymbolMaps.GlobDefs[G.getName()] = &G;
    }
    return SymbolMaps;
  }

  // Resolve GlobalValue to its definition across modules or fallback to the
  // declartion if not found (e.g., intrinsics). Add to WorkList if unseen.
  void resolveGV(const DefMaps &Defs, GlobalValue *G,
                 SmallVector<GlobalValue *> &WorkList,
                 SmallPtrSetImpl<GlobalValue *> &Found) {
    GlobalValue *ResolvedGV = nullptr;
    if (auto *F = dyn_cast<Function>(G)) {
      if (!F->isDeclaration())
        ResolvedGV = F;
      else if (auto *D = Defs.FuncDefs.lookup(F->getName()))
        ResolvedGV = D;
      else {
        if (FuncDecls.contains(F->getName()))
          FuncDecls[F->getName()].GVs.insert(G);
        else
          FuncDecls[F->getName()] = {F->getFunctionType(),
                                     F->getLinkage(),
                                     F->getAddressSpace(),
                                     F->getAttributes(),
                                     {G}};
      }
    } else if (auto *GV = dyn_cast<GlobalVariable>(G)) {
      if (GV->hasInitializer())
        ResolvedGV = GV;
      else if (auto *D = Defs.GlobDefs.lookup(GV->getName()))
        ResolvedGV = D;
      else {
        if (GlobDecls.contains(GV->getName()))
          GlobDecls[GV->getName()].GVs.insert(G);
        else
          GlobDecls[GV->getName()] = {
              GV->getValueType(),       GV->isConstant(),
              GV->getLinkage(),         nullptr,
              GV->getThreadLocalMode(), GV->getAddressSpace(),
              GV->getAttributes(),      {G}};
      }
    } else
      PROTEUS_FATAL_ERROR("Unsupported global value");

    if (ResolvedGV && Found.insert(ResolvedGV).second) {
      WorkList.push_back(ResolvedGV);
      ResolvedMap[ResolvedGV].push_back(G);
    }
  }

  void scanConstant(Constant *C, const DefMaps &Defs,
                    SmallVector<GlobalValue *> &WorkList,
                    SmallPtrSetImpl<GlobalValue *> &Found) {
    // If this is a constant expression (e.g.,  bitcast), unwrap and scan its
    // operand.
    if (auto *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->isCast())
        scanConstant(CE->getOperand(0), Defs, WorkList, Found);
    }

    // If C itself is a global value resolve its definition.
    if (auto *G = dyn_cast<GlobalValue>(C))
      resolveGV(Defs, G, WorkList, Found);

    // Recurse into any operand constants.
    for (unsigned I = 0, E = C->getNumOperands(); I != E; ++I)
      if (auto *OpC = dyn_cast<Constant>(C->getOperand(I)))
        scanConstant(OpC, Defs, WorkList, Found);
  }

  SmallPtrSet<GlobalValue *, 32> findTransitiveClosure(Function *Entry,
                                                       const DefMaps &Defs) {
    SmallPtrSet<GlobalValue *, 32> Found;
    SmallVector<GlobalValue *> WorkList;

    // Seed with the entry function.
    resolveGV(Defs, Entry, WorkList, Found);

    // Process DFS work-list.
    while (!WorkList.empty()) {
      auto *GV = WorkList.pop_back_val();

      if (auto *F = dyn_cast<Function>(GV)) {
        for (auto &BB : *F) {
          for (auto &I : BB) {
            // Add direct calls to other functions.
            if (auto *CB = dyn_cast<CallBase>(&I))
              if (Function *Callee = CB->getCalledFunction())
                resolveGV(Defs, Callee, WorkList, Found);

            // Scan GlobalValue operands or Constants.
            for (Use &U : I.operands()) {
              Value *Op = U.get()->stripPointerCasts();
              if (auto *GVOp = dyn_cast<GlobalValue>(Op)) {
                resolveGV(Defs, GVOp, WorkList, Found);
              } else if (auto *C = dyn_cast<Constant>(Op)) {
                scanConstant(C, Defs, WorkList, Found);
              }
            }
          }
        }
      } else if (auto *GVar = dyn_cast<GlobalVariable>(GV)) {
        if (auto *Init = GVar->getInitializer())
          scanConstant(Init, Defs, WorkList, Found);
      } else {
        PROTEUS_FATAL_ERROR("Unsupported global value");
      }
    }

    return Found;
  }

  std::unique_ptr<Module>
  cloneClosure(Module &M, LLVMContext &Ctx,
               SmallPtrSetImpl<GlobalValue *> const &Reachable) {
    auto ModuleOut =
        std::make_unique<Module>(M.getName().str() + ".closure.clone", Ctx);
    ModuleOut->setSourceFileName(M.getSourceFileName());
    ModuleOut->setDataLayout(M.getDataLayout());
    ModuleOut->setTargetTriple(M.getTargetTriple());
    ModuleOut->setModuleInlineAsm(M.getModuleInlineAsm());
#if LLVM_VERSION_MAJOR >= 18
    ModuleOut->IsNewDbgInfoFormat = M.IsNewDbgInfoFormat;
#endif

    ValueToValueMapTy VMap;

    // Emit the function declarations.
    for (auto &D : FuncDecls) {
      StringRef FuncName = D.getKey();
      FuncDeclInfo &FuncInfo = D.getValue();
      Function *NF =
          Function::Create(FuncInfo.FuncTy, FuncInfo.Linkage,
                           FuncInfo.AddrSpace, FuncName, ModuleOut.get());
      NF->setAttributes(FuncInfo.Attributes);
      for (auto *GV : FuncInfo.GVs)
        VMap[GV] = NF;
    }

    // Emit the global variable declarations.
    for (auto &D : GlobDecls) {
      StringRef GVName = D.getKey();
      GlobDeclInfo &GlobInfo = D.getValue();
      auto *NG = new GlobalVariable(
          *ModuleOut, GlobInfo.ValueType, GlobInfo.IsConstant, GlobInfo.Linkage,
          nullptr, GVName, nullptr, GlobInfo.TLM, GlobInfo.AddrSpace);
      NG->setAttributes(GlobInfo.Attributes);
      for (auto *GV : GlobInfo.GVs)
        VMap[GV] = NG;
    }

    // Create unpopulated declarations.
    for (GlobalValue *GV : Reachable) {
      if (auto *F = dyn_cast<Function>(GV)) {
        Function *NF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                        F->getAddressSpace(), F->getName(),
                                        ModuleOut.get());
        NF->copyAttributesFrom(F);
        VMap[F] = NF;

        for (auto *DeclGV : ResolvedMap[F])
          VMap[DeclGV] = NF;
      } else if (auto *GVar = dyn_cast<GlobalVariable>(GV)) {
        auto *NG = new GlobalVariable(
            *ModuleOut, GV->getValueType(), GVar->isConstant(),
            GV->getLinkage(), nullptr, GV->getName(), nullptr,
            GV->getThreadLocalMode(), GV->getAddressSpace());
        NG->copyAttributesFrom(GVar);
        VMap[GVar] = NG;

        for (auto *DeclGV : ResolvedMap[GVar])
          VMap[DeclGV] = NG;
      } else
        PROTEUS_FATAL_ERROR("Unsupported global value");
    }

    // Clone function bodies and global variable initializers.
    for (GlobalValue *GV : Reachable) {
      if (auto *F = dyn_cast<Function>(GV)) {
        Function *NF = cast<Function>(VMap[F]);
        SmallVector<ReturnInst *, 8> Returns;
        Function::arg_iterator DestI = NF->arg_begin();
        for (const Argument &I : F->args())
          if (VMap.count(&I) == 0) {
            DestI->setName(I.getName());
            VMap[&I] = &*DestI++;
          }

        CloneFunctionInto(NF, F, VMap,
                          /*Changes=*/CloneFunctionChangeType::DifferentModule,
                          Returns);
      } else if (auto *GVar = dyn_cast<GlobalVariable>(GV)) {

        auto *NG = cast<GlobalVariable>(VMap[GVar]);
        if (GVar->hasInitializer())
          NG->setInitializer(MapValue(GVar->getInitializer(), VMap));
      } else
        PROTEUS_FATAL_ERROR("Unsupported global value");
    }

    // Copy annotations from the entry module M into KernelModuleTmp now that
    // VMap has been populated.
    const std::string MetadataToCopy[] = {"llvm.annotations",
                                          "nvvm.annotations", "nvvmir.version",
                                          "llvm.module.flags"};
    for (auto &MetadataName : MetadataToCopy) {
      NamedMDNode *NamedMD = M.getNamedMetadata(MetadataName);
      if (!NamedMD)
        continue;

      auto *NewNamedMD = ModuleOut->getOrInsertNamedMetadata(MetadataName);
      for (unsigned I = 0, E = NamedMD->getNumOperands(); I < E; ++I) {
        MDNode *MDEntry = NamedMD->getOperand(I);
        bool ShouldClone = true;
        // Skip if the operands of an MDNode refer to non-existing,
        // unreachable global values.
        for (auto &Operand : MDEntry->operands()) {
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

        NewNamedMD->addOperand(MapMetadata(MDEntry, VMap));
      }
    }

#if PROTEUS_ENABLE_DEBUG
    if (verifyModule(*ModuleOut, &errs()))
      PROTEUS_FATAL_ERROR(
          "Broken cross-module clone found, JIT compilation aborted!");
#endif
    return ModuleOut;
  }
};

inline std::unique_ptr<Module>
cloneKernelFromModules(ArrayRef<std::unique_ptr<Module>> Mods,
                       StringRef EntryName) {
  auto Cloner = LinkingCloner();
  LinkingCloner::DefMaps Defs = Cloner.buildDefMaps(Mods);

  // Find the entry function and its module.
  Function *EntryF = nullptr;
  Module *EntryM = nullptr;
  for (auto &M : Mods) {
    if ((EntryF = M->getFunction(EntryName)) && !EntryF->isDeclaration()) {
      EntryM = M.get();
      break;
    }
  }
  if (!EntryF)
    PROTEUS_FATAL_ERROR("Expected non-null entry function");

  // Compute the transitive closure starting from the entry function.
  SmallVector<Function *> ToVisit{EntryF};
  SmallPtrSet<Function *, 32> VisitSet{EntryF};
  SmallPtrSet<GlobalValue *, 32> Reachable;
  while (!ToVisit.empty()) {
    auto *F = ToVisit.pop_back_val();
    // Due to lazy parsing, make sure the function is materialized before
    // traversing it.
    if (auto E = F->materialize())
      PROTEUS_FATAL_ERROR("Failed to materialize: " + toString(std::move(E)));

    auto ThisReachable = Cloner.findTransitiveClosure(F, Defs);
    for (auto *GV : ThisReachable) {
      Reachable.insert(GV);
      if (auto *ThisF = dyn_cast<Function>(GV)) {
        if (!VisitSet.contains(ThisF)) {
          VisitSet.insert(ThisF);
          ToVisit.push_back(ThisF);
        }
      }
    }
  }

  // Clone closure in new module.
  auto KernelModule =
      Cloner.cloneClosure(*EntryM, EntryF->getContext(), Reachable);

  return KernelModule;
}

} // namespace proteus

#endif

#endif
