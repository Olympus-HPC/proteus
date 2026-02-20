#ifndef PROTEUS_CLONING_H
#define PROTEUS_CLONING_H

#include "proteus/Config.h"
#include "proteus/Debug.h"
#include "proteus/Error.h"
#include "proteus/Logger.h"

#include <llvm/Analysis/CallGraph.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace proteus {

using namespace llvm;

inline std::unique_ptr<Module> cloneKernelFromModule(Module &M, StringRef Name,
                                                     CallGraph &CG) {
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
    reportFatalError("Expected function " + Name);

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

  if (Config::get().ProteusDebugOutput) {
    Logger::logfile(Name.str() + ".mini.ll", *KernelModuleTmp);
    if (verifyModule(*KernelModuleTmp, &errs()))
      reportFatalError("Broken mini-module found, JIT compilation aborted!");
  }

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

  DefMaps buildDefMaps(ArrayRef<std::reference_wrapper<Module>> Mods) {
    DefMaps SymbolMaps;
    for (Module &M : Mods) {
      for (Function &F : M.functions())
        if (!F.isDeclaration())
          SymbolMaps.FuncDefs[F.getName()] = &F;
      for (GlobalVariable &G : M.globals())
        if (G.hasInitializer() || G.hasExternalLinkage())
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
    } else if (auto *GA = dyn_cast<GlobalAlias>(G)) {
      auto *GVA = dyn_cast<GlobalValue>(GA->getAliasee()->stripPointerCasts());
      if (!GVA) {
        SmallVector<char> ErrMsg;
        raw_svector_ostream OS{ErrMsg};
        G->print(OS);
        reportFatalError("Expected aliasee to be a global value: " + ErrMsg);
      }
      ResolvedGV = GA;
    } else {
      reportFatalError("Unsupported global value: " + toString(*G));
    }

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
        // If the function has a personality function, resolve it as it needs to
        // be declared in the cloned module.
        if (F->hasPersonalityFn()) {
          if (auto *PersGV = dyn_cast<GlobalValue>(
                  F->getPersonalityFn()->stripPointerCasts())) {
            resolveGV(Defs, PersGV, WorkList, Found);
          }
        }

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
      } else if (auto *GA = dyn_cast<GlobalAlias>(GV)) {
        scanConstant(GA->getAliasee(), Defs, WorkList, Found);
      } else {
        reportFatalError("Unsupported global value: " + toString(*GV));
      }
    }

    return Found;
  }

  std::unique_ptr<Module> cloneClosure(
      Module &M, LLVMContext &Ctx,
      SmallPtrSetImpl<GlobalValue *> const &Reachable,
      function_ref<bool(const GlobalValue *)> ShouldCloneDefinition = nullptr) {
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
      } else if (auto *GA = dyn_cast<GlobalAlias>(GV)) {
        auto *NGA = GlobalAlias::create(GA->getValueType(),
                                        GA->getAddressSpace(), GA->getLinkage(),
                                        GA->getName(), ModuleOut.get());
        NGA->copyAttributesFrom(GA);
        NGA->setVisibility(GA->getVisibility());
        VMap[GA] = NGA;
      } else {
        reportFatalError("Unsupported global value: " + toString(*GV));
      }
    }

    // Clone function bodies and global variable initializers.
    for (GlobalValue *GV : Reachable) {
      // Check if the ShouldCloneDefinition callback exists and call to exclude
      // or include the definition of this GV.
      if (ShouldCloneDefinition)
        if (!ShouldCloneDefinition(GV))
          continue;

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
      } else if (auto *GA = dyn_cast<GlobalAlias>(GV)) {
        auto *Aliasee = GA->getAliasee();
        auto *NGA = cast<GlobalAlias>(VMap[GA]);
        NGA->setAliasee(MapValue(Aliasee, VMap));
      } else {
        reportFatalError("Unsupported global value: " + toString(*GV));
      }
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

    if (Config::get().ProteusDebugOutput) {
      if (verifyModule(*ModuleOut, &errs()))
        reportFatalError(
            "Broken cross-module clone found, JIT compilation aborted!");
    }
    return ModuleOut;
  }
};

inline std::unique_ptr<Module> cloneKernelFromModules(
    ArrayRef<std::reference_wrapper<Module>> Mods, StringRef EntryName,
    function_ref<bool(const GlobalValue *)> ShouldCloneDefinition = nullptr) {
  auto Cloner = LinkingCloner();
  LinkingCloner::DefMaps Defs = Cloner.buildDefMaps(Mods);

  // Find the entry function and its module.
  Function *EntryF = nullptr;
  Module *EntryM = nullptr;
  for (Module &M : Mods) {
    if ((EntryF = M.getFunction(EntryName)) && !EntryF->isDeclaration()) {
      EntryM = &M;
      break;
    }
  }
  if (!EntryF)
    reportFatalError("Expected non-null entry function");

  // Compute the transitive closure starting from the entry function.
  SmallVector<Function *> ToVisit{EntryF};
  SmallPtrSet<Function *, 32> VisitSet{EntryF};
  SmallPtrSet<GlobalValue *, 32> Reachable;

  // External global variable addresses can be captured and passed as
  // kernel argument. Prepopulate reachable with these values.
  for (Module &M : Mods) {
    for (GlobalVariable &G : M.globals())
      if (G.hasExternalLinkage())
        Reachable.insert(&G);
  }

  while (!ToVisit.empty()) {
    auto *F = ToVisit.pop_back_val();
    // Due to lazy parsing, make sure the function is materialized before
    // traversing it.
    if (auto E = F->materialize())
      reportFatalError("Failed to materialize: " + toString(std::move(E)));

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
  auto KernelModule = Cloner.cloneClosure(*EntryM, EntryF->getContext(),
                                          Reachable, ShouldCloneDefinition);

  return KernelModule;
}

} // namespace proteus

#endif
