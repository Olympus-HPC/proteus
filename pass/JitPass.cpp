//=============================================================================
// FILE:
//    JitPass.cpp
//
// DESCRIPTION:
//    Find functions annotated with "jit" plus input arguments that are
//    amenable to runtime constant propagation. Stores the IR for those
//    functions, replaces them with a stub function that calls the jit runtime
//    library to compile the IR and call the function pointer of the jit'ed
//    version.
//
// USAGE:
//    1. Legacy PM
//      opt -enable-new-pm=0 -load libJitPass.dylib -legacy-jit-pass
//      -disable-output `\`
//        <input-llvm-file>
//    2. New PM
//      opt -load-pass-plugin=libJitPass.dylib -passes="jit-pass" `\`
//        -disable-output <input-llvm-file>
//
//
// License: MIT
//=============================================================================
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include "llvm/Analysis/MemorySSA.h"

#include <iostream>

//#define ENABLE_RECURSIVE_JIT

using namespace llvm;

//-----------------------------------------------------------------------------
// JitPass implementation
//-----------------------------------------------------------------------------
// No need to expose the internals of the pass to the outside world - keep
// everything in an anonymous namespace.
namespace {

struct JitFunctionInfo {
  Function *Fn;
  std::string ModuleIR;
};

SmallVector<JitFunctionInfo, 8> JitFunctionInfoList;

void parseAnnotations(Module &M) {
  auto GlobalAnnotations = M.getNamedGlobal("llvm.global.annotations");
  if (GlobalAnnotations) {
    auto Array = cast<ConstantArray>(GlobalAnnotations->getOperand(0));
    dbgs() << "Array " << *Array << "\n";
    for (int i = 0; i < Array->getNumOperands(); i++) {
      auto Entry = cast<ConstantStruct>(Array->getOperand(i));
      dbgs() << "Entry " << *Entry << "\n";

      auto Fn = dyn_cast<Function>(Entry->getOperand(0));

      if (!Fn)
        continue;

      for (auto &JFI : JitFunctionInfoList)
        if (JFI.Fn == Fn)
          report_fatal_error("Duplicate jit annotation for Fn " + Fn->getName(),
                             false);

      dbgs() << "Function " << Fn->getName() << "\n";

      auto Annotation = cast<ConstantDataArray>(Entry->getOperand(1)->getOperand(0));

      dbgs() << "Annotation " << Annotation->getAsCString() << "\n";

      // TODO: needs CString for comparison to work, why?
      if (Annotation->getAsCString().compare("jit"))
        continue;

      JitFunctionInfo JFI;
      JFI.Fn = Fn;

      JitFunctionInfoList.push_back(JFI);
    }
  }
}

void getReachableFunctions(CallGraph &CG, Module &M, Function &F,
                           SmallPtrSetImpl<Function *> &ReachableFunctions) {
  SmallVector<Function *, 8> ToVisit;
  ToVisit.push_back(&F);
  while (!ToVisit.empty()) {
    Function *VisitF = ToVisit.pop_back_val();
    CallGraphNode *CGNode = CG[VisitF];

    for (const auto &Callee : *CGNode) {
      Function *CalleeF = Callee.second->getFunction();

      if (!CalleeF) {
        dbgs() << "Skip external node\n";
        continue;
      }

      if (CalleeF->isDeclaration()) {
        dbgs() << "Skip declaration of " << CalleeF->getName() << "\n";
        continue;
      }

      if (ReachableFunctions.contains(CalleeF)) {
        dbgs() << "Skip already visited " << CalleeF->getName() << "\n";
        continue;
      }

      dbgs() << "Found reachable " << CalleeF->getName() << " ... to visit\n";
      ReachableFunctions.insert(CalleeF);
      ToVisit.push_back(CalleeF);
    }
  }
}

// This method implements what the pass does
bool visitor(Module &M, CallGraph &CG,
             function_ref<MemorySSAAnalysis::Result &(Function &)> GetMSSAResult) {
  if (JitFunctionInfoList.empty()) {
    //dbgs() << "=== Empty Begin Mod\n" << M << "=== End Mod\n";
    return false;
  }

  //dbgs() << "=== Pre M\n" << M << "=== End of Pre M\n";

  // First pass creates the string Module IR per jit'ed function.
  for (JitFunctionInfo &JFI : JitFunctionInfoList) {
    Function *F = JFI.Fn;

    SmallPtrSet<Function *, 16> ReachableFunctions;
    getReachableFunctions(CG, M, *F, ReachableFunctions);
    ReachableFunctions.insert(F);

    ValueToValueMapTy VMap;
    auto JitMod = CloneModule(
        M, VMap, [&ReachableFunctions,&F](const GlobalValue *GV) {
          if (const GlobalVariable *G = dyn_cast<GlobalVariable>(GV)) {
            if (!G->isConstant())
              return false;
          }

          // TODO: do not clone aliases' definitions, it this sound?
          if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
            return false;

          if (const Function *OrigF = dyn_cast<Function>(GV)) {
            if (OrigF == F) {
              dbgs() << "OrigF " << OrigF->getName() << " == " << F->getName() << ", definitely keep\n";
              return true;
            }
            // Do not keep definitions of unreachable functions.
            if (!ReachableFunctions.contains(OrigF)) {
              //dbgs() << "Drop unreachable " << F->getName() << "\n";
              return false;
            }

#ifdef ENABLE_RECURSIVE_JIT
            // Enable recursive jit'ing.
            for (auto &JFIInner : JitFunctionInfoList)
              if (JFIInner.Fn == OrigF) {
                dbgs() << "Do not keep definitions of another jit function " << OrigF->getName() << "\n";
                return false;
              }
#endif

            // dbgs() << "Keep reachable " << F->getName() << "\n";
            // getchar();
          }

          // By default, clone the definition.
          return true;
        });

    Function *JitF = cast<Function>(VMap[F]);
    JitF->setLinkage(GlobalValue::ExternalLinkage);

    // Set global variables to external linkage, when needed.
    for (auto &GV : M.global_values()) {
      if (VMap[&GV])
        if (auto *GVar = dyn_cast<GlobalVariable>(&GV)) {
          if (GVar->isConstant())
            continue;
          if (GVar->getSection() == "llvm.metadata")
            continue;
          if (GVar->getName() == "llvm.global_ctors")
            continue;
          if (GVar->isDSOLocal())
            continue;
          if (GVar->hasCommonLinkage())
            continue;
          //dbgs() << "=== GV\n";
          //dbgs() << GV << "\n";
          //dbgs() << "Linkage " << GV.getLinkage() << "\n";
          //dbgs() << "Visibility " << GV.getVisibility() << "\n";
          GV.setLinkage(GlobalValue::ExternalLinkage);
          //dbgs() << "Make " << GV << " External\n";
          //dbgs() << GV << "\n";
          //dbgs() << "=== End GV\n";
        }
    }
#ifdef ENABLE_RECURSIVE_JIT
    // Set linkage to external for any reachable jit'ed function to enable
    // recursive jit'ing.
    for (auto &JFIInner : JitFunctionInfoList) {
      if (!ReachableFunctions.contains(JFIInner.Fn))
        continue;
      Function *JitF = cast<Function>(VMap[JFIInner.Fn]);
      JitF->setLinkage(GlobalValue::ExternalLinkage);
    }
    F->setLinkage(GlobalValue::ExternalLinkage);
#endif
    // TODO: Do we want to keep debug info?
    StripDebugInfo(*JitMod);

    if (verifyModule(*JitMod, &errs()))
      report_fatal_error("Broken module found, compilation aborted!", false);
    else
      dbgs() << "JitMod verified!\n";

    // TODO: is writing/reading the bitcode instead of the textual IR faster?
    raw_string_ostream OS(JFI.ModuleIR);
    WriteBitcodeToFile(*JitMod, OS);
    OS.flush();

    //dbgs() << "=== StrIR\n" << *JitMod << "=== End of StrIR\n";
    //dbgs() << "=== Post M\n" << M << "=== End of Post M\n";
  }

  // Create jit entry runtime function.
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  // Use Int64 type for the Value, big enough to hold primitives.
  /* struct {
    int64_t Value;
    TODO: have a common header file?
  }
  Must be kept aligned with libjit.
  */
  StructType *RuntimeConstantTy =
      StructType::create({Int64Ty}, "struct.args");

  // TODO: This works by embedding the jit.bc library.
  // Function *JitEntryFn = M.getFunction("__jit_entry");
  // assert(JitEntryFn && "Expected non-null JitEntryFn");
  // FunctionType *JitEntryFnTy = JitEntryFn->getFunctionType();
  FunctionType *JitEntryFnTy =
      FunctionType::get(VoidPtrTy,
                        {VoidPtrTy, VoidPtrTy, Int32Ty,
                         RuntimeConstantTy->getPointerTo(), Int32Ty},
                        /* isVarArg=*/false);
  Function *JitEntryFn = Function::Create(
      JitEntryFnTy, GlobalValue::ExternalLinkage, "__jit_entry", M);

  // Second pass replaces jit'ed functions in the original module with stubs to
  // call the runtime entry point that compiles and links.
  for (JitFunctionInfo &JFI : JitFunctionInfoList) {
    Function *F = JFI.Fn;

    auto GetAnnotatedIntrinsics = [](Function &F) {
      SmallVector<CallBase *> AnnotatedIntrinsics;
      for (auto &BB : F)
        for (auto &I : BB)
          if (CallBase *CB = dyn_cast<CallBase>(&I))
            if ((CB->getIntrinsicID() == Intrinsic::var_annotation) ||
                (CB->getIntrinsicID() == Intrinsic::ptr_annotation))
              AnnotatedIntrinsics.push_back(CB);
      return AnnotatedIntrinsics;
    };

    SmallVector<Instruction *> RuntimeConstants;
    auto &MSSA = GetMSSAResult(*F).getMSSA();
    MSSA.ensureOptimizedUses();
    //dbgs() << "=== MSSA\n";
    //MSSA.print(dbgs());
    //dbgs() << "=== End of MSSA\n";
    SmallVector<CallBase *> AnnotatedInstrinsics = GetAnnotatedIntrinsics(*F);

    // TODO: check using MemorySSA that values are indeed runtime constants.
    for (auto *CB : AnnotatedInstrinsics) {
      if (CB->getIntrinsicID() == Intrinsic::var_annotation) {
        Value *Ptr = CB->getArgOperand(0);
        MemoryDef *MA = cast<MemoryDef>(MSSA.getMemoryAccess(CB));
        MemoryDef *Clobber = cast<MemoryDef>(MA->getDefiningAccess());

        StoreInst *Store = dyn_cast<StoreInst>(Clobber->getMemoryInst());
        assert(Store &&
               "Expected store instruction to llvm.var.annotation value");
        Argument *Arg = dyn_cast<Argument>(Store->getValueOperand());
        assert(Arg && "Expected memory dependency on function argument");

        RuntimeConstants.push_back(Store->clone());
      }
      if (CB->getIntrinsicID() == Intrinsic::ptr_annotation) {
        Value *Ptr = CB->getArgOperand(0);
        GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr);
        assert(GEP && "Expected GEP for llvm.ptr.annotation");
        Argument *Arg = dyn_cast<Argument>(GEP->getPointerOperand());
        assert(Arg &&
               "Expected GEP with a pointer operation to a function argument");

        RuntimeConstants.push_back(GEP->clone());
      }
    }

    // Replace jit'ed function with a stub function.
    std::string FnName = F->getName().str();
    F->setName("");
    Function *StubFn =
        Function::Create(F->getFunctionType(), F->getLinkage(), FnName, M);
    ValueToValueMapTy VMap;
    for (size_t I = 0; I < F->arg_size(); ++I) {
      VMap[F->getArg(I)] = StubFn->getArg(I);
      // Name arguments for readability.
      StubFn->getArg(I)->setName(F->getArg(I)->getName());
    }

    // Replace the body of the jit'ed function to call the jit entry, grab the
    // address of the specialized jit version and execute it.
    IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", StubFn));

    // Create the runtime constant array type for the runtime constants passed
    // to the jit entry function.
    ArrayType *RuntimeConstantArrayTy =
        ArrayType::get(RuntimeConstantTy, RuntimeConstants.size());

    // Create globals for the function name and string IR passed to the jit
    // entry.
    auto *FnNameGlobal = Builder.CreateGlobalString(StubFn->getName());
    auto *StrIRGlobal = Builder.CreateGlobalString(JFI.ModuleIR);

    // Create the runtime constants data structure passed to the jit entry.
    Value *RuntimeConstantsAlloca = nullptr;
    if (RuntimeConstants.size() > 0) {
      RuntimeConstantsAlloca = Builder.CreateAlloca(RuntimeConstantArrayTy);
      // Zero-initialize the alloca to avoid stack garbage for caching.
      Builder.CreateStore(Constant::getNullValue(RuntimeConstantArrayTy),
                          RuntimeConstantsAlloca);
      constexpr int ValueIdx = 0;
      for (int ArgI = 0; ArgI < RuntimeConstants.size(); ++ArgI) {
        auto *GEP = Builder.CreateInBoundsGEP(
            RuntimeConstantArrayTy, RuntimeConstantsAlloca,
            {Builder.getInt32(0), Builder.getInt32(ArgI)});

        auto *GEPValue =
            Builder.CreateStructGEP(RuntimeConstantTy, GEP, ValueIdx);

        if (auto *Store = dyn_cast<StoreInst>(RuntimeConstants[ArgI])) {
          VMap[Store->getPointerOperand()] = GEPValue;
          RemapInstruction(Store, VMap);
          Store->dropUnknownNonDebugMetadata();
          Builder.Insert(Store);
        }
        else if (auto *GEPI = dyn_cast<GetElementPtrInst>(RuntimeConstants[ArgI])) {
          RemapInstruction(GEPI, VMap);
          GEPI->dropUnknownNonDebugMetadata();
          Builder.Insert(GEPI);

          SmallVector<Value *> Operands;
          for (auto &Op : GEPI->operands())
            Operands.push_back(Op.get());

          ArrayRef<Value *> IdxList = ArrayRef<Value *>(Operands).slice(1);
          Type *IdxTy =
              GEPI->getIndexedType(GEPI->getSourceElementType(), IdxList);


          auto *Load = Builder.CreateLoad(IdxTy, GEPI);
          Builder.CreateStore(Load, GEPValue);
        }
        else
          assert(false && "Expected store or GEP instruction");
      }
    } else
      RuntimeConstantsAlloca =
          Constant::getNullValue(RuntimeConstantArrayTy->getPointerTo());

    assert(RuntimeConstantsAlloca && "Expected non-null runtime constants alloca");

    auto *JitFnPtr = Builder.CreateCall(
        JitEntryFnTy, JitEntryFn,
        {FnNameGlobal, StrIRGlobal, Builder.getInt32(JFI.ModuleIR.size()),
         RuntimeConstantsAlloca, Builder.getInt32(RuntimeConstants.size())});
    SmallVector<Value *, 8> Args;
    for (auto &Arg : StubFn->args())
      Args.push_back(&Arg);
    auto *RetVal =
        Builder.CreateCall(StubFn->getFunctionType(), JitFnPtr, Args);
    if (StubFn->getReturnType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(RetVal);

    //dbgs() << "=== Begin OrigFn\n" << *F << "=== End of OrigFn\n";
    //dbgs() << "=== Begin StubFn\n" << *StubFn << "=== End of StubFn\n";

    F->replaceAllUsesWith(StubFn);
    F->eraseFromParent();
  }

  //dbgs() << "=== Begin Mod\n" << M << "=== End Mod\n";
  if (verifyModule(M, &errs()))
    report_fatal_error("Broken module found, compilation aborted!", false);
  else
    dbgs() << "Module verified!\n";

  return true;
}

// New PM implementation
struct JitPass : PassInfoMixin<JitPass> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    parseAnnotations(M);
    CallGraph &CG = AM.getResult<CallGraphAnalysis>(M);
    FunctionAnalysisManager &FAM =
        AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    auto GetMSSAResult = [&FAM](Function &F) -> MemorySSAAnalysis::Result & {
      return FAM.getResult<MemorySSAAnalysis>(F);
    };
    bool Changed = visitor(M, CG, GetMSSAResult);

    if (Changed)
      return PreservedAnalyses::none();
    else
      return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation
struct LegacyJitPass : public ModulePass {
  static char ID;
  LegacyJitPass() : ModulePass(ID) {}
  // Main entry point - the name conveys what unit of IR this is to be run on.
  bool runOnModule(Module &M) override {
    assert(false && "Should never reach");
    return false;
  #if 0
    parseAnnotations(M);
    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    bool Changed = visitor(M, CG);

    return Changed;
  #endif
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getJitPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    // TODO: decide where to insert it in the pipeline. Early avoids
    // inlining jit function (which disables jit'ing) but may require more
    // optimization, hence overhead, at runtime.
    //PB.registerPipelineStartEPCallback([&](ModulePassManager &MPM, auto) {
    PB.registerPipelineEarlySimplificationEPCallback([&](ModulePassManager &MPM, auto) {
    // XXX: LastEP can break jit'ing, jit function is inlined!
    //PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto) {
      MPM.addPass(JitPass());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "JitPass", LLVM_VERSION_STRING, callback};
}

// TODO: use by jit-pass name.
// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize JitPass when added to the pass pipeline on the
// command line, i.e. via '-passes=jit-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getJitPassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyJitPass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyJitPass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-jit-pass'
static RegisterPass<LegacyJitPass>
    X("legacy-jit-pass", "Jit Pass",
      false, // This pass doesn't modify the CFG => false
      false // This pass is not a pure analysis pass => false
    );
