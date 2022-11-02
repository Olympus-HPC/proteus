//===-- LLJITWithOptimizingIRTransform.cpp -- LLJIT with IR optimization --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In this example we will use an IR transform to optimize a module as it
// passes through LLJIT's IRTransformLayer.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Vectorize.h"

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Instructions.h"

#include <iostream>

#define ENABLE_TIME_PROFILING

using namespace llvm;
using namespace llvm::orc;

struct TimeTracerRAII {
  TimeTracerRAII() { timeTraceProfilerInitialize(500 /* us */, "jit"); }

  ~TimeTracerRAII() {
    if (auto E = timeTraceProfilerWrite("", "-")) {
      handleAllErrors(std::move(E));
      return;
    }
    timeTraceProfilerCleanup();
  }
};

#ifdef ENABLE_TIME_PROFILING
TimeTracerRAII TimeTracer;
#define TIMESCOPE(x) TimeTraceScope T(x);
#else
#define TIMESCOPE(x)
#endif

inline Error createSMDiagnosticError(llvm::SMDiagnostic &Diag) {
  std::string Msg;
  {
    raw_string_ostream OS(Msg);
    Diag.print("", OS);
  }
  return make_error<StringError>(std::move(Msg), inconvertibleErrorCode());
}
// Returns the TargetMachine instance or zero if no triple is provided.
static TargetMachine* GetTargetMachine(Triple TheTriple, StringRef CPUStr,
                                       StringRef FeaturesStr,
                                       const TargetOptions &Options) {
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
  // Some modules don't specify a triple, and this is okay.
  if (!TheTarget) {
    return nullptr;
  }

  return TheTarget->createTargetMachine(
      TheTriple.getTriple(), codegen::getCPUStr(), codegen::getFeaturesStr(),
      Options, codegen::getExplicitRelocModel(),
      // TODO: Experiment with other CodeGenOpt options besides Aggresive?
      codegen::getExplicitCodeModel(), CodeGenOpt::Aggressive);
}

// A function object that creates a simple pass pipeline to apply to each
// module as it passes through the IRTransformLayer.
class OptimizationTransform {
public:
  OptimizationTransform() {

#if 0
    PM->add(createTailCallEliminationPass());
    PM->add(createFunctionInliningPass());
    PM->add(createIndVarSimplifyPass());
    PM->add(createCFGSimplificationPass());
    PM->add(createLICMPass());
#endif
  }

  Expected<ThreadSafeModule> operator()(ThreadSafeModule TSM,
                                        MaterializationResponsibility &R) {
    TSM.withModuleDo([this](Module &M) {
#if 0
      dbgs() << "--- BEFORE OPTIMIZATION ---\n" << M << "\n";
#endif
      TIMESCOPE("OptimizationTransform");
      Triple ModuleTriple(M.getTargetTriple());
      std::string CPUStr, FeaturesStr;
      TargetMachine *Machine = nullptr;
      const TargetOptions Options =
          codegen::InitTargetOptionsFromCodeGenFlags(ModuleTriple);

      if (ModuleTriple.getArch()) {
        CPUStr = codegen::getCPUStr();
        FeaturesStr = codegen::getFeaturesStr();
        Machine = GetTargetMachine(ModuleTriple, CPUStr, FeaturesStr, Options);
      } else if (ModuleTriple.getArchName() != "unknown" &&
                 ModuleTriple.getArchName() != "") {
        errs() << "unrecognized architecture '" << ModuleTriple.getArchName()
               << "' provided.\n";
        abort();
      }
      std::unique_ptr<TargetMachine> TM(Machine);
      codegen::setFunctionAttributes(CPUStr, FeaturesStr, M);
      TargetLibraryInfoImpl TLII(ModuleTriple);
      legacy::PassManager MPasses;

      MPasses.add(new TargetLibraryInfoWrapperPass(TLII));
      MPasses.add(createTargetTransformInfoWrapperPass(
          TM ? TM->getTargetIRAnalysis() : TargetIRAnalysis()));

      std::unique_ptr<legacy::FunctionPassManager> FPasses;
      FPasses.reset(new legacy::FunctionPassManager(&M));
      FPasses->add(createTargetTransformInfoWrapperPass(
          TM ? TM->getTargetIRAnalysis() : TargetIRAnalysis()));

      if (TM) {
        // FIXME: We should dyn_cast this when supported.
        auto &LTM = static_cast<LLVMTargetMachine &>(*TM);
        Pass *TPC = LTM.createPassConfig(MPasses);
        MPasses.add(TPC);
      }

      unsigned int OptLevel = 3;

      {
        TIMESCOPE("Builder");
        PassManagerBuilder Builder;
        Builder.OptLevel = OptLevel;
        Builder.SizeLevel = 0;
        Builder.Inliner = createFunctionInliningPass(OptLevel, 0, false);
        Builder.DisableUnrollLoops = false;
        Builder.LoopVectorize = true;
        Builder.SLPVectorize = true;
        TM->adjustPassManager(Builder);
        Builder.populateFunctionPassManager(*FPasses);
        Builder.populateModulePassManager(MPasses);
      }

      {
        TIMESCOPE("RunPassPipeline");
        if (FPasses) {
          FPasses->doInitialization();
          for (Function &F : M)
            FPasses->run(F);
          FPasses->doFinalization();
        }
        MPasses.run(M);
      }
#if 0
      dbgs() << "--- AFTER OPTIMIZATION ---\n" << M << "\n";
#endif
    });
    return std::move(TSM);
  }
};

struct RuntimeConstant {
  union {
    int32_t Int32Val;
    int64_t Int64Val;
    float FloatVal;
    double DoubleVal;
  };
};


static codegen::RegisterCodeGenFlags CFG;
std::unique_ptr<LLJIT> J;
// TODO: make it a singleton?
class JitEngine {
public:

  ExitOnError ExitOnErr;

  struct JitCacheEntry {
    void *Ptr;
    int num_execs;
  };
  StringMap<JitCacheEntry> JitCache;
  int hits = 0;
  int total = 0;

  JitEngine(int argc, char *argv[]) {
    InitLLVM X(argc, argv);

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    ExitOnErr.setBanner("JIT: ");
    // Create the LLJIT instance.
    J = ExitOnErr(LLJITBuilder().create());
    // (2) Resolve symbols in the main process.
    orc::MangleAndInterner Mangle(J->getExecutionSession(), J->getDataLayout());
    J->getMainJITDylib().addGenerator(
        ExitOnErr(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            J->getDataLayout().getGlobalPrefix(),
            [MainName = Mangle("main")](const orc::SymbolStringPtr &Name) {
              // dbgs() << "Search name " << Name << "\n";
              return Name != MainName;
            })));

    // (2) Install transform to optimize modules when they're materialized.
    J->getIRTransformLayer().setTransform(OptimizationTransform());

    //dbgs() << "JIT inited\n";
    //getchar();
  }

  ~JitEngine() {
    std::cout << "JitCache hits " << hits << " total " << total << "\n";
    for (auto &It : JitCache) {
      StringRef FnName = It.getKey();
      JitCacheEntry &JCE = It.getValue();
      std::cout << "FnName " << FnName.str() << " num_execs " << JCE.num_execs
                << "\n";
    }
  }

  Expected<llvm::orc::ThreadSafeModule>
  parseSource(StringRef FnName, StringRef Suffix, StringRef IR,
              RuntimeConstant *RC, int NumRuntimeConstants) {

    TIMESCOPE("parseSource");
    auto Ctx = std::make_unique<LLVMContext>();
    SMDiagnostic Err;
    if (auto M =
            parseIR(MemoryBufferRef(IR, ("Mod-" + FnName + Suffix).str()), Err, *Ctx)) {
      //dbgs() << "=== Parsed Module\n" << *M << "=== End of Parsed Module\n";
      Function *F = M->getFunction(FnName);

      auto GetConstant = [](RuntimeConstant &RC, Type *ArgTy) {
        Constant *C = nullptr;
        if (ArgTy->isIntegerTy(32)) {
           //dbgs() << "RC is Int32\n";
          C = ConstantInt::get(ArgTy, RC.Int32Val);
        } else if (ArgTy->isIntegerTy(64)) {
           //dbgs() << "RC is Int64\n";
          C = ConstantInt::get(ArgTy, RC.Int64Val);
        } else if (ArgTy->isFloatTy()) {
           //dbgs() << "RC is Float\n";
          C = ConstantFP::get(ArgTy, RC.FloatVal);
        } else if (ArgTy->isDoubleTy()) {
           //dbgs() << "RC is Double\n";
          C = ConstantFP::get(ArgTy, RC.DoubleVal);
#if 0
        } else if (ArgTy->isPointerTy()) {
          auto *IntC = ConstantInt::get(Type::getInt64Ty(*Ctx), RC.Int64Val);
          C = ConstantExpr::getIntToPtr(IntC, ArgTy);
#endif
        } else
          report_fatal_error("JIT Incompatible type in runtime constant");

        return C;
      };

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

      auto GetGEPOperands = [](GetElementPtrInst *GEP) {
        SmallVector<Value *> Operands;
          for (auto &Op : GEP->operands()) {
            Operands.push_back(Op.get());
            dbgs() << "Pushed Op " << *Op.get() << "\n";
          }

          return Operands;
      };
      // Clone the function and replace argument uses with runtime constants.
      ValueToValueMapTy VMap;
      F->setName("");
      Function *NewF = CloneFunction(F, VMap);
      NewF->setName(FnName);
      int RCIdx = 0;

      auto AnnotatedIntrinsics = GetAnnotatedIntrinsics(*NewF);

      for (CallBase *CB : AnnotatedIntrinsics) {
        if (CB->getIntrinsicID() == Intrinsic::var_annotation) {
          Value *Ptr = CB->getArgOperand(0);
          AllocaInst *PtrAlloca = dyn_cast<AllocaInst>(Ptr);
          assert(PtrAlloca && "Expected alloca for llvm.var.annotation");
          Constant *C = GetConstant(RC[RCIdx],
                                    PtrAlloca->getAllocatedType());
          assert(C && "Expected non-null constant");
          auto *Store = new StoreInst(C, Ptr, CB);
          CB->eraseFromParent();
        } else if (CB->getIntrinsicID() == Intrinsic::ptr_annotation) {
          Value *Ptr = CB->getArgOperand(0);
          auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
          assert(GEP && "Expected GEP for llvm.ptr.annotation");
          ArrayRef<Value *> IdxList =
              ArrayRef<Value *>(GetGEPOperands(GEP)).slice(1);
          for (auto *V : IdxList)
            dbgs() << "IdxList " << *V << "\n";
          Type *IdxTy =
              GEP->getIndexedType(GEP->getSourceElementType(), IdxList);
          Constant *C = GetConstant(RC[RCIdx], IdxTy);
          assert(C && "Expected non-null constant");
          // TODO: fix addres space.
          auto *Alloca =
              new AllocaInst(C->getType(), 0, GEP->getName() + ".rc",
                             &*NewF->getEntryBlock().getFirstInsertionPt());
          auto *Store = new StoreInst(C, Alloca, CB);
          CB->replaceAllUsesWith(Alloca);
          CB->eraseFromParent();
        }

        RCIdx++;
      }

      assert(RCIdx == NumRuntimeConstants && "Expected to use all runtime constants");

      F->replaceAllUsesWith(NewF);
      F->eraseFromParent();

      //dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n";

      NewF->setName(FnName + Suffix);
      for (Function &F : *M) {
        if (F.isDeclaration())
          continue;

        if (&F == NewF)
          continue;

        // Internalize other functions in the module.
        F.setLinkage(GlobalValue::InternalLinkage);
        // Rename functions to internalize using jit'ed function name.
        F.setName(F.getName() + ".." + NewF->getName());
      }

      // dbgs() << "NewF " << *NewF << "\n";
      // getchar();
#if 0
      dbgs() << "=== Modified Module\n" << *M << "=== End of Modified Module\n";
      if (verifyModule(*M, &errs()))
        report_fatal_error("Broken module found, JIT compilation aborted!", false);
      else
        dbgs() << "Module verified!\n";
#endif
      return ThreadSafeModule(std::move(M), std::move(Ctx));
    }

    return createSMDiagnosticError(Err);
  }

  void *compileAndLink(StringRef FnName, StringRef IR, RuntimeConstant *RC,
                      int NumRuntimeConstants) {
    TIMESCOPE("compileAndLink");
    std::string Suffix = mangleSuffix(FnName, RC, NumRuntimeConstants);
    std::string MangledFnName = FnName.str() + Suffix;

    void *JitFnPtr = lookup(MangledFnName);
    if (JitFnPtr)
      return JitFnPtr;

    dbgs() << "======= COMPILING " << FnName << " =====================\n";
    // (3) Add modules.
    ExitOnErr(J->addIRModule(
        ExitOnErr(parseSource(FnName, Suffix, IR, RC, NumRuntimeConstants))));

    // (4) Look up the JIT'd function.
    //dbgs() << "Lookup FnName " << FnName << "\n";
    auto EntryAddr = ExitOnErr(J->lookup(MangledFnName));

    JitFnPtr = (void *)EntryAddr.getValue();
    insert(MangledFnName, JitFnPtr);

    return JitFnPtr;
  }

  std::string mangleSuffix(StringRef FnName, RuntimeConstant *RC,
                           int NumRuntimeConstants) {
    // Generate mangled name with runtime constants.
    std::string Suffix = ".";
    for (int I = 0; I < NumRuntimeConstants; ++I)
      Suffix += ("." + std::to_string(RC[I].Int64Val));
    return Suffix;
  }

  void *lookup(StringRef FnName) {
    TIMESCOPE("lookup");
    total++;

    auto It = JitCache.find(FnName.str());
    if (It == JitCache.end())
      return nullptr;

    It->getValue().num_execs++;
    hits++;
    return It->getValue().Ptr;
  }

  void insert(StringRef FnName, void *Ptr) {
    TIMESCOPE("insert");
    JitCache[FnName.str()] = {Ptr, /* num_execs */ 1};
  }
};

JitEngine Jit(0, (char *[]){ nullptr });

extern "C" {
__attribute__((used)) void *__jit_entry(char *FnName, char *IR, int IRSize,
                                        RuntimeConstant *RC,
                                        int NumRuntimeConstants) {
  TIMESCOPE("__jit_entry");
#if 0
  dbgs() << "FnName " << FnName << " NumRuntimeConstants " << NumRuntimeConstants << "\n";
  for (int I = 0; I < NumRuntimeConstants; ++I)
    dbgs() << "RC[" << I << "]: ArgNo=" << RC[I].ArgNo
           << " Value Int32=" << RC[I].Int32Val
           << " Value Int64=" << RC[I].Int64Val
           << " Value Float=" << RC[I].FloatVal
           << " Value Double=" << RC[I].DoubleVal
           << "\n";
#endif

  //dbgs() << "JIT Entry " << FnName << "\n";
  StringRef StrIR(IR, IRSize);
  void *JitFnPtr = Jit.compileAndLink(FnName, StrIR, RC, NumRuntimeConstants);

  return JitFnPtr;
}
}
