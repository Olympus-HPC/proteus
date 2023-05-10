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
#include "llvm/IR/DebugInfo.h"
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
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/Object/SymbolSize.h"

#include <iostream>

//#define ENABLE_TIME_PROFILING
//#define ENABLE_PERFMAP
//#define ENABLE_DEBUG

#define ENABLE_RUNTIME_CONSTPROP

#ifdef ENABLE_DEBUG
#define DBG(x) x;
#else
#define DBG(x)
#endif

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
      DBG(dbgs() << "=== Begin Before Optimization\n" << M << "=== End Before\n");
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

      unsigned int OptLevel = 2;

      {
        TIMESCOPE("Builder");
        PassManagerBuilder Builder;
        Builder.OptLevel = OptLevel;
        Builder.SizeLevel = 0;
        Builder.Inliner = nullptr;
        //Builder.Inliner = createAlwaysInlinerLegacyPass();
        //Builder.Inliner = createFunctionInliningPass(OptLevel, 0, false);
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
      DBG(dbgs() << "=== Begin After Optimization\n" << M << "=== End After Optimization\n");
#ifdef ENABLE_DEBUG
      if (verifyModule(M, &errs()))
        report_fatal_error(
            "Broken module found after optimization, JIT compilation aborted!",
            false);
      else
        dbgs() << "Module after optimization verified!\n";
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

inline hash_code hash_value(const RuntimeConstant &RC) {
  return hash_value(RC.Int64Val);
}


static codegen::RegisterCodeGenFlags CFG;
std::unique_ptr<LLJIT> J;
// TODO: make it a singleton?
class JitEngine {
public:

  ExitOnError ExitOnErr;

  struct JitCacheEntry {
    void *Ptr;
    int num_execs;
#ifdef ENABLE_DEBUG
    std::string FnName;
    SmallVector<RuntimeConstant, 8> RCVector;
#endif
  };
  DenseMap<uint64_t, JitCacheEntry> JitCache;
  int hits = 0;
  int total = 0;

  static void
  dumpSymbolInfo(const llvm::object::ObjectFile &loadedObj,
                 const llvm::RuntimeDyld::LoadedObjectInfo &objInfo) {
    // Dump information about symbols.
    auto pid = sys::Process::getProcessId();
    std::error_code EC;
    raw_fd_ostream ofd("/tmp/perf-" + std::to_string(pid) + ".map", EC,
                       sys::fs::OF_Append);
    if (EC)
      report_fatal_error("Cannot open perf map file");
    for (auto symSizePair : llvm::object::computeSymbolSizes(loadedObj)) {
      auto sym = symSizePair.first;
      auto size = symSizePair.second;
      auto symName = sym.getName();
      // Skip any unnamed symbols.
      if (!symName || symName->empty())
        continue;
      // The relative address of the symbol inside its section.
      auto symAddr = sym.getAddress();
      if (!symAddr)
        continue;
      // The address the functions was loaded at.
      auto loadedSymAddress = *symAddr;
      auto symbolSection = sym.getSection();
      if (symbolSection) {
        // Compute the load address of the symbol by adding the section load
        // address.
        loadedSymAddress += objInfo.getSectionLoadAddress(*symbolSection.get());
      }
      llvm::outs() << llvm::format("Address range: [%12p, %12p]",
                                   loadedSymAddress, loadedSymAddress + size)
                   << "\tSymbol: " << *symName << "\n";

      if (size > 0)
        ofd << llvm::format("%lx %x)", loadedSymAddress, size) << " "
            << *symName << "\n";
    }

    ofd.close();
  }
  static void notifyLoaded(MaterializationResponsibility &R,
                    const object::ObjectFile &Obj,
                    const RuntimeDyld::LoadedObjectInfo &LOI) {
    dumpSymbolInfo(Obj, LOI);
  }

  JitEngine(int argc, char *argv[]) {
    InitLLVM X(argc, argv);

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    ExitOnErr.setBanner("JIT: ");
    // Create the LLJIT instance.
    // TODO: Fix support for debugging jitted code. This appears to be
    // the correct interface (see orcv2 examples) but it does not work.
    // By dumpSymbolInfo() the debug sections are not populated. Why?
    J = ExitOnErr(LLJITBuilder()
                      .setObjectLinkingLayerCreator([&](ExecutionSession &ES,
                                                        const Triple &TT) {
                        auto GetMemMgr = []() {
                          return std::make_unique<SectionMemoryManager>();
                        };
                        auto ObjLinkingLayer =
                            std::make_unique<RTDyldObjectLinkingLayer>(
                                ES, std::move(GetMemMgr));

                        // Register the event listener.
                        ObjLinkingLayer->registerJITEventListener(
                            *JITEventListener::createGDBRegistrationListener());

                        // Make sure the debug info sections aren't stripped.
                        ObjLinkingLayer->setProcessAllSections(true);

#if defined(ENABLE_DEBUG) || defined(ENABLE_PERFMAP)
                        ObjLinkingLayer->setNotifyLoaded(notifyLoaded);
#endif

                        return ObjLinkingLayer;
                      })
                      .create());
    // (2) Resolve symbols in the main process.
    orc::MangleAndInterner Mangle(J->getExecutionSession(), J->getDataLayout());
    J->getMainJITDylib().addGenerator(
        ExitOnErr(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            J->getDataLayout().getGlobalPrefix(),
            [MainName = Mangle("main")](const orc::SymbolStringPtr &Name) {
              // dbgs() << "Search name " << Name << "\n";
              return Name != MainName;
            })));

    // (3) Install transform to optimize modules when they're materialized.
    J->getIRTransformLayer().setTransform(OptimizationTransform());

    //dbgs() << "JIT inited\n";
    //getchar();
  }

  ~JitEngine() {
    std::cout << std::dec;
    std::cout << "JitCache hits " << hits << " total " << total << "\n";
    for (auto &It : JitCache) {
      uint64_t HashValue = It.first;
      JitCacheEntry &JCE = It.second;
      std::cout << "HashValue " << HashValue << " num_execs " << JCE.num_execs;
#ifdef ENABLE_DEBUG
      std::cout << " FnName " << JCE.FnName << " RCs [";
      for (auto &RC : JCE.RCVector)
        std::cout << RC.Int64Val << ", ";
      std::cout << "]";
#endif
      std::cout << "\n";
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
      assert(F && "Expected non-null function!");
      MDNode *Node = F->getMetadata("jit_arg_nos");
      DBG(dbgs() << "Metadata jit for F " << F->getName() << " = " << *Node << "\n");

      // Replace argument uses with runtime constants.
#ifdef ENABLE_RUNTIME_CONSTPROP
      for (int I = 0; I < Node->getNumOperands(); ++I) {
        ConstantAsMetadata *CAM = cast<ConstantAsMetadata>(Node->getOperand(I));
        ConstantInt *ConstInt = cast<ConstantInt>(CAM->getValue());
        int ArgNo = ConstInt->getZExtValue();
        Value *Arg = F->getArg(ArgNo);
        // TODO: add constant creation for FP types too.
        Type *ArgType = Arg->getType();
        Constant *C = nullptr;
        if (ArgType->isIntegerTy(32)) {
          // dbgs() << "RC is Int32\n";
          C = ConstantInt::get(ArgType, RC[I].Int32Val);
        } else if (ArgType->isIntegerTy(64)) {
          // dbgs() << "RC is Int64\n";
          C = ConstantInt::get(ArgType, RC[I].Int64Val);
        } else if (ArgType->isFloatTy()) {
          // dbgs() << "RC is Float\n";
          C = ConstantFP::get(ArgType, RC[I].FloatVal);
        } else if (ArgType->isDoubleTy()) {
          // dbgs() << "RC is Double\n";
          C = ConstantFP::get(ArgType, RC[I].DoubleVal);
        } else if (ArgType->isPointerTy()) {
          auto *IntC = ConstantInt::get(Type::getInt64Ty(*Ctx), RC[I].Int64Val);
          C = ConstantExpr::getIntToPtr(IntC, ArgType);
        } else
          report_fatal_error("JIT Incompatible type in runtime constant");

        Arg->replaceAllUsesWith(C);
      }
#endif

      //dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n";

      F->setName(FnName + Suffix);

#if 0
      for(auto &GO: M->global_objects()) {
        GO.setComdat(nullptr);
      }
#endif

#ifdef ENABLE_DEBUG
      dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
      if (verifyModule(*M, &errs()))
        report_fatal_error("Broken module found, JIT compilation aborted!", false);
      else
        dbgs() << "Module verified!\n";
#endif
      return ThreadSafeModule(std::move(M), std::move(Ctx));
    }

    return createSMDiagnosticError(Err);
  }

  void *compileAndLink(StringRef FnName, char *IR, int IRSize, RuntimeConstant *RC,
                      int NumRuntimeConstants) {
    TIMESCOPE("compileAndLink");

    uint64_t HashValue = hash(FnName, RC, NumRuntimeConstants);
    void *JitFnPtr = lookupCache(HashValue);
    if (JitFnPtr)
      return JitFnPtr;

    std::string Suffix = mangleSuffix(HashValue);
    std::string MangledFnName = FnName.str() + Suffix;

    StringRef StrIR(IR, IRSize);
    // (3) Add modules.
    ExitOnErr(J->addIRModule(
        ExitOnErr(parseSource(FnName, Suffix, StrIR, RC, NumRuntimeConstants))));

    DBG(dbgs() << "===\n" << *J->getExecutionSession().getSymbolStringPool() << "===\n");

    // (4) Look up the JIT'd function.
    DBG(dbgs() << "Lookup FnName " << FnName << " mangled as " << MangledFnName << "\n");
    auto EntryAddr = ExitOnErr(J->lookup(MangledFnName));

    JitFnPtr = (void *)EntryAddr.getValue();
    DBG(dbgs() << "FnName " << FnName << " Mangled " << MangledFnName << " address " << JitFnPtr << "\n");
    assert(JitFnPtr && "Expected non-null JIT function pointer");
    insertCache(HashValue, JitFnPtr
#ifdef ENABLE_DEBUG
                ,
                FnName, RC, NumRuntimeConstants
#endif
    );

    dbgs() << "=== JIT compile: " << FnName << " Mangled " << MangledFnName
           << " RC HashValue " << HashValue << " Addr " << JitFnPtr << "\n";
    return JitFnPtr;
  }

  uint64_t hash(StringRef FnName, RuntimeConstant *RC, int NumRuntimeConstants) {
    ArrayRef<RuntimeConstant> Data(RC, NumRuntimeConstants);
    auto HashValue = hash_combine(FnName, Data);
    return HashValue;
  }

  std::string mangleSuffix(uint64_t HashValue) {
    // Generate mangled suffix from runtime constants.
    return "_" + utostr(HashValue);
  }

  void *lookupCache(uint64_t HashValue) {
    TIMESCOPE("lookup");
    total++;

    auto It = JitCache.find(HashValue);
    if (It == JitCache.end())
      return nullptr;

    It->second.num_execs++;
    hits++;
    return It->second.Ptr;
  }

  void insertCache(uint64_t HashValue, void *Ptr
#ifdef ENABLE_DEBUG
                   ,
                   StringRef FnName, RuntimeConstant *RC,
                   int NumRuntimeConstants
#endif
  ) {
#ifdef ENABLE_DEBUG
    if (JitCache.count(HashValue))
      report_fatal_error("JitCache collision detected");
#endif
    JitCache[HashValue] = {Ptr, /* num_execs */ 1};
#ifdef ENABLE_DEBUG
    JitCache[HashValue].FnName = FnName.str();
    for (size_t I = 0; I < NumRuntimeConstants; ++I)
      JitCache[HashValue].RCVector.push_back(RC[I]);
#endif
  }
};

JitEngine Jit(0, (char *[]){ nullptr });

extern "C" {
__attribute__((used)) void *__jit_entry(char *FnName, char *IR, int IRSize,
                                        RuntimeConstant *RC,
                                        int NumRuntimeConstants) {
  TIMESCOPE("__jit_entry");
#if 0
  dbgs() << "FnName " << FnName << " NumRuntimeConstants "
         << NumRuntimeConstants << "\n";
  for (int I = 0; I < NumRuntimeConstants; ++I)
    dbgs() << " Value Int32=" << RC[I].Int32Val
           << " Value Int64=" << RC[I].Int64Val
           << " Value Float=" << RC[I].FloatVal
           << " Value Double=" << RC[I].DoubleVal << "\n";
#endif

  //dbgs() << "JIT Entry " << FnName << "\n";
  void *JitFnPtr = Jit.compileAndLink(FnName, IR, IRSize, RC, NumRuntimeConstants);

  return JitFnPtr;
}
}
