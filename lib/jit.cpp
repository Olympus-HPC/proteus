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
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
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

#include <iostream>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#if ENABLE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#define hipErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      abort();                                                                 \
    }                                                                          \
  }

#define hiprtcErrCheck(CALL)                                                   \
  {                                                                            \
    hiprtcResult err = CALL;                                                   \
    if (err != HIPRTC_SUCCESS) {                                               \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hiprtcGetErrorString(err));                                       \
      abort();                                                                 \
    }                                                                          \
  }

#endif

// #define ENABLE_TIME_PROFILING
// #define ENABLE_PERFMAP
// #define ENABLE_DEBUG

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
static TargetMachine *GetTargetMachine(Triple TheTriple, StringRef CPUStr,
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
      DBG(dbgs() << "=== Begin Before Optimization\n"
                 << M << "=== End Before\n");
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
        // TODO: Use the new PM.
        PassManagerBuilder Builder;
        Builder.OptLevel = OptLevel;
        Builder.SizeLevel = 0;
        Builder.Inliner = nullptr;
        // Builder.Inliner = createAlwaysInlinerLegacyPass();
        // Builder.Inliner = createFunctionInliningPass(OptLevel, 0, false);
        Builder.DisableUnrollLoops = false;
        Builder.LoopVectorize = true;
        Builder.SLPVectorize = true;
        // TODO: This is unsupported in llvm 16 as unneeded, check.
        // TM->adjustPassManager(Builder);
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
      DBG(dbgs() << "=== Begin After Optimization\n"
                 << M << "=== End After Optimization\n");
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

  Expected<ThreadSafeModule> operator()(ThreadSafeModule TSM) {
    TSM.withModuleDo([this](Module &M) {
      DBG(dbgs() << "=== Begin Before Optimization\n"
                 << M << "=== End Before\n");
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
        // TODO: Use the new PM.
        PassManagerBuilder Builder;
        Builder.OptLevel = OptLevel;
        Builder.SizeLevel = 0;
        Builder.Inliner = nullptr;
        // Builder.Inliner = createAlwaysInlinerLegacyPass();
        // Builder.Inliner = createFunctionInliningPass(OptLevel, 0, false);
        Builder.DisableUnrollLoops = false;
        Builder.LoopVectorize = true;
        Builder.SLPVectorize = true;
        // TODO: This is unsupported in llvm 16 as unneeded, check.
        // TM->adjustPassManager(Builder);
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
      DBG(dbgs() << "=== Begin After Optimization\n"
                 << M << "=== End After Optimization\n");
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
    void *PtrVal; // TODO: think
  };
};

inline hash_code hash_value(const RuntimeConstant &RC) {
  return hash_value(RC.Int64Val);
}

// TODO: check if this global is needed.
static codegen::RegisterCodeGenFlags CFG;

class JitEngine {
public:
  std::unique_ptr<LLJIT> LLJITPtr;
  ExitOnError ExitOnErr;

  static JitEngine &instance() {
    static JitEngine Jit(0, (char *[]){nullptr});
    return Jit;
  }

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
  parseBitcode(StringRef FnName, StringRef Suffix, StringRef IR,
               RuntimeConstant *RC, int NumRuntimeConstants) {

    TIMESCOPE("parseBitcode");
    auto Ctx = std::make_unique<LLVMContext>();
    SMDiagnostic Err;
    if (auto M = parseIR(MemoryBufferRef(IR, ("Mod-" + FnName + Suffix).str()),
                         Err, *Ctx)) {
      // dbgs() << "=== Parsed Module\n" << *M << "=== End of Parsed Module\n";
      Function *F = M->getFunction(FnName);
      assert(F && "Expected non-null function!");
      MDNode *Node = F->getMetadata("jit_arg_nos");
      DBG(dbgs() << "Metadata jit for F " << F->getName() << " = " << *Node
                 << "\n");

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

      // dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n";

      F->setName(FnName + Suffix);

#if 0
      for(auto &GO: M->global_objects()) {
        GO.setComdat(nullptr);
      }
#endif

#ifdef ENABLE_DEBUG
      dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
      if (verifyModule(*M, &errs()))
        report_fatal_error("Broken module found, JIT compilation aborted!",
                           false);
      else
        dbgs() << "Module verified!\n";
#endif
      return ThreadSafeModule(std::move(M), std::move(Ctx));
    }

    return createSMDiagnosticError(Err);
  }

  void *compileAndLink(StringRef FnName, char *IR, int IRSize,
                       RuntimeConstant *RC, int NumRuntimeConstants) {
    TIMESCOPE("compileAndLink");

    uint64_t HashValue = hash(FnName, RC, NumRuntimeConstants);
    void *JitFnPtr = lookupCache(HashValue);
    if (JitFnPtr)
      return JitFnPtr;

    std::string Suffix = mangleSuffix(HashValue);
    std::string MangledFnName = FnName.str() + Suffix;

    StringRef StrIR(IR, IRSize);
    // (3) Add modules.
    ExitOnErr(LLJITPtr->addIRModule(ExitOnErr(
        parseBitcode(FnName, Suffix, StrIR, RC, NumRuntimeConstants))));

    DBG(dbgs() << "===\n"
               << *LLJITPtr->getExecutionSession().getSymbolStringPool()
               << "===\n");

    // (4) Look up the JIT'd function.
    DBG(dbgs() << "Lookup FnName " << FnName << " mangled as " << MangledFnName
               << "\n");
    auto EntryAddr = ExitOnErr(LLJITPtr->lookup(MangledFnName));

    JitFnPtr = (void *)EntryAddr.getValue();
    DBG(dbgs() << "FnName " << FnName << " Mangled " << MangledFnName
               << " address " << JitFnPtr << "\n");
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

  uint64_t hash(StringRef FnName, RuntimeConstant *RC,
                int NumRuntimeConstants) {
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

private:
  JitEngine(int argc, char *argv[]) {
    InitLLVM X(argc, argv);

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    ExitOnErr.setBanner("JIT: ");
    // Create the LLJIT instance.
    // TODO: Fix support for debugging jitted code. This appears to be
    // the correct interface (see orcv2 examples) but it does not work.
    // By dumpSymbolInfo() the debug sections are not populated. Why?
    LLJITPtr =
        ExitOnErr(LLJITBuilder()
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
    orc::MangleAndInterner Mangle(LLJITPtr->getExecutionSession(),
                                  LLJITPtr->getDataLayout());
    LLJITPtr->getMainJITDylib().addGenerator(
        ExitOnErr(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            LLJITPtr->getDataLayout().getGlobalPrefix(),
            [MainName = Mangle("main")](const orc::SymbolStringPtr &Name) {
              // dbgs() << "Search name " << Name << "\n";
              return Name != MainName;
            })));

    // (3) Install transform to optimize modules when they're materialized.
    LLJITPtr->getIRTransformLayer().setTransform(OptimizationTransform());

    // dbgs() << "JIT inited\n";
    // getchar();
  }
};

extern "C" {
__attribute__((used)) void *__jit_entry(char *FnName, char *IR, int IRSize,
                                        RuntimeConstant *RC,
                                        int NumRuntimeConstants) {
  TIMESCOPE("__jit_entry");
  JitEngine &Jit = JitEngine::instance();
  // JitEngi(0, (char *[]){ nullptr });
#if 0
    dbgs() << "FnName " << FnName << " NumRuntimeConstants "
      << NumRuntimeConstants << "\n";
    for (int I = 0; I < NumRuntimeConstants; ++I)
      dbgs() << " Value Int32=" << RC[I].Int32Val
        << " Value Int64=" << RC[I].Int64Val
        << " Value Float=" << RC[I].FloatVal
        << " Value Double=" << RC[I].DoubleVal << "\n";
#endif

  // dbgs() << "JIT Entry " << FnName << "\n";
  void *JitFnPtr =
      Jit.compileAndLink(FnName, IR, IRSize, RC, NumRuntimeConstants);

  return JitFnPtr;
}

int __hipRegisterFunction(void *, void *, void *, void *, int, void *, void *,
                          void *, void *, void *);
void *__hipRegisterFatBinary(void *);
void __hipRegisterVar(void *, void *, const char *, const char *, int32_t,
                      int64_t, int32_t, int32_t);

class JitEngineDevice {
public:
  static JitEngineDevice &instance() {
    static JitEngineDevice Jit{};
    return Jit;
  }

  ~JitEngineDevice() {}

  Expected<llvm::orc::ThreadSafeModule>
  parseBitcode(StringRef FnName, StringRef Suffix, StringRef IR,
               RuntimeConstant *RC, int NumRuntimeConstants) {

    TIMESCOPE("parseBitcode");
    auto Ctx = std::make_unique<LLVMContext>();
    SMDiagnostic Err;
    if (auto M = parseIR(MemoryBufferRef(IR, ("Mod-" + FnName + Suffix).str()),
                         Err, *Ctx)) {
      DBG(dbgs() << "=== Parsed Module\n"
                 << *M << "=== End of Parsed Module\n");
      Function *F = M->getFunction(FnName);
      assert(F && "Expected non-null function!");
      MDNode *Node = F->getMetadata("jit_arg_nos");
      DBG(dbgs() << "Metadata jit for F " << F->getName() << " = " << *Node
                 << "\n");

      // Re-link globals to fixed addresses provided by registered variables.
      for (auto RegisterVar : VarNameToDevPtr) {
        auto &VarName = RegisterVar.first;
        auto DevPtr = RegisterVar.second;
        auto *GV = M->getNamedGlobal(VarName);
        assert(GV && "Expected existing global variable");
        // Remove the re-linked global from llvm.compiler.used since it that use
        // is not replaceable by the fixed addr constant expression.
        removeFromUsedLists(*M, [&GV](Constant *C) {
          if (GV == C)
            return true;

          return false;
        });

        Constant *Addr =
            ConstantInt::get(Type::getInt64Ty(*Ctx), (uint64_t)DevPtr);
        Value *CE = ConstantExpr::getIntToPtr(Addr, GV->getType());
        GV->replaceAllUsesWith(CE);
      }

#ifdef ENABLE_DEBUG
      dbgs() << "=== Linked M\n" << *M << "=== End of Linked M\n";
      if (verifyModule(*M, &errs()))
        report_fatal_error(
            "After linking, broken module found, JIT compilation aborted!",
            false);
      else
        dbgs() << "Module verified!\n";
#endif

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

      DBG(dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n");

      F->setName(FnName + Suffix);

#if 0
      for(auto &GO: M->global_objects()) {
        GO.setComdat(nullptr);
      }
#endif

#ifdef ENABLE_DEBUG
      dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
      if (verifyModule(*M, &errs()))
        report_fatal_error("Broken module found, JIT compilation aborted!",
                           false);
      else
        dbgs() << "Module verified!\n";
#endif
      return ThreadSafeModule(std::move(M), std::move(Ctx));
    }

    return createSMDiagnosticError(Err);
  }

  hipError_t compileAndRun(StringRef FnName, char *IR, int IRSize,
                           RuntimeConstant *RC, int NumRuntimeConstants,
                           uint32_t GridDimX, uint32_t GridDimY,
                           uint32_t GridDimZ, uint32_t BlockDimX,
                           uint32_t BlockDimY, uint32_t BlockDimZ,
                           void **KernelArgs, uint64_t ShmemSize,
                           void *Stream) {
    TIMESCOPE("compileAndRun");

#ifdef ENABLE_HIP
    StringRef StrIR(IR, IRSize);
    // TODO: We don't need a unique suffix here if we use a different hip
    // module, function per specialized kernel. Depends on how we implement
    // loading and executing device kernels.
    auto TransformedBitcode =
        parseBitcode(FnName, "_jit", StrIR, RC, NumRuntimeConstants);
    if (auto E = TransformedBitcode.takeError())
      report_fatal_error(toString(std::move(E)).c_str());

    Module *M = TransformedBitcode->getModuleUnlocked();

    std::string MemBuf;
    raw_string_ostream OS(MemBuf);
    OS << *M;
    // WriteBitcodeToFile(*M, OS);
    OS.flush();

    Function *F = M->getFunction(Twine(FnName + "_jit").str());
    assert(F && "Expected non-null function");

    hipModule_t HipModule;
    hipFunction_t HipFunction;

    hiprtcLinkState hip_link_state_ptr;

    // TODO: Dynamic linking is to be supported through hiprtc. Currently the
    // interface is limited and lacks support for linking globals. Indicative
    // code here is for future re-visit.
#if DYNAMIC_LINK
    std::vector<hiprtcJIT_option> LinkOptions = {
        HIPRTC_JIT_GLOBAL_SYMBOL_NAMES, HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS,
        HIPRTC_JIT_GLOBAL_SYMBOL_COUNT};
    std::vector<const char *> GlobalNames;
    std::vector<const void *> GlobalAddrs;
    for (auto RegisterVar : VarNameToDevPtr) {
      auto &VarName = RegisterVar.first;
      auto DevPtr = RegisterVar.second;
      GlobalNames.push_back(VarName.c_str());
      GlobalAddrs.push_back(DevPtr);
    }

    std::size_t GlobalSize = GlobalNames.size();
    std::size_t NumOptions = LinkOptions.size();
    const void *LinkOptionsValues[] = {GlobalNames.data(), GlobalAddrs.data(),
                                       (void *)&GlobalSize};
    hiprtcErrCheck(hiprtcLinkCreate(LinkOptions.size(), LinkOptions.data(),
                                    (void **)&LinkOptionsValues,
                                    &hip_link_state_ptr));

    hiprtcErrCheck(hiprtcLinkAddData(
        hip_link_state_ptr, HIPRTC_JIT_INPUT_LLVM_BITCODE,
        (void *)MemBuf.data(), MemBuf.size(), FnName.data(), LinkOptions.size(),
        LinkOptions.data(), (void **)&LinkOptionsValues));
#endif

    void *BinOut;
    size_t BinSize;
    {
      TIMESCOPE("Device linker")
      hiprtcErrCheck(
          hiprtcLinkCreate(0, nullptr, nullptr, &hip_link_state_ptr));
      hiprtcErrCheck(hiprtcLinkAddData(hip_link_state_ptr,
                                       HIPRTC_JIT_INPUT_LLVM_BITCODE,
                                       (void *)MemBuf.data(), MemBuf.size(),
                                       FnName.data(), 0, nullptr, nullptr));
      hiprtcErrCheck(hiprtcLinkComplete(hip_link_state_ptr, &BinOut, &BinSize));
    }
    {
      TIMESCOPE("Load module");
      hipErrCheck(hipModuleLoadData(&HipModule, BinOut));
    }

    // TODO: remove, this is obsolete, version 5.7.1 accepts KernelArgs without
    // going through the hoops. This code is not correct if arguments are not
    // pointers or scalars of the same size as pointers.
#if 0
    std::vector<void *>ArgBuffer(F->arg_size());
    for(size_t I = 0; I < ArgBuffer.size() ; ++I)
      memcpy(&ArgBuffer[I], KernelArgs[I], sizeof(void *));

    size_t ArgBufferSize = ArgBuffer.size() * sizeof(void *);
    void *Config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ArgBuffer[0],
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &ArgBufferSize,
                      HIP_LAUNCH_PARAM_END};
#endif
    {
      TIMESCOPE("Module get function");
      hipErrCheck(hipModuleGetFunction(&HipFunction, HipModule,
                                       (FnName + "_jit").str().c_str()));
    }

    auto Ret = hipModuleLaunchKernel(
        HipFunction, GridDimX, GridDimY, GridDimZ, BlockDimX, BlockDimY,
        BlockDimZ, ShmemSize, (hipStream_t)Stream,
        /* kernel_args (unsupported)*/ KernelArgs, nullptr);
#else
#error "Device JIT requires ENABLE_HIP"
#endif
    return Ret;
  }

  void insertRegisterVar(const char *VarName, void *DevPtr) {
    VarNameToDevPtr[VarName] = DevPtr;
  }

private:
  JitEngineDevice() {}
  std::unordered_map<std::string, void *> VarNameToDevPtr;
};

// TODO: Guard with ENABLE_HIP.
// NOTE: A great mystery is: why does this work ONLY if HostAddr is a CONST
// void*
__attribute((used)) void __jit_register_var(const void *HostAddr,
                                            const char *VarName) {
  JitEngineDevice &Jit = JitEngineDevice::instance();
  void *DevPtr = nullptr;
  hipErrCheck(hipGetSymbolAddress(&DevPtr, HIP_SYMBOL(HostAddr)));
  Jit.insertRegisterVar(VarName, DevPtr);
}

__attribute__((used)) hipError_t
__jit_launch_kernel(char *KernelName, char *IR, int IRSize, RuntimeConstant *RC,
                    int NumRuntimeConstants, uint64_t GridDimXY,
                    uint32_t GridDimZ, uint64_t BlockDimXY, uint32_t BlockDimZ,
                    void **KernelArgs, uint64_t ShmemSize, void *Stream) {
  TIMESCOPE("__jit_launch_kernel");
  JitEngineDevice &Jit = JitEngineDevice::instance();
  DBG(dbgs() << "JIT Launch Kernel\n");
  uint32_t *GridDimPtr = (uint32_t *)&GridDimXY;
  uint32_t GridDimX = *GridDimPtr, GridDimY = *(GridDimPtr + 1);
  uint32_t *BlockDimPtr = (uint32_t *)&BlockDimXY;
  uint32_t BlockDimX = *BlockDimPtr, BlockDimY = *(BlockDimPtr + 1);
  DBG(dbgs() << "=== Kernel Info\n");
  DBG(dbgs() << "KernelName " << KernelName << "\n");
  DBG(dbgs() << "Grid " << GridDimX << ", " << GridDimY << ", " << GridDimZ
             << "\n");
  DBG(dbgs() << "Block " << BlockDimX << ", " << BlockDimY << ", " << BlockDimZ
             << "\n");
  DBG(dbgs() << "KernelArgs " << KernelArgs << "\n");
  DBG(dbgs() << "ShmemSize " << ShmemSize << "\n");
  DBG(dbgs() << "Stream " << Stream << "\n");
  DBG(dbgs() << "=== End Kernel Info\n");
  StringRef StrIR(IR, IRSize);
  return Jit.compileAndRun(KernelName, IR, IRSize, RC, NumRuntimeConstants,
                           GridDimX, GridDimY, GridDimZ, BlockDimX, BlockDimY,
                           BlockDimZ, KernelArgs, ShmemSize, Stream);
}
}
