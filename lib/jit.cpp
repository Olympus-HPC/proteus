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
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
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
#include "llvm/Object/ELF.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <filesystem>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <iostream>
#include <memory>
#include <string>

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

#if ENABLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvPTXCompiler.h>

#define cudaErrCheck(CALL)                                                     \
  {                                                                            \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      abort();                                                                 \
    }                                                                          \
  }

#define cuErrCheck(CALL)                                                       \
  {                                                                            \
    CUresult err = CALL;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *ErrStr;                                                      \
      cuGetErrorString(err, &ErrStr);                                          \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__, ErrStr);            \
      abort();                                                                 \
    }                                                                          \
  }

#define nvPTXCompilerErrCheck(CALL)                                            \
  {                                                                            \
    nvPTXCompileResult err = CALL;                                             \
    if (err != NVPTXCOMPILE_SUCCESS) {                                         \
      printf("ERROR @ %s:%d ->  %d\n", __FILE__, __LINE__, err);               \
      abort();                                                                 \
    }                                                                          \
  }

#endif

//  #define ENABLE_PERFMAP

#if ENABLE_DEBUG
#define DBG(x) x;
#else
#define DBG(x)
#endif

#define FATAL_ERROR(x)                                                         \
  report_fatal_error(Twine(std::string{} + __FILE__ + ":" +                    \
                           std::to_string(__LINE__) + " => " + x))

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

#if ENABLE_TIME_TRACING
TimeTracerRAII TimeTracer;
#define TIMESCOPE(x) TimeTraceScope T(x);
#else
#define TIMESCOPE(x)
#endif

static inline Error createSMDiagnosticError(llvm::SMDiagnostic &Diag) {
  std::string Msg;
  {
    raw_string_ostream OS(Msg);
    Diag.print("", OS);
  }
  return make_error<StringError>(std::move(Msg), inconvertibleErrorCode());
}

static Expected<std::unique_ptr<TargetMachine>>
createTargetMachine(Module &M, StringRef CPU /*, unsigned OptLevel*/) {
  Triple TT(M.getTargetTriple());
  // TODO: Parameterize optlevel.
  CodeGenOpt::Level CGOptLevel = CodeGenOpt::Aggressive;

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());

  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(TT);

  std::optional<Reloc::Model> RelocModel;
  if (M.getModuleFlag("PIC Level"))
    RelocModel =
        M.getPICLevel() == PICLevel::NotPIC ? Reloc::Static : Reloc::PIC_;

  std::optional<CodeModel::Model> CodeModel = M.getCodeModel();

  TargetOptions Options = codegen::InitTargetOptionsFromCodeGenFlags(TT);

  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(M.getTargetTriple(), CPU, Features.getString(),
                             Options, RelocModel, CodeModel, CGOptLevel));
  if (!TM)
    return make_error<StringError>("Failed to create target machine",
                                   inconvertibleErrorCode());
  return TM;
}

// A function object that creates a simple pass pipeline to apply to each
// module as it passes through the IRTransformLayer.
class OptimizationTransform {
public:
  OptimizationTransform() {}

  Expected<ThreadSafeModule> operator()(ThreadSafeModule TSM,
                                        MaterializationResponsibility &R) {
    TSM.withModuleDo([this](Module &M) {
      DBG(dbgs() << "=== Begin Before Optimization\n"
                 << M << "=== End Before\n");
      TIMESCOPE("Run Optimization Transform");
      PassBuilder PB;
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;

      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      ModulePassManager Passes =
          PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
      Passes.run(M, MAM);
      DBG(dbgs() << "=== Begin After Optimization\n"
                 << M << "=== End After Optimization\n");
#if ENABLE_DEBUG
      if (verifyModule(M, &errs()))
        FATAL_ERROR(
            "Broken module found after optimization, JIT compilation aborted!");
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
      TIMESCOPE("Run Optimization Transform");
      PassBuilder PB;
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;

      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      ModulePassManager Passes =
          PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
      Passes.run(M, MAM);
      DBG(dbgs() << "=== Begin After Optimization\n"
                 << M << "=== End After Optimization\n");
#if ENABLE_DEBUG
      if (verifyModule(M, &errs()))
        FATAL_ERROR(
            "Broken module found after optimization, JIT compilation aborted!");
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
    // TODO: This allows pointer as runtime constant values. How useful is that?
    void *PtrVal;
  };
};

inline hash_code hash_value(const RuntimeConstant &RC) {
  return hash_value(RC.Int64Val);
}

// TODO: check if this global is needed.
static codegen::RegisterCodeGenFlags CFG;

struct Codegen {
  static void relinkGlobals(
      Module &M,
      std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
    // Re-link globals to fixed addresses provided by registered
    // variables.
    for (auto RegisterVar : VarNameToDevPtr) {
#if ENABLE_HIP
      const void *DevPtr = RegisterVar.second;
#elif ENABLE_CUDA
      // For CUDA we must defer resolving the global symbol address here
      // instead when registering the variable in the constructor context.
      void *DevPtr = nullptr;
      cudaErrCheck(cudaGetSymbolAddress(&DevPtr, RegisterVar.second));
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif
      assert(DevPtr && "Expected non-null device pointer for global");
      auto &VarName = RegisterVar.first;
      auto *GV = M.getNamedGlobal(VarName);
      assert(GV && "Expected existing global variable");
      // Remove the re-linked global from llvm.compiler.used since that
      // use is not replaceable by the fixed addr constant expression.
      removeFromUsedLists(M, [&GV](Constant *C) {
        if (GV == C)
          return true;

        return false;
      });

      Constant *Addr =
          ConstantInt::get(Type::getInt64Ty(M.getContext()), (uint64_t)DevPtr);
      Value *CE = ConstantExpr::getIntToPtr(Addr, GV->getType());
      GV->replaceAllUsesWith(CE);
    }

#if ENABLE_DEBUG
    dbgs() << "=== Linked M\n" << M << "=== End of Linked M\n";
    if (verifyModule(M, &errs()))
      FATAL_ERROR(
          "After linking, broken module found, JIT compilation aborted!");
    else
      dbgs() << "Module verified!\n";
#endif
  }
};

// NOTE: Stored cache assumes that stored code is re-usable across runs!
// Source code changes should invalidate the cache (TODO). Also, if
// storing assembly (PTX) or binary (ELF), then device globals may
// have different addresses that render it invalid. In this case, store LLVM IR
// to re-link globals.
template <typename Function_t> class JitStoredCache {
public:
  Function_t lookup(uint64_t HashValue, StringRef Kernel) {
    TIMESCOPE("object lookup");
    Accesses++;

    Function_t DevFunction;
#if ENABLE_LLVMIR_STORED_CACHE
#error Unsupported yet
#endif
#if ENABLE_HIP
    hipModule_t HipModule;
    //  Load module from file.
    auto Err = hipModuleLoad(
        &HipModule, ("cache-jit-" + std::to_string(HashValue) + ".o").c_str());
    if (Err == hipErrorFileNotFound)
      return nullptr;

    hipErrCheck(Err);

    // Get function from loaded module.
    const std::string Suffix = "$jit$" + std::to_string(HashValue) + "$";
    const std::string KernelMangled = Kernel.str() + Suffix;

    hipErrCheck(
        hipModuleGetFunction(&DevFunction, HipModule, KernelMangled.c_str()));
#elif ENABLE_CUDA
    CUmodule CUMod;
    auto Err = cuModuleLoad(
        &CUMod, ("cache-jit-" + std::to_string(HashValue) + ".o").c_str());

    if (Err == CUDA_ERROR_FILE_NOT_FOUND)
      return nullptr;

    cuErrCheck(Err);

    // Get function from loaded module.
    const std::string Suffix = "$jit$" + std::to_string(HashValue) + "$";
    const std::string KernelMangled = Kernel.str() + Suffix;

    cuErrCheck(cuModuleGetFunction(&DevFunction, CUMod, KernelMangled.c_str()));
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined'
#endif

    // Return function.
    Hits++;
    return DevFunction;
  }

  void printStats() {
    // Use printf to avoid re-ordering outputs by outs() in HIP.
    printf("JitStoredCache hits %lu total %lu\n", Hits, Accesses);
  }

private:
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
};

template <typename Function_t> class JitCache {
public:
  uint64_t hash(StringRef ModuleUniqueId, StringRef FnName,
                const RuntimeConstant *RC, int NumRuntimeConstants) const {
    ArrayRef<RuntimeConstant> Data(RC, NumRuntimeConstants);
    auto HashValue = hash_combine(ExePath, ModuleUniqueId, FnName, Data);
    return HashValue;
  }

  Function_t lookup(uint64_t HashValue) {
    TIMESCOPE("lookup");
    Accesses++;

    auto It = CacheMap.find(HashValue);
    if (It == CacheMap.end())
      return nullptr;

    It->second.NumExecs++;
    It->second.NumHits++;
    Hits++;
    return It->second.FunctionPtr;
  }

  void insert(uint64_t HashValue, Function_t FunctionPtr
#if ENABLE_DEBUG
              ,
              StringRef FnName, RuntimeConstant *RC, int NumRuntimeConstants
#endif
  ) {
#if ENABLE_DEBUG
    if (CacheMap.count(HashValue))
      FATAL_ERROR("JitCache collision detected");
#endif

    CacheMap[HashValue] = {FunctionPtr, /* num_execs */ 1};

#if ENABLE_DEBUG
    CacheMap[HashValue].FnName = FnName.str();
    for (size_t I = 0; I < NumRuntimeConstants; ++I)
      CacheMap[HashValue].RCVector.push_back(RC[I]);
#endif
  }

  void printStats() {
    // outs() << "JitCache hits " << Hits << " total " << Accesses << "\n";
    // Use printf to avoid re-ordering outputs by outs() in HIP.
    printf("JitCache hits %lu total %lu\n", Hits, Accesses);
    for (auto &It : CacheMap) {
      uint64_t HashValue = It.first;
      JitCacheEntry &JCE = It.second;
      // outs() << "HashValue " << HashValue << " num_execs " <<
      // JCE.NumExecs;
      printf("HashValue %lu NumExecs %lu NumHits %lu", HashValue, JCE.NumExecs,
             JCE.NumHits);
#if ENABLE_DEBUG
      // outs() << " FnName " << JCE.FnName << " RCs [";
      printf(" FnName %s RCs [", JCE.FnName.c_str());
      for (auto &RC : JCE.RCVector)
        // outs() << RC.Int64Val << ", ";
        printf("%ld, ", RC.Int64Val);
      // outs() << "]";
      printf("]");
#endif
      // outs() << "\n";
      printf("\n");
    }
  }

  JitCache() {
    // NOTE: Linux-specific.
    ExePath = std::filesystem::canonical("/proc/self/exe");
  }

private:
  struct JitCacheEntry {
    Function_t FunctionPtr;
    uint64_t NumExecs;
    uint64_t NumHits;
#if ENABLE_DEBUG
    std::string FnName;
    SmallVector<RuntimeConstant, 8> RCVector;
#endif
  };

  DenseMap<uint64_t, JitCacheEntry> CacheMap;
  // Use the executable binary path when hashing to differentiate between
  // same-named kernels generated by other executables.
  std::filesystem::path ExePath;
  uint64_t Hits = 0;
  uint64_t Accesses = 0;
};

class JitEngine {
public:
  std::unique_ptr<LLJIT> LLJITPtr;
  ExitOnError ExitOnErr;

  static JitEngine &instance() {
    static JitEngine Jit(0, (char *[]){nullptr});
    return Jit;
  }

  static void
  dumpSymbolInfo(const llvm::object::ObjectFile &loadedObj,
                 const llvm::RuntimeDyld::LoadedObjectInfo &objInfo) {
    // Dump information about symbols.
    auto pid = sys::Process::getProcessId();
    std::error_code EC;
    raw_fd_ostream ofd("/tmp/perf-" + std::to_string(pid) + ".map", EC,
                       sys::fs::OF_Append);
    if (EC)
      FATAL_ERROR("Cannot open perf map file");
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

  ~JitEngine() { CodeCache.printStats(); }

  Expected<llvm::orc::ThreadSafeModule>
  parseBitcode(StringRef FnName, StringRef Suffix, StringRef IR,
               RuntimeConstant *RC, int NumRuntimeConstants) {

    TIMESCOPE("parseBitcode");
    auto Ctx = std::make_unique<LLVMContext>();
    SMDiagnostic Err;
    if (auto M = parseIR(MemoryBufferRef(IR, ("Mod-" + FnName + Suffix).str()),
                         Err, *Ctx)) {
      // dbgs() << "=== Parsed Module\n" << *M << "=== End of Parsed
      // Module\n";
      Function *F = M->getFunction(FnName);
      assert(F && "Expected non-null function!");
      MDNode *Node = F->getMetadata("jit_arg_nos");
      DBG(dbgs() << "Metadata jit for F " << F->getName() << " = " << *Node
                 << "\n");

      // Replace argument uses with runtime constants.
      // TODO: Env var to enable runtime constprop.
#if ENABLE_RUNTIME_CONSTPROP
      for (int I = 0; I < Node->getNumOperands(); ++I) {
        ConstantAsMetadata *CAM = cast<ConstantAsMetadata>(Node->getOperand(I));
        ConstantInt *ConstInt = cast<ConstantInt>(CAM->getValue());
        int ArgNo = ConstInt->getZExtValue();
        Value *Arg = F->getArg(ArgNo);
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
          FATAL_ERROR("JIT Incompatible type in runtime constant");

        Arg->replaceAllUsesWith(C);
      }
#endif

      // dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n";

      F->setName(FnName + Suffix);

#if ENABLE_DEBUG
      dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
      if (verifyModule(*M, &errs()))
        FATAL_ERROR("Broken module found, JIT compilation aborted!");
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
    void *JitFnPtr = CodeCache.lookup(HashValue);
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
    CodeCache.insert(HashValue, JitFnPtr
#if ENABLE_DEBUG
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

  JitCache<void *> CodeCache;
};

extern "C" {
__attribute__((used)) void *__jit_entry(char *FnName, char *IR, int IRSize,
                                        RuntimeConstant *RC,
                                        int NumRuntimeConstants) {
  TIMESCOPE("__jit_entry");
  JitEngine &Jit = JitEngine::instance();
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

struct FatbinWrapper_t {
  int32_t Magic;
  int32_t Version;
  const char *Binary;
  void *X;
};

static void runOptimizationPassPipeline(Module &M, StringRef CPU) {
  TIMESCOPE("Run opt passes");
  PipelineTuningOptions PTO;

  std::optional<PGOOptions> PGOOpt;
  auto TM = createTargetMachine(M, CPU);
  if (auto Err = TM.takeError())
    report_fatal_error(std::move(Err));
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

  PassBuilder PB(TM->get(), PTO, PGOOpt, nullptr);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager Passes =
      PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
  Passes.run(M, MAM);
}

class JitEngineDevice {
public:
  static JitEngineDevice &instance() {
    static JitEngineDevice Jit{};
    return Jit;
  }

  ~JitEngineDevice() {
    CodeCache.printStats();
    StoredCache.printStats();
  }

  std::unique_ptr<MemoryBuffer> extractDeviceBitcodeHip(StringRef KernelName,
                                                        const char *Binary) {
    constexpr char OFFLOAD_BUNDLER_MAGIC_STR[] = "__CLANG_OFFLOAD_BUNDLE__";
    size_t Pos = 0;
    StringRef Magic(Binary, sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
    if (!Magic.equals(OFFLOAD_BUNDLER_MAGIC_STR))
      FATAL_ERROR("Error missing magic string");
    Pos += sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;

    auto Read8ByteIntLE = [](const char *S, size_t Pos) {
      return llvm::support::endian::read64le(S + Pos);
    };

    uint64_t NumberOfBundles = Read8ByteIntLE(Binary, Pos);
    Pos += 8;
    DBG(dbgs() << "NumberOfbundles " << NumberOfBundles << "\n");

    StringRef DeviceBinary;
    for (uint64_t i = 0; i < NumberOfBundles; ++i) {
      uint64_t Offset = Read8ByteIntLE(Binary, Pos);
      Pos += 8;

      uint64_t Size = Read8ByteIntLE(Binary, Pos);
      Pos += 8;

      uint64_t TripleSize = Read8ByteIntLE(Binary, Pos);
      Pos += 8;

      StringRef Triple(Binary + Pos, TripleSize);
      Pos += TripleSize;

      DBG(dbgs() << "Offset " << Offset << "\n");
      DBG(dbgs() << "Size " << Size << "\n");
      DBG(dbgs() << "TripleSize " << TripleSize << "\n");
      DBG(dbgs() << "Triple " << Triple << "\n");

      if (!Triple.contains("amdgcn"))
        continue;

      DeviceBinary = StringRef(Binary + Offset, Size);
      break;
    }

#if ENABLE_DEBUG
    {
      std::error_code EC;
      raw_fd_ostream OutBin("device.bin", EC);
      if (EC)
        FATAL_ERROR("Cannot open device binary file");
      OutBin << DeviceBinary;
      OutBin.close();
      dbgs() << "Binary image found\n";
    }
#endif

    Expected<object::ELF64LEFile> DeviceElf =
        llvm::object::ELF64LEFile::create(DeviceBinary);
    if (DeviceElf.takeError())
      FATAL_ERROR("Cannot create the device elf");

    auto Sections = DeviceElf->sections();
    if (Sections.takeError())
      FATAL_ERROR("Error reading sections");

    // NOTE: We must extract the .jit sections per kernel to avoid linked
    // device libraries. Otherwise, the hiprtc linker complains that it
    // cannot link device libraries (working assumption).
    ArrayRef<uint8_t> DeviceBitcode;
    Twine TargetSection = ".jit." + KernelName;
    for (auto Section : *Sections) {
      auto SectionName = DeviceElf->getSectionName(Section);
      if (SectionName.takeError())
        FATAL_ERROR("Error reading section name");
      DBG(dbgs() << "SectionName " << *SectionName << "\n");
      DBG(dbgs() << "TargetSection " << TargetSection << "\n");
      if (!SectionName->equals(TargetSection.str()))
        continue;

      auto SectionContents = DeviceElf->getSectionContents(Section);
      if (SectionContents.takeError())
        FATAL_ERROR("Error reading section contents");

      DeviceBitcode = *SectionContents;
    }

    if (DeviceBitcode.empty())
      FATAL_ERROR("Error finding the device bitcode");

#if ENABLE_DEBUG
    {
      std::error_code EC;
      raw_fd_ostream OutBC(Twine(".jit." + KernelName + ".bc").str(), EC);
      if (EC)
        FATAL_ERROR("Cannot open device bitcode file");
      OutBC << StringRef(reinterpret_cast<const char *>(DeviceBitcode.data()),
                         DeviceBitcode.size());
      OutBC.close();
    }
#endif

    return MemoryBuffer::getMemBufferCopy(
        StringRef(reinterpret_cast<const char *>(DeviceBitcode.data()),
                  DeviceBitcode.size()));
  }

#if ENABLE_CUDA
  std::unique_ptr<MemoryBuffer> extractDeviceBitcodeCuda(StringRef KernelName,
                                                         const char *Binary,
                                                         size_t FatbinSize) {
    CUmodule CUMod;
    CUlinkState CULinkState = nullptr;
    CUdeviceptr DevPtr;
    size_t Bytes;
    std::string Symbol = Twine("__jit_bc_" + KernelName).str();

    // NOTE: loading a module OR getting the global fails if rdc compilation
    // is enabled. Try to use the linker interface to load the binary image.
    // If that fails too, abort.
    // TODO: detect rdc compilation in the JitPass, see
    // __cudaRegisterLinkedLibrary or __nv_relfatbin section existences.
    if (cuModuleLoadFatBinary(&CUMod, Binary) != CUDA_SUCCESS ||
        cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, Symbol.c_str()) ==
            CUDA_ERROR_NOT_FOUND) {
      cuErrCheck(cuLinkCreate(0, nullptr, nullptr, &CULinkState));
      cuErrCheck(cuLinkAddData(CULinkState, CU_JIT_INPUT_FATBINARY,
                               (void *)Binary, FatbinSize, "", 0, 0, 0));
      void *BinOut;
      size_t BinSize;
      cuErrCheck(cuLinkComplete(CULinkState, &BinOut, &BinSize));
      cuErrCheck(cuModuleLoadFatBinary(&CUMod, BinOut));
    }

    cuErrCheck(cuModuleGetGlobal(&DevPtr, &Bytes, CUMod, Symbol.c_str()));

    SmallString<4096> DeviceBitcode;
    DeviceBitcode.reserve(Bytes);
    cuErrCheck(cuMemcpyDtoH(DeviceBitcode.data(), DevPtr, Bytes));
#ifdef ENABLE_DEBUG
    {
      std::error_code EC;
      raw_fd_ostream OutBC(Twine("from-device-jit-" + KernelName + ".bc").str(),
                           EC);
      if (EC)
        FATAL_ERROR("Cannot open device memory jit file");
      OutBC << StringRef(DeviceBitcode.data(), Bytes);
      OutBC.close();
    }
#endif

    cuErrCheck(cuModuleUnload(CUMod));
    if (CULinkState)
      cuErrCheck(cuLinkDestroy(CULinkState));
    return MemoryBuffer::getMemBufferCopy(
        StringRef(DeviceBitcode.data(), Bytes));
  }
#endif

  Expected<llvm::orc::ThreadSafeModule>
  parseBitcode(StringRef FnName, StringRef Suffix, StringRef IR, int BlockSize,
               int GridSize, RuntimeConstant *RC, int NumRuntimeConstants) {

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
      assert(Node && "Expected metadata for jit arguments");
      DBG(dbgs() << "Metadata jit for F " << F->getName() << " = " << *Node
                 << "\n");

      // Relink device globals.
      Codegen::relinkGlobals(*M, VarNameToDevPtr);

      // Replace argument uses with runtime constants.
#if ENABLE_RUNTIME_CONSTPROP
      if (Config.ENV_JIT_RUNTIME_CONSTPROP)
        for (int I = 0; I < Node->getNumOperands(); ++I) {
          ConstantAsMetadata *CAM =
              cast<ConstantAsMetadata>(Node->getOperand(I));
          ConstantInt *ConstInt = cast<ConstantInt>(CAM->getValue());
          int ArgNo = ConstInt->getZExtValue();
          Value *Arg = F->getArg(ArgNo);
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
            auto *IntC =
                ConstantInt::get(Type::getInt64Ty(*Ctx), RC[I].Int64Val);
            C = ConstantExpr::getIntToPtr(IntC, ArgType);
          } else
            FATAL_ERROR("JIT Incompatible type in runtime constant");

          Arg->replaceAllUsesWith(C);
        }
#endif

      DBG(dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n");

      F->setName(FnName + Suffix);

#if ENABLE_JIT_LAUNCH_BOUNDS
      if (Config.ENV_JIT_LAUNCH_BOUNDS) {
// TODO: Launch bounds for cuda.
// TODO: Environment variable to enable launch bounds.
#if ENABLE_HIP
        // TODO: fix calculation of launch bounds.
        // TODO: find maximum (hardcoded 1024) from device info.
        // TODO: Setting as 1, BlockSize to replicate launch bounds settings
        // Does setting it as BlockSize, BlockSize help?
        F->addFnAttr("amdgpu-flat-work-group-size",
                     "1," + std::to_string(std::min(1024, BlockSize)));
        // TODO: find warp size (hardcoded 64) from device info.
        // int WavesPerEU = (GridSize * BlockSize) / 64 / 110 / 4 / 2;
        int WavesPerEU = 0;
        // F->addFnAttr("amdgpu-waves-per-eu", std::to_string(WavesPerEU));
        DBG(dbgs() << "BlockSize " << BlockSize << " GridSize " << GridSize
                   << " => Set Wokgroup size " << BlockSize
                   << " WavesPerEU (unused) " << WavesPerEU << "\n");

#elif ENABLE_CUDA
        NamedMDNode *NvvmAnnotations = M->getNamedMetadata("nvvm.annotations");
        assert(NvvmAnnotations &&
               "Expected non-null nvvm.annotations metadata");
        // TODO: fix hardcoded 1024 as the maximum, by reading device
        // properties.
        // TODO: set min GridSize.
        int MaxThreads = std::min(1024, BlockSize);
        llvm::Metadata *MDVals[] = {
            llvm::ConstantAsMetadata::get(F),
            llvm::MDString::get(M->getContext(), "maxntidx"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(M->getContext()), MaxThreads))};
        NvvmAnnotations->addOperand(llvm::MDNode::get(M->getContext(), MDVals));
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined'
#endif
      }
#endif

#if ENABLE_DEBUG
      dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
      if (verifyModule(*M, &errs()))
        FATAL_ERROR("Broken module found, JIT compilation aborted!");
      else
        dbgs() << "Module verified!\n";
#endif
      return ThreadSafeModule(std::move(M), std::move(Ctx));
    }

    return createSMDiagnosticError(Err);
  }

  void storeObjectToStoredCache(StringRef ObjectRef, uint64_t HashValue) {
    TIMESCOPE("Store object");
    if (Config.ENV_JIT_USE_STORED_CACHE) {
      std::error_code EC;
      raw_fd_ostream OutBin(
          Twine("cache-jit-" + std::to_string(HashValue) + ".o").str(), EC);
      if (EC)
        FATAL_ERROR("Cannot open device object file");
      OutBin << ObjectRef;
      OutBin.close();
    }
  }

#if ENABLE_HIP
  hipError_t codegenAndLaunchHip(Module *M, StringRef HipArch,
                                 StringRef KernelName, StringRef Suffix,
                                 uint64_t HashValue, RuntimeConstant *RC,
                                 int NumRuntimeConstants, dim3 GridDim,
                                 dim3 BlockDim, void **KernelArgs,
                                 uint64_t ShmemSize, hipStream_t Stream) {
    // Remove extras to get a working CPU architecture value, e.g., from
    // gfx90a:sramecc+:xnack- drop everything after the first :.
    // HipArch = HipArch.substr(0, HipArch.find_first_of(":"));
    // TODO: Do not run optimization pipeline for hip, hiprtc adds O3 by
    // default. Also, need to fine-tune the pipeline: issue with libor where
    // aggressive unrolling creates huge, slow binary code.
    // runOptimizationPassPipeline(*M, HipArch);
#if ENABLE_DEBUG
    {
      if (verifyModule(*M, &errs()))
        FATAL_ERROR("Broken module found after optimization, JIT "
                    "compilation aborted!");
      std::error_code EC;
      raw_fd_ostream OutBC(
          Twine("opt-transformed-jit-" + KernelName + Suffix + ".bc").str(),
          EC);
      if (EC)
        FATAL_ERROR("Cannot open device transformed bitcode file");
      // TODO: Remove or leave it only for debugging.
      OutBC << *M;
      OutBC.close();
    }
#endif

    SmallString<4096> ModuleBuffer;
    raw_svector_ostream ModuleBufferOS(ModuleBuffer);
    WriteBitcodeToFile(*M, ModuleBufferOS);

    char *BinOut;
    size_t BinSize;
    hipModule_t HipModule;

    hiprtcLinkState hip_link_state_ptr;

    // TODO: Dynamic linking is to be supported through hiprtc. Currently
    // the interface is limited and lacks support for linking globals.
    // Indicative code here is for future re-visit.
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
        (void *)ModuleBuffer.data(), ModuleBuffer.size(), KernelName.data(),
        LinkOptions.size(), LinkOptions.data(), (void **)&LinkOptionsValues));
#endif

    {
      TIMESCOPE("Device linker");
// #if CUSTOM_OPTIONS
// TODO: Toggle this with an env var.
#if 1
      // NOTE: This code is an example of passing custom, AMD-specific
      // options to the compiler/linker. NOTE: Unrolling can have a dramatic
      // (time-consuming) effect on JIT compilation time and on the
      // resulting optimization, better or worse depending on code
      // specifics. const char *OptArgs[] = {"-mllvm",
      // "-amdgpu-internalize-symbols",
      //                         "-save-temps", "-mllvm",
      //                         "-unroll-threshold=100"};
      const char *OptArgs[] = {"-mllvm", "-amdgpu-internalize-symbols",
                               "-mllvm", "-unroll-threshold=1000",
                               "-march=gfx90a"};
      std::vector<hiprtcJIT_option> JITOptions = {
          HIPRTC_JIT_IR_TO_ISA_OPT_EXT, HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
      size_t OptArgsSize = 5;
      const void *JITOptionsValues[] = {(void *)OptArgs, (void *)(OptArgsSize)};
      hiprtcErrCheck(hiprtcLinkCreate(JITOptions.size(), JITOptions.data(),
                                      (void **)JITOptionsValues,
                                      &hip_link_state_ptr));
#else
      hiprtcErrCheck(
          hiprtcLinkCreate(0, nullptr, nullptr, &hip_link_state_ptr));
#endif

      hiprtcErrCheck(
          hiprtcLinkAddData(hip_link_state_ptr, HIPRTC_JIT_INPUT_LLVM_BITCODE,
                            (void *)ModuleBuffer.data(), ModuleBuffer.size(),
                            KernelName.data(), 0, nullptr, nullptr));
      hiprtcErrCheck(
          hiprtcLinkComplete(hip_link_state_ptr, (void **)&BinOut, &BinSize));
    }
    {
      TIMESCOPE("Load module");
      hipErrCheck(hipModuleLoadData(&HipModule, BinOut));
    }

    hipFunction_t HipFunction;
    {
      TIMESCOPE("Module get function");
      hipErrCheck(hipModuleGetFunction(&HipFunction, HipModule,
                                       (KernelName + Suffix).str().c_str()));
    }
    CodeCache.insert(HashValue, HipFunction
#if ENABLE_DEBUG
                     ,
                     KernelName, RC, NumRuntimeConstants
#endif
    );

    StringRef ObjectRef(BinOut, BinSize);
    storeObjectToStoredCache(ObjectRef, HashValue);

    hipErrCheck(hipModuleLaunchKernel(
        HipFunction, GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y,
        BlockDim.z, ShmemSize, (hipStream_t)Stream, KernelArgs, nullptr));
    return hipSuccess;
  }
#endif

#if ENABLE_CUDA

  void codegenPTX(Module &M, StringRef CudaArch,
                  SmallVectorImpl<char> &PTXStr) {
    TIMESCOPE("Codegen PTX");
    auto TMExpected = createTargetMachine(M, CudaArch);
    if (!TMExpected)
      FATAL_ERROR(toString(TMExpected.takeError()));

    std::unique_ptr<TargetMachine> TM = std::move(*TMExpected);
    TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));

    legacy::PassManager PM;
    PM.add(new TargetLibraryInfoWrapperPass(TLII));
    MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(
        reinterpret_cast<LLVMTargetMachine *>(TM.get()));

    raw_svector_ostream PTXOS(PTXStr);
    TM->addPassesToEmitFile(PM, PTXOS, nullptr, CGFT_AssemblyFile,
                            /* DisableVerify */ false, MMIWP);

    PM.run(M);
  }

  cudaError_t codegenAndLaunchCuda(Module *M, StringRef CudaArch,
                                   StringRef KernelName, StringRef Suffix,
                                   uint64_t HashValue, RuntimeConstant *RC,
                                   int NumRuntimeConstants, dim3 GridDim,
                                   dim3 BlockDim, void **KernelArgs,
                                   uint64_t ShmemSize, CUstream Stream) {
    // Codegen PTX, load the module and run through the CUDA PTX JIT
    // interface. Check this reference for JIT caching:
    // https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
    // Interesting env vars: CUDA_CACHE_DISABLE, CUDA_CACHE_MAXSIZE,
    // CUDA_CACHE_PATH, CUDA_FORCE_PTX_JIT.
    // For CUDA, run the target-specific optimization pipeline to optimize the
    // LLVM IR before handing over to the CUDA driver PTX compiler.
    runOptimizationPassPipeline(*M, CudaArch);

    SmallVector<char, 4096> PTXStr;
    SmallVector<char, 4096> FinalIR;
    size_t BinSize;

#if ENABLE_DEBUG
    {
      if (verifyModule(*M, &errs()))
        FATAL_ERROR("Broken module found after optimization, JIT "
                    "compilation aborted!");
      std::error_code EC;
      raw_fd_ostream OutBC(
          Twine("opt-transformed-jit-" + KernelName + Suffix + ".bc").str(),
          EC);
      if (EC)
        FATAL_ERROR("Cannot open device transformed bitcode file");
      OutBC << *M;
      OutBC.close();
    }
#endif

    codegenPTX(*M, CudaArch, PTXStr);

#if ENABLE_DEBUG
    {
      std::error_code EC;
      raw_fd_ostream OutPtx(
          Twine("jit-" + std::to_string(HashValue) + ".ptx").str(), EC);
      if (EC)
        FATAL_ERROR("Cannot open ptx output file");
      OutPtx << PTXStr;
      OutPtx.close();
    }
#endif

    CUmodule CUMod;
    CUfunction CUFunc;

    {
      TIMESCOPE("Create object");
      // CUDA requires null-terminated PTX.
      PTXStr.push_back('\0');
#if ENABLE_LLVMIR_STORED_CACHE
      {
        raw_svector_ostream IROS(FinalIR);
        WriteBitcodeToFile(*M, IROS);
      }
      StringRef ObjectRef(FinalIR.data(), FinalIR.size());
#elif ENABLE_CUDA_PTX_STORED_CACHE
      cuErrCheck(cuModuleLoadData(&CUMod, PTXStr.data()));
      cuErrCheck(cuModuleGetFunction(&CUFunc, CUMod,
                                     Twine(KernelName + Suffix).str().c_str()));
      StringRef ObjectRef(PTXStr.data(), PTXStr.size());
#else
      // Store ELF object.
      nvPTXCompilerHandle PTXCompiler;
      nvPTXCompilerErrCheck(
          nvPTXCompilerCreate(&PTXCompiler, PTXStr.size(), PTXStr.data()));
      std::string ArchOpt = ("--gpu-name=" + CudaArch).str();
#if ENABLE_DEBUG
      const char *CompileOptions[] = {ArchOpt.c_str(), "--verbose"};
      size_t NumCompileOptions = 2;
#else
      const char *CompileOptions[] = {ArchOpt.c_str()};
      size_t NumCompileOptions = 1;
#endif
      nvPTXCompilerErrCheck(
          nvPTXCompilerCompile(PTXCompiler, NumCompileOptions, CompileOptions));
      nvPTXCompilerErrCheck(
          nvPTXCompilerGetCompiledProgramSize(PTXCompiler, &BinSize));
      auto BinOut = std::make_unique<char[]>(BinSize);
      nvPTXCompilerErrCheck(
          nvPTXCompilerGetCompiledProgram(PTXCompiler, BinOut.get()));

#if ENABLE_DEBUG
      {
        size_t LogSize;
        nvPTXCompilerErrCheck(
            nvPTXCompilerGetInfoLogSize(PTXCompiler, &LogSize));
        auto Log = std::make_unique<char[]>(LogSize);
        nvPTXCompilerErrCheck(nvPTXCompilerGetInfoLog(PTXCompiler, Log.get()));
        dbgs() << "=== nvPTXCompiler Log\n" << Log.get() << "\n";
      }
#endif
      nvPTXCompilerErrCheck(nvPTXCompilerDestroy(&PTXCompiler));

      cuErrCheck(cuModuleLoadData(&CUMod, BinOut.get()));
      cuErrCheck(cuModuleGetFunction(&CUFunc, CUMod,
                                     Twine(KernelName + Suffix).str().c_str()));

      StringRef ObjectRef(BinOut.get(), BinSize);
#endif

      CodeCache.insert(HashValue, CUFunc
#if ENABLE_DEBUG
                       ,
                       KernelName, RC, NumRuntimeConstants
#endif
      );

      storeObjectToStoredCache(ObjectRef, HashValue);
    }

    cuLaunchKernel(CUFunc, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                   BlockDim.y, BlockDim.z, ShmemSize, (CUstream)Stream,
                   KernelArgs, nullptr);
    // TODO: cuModuleUnload and ctxCtxDestroy at program exit.
    return cudaGetLastError();
  }
#endif

#if ENABLE_HIP
  hipError_t
#elif ENABLE_CUDA
  cudaError_t
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined'
#endif
  compileAndRun(StringRef ModuleUniqueId, StringRef KernelName,
                FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
                RuntimeConstant *RC, int NumRuntimeConstants, dim3 GridDim,
                dim3 BlockDim, void **KernelArgs, uint64_t ShmemSize,
                void *Stream) {
    TIMESCOPE("compileAndRun");

    uint64_t HashValue =
        CodeCache.hash(ModuleUniqueId, KernelName, RC, NumRuntimeConstants);
#if ENABLE_HIP
    hipFunction_t HipFunction = CodeCache.lookup(HashValue);
    if (HipFunction)
      return hipModuleLaunchKernel(
          HipFunction, GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y,
          BlockDim.z, ShmemSize, (hipStream_t)Stream, KernelArgs, nullptr);

    if (Config.ENV_JIT_USE_STORED_CACHE)
      if ((HipFunction = StoredCache.lookup(HashValue, KernelName))) {
        CodeCache.insert(HashValue, HipFunction
#if ENABLE_DEBUG
                         ,
                         KernelName, RC, NumRuntimeConstants
#endif
        );

        return hipModuleLaunchKernel(HipFunction, GridDim.x, GridDim.y,
                                     GridDim.z, BlockDim.x, BlockDim.y,
                                     BlockDim.z, ShmemSize, (hipStream_t)Stream,
                                     KernelArgs, nullptr);
      }
#elif ENABLE_CUDA
    CUfunction CUFunc = CodeCache.lookup(HashValue);
    if (CUFunc) {
      cuLaunchKernel(CUFunc, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                     BlockDim.y, BlockDim.z, ShmemSize, (CUstream)Stream,
                     KernelArgs, nullptr);
      return cudaGetLastError();
    }

    if (Config.ENV_JIT_USE_STORED_CACHE)
      if ((CUFunc = StoredCache.lookup(HashValue, KernelName))) {
        CodeCache.insert(HashValue, CUFunc
#if ENABLE_DEBUG
                         ,
                         KernelName, RC, NumRuntimeConstants
#endif
        );

        cuLaunchKernel(CUFunc, GridDim.x, GridDim.y, GridDim.z, BlockDim.x,
                       BlockDim.y, BlockDim.z, ShmemSize, (CUstream)Stream,
                       KernelArgs, nullptr);
        return cudaGetLastError();
      }
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif

#if ENABLE_HIP
    hipDeviceProp_t devProp;
    hipErrCheck(hipGetDeviceProperties(&devProp, 0));
    auto IRBuffer = extractDeviceBitcodeHip(KernelName, FatbinWrapper->Binary);
#elif ENABLE_CUDA
    auto IRBuffer =
        extractDeviceBitcodeCuda(KernelName, FatbinWrapper->Binary, FatbinSize);
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif

    // NOTE: we don't need a suffix to differentiate kernels, each
    // specialization will be in its own module. It exists only for
    // debugging purposes to verify that the jitted kernel executes.
    std::string Suffix = "$jit$" + std::to_string(HashValue) + "$";
    auto TransformedBitcode = parseBitcode(
        KernelName, Suffix, IRBuffer->getBuffer(),
        BlockDim.x * BlockDim.y * BlockDim.z, GridDim.x * GridDim.y * GridDim.z,
        RC, NumRuntimeConstants);
    if (auto E = TransformedBitcode.takeError())
      FATAL_ERROR(toString(std::move(E)).c_str());

    Module *M = TransformedBitcode->getModuleUnlocked();

#if ENABLE_DEBUG
    {
      std::error_code EC;
      raw_fd_ostream OutBC(
          Twine("transformed-jit-" + KernelName + Suffix + ".bc").str(), EC);
      if (EC)
        FATAL_ERROR("Cannot open device transformed bitcode file");
      OutBC << *M;
      OutBC.close();
    }
#endif

#if ENABLE_HIP
    auto Ret = codegenAndLaunchHip(M, devProp.gcnArchName, KernelName, Suffix,
                                   HashValue, RC, NumRuntimeConstants, GridDim,
                                   BlockDim, KernelArgs, ShmemSize,
                                   (hipStream_t)Stream);

#elif ENABLE_CUDA
    auto Ret = codegenAndLaunchCuda(M, CudaArch, KernelName, Suffix, HashValue,
                                    RC, NumRuntimeConstants, GridDim, BlockDim,
                                    KernelArgs, ShmemSize, (CUstream)Stream);

#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif

    return Ret;
  }

  void insertRegisterVar(const char *VarName, const void *Addr) {
    VarNameToDevPtr[VarName] = Addr;
  }

private:
  JitEngineDevice() {
#if ENABLE_HIP
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
#elif ENABLE_CUDA
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    // Initialize CUDA and retrieve the compute capability, needed for later
    // operations.
    CUdevice CUDev;
    CUcontext CUCtx;

    cuErrCheck(cuInit(0));

    CUresult CURes = cuCtxGetDevice(&CUDev);
    if (CURes == CUDA_ERROR_INVALID_CONTEXT or !CUDev)
      // TODO: is selecting device 0 correct?
      cuErrCheck(cuDeviceGet(&CUDev, 0));

    cuErrCheck(cuCtxGetCurrent(&CUCtx));
    if (!CUCtx)
      cuErrCheck(cuCtxCreate(&CUCtx, 0, CUDev));

    int CCMajor;
    cuErrCheck(cuDeviceGetAttribute(
        &CCMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUDev));
    int CCMinor;
    cuErrCheck(cuDeviceGetAttribute(
        &CCMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUDev));
    CudaArch = "sm_" + std::to_string(CCMajor * 10 + CCMinor);

    DBG(dbgs() << "CUDA Arch " << CudaArch << "\n");
#endif

    Config.ENV_JIT_USE_STORED_CACHE =
        getEnvOrDefaultBool("ENV_JIT_USE_STORED_CACHE", true);
    Config.ENV_JIT_LAUNCH_BOUNDS =
        getEnvOrDefaultBool("ENV_JIT_LAUNCH_BOUNDS", true);
    Config.ENV_JIT_RUNTIME_CONSTPROP =
        getEnvOrDefaultBool("ENV_JIT_RUNTIME_CONSTPROP", true);

#if ENABLE_DEBUG
    dbgs() << "ENV_JIT_USE_STORED_CACHE " << Config.ENV_JIT_USE_STORED_CACHE
           << "\n";
    dbgs() << "ENV_JIT_LAUNCH_BOUNDS " << Config.ENV_JIT_LAUNCH_BOUNDS << "\n";
    dbgs() << "ENV_JIT_RUNTIME_CONSTPROP " << Config.ENV_JIT_RUNTIME_CONSTPROP
           << "\n";
#endif
  }

  bool getEnvOrDefaultBool(const char *VarName, bool Default) {
    const char *EnvValue = std::getenv(VarName);
    return EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : Default;
  }

  std::unordered_map<std::string, const void *> VarNameToDevPtr;
  struct {
    bool ENV_JIT_USE_STORED_CACHE;
    bool ENV_JIT_LAUNCH_BOUNDS;
    bool ENV_JIT_RUNTIME_CONSTPROP;
  } Config;
#if ENABLE_HIP
  JitCache<hipFunction_t> CodeCache;
  JitStoredCache<hipFunction_t> StoredCache;
#elif ENABLE_CUDA
  JitCache<CUfunction> CodeCache;
  JitStoredCache<CUfunction> StoredCache;
  std::string CudaArch;
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif
};

// NOTE: A great mystery is: why does this work ONLY if HostAddr is a CONST
// void* for HIP
__attribute((used)) void __jit_register_var(const void *HostAddr,
                                            const char *VarName) {
  JitEngineDevice &Jit = JitEngineDevice::instance();
#if ENABLE_HIP
  void *DevPtr = nullptr;
  // NOTE: For HIP it works to get the symobl address during the call
  // inside a constructor context.
  hipErrCheck(hipGetSymbolAddress(&DevPtr, HIP_SYMBOL(HostAddr)));
  Jit.insertRegisterVar(VarName, DevPtr);
#elif ENABLE_CUDA
  // NOTE: For CUDA, it fails to get the symbol address inside the constructor
  // context, so we save the host address and defer resolving the symbol
  // address when patching the bitcode.
  Jit.insertRegisterVar(VarName, HostAddr);
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined"
#endif
}

__attribute__((used))
#if ENABLE_HIP
// NOTE: Using the ABI With scalars for GridDim, BlockDim instead of dim3 to
// avoid issues with aggregate coercion of parameters.
hipError_t
__jit_launch_kernel(const char *ModuleUniqueId, char *KernelName,
                    FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
                    RuntimeConstant *RC, int NumRuntimeConstants,
                    uint64_t GridDimXY, uint32_t GridDimZ, uint64_t BlockDim_XY,
                    uint32_t BlockDimZ, void **KernelArgs, uint64_t ShmemSize,
                    void *Stream) {

#elif ENABLE_CUDA
cudaError_t
__jit_launch_kernel(const char *ModuleUniqueId, char *KernelName,
                    FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
                    RuntimeConstant *RC, int NumRuntimeConstants, dim3 GridDim,
                    dim3 BlockDim, void **KernelArgs, uint64_t ShmemSize,
                    void *Stream) {
#else
#error "Expected ENABLE_HIP or ENABLE_CUDA to be defined'
#endif
#if ENABLE_HIP
  dim3 GridDim = {*(uint32_t *)&GridDimXY, *(((uint32_t *)&GridDimXY) + 1),
                  GridDimZ};
  dim3 BlockDim = {*(uint32_t *)&BlockDim_XY, *(((uint32_t *)&BlockDim_XY) + 1),
                   BlockDimZ};
#endif
  TIMESCOPE("__jit_launch_kernel");
  JitEngineDevice &Jit = JitEngineDevice::instance();
  DBG(dbgs() << "JIT Launch Kernel\n");
  DBG(dbgs() << "=== Kernel Info\n");
  DBG(dbgs() << "KernelName " << KernelName << "\n");
  DBG(dbgs() << "FatbinSize " << FatbinSize << "\n");
  DBG(dbgs() << "Grid " << GridDim.x << ", " << GridDim.y << ", " << GridDim.z
             << "\n");
  DBG(dbgs() << "Block " << BlockDim.x << ", " << BlockDim.y << ", "
             << BlockDim.z << "\n");
  DBG(dbgs() << "KernelArgs " << KernelArgs << "\n");
  DBG(dbgs() << "ShmemSize " << ShmemSize << "\n");
  DBG(dbgs() << "Stream " << Stream << "\n");
  DBG(dbgs() << "=== End Kernel Info\n");
  return Jit.compileAndRun(ModuleUniqueId, KernelName, FatbinWrapper,
                           FatbinSize, RC, NumRuntimeConstants, GridDim,
                           BlockDim, KernelArgs, ShmemSize, Stream);
}
}
