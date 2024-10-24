//===-- JitEngineHost.cpp -- JIT Engine for CPU using ORC ---===//
//
// Part of Proteus Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the JitEngine interface for dynamic compilation and optimization
// of CPU code.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/SymbolSize.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "CompilerInterfaceTypes.h"
#include "JitEngineHost.hpp"
#include "TransformArgumentSpecialization.hpp"

using namespace proteus;
using namespace llvm;
using namespace llvm::orc;

// TODO: check if this global is needed.
static codegen::RegisterCodeGenFlags CFG;

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

JitEngineHost &JitEngineHost::instance() {
  static JitEngineHost Jit(0, (char *[]){nullptr});
  return Jit;
}

void JitEngineHost::addStaticLibrarySymbols() {
  // Create a SymbolMap for static symbols.
  SymbolMap SymbolMap;

#if ENABLE_CUDA
  // NOTE: For CUDA codes we link the CUDA runtime statically to access device
  // global variables. So, if the host JIT module uses CUDA functions, we
  // need to resolve them statically in the JIT module's linker.

  // TODO: Instead of manually adding symbols, can we handle
  // materialization errors through a notify callback and automatically add
  // those symbols?

  // Insert symbols in the SymbolMap, disambiguate if needed.
  using CudaLaunchKernelFn =
      cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  auto CudaLaunchKernel = static_cast<CudaLaunchKernelFn>(&cudaLaunchKernel);
  SymbolMap[LLJITPtr->mangleAndIntern("cudaLaunchKernel")] =
      llvm::orc::ExecutorSymbolDef(
          llvm::orc::ExecutorAddr{
              reinterpret_cast<uintptr_t>(CudaLaunchKernel)},
          llvm::JITSymbolFlags::Exported);

#endif
  // Register the symbol manually.
  llvm::cantFail(
      LLJITPtr->getMainJITDylib().define(absoluteSymbols(SymbolMap)));
}

void JitEngineHost::dumpSymbolInfo(
    const llvm::object::ObjectFile &loadedObj,
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
      ofd << llvm::format("%lx %x)", loadedSymAddress, size) << " " << *symName
          << "\n";
  }

  ofd.close();
}

void JitEngineHost::notifyLoaded(MaterializationResponsibility &R,
                                 const object::ObjectFile &Obj,
                                 const RuntimeDyld::LoadedObjectInfo &LOI) {
  dumpSymbolInfo(Obj, LOI);
}

JitEngineHost::~JitEngineHost() { CodeCache.printStats("Host engine"); }

Expected<llvm::orc::ThreadSafeModule>
JitEngineHost::specializeBitcode(StringRef FnName, StringRef Suffix,
                                 StringRef IR, RuntimeConstant *RC,
                                 int NumRuntimeConstants) {
  TIMESCOPE("specializeBitcode");
  auto Ctx = std::make_unique<LLVMContext>();
  SMDiagnostic Err;
  if (auto M = parseIR(MemoryBufferRef(IR, ("Mod-" + FnName + Suffix).str()),
                       Err, *Ctx)) {
    // dbgs() << "=== Parsed Module\n" << *M << "=== End of Parsed Module\n ";
    Function *F = M->getFunction(FnName);
    assert(F && "Expected non-null function!");
    // Replace argument uses with runtime constants.
    // TODO: change NumRuntimeConstants to size_t at interface.
    TransformArgumentSpecialization::transform(
        *M, *F,
        ArrayRef<RuntimeConstant>{RC,
                                  static_cast<size_t>(NumRuntimeConstants)});

    // dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n";

    F->setName(FnName + Suffix);

#if ENABLE_DEBUG
    dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
    if (verifyModule(*M, &errs()))
      FATAL_ERROR("Broken module found, JIT compilation aborted!");
    else
      dbgs() << "Module verified!\n";
    getchar();
#endif
    return ThreadSafeModule(std::move(M), std::move(Ctx));
  }

  return createSMDiagnosticError(Err);
}

void *JitEngineHost::compileAndLink(StringRef FnName, char *IR, int IRSize,
                                    RuntimeConstant *RC,
                                    int NumRuntimeConstants) {
  TIMESCOPE("compileAndLink");

  // TODO: implement ModuleUniqueId for host code.
  uint64_t HashValue = CodeCache.hash("", FnName, RC, NumRuntimeConstants);
  void *JitFnPtr = CodeCache.lookup(HashValue);
  if (JitFnPtr)
    return JitFnPtr;

  std::string Suffix = mangleSuffix(HashValue);
  std::string MangledFnName = FnName.str() + Suffix;

  StringRef StrIR(IR, IRSize);
  // (3) Add modules.
  ExitOnErr(LLJITPtr->addIRModule(ExitOnErr(
      specializeBitcode(FnName, Suffix, StrIR, RC, NumRuntimeConstants))));

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
  CodeCache.insert(HashValue, JitFnPtr, FnName, RC, NumRuntimeConstants);

  dbgs() << "=== JIT compile: " << FnName << " Mangled " << MangledFnName
         << " RC HashValue " << HashValue << " Addr " << JitFnPtr << "\n";
  return JitFnPtr;
}

JitEngineHost::JitEngineHost(int argc, char *argv[]) {
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

  // Add static library functions for JIT linking.
  addStaticLibrarySymbols();

  // (3) Install transform to optimize modules when they're materialized.
  LLJITPtr->getIRTransformLayer().setTransform(OptimizationTransform());
}
