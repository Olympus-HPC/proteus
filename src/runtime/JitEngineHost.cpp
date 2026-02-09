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

#include "proteus/impl/JitEngineHost.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/LambdaRegistry.h"
#include "proteus/impl/TransformArgumentSpecialization.h"
#include "proteus/impl/TransformLambdaSpecialization.h"
#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
#include "proteus/impl/CompilerInterfaceDevice.h"
#endif

#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Object/SymbolSize.h>
#include <llvm/TargetParser/Host.h>

#include <memory>

using namespace proteus;
using namespace llvm;
using namespace llvm::orc;

JitEngineHost &JitEngineHost::instance() {
  static JitEngineHost Jit;
  return Jit;
}

void JitEngineHost::addStaticLibrarySymbols() {
  // Create a SymbolMap for static symbols.
  SymbolMap SymbolMap;

#if PROTEUS_ENABLE_CUDA
  // NOTE: For CUDA codes we link the CUDA runtime statically to access device
  // global variables. So, if the host JIT module uses CUDA functions, we
  // need to resolve them statically in the JIT module's linker.

  // Insert cudaLaunchKernel if available, using the Proteus CUDA runtime
  // builtin initialization to avoid dependencies on the CUDA runtime library.
  if (__proteus_cudaLaunchKernel_ptr) {
    SymbolMap[LLJITPtr->mangleAndIntern("cudaLaunchKernel")] =
        orc::ExecutorSymbolDef(orc::ExecutorAddr{reinterpret_cast<uintptr_t>(
                                   __proteus_cudaLaunchKernel_ptr)},
                               JITSymbolFlags::Exported);
  }

#endif

#if PROTEUS_ENABLE_CUDA || PROTEUS_ENABLE_HIP
  // Add __jit_launch_kernel as a static symbol.
  SymbolMap[LLJITPtr->mangleAndIntern("__jit_launch_kernel")] =
      orc::ExecutorSymbolDef(
          orc::ExecutorAddr{reinterpret_cast<uintptr_t>(__jit_launch_kernel)},
          JITSymbolFlags::Exported);

#endif
  // Register the symbol in the main JIT dynamic library.
  cantFail(LLJITPtr->getMainJITDylib().define(absoluteSymbols(SymbolMap)));
}

void JitEngineHost::dumpSymbolInfo(
    const object::ObjectFile &LoadedObj,
    const RuntimeDyld::LoadedObjectInfo &ObjInfo) {
  // Dump information about symbols.
  auto Pid = sys::Process::getProcessId();
  std::error_code EC;
  raw_fd_ostream OFD("/tmp/perf-" + std::to_string(Pid) + ".map", EC,
                     sys::fs::OF_Append);
  if (EC)
    reportFatalError("Cannot open perf map file");
  for (auto SymSizePair : object::computeSymbolSizes(LoadedObj)) {
    auto Sym = SymSizePair.first;
    auto Size = SymSizePair.second;
    auto SymName = Sym.getName();
    // Skip any unnamed symbols.
    if (!SymName || SymName->empty())
      continue;
    // The relative address of the symbol inside its section.
    auto SymAddr = Sym.getAddress();
    if (!SymAddr)
      continue;
    // The address the functions was loaded at.
    auto LoadedSymAddress = *SymAddr;
    auto SymbolSection = Sym.getSection();
    if (SymbolSection) {
      // Compute the load address of the symbol by adding the section load
      // address.
      LoadedSymAddress += ObjInfo.getSectionLoadAddress(*SymbolSection.get());
    }
    PROTEUS_DBG(Logger::logs("proteus")
                << format("Address range: [%12p, %12p]", LoadedSymAddress,
                          LoadedSymAddress + Size)
                << "\tSymbol: " << *SymName << "\n");

    if (Size > 0)
      OFD << format("%lx %x)", LoadedSymAddress, Size) << " " << *SymName
          << "\n";
  }

  OFD.close();
}

void JitEngineHost::notifyLoaded(MaterializationResponsibility & /*R*/,
                                 const object::ObjectFile &Obj,
                                 const RuntimeDyld::LoadedObjectInfo &LOI) {
  dumpSymbolInfo(Obj, LOI);
}

JitEngineHost::~JitEngineHost() {
  CodeCache.printStats();
  CodeCache.printKernelTrace();
  if (CacheChain)
    CacheChain->printStats();
}

void JitEngineHost::specializeIR(Module &M, StringRef FnName, StringRef Suffix,
                                 ArrayRef<RuntimeConstant> RCArray) {
  TIMESCOPE("specializeIR");
  Function *F = M.getFunction(FnName);
  assert(F && "Expected non-null function!");

  // Replace argument uses with runtime constants.
  // TODO: change NumRuntimeConstants to size_t at interface.
  MDNode *Node = F->getMetadata("jit_arg_nos");
  assert(Node && "Expected metata for jit argument positions");
  PROTEUS_DBG(Logger::logs("proteus") << "Metadata jit for F " << F->getName()
                                      << " = " << *Node << "\n");

  // Replace argument uses with runtime constants.
  SmallVector<int32_t> ArgPos;
  for (unsigned int I = 0; I < Node->getNumOperands(); ++I) {
    ConstantAsMetadata *CAM = cast<ConstantAsMetadata>(Node->getOperand(I));
    ConstantInt *ConstInt = cast<ConstantInt>(CAM->getValue());
    int ArgNo = ConstInt->getZExtValue();
    ArgPos.push_back(ArgNo);
  }

  TransformArgumentSpecialization::transform(M, *F, RCArray);

  if (!LambdaRegistry::instance().empty()) {
    if (auto OptionalMapIt =
            LambdaRegistry::instance().matchJitVariableMap(F->getName())) {
      auto &RCVec = OptionalMapIt.value()->getSecond();
      TransformLambdaSpecialization::transform(M, *F, RCVec);
    }
  }

  F->setName(FnName + Suffix);

  if (Config::get().ProteusDebugOutput) {
    if (verifyModule(M, &errs()))
      reportFatalError("Broken module found, JIT compilation aborted!");
    else
      Logger::logs("proteus") << "Module verified!\n";
  }
}

void getLambdaJitValues(StringRef FnName,
                        SmallVector<RuntimeConstant> &LambdaJitValuesVec) {
  LambdaRegistry &LR = LambdaRegistry::instance();
  if (LR.empty())
    return;

  PROTEUS_DBG(Logger::logs("proteus") << "=== Host LAMBDA MATCHING\n"
                                      << "Caller trigger " << FnName << " -> "
                                      << demangle(FnName.str()) << "\n");

  SmallVector<StringRef> LambdaCalleeInfo;
  PROTEUS_DBG(Logger::logs("proteus")
              << " Trying F " << demangle(FnName.str()) << "\n ");
  auto OptionalMapIt = LR.matchJitVariableMap(FnName);
  if (!OptionalMapIt)
    return;

  LambdaJitValuesVec = OptionalMapIt.value()->getSecond();
}
namespace {
void flushLambdaRuntimeConstants(StringRef FnName) {
  // Whether we (a) specializeIR of a lambda (b) load from cache or (c) load
  // from storage, we need to flush out the runtime constants from the lambda
  // registry
  auto &LR = LambdaRegistry::instance();
  if (auto OptionalMapIt = LR.matchJitVariableMap(FnName); OptionalMapIt) {
    LR.flushRuntimeConstants(OptionalMapIt.value()->first);
  }
}
} // namespace

void *
JitEngineHost::compileAndLink(StringRef FnName, char *IR, int IRSize,
                              void **Args,
                              ArrayRef<RuntimeConstantInfo *> RCInfoArray) {
  TIMESCOPE("compileAndLink");

  StringRef StrIR(IR, IRSize);
  auto Ctx = std::make_unique<LLVMContext>();

  Timer T;
  SMDiagnostic Diag;
  auto M = parseIR(MemoryBufferRef(StrIR, "JitModule"), Diag, *Ctx);
  if (!M)
    reportFatalError("Error parsing IR: " + Diag.getMessage());

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus") << "Parse IR " << FnName << " "
                                               << T.elapsed() << " ms\n");

  SmallVector<RuntimeConstant> RCVec =
      getRuntimeConstantValues(Args, RCInfoArray);
  SmallVector<RuntimeConstant> LambdaJitValuesVec;
  getLambdaJitValues(FnName, LambdaJitValuesVec);

  HashT HashValue = hash(StrIR, FnName, RCVec, LambdaJitValuesVec);
  if (Config::get().ProteusDebugOutput) {
    Logger::logs("proteus")
        << "Hashing: " << " FnName " << FnName << " RCVec [ ";
    for (const auto &RC : RCVec)
      Logger::logs("proteus") << RC.Value.Int64Val << ",";
    Logger::logs("proteus") << " ] LambdaVec [ ";
    for (auto &RC : LambdaJitValuesVec)
      Logger::logs("proteus") << RC.Value.Int64Val << ",";
    Logger::logs("proteus") << " ] -> Hash " << HashValue.getValue() << "\n";
  }

  // Lookup the function pointer in the code cache.
  void *JitFnPtr = CodeCache.lookup(HashValue);
  if (JitFnPtr) {
    flushLambdaRuntimeConstants(FnName);
    return JitFnPtr;
  }

  std::string Suffix = HashValue.toMangledSuffix();
  std::string MangledFnName = FnName.str() + Suffix;
  std::unique_ptr<CompiledLibrary> Library;

  // Lookup the code library in the object cache chain to load without
  // compiling, if found.
  if (CacheChain && (Library = CacheChain->lookup(HashValue))) {
    loadCompiledLibrary(*Library);
  } else {
    PROTEUS_DBG(Logger::logfile(HashValue.toString() + ".input.ll", *M));
    // Specialize the module using runtime values.
    specializeIR(*M, FnName, Suffix, RCVec);
    PROTEUS_DBG(Logger::logfile(HashValue.toString() + ".specialized.ll", *M));
    // Compile the object.
    auto ObjectModule = compileOnly(*M);

    if (CacheChain)
      CacheChain->store(
          HashValue, CacheEntry::staticObject(ObjectModule->getMemBufferRef()));

    // Create the compiled library and load it.
    Library = std::make_unique<CompiledLibrary>(std::move(ObjectModule));
    loadCompiledLibrary(*Library);
  }

  // Retrieve the function address and store it in the code cache.
  JitFnPtr = getFunctionAddress(MangledFnName, *Library);
  CodeCache.insert(HashValue, JitFnPtr, FnName);

  PROTEUS_DBG(Logger::logs("proteus")
              << "===\n"
              << *LLJITPtr->getExecutionSession().getSymbolStringPool()
              << "===\n");

  PROTEUS_DBG(Logger::logs("proteus")
              << "=== JIT compile: " << FnName << " Mangled " << MangledFnName
              << " RC HashValue " << HashValue.toString() << " Addr "
              << JitFnPtr << "\n");
  flushLambdaRuntimeConstants(FnName);

  return JitFnPtr;
}

std::unique_ptr<MemoryBuffer> JitEngineHost::compileOnly(Module &M,
                                                         bool DisableIROpt) {
  // Create the target machine using JITTargetMachineBuilder to match ORC JIT
  // loading.
  auto ExpectedTM =
      JITTargetMachineBuilder::detectHost()->createTargetMachine();
  if (auto E = ExpectedTM.takeError())
    reportFatalError("Expected target machine: " + toString(std::move(E)));
  std::unique_ptr<TargetMachine> TM = std::move(*ExpectedTM);

  // Set up the output stream.
  SmallVector<char, 0> ObjBuffer;
  raw_svector_ostream ObjStream(ObjBuffer);

  // Set up the pass manager.
  legacy::PassManager PM;
  // Add optimization passes.
  if (!DisableIROpt) {
    const auto &CGConfig = Config::get().getCGConfig();
    if (CGConfig.optPipeline()) {
      optimizeIR(M, sys::getHostCPUName(), CGConfig.optPipeline().value(),
                 CGConfig.codeGenOptLevel());
    } else
      optimizeIR(M, sys::getHostCPUName(), CGConfig.optLevel(),
                 CGConfig.codeGenOptLevel());
  } else {
    if (Config::get().traceSpecializations())
      Logger::trace("[SkipOpt] Skipping JitEngine IR optimization\n");
  }

  // Add the target passes to emit object code.
  if (TM->addPassesToEmitFile(PM, ObjStream, nullptr,
                              CodeGenFileType::ObjectFile)) {
    reportFatalError("Target machine cannot emit object file");
  }

  // Run the passes.
  PM.run(M);

  return MemoryBuffer::getMemBufferCopy(
      StringRef(ObjBuffer.data(), ObjBuffer.size()));
}

void JitEngineHost::loadCompiledLibrary(CompiledLibrary &Library) {
  // Create an isolated JITDyLib context and load the compiled library for
  // linking and symbol retrieval.
  auto &ES = LLJITPtr->getExecutionSession();
  auto ExpectedJitDyLib = ES.createJITDylib(
      "JitDyLib_" + std::to_string(reinterpret_cast<uintptr_t>(&Library)));
  if (auto E = ExpectedJitDyLib.takeError()) {
    reportFatalError("Error creating library jit dylib: " +
                     toString(std::move(E)));
  }

  JITDylib &CreatedDylib = *ExpectedJitDyLib;
  Library.JitDyLib = &CreatedDylib;

  if (Library.isSharedObject()) {
    // Load the dynamic library through a generator using the dynamic library
    // file.
    auto Gen = ExitOnErr(llvm::orc::DynamicLibrarySearchGenerator::Load(
        Library.DynLibPath.c_str(),
        LLJITPtr->getDataLayout().getGlobalPrefix()));
    Library.JitDyLib->addGenerator(std::move(Gen));
  } else {
    // Resolve symbols from main/process before materialization through the main
    // JIT dynamic library. It is simpler and avoids duplication than adding
    // symbols to the code library JIT dylib every time.
    Library.JitDyLib->addToLinkOrder(LLJITPtr->getMainJITDylib(),
                                     JITDylibLookupFlags::MatchAllSymbols);

    // Add the object from the compiled library.
    if (auto E = LLJITPtr->addObjectFile(*Library.JitDyLib,
                                         std::move(Library.ObjectModule)))
      reportFatalError("Error loading object file: " + toString(std::move(E)));
  }
}

void *JitEngineHost::getFunctionAddress(StringRef FnName,
                                        CompiledLibrary &Library) {
  // Lookup the function address corresponding the dynamic library context of
  // the compiled library.
  assert(Library.JitDyLib && "Expected non-null JIT dylib");
  auto EntryAddr = ExitOnErr(LLJITPtr->lookup(*Library.JitDyLib, FnName));

  void *JitFnPtr = (void *)EntryAddr.getValue();
  assert(JitFnPtr && "Expected non-null JIT function pointer");

  return JitFnPtr;
}

JitEngineHost::JitEngineHost() {
  ExitOnErr.setBanner("JIT: ");
  // Create the LLJIT instance.
  // TODO: Fix support for debugging jitted code. This appears to be
  // the correct interface (see orcv2 examples) but it does not work.
  // By dumpSymbolInfo() the debug sections are not populated. Why?
  LLJITPtr =
      ExitOnErr(LLJITBuilder()
                    .setObjectLinkingLayerCreator([&](ExecutionSession &ES,
                                                      const Triple & /*TT*/) {
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

                      if (Config::get().ProteusDebugOutput) {
                        ObjLinkingLayer->setNotifyLoaded(notifyLoaded);
                      }
                      return ObjLinkingLayer;
                    })
                    .create());
  // Use the main JIT dynamic library to add a generator for host process
  // symbols.
  orc::MangleAndInterner Mangle(LLJITPtr->getExecutionSession(),
                                LLJITPtr->getDataLayout());
  LLJITPtr->getMainJITDylib().addGenerator(
      ExitOnErr(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          LLJITPtr->getDataLayout().getGlobalPrefix(),
          [MainName = Mangle("main")](const orc::SymbolStringPtr &Name) {
            return Name != MainName;
          })));

  // Add static library functions to the main JIT dynamic library.
  addStaticLibrarySymbols();

  if (Config::get().ProteusUseStoredCache)
    CacheChain.emplace("JitEngineHost");
}
