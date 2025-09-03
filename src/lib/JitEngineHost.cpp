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

#include "proteus/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/CoreLLVM.hpp"
#include "proteus/JitEngine.hpp"
#include "proteus/JitEngineHost.hpp"
#include "proteus/LambdaRegistry.hpp"
#include "proteus/TransformArgumentSpecialization.hpp"
#include "proteus/TransformLambdaSpecialization.hpp"
#include "proteus/Utils.h"

using namespace proteus;
using namespace llvm;
using namespace llvm::orc;

#if PROTEUS_ENABLE_HIP || PROTEUS_ENABLE_CUDA
#include "proteus/CompilerInterfaceDevice.h"
#endif

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

  // Insert symbols in the SymbolMap, disambiguate if needed.
  using CudaLaunchKernelFn =
      cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  auto CudaLaunchKernel = static_cast<CudaLaunchKernelFn>(&cudaLaunchKernel);
  SymbolMap[LLJITPtr->mangleAndIntern("cudaLaunchKernel")] =
      orc::ExecutorSymbolDef(
          orc::ExecutorAddr{reinterpret_cast<uintptr_t>(CudaLaunchKernel)},
          JITSymbolFlags::Exported);

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
    PROTEUS_FATAL_ERROR("Cannot open perf map file");
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
  StorageCache.printStats();
}

void JitEngineHost::specializeIR(Module &M, StringRef FnName, StringRef Suffix,
                                 ArrayRef<RuntimeConstant> RCArray) {
  TIMESCOPE("specializeIR");
  Function *F = M.getFunction(FnName);
  assert(F && "Expected non-null function!");

#if PROTEUS_ENABLE_DEBUG
  PROTEUS_DBG(Logger::logfile(FnName.str() + ".input.ll", M));
#endif
  // Find GlobalValue declarations that are externally defined. Resolve them
  // statically as absolute symbols in the ORC linker. Required for resolving
  // __jit_launch_kernel for a host JIT function when libproteus is compiled
  // as a static library. For other non-resolved symbols, return a fatal error
  // to investigate.
  for (auto &GV : M.global_values()) {
    if (!GV.isDeclaration())
      continue;

    if (Function *F = dyn_cast<Function>(&GV))
      if (F->isIntrinsic())
        continue;

    auto ExecutorAddr = LLJITPtr->lookup(GV.getName());
    if (auto Error = ExecutorAddr.takeError())
      PROTEUS_FATAL_ERROR("Unknown global value " + GV.getName() +
                          " to resolve: " + toString(std::move(Error)));
  }
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

#if PROTEUS_ENABLE_DEBUG
  Logger::logfile(FnName.str() + ".specialized.ll", M);
  if (verifyModule(M, &errs()))
    PROTEUS_FATAL_ERROR("Broken module found, JIT compilation aborted!");
  else
    Logger::logs("proteus") << "Module verified!\n";
#endif
}

void getLambdaJitValues(StringRef FnName,
                        SmallVector<RuntimeConstant> &LambdaJitValuesVec) {
  LambdaRegistry LR = LambdaRegistry::instance();
  if (LR.empty())
    return;

  PROTEUS_DBG(Logger::logs("proteus") << "=== Host LAMBDA MATCHING\n"
                                      << "Caller trigger " << FnName << " -> "
                                      << demangle(FnName.str()) << "\n");

  SmallVector<StringRef> LambdaCalleeInfo;
  PROTEUS_DBG(Logger::logs("proteus")
              << " Trying F " << demangle(FnName.str()) << "\n ");
  auto OptionalMapIt = LambdaRegistry::instance().matchJitVariableMap(FnName);
  if (!OptionalMapIt)
    return;

  LambdaJitValuesVec = OptionalMapIt.value()->getSecond();
}

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
    PROTEUS_FATAL_ERROR("Error parsing IR: " + Diag.getMessage());

  PROTEUS_TIMER_OUTPUT(Logger::outs("proteus") << "Parse IR " << FnName << " "
                                               << T.elapsed() << " ms\n");

  SmallVector<RuntimeConstant> RCVec =
      getRuntimeConstantValues(Args, RCInfoArray);
  SmallVector<RuntimeConstant> LambdaJitValuesVec;
  getLambdaJitValues(FnName, LambdaJitValuesVec);

  HashT HashValue = hash(StrIR, FnName, RCVec, LambdaJitValuesVec);
#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "Hashing: " << " FnName " << FnName << " RCVec [ ";
  for (const auto &RC : RCVec)
    Logger::logs("proteus") << RC.Value.Int64Val << ",";
  Logger::logs("proteus") << " ] LambdaVec [ ";
  for (auto &RC : LambdaJitValuesVec)
    Logger::logs("proteus") << RC.Value.Int64Val << ",";
  Logger::logs("proteus") << " ] -> Hash " << HashValue.getValue() << "\n";
#endif

  // Lookup the function pointer in the code cache.
  void *JitFnPtr = CodeCache.lookup(HashValue);
  if (JitFnPtr)
    return JitFnPtr;

  std::string Suffix = mangleSuffix(HashValue);
  std::string MangledFnName = FnName.str() + Suffix;
  std::unique_ptr<CompiledLibrary> Library;
  // Lookup the code library in the storage cache to load without compiling, if
  // found.
  if ((Library = StorageCache.lookup(HashValue))) {
    loadCompiledLibrary(*Library);
  } else {
    // Specialize the module using runtime values.
    specializeIR(*M, FnName, Suffix, RCVec);
    // Compile the object.
    auto ObjectModule = compileOnly(*M);

    StorageCache.store(HashValue, ObjectModule->getMemBufferRef());

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
  return JitFnPtr;
}

std::unique_ptr<MemoryBuffer> JitEngineHost::compileOnly(Module &M) {
  // Create the target machine using JITTargetMachineBuilder to match ORC JIT
  // loading.
  auto ExpectedTM =
      JITTargetMachineBuilder::detectHost()->createTargetMachine();
  if (auto E = ExpectedTM.takeError())
    PROTEUS_FATAL_ERROR("Expected target machine: " + toString(std::move(E)));
  std::unique_ptr<TargetMachine> TM = std::move(*ExpectedTM);

  // Set up the output stream.
  SmallVector<char, 0> ObjBuffer;
  raw_svector_ostream ObjStream(ObjBuffer);

  // Set up the pass manager.
  legacy::PassManager PM;
  // Add optimization passes.
  if (Config::get().ProteusOptPipeline) {
    optimizeIR(M, sys::getHostCPUName(),
               Config::get().ProteusOptPipeline.value(), 3);
  } else
    optimizeIR(M, sys::getHostCPUName(), '3', 3);

  // Add the target passes to emit object code.
  if (TM->addPassesToEmitFile(PM, ObjStream, nullptr,
                              CodeGenFileType::ObjectFile)) {
    PROTEUS_FATAL_ERROR("Target machine cannot emit object file");
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
    PROTEUS_FATAL_ERROR("Error creating library jit dylib: " +
                        toString(std::move(E)));
  }

  JITDylib &CreatedDylib = *ExpectedJitDyLib;
  Library.JitDyLib = &CreatedDylib;

  if (Library.isDynLib()) {
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
      PROTEUS_FATAL_ERROR("Error loading object file: " +
                          toString(std::move(E)));
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

#if PROTEUS_ENABLE_DEBUG
                      ObjLinkingLayer->setNotifyLoaded(notifyLoaded);
#endif
                      return ObjLinkingLayer;
                    })
                    .create());
  // Use the main JIT dynamic library to add a generate for host proces symbols.
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
}
