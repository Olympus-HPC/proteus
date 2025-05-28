//===-- JitEngineDevice.cpp -- Base JIT Engine Device header impl. --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINEDEVICE_HPP
#define PROTEUS_JITENGINEDEVICE_HPP

#include <cstdint>
#include <functional>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/CallGraph.h>
#include <memory>
#include <optional>
#include <string>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "proteus/Cloning.h"
#include "proteus/CompilerAsync.hpp"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/CompilerSync.hpp"
#include "proteus/CoreDevice.hpp"
#include "proteus/CoreLLVM.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"
#include "proteus/JitCache.hpp"
#include "proteus/JitEngine.hpp"
#include "proteus/JitStorageCache.hpp"
#include "proteus/TimeTracing.hpp"
#include "proteus/Utils.h"

namespace proteus {

using namespace llvm;

struct FatbinWrapperT {
  int32_t Magic;
  int32_t Version;
  const char *Binary;
  void **PrelinkedFatbins;
};

class BinaryInfo {
private:
  FatbinWrapperT *FatbinWrapper;
  std::unique_ptr<LLVMContext> Ctx;
  SmallVector<std::string> LinkedModuleIds;
  Module *LinkedModule;
  std::optional<SmallVector<std::unique_ptr<Module>>> ExtractedModules;
  std::optional<HashT> ExtractedModuleHash;
  std::optional<CallGraph> ModuleCallGraph;

public:
  BinaryInfo() = default;
  BinaryInfo(FatbinWrapperT *FatbinWrapper,
             SmallVector<std::string> &&LinkedModuleIds)
      : FatbinWrapper(FatbinWrapper), Ctx(std::make_unique<LLVMContext>()),
        LinkedModuleIds(LinkedModuleIds), LinkedModule(nullptr),
        ExtractedModules(std::nullopt), ModuleCallGraph(std::nullopt) {}

  FatbinWrapperT *getFatbinWrapper() const { return FatbinWrapper; }

  std::unique_ptr<LLVMContext> &getLLVMContext() { return Ctx; }

  bool hasLinkedModule() const { return (LinkedModule != nullptr); }
  Module &getLinkedModule() {
    if (!LinkedModule) {
      if (!hasExtractedModules())
        PROTEUS_FATAL_ERROR("Expected extracted modules");

      Timer T;
      // Avoid linking when there's a single module by moving it instead and
      // making sure it's materialized for call graph analysis.
      if (ExtractedModules->size() == 1) {
        LinkedModule = ExtractedModules->front().get();
        if (auto E = LinkedModule->materializeAll())
          PROTEUS_FATAL_ERROR("Error materializing " + toString(std::move(E)));
      } else {
        // By the LLVM API, linkModules takes ownership of module pointers in
        // ExtractedModules and returns a new unique ptr to the linked module.
        // We update ExtractedModules to contain and own only the generated
        // LinkedModule.
        auto GeneratedLinkedModule =
            proteus::linkModules(*Ctx, std::move(ExtractedModules.value()));
        SmallVector<std::unique_ptr<Module>> NewExtractedModules;
        NewExtractedModules.emplace_back(std::move(GeneratedLinkedModule));
        setExtractedModules(NewExtractedModules);

        LinkedModule = ExtractedModules->front().get();
      }

      PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                           << "getLinkedModule " << T.elapsed() << " ms\n");
    }

    return *LinkedModule;
  }

  bool hasExtractedModules() const { return ExtractedModules.has_value(); }
  const SmallVector<std::reference_wrapper<Module>>
  getExtractedModules() const {
    // This should be called only once when cloning the kernel module to cache.
    SmallVector<std::reference_wrapper<Module>> ModulesRef;
    for (auto &M : ExtractedModules.value())
      ModulesRef.emplace_back(*M);

    return ModulesRef;
  }
  void setExtractedModules(SmallVector<std::unique_ptr<Module>> &Modules) {
    ExtractedModules = std::move(Modules);
  }

  bool hasModuleHash() const { return ExtractedModuleHash.has_value(); }
  HashT getModuleHash() const { return ExtractedModuleHash.value(); }
  void setModuleHash(HashT HashValue) { ExtractedModuleHash = HashValue; }
  void updateModuleHash(HashT HashValue) {
    if (ExtractedModuleHash)
      ExtractedModuleHash = hashCombine(ExtractedModuleHash.value(), HashValue);
    else
      ExtractedModuleHash = HashValue;
  }

  CallGraph &getCallGraph() {
    if (!ModuleCallGraph.has_value()) {
      if (!LinkedModule)
        PROTEUS_FATAL_ERROR("Expected non-null linked module");
      ModuleCallGraph.emplace(CallGraph(*LinkedModule));
    }
    return ModuleCallGraph.value();
  }

  void addModuleId(const char *ModuleId) {
    LinkedModuleIds.push_back(ModuleId);
  }

  auto &getModuleIds() { return LinkedModuleIds; }
};

class JITKernelInfo {
  std::optional<void *> Kernel;
  std::unique_ptr<LLVMContext> Ctx;
  std::string Name;
  SmallVector<int32_t> RCTypes;
  SmallVector<int32_t> RCIndices;
  std::optional<std::unique_ptr<Module>> ExtractedModule;
  std::optional<std::unique_ptr<MemoryBuffer>> Bitcode;
  std::optional<std::reference_wrapper<BinaryInfo>> BinInfo;
  std::optional<HashT> StaticHash;
  std::optional<SmallVector<std::pair<std::string, StringRef>>>
      LambdaCalleeInfo;

public:
  JITKernelInfo(void *Kernel, BinaryInfo &BinInfo, char const *Name,
                int32_t *RCIndices, int32_t *RCTypes, int32_t NumRCs)
      : Kernel(Kernel), Ctx(std::make_unique<LLVMContext>()), Name(Name),
        RCTypes{ArrayRef{RCTypes, static_cast<size_t>(NumRCs)}},
        RCIndices{ArrayRef{RCIndices, static_cast<size_t>(NumRCs)}},
        ExtractedModule(std::nullopt), Bitcode{std::nullopt}, BinInfo(BinInfo),
        LambdaCalleeInfo(std::nullopt) {}

  JITKernelInfo() = default;
  void *getKernel() const {
    assert(Kernel.has_value() && "Expected Kernel is inited");
    return Kernel.value();
  }
  std::unique_ptr<LLVMContext> &getLLVMContext() { return Ctx; }
  const std::string &getName() const { return Name; }
  const auto &getRCIndices() const { return RCIndices; }
  const auto &getRCTypes() const { return RCTypes; }
  bool hasModule() const { return ExtractedModule.has_value(); }
  Module &getModule() const { return *ExtractedModule->get(); }
  BinaryInfo &getBinaryInfo() const { return BinInfo.value(); }
  void setModule(std::unique_ptr<llvm::Module> Mod) {
    ExtractedModule = std::move(Mod);
  }

  bool hasBitcode() { return Bitcode.has_value(); }
  void setBitcode(std::unique_ptr<MemoryBuffer> ExtractedBitcode) {
    Bitcode = std::move(ExtractedBitcode);
  }
  MemoryBufferRef getBitcode() { return Bitcode.value()->getMemBufferRef(); }

  bool hasStaticHash() const { return StaticHash.has_value(); }
  const HashT getStaticHash() const { return StaticHash.value(); }
  void createStaticHash(HashT ModuleHash) {
    StaticHash = hash(Name);
    StaticHash = hashCombine(StaticHash.value(), ModuleHash);
  }

  bool hasLambdaCalleeInfo() { return LambdaCalleeInfo.has_value(); }
  const auto &getLambdaCalleeInfo() { return LambdaCalleeInfo.value(); }
  void setLambdaCalleeInfo(
      SmallVector<std::pair<std::string, StringRef>> &&LambdaInfo) {
    LambdaCalleeInfo = std::move(LambdaInfo);
  }
};

template <typename ImplT> struct DeviceTraits;

template <typename ImplT> class JitEngineDevice : public JitEngine {

public:
  using DeviceError_t = typename DeviceTraits<ImplT>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<ImplT>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<ImplT>::KernelFunction_t;

  DeviceError_t
  compileAndRun(JITKernelInfo &KernelInfo, dim3 GridDim, dim3 BlockDim,
                void **KernelArgs, uint64_t ShmemSize,
                typename DeviceTraits<ImplT>::DeviceStream_t Stream);

  std::pair<std::unique_ptr<Module>, std::unique_ptr<MemoryBuffer>>
  extractKernelModule(BinaryInfo &BinInfo, StringRef KernelName,
                      LLVMContext &Ctx) {
    std::unique_ptr<Module> KernelModule =
        static_cast<ImplT &>(*this).tryExtractKernelModule(BinInfo, KernelName,
                                                           Ctx);
    std::unique_ptr<MemoryBuffer> Bitcode = nullptr;

    // If there is no ready-made kernel module from AOT, extract per-TU or the
    // single linked module and clone the kernel module.
    if (!KernelModule) {
      Timer T;
      if (!BinInfo.hasExtractedModules())
        static_cast<ImplT &>(*this).extractModules(BinInfo);

      std::unique_ptr<Module> KernelModuleTmp = nullptr;
      switch (Config::get().ProteusKernelClone) {
      case proteus::KernelCloneOption::LinkClonePrune: {
        auto &LinkedModule = BinInfo.getLinkedModule();
        KernelModule = llvm::CloneModule(LinkedModule);
        break;
      }
      case proteus::KernelCloneOption::LinkCloneLight: {
        auto &LinkedModule = BinInfo.getLinkedModule();
        KernelModule =
            proteus::cloneKernelFromModules({LinkedModule}, KernelName);
        break;
      }
      case proteus::KernelCloneOption::CrossClone: {
        KernelModule = proteus::cloneKernelFromModules(
            BinInfo.getExtractedModules(), KernelName);
        break;
      }
      default:
        PROTEUS_FATAL_ERROR("Unsupported kernel cloning option");
      }

      PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                           << "Cloning "
                           << toString(Config::get().ProteusKernelClone) << " "
                           << T.elapsed() << " ms\n");
    }

    // Internalize and cleanup to simplify the module and prepare it for
    // optimization.
    internalize(*KernelModule, KernelName);
    proteus::runCleanupPassPipeline(*KernelModule);

    // If the module is not in the provided context due to cloning, roundtrip it
    // using bitcode. Re-use the roundtrip bitcode to return it.
    if (&KernelModule->getContext() != &Ctx) {
      SmallVector<char> CloneBuffer;
      raw_svector_ostream OS(CloneBuffer);
      WriteBitcodeToFile(*KernelModule, OS);
      StringRef CloneStr = StringRef(CloneBuffer.data(), CloneBuffer.size());
      auto ExpectedKernelModule =
          parseBitcodeFile(MemoryBufferRef{CloneStr, KernelName}, Ctx);
      if (auto E = ExpectedKernelModule.takeError())
        PROTEUS_FATAL_ERROR("Error parsing bitcode: " + toString(std::move(E)));

      KernelModule = std::move(*ExpectedKernelModule);
      Bitcode = MemoryBuffer::getMemBufferCopy(CloneStr);
    } else {
      // Parse the kernel module to create the bitcode since it has not been
      // created by roundtripping.
      SmallVector<char> BitcodeBuffer;
      raw_svector_ostream OS(BitcodeBuffer);
      WriteBitcodeToFile(*KernelModule, OS);
      auto BitcodeStr = StringRef{BitcodeBuffer.data(), BitcodeBuffer.size()};
      Bitcode = MemoryBuffer::getMemBufferCopy(BitcodeStr);
    }

    return std::make_pair(std::move(KernelModule), std::move(Bitcode));
  }

  void extractModuleAndBitcode(JITKernelInfo &KernelInfo) {
    TIMESCOPE(__FUNCTION__)

    if (KernelInfo.hasModule() && KernelInfo.hasBitcode())
      return;

    if (KernelInfo.hasModule())
      PROTEUS_FATAL_ERROR("Unexpected KernelInfo has module but not bitcode");

    if (KernelInfo.hasBitcode())
      PROTEUS_FATAL_ERROR("Unexpected KernelInfo has bitcode but not module");

    BinaryInfo &BinInfo = KernelInfo.getBinaryInfo();

    Timer T;
    auto [KernelModule, BitcodeBuffer] = extractKernelModule(
        BinInfo, KernelInfo.getName(), *KernelInfo.getLLVMContext());

    if (!KernelModule)
      PROTEUS_FATAL_ERROR("Expected non-null kernel module");
    if (!BitcodeBuffer)
      PROTEUS_FATAL_ERROR("Expected non-null kernel bitcode");

    KernelInfo.setModule(std::move(KernelModule));
    KernelInfo.setBitcode(std::move(BitcodeBuffer));
    PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                         << "Extract kernel module " << T.elapsed() << " ms\n");
  }

  Module &getModule(JITKernelInfo &KernelInfo) {
    if (!KernelInfo.hasModule())
      extractModuleAndBitcode(KernelInfo);

    if (!KernelInfo.hasModule())
      PROTEUS_FATAL_ERROR("Expected module in KernelInfo");

    return KernelInfo.getModule();
  }

  MemoryBufferRef getBitcode(JITKernelInfo &KernelInfo) {
    if (!KernelInfo.hasBitcode())
      extractModuleAndBitcode(KernelInfo);

    if (!KernelInfo.hasBitcode())
      PROTEUS_FATAL_ERROR("Expected bitcode in KernelInfo");

    return KernelInfo.getBitcode();
  }

  void getLambdaJitValues(JITKernelInfo &KernelInfo,
                          SmallVector<RuntimeConstant> &LambdaJitValuesVec) {
    LambdaRegistry LR = LambdaRegistry::instance();
    if (LR.empty()) {
      KernelInfo.setLambdaCalleeInfo({});
      return;
    }

    if (!KernelInfo.hasLambdaCalleeInfo()) {
      Module &KernelModule = getModule(KernelInfo);
      PROTEUS_DBG(Logger::logs("proteus")
                  << "=== LAMBDA MATCHING\n"
                  << "Caller trigger " << KernelInfo.getName() << " -> "
                  << demangle(KernelInfo.getName()) << "\n");

      SmallVector<std::pair<std::string, StringRef>> LambdaCalleeInfo;
      for (auto &F : KernelModule.getFunctionList()) {
        PROTEUS_DBG(Logger::logs("proteus")
                    << " Trying F " << demangle(F.getName().str()) << "\n ");
        auto OptionalMapIt =
            LambdaRegistry::instance().matchJitVariableMap(F.getName());
        if (OptionalMapIt)
          LambdaCalleeInfo.emplace_back(F.getName(),
                                        OptionalMapIt.value()->first);
      }

      KernelInfo.setLambdaCalleeInfo(std::move(LambdaCalleeInfo));
    }

    for (auto &[FnName, LambdaType] : KernelInfo.getLambdaCalleeInfo()) {
      const SmallVector<RuntimeConstant> &Values =
          LR.getJitVariables(LambdaType);
      LambdaJitValuesVec.insert(LambdaJitValuesVec.end(), Values.begin(),
                                Values.end());
    }
  }

  void insertRegisterVar(const char *VarName, const void *Addr) {
    VarNameToDevPtr[VarName] = Addr;
  }
  void registerLinkedBinary(FatbinWrapperT *FatbinWrapper,
                            const char *ModuleId);
  void registerFatBinary(void *Handle, FatbinWrapperT *FatbinWrapper,
                         const char *ModuleId);
  void registerFatBinaryEnd();
  void registerFunction(void *Handle, void *Kernel, char *KernelName,
                        int32_t *RCIndices, int32_t *RCTypes, int32_t NumRCs);

  void *CurHandle = nullptr;
  std::unordered_map<std::string, FatbinWrapperT *> ModuleIdToFatBinary;
  std::unordered_map<const void *, BinaryInfo> HandleToBinaryInfo;
  SmallVector<std::string> GlobalLinkedModuleIds;
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;

  bool containsJITKernelInfo(const void *Func) {
    return JITKernelInfoMap.contains(Func);
  }

  std::optional<std::reference_wrapper<JITKernelInfo>>
  getJITKernelInfo(const void *Func) {
    if (!containsJITKernelInfo(Func)) {
      return std::nullopt;
    }
    return JITKernelInfoMap[Func];
  }

  HashT getStaticHash(JITKernelInfo &KernelInfo) {
    if (KernelInfo.hasStaticHash())
      return KernelInfo.getStaticHash();

    BinaryInfo &BinInfo = KernelInfo.getBinaryInfo();

    if (BinInfo.hasModuleHash()) {
      KernelInfo.createStaticHash(BinInfo.getModuleHash());
      return KernelInfo.getStaticHash();
    }

    HashT ModuleHash = static_cast<ImplT &>(*this).getModuleHash(BinInfo);

    KernelInfo.createStaticHash(BinInfo.getModuleHash());
    return KernelInfo.getStaticHash();
  }

  void finalize() {
    if (Config::get().ProteusAsyncCompilation)
      CompilerAsync::instance(Config::get().ProteusAsyncThreads)
          .joinAllThreads();
  }

private:
  //------------------------------------------------------------------
  // Begin Methods implemented in the derived device engine class.
  //------------------------------------------------------------------
  void *resolveDeviceGlobalAddr(const void *Addr) {
    return static_cast<ImplT &>(*this).resolveDeviceGlobalAddr(Addr);
  }

  void setLaunchBoundsForKernel(Module &M, Function &F, size_t GridSize,
                                int BlockSize) {
    static_cast<ImplT &>(*this).setLaunchBoundsForKernel(M, F, GridSize,
                                                         BlockSize);
  }

  void setKernelDims(Module &M, dim3 &GridDim, dim3 &BlockDim) {
    proteus::setKernelDims(M, GridDim, BlockDim);
  }

  DeviceError_t launchKernelFunction(KernelFunction_t KernelFunc, dim3 GridDim,
                                     dim3 BlockDim, void **KernelArgs,
                                     uint64_t ShmemSize,
                                     DeviceStream_t Stream) {
    TIMESCOPE(__FUNCTION__);
    return static_cast<ImplT &>(*this).launchKernelFunction(
        KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize, Stream);
  }

  void relinkGlobalsObject(MemoryBufferRef Object) {
    TIMESCOPE(__FUNCTION__);
    proteus::relinkGlobalsObject(Object, VarNameToDevPtr);
  }

  KernelFunction_t getKernelFunctionFromImage(StringRef KernelName,
                                              const void *Image) {
    TIMESCOPE(__FUNCTION__);
    return static_cast<ImplT &>(*this).getKernelFunctionFromImage(KernelName,
                                                                  Image);
  }

  //------------------------------------------------------------------
  // End Methods implemented in the derived device engine class.
  //------------------------------------------------------------------

  void pruneIR(Module &M);

  void internalize(Module &M, StringRef KernelName);

  void replaceGlobalVariablesWithPointers(Module &M);

protected:
  JitEngineDevice() {}

  ~JitEngineDevice() {
    CodeCache.printStats();
    StorageCache.printStats();
  }

  JitCache<KernelFunction_t> CodeCache;
  JitStorageCache<KernelFunction_t> StorageCache;
  std::string DeviceArch;
  std::unordered_map<std::string, const void *> VarNameToDevPtr;

  DenseMap<const void *, JITKernelInfo> JITKernelInfoMap;
};

template <typename ImplT> void JitEngineDevice<ImplT>::pruneIR(Module &M) {
  TIMESCOPE("pruneIR");
  proteus::pruneIR(M);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::internalize(Module &M, StringRef KernelName) {
  proteus::internalize(M, KernelName);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::replaceGlobalVariablesWithPointers(Module &M) {
  TIMESCOPE(__FUNCTION__)

  proteus::replaceGlobalVariablesWithPointers(M, VarNameToDevPtr);

#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "=== Linked M\n" << M << "=== End of Linked M\n";
  if (verifyModule(M, &errs()))
    PROTEUS_FATAL_ERROR(
        "After linking, broken module found, JIT compilation aborted!");
  else
    Logger::logs("proteus") << "Module verified!\n";
#endif
}

template <typename ImplT>
typename DeviceTraits<ImplT>::DeviceError_t
JitEngineDevice<ImplT>::compileAndRun(
    JITKernelInfo &KernelInfo, dim3 GridDim, dim3 BlockDim, void **KernelArgs,
    uint64_t ShmemSize, typename DeviceTraits<ImplT>::DeviceStream_t Stream) {
  TIMESCOPE("compileAndRun");

  // Lazy initialize the map of device global variables to device pointers by
  // resolving the host address to the device address. For HIP it is fine to do
  // this earlier (e.g., instertRegisterVar), but CUDA can't. So, we initialize
  // this here the first time we need to compile a kernel.
  static std::once_flag Flag;
  std::call_once(Flag, [&]() {
    for (auto &[GlobalName, HostAddr] : VarNameToDevPtr) {
      void *DevPtr = resolveDeviceGlobalAddr(HostAddr);
      VarNameToDevPtr.at(GlobalName) = DevPtr;
    }
  });

  SmallVector<RuntimeConstant> RCVec;
  SmallVector<RuntimeConstant> LambdaJitValuesVec;

  getRuntimeConstantValues(KernelArgs, KernelInfo.getRCIndices(),
                           KernelInfo.getRCTypes(), RCVec);
  getLambdaJitValues(KernelInfo, LambdaJitValuesVec);

  HashT HashValue =
      hash(getStaticHash(KernelInfo), RCVec, LambdaJitValuesVec, GridDim.x,
           GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z);

  typename DeviceTraits<ImplT>::KernelFunction_t KernelFunc =
      CodeCache.lookup(HashValue);
  if (KernelFunc)
    return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                ShmemSize, Stream);

  // NOTE: we don't need a suffix to differentiate kernels, each specialization
  // will be in its own module uniquely identify by HashValue. It exists only
  // for debugging purposes to verify that the jitted kernel executes.
  std::string Suffix = mangleSuffix(HashValue);
  std::string KernelMangled = (KernelInfo.getName() + Suffix);

  if (Config::get().ProteusUseStoredCache) {
    // If there device global variables, lookup the IR and codegen object
    // before launching. Else, if there aren't device global variables, lookup
    // the object and launch.

    // TODO: Check for globals is very conservative and always re-builds from
    // LLVM IR even if the Jit module does not use global variables.  A better
    // solution is to keep track of whether a kernel uses gvars (store a flag in
    // the cache file?) and load the object in case it does not use any.
    // TODO: Can we use RTC interfaces for fast linking on object files?
    auto CacheBuf = StorageCache.lookup(HashValue);
    if (CacheBuf) {
      if (!Config::get().ProteusRelinkGlobalsByCopy)
        relinkGlobalsObject(CacheBuf->getMemBufferRef());

      auto KernelFunc =
          getKernelFunctionFromImage(KernelMangled, CacheBuf->getBufferStart());

      CodeCache.insert(HashValue, KernelFunc, KernelInfo.getName(), RCVec);

      return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                  ShmemSize, Stream);
    }
  }

  MemoryBufferRef KernelBitcode = getBitcode(KernelInfo);
  std::unique_ptr<MemoryBuffer> ObjBuf = nullptr;

  if (Config::get().ProteusAsyncCompilation) {
    auto &Compiler = CompilerAsync::instance(Config::get().ProteusAsyncThreads);
    // If there is no compilation pending for the specialization, post the
    // compilation task to the compiler.
    if (!Compiler.isCompilationPending(HashValue)) {
      PROTEUS_DBG(Logger::logs("proteus") << "Compile async for HashValue "
                                          << HashValue.toString() << "\n");

      Compiler.compile(CompilationTask{
          KernelBitcode, HashValue, KernelInfo.getName(), Suffix, BlockDim,
          GridDim, KernelInfo.getRCIndices(), RCVec,
          KernelInfo.getLambdaCalleeInfo(), VarNameToDevPtr,
          GlobalLinkedBinaries, DeviceArch,
          /* CGOption */ Config::get().ProteusCodegen,
          /* DumpIR */ Config::get().ProteusDumpLLVMIR,
          /* RelinkGlobalsByCopy */ Config::get().ProteusRelinkGlobalsByCopy,
          /*SpecializeArgs=*/Config::get().ProteusSpecializeArgs,
          /*SpecializeDims=*/Config::get().ProteusSpecializeDims,
          /*SpecializeLaunchBounds=*/
          Config::get().ProteusSpecializeLaunchBounds});
    }

    // Compilation is pending, try to get the compilation result buffer. If
    // buffer is null, compilation is not done, so execute the AOT version
    // directly.
    ObjBuf = Compiler.takeCompilationResult(
        HashValue, Config::get().ProteusAsyncTestBlocking);
    if (!ObjBuf) {
      return launchKernelDirect(KernelInfo.getKernel(), GridDim, BlockDim,
                                KernelArgs, ShmemSize, Stream);
    }
  } else {
    // Process through synchronous compilation.
    ObjBuf = CompilerSync::instance().compile(CompilationTask{
        KernelBitcode, HashValue, KernelInfo.getName(), Suffix, BlockDim,
        GridDim, KernelInfo.getRCIndices(), RCVec,
        KernelInfo.getLambdaCalleeInfo(), VarNameToDevPtr, GlobalLinkedBinaries,
        DeviceArch,
        /* CGOption */ Config::get().ProteusCodegen,
        /* DumpIR */ Config::get().ProteusDumpLLVMIR,
        /* RelinkGlobalsByCopy */ Config::get().ProteusRelinkGlobalsByCopy,
        /*SpecializeArgs=*/Config::get().ProteusSpecializeArgs,
        /*SpecializeDims=*/Config::get().ProteusSpecializeDims,
        /*SpecializeLaunchBounds=*/
        Config::get().ProteusSpecializeLaunchBounds});
  }

  if (!ObjBuf)
    PROTEUS_FATAL_ERROR("Expected non-null object");

  KernelFunc = proteus::getKernelFunctionFromImage(
      KernelMangled, ObjBuf->getBufferStart(),
      Config::get().ProteusRelinkGlobalsByCopy, VarNameToDevPtr);

  CodeCache.insert(HashValue, KernelFunc, KernelInfo.getName(), RCVec);
  if (Config::get().ProteusUseStoredCache) {
    StorageCache.store(HashValue, ObjBuf->getMemBufferRef());
  }

  return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                              ShmemSize, Stream);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerFatBinary(void *Handle,
                                               FatbinWrapperT *FatbinWrapper,
                                               const char *ModuleId) {
  CurHandle = Handle;
  PROTEUS_DBG(Logger::logs("proteus")
              << "Register fatbinary Handle " << Handle << " FatbinWrapper "
              << FatbinWrapper << " Binary " << (void *)FatbinWrapper->Binary
              << " ModuleId " << ModuleId << "\n");
  if (FatbinWrapper->PrelinkedFatbins) {
    // This is RDC compilation, just insert the FatbinWrapper and ignore the
    // ModuleId coming from the link.stub.
    HandleToBinaryInfo.emplace(Handle, BinaryInfo{FatbinWrapper, {}});

    // Initialize GlobalLinkedBinaries with prelinked fatbins.
    void *Ptr = FatbinWrapper->PrelinkedFatbins[0];
    for (int I = 0; Ptr != nullptr;
         ++I, Ptr = FatbinWrapper->PrelinkedFatbins[I]) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << "I " << I << " PrelinkedFatbin " << Ptr << "\n");
      GlobalLinkedBinaries.insert(Ptr);
    }
  } else {
    // This is non-RDC compilation, associate the ModuleId of the JIT bitcode in
    // the module with the FatbinWrapper.
    ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
    HandleToBinaryInfo.emplace(Handle, BinaryInfo{FatbinWrapper, {ModuleId}});
  }
}

template <typename ImplT> void JitEngineDevice<ImplT>::registerFatBinaryEnd() {
  PROTEUS_DBG(Logger::logs("proteus") << "Register fatbinary end\n");
  // Erase linked binaries for which we have LLVM IR code, those binaries are
  // stored in the ModuleIdToFatBinary map.
  for (auto &[ModuleId, FatbinWrapper] : ModuleIdToFatBinary)
    GlobalLinkedBinaries.erase((void *)FatbinWrapper->Binary);

  CurHandle = nullptr;
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerFunction(void *Handle, void *Kernel,
                                              char *KernelName,
                                              int32_t *RCIndices,
                                              int32_t *RCTypes,
                                              int32_t NumRCs) {
  PROTEUS_DBG(Logger::logs("proteus") << "Register function " << Kernel
                                      << " To Handle " << Handle << "\n");
  // NOTE: HIP RDC might call multiple times the registerFunction for the same
  // kernel, which has weak linkage, when it comes from different translation
  // units. Either the first or the second call can prevail and should be
  // equivalent. We let the first one prevail.
  if (JITKernelInfoMap.contains(Kernel)) {
    PROTEUS_DBG(Logger::logs("proteus")
                << "Warning: duplicate register function for kernel " +
                       std::string(KernelName)
                << "\n");
    return;
  }

  if (!HandleToBinaryInfo.count(Handle))
    PROTEUS_FATAL_ERROR("Expected Handle in map");
  BinaryInfo &BinInfo = HandleToBinaryInfo[Handle];

  JITKernelInfoMap[Kernel] =
      JITKernelInfo{Kernel, BinInfo, KernelName, RCIndices, RCTypes, NumRCs};
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerLinkedBinary(FatbinWrapperT *FatbinWrapper,
                                                  const char *ModuleId) {
  PROTEUS_DBG(Logger::logs("proteus")
              << "Register linked binary FatBinary " << FatbinWrapper
              << " Binary " << (void *)FatbinWrapper->Binary << " ModuleId "
              << ModuleId << "\n");
  if (CurHandle) {
    if (!HandleToBinaryInfo.count(CurHandle))
      PROTEUS_FATAL_ERROR("Expected CurHandle in map");

    HandleToBinaryInfo[CurHandle].addModuleId(ModuleId);
  } else
    GlobalLinkedModuleIds.push_back(ModuleId);

  ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
}

} // namespace proteus

#endif
