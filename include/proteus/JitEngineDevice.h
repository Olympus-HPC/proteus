//===-- JitEngineDevice.cpp -- Base JIT Engine Device header impl. --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINEDEVICE_H
#define PROTEUS_JITENGINEDEVICE_H

#include "proteus/Caching/MemoryCache.h"
#include "proteus/Caching/ObjectCacheRegistry.h"
#include "proteus/Cloning.h"
#include "proteus/CompilerAsync.h"
#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/CompilerSync.h"
#include "proteus/CoreDevice.h"
#include "proteus/CoreLLVM.h"
#include "proteus/Debug.h"
#include "proteus/Hashing.h"
#include "proteus/JitEngine.h"
#include "proteus/TimeTracing.h"
#include "proteus/Utils.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/CallGraph.h>
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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

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
  std::unique_ptr<MemoryBuffer> DeviceBinary;
  std::unordered_map<std::string, GlobalVarInfo> VarNameToGlobalInfo;
  bool GlobalsMapped;
  std::once_flag Flag;

public:
  BinaryInfo() = default;
  BinaryInfo(FatbinWrapperT *FatbinWrapper,
             SmallVector<std::string> &&LinkedModuleIds)
      : FatbinWrapper(FatbinWrapper), Ctx(std::make_unique<LLVMContext>()),
        LinkedModuleIds(LinkedModuleIds), LinkedModule(nullptr),
        ExtractedModules(std::nullopt), ModuleCallGraph(std::nullopt),
        DeviceBinary(nullptr), GlobalsMapped(false) {}

  FatbinWrapperT *getFatbinWrapper() const { return FatbinWrapper; }

  std::unique_ptr<LLVMContext> &getLLVMContext() { return Ctx; }

  bool hasLinkedModule() const { return (LinkedModule != nullptr); }
  Module &getLinkedModule() {
    if (!LinkedModule) {
      if (!hasExtractedModules())
        reportFatalError("Expected extracted modules");

      Timer T;
      // Avoid linking when there's a single module by moving it instead and
      // making sure it's materialized for call graph analysis.
      if (ExtractedModules->size() == 1) {
        LinkedModule = ExtractedModules->front().get();
        if (auto E = LinkedModule->materializeAll())
          reportFatalError("Error materializing " + toString(std::move(E)));
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
    // This should be called only once when cloning the kernel module to
    // cache.
    SmallVector<std::reference_wrapper<Module>> ModulesRef;
    for (auto &M : ExtractedModules.value())
      ModulesRef.emplace_back(*M);

    return ModulesRef;
  }
  void setExtractedModules(SmallVector<std::unique_ptr<Module>> &Modules) {
    ExtractedModules = std::move(Modules);
  }

  bool hasModuleHash() const { return ExtractedModuleHash.has_value(); }
  HashT getModuleHash() const {
    if (!hasModuleHash())
      reportFatalError("Expected module hash to be set");

    return ExtractedModuleHash.value();
  }
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
        reportFatalError("Expected non-null linked module");
      ModuleCallGraph.emplace(CallGraph(*LinkedModule));
    }
    return ModuleCallGraph.value();
  }

  bool hasDeviceBinary() { return (DeviceBinary != nullptr); }
  MemoryBufferRef getDeviceBinary() {
    if (!hasDeviceBinary())
      reportFatalError("Expected non-null device binary");
    return DeviceBinary->getMemBufferRef();
  }
  void setDeviceBinary(std::unique_ptr<MemoryBuffer> DeviceBinaryBuffer) {
    DeviceBinary = std::move(DeviceBinaryBuffer);
  }

  void addModuleId(const char *ModuleId) {
    LinkedModuleIds.push_back(ModuleId);
  }

  void registerGlobalVar(const char *VarName, const void *Addr,
                         uint64_t VarSize) {
    VarNameToGlobalInfo.emplace(VarName, GlobalVarInfo(Addr, nullptr, VarSize));
  }

  void mapGlobals() {
    std::call_once(Flag, [&]() {
      for (auto &[GlobalName, GVI] : VarNameToGlobalInfo) {
        void *DevPtr = resolveDeviceGlobalAddr(GVI.HostAddr);
        VarNameToGlobalInfo.at(GlobalName).DevAddr = DevPtr;
      }
      auto TraceOut = [](std::unordered_map<std::string, GlobalVarInfo>
                             &VarNameToGlobalInfo) {
        SmallString<128> S;
        raw_svector_ostream OS(S);
        for (auto &[GlobalName, GVI] : VarNameToGlobalInfo) {
          OS << "[GVarInfo]: " << GlobalName << " HAddr:" << GVI.HostAddr
             << " DevAddr:" << GVI.DevAddr << " VarSize:" << GVI.VarSize
             << "\n";
        }

        return S;
      };
      if (Config::get().ProteusTraceOutput >= 1)
        Logger::trace(TraceOut(VarNameToGlobalInfo));
      GlobalsMapped = true;
    });
  }

  std::unordered_map<std::string, GlobalVarInfo> &getVarNameToGlobalInfo() {
    return VarNameToGlobalInfo;
  }

  auto &getModuleIds() { return LinkedModuleIds; }
};

class JITKernelInfo {
  std::optional<void *> Kernel;
  std::unique_ptr<LLVMContext> Ctx;
  std::string Name;
  ArrayRef<RuntimeConstantInfo *> RCInfoArray;
  std::optional<std::unique_ptr<Module>> ExtractedModule;
  std::optional<std::unique_ptr<MemoryBuffer>> Bitcode;
  std::optional<std::reference_wrapper<BinaryInfo>> BinInfo;
  std::optional<HashT> StaticHash;
  std::optional<SmallVector<std::pair<std::string, StringRef>>>
      LambdaCalleeInfo;

public:
  JITKernelInfo(void *Kernel, BinaryInfo &BinInfo, char const *Name,
                ArrayRef<RuntimeConstantInfo *> RCInfoArray)
      : Kernel(Kernel), Ctx(std::make_unique<LLVMContext>()), Name(Name),
        RCInfoArray(RCInfoArray), ExtractedModule(std::nullopt),
        Bitcode{std::nullopt}, BinInfo(BinInfo),
        LambdaCalleeInfo(std::nullopt) {}

  JITKernelInfo() = default;
  void *getKernel() const {
    assert(Kernel.has_value() && "Expected Kernel is inited");
    return Kernel.value();
  }
  std::unique_ptr<LLVMContext> &getLLVMContext() { return Ctx; }
  const std::string &getName() const { return Name; }
  ArrayRef<RuntimeConstantInfo *> getRCInfoArray() const { return RCInfoArray; }
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
        reportFatalError("Unsupported kernel cloning option");
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

    // If the module is not in the provided context due to cloning, roundtrip
    // it using bitcode. Re-use the roundtrip bitcode to return it.
    if (&KernelModule->getContext() != &Ctx) {
      SmallVector<char> CloneBuffer;
      raw_svector_ostream OS(CloneBuffer);
      WriteBitcodeToFile(*KernelModule, OS);
      StringRef CloneStr = StringRef(CloneBuffer.data(), CloneBuffer.size());
      auto ExpectedKernelModule =
          parseBitcodeFile(MemoryBufferRef{CloneStr, KernelName}, Ctx);
      if (auto E = ExpectedKernelModule.takeError())
        reportFatalError("Error parsing bitcode: " + toString(std::move(E)));

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
      reportFatalError("Unexpected KernelInfo has module but not bitcode");

    if (KernelInfo.hasBitcode())
      reportFatalError("Unexpected KernelInfo has bitcode but not module");

    BinaryInfo &BinInfo = KernelInfo.getBinaryInfo();

    Timer T;
    auto [KernelModule, BitcodeBuffer] = extractKernelModule(
        BinInfo, KernelInfo.getName(), *KernelInfo.getLLVMContext());

    if (!KernelModule)
      reportFatalError("Expected non-null kernel module");
    if (!BitcodeBuffer)
      reportFatalError("Expected non-null kernel bitcode");

    KernelInfo.setModule(std::move(KernelModule));
    KernelInfo.setBitcode(std::move(BitcodeBuffer));
    PROTEUS_TIMER_OUTPUT(Logger::outs("proteus")
                         << "Extract kernel module " << T.elapsed() << " ms\n");
  }

  Module &getModule(JITKernelInfo &KernelInfo) {
    if (!KernelInfo.hasModule())
      extractModuleAndBitcode(KernelInfo);

    if (!KernelInfo.hasModule())
      reportFatalError("Expected module in KernelInfo");

    return KernelInfo.getModule();
  }

  MemoryBufferRef getBitcode(JITKernelInfo &KernelInfo) {
    if (!KernelInfo.hasBitcode())
      extractModuleAndBitcode(KernelInfo);

    if (!KernelInfo.hasBitcode())
      reportFatalError("Expected bitcode in KernelInfo");

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

  void insertRegisterVar(void *Handle, const char *VarName, const void *Addr,
                         uint64_t VarSize) {
    if (!HandleToBinaryInfo.count(Handle))
      reportFatalError("Expected Handle in map");
    BinaryInfo &BinInfo = HandleToBinaryInfo[Handle];

    BinInfo.registerGlobalVar(VarName, Addr, VarSize);
  }

  void registerLinkedBinary(FatbinWrapperT *FatbinWrapper,
                            const char *ModuleId);
  void registerFatBinary(void *Handle, FatbinWrapperT *FatbinWrapper,
                         const char *ModuleId);
  void registerFatBinaryEnd();
  void registerFunction(void *Handle, void *Kernel, char *KernelName,
                        ArrayRef<RuntimeConstantInfo *> RCInfoArray);

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

  static void initCacheChain() {
    ObjectCacheRegistry::instance().create("JitEngineDevice");
  }

  static void finalizeCacheChain() {
    if (auto CacheOpt =
            ObjectCacheRegistry::instance().get("JitEngineDevice")) {
      CacheOpt->get().finalize();
    }
  }

  void ensureProteusInitialized() const {
    if (!ObjectCacheRegistry::instance().get("JitEngineDevice"))
      reportFatalError(
          "proteus not initialized. Call proteus::init() before using JIT "
          "compilation.");
  }

public:
  void finalize() {
    if (Config::get().ProteusAsyncCompilation)
      CompilerAsync::instance(Config::get().ProteusAsyncThreads)
          .joinAllThreads();

    finalizeCacheChain();
  }

  StringRef getDeviceArch() const { return DeviceArch; }

protected:
  JitEngineDevice() {}

  ~JitEngineDevice() {
    CodeCache.printStats();
    if (auto CacheOpt =
            ObjectCacheRegistry::instance().get("JitEngineDevice")) {
      CacheOpt->get().printStats();
    }
  }

  MemoryCache<KernelFunction_t> CodeCache{"JitEngineDevice"};
  std::string DeviceArch;

  DenseMap<const void *, JITKernelInfo> JITKernelInfoMap;
};

template <typename ImplT>
typename DeviceTraits<ImplT>::DeviceError_t
JitEngineDevice<ImplT>::compileAndRun(
    JITKernelInfo &KernelInfo, dim3 GridDim, dim3 BlockDim, void **KernelArgs,
    uint64_t ShmemSize, typename DeviceTraits<ImplT>::DeviceStream_t Stream) {
  TIMESCOPE("compileAndRun");
  ensureProteusInitialized();

  auto &BinInfo = KernelInfo.getBinaryInfo();

  // Lazy initialize the map of device global variables to device pointers by
  // resolving the host address to the device address. For HIP it is fine to
  // do this earlier (e.g., instertRegisterVar), but CUDA can't. So, we
  // initialize this here the first time we need to compile a kernel.
  BinInfo.mapGlobals();

  SmallVector<RuntimeConstant> RCVec =
      getRuntimeConstantValues(KernelArgs, KernelInfo.getRCInfoArray());

  SmallVector<RuntimeConstant> LambdaJitValuesVec;
  getLambdaJitValues(KernelInfo, LambdaJitValuesVec);

  HashT HashValue =
      hash(getStaticHash(KernelInfo), RCVec, LambdaJitValuesVec, GridDim.x,
           GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z);

  typename DeviceTraits<ImplT>::KernelFunction_t KernelFunc =
      CodeCache.lookup(HashValue);
  if (KernelFunc)
    return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                ShmemSize, Stream);

  // NOTE: we don't need a suffix to differentiate kernels, each
  // specialization will be in its own module uniquely identify by HashValue.
  // It exists only for debugging purposes to verify that the jitted kernel
  // executes.
  std::string Suffix = HashValue.toMangledSuffix();
  std::string KernelMangled = (KernelInfo.getName() + Suffix);

  if (auto CacheOpt = getLibraryCache("JitEngineDevice")) {
    auto CompiledLib = CacheOpt->get().lookup(HashValue);
    if (CompiledLib) {
      if (!Config::get().ProteusRelinkGlobalsByCopy)
        relinkGlobalsObject(CompiledLib->ObjectModule->getMemBufferRef(),
                            BinInfo.getVarNameToGlobalInfo());

      auto KernelFunc = proteus::getKernelFunctionFromImage(
          KernelMangled, CompiledLib->ObjectModule->getBufferStart(),
          Config::get().ProteusRelinkGlobalsByCopy,
          BinInfo.getVarNameToGlobalInfo());

      CodeCache.insert(HashValue, KernelFunc, KernelInfo.getName());

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
          GridDim, RCVec, KernelInfo.getLambdaCalleeInfo(),
          BinInfo.getVarNameToGlobalInfo(), GlobalLinkedBinaries, DeviceArch,
          /*CodeGenConfig */ Config::get().getCGConfig(KernelInfo.getName()),
          /*DumpIR*/ Config::get().ProteusDumpLLVMIR,
          /*RelinkGlobalsByCopy*/ Config::get().ProteusRelinkGlobalsByCopy});
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
        GridDim, RCVec, KernelInfo.getLambdaCalleeInfo(),
        BinInfo.getVarNameToGlobalInfo(), GlobalLinkedBinaries, DeviceArch,
        /*CodeGenConfig */ Config::get().getCGConfig(KernelInfo.getName()),
        /*DumpIR*/ Config::get().ProteusDumpLLVMIR,
        /*RelinkGlobalsByCopy*/ Config::get().ProteusRelinkGlobalsByCopy});
  }

  if (!ObjBuf)
    reportFatalError("Expected non-null object");

  KernelFunc = proteus::getKernelFunctionFromImage(
      KernelMangled, ObjBuf->getBufferStart(),
      Config::get().ProteusRelinkGlobalsByCopy,
      BinInfo.getVarNameToGlobalInfo());

  CodeCache.insert(HashValue, KernelFunc, KernelInfo.getName());
  if (auto CacheOpt = getLibraryCache("JitEngineDevice"))
    CacheOpt->get().store(HashValue,
                          CacheEntry::staticObject(ObjBuf->getMemBufferRef()));

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
    HandleToBinaryInfo.try_emplace(Handle, FatbinWrapper,
                                   SmallVector<std::string>{});

    // Initialize GlobalLinkedBinaries with prelinked fatbins.
    void *Ptr = FatbinWrapper->PrelinkedFatbins[0];
    for (int I = 0; Ptr != nullptr;
         ++I, Ptr = FatbinWrapper->PrelinkedFatbins[I]) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << "I " << I << " PrelinkedFatbin " << Ptr << "\n");
      GlobalLinkedBinaries.insert(Ptr);
    }
  } else {
    // This is non-RDC compilation, associate the ModuleId of the JIT bitcode
    // in the module with the FatbinWrapper.
    ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
    HandleToBinaryInfo.try_emplace(Handle, FatbinWrapper,
                                   SmallVector<std::string>{ModuleId});
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
void JitEngineDevice<ImplT>::registerFunction(
    void *Handle, void *Kernel, char *KernelName,
    ArrayRef<RuntimeConstantInfo *> RCInfoArray) {
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
    reportFatalError("Expected Handle in map");
  BinaryInfo &BinInfo = HandleToBinaryInfo[Handle];

  PROTEUS_DBG(Logger::logs("proteus")
              << "Register function  " << KernelName << " with binary handle "
              << Handle << "\n");

  JITKernelInfoMap[Kernel] =
      JITKernelInfo{Kernel, BinInfo, KernelName, RCInfoArray};
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
      reportFatalError("Expected CurHandle in map");

    HandleToBinaryInfo[CurHandle].addModuleId(ModuleId);
  } else
    GlobalLinkedModuleIds.push_back(ModuleId);

  ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
}

} // namespace proteus

#endif
