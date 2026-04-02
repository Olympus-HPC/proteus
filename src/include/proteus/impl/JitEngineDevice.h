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

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Init.h"
#include "proteus/TimeTracing.h"
#include "proteus/impl/AutoReadOnlyCaptures.h"
#include "proteus/impl/Caching/MemoryCache.h"
#include "proteus/impl/Caching/ObjectCacheChain.h"
#include "proteus/impl/Cloning.h"
#include "proteus/impl/CompilerAsync.h"
#include "proteus/impl/CompilerSync.h"
#include "proteus/impl/CoreDevice.h"
#include "proteus/impl/CoreLLVM.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Hashing.h"
#include "proteus/impl/JitEngine.h"
#include "proteus/impl/JitEngineInfoRegistry.h"
#include "proteus/impl/LambdaRegistry.h"
#include "proteus/impl/LambdaSpecializationInfo.h"
#include "proteus/impl/Utils.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
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
#include <llvm/IR/IntrinsicInst.h>
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
  std::once_flag Flag;

public:
  BinaryInfo() = default;
  BinaryInfo(FatbinWrapperT *FatbinWrapper,
             SmallVector<std::string> &&LinkedModuleIds)
      : FatbinWrapper(FatbinWrapper), Ctx(std::make_unique<LLVMContext>()),
        LinkedModuleIds(LinkedModuleIds), LinkedModule(nullptr),
        ExtractedModules(std::nullopt), ModuleCallGraph(std::nullopt),
        DeviceBinary(nullptr) {}

  FatbinWrapperT *getFatbinWrapper() const { return FatbinWrapper; }

  std::unique_ptr<LLVMContext> &getLLVMContext() { return Ctx; }

  bool hasLinkedModule() const { return (LinkedModule != nullptr); }
  Module &getLinkedModule() {
    TIMESCOPE(BinaryInfo, getLinkedModule);
    if (!LinkedModule) {
      if (!hasExtractedModules())
        reportFatalError("Expected extracted modules");

      Timer T(Config::get().ProteusEnableTimers);
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

  void insertGlobalVar(const char *VarName, const void *HostAddr,
                       const void *DeviceAddr, uint64_t VarSize) {
    auto KV = VarNameToGlobalInfo.emplace(
        VarName, GlobalVarInfo(HostAddr, DeviceAddr, VarSize));

    auto TraceOut = [&KV]() {
      auto GlobalName = KV.first->first;
      auto &GVI = KV.first->second;

      SmallString<128> S;
      raw_svector_ostream OS(S);
      OS << "[GVarInfo]: " << GlobalName << " HAddr:" << GVI.HostAddr
         << " DevAddr:" << GVI.DevAddr << " VarSize:" << GVI.VarSize << "\n";

      return S;
    };

    if (Config::get().traceSpecializations())
      Logger::trace(TraceOut());
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
  std::optional<SmallVector<proteus::LambdaCalleeInfo>> CachedLambdaCalleeInfo;

public:
  JITKernelInfo(void *Kernel, BinaryInfo &BinInfo, char const *Name,
                ArrayRef<RuntimeConstantInfo *> RCInfoArray)
      : Kernel(Kernel), Ctx(std::make_unique<LLVMContext>()), Name(Name),
        RCInfoArray(RCInfoArray), ExtractedModule(std::nullopt),
        Bitcode{std::nullopt}, BinInfo(BinInfo),
        CachedLambdaCalleeInfo(std::nullopt) {}

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

  bool hasLambdaCalleeInfo() { return CachedLambdaCalleeInfo.has_value(); }
  const auto &getLambdaCalleeInfo() { return CachedLambdaCalleeInfo.value(); }
  void
  setLambdaCalleeInfo(SmallVector<proteus::LambdaCalleeInfo> &&LambdaInfo) {
    CachedLambdaCalleeInfo = std::move(LambdaInfo);
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
    TIMESCOPE(JitEngineDevice, extractKernelModule);
    std::unique_ptr<Module> KernelModule =
        static_cast<ImplT &>(*this).tryExtractKernelModule(BinInfo, KernelName,
                                                           Ctx);
    std::unique_ptr<MemoryBuffer> Bitcode = nullptr;

    // If there is no ready-made kernel module from AOT, extract per-TU or the
    // single linked module and clone the kernel module.
    if (!KernelModule) {
      Timer T(Config::get().ProteusEnableTimers);
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
    TIMESCOPE(JitEngineDevice, extractModuleAndBitcode);

    if (KernelInfo.hasModule() && KernelInfo.hasBitcode())
      return;

    if (KernelInfo.hasModule())
      reportFatalError("Unexpected KernelInfo has module but not bitcode");

    if (KernelInfo.hasBitcode())
      reportFatalError("Unexpected KernelInfo has bitcode but not module");

    BinaryInfo &BinInfo = KernelInfo.getBinaryInfo();

    Timer T(Config::get().ProteusEnableTimers);
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

  std::optional<unsigned> mergeKernelArgIndex(std::optional<unsigned> Current,
                                              std::optional<unsigned> Candidate,
                                              bool &HasAmbiguity) const {
    if (!Candidate)
      return Current;
    if (!Current)
      return Candidate;
    if (*Current != *Candidate)
      HasAmbiguity = true;
    return Current;
  }

  std::optional<unsigned> matchKernelArgIndexByType(Function &KernelFn,
                                                    Type *ClosureTy) const {
    if (!ClosureTy)
      return std::nullopt;

    std::optional<unsigned> Result;
    bool HasAmbiguity = false;
    for (Argument &Arg : KernelFn.args()) {
      Type *KernelArgTy = Arg.getParamByRefType();
      if (!KernelArgTy)
        KernelArgTy = Arg.getParamByValType();
      if (KernelArgTy != ClosureTy)
        continue;

      Result = mergeKernelArgIndex(Result, Arg.getArgNo(), HasAmbiguity);
      if (HasAmbiguity)
        return std::nullopt;
    }

    return Result;
  }

  std::optional<Type *>
  findClosureStorageType(Value *V, SmallPtrSetImpl<Value *> &Seen) const {
    if (!V)
      return std::nullopt;
    if (!Seen.insert(V).second)
      return std::nullopt;

    if (auto *AI = dyn_cast<AllocaInst>(V))
      return AI->getAllocatedType();

    if (auto *Arg = dyn_cast<Argument>(V)) {
      if (Type *Ty = Arg->getParamByRefType())
        return Ty;
      if (Type *Ty = Arg->getParamByValType())
        return Ty;
      return std::nullopt;
    }

    if (auto *BC = dyn_cast<BitCastInst>(V))
      return findClosureStorageType(BC->getOperand(0), Seen);

    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V))
      return findClosureStorageType(ASC->getOperand(0), Seen);

    if (auto *LI = dyn_cast<LoadInst>(V))
      return findClosureStorageType(LI->getPointerOperand(), Seen);

    if (auto *GEP = dyn_cast<GetElementPtrInst>(V))
      return findClosureStorageType(GEP->getPointerOperand(), Seen);

    if (auto *PN = dyn_cast<PHINode>(V)) {
      std::optional<Type *> Result;
      for (Value *Incoming : PN->incoming_values()) {
        auto Candidate = findClosureStorageType(Incoming, Seen);
        if (!Candidate)
          continue;
        if (!Result) {
          Result = Candidate;
          continue;
        }
        if (*Result != *Candidate)
          return std::nullopt;
      }
      return Result;
    }

    if (auto *SI = dyn_cast<SelectInst>(V)) {
      auto TrueTy = findClosureStorageType(SI->getTrueValue(), Seen);
      auto FalseTy = findClosureStorageType(SI->getFalseValue(), Seen);
      if (!TrueTy)
        return FalseTy;
      if (!FalseTy)
        return TrueTy;
      if (*TrueTy != *FalseTy)
        return std::nullopt;
      return TrueTy;
    }

    return std::nullopt;
  }

  std::optional<unsigned>
  traceStoredKernelArgIndex(Function &KernelFn, Value *Ptr,
                            SmallPtrSetImpl<Value *> &SeenPointers) const {
    if (!SeenPointers.insert(Ptr).second)
      return std::nullopt;

    std::optional<unsigned> Result;
    bool HasAmbiguity = false;
    auto MergeCandidate = [&](Value *Candidate) {
      if (!Candidate)
        return;

      SmallPtrSet<Value *, 32> CandidateSeenValues;
      SmallPtrSet<Function *, 8> CandidateSeenFunctions;
      auto CandidateIndex =
          traceKernelArgIndex(KernelFn, Candidate->stripPointerCasts(),
                              CandidateSeenValues, CandidateSeenFunctions);
      Result = mergeKernelArgIndex(Result, CandidateIndex, HasAmbiguity);
    };

    for (User *User : Ptr->users()) {
      if (auto *MemCpy = dyn_cast<MemCpyInst>(User)) {
        if (MemCpy->getDest() == Ptr)
          MergeCandidate(MemCpy->getSource());
        continue;
      }

      if (auto *SI = dyn_cast<StoreInst>(User)) {
        if (SI->getPointerOperand() == Ptr)
          MergeCandidate(SI->getValueOperand());
        continue;
      }

      if (isa<BitCastInst>(User) || isa<AddrSpaceCastInst>(User) ||
          isa<GetElementPtrInst>(User) || isa<PHINode>(User) ||
          isa<SelectInst>(User)) {
        auto Candidate = traceStoredKernelArgIndex(KernelFn, cast<Value>(User),
                                                   SeenPointers);
        Result = mergeKernelArgIndex(Result, Candidate, HasAmbiguity);
      }

      if (HasAmbiguity)
        return std::nullopt;
    }

    return Result;
  }

  std::optional<unsigned>
  traceKernelArgIndex(Function &KernelFn, Value *V,
                      SmallPtrSetImpl<Value *> &SeenValues,
                      SmallPtrSetImpl<Function *> &SeenFunctions) const {
    if (!V)
      return std::nullopt;

    if (!SeenValues.insert(V).second)
      return std::nullopt;

    if (auto *BC = dyn_cast<BitCastInst>(V))
      return traceKernelArgIndex(KernelFn, BC->getOperand(0), SeenValues,
                                 SeenFunctions);

    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V))
      return traceKernelArgIndex(KernelFn, ASC->getOperand(0), SeenValues,
                                 SeenFunctions);

    if (auto *LI = dyn_cast<LoadInst>(V))
      return traceKernelArgIndex(KernelFn, LI->getPointerOperand(), SeenValues,
                                 SeenFunctions);

    if (auto *Arg = dyn_cast<Argument>(V)) {
      if (Arg->getParent() == &KernelFn)
        return Arg->getArgNo();

      Function *ParentFn = Arg->getParent();
      if (!SeenFunctions.insert(ParentFn).second)
        return std::nullopt;

      std::optional<unsigned> Result;
      bool HasAmbiguity = false;
      for (User *User : ParentFn->users()) {
        auto *CB = dyn_cast<CallBase>(User);
        if (!CB)
          continue;
        if (CB->getCalledOperand()->stripPointerCasts() != ParentFn)
          continue;
        if (Arg->getArgNo() >= CB->arg_size())
          continue;

        auto Candidate =
            traceKernelArgIndex(KernelFn, CB->getArgOperand(Arg->getArgNo()),
                                SeenValues, SeenFunctions);
        Result = mergeKernelArgIndex(Result, Candidate, HasAmbiguity);
        if (HasAmbiguity)
          return std::nullopt;
      }

      return Result;
    }

    if (auto *AI = dyn_cast<AllocaInst>(V)) {
      SmallPtrSet<Value *, 16> SeenPointers;
      return traceStoredKernelArgIndex(KernelFn, AI, SeenPointers);
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(V))
      return traceKernelArgIndex(KernelFn, GEP->getPointerOperand(), SeenValues,
                                 SeenFunctions);

    if (auto *PN = dyn_cast<PHINode>(V)) {
      std::optional<unsigned> Result;
      bool HasAmbiguity = false;
      for (Value *Incoming : PN->incoming_values()) {
        auto Candidate =
            traceKernelArgIndex(KernelFn, Incoming, SeenValues, SeenFunctions);
        Result = mergeKernelArgIndex(Result, Candidate, HasAmbiguity);
        if (HasAmbiguity)
          return std::nullopt;
      }
      return Result;
    }

    if (auto *SI = dyn_cast<SelectInst>(V)) {
      auto TrueCandidate = traceKernelArgIndex(KernelFn, SI->getTrueValue(),
                                               SeenValues, SeenFunctions);
      bool HasAmbiguity = false;
      auto Result =
          mergeKernelArgIndex(std::nullopt, TrueCandidate, HasAmbiguity);
      auto FalseCandidate = traceKernelArgIndex(KernelFn, SI->getFalseValue(),
                                                SeenValues, SeenFunctions);
      Result = mergeKernelArgIndex(Result, FalseCandidate, HasAmbiguity);
      if (HasAmbiguity)
        return std::nullopt;
      return Result;
    }

    return std::nullopt;
  }

  SmallVector<LambdaCalleeInfo>
  discoverLambdaCalleeInfo(JITKernelInfo &KernelInfo, Module &KernelModule) {
    Function *KernelFn = KernelModule.getFunction(KernelInfo.getName());
    if (!KernelFn)
      reportFatalError("Expected non-null kernel function");

    SmallVector<LambdaCalleeInfo> LambdaInfo;
    for (auto &F : KernelModule.getFunctionList()) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << " Trying F " << demangle(F.getName().str()) << "\n ");
      auto OptionalMapIt =
          LambdaRegistry::instance().matchJitVariableMap(F.getName());
      if (!OptionalMapIt)
        continue;

      std::optional<unsigned> KernelArgIndex;
      bool HasAmbiguity = false;
      for (User *User : F.users()) {
        auto *CB = dyn_cast<CallBase>(User);
        if (!CB)
          continue;
        if (CB->getCalledOperand()->stripPointerCasts() != &F)
          continue;
        if (CB->arg_empty())
          continue;

        SmallPtrSet<Value *, 32> SeenValues;
        SmallPtrSet<Function *, 8> SeenFunctions;
        auto Candidate = traceKernelArgIndex(*KernelFn, CB->getArgOperand(0),
                                             SeenValues, SeenFunctions);
        if (!Candidate) {
          SmallPtrSet<Value *, 16> SeenTypes;
          auto ClosureTy =
              findClosureStorageType(CB->getArgOperand(0), SeenTypes);
          Candidate =
              matchKernelArgIndexByType(*KernelFn, ClosureTy.value_or(nullptr));
        }
        KernelArgIndex =
            mergeKernelArgIndex(KernelArgIndex, Candidate, HasAmbiguity);
        if (HasAmbiguity)
          break;
      }

      LambdaCalleeInfo Info{
          F.getName().str(), OptionalMapIt.value()->first.str(),
          KernelArgIndex ? static_cast<int32_t>(*KernelArgIndex) : -1};
      LambdaInfo.push_back(std::move(Info));
    }

    return LambdaInfo;
  }

  void resolveLambdaSpecializations(
      JITKernelInfo &KernelInfo,
      SmallVector<ResolvedLambdaSpecializationInfo> &LambdaSpecializations,
      void **KernelArgs) {
    TIMESCOPE(JitEngineDevice, resolveLambdaSpecializations);
    LambdaRegistry &LR = LambdaRegistry::instance();
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

      KernelInfo.setLambdaCalleeInfo(
          discoverLambdaCalleeInfo(KernelInfo, KernelModule));
    }

    Module &KernelModule = getModule(KernelInfo);
    for (const auto &Info : KernelInfo.getLambdaCalleeInfo()) {
      // Get explicit jit_variable captures
      const SmallVector<RuntimeConstant> &ExplicitValues =
          LR.getJitVariables(Info.LambdaType);

      Function *LambdaFn = KernelModule.getFunction(Info.CalleeName);
      const void *LambdaClosure =
          (KernelArgs && Info.KernelArgIndex >= 0)
              ? KernelArgs[static_cast<size_t>(Info.KernelArgIndex)]
              : nullptr;
      SmallVector<RuntimeConstant> MergedValues =
          resolveLambdaSpecializationValues(
              ExplicitValues, LambdaFn, LambdaClosure,
              Config::get().ProteusAutoReadOnlyCaptures,
              Config::get().traceSpecializations());

      LambdaSpecializations.push_back(ResolvedLambdaSpecializationInfo{
          Info.CalleeName, std::move(MergedValues)});
    }
  }

  void registerVar(void *Handle, const char *VarName, const void *HostAddr,
                   uint64_t VarSize) {
    if (!HandleToBinaryInfo.count(Handle))
      reportFatalError("Expected Handle in map");
    BinaryInfo &BinInfo = HandleToBinaryInfo[Handle];

    void *DeviceAddr = resolveDeviceGlobalAddr(HostAddr);
    assert(DeviceAddr &&
           "Expected non-null device address for global variable");

    BinInfo.insertGlobalVar(VarName, HostAddr, DeviceAddr, VarSize);
  }

  void registerLinkedBinary(void *Handle, FatbinWrapperT *FatbinWrapper,
                            const char *ModuleId);
  void registerFatBinary(void *Handle, FatbinWrapperT *FatbinWrapper,
                         const char *ModuleId);
  void finalizeRegistration();
  void registerFunction(void *Handle, void *Kernel, char *KernelName,
                        ArrayRef<RuntimeConstantInfo *> RCInfoArray);

  std::unordered_map<std::string, FatbinWrapperT *> ModuleIdToFatBinary;
  std::unordered_map<const void *, BinaryInfo> HandleToBinaryInfo;
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

public:
  StringRef getDeviceArch() const { return DeviceArch; }

protected:
  JitEngineDevice() {
    TIMESCOPE(JitEngineDevice, JitEngineDevice);
    auto &JitEngineInfo = JitEngineInfoRegistry::instance();

    for (auto &[Handle, FatbinInfo] : JitEngineInfo.FatbinaryMap) {
      registerFatBinary(
          Handle, reinterpret_cast<FatbinWrapperT *>(FatbinInfo.FatbinWrapper),
          FatbinInfo.ModuleId);

      for (auto &LinkedBin : FatbinInfo.LinkedBinaries)
        registerLinkedBinary(
            Handle, reinterpret_cast<FatbinWrapperT *>(LinkedBin.FatbinWrapper),
            LinkedBin.ModuleId);

      for (auto &Func : FatbinInfo.Functions)
        registerFunction(Handle, Func.Kernel, Func.KernelName,
                         Func.RCInfoArray);

      for (auto &Var : FatbinInfo.Vars)
        registerVar(Var.Handle, Var.VarName, Var.HostAddr, Var.VarSize);
    }

    finalizeRegistration();

    if (Config::get().ProteusUseStoredCache)
      CacheChain.emplace("JitEngineDevice");

    if (Config::get().ProteusAsyncCompilation)
      AsyncCompiler =
          std::make_unique<CompilerAsync>(Config::get().ProteusAsyncThreads);
  }

  ~JitEngineDevice() {
    // Thread joining is handled by CompilerAsync's shutdown guard to ensure it
    // happens before static objects are destroyed. If this destructor does run,
    // joinAllThreads() is idempotent.
    if (AsyncCompiler)
      AsyncCompiler->joinAllThreads();

    if (Config::get().traceCacheStats())
      CodeCache.printStats();
    CodeCache.printKernelTrace();
    if (Config::get().traceCacheStats() && CacheChain)
      CacheChain->printStats();
  }

  MemoryCache<KernelFunction_t> CodeCache{"JitEngineDevice"};
  std::optional<ObjectCacheChain> CacheChain;
  std::string DeviceArch;

  DenseMap<const void *, JITKernelInfo> JITKernelInfoMap;
  std::unique_ptr<CompilerAsync> AsyncCompiler;
};

template <typename ImplT>
typename DeviceTraits<ImplT>::DeviceError_t
JitEngineDevice<ImplT>::compileAndRun(
    JITKernelInfo &KernelInfo, dim3 GridDim, dim3 BlockDim, void **KernelArgs,
    uint64_t ShmemSize, typename DeviceTraits<ImplT>::DeviceStream_t Stream) {
  TIMESCOPE(JitEngineDevice, compileAndRun);

  auto &BinInfo = KernelInfo.getBinaryInfo();

  SmallVector<RuntimeConstant> RCVec =
      getRuntimeConstantValues(KernelArgs, KernelInfo.getRCInfoArray());

  SmallVector<ResolvedLambdaSpecializationInfo> LambdaSpecializations;
  resolveLambdaSpecializations(KernelInfo, LambdaSpecializations, KernelArgs);
  // Determine the hash based on dimension specialization.  If we do not
  // specialize IR based on grid dimensions, avoid hashing on those to
  // eliminate repeated compilation overhead.
  HashT HashValue =
      hash(getStaticHash(KernelInfo), RCVec,
           ArrayRef<ResolvedLambdaSpecializationInfo>{LambdaSpecializations},
           BlockDim.x, BlockDim.y, BlockDim.z);
  if (Config::get().getCGConfig().specializeDims() ||
      Config::get().getCGConfig().specializeDimsRange())
    HashValue = hash(HashValue, GridDim.x, GridDim.y, GridDim.z);

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

  if (CacheChain) {
    auto CompiledLib = CacheChain->lookup(HashValue);
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
    //  If there is no compilation pending for the specialization, post the
    //  compilation task to the compiler.
    if (!AsyncCompiler->isCompilationPending(HashValue)) {
      PROTEUS_DBG(Logger::logs("proteus") << "Compile async for HashValue "
                                          << HashValue.toString() << "\n");

      AsyncCompiler->compile(CompilationTask{
          KernelBitcode, HashValue, KernelInfo.getName(), Suffix, BlockDim,
          GridDim, RCVec, LambdaSpecializations,
          BinInfo.getVarNameToGlobalInfo(), GlobalLinkedBinaries, DeviceArch,
          /*CodeGenConfig */ Config::get().getCGConfig(KernelInfo.getName()),
          /*DumpIR*/ Config::get().ProteusDumpLLVMIR,
          /*RelinkGlobalsByCopy*/ Config::get().ProteusRelinkGlobalsByCopy});
    }

    // Compilation is pending, try to get the compilation result buffer. If
    // buffer is null, compilation is not done, so execute the AOT version
    // directly.
    ObjBuf = AsyncCompiler->takeCompilationResult(
        HashValue, Config::get().ProteusAsyncTestBlocking);
    if (!ObjBuf) {
      return launchKernelDirect(KernelInfo.getKernel(), GridDim, BlockDim,
                                KernelArgs, ShmemSize, Stream);
    }
  } else {
    // Process through synchronous compilation.
    ObjBuf = CompilerSync::instance().compile(CompilationTask{
        KernelBitcode, HashValue, KernelInfo.getName(), Suffix, BlockDim,
        GridDim, RCVec, LambdaSpecializations, BinInfo.getVarNameToGlobalInfo(),
        GlobalLinkedBinaries, DeviceArch,
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
  if (CacheChain)
    CacheChain->store(HashValue,
                      CacheEntry::staticObject(ObjBuf->getMemBufferRef()));

  return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                              ShmemSize, Stream);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerFatBinary(void *Handle,
                                               FatbinWrapperT *FatbinWrapper,
                                               const char *ModuleId) {
  TIMESCOPE(JitEngineDevice, registerFatBinary);
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

template <typename ImplT> void JitEngineDevice<ImplT>::finalizeRegistration() {
  TIMESCOPE(JitEngineDevice, finalizeRegistration);
  PROTEUS_DBG(Logger::logs("proteus") << "Finalize registration\n");
  // Erase linked binaries for which we have LLVM IR code, those binaries are
  // stored in the ModuleIdToFatBinary map.
  for (auto &[ModuleId, FatbinWrapper] : ModuleIdToFatBinary)
    GlobalLinkedBinaries.erase((void *)FatbinWrapper->Binary);
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
void JitEngineDevice<ImplT>::registerLinkedBinary(void *Handle,
                                                  FatbinWrapperT *FatbinWrapper,
                                                  const char *ModuleId) {
  TIMESCOPE(JitEngineDevice, registerLinkedBinary);
  PROTEUS_DBG(Logger::logs("proteus")
              << "Register linked binary FatBinary " << FatbinWrapper
              << " Binary " << (void *)FatbinWrapper->Binary << " ModuleId "
              << ModuleId << "\n");
  if (!HandleToBinaryInfo.count(Handle))
    reportFatalError("Expected Handle in map");

  HandleToBinaryInfo[Handle].addModuleId(ModuleId);
  ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
}

} // namespace proteus

#endif
