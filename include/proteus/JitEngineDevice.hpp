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
#include "llvm/Config/llvm-config.h"
#include <filesystem>
#include <functional>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/Support/MemoryBuffer.h>

#include "llvm/Linker/Linker.h"
#include "llvm/Object/ELFObjectFile.h"
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <memory>

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <optional>
#include <string>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Hashing.hpp"
#include "proteus/JitCache.hpp"
#include "proteus/JitEngine.hpp"
#include "proteus/JitStorageCache.hpp"
#include "proteus/TimeTracing.hpp"
#include "proteus/TransformArgumentSpecialization.hpp"
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
  SmallVector<std::string> LinkedModuleIds;
  std::unique_ptr<Module> ExtractedModule;
  std::optional<HashT> ExtractedModuleHash;

public:
  BinaryInfo() = default;
  BinaryInfo(FatbinWrapperT *FatbinWrapper,
             SmallVector<std::string> &&LinkedModuleIds)
      : FatbinWrapper(FatbinWrapper), LinkedModuleIds(LinkedModuleIds) {}

  FatbinWrapperT *getFatbinWrapper() const { return FatbinWrapper; }

  bool hasModule() const { return (ExtractedModule != nullptr); }
  Module &getModule() const { return *ExtractedModule; }
  void setModule(std::unique_ptr<Module> Module) {
    ExtractedModule = std::move(Module);
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

  void addModuleId(const char *ModuleId) {
    LinkedModuleIds.push_back(ModuleId);
  }

  auto &getModuleIds() { return LinkedModuleIds; }
};

class JITKernelInfo {
  std::string Name;
  SmallVector<int32_t> RCTypes;
  SmallVector<int32_t> RCIndices;
  int32_t NumRCs;
  std::optional<std::reference_wrapper<Module>> ExtractedModule;
  std::optional<std::reference_wrapper<BinaryInfo>> BinInfo;
  std::optional<HashT> StaticHash;

public:
  JITKernelInfo(BinaryInfo &BinInfo, char const *Name, int32_t *RCIndices,
                int32_t *RCTypes, int32_t NumRCs)
      : BinInfo(BinInfo), Name(Name),
        RCIndices{ArrayRef{RCIndices, static_cast<size_t>(NumRCs)}},
        RCTypes{ArrayRef{RCTypes, static_cast<size_t>(NumRCs)}}, NumRCs(NumRCs),
        ExtractedModule(std::nullopt) {}

  JITKernelInfo() = default;
  const std::string &getName() const { return Name; }
  const auto &getRCIndices() const { return RCIndices; }
  const auto &getRCTypes() const { return RCTypes; }
  const auto &getNumRCs() const { return NumRCs; }
  const bool hasModule() const { return ExtractedModule.has_value(); }
  Module &getModule() const { return ExtractedModule->get(); }
  BinaryInfo &getBinaryInfo() const { return BinInfo.value(); }
  void setModule(llvm::Module &Mod) { ExtractedModule = Mod; }
  const bool hasStaticHash() const { return StaticHash.has_value(); }
  const HashT getStaticHash() const { return StaticHash.value(); }
  void createStaticHash(HashT ModuleHash) {
    StaticHash = hash(Name);
    StaticHash = hashCombine(StaticHash.value(), ModuleHash);
  }
};

template <typename ImplT> struct DeviceTraits;

template <typename ImplT> class JitEngineDevice : public JitEngine {

private:
  // LLVMContext needs to destroy after all associated Module objects have been
  // destroyed. Declared first to destroy last.
  LLVMContext Ctx;

public:
  using DeviceError_t = typename DeviceTraits<ImplT>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<ImplT>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<ImplT>::KernelFunction_t;

  DeviceError_t
  compileAndRun(JITKernelInfo &KernelInfo, dim3 GridDim, dim3 BlockDim,
                void **KernelArgs, uint64_t ShmemSize,
                typename DeviceTraits<ImplT>::DeviceStream_t Stream);

  Module &getModule(JITKernelInfo &KernelInfo) {
    TIMESCOPE(__FUNCTION__)

    if (KernelInfo.hasModule())
      return KernelInfo.getModule();

    BinaryInfo &BinInfo = KernelInfo.getBinaryInfo();

    if (BinInfo.hasModule()) {
      KernelInfo.setModule(BinInfo.getModule());
      return KernelInfo.getModule();
    }

    std::unique_ptr<Module> ExtractedModule =
        static_cast<ImplT &>(*this).extractModule(BinInfo);

    BinInfo.setModule(std::move(ExtractedModule));
    KernelInfo.setModule(BinInfo.getModule());
    return KernelInfo.getModule();
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
  DenseMap<const void *, BinaryInfo> HandleToBinaryInfo;
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
    auto ReplaceIntrinsicDim = [&](ArrayRef<StringRef> IntrinsicNames,
                                   uint32_t DimValue) {
      auto CollectCallUsers = [](Function &F) {
        SmallVector<CallInst *> CallUsers;
        for (auto *User : F.users()) {
          auto *Call = dyn_cast<CallInst>(User);
          if (!Call)
            continue;
          CallUsers.push_back(Call);
        }

        return CallUsers;
      };

      for (auto IntrinsicName : IntrinsicNames) {

        Function *IntrinsicFunction = M.getFunction(IntrinsicName);
        if (!IntrinsicFunction)
          continue;

        for (auto *Call : CollectCallUsers(*IntrinsicFunction)) {
          Value *ConstantValue =
              ConstantInt::get(Type::getInt32Ty(M.getContext()), DimValue);
          Call->replaceAllUsesWith(ConstantValue);
          Call->eraseFromParent();
        }
      }
    };

    ReplaceIntrinsicDim(ImplT::gridDimXFnName(), GridDim.x);
    ReplaceIntrinsicDim(ImplT::gridDimYFnName(), GridDim.y);
    ReplaceIntrinsicDim(ImplT::gridDimZFnName(), GridDim.z);

    ReplaceIntrinsicDim(ImplT::blockDimXFnName(), BlockDim.x);
    ReplaceIntrinsicDim(ImplT::blockDimYFnName(), BlockDim.y);
    ReplaceIntrinsicDim(ImplT::blockDimZFnName(), BlockDim.z);

    auto InsertAssume = [&](ArrayRef<StringRef> IntrinsicNames, int DimValue) {
      for (auto IntrinsicName : IntrinsicNames) {
        Function *IntrinsicFunction = M.getFunction(IntrinsicName);
        if (!IntrinsicFunction || IntrinsicFunction->use_empty())
          continue;

        // Iterate over all uses of the intrinsic.
        for (auto U : IntrinsicFunction->users()) {
          auto *Call = dyn_cast<CallInst>(U);
          if (!Call)
            continue;

          // Insert the llvm.assume intrinsic.
          IRBuilder<> Builder(Call->getNextNode());
          Value *Bound = ConstantInt::get(Call->getType(), DimValue);
          Value *Cmp = Builder.CreateICmpULT(Call, Bound);

          Function *AssumeIntrinsic =
              Intrinsic::getDeclaration(&M, Intrinsic::assume);
          Builder.CreateCall(AssumeIntrinsic, Cmp);
        }
      }
    };

    // Inform LLVM about the range of possible values of threadIdx.*.
    InsertAssume(ImplT::threadIdxXFnName(), BlockDim.x);
    InsertAssume(ImplT::threadIdxYFnName(), BlockDim.y);
    InsertAssume(ImplT::threadIdxZFnName(), BlockDim.z);

    // Inform LLVM about the range of possible values of blockIdx.*.
    InsertAssume(ImplT::blockIdxXFnName(), GridDim.x);
    InsertAssume(ImplT::blockIdxYFnName(), GridDim.y);
    InsertAssume(ImplT::blockIdxZFnName(), GridDim.z);
  }

  void getRuntimeConstantValues(void **KernelArgs,
                                const SmallVector<int32_t> &RCIndices,
                                const SmallVector<int32_t> &RCTypes,
                                SmallVector<RuntimeConstant> &RCsVec) {
    TIMESCOPE(__FUNCTION__);
    for (int I = 0; I < RCIndices.size(); ++I) {
      PROTEUS_DBG(Logger::logs("proteus") << "RC Index " << RCIndices[I]
                                          << " Type " << RCTypes[I] << "\n");
      RuntimeConstant RC;
      switch (RCTypes[I]) {
      case RuntimeConstantTypes::BOOL:
        RC.Value.BoolVal = *(bool *)KernelArgs[RCIndices[I]];
        break;
      case RuntimeConstantTypes::INT8:
        RC.Value.Int8Val = *(int8_t *)KernelArgs[RCIndices[I]];
        break;
      case RuntimeConstantTypes::INT32:
        RC.Value.Int32Val = *(int32_t *)KernelArgs[RCIndices[I]];
        break;
      case RuntimeConstantTypes::INT64:
        RC.Value.Int64Val = *(int64_t *)KernelArgs[RCIndices[I]];
        break;
      case RuntimeConstantTypes::FLOAT:
        RC.Value.FloatVal = *(float *)KernelArgs[RCIndices[I]];
        break;
      case RuntimeConstantTypes::DOUBLE:
        RC.Value.DoubleVal = *(double *)KernelArgs[RCIndices[I]];
        break;
      // NOTE: long double on device should correspond to plain double.
      // XXX: CUDA with a long double SILENTLY fails to create a working
      // kernel in AOT compilation, with or without JIT.
      case RuntimeConstantTypes::LONG_DOUBLE:
        RC.Value.LongDoubleVal = *(long double *)KernelArgs[RCIndices[I]];
        break;
      case RuntimeConstantTypes::PTR:
        RC.Value.PtrVal = (void *)KernelArgs[RCIndices[I]];
        break;
      default:
        FATAL_ERROR("JIT Incompatible type in runtime constant: " +
                    std::to_string(RCTypes[I]));
      }

      RCsVec.push_back(RC);
    }
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

    Expected<object::ELF64LEObjectFile> DeviceElfOrErr =
        object::ELF64LEObjectFile::create(Object);
    if (DeviceElfOrErr.takeError())
      FATAL_ERROR("Cannot create the device elf");
    auto &DeviceElf = *DeviceElfOrErr;

    for (auto &[GlobalName, HostAddr] : VarNameToDevPtr) {
      for (auto &Symbol : DeviceElf.symbols()) {
        auto SymbolNameOrErr = Symbol.getName();
        if (!SymbolNameOrErr)
          continue;
        auto SymbolName = *SymbolNameOrErr;

        if (!SymbolName.equals(GlobalName + "$ptr"))
          continue;

        Expected<uint64_t> ValueOrErr = Symbol.getValue();
        if (!ValueOrErr)
          FATAL_ERROR("Expected symbol value");
        uint64_t SymbolValue = *ValueOrErr;

        // Get the section containing the symbol
        auto SectionOrErr = Symbol.getSection();
        if (!SectionOrErr)
          FATAL_ERROR("Cannot retrieve section");
        const auto &Section = *SectionOrErr;
        if (Section == DeviceElf.section_end())
          FATAL_ERROR("Expected sybmol in section");

        // Get the section's address and data
        Expected<StringRef> SectionDataOrErr = Section->getContents();
        if (!SectionDataOrErr)
          FATAL_ERROR("Error retrieving section data");
        StringRef SectionData = *SectionDataOrErr;

        // Calculate offset within the section
        uint64_t SectionAddr = Section->getAddress();
        uint64_t Offset = SymbolValue - SectionAddr;
        if (Offset >= SectionData.size())
          FATAL_ERROR("Expected offset within section size");

        uint64_t *Data = (uint64_t *)(SectionData.data() + Offset);
        *Data = reinterpret_cast<uint64_t>(resolveDeviceGlobalAddr(HostAddr));
        break;
      }
    }
  }

  std::unique_ptr<MemoryBuffer> codegenObject(Module &M, StringRef DeviceArch) {
    return static_cast<ImplT &>(*this).codegenObject(M, DeviceArch);
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

  void pruneIR(Module &M, StringRef FnName);

  void specializeIR(Module &M, StringRef FnName, StringRef Suffix,
                    dim3 &BlockDim, dim3 &GridDim,
                    const SmallVector<int32_t> &RCIndices, RuntimeConstant *RC,
                    int NumRuntimeConstants);

  void replaceGlobalVariablesWithPointers(
      Module &M,
      std::unordered_map<std::string, const void *> &VarNameToDevPtr);

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
  std::unique_ptr<Module>
  linkJitModule(SmallVector<std::unique_ptr<Module>> &LinkedModules);

  LLVMContext &getLLVMContext() { return Ctx; }

protected:
  DenseMap<const void *, JITKernelInfo> JITKernelInfoMap;
};

template <typename ImplT>
void JitEngineDevice<ImplT>::specializeIR(Module &M, StringRef FnName,
                                          StringRef Suffix, dim3 &BlockDim,
                                          dim3 &GridDim,
                                          const SmallVector<int32_t> &RCIndices,
                                          RuntimeConstant *RC,
                                          int NumRuntimeConstants) {
  TIMESCOPE("specializeIR");
  Function *F = M.getFunction(FnName);

  assert(F && "Expected non-null function!");
  // Replace argument uses with runtime constants.
  if (Config.ENV_PROTEUS_SPECIALIZE_ARGS)
    // TODO: change NumRuntimeConstants to size_t at interface.
    TransformArgumentSpecialization::transform(
        M, *F, RCIndices,
        ArrayRef<RuntimeConstant>{RC,
                                  static_cast<size_t>(NumRuntimeConstants)});

  // Replace uses of blockDim.* and gridDim.* with constants.
  if (Config.ENV_PROTEUS_SPECIALIZE_DIMS) {
    setKernelDims(M, GridDim, BlockDim);
  }

  // Internalize others besides the kernel function.
  internalizeModule(M, [&F](const GlobalValue &GV) {
    // Do not internalize the kernel function.
    if (&GV == F)
      return true;

    // Internalize everything else.
    return false;
  });

  PROTEUS_DBG(Logger::logs("proteus") << "=== JIT Module\n"
                                      << M << "=== End of JIT Module\n");

  F->setName(FnName + Suffix);

  if (Config.ENV_PROTEUS_SET_LAUNCH_BOUNDS)
    setLaunchBoundsForKernel(M, *F, GridDim.x * GridDim.y * GridDim.z,
                             BlockDim.x * BlockDim.y * BlockDim.z);

  runCleanupPassPipeline(M);

#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "=== Final Module\n"
                          << M << "=== End Final Module\n";
  if (verifyModule(M, &errs()))
    FATAL_ERROR("Broken module found, JIT compilation aborted!");
  else
    Logger::logs("proteus") << "Module verified!\n";
#endif
}

template <typename ImplT>
void JitEngineDevice<ImplT>::pruneIR(Module &M, StringRef FnName) {
  TIMESCOPE("pruneIR");
  PROTEUS_DBG(Logger::logs("proteus") << "=== Parsed Module\n"
                                      << M << "=== End of Parsed Module\n");
  // Remove llvm.global.annotations now that we have read them.
  if (auto *GlobalAnnotations = M.getGlobalVariable("llvm.global.annotations"))
    M.eraseGlobalVariable(GlobalAnnotations);

  // Remove llvm.compiler.used
  if (auto *CompilerUsed = M.getGlobalVariable("llvm.compiler.used"))
    M.eraseGlobalVariable(CompilerUsed);

  // Remove the __clang_gpu_used_external used in HIP RDC compilation and its
  // uses in llvm.used, llvm.compiler.used.
  SmallVector<GlobalVariable *> GlobalsToErase;
  for (auto &GV : M.globals()) {
    auto Name = GV.getName();
    if (Name.starts_with("__clang_gpu_used_external") ||
        Name.starts_with("_jit_bitcode") || Name.starts_with("__hip_cuid")) {
      GlobalsToErase.push_back(&GV);
      removeFromUsedLists(M, [&GV](Constant *C) {
        if (auto *Global = dyn_cast<GlobalVariable>(C))
          return Global == &GV;
        return false;
      });
    }
  }
  for (auto GV : GlobalsToErase) {
    M.eraseGlobalVariable(GV);
  }

  // Remove externaly_initialized attributes.
  for (auto &GV : M.globals())
    if (GV.isExternallyInitialized())
      GV.setExternallyInitialized(false);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::replaceGlobalVariablesWithPointers(
    Module &M, std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
  TIMESCOPE(__FUNCTION__)
  // Re-link globals to fixed addresses provided by registered
  // variables.
  for (auto RegisterVar : VarNameToDevPtr) {
    auto &VarName = RegisterVar.first;
    auto *GV = M.getNamedGlobal(VarName);
    // Skip linking if the GV does not exist in the module.
    if (!GV)
      continue;

    // This will convert constant users of GV to instructions so that we can
    // replace with the GV ptr.
    convertUsersOfConstantsToInstructions({GV});

    Constant *Addr =
        ConstantInt::get(Type::getInt64Ty(M.getContext()), 0xDEADBEEFDEADBEEF);
    auto *CE = ConstantExpr::getIntToPtr(Addr, GV->getType()->getPointerTo());
    auto *GVarPtr = new GlobalVariable(
        M, GV->getType()->getPointerTo(), false, GlobalValue::ExternalLinkage,
        CE, GV->getName() + "$ptr", nullptr, GV->getThreadLocalMode(),
        GV->getAddressSpace(), true);

    SmallVector<Instruction *> ToReplace;
    for (auto *User : GV->users()) {
      auto *Inst = dyn_cast<Instruction>(User);
      if (!Inst)
        FATAL_ERROR("Expected Instruction User for GV");

      ToReplace.push_back(Inst);
    }

    for (auto *Inst : ToReplace) {
      IRBuilder Builder{Inst};
      auto *Load = Builder.CreateLoad(GV->getType(), GVarPtr);
      Inst->replaceUsesOfWith(GV, Load);
    }
  }

#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "=== Linked M\n" << M << "=== End of Linked M\n";
  if (verifyModule(M, &errs()))
    FATAL_ERROR("After linking, broken module found, JIT compilation aborted!");
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

  SmallVector<RuntimeConstant> RCsVec;

  getRuntimeConstantValues(KernelArgs, KernelInfo.getRCIndices(),
                           KernelInfo.getRCTypes(), RCsVec);

  HashT HashValue =
      hash(getStaticHash(KernelInfo), RCsVec, GridDim.x, GridDim.y, GridDim.z,
           BlockDim.x, BlockDim.y, BlockDim.z);

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

  if (Config.ENV_PROTEUS_USE_STORED_CACHE) {
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
      if (!Config.ENV_PROTEUS_RELINK_GLOBALS_BY_COPY)
        relinkGlobalsObject(CacheBuf->getMemBufferRef());

      auto KernelFunc =
          getKernelFunctionFromImage(KernelMangled, CacheBuf->getBufferStart());

      CodeCache.insert(HashValue, KernelFunc, KernelInfo.getName(),
                       RCsVec.data(), KernelInfo.getNumRCs());

      return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                  ShmemSize, Stream);
    }
  }

  // We need to clone, as getModule returns a generic LLVM IR to be
  // used by any kernel that will be specialized
  auto JitModule = llvm::CloneModule(getModule(KernelInfo));
  // NOTE: There is potential oportunity here, to reduce some of the JIT costs
  // further. We can have a pruneIR in which we do not do any RC/Grid/Block
  // specializations. We only internalize symbols. Then we can use that IR
  // for all upcoming specializations of dynamic information.
  // There is a memory trade off in such case, We will need to have a peristent
  // in memory module, for every annotated kernel. If we have a case of 1000s of
  // kernels, this can be an issue

  pruneIR(*JitModule, KernelInfo.getName());

  specializeIR(*JitModule, KernelInfo.getName(), Suffix, BlockDim, GridDim,
               KernelInfo.getRCIndices(), RCsVec.data(),
               KernelInfo.getNumRCs());

  replaceGlobalVariablesWithPointers(*JitModule, VarNameToDevPtr);

  // For HIP RTC codegen do not run the optimization pipeline since HIP RTC
  // internally runs it. For the rest of cases, that is CUDA or HIP with our own
  // codegen instead of RTC, run the target-specific optimization pipeline to
  // optimize the LLVM IR before handing over to codegen.
#if PROTEUS_ENABLE_CUDA
  optimizeIR(*JitModule, DeviceArch);
#elif PROTEUS_ENABLE_HIP
  if (!Config.ENV_PROTEUS_USE_HIP_RTC_CODEGEN)
    optimizeIR(*JitModule, DeviceArch);
#else
#error "JitEngineDevice requires PROTEUS_ENABLE_CUDA or PROTEUS_ENABLE_HIP"
#endif

  if (Config.ENV_PROTEUS_DUMP_LLVM_IR) {
    const auto CreateDumpDirectory = []() {
      const std::string DumpDirectory = ".proteus-dump";
      std::filesystem::create_directory(DumpDirectory);
      return DumpDirectory;
    };

    static const std::string DumpDirectory = CreateDumpDirectory();

    saveToFile(DumpDirectory + "/device-jit-" + HashValue.toString() + ".ll",
               *JitModule);
  }

  auto ObjBuf = codegenObject(*JitModule, DeviceArch);
  if (Config.ENV_PROTEUS_USE_STORED_CACHE) {
    StorageCache.store(HashValue, ObjBuf->getMemBufferRef());
  }
  if (!Config.ENV_PROTEUS_RELINK_GLOBALS_BY_COPY)
    relinkGlobalsObject(ObjBuf->getMemBufferRef());

  KernelFunc =
      getKernelFunctionFromImage(KernelMangled, ObjBuf->getBufferStart());

  CodeCache.insert(HashValue, KernelFunc, KernelInfo.getName(), RCsVec.data(),
                   KernelInfo.getNumRCs());

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
    HandleToBinaryInfo[Handle] = {FatbinWrapper, {}};

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
    HandleToBinaryInfo[Handle] = {FatbinWrapper, {ModuleId}};
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

  if (!HandleToBinaryInfo.contains(Handle))
    FATAL_ERROR("Expected Handle in map");
  BinaryInfo &BinInfo = HandleToBinaryInfo[Handle];

  JITKernelInfoMap[Kernel] =
      JITKernelInfo{BinInfo, KernelName, RCIndices, RCTypes, NumRCs};
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerLinkedBinary(FatbinWrapperT *FatbinWrapper,
                                                  const char *ModuleId) {
  PROTEUS_DBG(Logger::logs("proteus")
              << "Register linked binary FatBinary " << FatbinWrapper
              << " Binary " << (void *)FatbinWrapper->Binary << " ModuleId "
              << ModuleId << "\n");
  if (CurHandle) {
    if (!HandleToBinaryInfo.contains(CurHandle))
      FATAL_ERROR("Expected CurHandle in map");

    HandleToBinaryInfo[CurHandle].addModuleId(ModuleId);
  } else
    GlobalLinkedModuleIds.push_back(ModuleId);

  ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
}

template <typename ImplT>
std::unique_ptr<Module> JitEngineDevice<ImplT>::linkJitModule(
    SmallVector<std::unique_ptr<Module>> &LinkedModules) {
  if (LinkedModules.empty())
    FATAL_ERROR("Expected jit module");

  auto LinkedModule =
      std::make_unique<llvm::Module>("JitModule", getLLVMContext());
  Linker IRLinker(*LinkedModule);
  for (auto &LinkedM : LinkedModules) {
    // Returns true if linking failed.
    if (IRLinker.linkInModule(std::move(LinkedM)))
      FATAL_ERROR("Linking failed");
  }

  return LinkedModule;
}

} // namespace proteus

#endif
