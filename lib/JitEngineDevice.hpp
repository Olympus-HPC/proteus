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
#if LLVM_VERSION_MAJOR == 18
#include "llvm/ADT/StableHashing.h"
#else
#include "llvm/CodeGen/StableHashing.h"
#endif

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

#include "CompilerInterfaceTypes.h"
#include "JitCache.hpp"
#include "JitEngine.hpp"
#include "JitStorageCache.hpp"
#include "TimeTracing.hpp"
#include "TransformArgumentSpecialization.hpp"
#include "Utils.h"

namespace proteus {

using namespace llvm;

class JITKernelInfo {
  char const *Name;
  SmallVector<int32_t> RCTypes;
  SmallVector<int32_t> RCIndices;
  int32_t NumRCs;
  std::optional<std::reference_wrapper<Module>> LinkedIR;

public:
  JITKernelInfo(char const *Name, int32_t *RCIndices, int32_t *RCTypes,
                int32_t NumRCs)
      : Name(Name), RCIndices{ArrayRef{RCIndices, static_cast<size_t>(NumRCs)}},
        RCTypes{ArrayRef{RCTypes, static_cast<size_t>(NumRCs)}}, NumRCs(NumRCs),
        LinkedIR(std::nullopt) {}

  JITKernelInfo() : Name(nullptr), NumRCs(0), RCIndices(), RCTypes() {}
  const auto &getName() const { return Name; }
  const auto &getRCIndices() const { return RCIndices; }
  const auto &getRCTypes() const { return RCTypes; }
  const auto &getNumRCs() const { return NumRCs; }
  const bool hasLinkedIR() const { return LinkedIR.has_value(); }
  Module &getLinkedModule() const { return LinkedIR->get(); }
  void setLinkedModule(llvm::Module &Mod) { LinkedIR = Mod; }
};

struct FatbinWrapper_t {
  int32_t Magic;
  int32_t Version;
  const char *Binary;
  void **PrelinkedFatbins;
};

template <typename ImplT> struct DeviceTraits;

template <typename ImplT> class JitEngineDevice : protected JitEngine {
public:
  using DeviceError_t = typename DeviceTraits<ImplT>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<ImplT>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<ImplT>::KernelFunction_t;

  DeviceError_t
  compileAndRun(StringRef ModuleUniqueId, void *Kernel, StringRef KernelName,
                const SmallVector<int32_t> &RCIndices,
                const SmallVector<int32_t> &RCTypes, int NumRuntimeConstants,
                dim3 GridDim, dim3 BlockDim, void **KernelArgs,
                uint64_t ShmemSize,
                typename DeviceTraits<ImplT>::DeviceStream_t Stream);

  void insertRegisterVar(const char *VarName, const void *Addr) {
    VarNameToDevPtr[VarName] = Addr;
  }
  void registerLinkedBinary(FatbinWrapper_t *FatbinWrapper,
                            const char *ModuleId);
  void registerFatBinary(void *Handle, FatbinWrapper_t *FatbinWrapper,
                         const char *ModuleId);
  void registerFatBinaryEnd();
  void registerFunction(void *Handle, void *Kernel, char *KernelName,
                        int32_t *RCIndices, int32_t *RCTypes, int32_t NumRCs);

  struct BinaryInfo {
    FatbinWrapper_t *FatbinWrapper;
    SmallVector<std::string> LinkedModuleIds;
  };

  void *CurHandle = nullptr;
  std::unordered_map<std::string, FatbinWrapper_t *> ModuleIdToFatBinary;
  DenseMap<void *, BinaryInfo> HandleToBinaryInfo;
  DenseMap<void *, void *> KernelToHandleMap;
  SmallVector<std::string> GlobalLinkedModuleIds;
  SmallPtrSet<void *, 8> GlobalLinkedBinaries;

  bool containsJITKernelInfo(const void *Func) {
    return JITKernelInfoMap.contains(Func);
  }

  std::optional<JITKernelInfo> getJITKernelInfo(const void *Func) {
    if (!containsJITKernelInfo(Func)) {
      return std::nullopt;
    }
    return JITKernelInfoMap[Func];
  }

  void addLinkedModule(std::unique_ptr<Module> Mod) {
    LinkedIRModules.emplace_back(std::move(Mod));
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
    auto ReplaceIntrinsicDim = [&](StringRef IntrinsicName, uint32_t DimValue) {
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
      Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction)
        return;

      for (auto *Call : CollectCallUsers(*IntrinsicFunction)) {
        Value *ConstantValue =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), DimValue);
        Call->replaceAllUsesWith(ConstantValue);
        Call->eraseFromParent();
      }
    };

    ReplaceIntrinsicDim(ImplT::gridDimXFnName(), GridDim.x);
    ReplaceIntrinsicDim(ImplT::gridDimYFnName(), GridDim.y);
    ReplaceIntrinsicDim(ImplT::gridDimZFnName(), GridDim.z);

    ReplaceIntrinsicDim(ImplT::blockDimXFnName(), BlockDim.x);
    ReplaceIntrinsicDim(ImplT::blockDimYFnName(), BlockDim.y);
    ReplaceIntrinsicDim(ImplT::blockDimZFnName(), BlockDim.z);

    auto InsertAssume = [&](StringRef IntrinsicName, int DimValue) {
      Function *IntrinsicFunction = M.getFunction(IntrinsicName);
      if (!IntrinsicFunction || IntrinsicFunction->use_empty())
        return;

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
    for (int I = 0; I < RCIndices.size(); ++I) {
      DBG(Logger::logs("proteus")
          << "RC Index " << RCIndices[I] << " Type " << RCTypes[I] << "\n");
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

  DeviceError_t launchKernelDirect(void *KernelFunc, dim3 GridDim,
                                   dim3 BlockDim, void **KernelArgs,
                                   uint64_t ShmemSize, DeviceStream_t Stream) {
    return static_cast<ImplT &>(*this).launchKernelDirect(
        KernelFunc, GridDim, BlockDim, KernelArgs, ShmemSize, Stream);
  }

  std::unique_ptr<MemoryBuffer> codegenObject(Module &M, StringRef DeviceArch) {
    return static_cast<ImplT &>(*this).codegenObject(M, DeviceArch);
  }

  KernelFunction_t getKernelFunctionFromImage(StringRef KernelName,
                                              const void *Image) {
    return static_cast<ImplT &>(*this).getKernelFunctionFromImage(KernelName,
                                                                  Image);
  }

  Module &extractDeviceBitcode(StringRef KernelName, void *Kernel) {
    TIMESCOPE(__FUNCTION__)
    return static_cast<ImplT &>(*this).extractDeviceBitcode(KernelName, Kernel);
  }
  //------------------------------------------------------------------
  // End Methods implemented in the derived device engine class.
  //------------------------------------------------------------------

  void specializeIR(Module &M, StringRef FnName, StringRef Suffix,
                    dim3 &BlockDim, dim3 &GridDim,
                    const SmallVector<int32_t> &RCIndices, RuntimeConstant *RC,
                    int NumRuntimeConstants);

  void
  relinkGlobals(Module &M,
                std::unordered_map<std::string, const void *> &VarNameToDevPtr);

  static stable_hash computeDeviceFatBinHash() {
    TIMESCOPE("computeDeviceFatBinHash");
    using namespace llvm::object;
    stable_hash L1Hash{0};
    auto ExePath = std::filesystem::canonical("/proc/self/exe");

    DBG(Logger::logs("proteus")
        << "Reading file from path " << ExePath.string() << "\n");

    auto bufferOrErr = MemoryBuffer::getFile(ExePath.string());
    if (!bufferOrErr) {
      FATAL_ERROR("Failed to open binary file");
    }

    auto objOrErr =
        ObjectFile::createELFObjectFile(bufferOrErr.get()->getMemBufferRef());
    if (!objOrErr) {
      FATAL_ERROR("Failed to create Object File");
    }

    ObjectFile &elfObj = **objOrErr;

    for (const SectionRef &section : elfObj.sections()) {
      auto nameOrErr = section.getName();
      if (!nameOrErr)
        FATAL_ERROR("Error getting section name: ");

      StringRef sectionName = nameOrErr.get();

      if (!ImplT::isHashedSection(sectionName))
        continue;

      DBG(Logger::logs("proteus")
          << "Hashing section " << sectionName.str() << "\n");

      auto contentsOrErr = section.getContents();
      if (!contentsOrErr) {
        FATAL_ERROR("Error getting section contents: ");
        continue;
      }
      StringRef sectionContents = contentsOrErr.get();
      auto sectionHash = stable_hash_combine_string(sectionContents);
      L1Hash = stable_hash_combine(sectionHash, L1Hash);
    }
    return L1Hash;
  }

protected:
  JitEngineDevice() {
    L1Hash = computeDeviceFatBinHash();
    Ctx = std::make_unique<LLVMContext>();
    DBG(Logger::logs("proteus") << "L1-Hash is " << L1Hash << "\n");
  }

  ~JitEngineDevice() {
    LinkedIRModules.clear();
    CodeCache.printStats();
    StorageCache.printStats();
    Ctx.reset();
  }

  JitCache<KernelFunction_t> CodeCache;
  JitStorageCache<KernelFunction_t> StorageCache;
  std::string DeviceArch;
  std::unordered_map<std::string, const void *> VarNameToDevPtr;
  std::unique_ptr<Module>
  linkJitModule(StringRef KernelName,
                SmallVector<std::unique_ptr<Module>> &LinkedModules);

  LLVMContext &getProteusLLVMCtx() const { return *Ctx.get(); }

  const stable_hash getL1Hash() const { return L1Hash; }

protected:
  DenseMap<const void *, JITKernelInfo> JITKernelInfoMap;

private:
  // All the LLVM Modules that have been loaded and linked;
  SmallVector<std::unique_ptr<Module>> LinkedIRModules;
  stable_hash L1Hash;
  std::unique_ptr<LLVMContext> Ctx;
};

template <typename ImplT>
void JitEngineDevice<ImplT>::specializeIR(Module &M, StringRef FnName,
                                          StringRef Suffix, dim3 &BlockDim,
                                          dim3 &GridDim,
                                          const SmallVector<int32_t> &RCIndices,
                                          RuntimeConstant *RC,
                                          int NumRuntimeConstants) {

  TIMESCOPE("specializeIR");
  DBG(Logger::logs("proteus") << "=== Parsed Module\n"
                              << M << "=== End of Parsed Module\n");
  Function *F = M.getFunction(FnName);
  assert(F && "Expected non-null function!");

  // Remove llvm.global.annotations now that we have read them.
  if (auto *GlobalAnnotations = M.getGlobalVariable("llvm.global.annotations"))
    M.eraseGlobalVariable(GlobalAnnotations);

  // Remove the __clang_gpu_used_external used in HIP RDC compilation and its
  // uses in llvm.used, llvm.compiler.used.
  SmallVector<GlobalVariable *> GlobalsToErase;
  for (auto &GV : M.globals()) {
    auto Name = GV.getName();
    if (Name.starts_with("__clang_gpu_used_external") ||
        Name.starts_with("_jit_bitcode") ||
        Name.starts_with("__hip_cuid")) {
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

  DBG(Logger::logs("proteus") << "=== JIT Module\n"
                              << M << "=== End of JIT Module\n");

  F->setName(FnName + Suffix);

  if (Config.ENV_PROTEUS_SET_LAUNCH_BOUNDS)
    setLaunchBoundsForKernel(M, *F, GridDim.x * GridDim.y * GridDim.z,
                             BlockDim.x * BlockDim.y * BlockDim.z);

#if ENABLE_DEBUG
  Logger::logs("proteus") << "=== Final Module\n"
                          << M << "=== End Final Module\n";
  if (verifyModule(M, &errs()))
    FATAL_ERROR("Broken module found, JIT compilation aborted!");
  else
    Logger::logs("proteus") << "Module verified!\n";
#endif
}

template <typename ImplT>
void JitEngineDevice<ImplT>::relinkGlobals(
    Module &M, std::unordered_map<std::string, const void *> &VarNameToDevPtr) {
  // Re-link globals to fixed addresses provided by registered
  // variables.
  for (auto RegisterVar : VarNameToDevPtr) {
    // For CUDA we must defer resolving the global symbol address here
    // instead when registering the variable in the constructor context.
    void *DevPtr = resolveDeviceGlobalAddr(RegisterVar.second);
    auto &VarName = RegisterVar.first;
    auto *GV = M.getNamedGlobal(VarName);
    // Skip linking if the GV does not exist in the module.
    if (!GV)
      continue;
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
    StringRef ModuleUniqueId, void *Kernel, StringRef KernelName,
    const SmallVector<int32_t> &RCIndices, const SmallVector<int32_t> &RCTypes,
    int NumRuntimeConstants, dim3 GridDim, dim3 BlockDim, void **KernelArgs,
    uint64_t ShmemSize, typename DeviceTraits<ImplT>::DeviceStream_t Stream) {
  TIMESCOPE("compileAndRun");

  SmallVector<RuntimeConstant> RCsVec;

  getRuntimeConstantValues(KernelArgs, RCIndices, RCTypes, RCsVec);
  auto L1Hash = getL1Hash();

  uint64_t HashValue = CodeCache.hash(
      L1Hash, ModuleUniqueId, KernelName, RCsVec.data(), NumRuntimeConstants,
      GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z);

  typename DeviceTraits<ImplT>::KernelFunction_t KernelFunc =
      CodeCache.lookup(HashValue);
  if (KernelFunc)
    return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                ShmemSize, Stream);

  // NOTE: we don't need a suffix to differentiate kernels, each specialization
  // will be in its own module uniquely identify by HashValue. It exists only
  // for debugging purposes to verify that the jitted kernel executes.
  std::string Suffix = mangleSuffix(HashValue);
  std::string KernelMangled = (KernelName + Suffix).str();

  if (Config.ENV_PROTEUS_USE_STORED_CACHE) {
    // If there device global variables, lookup the IR and codegen object
    // before launching. Else, if there aren't device global variables, lookup
    // the object and launch.

    // TODO: Check for globals is very conservative and always re-builds from
    // LLVM IR even if the Jit module does not use global variables.  A better
    // solution is to keep track of whether a kernel uses gvars (store a flag in
    // the cache file?) and load the object in case it does not use any.
    // TODO: Can we use RTC interfaces for fast linking on object files?
    bool HasDeviceGlobals = !VarNameToDevPtr.empty();
    if (auto CacheBuf =
            (HasDeviceGlobals
                 ? StorageCache.lookupBitcode(HashValue, KernelMangled)
                 : StorageCache.lookupObject(HashValue, KernelMangled))) {
      std::unique_ptr<MemoryBuffer> ObjBuf;
      if (HasDeviceGlobals) {
        auto Ctx = std::make_unique<LLVMContext>();
        SMDiagnostic Err;
        auto M = parseIR(CacheBuf->getMemBufferRef(), Err, *Ctx);
        relinkGlobals(*M, VarNameToDevPtr);
        ObjBuf = codegenObject(*M, DeviceArch);
      } else {
        ObjBuf = std::move(CacheBuf);
      }

      auto KernelFunc =
          getKernelFunctionFromImage(KernelMangled, ObjBuf->getBufferStart());

      CodeCache.insert(HashValue, KernelFunc, KernelName, RCsVec.data(),
                       NumRuntimeConstants);

      return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                  ShmemSize, Stream);
    }
  }

  // We need to clone, as extractDeviceBitcode returns a generic LLVM IR to be
  // used by any kernel that will be specialized
  auto JitModule = llvm::CloneModule(extractDeviceBitcode(KernelName, Kernel));
  // NOTE: There is potential oportunity here, to reduce some of the JIT costs
  // further. We can have a specializeIR in which we do not do any RC/Grid/Block
  // specializations. We only internalize symbols. Then we can use that IR
  // for all upcoming specializations of dynamic information.
  // There is a memory trade off in such case, We will need to have a peristent
  // in memory module, for every annotated kernel. If we have a case of 1000s of
  // kernels, this can be an issue

  specializeIR(*JitModule, KernelName, Suffix, BlockDim, GridDim, RCIndices,
               RCsVec.data(), NumRuntimeConstants);

  // For CUDA, run the target-specific optimization pipeline to optimize the
  // LLVM IR before handing over to the CUDA driver PTX compiler.
  optimizeIR(*JitModule, DeviceArch);

  SmallString<4096> ModuleBuffer;
  raw_svector_ostream ModuleBufferOS(ModuleBuffer);
  WriteBitcodeToFile(*JitModule, ModuleBufferOS);
  StorageCache.storeBitcode(HashValue, ModuleBuffer);

  relinkGlobals(*JitModule, VarNameToDevPtr);

  auto ObjBuf = codegenObject(*JitModule, DeviceArch);
  if (Config.ENV_PROTEUS_USE_STORED_CACHE)
    StorageCache.storeObject(HashValue, ObjBuf->getMemBufferRef());

  KernelFunc =
      getKernelFunctionFromImage(KernelMangled, ObjBuf->getBufferStart());

  CodeCache.insert(HashValue, KernelFunc, KernelName, RCsVec.data(),
                   NumRuntimeConstants);

  return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                              ShmemSize, Stream);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerFatBinary(void *Handle,
                                               FatbinWrapper_t *FatbinWrapper,
                                               const char *ModuleId) {
  CurHandle = Handle;
  DBG(Logger::logs("proteus")
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
      DBG(Logger::logs("proteus")
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
  DBG(Logger::logs("proteus") << "Register fatbinary end\n");
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
  DBG(Logger::logs("proteus")
      << "Register function " << Kernel << " To Handle " << Handle << "\n");
  // NOTE: HIP RDC might call multiple times the registerFunction for the same
  // kernel, which has weak linkage, when it comes from different translation
  // units. Either the first or the second call can prevail and should be
  // equivalent. We let the first one prevail.
  if (KernelToHandleMap.contains(Kernel)) {
    DBG(Logger::logs("proteus")
        << "Warning: duplicate register function for kernel " +
               std::string(KernelName)
        << "\n");
    return;
  }
  KernelToHandleMap[Kernel] = Handle;

  JITKernelInfoMap[Kernel] =
      JITKernelInfo(KernelName, RCIndices, RCTypes, NumRCs);
}

template <typename ImplT>
void JitEngineDevice<ImplT>::registerLinkedBinary(
    FatbinWrapper_t *FatbinWrapper, const char *ModuleId) {
  DBG(Logger::logs("proteus")
      << "Register linked binary FatBinary " << FatbinWrapper << " Binary "
      << (void *)FatbinWrapper->Binary << " ModuleId " << ModuleId << "\n");
  if (CurHandle) {
    if (!HandleToBinaryInfo.contains(CurHandle))
      FATAL_ERROR("Expected CurHandle in map");

    HandleToBinaryInfo[CurHandle].LinkedModuleIds.push_back(ModuleId);
  } else
    GlobalLinkedModuleIds.push_back(ModuleId);

  ModuleIdToFatBinary[ModuleId] = FatbinWrapper;
}

template <typename ImplT>
std::unique_ptr<Module> JitEngineDevice<ImplT>::linkJitModule(
    StringRef KernelName, SmallVector<std::unique_ptr<Module>> &LinkedModules) {
  if (LinkedModules.empty())
    FATAL_ERROR("Expected jit module");

  auto LinkedModule =
      std::make_unique<llvm::Module>("JitModule", getProteusLLVMCtx());
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
