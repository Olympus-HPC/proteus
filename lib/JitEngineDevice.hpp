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

#include "llvm/Linker/Linker.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/SHA256.h"
#include <cstdint>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/raw_ostream.h>
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

public:
  JITKernelInfo(char const *Name, int32_t *RCIndices, int32_t *RCTypes,
                int32_t NumRCs)
      : Name(Name), RCIndices{ArrayRef{RCIndices, static_cast<size_t>(NumRCs)}},
        RCTypes{ArrayRef{RCTypes, static_cast<size_t>(NumRCs)}},
        NumRCs(NumRCs) {}

  JITKernelInfo() : Name(nullptr), NumRCs(0), RCIndices(), RCTypes() {}
  const auto &getName() const { return Name; }
  const auto &getRCIndices() const { return RCIndices; }
  const auto &getRCTypes() const { return RCTypes; }
  const auto &getNumRCs() const { return NumRCs; }
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

  SmallVector<std::pair<std::string, SmallVector<std::unique_ptr<Module>>>>
      SHA256HashWithBitcodes;

  DenseMap<void *, std::unique_ptr<llvm::Module>> LinkedLLVMIRModules;
  DenseMap<void *, int> KernelToBitcodeIndex;
  /* @Brief After proteus initialization contains all kernels annotathed with
   * proteus */
  DenseSet<void *> ProteusAnnotatedKernels;
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

  std::unique_ptr<llvm::Module>
  createLinkedModule(ArrayRef<std::unique_ptr<Module>> LinkedModules,
                     StringRef KernelName) {
    TIMESCOPE(__FUNCTION__)
    return static_cast<ImplT &>(*this).createLinkedModule(LinkedModules,
                                                          KernelName);
  }

  int extractDeviceBitcode(StringRef KernelName, void *Kernel) {
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

  static std::string computeDeviceFatBinHash() {
    TIMESCOPE("computeDeviceFatBinHash");
    using namespace llvm::object;
    llvm::SHA256 sha256;
    auto ExePath = std::filesystem::canonical("/proc/self/exe");

    std::cout << "Reading file from path " << ExePath.string() << "\n";

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

    // Step 3: Iterate through sections and get their contents
    for (const SectionRef &section : elfObj.sections()) {
      auto nameOrErr = section.getName();
      if (!nameOrErr)
        FATAL_ERROR("Error getting section name: ");

      StringRef sectionName = nameOrErr.get();
      if (sectionName.compare(ImplT::getFatBinSectionName()) != 0)
        continue;

      // Get the contents of the section
      auto contentsOrErr = section.getContents();
      if (!contentsOrErr) {
        FATAL_ERROR("Error getting section contents: ");
        continue;
      }
      StringRef sectionContents = contentsOrErr.get();

      // Print section name and size
      outs() << "Section: " << sectionName
             << ", Size: " << sectionContents.size() << " bytes\n";
      sha256.update(sectionContents);
      break;
    }
    auto sha256Hash = sha256.final();
    return llvm::toHex(sha256Hash);
  }

protected:
  JitEngineDevice() {
    ProteusCtx = std::make_unique<LLVMContext>();
    ProteusDeviceBinHash = computeDeviceFatBinHash();
    std::cout << "Device Bin Hash is " << ProteusDeviceBinHash << "\n";
  }
  ~JitEngineDevice() {
    CodeCache.printStats();
    StorageCache.printStats();
    // Note: We manually clear or unique_ptr to Modules before the destructor
    // releases the ProteusCtx.
    //
    // Explicitly clear the LinkedLLVMIRModules
    LinkedLLVMIRModules.clear();

    // Explicitly clear SHA256HashWithBitcodes
    for (auto &Entry : SHA256HashWithBitcodes)
      Entry.second.clear();
    SHA256HashWithBitcodes.clear();
  }

  JitCache<KernelFunction_t> CodeCache;
  JitStorageCache<KernelFunction_t> StorageCache;
  std::string DeviceArch;
  std::unordered_map<std::string, const void *> VarNameToDevPtr;
  void linkJitModule(Module &M, StringRef KernelName,
                     ArrayRef<std::unique_ptr<Module>> LinkedModules);
  std::string
  getCombinedModuleHash(ArrayRef<std::unique_ptr<Module>> LinkedModules);

  // All modules are associated with context, to guarantee correct lifetime.
  std::unique_ptr<LLVMContext> ProteusCtx;
  std::string ProteusDeviceBinHash;

private:
  // This map is private and only accessible via the API.
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
  DBG(Logger::logs("proteus") << "=== Parsed Module\n"
                              << M << "=== End of Parsed Module\n");
  Function *F = M.getFunction(FnName);
  assert(F && "Expected non-null function!");

  // Remove llvm.global.annotations now that we have read them.
  if (auto *GlobalAnnotations = M.getGlobalVariable("llvm.global.annotations"))
    M.eraseGlobalVariable(GlobalAnnotations);
  // Remove the __clang_gpu_used_external used in HIP RDC compilation and its
  // uses in llvm.used, llvm.compiler.used.
  if (auto *ClangGPUUsedExternal =
          M.getNamedGlobal("__clang_gpu_used_external")) {
    removeFromUsedLists(M, [&ClangGPUUsedExternal](Constant *C) {
      if (auto *GV = dyn_cast<GlobalVariable>(C))
        return GV == ClangGPUUsedExternal;
      return false;
    });
    M.eraseGlobalVariable(ClangGPUUsedExternal);
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

  // This was never registered, return immediately
  if (!KernelToHandleMap.contains(Kernel))
    return launchKernelDirect(Kernel, GridDim, BlockDim, KernelArgs, ShmemSize,
                              Stream);

  SmallVector<RuntimeConstant> RCsVec;

  getRuntimeConstantValues(KernelArgs, RCIndices, RCTypes, RCsVec);

  typename DeviceTraits<ImplT>::KernelFunction_t KernelFunc;

  auto Index = KernelToBitcodeIndex.contains(Kernel)
                   ? KernelToBitcodeIndex[Kernel]
                   : extractDeviceBitcode(KernelName, Kernel);

  // I have already read the LLVM IR from the Binary. Pick the Static Hash
  auto StaticHash = SHA256HashWithBitcodes[Index].first;
  // TODO: This does not include the GridDims/BlockDims. We need to fix it.
  auto PersistentHash = ProteusDeviceBinHash;
  uint64_t DynamicHashValue =
      CodeCache.hash(PersistentHash, KernelName, GridDim, BlockDim,
                     RCsVec.data(), NumRuntimeConstants);
  KernelFunc = CodeCache.lookup(DynamicHashValue);
  std::cout << " Function with name " << KernelName.str() << "at address "
            << Kernel << " has PersistentHash " << PersistentHash
            << " Static Hash:" << StaticHash
            << " Dynamic Hash:" << DynamicHashValue << "\n";

  // We found the kernel, execute
  if (KernelFunc)
    return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                ShmemSize, Stream);

  // NOTE: we don't need a suffix to differentiate kernels, each specialization
  // will be in its own module uniquely identify by HashValue. It exists only
  // for debugging purposes to verify that the jitted kernel executes.
  std::string Suffix = mangleSuffix(DynamicHashValue);
  std::string KernelMangled = (KernelName + Suffix).str();

  if (Config.ENV_PROTEUS_USE_STORED_CACHE) {
    // FIXME: The code cache is completely broken as of now. I need to revisit
    // this. If there device global variables, lookup the IR and codegen object
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
                 ? StorageCache.lookupBitcode(DynamicHashValue, KernelMangled)
                 : StorageCache.lookupObject(DynamicHashValue,
                                             KernelMangled))) {
      std::unique_ptr<MemoryBuffer> ObjBuf;
      if (HasDeviceGlobals) {
        SMDiagnostic Err;
        auto M = parseIR(CacheBuf->getMemBufferRef(), Err, *ProteusCtx.get());
        relinkGlobals(*M, VarNameToDevPtr);
        ObjBuf = codegenObject(*M, DeviceArch);
      } else {
        ObjBuf = std::move(CacheBuf);
      }

      auto KernelFunc =
          getKernelFunctionFromImage(KernelMangled, ObjBuf->getBufferStart());

      CodeCache.insert(DynamicHashValue, KernelFunc, KernelName, RCsVec.data(),
                       NumRuntimeConstants);

      return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                  ShmemSize, Stream);
    }
  }

  if (!LinkedLLVMIRModules.contains(Kernel)) {
    // if we get here, we have access to the LLVM-IR of the module, but we
    // have never linked everything together and internalized the symbols.
    LinkedLLVMIRModules.insert(
        {Kernel,
         createLinkedModule(SHA256HashWithBitcodes[Index].second, KernelName)});
  }

  // We need to clone, The JitModule will be specialized later, and we need
  // the one stored under LinkedLLVMIRModules to be a generic version prior
  // specialization.
  auto JitModule = llvm::CloneModule(*LinkedLLVMIRModules[Kernel]);

  specializeIR(*JitModule, KernelName, Suffix, BlockDim, GridDim, RCIndices,
               RCsVec.data(), NumRuntimeConstants);

  // For CUDA, run the target-specific optimization pipeline to optimize the
  // LLVM IR before handing over to the CUDA driver PTX compiler.
  optimizeIR(*JitModule, DeviceArch);

  SmallString<4096> ModuleBuffer;
  raw_svector_ostream ModuleBufferOS(ModuleBuffer);
  WriteBitcodeToFile(*JitModule, ModuleBufferOS);
  StorageCache.storeBitcode(DynamicHashValue, ModuleBuffer);

  relinkGlobals(*JitModule, VarNameToDevPtr);

  auto ObjBuf = codegenObject(*JitModule, DeviceArch);
  if (Config.ENV_PROTEUS_USE_STORED_CACHE)
    StorageCache.storeObject(DynamicHashValue, ObjBuf->getMemBufferRef());

  KernelFunc =
      getKernelFunctionFromImage(KernelMangled, ObjBuf->getBufferStart());

  CodeCache.insert(DynamicHashValue, KernelFunc, KernelName, RCsVec.data(),
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
  assert(!KernelToHandleMap.contains(Kernel) &&
         "Expected kernel inserted only once in the map");
  KernelToHandleMap[Kernel] = Handle;

  assert(!ProteusAnnotatedKernels.contains(Kernel) &&
         "Expected kernel inserted only once in proteus kernel map");
  ProteusAnnotatedKernels.insert(Kernel);

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
std::string JitEngineDevice<ImplT>::getCombinedModuleHash(
    ArrayRef<std::unique_ptr<Module>> LinkedModules) {
  SmallVector<std::string> SHA256HashCodes;
  for (auto &Mod : LinkedModules) {
    NamedMDNode *ProteusSHANode =
        Mod->getNamedMetadata("proteus.module.sha256");
    assert(ProteusSHANode != nullptr &&
           "Expected non-null proteus.module.sha256 metadata");
    assert(ProteusSHANode->getNumOperands() == 1 &&
           "Hash MD Node should have a single operand");
    auto MDHash = ProteusSHANode->getOperand(0);
    MDString *sha256 = dyn_cast<MDString>(MDHash->getOperand(0));
    if (!sha256) {
      FATAL_ERROR("Could not read sha256 from module\n");
    }
    SHA256HashCodes.push_back(sha256->getString().str());
    Mod->eraseNamedMetadata(ProteusSHANode);
  }

  std::sort(SHA256HashCodes.begin(), SHA256HashCodes.end());
  std::string combinedHash;
  for (auto hash : SHA256HashCodes) {
    combinedHash += hash;
  }
  return combinedHash;
}

template <typename ImplT>
void JitEngineDevice<ImplT>::linkJitModule(
    Module &M, StringRef KernelName,
    ArrayRef<std::unique_ptr<Module>> LinkedModules) {
  if (LinkedModules.empty())
    FATAL_ERROR("Expected jit module");

  Linker IRLinker(M);
  for (auto &LinkedM : llvm::reverse(LinkedModules)) {
    // Returns true if linking failed.
    if (IRLinker.linkInModule(std::move(LinkedM)))
      FATAL_ERROR("Linking failed");
  }
}

} // namespace proteus

#endif
