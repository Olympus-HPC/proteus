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
#include <llvm/IR/Constants.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "CompilerInterfaceTypes.h"
#include "JitCache.hpp"
#include "JitEngine.hpp"
#include "JitStorageCache.hpp"
#include "TransformArgumentSpecialization.hpp"
#include "Utils.h"

// TODO: check if this global is needed.
static llvm::codegen::RegisterCodeGenFlags CFG;

namespace proteus {

using namespace llvm;

struct FatbinWrapper_t {
  int32_t Magic;
  int32_t Version;
  const char *Binary;
  void *X;
};

template <typename ImplT> struct DeviceTraits;

template <typename ImplT> class JitEngineDevice : protected JitEngine {
public:
  using DeviceError_t = typename DeviceTraits<ImplT>::DeviceError_t;
  using DeviceStream_t = typename DeviceTraits<ImplT>::DeviceStream_t;
  using KernelFunction_t = typename DeviceTraits<ImplT>::KernelFunction_t;

  DeviceError_t compileAndRun(StringRef ModuleUniqueId, StringRef KernelName,
                              FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
                              RuntimeConstant *RC, int NumRuntimeConstants,
                              dim3 GridDim, dim3 BlockDim, void **KernelArgs,
                              uint64_t ShmemSize, DeviceStream_t Stream);

  void insertRegisterVar(const char *VarName, const void *Addr) {
    VarNameToDevPtr[VarName] = Addr;
  }

private:
  //------------------------------------------------------------------
  // Begin Methods implemented in the derived device engine class.
  //------------------------------------------------------------------
  void *resolveDeviceGlobalAddr(const void *Addr) {
    return static_cast<ImplT &>(*this).resolveDeviceGlobalAddr(Addr);
  }

  void setLaunchBoundsForKernel(Module *M, Function *F, int GridSize,
                                int BlockSize) {
    static_cast<ImplT &>(*this).setLaunchBoundsForKernel(M, F, GridSize,
                                                         BlockSize);
  }

  DeviceError_t launchKernelFunction(KernelFunction_t KernelFunc, dim3 GridDim,
                                     dim3 BlockDim, void **KernelArgs,
                                     uint64_t ShmemSize,
                                     DeviceStream_t Stream) {
    return static_cast<ImplT &>(*this).launchKernelFunction(
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

  std::unique_ptr<MemoryBuffer> extractDeviceBitcode(StringRef KernelName,
                                                     const char *Binary,
                                                     size_t FatbinSize = 0) {
    return static_cast<ImplT &>(*this).extractDeviceBitcode(KernelName, Binary,
                                                            FatbinSize);
  }
  //------------------------------------------------------------------
  // End Methods implemented in the derived device engine class.
  //------------------------------------------------------------------

  Expected<orc::ThreadSafeModule>
  specializeIR(StringRef FnName, StringRef Suffix, StringRef IR, int BlockSize,
               int GridSize, RuntimeConstant *RC, int NumRuntimeConstants);

  void
  relinkGlobals(Module &M,
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
};

template <typename ImplT>
Expected<orc::ThreadSafeModule> JitEngineDevice<ImplT>::specializeIR(
    StringRef FnName, StringRef Suffix, StringRef IR, int BlockSize,
    int GridSize, RuntimeConstant *RC, int NumRuntimeConstants) {

  TIMESCOPE("specializeIR");
  auto Ctx = std::make_unique<LLVMContext>();
  SMDiagnostic Err;
  if (auto M = parseIR(MemoryBufferRef(IR, ("Mod-" + FnName + Suffix).str()),
                       Err, *Ctx)) {
    DBG(dbgs() << "=== Parsed Module\n" << *M << "=== End of Parsed Module\n");
    Function *F = M->getFunction(FnName);
    assert(F && "Expected non-null function!");
    MDNode *Node = F->getMetadata("jit_arg_nos");
    assert(Node && "Expected metadata for jit arguments");
    DBG(dbgs() << "Metadata jit for F " << F->getName() << " = " << *Node
               << "\n");

    // Replace argument uses with runtime constants.
    if (Config.ENV_PROTEUS_SPECIALIZE_ARGS)
      // TODO: change NumRuntimeConstants to size_t at interface.
      TransformArgumentSpecialization::transform(
          *M, *F,
          ArrayRef<RuntimeConstant>{RC,
                                    static_cast<size_t>(NumRuntimeConstants)});

    DBG(dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n");

    F->setName(FnName + Suffix);

    if (Config.ENV_PROTEUS_SET_LAUNCH_BOUNDS)
      setLaunchBoundsForKernel(M.get(), F, GridSize, BlockSize);

#if ENABLE_DEBUG
    dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
    if (verifyModule(*M, &errs()))
      FATAL_ERROR("Broken module found, JIT compilation aborted!");
    else
      dbgs() << "Module verified!\n";
#endif
    return orc::ThreadSafeModule(std::move(M), std::move(Ctx));
  }

  return createSMDiagnosticError(Err);
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
  dbgs() << "=== Linked M\n" << M << "=== End of Linked M\n";
  if (verifyModule(M, &errs()))
    FATAL_ERROR("After linking, broken module found, JIT compilation aborted!");
  else
    dbgs() << "Module verified!\n";
#endif
}

template <typename ImplT>
typename DeviceTraits<ImplT>::DeviceError_t
JitEngineDevice<ImplT>::compileAndRun(
    StringRef ModuleUniqueId, StringRef KernelName,
    FatbinWrapper_t *FatbinWrapper, size_t FatbinSize, RuntimeConstant *RC,
    int NumRuntimeConstants, dim3 GridDim, dim3 BlockDim, void **KernelArgs,
    uint64_t ShmemSize, typename DeviceTraits<ImplT>::DeviceStream_t Stream) {
  TIMESCOPE("compileAndRun");

  uint64_t HashValue =
      CodeCache.hash(ModuleUniqueId, KernelName, RC, NumRuntimeConstants);
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

      CodeCache.insert(HashValue, KernelFunc, KernelName, RC,
                       NumRuntimeConstants);

      return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                                  ShmemSize, Stream);
    }
  }

  auto IRBuffer =
      extractDeviceBitcode(KernelName, FatbinWrapper->Binary, FatbinSize);

  auto SpecializedModule =
      specializeIR(KernelName, Suffix, IRBuffer->getBuffer(),
                   BlockDim.x * BlockDim.y * BlockDim.z,
                   GridDim.x * GridDim.y * GridDim.z, RC, NumRuntimeConstants);
  if (auto E = SpecializedModule.takeError())
    FATAL_ERROR(toString(std::move(E)).c_str());

  Module *M = SpecializedModule->getModuleUnlocked();

  // For CUDA, run the target-specific optimization pipeline to optimize the
  // LLVM IR before handing over to the CUDA driver PTX compiler.
  optimizeIR(*M, DeviceArch);

  SmallString<4096> ModuleBuffer;
  raw_svector_ostream ModuleBufferOS(ModuleBuffer);
  WriteBitcodeToFile(*M, ModuleBufferOS);
  StorageCache.storeBitcode(HashValue, ModuleBuffer);

  relinkGlobals(*M, VarNameToDevPtr);

  auto ObjBuf = codegenObject(*M, DeviceArch);
  if (Config.ENV_PROTEUS_USE_STORED_CACHE)
    StorageCache.storeObject(HashValue, ObjBuf->getMemBufferRef());

  KernelFunc =
      getKernelFunctionFromImage(KernelMangled, ObjBuf->getBufferStart());

  CodeCache.insert(HashValue, KernelFunc, KernelName, RC, NumRuntimeConstants);

  return launchKernelFunction(KernelFunc, GridDim, BlockDim, KernelArgs,
                              ShmemSize, Stream);
}

} // namespace proteus

#endif
