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
#include "JitStoredCache.hpp"
#include "TransformArgumentSpecialization.hpp"
#include "Utils.h"

// TODO: check if this global is needed.
static llvm::codegen::RegisterCodeGenFlags CFG;

namespace proteus {

using namespace llvm;
using namespace llvm::orc;

struct FatbinWrapper_t {
  int32_t Magic;
  int32_t Version;
  const char *Binary;
  void *X;
};

template <typename ImplT> class JitEngineDevice : protected JitEngine {
public:
  Expected<llvm::orc::ThreadSafeModule>
  specializeBitcode(StringRef FnName, StringRef Suffix, StringRef IR,
                    int BlockSize, int GridSize, RuntimeConstant *RC,
                    int NumRuntimeConstants);

  auto compileAndRun(StringRef ModuleUniqueId, StringRef KernelName,
                     FatbinWrapper_t *FatbinWrapper, size_t FatbinSize,
                     RuntimeConstant *RC, int NumRuntimeConstants, dim3 GridDim,
                     dim3 BlockDim, void **KernelArgs, uint64_t ShmemSize,
                     void *Stream) {
    return static_cast<ImplT &>(*this).compileAndRun(
        ModuleUniqueId, KernelName, FatbinWrapper, FatbinSize, RC,
        NumRuntimeConstants, GridDim, BlockDim, KernelArgs, ShmemSize, Stream);
  }

  void insertRegisterVar(const char *VarName, const void *Addr) {
    VarNameToDevPtr[VarName] = Addr;
  }

  void
  relinkGlobals(Module &M,
                std::unordered_map<std::string, const void *> &VarNameToDevPtr);

  void *resolveDeviceGlobalAddr(const void *Addr) {
    return static_cast<ImplT &>(*this).resolveDeviceGlobalAddr(Addr);
  }

  void setLaunchBoundsForKernel(Module *M, llvm::Function *F, int GridSize,
                                int BlockSize) {
    static_cast<ImplT &>(*this).setLaunchBoundsForKernel(M, F, GridSize,
                                                         BlockSize);
  }

protected:
  JitEngineDevice() {
    Config.ENV_JIT_USE_STORED_CACHE =
        getEnvOrDefaultBool("ENV_JIT_USE_STORED_CACHE", true);
    Config.ENV_JIT_LAUNCH_BOUNDS =
        getEnvOrDefaultBool("ENV_JIT_LAUNCH_BOUNDS", true);
    Config.ENV_JIT_SPECIALIZE_ARGS =
        getEnvOrDefaultBool("ENV_JIT_SPECIALIZE_ARGS", true);

#if ENABLE_DEBUG
    dbgs() << "ENV_JIT_USE_STORED_CACHE " << Config.ENV_JIT_USE_STORED_CACHE
           << "\n";
    dbgs() << "ENV_JIT_LAUNCH_BOUNDS " << Config.ENV_JIT_LAUNCH_BOUNDS << "\n";
    dbgs() << "ENV_JIT_SPECIALIZE_ARGS " << Config.ENV_JIT_SPECIALIZE_ARGS
           << "\n";
#endif
  }

  bool getEnvOrDefaultBool(const char *VarName, bool Default) {
    const char *EnvValue = std::getenv(VarName);
    return EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : Default;
  }

  std::unordered_map<std::string, const void *> VarNameToDevPtr;
  struct {
    bool ENV_JIT_USE_STORED_CACHE;
    bool ENV_JIT_LAUNCH_BOUNDS;
    bool ENV_JIT_SPECIALIZE_ARGS;
  } Config;
};

template <typename ImplT>
Expected<llvm::orc::ThreadSafeModule> JitEngineDevice<ImplT>::specializeBitcode(
    StringRef FnName, StringRef Suffix, StringRef IR, int BlockSize,
    int GridSize, RuntimeConstant *RC, int NumRuntimeConstants) {

  TIMESCOPE("specializeBitcode");
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

    // Relink device globals.
    relinkGlobals(*M, VarNameToDevPtr);

    // Replace argument uses with runtime constants.
    if (Config.ENV_JIT_SPECIALIZE_ARGS)
      // TODO: change NumRuntimeConstants to size_t at interface.
      TransformArgumentSpecialization::transform(
          *M, *F,
          ArrayRef<RuntimeConstant>{RC,
                                    static_cast<size_t>(NumRuntimeConstants)});

    DBG(dbgs() << "=== JIT Module\n" << *M << "=== End of JIT Module\n");

    F->setName(FnName + Suffix);

    if (Config.ENV_JIT_LAUNCH_BOUNDS)
      setLaunchBoundsForKernel(M.get(), F, GridSize, BlockSize);

#if ENABLE_DEBUG
    dbgs() << "=== Final Module\n" << *M << "=== End Final Module\n";
    if (verifyModule(*M, &errs()))
      FATAL_ERROR("Broken module found, JIT compilation aborted!");
    else
      dbgs() << "Module verified!\n";
#endif
    return ThreadSafeModule(std::move(M), std::move(Ctx));
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
    assert(GV && "Expected existing global variable");
    // Remove the re-linked global from llvm.compiler.used since that
    // use is not replaceable by the fixed addr constant expression.
    llvm::removeFromUsedLists(M, [&GV](Constant *C) {
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
  llvm::dbgs() << "=== Linked M\n" << M << "=== End of Linked M\n";
  if (verifyModule(M, &llvm::errs()))
    FATAL_ERROR("After linking, broken module found, JIT compilation aborted!");
  else
    llvm::dbgs() << "Module verified!\n";
#endif
}

} // namespace proteus

#endif
