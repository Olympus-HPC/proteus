//===-- JitEngineInfoRegistry.h -- JIT engine info registry --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Registry for JIT engine information: tracks registered fat binaries,
// linked binaries, functions, and variables used by Proteus JIT engines.
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JIT_ENGINE_INFO_REGISTRY_H
#define PROTEUS_JIT_ENGINE_INFO_REGISTRY_H

#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"

#include <llvm/ADT/SmallVector.h>

#include <cstdint>
#include <unordered_map>

namespace proteus {

struct RegisterVarInfo {
  void *Handle;
  const void *HostAddr;
  const char *VarName;
  uint64_t VarSize;
};

struct RegisterLinkedBinaryInfo {
  void *FatbinWrapper;
  const char *ModuleId;
};

struct RegisterFunctionInfo {
  void *Handle;
  void *Kernel;
  char *KernelName;
  ArrayRef<RuntimeConstantInfo *> RCInfoArray;
};

struct RegisterFatBinaryInfo {
  void *Handle;
  void *FatbinWrapper;
  const char *ModuleId;

  SmallVector<RegisterVarInfo> Vars;
  SmallVector<RegisterLinkedBinaryInfo> LinkedBinaries;
  SmallVector<RegisterFunctionInfo> Functions;

  RegisterFatBinaryInfo(void *Handle, void *FatbinWrapper, const char *ModuleId)
      : Handle(Handle), FatbinWrapper(FatbinWrapper), ModuleId(ModuleId) {}
};

class JitEngineInfoRegistry {
public:
  static JitEngineInfoRegistry &instance() {
    static JitEngineInfoRegistry Instance;
    return Instance;
  }

  std::unordered_map<void *, RegisterFatBinaryInfo> FatbinaryMap;
  SmallVector<RegisterLinkedBinaryInfo> RegisteredLinkedBinaries;

  void registerFatBinary(void *Handle, void *FatbinWrapper,
                         const char *ModuleId) {
    FatbinaryMap.emplace(
        Handle, RegisterFatBinaryInfo(Handle, FatbinWrapper, ModuleId));
  }

  // Register linked binary is CUDA specific.
  void registerLinkedBinary(void *FatbinWrapper, const char *ModuleId) {
    RegisteredLinkedBinaries.push_back({FatbinWrapper, ModuleId});
  }

  void registerFunction(void *Handle, void *Kernel, char *KernelName,
                        ArrayRef<RuntimeConstantInfo *> RCInfoArray) {
    auto &FatbinInfo = FatbinaryMap.at(Handle);
    FatbinInfo.Functions.push_back({Handle, Kernel, KernelName, RCInfoArray});
  }

  void registerVar(void *Handle, const void *HostAddr, const char *VarName,
                   uint64_t VarSize) {
    auto &FatbinInfo = FatbinaryMap.at(Handle);
    FatbinInfo.Vars.push_back({Handle, HostAddr, VarName, VarSize});
  }

  void registerFatBinaryEnd(void *Handle) {
    // CUDA specific: associate any registered linked binary with this fat
    // binary.
    auto &FatbinInfo = FatbinaryMap.at(Handle);
    FatbinInfo.LinkedBinaries = RegisteredLinkedBinaries;
    RegisteredLinkedBinaries.clear();
  }

private:
  JitEngineInfoRegistry() = default;
};

} // namespace proteus

#endif
