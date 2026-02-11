#ifndef PROTEUS_JIT_ENGINE_INFO_REGISTRY_H
#define PROTEUS_JIT_ENGINE_INFO_REGISTRY_H

#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"

#include <llvm/ADT/SmallVector.h>

#include <cstdint>

namespace proteus {

struct RegisterVarInfo {
  void *Handle;
  const void *HostAddr;
  const char *VarName;
  uint64_t VarSize;
};

struct RegisterFatBinaryInfo {
  void *Handle;
  void *FatbinWrapper;
  const char *ModuleId;
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

class JitEngineInfoRegistry {
public:
  static JitEngineInfoRegistry &instance() {
    static JitEngineInfoRegistry Instance;
    return Instance;
  }

  SmallVector<RegisterVarInfo> RegisteredVars;
  SmallVector<RegisterFatBinaryInfo> RegisteredFatBinaries;
  SmallVector<RegisterLinkedBinaryInfo> RegisteredLinkedBinaries;
  SmallVector<RegisterFunctionInfo> RegisteredFunctions;

  void registerVar(void *Handle, const void *HostAddr, const char *VarName,
                   uint64_t VarSize) {
    RegisteredVars.push_back({Handle, HostAddr, VarName, VarSize});
  }

  void registerFatBinary(void *Handle, void *FatbinWrapper,
                         const char *ModuleId) {
    RegisteredFatBinaries.push_back({Handle, FatbinWrapper, ModuleId});
  }

  void registerLinkedBinary(void *FatbinWrapper, const char *ModuleId) {
    RegisteredLinkedBinaries.push_back({FatbinWrapper, ModuleId});
  }

  void registerFunction(void *Handle, void *Kernel, char *KernelName,
                        ArrayRef<RuntimeConstantInfo *> RCInfoArray) {
    RegisteredFunctions.push_back({Handle, Kernel, KernelName, RCInfoArray});
  }

private:
  JitEngineInfoRegistry() = default;
};

} // namespace proteus

#endif
