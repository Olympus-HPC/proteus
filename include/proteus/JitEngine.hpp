//===-- JitEngine.hpp -- Base JIT Engine header --===//
//
// Part of the Proteus Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef PROTEUS_JITENGINE_HPP
#define PROTEUS_JITENGINE_HPP

#include <cstdlib>
#include <optional>
#include <string>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"
#include "proteus/Logger.hpp"

namespace proteus {

using namespace llvm;

inline bool getEnvOrDefaultBool(const char *VarName, bool Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : Default;
}

class JitEngine {
public:
  void optimizeIR(Module &M, StringRef Arch, char OptLevel = '3',
                  unsigned CodegenOptLevel = 3);

  bool isProteusDisabled() { return Config.ENV_PROTEUS_DISABLE; }

  void pushJitVariable(RuntimeConstant &RC);
  void registerLambda(const char *Symbol);

protected:
  void runCleanupPassPipeline(Module &M);

  JitEngine();

  std::string mangleSuffix(HashT &HashValue);

  struct {
    bool ENV_PROTEUS_USE_STORED_CACHE;
    bool ENV_PROTEUS_SET_LAUNCH_BOUNDS;
    bool ENV_PROTEUS_SPECIALIZE_ARGS;
    bool ENV_PROTEUS_SPECIALIZE_DIMS;
    bool ENV_PROTEUS_USE_HIP_RTC_CODEGEN;
    bool ENV_PROTEUS_DISABLE;
    bool ENV_PROTEUS_DUMP_LLVM_IR;
    bool ENV_PROTEUS_RELINK_GLOBALS_BY_COPY;
  } Config;
};

} // namespace proteus

#endif
