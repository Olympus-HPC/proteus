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

#include "proteus/CoreLLVM.hpp"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"
#include "proteus/Logger.hpp"

namespace proteus {

using namespace llvm;

inline bool getEnvOrDefaultBool(const char *VarName, bool Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : Default;
}

inline int getEnvOrDefaultInt(const char *VarName, int Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? std::stoi(EnvValue) : Default;
}

class JitEngine {
public:
  InitLLVMTargets Init;
  void optimizeIR(Module &M, StringRef Arch, StringRef OptLevel = "default<O3>",
                  unsigned CodegenOptLevel = 3);

  bool isProteusDisabled() { return Config.PROTEUS_DISABLE; }

  void enable() { Config.PROTEUS_DISABLE = false; }

  void disable() { Config.PROTEUS_DISABLE = true; }

protected:
  void runCleanupPassPipeline(Module &M);

  JitEngine();

  std::string mangleSuffix(HashT &HashValue);

  struct {
    bool PROTEUS_USE_STORED_CACHE;
    bool PROTEUS_SET_LAUNCH_BOUNDS;
    bool PROTEUS_SPECIALIZE_ARGS;
    bool PROTEUS_SPECIALIZE_DIMS;
    bool PROTEUS_USE_HIP_RTC_CODEGEN;
    bool PROTEUS_DISABLE;
    bool PROTEUS_DUMP_LLVM_IR;
    bool PROTEUS_RELINK_GLOBALS_BY_COPY;
    bool PROTEUS_ASYNC_COMPILATION;
    int PROTEUS_ASYNC_THREADS;
    bool PROTEUS_ASYNC_TEST_BLOCKING;
    bool PROTEUS_USE_LIGHTWEIGHT_KERNEL_CLONE;
  } Config;
};

} // namespace proteus

#endif
