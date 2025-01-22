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
#include <memory>
#include <string>

#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

#include "Utils.h"

namespace proteus {

using namespace llvm;

static inline Error createSMDiagnosticError(SMDiagnostic &Diag) {
  std::string Msg;
  {
    raw_string_ostream OS(Msg);
    Diag.print("", OS);
  }
  return make_error<StringError>(std::move(Msg), inconvertibleErrorCode());
}

static inline bool getEnvOrDefaultBool(const char *VarName, bool Default) {

  const char *EnvValue = std::getenv(VarName);
  return EnvValue ? static_cast<bool>(std::stoi(EnvValue)) : Default;
}

class JitEngine {
public:
  static void optimizeIR(Module &M, StringRef Arch);

  bool isProteusDisabled() { return Config.ENV_PROTEUS_DISABLE; }

  static void runCleanupPassPipeline(Module &M);

protected:
  static Expected<std::unique_ptr<TargetMachine>>
  createTargetMachine(Module &M, StringRef Arch, unsigned OptLevel = 3);

  static void runOptimizationPassPipeline(Module &M, StringRef Arch,
                                          unsigned OptLevel = 3);

  JitEngine();

  static std::string mangleSuffix(uint64_t HashValue);

  struct {
    bool ENV_PROTEUS_USE_STORED_CACHE;
    bool ENV_PROTEUS_SET_LAUNCH_BOUNDS;
    bool ENV_PROTEUS_SPECIALIZE_ARGS;
    bool ENV_PROTEUS_SPECIALIZE_DIMS;
    bool ENV_PROTEUS_USE_HIP_RTC_CODEGEN;
    bool ENV_PROTEUS_DISABLE;
  } Config;
};

} // namespace proteus

#endif
