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

static inline Error createSMDiagnosticError(llvm::SMDiagnostic &Diag) {
  std::string Msg;
  {
    llvm::raw_string_ostream OS(Msg);
    Diag.print("", OS);
  }
  return llvm::make_error<llvm::StringError>(std::move(Msg),
                                             llvm::inconvertibleErrorCode());
}

class JitEngine {
protected:
  Expected<std::unique_ptr<TargetMachine>>
  createTargetMachine(Module &M, StringRef CPU /*, unsigned OptLevel*/);

  void runOptimizationPassPipeline(Module &M, StringRef CPU);

  JitEngine();

  bool getEnvOrDefaultBool(const char *VarName, bool Default);

  std::string mangleSuffix(uint64_t HashValue);

  struct {
    bool ENV_JIT_USE_STORED_CACHE;
    bool ENV_JIT_LAUNCH_BOUNDS;
    bool ENV_JIT_SPECIALIZE_ARGS;
  } Config;
};

} // namespace proteus

#endif