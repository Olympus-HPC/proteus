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
#include <llvm/ADT/DenseMap.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <memory>
#include <optional>
#include <string>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Hashing.hpp"
#include "proteus/Utils.h"

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
  void optimizeIR(Module &M, StringRef Arch);

  bool isProteusDisabled() { return Config.ENV_PROTEUS_DISABLE; }

  void pushJitVariable(RuntimeConstant &RC);
  void registerLambda(const char *Symbol);

protected:
  Expected<std::unique_ptr<TargetMachine>>
  createTargetMachine(Module &M, StringRef Arch, unsigned OptLevel = 3);

  void runOptimizationPassPipeline(Module &M, StringRef Arch,
                                   unsigned OptLevel = 3);
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

inline SmallVector<RuntimeConstant> &getPendingJitVariables() {
  static SmallVector<RuntimeConstant> PendingJitVariables;
  return PendingJitVariables;
}

inline DenseMap<StringRef, SmallVector<RuntimeConstant>> &getJitVariableMap() {
  // NOTE: The StringRef key refers to the global variable string of the lambda
  // class symbol created in the proteus pass.
  static DenseMap<StringRef, SmallVector<RuntimeConstant>> JitVariableMap;
  return JitVariableMap;
}

inline std::optional<
    DenseMap<StringRef, SmallVector<RuntimeConstant>>::iterator>
matchJitVariableMap(StringRef FnName) {
  std::string Operator = llvm::demangle(FnName.str());
  std::size_t Sep = Operator.rfind("::operator()");
  if (Sep == std::string::npos) {
    PROTEUS_DBG(Logger::logs("proteus") << "... SKIP ::operator() not found\n");
    return std::nullopt;
  }

  StringRef Symbol = StringRef{Operator}.slice(0, Sep);
#if PROTEUS_ENABLE_DEBUG
  Logger::logs("proteus") << "Operator " << Operator << "\n=> Symbol to match "
                          << Symbol << "\n";
  Logger::logs("proteus") << "Available Keys\n";
  for (auto &[Key, Val] : getJitVariableMap()) {
    Logger::logs("proteus") << "\tKey: " << Key << "\n";
  }
  Logger::logs("proteus") << "===\n";
#endif

  const auto SymToRC = getJitVariableMap().find(Symbol);
  if (SymToRC == getJitVariableMap().end())
    return std::nullopt;

  return SymToRC;
}

inline void pushJitVariable(RuntimeConstant RC) {
  getPendingJitVariables().push_back(RC);
}

inline void registerLambda(const char *Symbol) {
  const StringRef SymbolStr{Symbol};
  PROTEUS_DBG(Logger::logs("proteus")
              << "=> RegisterLambda " << Symbol << "\n");
  auto &JitVariables = getPendingJitVariables();
  auto &VariableMap = getJitVariableMap();
  VariableMap[SymbolStr] = JitVariables;
  JitVariables.clear();
}

} // namespace proteus

#endif
