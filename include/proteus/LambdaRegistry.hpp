#ifndef PROTEUS_LAMBDA_INTERFACE_HPP
#define PROTEUS_LAMBDA_INTERFACE_HPP

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Error.h"
#include "proteus/Logger.hpp"

namespace proteus {

using namespace llvm;

// The LambdaRegistry stores the unique lambda type symbol and JIT member
// variables. We assume there is always ONLY ONE pending lambda registration
// which is used up when JIT compiling the caller kernel or running its cached
// instance. The helper class LambdaRegistryRAII clears the registration upon
// compilation or running from cache. The single pending lambda assumption is
// checked at runtime to abort execution with an error.

class LambdaRegistry {
public:
  static LambdaRegistry &instance() {
    static LambdaRegistry Singleton;
    return Singleton;
  }

  std::optional<const std::reference_wrapper<SmallVector<RuntimeConstant>>>
  matchJitVariableMap(StringRef FnName) {
    std::string Operator = llvm::demangle(FnName.str());
    std::size_t Sep = Operator.rfind("::operator()");
    if (Sep == std::string::npos) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << "... SKIP ::operator() not found\n");
      return std::nullopt;
    }

    StringRef Symbol = StringRef{Operator}.slice(0, Sep);
#if PROTEUS_ENABLE_DEBUG
    Logger::logs("proteus")
        << "Operator " << Operator << "\n=> Symbol to match " << Symbol
        << " <-> Registered Symbol " << SymbolRef << "\n";
#endif

    if (SymbolRef == Symbol)
      return PendingJitVariables;

    return std::nullopt;
  }

  void pushJitVariable(RuntimeConstant &RC) {
    PendingJitVariables.emplace_back(RC);
  }

  // The Symbol input argument is created as a global variable in the
  // ProteusPass, thus it has program-wide lifetime. Hence it is valid for
  // SymbolRef to store a reference to it.
  inline void registerLambda(const char *Symbol) {
    if (!empty())
      PROTEUS_FATAL_ERROR("Expected a single lambda registration");

    SymbolRef = Symbol;
    PROTEUS_DBG(Logger::logs("proteus")
                << "=> RegisterLambda " << Symbol << "\n");
  }

  bool empty() { return (SymbolRef == std::nullopt); }

  const SmallVector<RuntimeConstant> &getPendingJitVariables() const {
    return PendingJitVariables;
  }

  void clear() {
    SymbolRef = std::nullopt;
    PendingJitVariables.clear();
  }

private:
  explicit LambdaRegistry() = default;
  std::optional<StringRef> SymbolRef;
  SmallVector<RuntimeConstant> PendingJitVariables;
};

class LambdaRegistryRAII {
public:
  explicit LambdaRegistryRAII() = default;
  ~LambdaRegistryRAII() { LambdaRegistry::instance().clear(); }
};

} // namespace proteus

#endif
