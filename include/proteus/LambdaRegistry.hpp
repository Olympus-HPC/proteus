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

// The LambdaRegistry stores the unique lambda type symbol and the values of Jit
// member variables in a map for retrieval by the Jit engines.
class LambdaRegistry {
public:
  static LambdaRegistry &instance() {
    static LambdaRegistry Singleton;
    return Singleton;
  }

  std::optional<DenseMap<StringRef, SmallVector<RuntimeConstant>>::iterator>
  matchJitVariableMap(StringRef FnName) {
    std::string Operator = llvm::demangle(FnName.str());
    std::size_t Sep = Operator.rfind("::operator()");
    if (Sep == std::string::npos) {
      PROTEUS_DBG(Logger::logs("proteus")
                  << "... SKIP ::operator() not found\n");
      return std::nullopt;
    }

    StringRef LambdaType = StringRef{Operator}.slice(0, Sep);
#if PROTEUS_ENABLE_DEBUG
    Logger::logs("proteus")
        << "Operator " << Operator << "\n=> Symbol to match " << Symbol << "\n";
    Logger::logs("proteus") << "Available Keys\n";
    for (auto &[Key, Val] : JitVariableMap) {
      Logger::logs("proteus") << "\tKey: " << Key << "\n";
    }
    Logger::logs("proteus") << "===\n";
#endif

    const auto It = JitVariableMap.find(LambdaType);
    if (It == JitVariableMap.end())
      return std::nullopt;

    return It;
  }

  void pushJitVariable(RuntimeConstant &RC) {
    PendingJitVariables.emplace_back(RC);
  }

  // The LambdaType input argument is created as a global variable in the
  // ProteusPass, thus it has program-wide lifetime. Hence it is valid for
  // LambdaTypeRef to store a reference to it.
  inline void registerLambda(const char *LambdaType) {
    const StringRef LambdaTypeRef{LambdaType};
    PROTEUS_DBG(Logger::logs("proteus")
                << "=> RegisterLambda " << LambdaTypeRef << "\n");
    // Copy PendingJitVariables if there were changed, otherwise the runtime
    // values for the lambda definition have not changed.
    if (!PendingJitVariables.empty()) {
      JitVariableMap[LambdaTypeRef] = PendingJitVariables;
      PendingJitVariables.clear();
    }
  }

  const SmallVector<RuntimeConstant> &getJitVariables(StringRef LambdaTypeRef) {
    return JitVariableMap[LambdaTypeRef];
  }

  bool empty() { return JitVariableMap.empty(); }

private:
  explicit LambdaRegistry() = default;
  SmallVector<RuntimeConstant> PendingJitVariables;
  DenseMap<StringRef, SmallVector<RuntimeConstant>> JitVariableMap;
};

} // namespace proteus

#endif
