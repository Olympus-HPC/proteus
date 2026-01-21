#ifndef PROTEUS_LAMBDA_INTERFACE_H
#define PROTEUS_LAMBDA_INTERFACE_H

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Config.h"
#include "proteus/Debug.h"
#include "proteus/Error.h"
#include "proteus/Logger.h"

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
    if (Config::get().ProteusDebugOutput) {
      Logger::logs("proteus")
          << "Operator " << Operator << "\n=> LambdaType to match "
          << LambdaType << "\n";
      Logger::logs("proteus") << "Available Keys\n";
      for (auto &[Key, Val] : JitVariableMap) {
        Logger::logs("proteus") << "\tKey: " << Key << "\n";
      }
      Logger::logs("proteus") << "===\n";
    }

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
    // Always register the lambda type in the map, even if there are no
    // explicit jit_variable calls. This allows auto-detection to work for
    // lambdas with only auto-detected captures.
    JitVariableMap[LambdaTypeRef] = PendingJitVariables;
    PendingJitVariables.clear();
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
