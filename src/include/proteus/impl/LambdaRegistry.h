#ifndef PROTEUS_LAMBDA_INTERFACE_H
#define PROTEUS_LAMBDA_INTERFACE_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Logger.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
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

  void dump() {
    llvm::outs() << "DUMPING TMP REGISTRY \n";
    for (const auto &[Key, Value] : PendingJitVariableMap) {
      llvm::outs() << Key << "\n";
      for (const auto &V : Value) {
        llvm::outs() << "   " << V.Value.DoubleVal << "\n";
      }
    }
    llvm::outs() << "DUMPING REGISTRY \n";
    for (const auto &[Key, Value] : JitVariableMap) {
      llvm::outs() << Key << "\n";
      for (const auto &V : Value) {
        llvm::outs() << "   " << V.Value.DoubleVal << "\n";
      }
    }
  }

  void setJitVariable(const char *LambdaType, RuntimeConstant &RC) {
    assert(PendingJitVariableMap.contains(LambdaType) &&
           "Lambda must be registered prior to register JIT variable!");
    PendingJitVariableMap[LambdaType].emplace_back(RC);
  }

  inline void registerLambda(const char *LambdaType) {
    const StringRef LambdaTypeRef{LambdaType};
    PROTEUS_DBG(Logger::logs("proteus")
                << "=> RegisterLambda " << LambdaTypeRef << "\n");
    // Copy PendingJitVariables if there were changed, otherwise the runtime
    // values for the lambda definition have not changed.
    if (!PendingJitVariableMap[LambdaTypeRef].empty()) {
      JitVariableMap[LambdaTypeRef] = PendingJitVariableMap[LambdaTypeRef];
      PendingJitVariableMap[LambdaTypeRef].clear();
    }
  }

  const SmallVector<RuntimeConstant> &getJitVariables(StringRef LambdaTypeRef) {
    return JitVariableMap[LambdaTypeRef];
  }

  bool empty() { return JitVariableMap.empty(); }

private:
  explicit LambdaRegistry() = default;
  DenseMap<StringRef, SmallVector<RuntimeConstant>> JitVariableMap;
  DenseMap<StringRef, SmallVector<RuntimeConstant>> PendingJitVariableMap;
};

} // namespace proteus

#endif
