#ifndef PROTEUS_LAMBDA_INTERFACE_HPP
#define PROTEUS_LAMBDA_INTERFACE_HPP

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Debug.h"
#include "proteus/Hashing.hpp"
#include "proteus/Logger.hpp"

namespace proteus {

using namespace llvm;

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

    StringRef Symbol = StringRef{Operator}.slice(0, Sep);
#if PROTEUS_ENABLE_DEBUG
    Logger::logs("proteus")
        << "Operator " << Operator << "\n=> Symbol to match " << Symbol << "\n";
    Logger::logs("proteus") << "Available Keys\n";
    for (auto &[Key, Val] : JitVariableMap) {
      Logger::logs("proteus") << "\tKey: " << Key << "\n";
    }
    Logger::logs("proteus") << "===\n";
#endif

    const auto SymToRC = JitVariableMap.find(Symbol);
    if (SymToRC == JitVariableMap.end())
      return std::nullopt;

    return SymToRC;
  }

  void pushJitVariable(RuntimeConstant &RC) {
    PendingJitVariables.emplace_back(RC);
  }

  inline void registerLambda(const char *Symbol) {
    const StringRef SymbolStr{Symbol};
    PROTEUS_DBG(Logger::logs("proteus")
                << "=> RegisterLambda " << Symbol << "\n");
    JitVariableMap[SymbolStr] = PendingJitVariables;
    PendingJitVariables.clear();
  }

  bool empty() { return JitVariableMap.empty(); }

  void erase(DenseMap<StringRef, SmallVector<RuntimeConstant>>::iterator It) {
    JitVariableMap.erase(It);
  }

private:
  explicit LambdaRegistry() = default;
  SmallVector<RuntimeConstant> PendingJitVariables;
  DenseMap<StringRef, SmallVector<RuntimeConstant>> JitVariableMap;
};

} // namespace proteus

#endif
