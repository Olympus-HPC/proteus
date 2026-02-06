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

#include <cstring>

namespace proteus {

using namespace llvm;

// Stores a copy of the lambda closure data captured during registration
struct ClosureData {
  SmallVector<char> Data;

  ClosureData() = default;
  ClosureData(const void *ClosurePtr, size_t ClosureSize) {
    if (ClosurePtr && ClosureSize > 0) {
      Data.resize(ClosureSize);
      std::memcpy(Data.data(), ClosurePtr, ClosureSize);
    }
  }

  const void *data() const { return Data.data(); }
  size_t size() const { return Data.size(); }
  bool empty() const { return Data.empty(); }
};

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
  inline void registerLambda(const char *LambdaType,
                             const void *ClosurePtr = nullptr,
                             size_t ClosureSize = 0) {
    const StringRef LambdaTypeRef{LambdaType};
    PROTEUS_DBG(Logger::logs("proteus")
                << "=> RegisterLambda " << LambdaTypeRef << "\n");

    // Only update JitVariableMap if this is a new registration or if we have
    // pending variables This prevents overwriting explicit captures when
    // register_lambda is called multiple times
    auto It = JitVariableMap.find(LambdaTypeRef);
    if (It == JitVariableMap.end() || !PendingJitVariables.empty()) {
      JitVariableMap[LambdaTypeRef] = PendingJitVariables;
      PendingJitVariables.clear();
    }

    // Store closure data if auto-detection is enabled
    // Only store on first registration to avoid unnecessary copies
    if (Config::get().ProteusAutoReadOnlyCaptures && ClosurePtr &&
        ClosureSize > 0) {
      auto ClosureIt = ClosureMap.find(LambdaTypeRef);
      if (ClosureIt == ClosureMap.end()) {
        ClosureMap[LambdaTypeRef] = ClosureData(ClosurePtr, ClosureSize);
        PROTEUS_DBG(Logger::logs("proteus")
                    << "=> Cached closure for " << LambdaTypeRef
                    << " size=" << ClosureSize << "\n");
      }
    }
  }

  const SmallVector<RuntimeConstant> &getJitVariables(StringRef LambdaTypeRef) {
    return JitVariableMap[LambdaTypeRef];
  }

  const ClosureData *getClosureData(StringRef LambdaTypeRef) const {
    auto It = ClosureMap.find(LambdaTypeRef);
    if (It == ClosureMap.end())
      return nullptr;
    return &It->second;
  }

  bool empty() { return JitVariableMap.empty(); }

private:
  explicit LambdaRegistry() = default;
  SmallVector<RuntimeConstant> PendingJitVariables;
  DenseMap<StringRef, SmallVector<RuntimeConstant>> JitVariableMap;
  DenseMap<StringRef, ClosureData> ClosureMap;
};

} // namespace proteus

#endif
