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
#include <optional>

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

  LambdaRegistry(const LambdaRegistry &) = delete;
  LambdaRegistry &operator=(const LambdaRegistry &) = delete;
  LambdaRegistry(LambdaRegistry &&) = delete;
  LambdaRegistry &operator=(LambdaRegistry &&) = delete;

  std::optional<DenseMap<int32_t, RuntimeConstant>>
  matchJitVariableMap(uint64_t ID) {
    if (JitVariableMap.empty())
      return std::nullopt;

    auto It = JitVariableMap.find(ID);
    if (It == JitVariableMap.end())
      return std::nullopt;

    return It->second;
  }

  void setJitVariable(uint64_t ID, RuntimeConstant &RC) {
    JitVariableMap[ID][RC.Pos] = RC;
  }

  std::optional<DenseMap<int32_t, RuntimeConstant>> getJitVariables(uint64_t ID) {
    auto It = JitVariableMap.find(ID);
    if (It != JitVariableMap.end())
      return It->second;
    return std::nullopt;
  }

  bool empty() { return JitVariableMap.empty(); }

private:
  explicit LambdaRegistry() = default;
  // First integral key is the preprocessor/constexpr functor ID generated
  // inside PROTEUS_REGISTER_LAMBDA.  The key of the value DenseMap is the slot
  // within the lambda storage.
  DenseMap<uint64_t, DenseMap<int32_t, RuntimeConstant>> JitVariableMap;

};

} // namespace proteus

#endif
