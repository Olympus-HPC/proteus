#ifndef PROTEUS_LAMBDA_INTERFACE_H
#define PROTEUS_LAMBDA_INTERFACE_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/Logger.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
#include <cstring>
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

  using JitVariantMap = DenseMap<int32_t, RuntimeConstant>;
  using JitVariantVec = SmallVector<JitVariantMap, 4>;

  std::optional<ArrayRef<JitVariantMap>> getJitVariants(uint64_t ID) const {
    if (JitVariableVariants.empty())
      return std::nullopt;

    auto It = JitVariableVariants.find(ID);
    if (It == JitVariableVariants.end())
      return std::nullopt;

    return ArrayRef<JitVariantMap>{It->second};
  }

  void appendJitVariable(uint64_t ID, const RuntimeConstant &RC) {
    PendingJitVariables[ID][RC.Pos] = RC;
  }

  void commitJitVariables(uint64_t ID) {
    auto PendingIt = PendingJitVariables.find(ID);
    if (PendingIt == PendingJitVariables.end())
      return;
    if (PendingIt->second.empty())
      return;

    JitVariantMap &Pending = PendingIt->second;
    JitVariantVec &Variants = JitVariableVariants[ID];

    auto RuntimeConstantEqual = [](const RuntimeConstant &A,
                                  const RuntimeConstant &B) -> bool {
      if (A.Type != B.Type)
        return false;
      if (A.Pos != B.Pos)
        return false;
      if (A.Offset != B.Offset)
        return false;
      // Today we only register scalar jit_variable captures via
      // __proteus_register_lambda_runtime_constant.
      return std::memcmp(&A.Value, &B.Value, sizeof(A.Value)) == 0;
    };

    auto VariantEqual = [&](const JitVariantMap &A,
                            const JitVariantMap &B) -> bool {
      if (A.size() != B.size())
        return false;
      for (const auto &KV : A) {
        auto It = B.find(KV.first);
        if (It == B.end())
          return false;
        if (!RuntimeConstantEqual(KV.second, It->second))
          return false;
      }
      return true;
    };

    bool AlreadyPresent = false;
    for (const auto &V : Variants) {
      if (VariantEqual(Pending, V)) {
        AlreadyPresent = true;
        break;
      }
    }

    if (!AlreadyPresent) {
      Variants.push_back(Pending);
    }

    Pending.clear();
  }

  bool empty() const { return JitVariableVariants.empty(); }

private:
  explicit LambdaRegistry() = default;
  // First integral key is the preprocessor/constexpr functor ID generated
  // inside PROTEUS_REGISTER_LAMBDA.  The key of the value DenseMap is the slot
  // within the lambda storage.
  DenseMap<uint64_t, JitVariantVec> JitVariableVariants;
  DenseMap<uint64_t, JitVariantMap> PendingJitVariables;

};

} // namespace proteus

#endif
