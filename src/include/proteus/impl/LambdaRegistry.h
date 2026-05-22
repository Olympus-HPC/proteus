#ifndef PROTEUS_LAMBDA_INTERFACE_H
#define PROTEUS_LAMBDA_INTERFACE_H

#include "proteus/CompilerInterfaceTypes.h"
#include "proteus/Error.h"
#include "proteus/impl/Config.h"
#include "proteus/impl/Debug.h"
#include "proteus/impl/LambdaCallsite.h"
#include "proteus/impl/Logger.h"

#include <cstring>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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

  using JitVariantMap = DenseMap<int32_t, RuntimeConstant>;
  using JitVariantVec = SmallVector<JitVariantMap, 4>;
  struct DeviceLaunchInfo {
    SmallVector<uint64_t, 4> LambdaCalleeInfo;
    LambdaCallsiteRuntimeConstantsMap CallsiteRuntimeConstants;
  };

  std::optional<ArrayRef<JitVariantMap>>
  getHostJitVariants(uint64_t FunctorID) const {
    if (HostJitVariableVariants.empty())
      return std::nullopt;

    auto It = HostJitVariableVariants.find(FunctorID);
    if (It == HostJitVariableVariants.end())
      return std::nullopt;

    return ArrayRef<JitVariantMap>{It->second};
  }

  void appendHostJitVariable(uint64_t FunctorID, const RuntimeConstant &RC) {
    PendingHostJitVariables[FunctorID][RC.Pos] = RC;
  }

  void commitHostJitVariables(uint64_t FunctorID) {
    auto PendingIt = PendingHostJitVariables.find(FunctorID);
    if (PendingIt == PendingHostJitVariables.end())
      return;
    if (PendingIt->second.empty())
      return;

    JitVariantMap &Pending = PendingIt->second;
    JitVariantVec &Variants = HostJitVariableVariants[FunctorID];

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

  void eraseHostJitVariables(uint64_t FunctorID) {
    if (HostJitVariableVariants.contains(FunctorID))
      HostJitVariableVariants.erase(FunctorID);
  }

  bool emptyHost() const { return HostJitVariableVariants.empty(); }

  void beginDeviceLaunch() {
    PendingDeviceLaunchInfo.LambdaCalleeInfo.clear();
    PendingDeviceLaunchInfo.CallsiteRuntimeConstants.clear();
    CurrentDeviceLaunchInfo.LambdaCalleeInfo.clear();
    CurrentDeviceLaunchInfo.CallsiteRuntimeConstants.clear();
  }

  void appendDeviceCallsiteRuntimeConstant(uint64_t LambdaID,
                                           uint32_t CallsiteIndex,
                                           const RuntimeConstant &RC) {
    auto &RCVec =
        PendingDeviceLaunchInfo.CallsiteRuntimeConstants[CallsiteIndex];
    RCVec.push_back(RC);
    if (llvm::find(PendingDeviceLaunchInfo.LambdaCalleeInfo, LambdaID) ==
        PendingDeviceLaunchInfo.LambdaCalleeInfo.end()) {
      PendingDeviceLaunchInfo.LambdaCalleeInfo.push_back(LambdaID);
    }
  }

  void finalizeDeviceLaunch() {
    for (auto &KV : PendingDeviceLaunchInfo.CallsiteRuntimeConstants) {
      llvm::sort(KV.second,
                 [](const RuntimeConstant &L, const RuntimeConstant &R) {
                   return L.Pos < R.Pos;
                 });
    }
    CurrentDeviceLaunchInfo = std::move(PendingDeviceLaunchInfo);
    PendingDeviceLaunchInfo.LambdaCalleeInfo.clear();
    PendingDeviceLaunchInfo.CallsiteRuntimeConstants.clear();
  }

  DeviceLaunchInfo takeDeviceLaunchInfo() {
    DeviceLaunchInfo LaunchInfo = std::move(CurrentDeviceLaunchInfo);
    CurrentDeviceLaunchInfo.LambdaCalleeInfo.clear();
    CurrentDeviceLaunchInfo.CallsiteRuntimeConstants.clear();
    return LaunchInfo;
  }

private:
  explicit LambdaRegistry() = default;
  // First integral key is the preprocessor/constexpr functor ID generated
  // inside PROTEUS_REGISTER_LAMBDA.  The key of the value DenseMap is the slot
  // within the lambda storage.
  DenseMap<uint64_t, JitVariantVec> HostJitVariableVariants;
  DenseMap<uint64_t, JitVariantMap> PendingHostJitVariables;
  DeviceLaunchInfo PendingDeviceLaunchInfo;
  DeviceLaunchInfo CurrentDeviceLaunchInfo;
};

} // namespace proteus

#endif
