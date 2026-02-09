// NOLINTBEGIN(readability-identifier-naming)

#include "proteus/impl/CompilerInterfaceRuntimeConstantInfo.h"
#include "proteus/CompilerInterfaceTypes.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdio>
#include <memory>

using namespace proteus;
using namespace llvm;

SmallVectorImpl<std::unique_ptr<RuntimeConstantInfo>> &
getRuntimeConstantInfoStorage() {
  static SmallVector<std::unique_ptr<RuntimeConstantInfo>, 64> RCStorage;
  return RCStorage;
}

extern "C" {
RuntimeConstantInfo *
__proteus_create_runtime_constant_info_scalar(RuntimeConstantType RCType,
                                              int32_t Pos) {
  auto &Ptr = getRuntimeConstantInfoStorage().emplace_back(
      std::make_unique<RuntimeConstantInfo>(RCType, Pos));
  return Ptr.get();
}

RuntimeConstantInfo *__proteus_create_runtime_constant_info_array_const_size(
    RuntimeConstantType RCType, int32_t Pos, int32_t NumElts,
    RuntimeConstantType EltType) {
  auto &Ptr = getRuntimeConstantInfoStorage().emplace_back(
      std::make_unique<RuntimeConstantInfo>(RCType, Pos, NumElts, EltType));
  return Ptr.get();
}

RuntimeConstantInfo *__proteus_create_runtime_constant_info_array_runconst_size(
    RuntimeConstantType RCType, int32_t Pos, RuntimeConstantType EltType,
    RuntimeConstantType NumEltsType, int32_t NumEltsPos) {
  auto &Ptr = getRuntimeConstantInfoStorage().emplace_back(
      std::make_unique<RuntimeConstantInfo>(RCType, Pos, EltType, NumEltsType,
                                            NumEltsPos));
  return Ptr.get();
}

RuntimeConstantInfo *__proteus_create_runtime_constant_info_object(
    RuntimeConstantType RCType, int32_t Pos, int32_t Size, bool PassByValue) {
  auto &Ptr = getRuntimeConstantInfoStorage().emplace_back(
      std::make_unique<RuntimeConstantInfo>(RCType, Pos, Size, PassByValue));
  return Ptr.get();
}
}
// NOLINTEND(readability-identifier-naming)
