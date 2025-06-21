// NOLINTBEGIN(readability-identifier-naming)

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
__proteus_create_runtime_constant_info(RuntimeConstantType RCType,
                                       int32_t Pos) {
  auto &Ptr = getRuntimeConstantInfoStorage().emplace_back(
      std::make_unique<RuntimeConstantInfo>(RCType, Pos));
  return Ptr.get();
}
}
// NOLINTEND(readability-identifier-naming)
